"""Tests for ``styxx.v8.verify`` -- ``--diff``, ``--ref``, result certs, challenge bodies.

Contract: ``styxx/v8/INTERFACES_layer2.md`` section 8.  Spec: v0.2 draft sections 5.2, 5.3, 5.4,
6, 6.1, 9, Appendix B, Appendix D.

The whole exit-code table of section 6 is driven end to end against ``MockRunner``.  Every number
asserted here is derived from the mock's own construction, with the arithmetic in the comment
beside it: the 8-item pool at batch 1 has floor 0.0 on every channel (the mock only flips a
nuisance item when ``batch_size != 1``), so one drifted item out of eight is an exact distance of
1/8 = 0.125.  Nothing skips.
"""
from __future__ import annotations

import copy

import pytest

from styxx.v8 import cert as certmod
from styxx.v8 import fingerprint as FP
from styxx.v8 import floor as floormod
from styxx.v8 import keys
from styxx.v8 import verify as V
from styxx.v8.consts import CHANNELS, EXIT, ROUND_PLACES
from styxx.v8.runner import MockRunner
from tests import v8_fixtures as F

# --------------------------------------------------------------------------- runners

N_ITEMS = 8

# Five batch-1 runs varying only item order (section 5.1 step 2).  The mock ignores order at
# batch 1, so every run is identical and every floor is 0.0 -- the probe's finding (section 5.6).
PLAN_ORDERS = [
    None,  # A.3 order
    ["i02", "i00", "i01", "i03", "i04", "i05", "i06", "i07"],
    ["i07", "i06", "i05", "i04", "i03", "i02", "i01", "i00"],
    ["i01", "i02", "i03", "i04", "i05", "i06", "i07", "i00"],
    ["i03", "i01", "i00", "i02", "i05", "i04", "i07", "i06"],
]


class ObservingRunner(MockRunner):
    """A ``MockRunner`` that names a subject other than the one asked for, and can fail on demand.

    ``subject(requested)`` is required of every runner (``styxx/v8/runner.py``); this one
    reports ``observed_subject`` when it was given one, which is how a verifier that loaded
    other weights is modelled.  ``observed_subject=None`` reports what was requested, like the
    plain mock.
    """

    def __init__(self, *, observed_subject=None, fail_run=None, fail_env=None, **kw):
        super().__init__(**kw)
        self._observed_subject = observed_subject
        self._fail_run = fail_run
        self._fail_env = fail_env

    def subject(self, requested):
        if self._observed_subject is None:
            return super().subject(requested)
        return dict(self._observed_subject)

    def environment(self):
        if self._fail_env is not None:
            raise self._fail_env
        return super().environment()

    def run(self, items, recipe, subject):
        if self._fail_run is not None:
            raise self._fail_run
        return super().run(items, recipe, subject)


# --------------------------------------------------------------------------- fixtures


def pool_cert(*, battery_id: str = F.BATTERY_ID, roles: dict | None = None) -> dict:
    """A pool-v1 battery cert with ``N_ITEMS`` items, addressed by ``battery_id``."""
    body = F.battery_body("pool-v1", n=N_ITEMS)
    if roles:
        for item in body["items"]:
            item["role"] = roles.get(item["item_id"], item.get("role", "item"))
    return {"id": battery_id, "type": "battery", "body": body}


def run_bodies(runner, subject, recipe, battery) -> list[dict]:
    return [
        FP.run_fingerprint(runner, subject, recipe, battery, run_index=k, nuisance={}, order=order)
        for k, order in enumerate(PLAN_ORDERS)
    ]


def fp_body(
    runner,
    subject,
    recipe,
    battery,
    *,
    with_floor: bool = True,
    with_sensitivity: bool = True,
    not_covered: list[str] | None = None,
) -> dict:
    """The canonical fingerprint body: run_index 0, a five-run floor, a sensitivity receipt."""
    runs = run_bodies(runner, subject, recipe, battery)
    if not with_floor:
        body = runs[0]
    else:
        body = FP.attach_floor(
            runs[0],
            runs,
            F.NOISE_PLAN_ID,
            F.RUN_IDS,
            ["order"],
            ["hardware.gpu"] if not_covered is None else not_covered,
        )
    if with_sensitivity:
        body["sensitivity"] = F.SENSITIVITY_ID
    return body


def fp_cert(body: dict, subject: dict, recipe: dict, *, battery_id: str = F.BATTERY_ID, **over) -> dict:
    """Sign ``body`` into a fingerprint cert carrying a ref for every id it embeds (A-09)."""
    refs = [{"role": "battery", "id": battery_id}]
    floor_block = body.get("noise_floor")
    if isinstance(floor_block, dict):
        refs.append({"role": "noise_plan", "id": floor_block["plan"]})
        refs.extend({"role": "run", "id": rid} for rid in floor_block["runs"])
    if body.get("sensitivity"):
        refs.append({"role": "sensitivity", "id": body["sensitivity"]})
    return F.make_cert(
        "fingerprint", subject=subject, recipe=recipe, body=body, refs=refs, **over
    )


def resolver_for(*certs: dict) -> dict:
    """A resolver mapping: the certs given plus the stub targets every ref role needs."""
    out: dict = {}
    for cert in certs:
        out[cert["id"]] = cert
    out[F.NOISE_PLAN_ID] = {"id": F.NOISE_PLAN_ID, "type": "prereg"}
    out[F.SENSITIVITY_ID] = {"id": F.SENSITIVITY_ID, "type": "result"}
    for rid in F.RUN_IDS:
        out[rid] = {"id": rid, "type": "fingerprint"}
    return out


def reference(**kw) -> tuple[dict, dict, dict]:
    """(battery cert, reference fingerprint cert, resolver) on the clean mock."""
    battery = pool_cert(roles=kw.pop("roles", None))
    subject = kw.pop("subject", None) or F.weights_subject()
    recipe = kw.pop("recipe", None) or F.recipe()
    runner = kw.pop("runner", None) or MockRunner()
    body = fp_body(runner, subject, recipe, battery, **kw)
    cert = fp_cert(body, subject, recipe)
    return battery, cert, resolver_for(battery, cert)


# --------------------------------------------------------------------------- the floor itself


def test_the_batch_one_plan_has_floor_zero_on_every_channel():
    """Section 5.6 / the probe: at batch 1 the mock does not move, so max(D_c) = 0.0."""
    battery = pool_cert()
    runs = run_bodies(MockRunner(nuisance_items={"i01", "i05"}), F.weights_subject(), F.recipe(), battery)
    per = floormod.floors(runs)
    assert per["exact"]["floor"] == 0.0
    assert per["seqlp"]["floor"] == 0.0
    assert per["topk"]["floor"] == 0.0
    assert per["exact"]["pairs"] == 10  # 5 runs -> 5*4/2
    assert per["exact"]["alpha_single"] == 1 / 11


# --------------------------------------------------------------------------- exit 0


def test_ref_same_exits_zero():
    battery, cert, res = reference()
    out = V.ref(cert, MockRunner(), res)
    assert out.verdict == "same"
    assert out.exit_code == EXIT["same"] == 0
    assert out.mismatched == []
    assert out.result_body["coverage"] == "within"
    assert out.result_body["confirmation_run"] is None
    for channel in ("exact", "seqlp", "topk"):
        assert out.result_body["per_channel"][channel]["distance"] == 0.0
        assert out.result_body["per_channel"][channel]["verdict"] == "same"
        assert out.result_body["per_channel"][channel]["ratio"] is None  # floor 0 -> null, not NaN


def test_diff_of_a_cert_with_itself_exits_zero():
    battery, cert, res = reference()
    out = V.diff(cert, cert, res)
    assert (out.verdict, out.exit_code) == ("same", 0)
    assert out.result_body["floor_owner"] == "A"
    assert out.result_body["floor_owner_id"] == cert["id"]


def test_same_without_a_sensitivity_receipt_exits_two():
    """Section 5.3: an agreement number without its detection power is not a number."""
    battery, cert, res = reference(with_sensitivity=False)
    out = V.ref(cert, MockRunner(), res)
    assert out.verdict == "same (sensitivity unmeasured)"
    assert out.exit_code == EXIT["sensitivity-unmeasured"] == 2
    assert out.result_body["sensitivity"] is None


# --------------------------------------------------------------------------- exit 1


def test_ref_drift_exits_one_with_a_confirmation_run():
    battery, cert, res = reference()
    out = V.ref(cert, MockRunner(drift_items={"i01"}), res)
    assert out.verdict == "drift"
    assert out.exit_code == EXIT["drift"] == 1
    exact = out.result_body["per_channel"]["exact"]
    assert exact["distance"] == 0.125  # 1 of 8 items differs -> 1 - 7/8
    assert exact["floor"] == 0.0
    assert exact["ratio"] is None  # floor 0 -> ratio null (section 6)
    assert exact["verdict"] == "drift"
    assert exact["confirmation_distance"] == 0.125
    assert out.result_body["confirmation_run"] is not None
    assert out.result_body["confirmation_run"]["channels"]["exact"]["hash"] == (
        out.result_body["new_run"]["channels"]["exact"]["hash"]
    )


def test_ref_without_confirm_never_reaches_drift():
    battery, cert, res = reference()
    out = V.ref(cert, MockRunner(drift_items={"i01"}), res, confirm=False)
    assert out.verdict == "exceeds_floor"
    assert out.exit_code == 2
    assert out.result_body["confirmation_run"] is None


def test_ref_identity_exits_one_and_names_the_field():
    """Section 5.2: identity is a subject verdict about the subject actually obtained."""
    battery, cert, res = reference()
    swapped = F.weights_subject(weights_sha256="e" * 64)
    out = V.ref(cert, ObservingRunner(observed_subject=swapped), res)
    assert out.verdict == "identity (weights_sha256)"
    assert out.exit_code == EXIT["identity"] == 1
    assert out.result_body["identity_diff"] == ["weights_sha256"]
    assert "identity: weights_sha256" in out.printed


def test_identity_names_every_differing_field_in_spec_order():
    battery, cert, res = reference()
    swapped = F.weights_subject(config_sha256="e" * 64, weights_sha256="f" * 64)
    out = V.ref(cert, ObservingRunner(observed_subject=swapped), res)
    # floor.IDENTITY_FIELDS order, not the caller's: weights, tokenizer, config, generation_config
    assert out.verdict == "identity (weights_sha256, config_sha256)"
    assert out.exit_code == 1


def test_a_diff_that_differs_on_an_identity_field_is_exit_three_not_identity():
    """weights_sha256 is inside S_identity, so two such certs are not comparable (section 2.3)."""
    battery = pool_cert()
    subject_a = F.weights_subject()
    subject_b = F.weights_subject(weights_sha256="e" * 64)
    recipe = F.recipe()
    runner = MockRunner()
    a = fp_cert(fp_body(runner, subject_a, recipe, battery), subject_a, recipe)
    b = fp_cert(fp_body(runner, subject_b, recipe, battery), subject_b, recipe)
    out = V.diff(a, b, resolver_for(battery, a, b))
    assert out.exit_code == EXIT["mismatch"] == 3
    assert out.mismatched == ["subject.weights_sha256"]
    assert out.result_body["per_channel"] == {}


# --------------------------------------------------------------------------- exit 2


def test_no_floor_exits_two():
    battery, cert, res = reference(with_floor=False)
    out = V.ref(cert, MockRunner(), res)
    assert out.verdict == "inconclusive"
    assert out.exit_code == EXIT["inconclusive"] == 2
    for block in out.result_body["per_channel"].values():
        assert block["floor"] is None
        assert block["verdict"] == "inconclusive"


def test_diff_without_a_usable_floor_on_the_left_cert_exits_two():
    """Section 6: `--diff A B` exits 2 if A has no floor, even when B has one."""
    battery = pool_cert()
    subject, recipe, runner = F.weights_subject(), F.recipe(), MockRunner()
    a = fp_cert(fp_body(runner, subject, recipe, battery, with_floor=False), subject, recipe)
    b = fp_cert(fp_body(runner, subject, recipe, battery), subject, recipe)
    out = V.diff(a, b, resolver_for(battery, a, b))
    assert (out.verdict, out.exit_code) == ("inconclusive", 2)
    assert out.result_body["floor_owner"] == "A"
    assert out.result_body["floor_b"]["exact"] == 0.0  # B's floor is printed beside it


def test_skew_via_harness_version_exits_two():
    """Section 2.3: a verifier change never reads as model drift."""
    battery, cert, res = reference()
    environment = {
        "runtime": {"framework": "mock", "version": "0", "backend": "cpu"},
        "hardware": {"gpu": "none", "driver": "none", "count": 0},
        "harness": {"name": "styxx", "version": "8.0.1", "commit": "1" * 40},
    }
    out = V.ref(cert, MockRunner(drift_items={"i01"}), res, environment=environment)
    assert out.verdict == "skew"
    assert out.exit_code == EXIT["skew"] == 2
    assert out.result_body["skew_fields"] == ["harness.version"]
    assert out.result_body["per_channel"]["exact"]["distance"] == 0.125
    assert out.result_body["confirmation_run"] is None  # skew is not a drift claim to confirm


def test_skew_below_the_floor_is_still_same():
    """Section 5.2: skew is a verdict only when the distance exceeds the floor."""
    battery, cert, res = reference()
    environment = {
        "runtime": {"framework": "mock", "version": "0", "backend": "cpu"},
        "hardware": {"gpu": "none", "driver": "none", "count": 0},
        "env_lock_sha256": F.sha256_text("a different lockfile"),
    }
    out = V.ref(cert, MockRunner(), res, environment=environment)
    assert out.result_body["skew_fields"] == ["env_lock_sha256"]
    assert (out.verdict, out.exit_code) == ("same", 0)


def test_beyond_floor_coverage_exits_two():
    """Section 5.4: a run from outside `covers` never gets a drift verdict."""
    battery, cert, res = reference()
    environment = {
        "runtime": {"framework": "mock", "version": "0", "backend": "cpu"},
        "hardware": {"gpu": "NVIDIA GeForce RTX 4070 Laptop GPU", "driver": "560.94", "count": 1},
    }
    out = V.ref(cert, MockRunner(drift_items={"i01"}), res, environment=environment)
    assert out.verdict == "beyond-floor-coverage"
    assert out.exit_code == EXIT["beyond-floor-coverage"] == 2
    assert out.result_body["coverage"] == "beyond-floor-coverage"
    assert out.result_body["coverage_diff"] == ["hardware.gpu"]
    assert out.result_body["per_channel"]["exact"]["distance"] == 0.125  # printed, not acted on


def test_diff_never_says_drift():
    """GATED S5-02 recommendation A: an unconfirmed exceedance is `exceeds_floor`."""
    battery = pool_cert()
    subject, recipe = F.weights_subject(), F.recipe()
    a = fp_cert(fp_body(MockRunner(), subject, recipe, battery), subject, recipe)
    b = fp_cert(
        fp_body(MockRunner(drift_items={"i01", "i04"}), subject, recipe, battery), subject, recipe
    )
    out = V.diff(a, b, resolver_for(battery, a, b))
    assert out.verdict == "exceeds_floor"
    assert out.exit_code == 2
    assert out.result_body["per_channel"]["exact"]["distance"] == 0.25  # 2 of 8
    assert all(block["verdict"] != "drift" for block in out.result_body["per_channel"].values())
    assert "drift" not in out.printed


def test_cross_subject_diff_is_labelled_and_can_never_say_same():
    """Section 6: a precision-crossing --diff is comparable, labelled, and never `same`."""
    battery = pool_cert()
    subject_a = F.weights_subject()
    subject_b = F.weights_subject(precision="int8-bnb")
    recipe = F.recipe()
    body = fp_body(MockRunner(), subject_a, recipe, battery)
    a = fp_cert(body, subject_a, recipe)
    # the same measured body under the other precision: distances are 0 on every channel
    b = fp_cert(copy.deepcopy(body), subject_b, recipe)
    out = V.diff(a, b, resolver_for(battery, a, b))
    assert out.result_body["cross_subject"] == ["precision"]
    assert out.result_body["per_channel"]["exact"]["distance"] == 0.0
    assert out.verdict == "inconclusive (cross-subject)"
    assert out.exit_code == 2
    assert "cross-subject: precision" in out.printed


def test_cross_subject_diff_that_exceeds_the_floor_keeps_its_channel_verdict():
    battery = pool_cert()
    subject_a = F.weights_subject()
    subject_b = F.weights_subject(precision="int8-bnb")
    recipe = F.recipe()
    runner = MockRunner(precision_items={"int8-bnb": {"i03"}})
    a = fp_cert(fp_body(runner, subject_a, recipe, battery), subject_a, recipe)
    b = fp_cert(fp_body(runner, subject_b, recipe, battery), subject_b, recipe)
    out = V.diff(a, b, resolver_for(battery, a, b))
    assert out.result_body["cross_subject"] == ["precision"]
    assert out.result_body["per_channel"]["exact"]["distance"] == 0.125  # 1 of 8 under int8-bnb
    assert (out.verdict, out.exit_code) == ("exceeds_floor", 2)


def test_a_revision_crossing_diff_is_also_cross_subject():
    battery = pool_cert()
    subject_a = F.weights_subject()
    subject_b = F.weights_subject(revision="b" * 40)
    recipe = F.recipe()
    body = fp_body(MockRunner(), subject_a, recipe, battery)
    a = fp_cert(body, subject_a, recipe)
    b = fp_cert(copy.deepcopy(body), subject_b, recipe)
    out = V.diff(a, b, resolver_for(battery, a, b))
    assert out.result_body["cross_subject"] == ["revision"]
    assert out.exit_code == 2


# --------------------------------------------------------------------------- exit 3


def test_mismatch_via_a_different_battery_exits_three_with_no_distances():
    battery_a = pool_cert()
    other_id = F.fake_id("battery-two")
    battery_b = pool_cert(battery_id=other_id)
    subject = F.weights_subject()
    recipe_a = F.recipe()
    recipe_b = F.recipe(battery=other_id)
    runner = MockRunner()
    a = fp_cert(fp_body(runner, subject, recipe_a, battery_a), subject, recipe_a)
    b = fp_cert(
        fp_body(runner, subject, recipe_b, battery_b), subject, recipe_b, battery_id=other_id
    )
    out = V.diff(a, b, resolver_for(battery_a, battery_b, a, b))
    assert out.exit_code == EXIT["mismatch"] == 3
    assert out.mismatched == ["recipe.battery"]
    assert out.result_body["per_channel"] == {}
    assert out.result_body["skipped_channels"] == []
    assert "not comparable: recipe.battery" in out.printed
    # section 6: the mismatched fields are printed and nothing else
    assert "distance=" not in out.printed


def test_a_schema_version_this_verifier_does_not_implement_exits_three():
    """Section 2.5: `major.minor` above our own is refused with exit 3, not exit 4."""
    battery, cert, res = reference()
    future = copy.deepcopy(cert)
    future["styxx"] = "8.1"
    out = V.diff(cert, future, res)
    assert out.exit_code == EXIT["mismatch"] == 3
    assert out.mismatched == ["schema_version:8.1"]
    assert out.verdict == "mismatch (schema version)"


def test_version_eight_ten_is_also_refused():
    """`8.10` > `8.0` numerically, not lexically."""
    battery, cert, res = reference()
    future = copy.deepcopy(cert)
    future["styxx"] = "8.10"
    out = V.ref(future, MockRunner(), res)
    assert (out.exit_code, out.mismatched) == (3, ["schema_version:8.10"])


def test_a_malformed_version_string_is_invalid_not_a_version_mismatch():
    battery, cert, res = reference()
    broken = copy.deepcopy(cert)
    broken["styxx"] = "not-a-version"
    out = V.ref(broken, MockRunner(), res)
    assert out.exit_code == EXIT["invalid"] == 4


def test_ref_refuses_a_subject_that_crosses_precision():
    """A --ref must re-run the same subject; `cross-subject` is a --diff affordance (section 6)."""
    battery, cert, res = reference()
    other = F.weights_subject(precision="nf4-bnb")
    out = V.ref(cert, ObservingRunner(observed_subject=other), res)
    assert out.exit_code == 3
    assert out.mismatched == ["cross-subject:precision"]
    assert out.result_body["per_channel"] == {}


def test_diff_of_a_non_fingerprint_is_a_mismatch():
    battery, cert, res = reference()
    other = F.make_cert("prereg")
    res2 = dict(res)
    res2[other["id"]] = other
    out = V.diff(cert, other, res2)
    assert out.exit_code == 3
    assert out.mismatched == ["cert.type:'fingerprint' vs 'prereg'"]


# --------------------------------------------------------------------------- exit 4


def test_a_tampered_cert_exits_four():
    battery, cert, res = reference()
    tampered = copy.deepcopy(cert)
    tampered["created"] = "2026-09-09T00:00:00Z"  # inside the digested bytes
    out = V.ref(tampered, MockRunner(), res)
    assert out.verdict == "invalid"
    assert out.exit_code == EXIT["invalid"] == 4
    assert any("id: does not recompute" in r for r in out.result_body["invalid_reasons"])


def test_a_cert_signed_by_the_wrong_key_exits_four():
    """The id recomputes, but the signature was made by a key that is not `issuer.key`.

    ``F.make_cert(seed=...)`` cannot build this: ``cert.sign`` refuses a seed whose public key is
    not ``issuer.key``, so the envelope is assembled by hand here (as ``test_v8_cert.py`` does).
    """
    battery = pool_cert()
    subject, recipe = F.weights_subject(), F.recipe()
    body = fp_body(MockRunner(), subject, recipe, battery)
    core = fp_cert(body, subject, recipe, signed=False, issuer_label="other")
    digest = certmod.digest_bytes(core)
    cert = dict(core)
    cert["id"] = "sha256:" + digest.hex()
    cert["sig"] = keys.encode_signature(
        keys.sign(F.keypair("issuer")[0], keys.tagged(certmod.CERT_TAG, digest))
    )
    out = V.ref(cert, MockRunner(), resolver_for(battery, cert))
    assert out.exit_code == 4
    assert any("sig: does not verify" in r for r in out.result_body["invalid_reasons"])


def test_a_ref_that_does_not_resolve_exits_four():
    battery, cert, res = reference()
    short = {k: v for k, v in res.items() if k != F.SENSITIVITY_ID}
    out = V.ref(cert, MockRunner(), short)
    assert out.exit_code == 4
    assert any("does not resolve" in r for r in out.result_body["invalid_reasons"])


def test_a_ref_that_resolves_to_the_wrong_type_exits_four():
    battery, cert, res = reference()
    wrong = dict(res)
    wrong[F.NOISE_PLAN_ID] = {"id": F.NOISE_PLAN_ID, "type": "battery"}
    out = V.ref(cert, MockRunner(), wrong)
    assert out.exit_code == 4
    assert any("not 'prereg'" in r for r in out.result_body["invalid_reasons"])


def test_an_unresolvable_battery_exits_four():
    battery, cert, res = reference()
    without = {k: v for k, v in res.items() if k != F.BATTERY_ID}
    without[F.BATTERY_ID] = {"id": F.BATTERY_ID, "type": "battery"}  # resolves, but no body
    out = V.ref(cert, MockRunner(), without)
    # the ref resolves to the right type, so the failure is at run time, not at the ref gate
    assert out.exit_code == EXIT["unavailable"] == 5


def test_no_resolver_means_no_ref_resolution_in_diff():
    """`resolver=None` is a caller that did not ask for resolution; the comparison still runs."""
    battery, cert, res = reference()
    out = V.diff(cert, cert, None)
    assert (out.verdict, out.exit_code) == ("same", 0)


# --------------------------------------------------------------------------- exit 5


def test_a_runner_that_cannot_run_exits_five():
    battery, cert, res = reference()
    out = V.ref(cert, ObservingRunner(fail_run=RuntimeError("unavailable")), res)
    assert out.verdict == "unavailable"
    assert out.exit_code == EXIT["unavailable"] == 5
    assert out.result_body["unavailable_reason"] == "RuntimeError: unavailable"
    assert out.result_body["attempted"]["battery"] == F.BATTERY_ID
    assert out.result_body["per_channel"] == {}
    assert "attempted battery" in out.printed


def test_a_runner_whose_environment_fails_exits_five():
    battery, cert, res = reference()
    out = V.ref(cert, ObservingRunner(fail_env=OSError("no nvidia-smi")), res)
    assert (out.verdict, out.exit_code) == ("unavailable", 5)
    # The reason names the member that failed as well as the cause: the environment is an
    # obligation of the runner now (`runner.reported_environment`), so a reader has to be able
    # to tell a runner that would not say from a cert that could not be read.
    assert out.result_body["unavailable_reason"] == (
        "runner ObservingRunner.environment(): OSError: no nvidia-smi"
    )


def test_a_confirmation_run_that_fails_exits_five():
    class OnceRunner(ObservingRunner):
        def __init__(self, **kw):
            super().__init__(**kw)
            self._calls = 0

        def run(self, items, recipe, subject):
            self._calls += 1
            if self._calls > 1:
                raise RuntimeError("gpu fell off the bus")
            return MockRunner.run(self, items, recipe, subject)

    battery, cert, res = reference()
    out = V.ref(cert, OnceRunner(drift_items={"i01"}), res)
    assert (out.verdict, out.exit_code) == ("unavailable", 5)
    assert out.result_body["unavailable_reason"].startswith("confirmation run: RuntimeError")


# --------------------------------------------------------------------------- the whole table


EXPECTED_EXITS = {
    "same": 0,
    "drift": 1,
    "identity": 1,
    "exceeds_floor": 2,
    "inconclusive": 2,
    "skew": 2,
    "beyond-floor-coverage": 2,
    "sensitivity-unmeasured": 2,
    "mismatch": 3,
    "invalid": 4,
    "unavailable": 5,
}


def test_the_exit_table_is_exactly_the_constant():
    assert EXIT == {k: v for k, v in EXPECTED_EXITS.items() if k != "exceeds_floor"}


@pytest.mark.parametrize(
    "scenario, verdict, code",
    [
        ("same", "same", 0),
        ("drift", "drift", 1),
        ("identity", "identity (weights_sha256)", 1),
        ("no-floor", "inconclusive", 2),
        ("skew", "skew", 2),
        ("beyond-coverage", "beyond-floor-coverage", 2),
        ("no-sensitivity", "same (sensitivity unmeasured)", 2),
        ("mismatch", "mismatch", 3),
        ("invalid", "invalid", 4),
        ("unavailable", "unavailable", 5),
    ],
)
def test_the_exit_code_table_end_to_end(scenario, verdict, code):
    """Section 6, the CI contract, one row at a time on the mock."""
    battery = pool_cert()
    subject, recipe = F.weights_subject(), F.recipe()
    kw: dict = {}
    if scenario == "no-floor":
        kw["with_floor"] = False
    if scenario == "no-sensitivity":
        kw["with_sensitivity"] = False
    body = fp_body(MockRunner(), subject, recipe, battery, **kw)
    cert = fp_cert(body, subject, recipe)
    res = resolver_for(battery, cert)

    runner = MockRunner()
    call: dict = {}
    if scenario == "drift":
        runner = MockRunner(drift_items={"i01"})
    elif scenario == "identity":
        runner = ObservingRunner(observed_subject=F.weights_subject(weights_sha256="e" * 64))
    elif scenario == "skew":
        runner = MockRunner(drift_items={"i01"})
        call["environment"] = {
            "runtime": {"framework": "mock", "version": "0", "backend": "cpu"},
            "hardware": {"gpu": "none", "driver": "none", "count": 0},
            "harness": {"name": "styxx", "version": "8.0.1", "commit": "1" * 40},
        }
    elif scenario == "beyond-coverage":
        runner = MockRunner(drift_items={"i01"})
        call["environment"] = {
            "runtime": {"framework": "mock", "version": "0", "backend": "cpu"},
            "hardware": {"gpu": "other", "driver": "none", "count": 1},
        }
    elif scenario == "unavailable":
        runner = ObservingRunner(fail_run=RuntimeError("unavailable"))
    elif scenario == "invalid":
        cert = copy.deepcopy(cert)
        cert["created"] = "2026-09-09T00:00:00Z"
    elif scenario == "mismatch":
        runner = ObservingRunner(observed_subject=F.weights_subject(hf_repo="other/repo"))

    out = V.ref(cert, runner, res, **call)
    assert (out.verdict, out.exit_code) == (verdict, code)
    assert f"(exit {code})" in out.printed


# --------------------------------------------------------------------------- reporting


def test_the_report_carries_the_coverage_line_with_every_verdict():
    """Section 5.4: `verify` prints the coverage line beside every verdict."""
    battery, cert, res = reference()
    out = V.ref(cert, MockRunner(), res)
    assert "coverage: within" in out.printed
    assert "covers=order" in out.printed
    assert "not_covered=hardware.gpu" in out.printed
    assert out.result_body["covers"] == ["order"]
    assert out.result_body["not_covered"] == ["hardware.gpu"]
    assert out.result_body["coverage_diff"] == []


def test_an_alias_verdict_carries_the_section_2_2_sentence():
    battery = pool_cert()
    subject, recipe = F.alias_subject(), F.recipe()
    runner = MockRunner(logprobs=False)
    # an alias subject has no `environment` block, so its floor holds no environment field fixed
    body = fp_body(runner, subject, recipe, battery, not_covered=[])
    cert = fp_cert(body, subject, recipe)
    out = V.ref(cert, MockRunner(logprobs=False), resolver_for(battery, cert))
    assert out.result_body["alias_note"] == V.ALIAS_NOTE
    assert V.ALIAS_NOTE in out.printed
    assert out.result_body["new_run"]["tier"] == "black-box"
    # no log-probs: seqlp and topk are absent on BOTH sides, so nothing is compared or skipped
    assert set(out.result_body["per_channel"]) == {"exact"}
    assert out.result_body["skipped_channels"] == []
    assert (out.verdict, out.exit_code) == ("same", 0)


def test_the_report_never_uses_a_claim_word():
    battery, cert, res = reference()
    for out in (V.ref(cert, MockRunner(), res), V.diff(cert, cert, res)):
        lowered = out.printed.lower()
        for word in ("immutable", "tamper-proof", "self-verifying"):
            assert word not in lowered


def test_the_rounding_is_recorded_in_the_result():
    battery, cert, res = reference()
    out = V.ref(cert, MockRunner(), res)
    assert out.result_body["rounding"] == ROUND_PLACES == 9


# --------------------------------------------------------------------------- channels


def test_a_channel_present_on_one_side_only_is_skipped_and_never_moves_the_exit_code():
    """Section 5.2: skipped channels are listed and never affect the exit code."""
    battery, cert, res = reference()
    out = V.ref(cert, MockRunner(logprobs=False), res)
    assert out.result_body["skipped_channels"] == ["seqlp", "topk"]
    assert set(out.result_body["per_channel"]) == {"exact"}
    assert (out.verdict, out.exit_code) == ("same", 0)


def test_anchor_flips_are_counted_beside_the_verdicts_and_do_not_move_the_distance():
    """Appendix B: anchors are counted separately and never produce a verdict."""
    battery = pool_cert(roles={"i02": "anchor"})
    subject, recipe = F.weights_subject(), F.recipe()
    # log-probs off on both sides: `exact` is the only channel, so the anchor flip stands alone
    # (seqlp and topk have no role filter in Appendix B and would move on any flipped item)
    cert = fp_cert(fp_body(MockRunner(logprobs=False), subject, recipe, battery), subject, recipe)
    runner = MockRunner(drift_items={"i02"}, logprobs=False)
    out = V.ref(cert, runner, resolver_for(battery, cert))
    assert out.result_body["anchor_flips"] == 1
    assert out.result_body["per_channel"]["exact"]["distance"] == 0.0
    assert (out.verdict, out.exit_code) == ("same", 0)
    assert "anchor_flips: 1" in out.printed


def test_an_anchor_flip_still_moves_the_log_prob_channels():
    """Appendix B: only `exact` filters by role; seqlp and topk are over every item."""
    battery = pool_cert(roles={"i02": "anchor"})
    subject, recipe = F.weights_subject(), F.recipe()
    cert = fp_cert(fp_body(MockRunner(), subject, recipe, battery), subject, recipe)
    out = V.ref(cert, MockRunner(drift_items={"i02"}), resolver_for(battery, cert))
    assert out.result_body["per_channel"]["exact"]["verdict"] == "same"
    assert out.result_body["per_channel"]["seqlp"]["verdict"] == "drift"
    assert (out.verdict, out.exit_code) == ("drift", 1)


def test_a_channel_that_cannot_be_compared_is_inconclusive_not_an_exception():
    """Appendix B: `lens` with differing n_layers is inconclusive; a verifier reports, not raises."""
    battery, cert, res = reference()
    a_body = copy.deepcopy(cert["body"])
    b_body = copy.deepcopy(cert["body"])
    n = len(a_body["items"])
    a_body["channels"]["lens"] = {
        "present": True,
        "n_layers": 26,
        "converge_layer": [14.0] * n,
        "mean": 14.0,
    }
    b_body["channels"]["lens"] = {
        "present": True,
        "n_layers": 24,
        "converge_layer": [14.0] * n,
        "mean": 14.0,
    }
    a_body["noise_floor"]["per_channel"]["lens"] = {
        "floor": 0.0, "distances": [0.0], "runs": 2, "pairs": 1, "alpha_single": 0.5
    }
    subject, recipe = F.weights_subject(), F.recipe()
    a = fp_cert(a_body, subject, recipe)
    b = fp_cert(b_body, subject, recipe)
    out = V.diff(a, b, resolver_for(pool_cert(), a, b))
    lens = out.result_body["per_channel"]["lens"]
    assert lens["distance"] is None
    assert lens["verdict"] == "inconclusive"
    assert "n_layers" in lens["note"]
    assert out.exit_code == 2


def test_the_ratio_is_the_quotient_when_the_floor_is_above_zero():
    battery, cert, res = reference()
    raised = copy.deepcopy(cert["body"])
    raised["noise_floor"]["per_channel"]["exact"]["floor"] = 0.5
    subject, recipe = F.weights_subject(), F.recipe()
    a = fp_cert(raised, subject, recipe)
    b = fp_cert(
        fp_body(MockRunner(drift_items={"i01", "i04"}), subject, recipe, pool_cert()),
        subject,
        recipe,
    )
    out = V.diff(a, b, resolver_for(pool_cert(), a, b))
    exact = out.result_body["per_channel"]["exact"]
    assert exact["distance"] == 0.25  # 2 of 8
    assert exact["floor"] == 0.5
    assert exact["ratio"] == 0.5  # 0.25 / 0.5
    assert exact["verdict"] == "same"  # 0.25 <= 0.5


# --------------------------------------------------------------------------- section 9


def test_challenge_body_is_the_section_9_shape():
    battery, cert, res = reference()
    out = V.ref(cert, MockRunner(drift_items={"i01"}), res)
    environment = {
        "runtime": {"framework": "mock", "version": "0", "backend": "cpu"},
        "hardware": {"gpu": "none", "driver": "none", "count": 0},
    }
    body = V.challenge_body(out, environment, note="reproduced on a second box")
    assert set(body) == {
        "per_channel", "coverage", "environment", "note", "subject", "recipe_core", "synthetic",
    }
    assert body["coverage"] == "within"
    assert body["per_channel"]["exact"] == {"distance": 0.125, "target_floor": 0.0}
    assert body["environment"] == environment
    # C3: the body says what produced the distances, and the mock says they are not a model's.
    assert body["subject"] == certmod.identity_fields(cert["subject"])
    assert body["recipe_core"] == certmod.recipe_core(cert["recipe"])
    assert body["synthetic"] is True


def test_a_challenge_cert_built_from_the_body_checks():
    battery, cert, res = reference()
    out = V.ref(cert, MockRunner(drift_items={"i01"}), res)
    own = fp_cert(
        fp_body(MockRunner(drift_items={"i01"}), F.weights_subject(), F.recipe(), pool_cert()),
        F.weights_subject(),
        F.recipe(),
        issuer_label="other",
    )
    body = V.challenge_body(out, {"runtime": {"framework": "mock"}})
    challenge = F.make_cert(
        "challenge",
        body=body,
        refs=[{"role": "target", "id": cert["id"]}, {"role": "own", "id": own["id"]}],
        issuer_label="other",
    )
    outcome = certmod.check(challenge)
    assert outcome.ok, outcome.reasons


def test_a_challenge_from_beyond_coverage_says_so():
    """Section 9: such a challenge is classified beyond-floor-coverage, never a dispute."""
    battery, cert, res = reference()
    environment = {
        "runtime": {"framework": "mock", "version": "0", "backend": "cpu"},
        "hardware": {"gpu": "other", "driver": "none", "count": 1},
    }
    out = V.ref(cert, MockRunner(drift_items={"i01"}), res, environment=environment)
    body = V.challenge_body(out, environment)
    assert body["coverage"] == "beyond-floor-coverage"


@pytest.mark.parametrize("scenario", ["mismatch", "invalid", "unavailable"])
def test_an_outcome_that_compared_nothing_cannot_become_a_challenge(scenario):
    battery, cert, res = reference()
    if scenario == "unavailable":
        out = V.ref(cert, ObservingRunner(fail_run=RuntimeError("unavailable")), res)
    elif scenario == "invalid":
        broken = copy.deepcopy(cert)
        broken["created"] = "2026-09-09T00:00:00Z"
        out = V.ref(broken, MockRunner(), res)
    else:
        out = V.ref(cert, ObservingRunner(observed_subject=F.weights_subject(hf_repo="x/y")), res)
    with pytest.raises(ValueError):
        V.challenge_body(out, {"runtime": {}})


def test_challenge_body_refuses_a_non_outcome():
    with pytest.raises(TypeError):
        V.challenge_body({"coverage": "within"}, {})


# --------------------------------------------------------------------------- the result cert


def test_make_result_cert_checks_as_a_signed_result_cert():
    battery, cert, res = reference()
    out = V.diff(cert, cert, res)
    seed, _public = F.keypair("issuer")
    result = V.make_result_cert(
        out,
        F.issuer(),
        seed,
        out.result_body["refs_suggested"],
        {},
        {},
        created=F.CREATED,
    )
    outcome = certmod.check(result)
    assert outcome.ok, outcome.reasons
    assert result["type"] == "result"
    assert result["body"]["kind"] == "verify"
    assert result["body"]["overall"] == "same"
    assert [r["role"] for r in result["refs"]] == ["target", "own", "sensitivity"]


def test_make_result_cert_from_a_ref_run_checks():
    battery, cert, res = reference()
    out = V.ref(cert, MockRunner(drift_items={"i01"}), res)
    seed, _public = F.keypair("issuer")
    result = V.make_result_cert(
        out, F.public_key("issuer"), seed, out.result_body["refs_suggested"], {}, {},
        created=F.CREATED,
    )
    outcome = certmod.check(result)
    assert outcome.ok, outcome.reasons
    assert result["issuer"]["name"] == "styxx verify"
    assert result["body"]["overall"] == "drift"
    assert result["body"]["new_run"]["run_index"] == 0


def test_make_result_cert_accepts_role_id_tuples():
    battery, cert, res = reference()
    out = V.diff(cert, cert, res)
    seed, _public = F.keypair("issuer")
    pairs = [(r["role"], r["id"]) for r in out.result_body["refs_suggested"]]
    result = V.make_result_cert(out, F.issuer(), seed, pairs, {}, {}, created=F.CREATED)
    assert certmod.check(result).ok


def test_make_result_cert_refuses_a_body_whose_ids_are_not_in_refs():
    """Section 2.1 / A-09: every cert id inside a cert must appear in refs."""
    battery, cert, res = reference()
    out = V.diff(cert, cert, res)
    seed, _public = F.keypair("issuer")
    with pytest.raises(ValueError, match="does not carry every cert id"):
        V.make_result_cert(out, F.issuer(), seed, [], {}, {}, created=F.CREATED)


def test_make_result_cert_refuses_a_seed_that_is_not_the_issuer_key():
    battery, cert, res = reference()
    out = V.diff(cert, cert, res)
    other_seed, _public = F.keypair("other")
    with pytest.raises(ValueError, match="refusing to sign"):
        V.make_result_cert(
            out, F.issuer(), other_seed, out.result_body["refs_suggested"], {}, {},
            created=F.CREATED,
        )


def test_make_result_cert_is_byte_stable_for_the_same_outcome():
    battery, cert, res = reference()
    out = V.diff(cert, cert, res)
    seed, _public = F.keypair("issuer")
    args = (out, F.issuer(), seed, out.result_body["refs_suggested"], {}, {})
    first = V.make_result_cert(*args, created=F.CREATED)
    second = V.make_result_cert(*args, created=F.CREATED)
    assert first == second
    assert first["id"] == second["id"]


def test_make_result_cert_refuses_a_bad_issuer():
    battery, cert, res = reference()
    out = V.diff(cert, cert, res)
    seed, _public = F.keypair("issuer")
    with pytest.raises(TypeError):
        V.make_result_cert(out, 7, seed, [], {}, {})


def test_make_result_cert_refuses_a_non_outcome():
    seed, _public = F.keypair("issuer")
    with pytest.raises(TypeError):
        V.make_result_cert({"kind": "verify"}, F.issuer(), seed, [], {}, {})


# --------------------------------------------------------------------------- shape


def test_the_result_body_carries_every_field_section_6_1_names():
    battery, cert, res = reference()
    for out in (V.ref(cert, MockRunner(), res), V.diff(cert, cert, res)):
        body = out.result_body
        for name in (
            "kind", "ref", "new_run", "confirmation_run", "per_channel", "overall",
            "floor_owner", "coverage", "skipped_channels", "rounding",
        ):
            assert name in body, name
        assert body["kind"] == "verify"
        assert "exit_code" not in body  # the exit code is the process's, not the cert's


def test_the_outcome_exit_code_matches_the_verdict_for_every_channel_verdict():
    """Every verdict `floor.decide` can produce reaches a code in the section 6 table."""
    codes = set()
    battery = pool_cert()
    subject, recipe = F.weights_subject(), F.recipe()
    cert = fp_cert(fp_body(MockRunner(), subject, recipe, battery), subject, recipe)
    res = resolver_for(battery, cert)
    for runner in (MockRunner(), MockRunner(drift_items={"i01"})):
        for confirm in (True, False):
            out = V.ref(cert, runner, res, confirm=confirm)
            codes.add(out.exit_code)
            assert out.exit_code in set(EXIT.values())
    assert codes == {0, 1, 2}


def test_verify_outcome_fields_have_the_contract_types():
    battery, cert, res = reference()
    out = V.diff(cert, cert, res)
    assert isinstance(out.result_body, dict)
    assert isinstance(out.verdict, str)
    assert isinstance(out.exit_code, int)
    assert isinstance(out.mismatched, list)
    assert isinstance(out.printed, str)


def test_diff_and_ref_refuse_a_non_dict():
    battery, cert, res = reference()
    with pytest.raises(TypeError):
        V.diff("not a cert", cert, res)
    with pytest.raises(TypeError):
        V.ref(["not a cert"], MockRunner(), res)


def test_a_callable_resolver_and_a_log_shaped_resolver_both_work():
    """`resolver` may be a mapping, a callable, or anything with `find`/`cert` (a ``Log``)."""
    battery, cert, res = reference()

    class LogShaped:
        def __init__(self, mapping):
            self._ids = list(mapping)
            self._certs = mapping

        def find(self, cert_id):
            return self._ids.index(cert_id) if cert_id in self._certs else None

        def cert(self, index):
            return self._certs[self._ids[index]]

    for resolver in (res.get, LogShaped(res)):
        out = V.ref(cert, MockRunner(), resolver)
        assert (out.verdict, out.exit_code) == ("same", 0)

    missing = LogShaped({k: v for k, v in res.items() if k != F.SENSITIVITY_ID})
    assert V.ref(cert, MockRunner(), missing).exit_code == 4


def test_an_unknown_resolver_kind_is_refused():
    battery, cert, res = reference()
    with pytest.raises(TypeError, match="resolver must be"):
        V.diff(cert, cert, 7)


def test_every_channel_name_in_a_result_body_is_a_known_channel():
    battery, cert, res = reference()
    out = V.ref(cert, MockRunner(), res)
    for name in out.result_body["per_channel"]:
        assert name in CHANNELS
    for name in out.result_body["skipped_channels"]:
        assert name in CHANNELS

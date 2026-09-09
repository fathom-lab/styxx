"""The four holes the second adversarial pass opened around the section 9 repair, closed.

`papers/v8/challenge_and_attack_2026_09_09/RESULT_challenge_and_attack_2026_09_09.md` records a
pass whose central sentence is the brief for this file: **the repair moved the trust boundary
rather than removing it.**  Section 9 rule 1 had been implemented three times over — client,
mint and append — and the attacker walked around all three rather than through any of them.

One test per hole, each named for the finding, each asserting the refusal AND that the honest
path still works:

* **C-MOCK** — `--runner mock` ships in the CLI and is its default; `MockRunner` reports back
  whatever subject it is handed, so the section 6 guard was true by construction and a challenge
  was signed against a published canonical carrying distances from a model that was never
  loaded.  The rule taken: a synthetic runner's numbers carry `synthetic: true` in the signed
  body, and `cert.comparable` refuses to cross that line.  See `styxx/v8/cert.py`, "Decisions",
  for why the marker is a body member and why the alternative (refuse to sign at all) lost.
* **C-DUPREF** — `Log._check_challenge_subject` built `by_role[role] = ...` in a loop, so
  `refs = [target, own=<fp16>, own=<bf16>]` gave the log the last `own` and a reader the earlier
  one.  Refused in `cert.check` for every single-valued role, and again in the log.
* **C-MINT-UNCOMPUTED** — `cli._challenge` signed when it could not compute rule 1, and the fact
  lived only in stdout.  Now it refuses; the CLI half of this is
  `tests/test_v8_cli.py::test_verify_challenge_refuses_to_sign_when_it_could_not_compute_rule_one`.
* **C-NONFP-TARGET** — `ROLE_TYPES['target']` is `'*'` (a `response` result's target is a
  challenge), so a challenge could name the battery cert as its target and the subject check
  never ran at all.  `cert.role_type` fixes a challenge's target to a fingerprint.

and **C3** — a challenge asserted nothing in its own bytes about what produced its distances.
The body now carries the challenger's `subject` and `recipe_core`, and the log refuses a body
that disagrees with the `own` cert it names.

Everything here runs on the mock; nothing here is evidence about a model.
"""
from __future__ import annotations

import copy

import pytest

from styxx.v8 import cert as certmod
from styxx.v8 import fingerprint as FP
from styxx.v8 import runner as runnermod
from styxx.v8 import verify as V
from styxx.v8.log import AppendRefused
from styxx.v8.runner import MockRunner
from tests import v8_fixtures as F
from tests.test_v8_log import battery_cert, challenge_body_for, challenge_cert, fp_cert, fresh
from tests.test_v8_verify import pool_cert, reference


# --------------------------------------------------------------------------- a ladder to attack


def two_fingerprints(tmp_path):
    """A log holding one battery, a target fingerprint and a comparable `own` fingerprint."""
    log = fresh(tmp_path / "log")
    battery = battery_cert()
    log.append(battery)
    target = fp_cert(battery["id"])
    log.append(target)
    own = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"]),
        refs=[{"role": "battery", "id": battery["id"]}, {"role": "previous", "id": target["id"]}],
        body=F.fingerprint_body(run_index=1),
    )
    log.append(own)
    return log, battery, target, own


# --------------------------------------------------------------------------- C-MOCK


def measured_body_from(mock_body: dict) -> dict:
    """A body a model could have produced, out of one the mock produced.

    Dropping ``synthetic`` is no longer enough to make a mock's body pass for a measurement, and
    that is C-MOCK-2's repair: ``cert.check`` re-derives ``MockRunner``'s item digests from the
    cert's own subject, recipe core and item ids. So the digests move too -- one flipped hex
    character per item, the preimages dropped, ``channels.exact.hash`` recomputed over the new
    ones -- and what is left is a body carrying numbers no mock produced and no marker, which is
    exactly what a measured cert is.
    """
    body = copy.deepcopy(mock_body)
    body.pop(certmod.SYNTHETIC, None)
    for item in body["items"]:
        digest = item["token_ids_sha256"]
        item["token_ids_sha256"] = ("1" if digest[0] == "0" else "0") + digest[1:]
        item.pop("token_ids", None)
        item.pop("output_text", None)
    body["channels"]["exact"]["hash"] = FP.exact_hash(body["items"])
    return body


def test_c_mock_a_synthetic_run_cannot_challenge_a_measured_cert():
    """The exploit, end to end: verify a cert no synthetic runner produced, with the mock.

    Before the repair `MockRunner.subject(requested)` echoed the request, `comparable` was
    trivially empty, and the run reached a verdict with distances a challenge could be built
    from.  Now the body the mock would produce is labelled and the comparison fails on
    `body.synthetic` — a mismatch, exit 3, with no distances and nothing to file.
    """
    _battery, cert, resolver = reference()
    body = measured_body_from(cert["body"])  # a cert a model produced
    measured = F.make_cert(
        "fingerprint",
        subject=cert["subject"],
        recipe=cert["recipe"],
        body=body,
        refs=cert["refs"],
    )

    out = V.ref(measured, MockRunner(drift_items={"i01"}), resolver)

    assert out.verdict == "mismatch"
    assert out.exit_code == 3
    assert f"body.{certmod.SYNTHETIC}" in out.mismatched
    assert out.result_body.get("per_channel") in (None, {})
    with pytest.raises(ValueError):
        V.challenge_body(out, MockRunner().environment())


def test_c_mock_the_marker_is_in_the_signed_bytes_of_every_body_the_mock_builds():
    """`run_fingerprint` stamps it, so no minting path reaches a body without it."""
    body = FP.run_fingerprint(
        MockRunner(), F.weights_subject(), F.recipe(), pool_cert(), run_index=0, nuisance={}
    )
    assert body[certmod.SYNTHETIC] is True
    cert = F.make_cert("fingerprint", body=body, refs=[{"role": "battery", "id": F.BATTERY_ID}])
    outcome = certmod.check(cert)
    assert outcome.ok, outcome.reasons
    assert certmod.is_synthetic(cert) is True
    # The marker is inside the digested core, so removing it is a different cert, not an edit.
    stripped = copy.deepcopy(cert)
    stripped["body"].pop(certmod.SYNTHETIC)
    assert certmod.compute_id(stripped) != cert["id"]


# --------------------------------------------------------------------------- C-MOCK-2


def test_c_mock_2_deleting_the_marker_and_resigning_is_refused():
    """C-MOCK-2: `schema/fingerprint.json` did not require `synthetic` and did not restrict what
    else a body may carry, so deleting the key and re-signing produced an unmarked synthetic body
    that `comparable` read as a measurement -- the marker was load-bearing for the shipped tool
    and for nothing else.

    The repair does not require the key (a measured body does not carry it). It re-derives it:
    `MockRunner` computes from a hash whose inputs -- subject identity, recipe core, item ids --
    are in the cert, so the digests say for themselves whose they are.

    This test fails if the oracle is removed from `cert.check`."""
    body = FP.run_fingerprint(
        MockRunner(), F.weights_subject(), F.recipe(), pool_cert(), run_index=0, nuisance={}
    )
    stripped = copy.deepcopy(body)
    assert stripped.pop(certmod.SYNTHETIC) is True
    forged = F.make_cert(  # re-signed: a fresh id and a valid signature over the new bytes
        "fingerprint", body=stripped, refs=[{"role": "battery", "id": F.BATTERY_ID}]
    )
    assert certmod.is_synthetic(forged) is False  # the marker really is gone

    outcome = certmod.check(forged)
    assert outcome.ok is False
    assert any(certmod.SYNTHETIC in reason and "MockRunner" in reason for reason in outcome.reasons), \
        outcome.reasons
    # every item is named, not just the one that was sampled first
    named = runnermod.mock_derived_items(stripped, forged["subject"], forged["recipe"])
    assert named == [item["item_id"] for item in stripped["items"]]


def test_c_mock_2_a_runner_that_lies_about_itself_is_caught_too():
    """The marker is a class member a runner sets, so a runner can decline to set it. The oracle
    does not ask the runner either: it asks the numbers."""

    class QuietRunner(MockRunner):
        synthetic = False

    body = FP.run_fingerprint(
        QuietRunner(), F.weights_subject(), F.recipe(), pool_cert(), run_index=0, nuisance={}
    )
    assert certmod.SYNTHETIC not in body
    cert = F.make_cert("fingerprint", body=body, refs=[{"role": "battery", "id": F.BATTERY_ID}])
    assert certmod.check(cert).ok is False


@pytest.mark.parametrize("configured", [
    {},
    {"nuisance_items": {"i01"}},
    {"drift_items": {"i01", "i02"}},
    {"logprobs": False},
])
def test_c_mock_2_the_oracle_covers_the_variants_the_mock_can_take(configured):
    """Which variant list an item took is runner configuration no cert records, so the oracle
    tries all of them; a mock configured to move some items is still the mock."""
    body = FP.run_fingerprint(
        MockRunner(**configured), F.weights_subject(),
        F.recipe(decoding=F.decoding(batch_size=8)), pool_cert(),
        run_index=0, nuisance={"batch_size": 8},
    )
    stripped = {k: v for k, v in body.items() if k != certmod.SYNTHETIC}
    forged = F.make_cert(
        "fingerprint", body=stripped, refs=[{"role": "battery", "id": F.BATTERY_ID}]
    )
    assert certmod.check(forged).ok is False


def test_c_mock_2_a_body_no_mock_produced_is_not_accused():
    """The oracle is an accusation only when it MATCHES. A body whose digests reproduce nothing
    checks out with no marker, which is what a measured cert is -- "I could not check" must never
    read as "this is a fabrication", and it must never read as the reverse either."""
    _battery, cert, _resolver = reference()
    measured = F.make_cert(
        "fingerprint",
        subject=cert["subject"],
        recipe=cert["recipe"],
        body=measured_body_from(cert["body"]),
        refs=cert["refs"],
    )
    assert certmod.is_synthetic(measured) is False
    assert runnermod.mock_derived_items(
        measured["body"], measured["subject"], measured["recipe"]
    ) == []
    assert certmod.check(measured).ok, certmod.check(measured).reasons


@pytest.mark.parametrize("value", ["no", None, 0, "true", False])
def test_c_mock_2_the_schema_pins_the_markers_value(value):
    """The schema cannot require a key a measured body does not carry, but it can say what the
    key means when it IS carried. `cert.is_synthetic` fails closed on a hostile value; the schema
    refuses it outright, so the two do not have to agree about nonsense."""
    body = FP.run_fingerprint(
        MockRunner(), F.weights_subject(), F.recipe(), pool_cert(), run_index=0, nuisance={}
    )
    body[certmod.SYNTHETIC] = value
    cert = F.make_cert("fingerprint", body=body, refs=[{"role": "battery", "id": F.BATTERY_ID}])
    reasons = certmod.schema_errors(cert)
    assert any(certmod.SYNTHETIC in r for r in reasons), reasons
    # `synthetic: false` is the interesting one: `is_synthetic` reads it as "a model produced
    # this" (it is not the marker), so without the schema pin it would be the deletion attack
    # spelled differently. The oracle catches it from the other side.
    if value is False:
        assert certmod.is_synthetic(cert) is False
        assert runnermod.mock_derived_items(
            cert["body"], cert["subject"], cert["recipe"]
        ) != []


def test_c_mock_a_real_runner_is_not_labelled_and_the_two_do_not_compare():
    """A runner that never heard of the member is treated as a model; the default is fail-safe."""

    class QuietRunner(MockRunner):
        synthetic = False  # a stand-in for a runner class that predates the member

    real = FP.run_fingerprint(
        QuietRunner(), F.weights_subject(), F.recipe(), pool_cert(), run_index=0, nuisance={}
    )
    fake = FP.run_fingerprint(
        MockRunner(), F.weights_subject(), F.recipe(), pool_cert(), run_index=0, nuisance={}
    )
    assert certmod.SYNTHETIC not in real
    assert runnermod.is_synthetic(QuietRunner()) is False

    a = F.make_cert("fingerprint", body=real, refs=[{"role": "battery", "id": F.BATTERY_ID}])
    b = F.make_cert("fingerprint", body=fake, refs=[{"role": "battery", "id": F.BATTERY_ID}])
    assert certmod.comparable(a, b) == [f"body.{certmod.SYNTHETIC}"]
    # Section 9 rule 1 IS `comparable`, so the challenge layer gets this with no rule of its own.
    assert certmod.challenge_validity(a, b) == [f"body.{certmod.SYNTHETIC}"]
    # And it is never softened the way `precision` and `revision` are.
    assert not any(e.startswith("cross-subject:") for e in certmod.comparable(a, b))


@pytest.mark.parametrize("value", [None, "no", 0, {}, []])
def test_c_mock_the_marker_reads_fail_closed_on_a_hostile_value(value):
    """Present and not `false` is the marker; `synthetic: null` is not a measurement.

    `styxx/_data/v8_verify.js` implements the same rule, so the two implementations agree on
    hostile bytes as well as on well-formed ones (`tests/js/v8_verify.test.js`).
    """
    assert certmod.is_synthetic({"body": {certmod.SYNTHETIC: value}}) is True
    assert certmod.is_synthetic({"body": {certmod.SYNTHETIC: False}}) is False
    assert certmod.is_synthetic({"body": {}}) is False
    assert certmod.is_synthetic({"body": "not an object"}) is False
    assert certmod.is_synthetic(None) is False


def test_c_mock_the_log_refuses_a_synthetic_challenge_against_a_measured_target(tmp_path):
    log, battery, target, own = two_fingerprints(tmp_path)
    # The target is a cert a model produced; the challenger's own run is the mock's.
    measured = copy.deepcopy(target)
    assert certmod.SYNTHETIC not in measured["body"]
    synthetic_own = copy.deepcopy(own)
    synthetic_own["body"][certmod.SYNTHETIC] = True
    synthetic_own = F.make_cert(
        "fingerprint",
        recipe=synthetic_own["recipe"],
        refs=synthetic_own["refs"],
        body=synthetic_own["body"],
    )
    log.append(synthetic_own)
    with pytest.raises(AppendRefused) as exc:
        log.append(challenge_cert(measured["id"], synthetic_own))
    assert f"body.{certmod.SYNTHETIC}" in exc.value.reason


# --------------------------------------------------------------------------- C-DUPREF


def test_c_dupref_two_own_refs_are_refused_by_check(tmp_path):
    """The construction: `refs = [target, own=<A>, own=<B>]`, one cert with two readings."""
    log, battery, target, own = two_fingerprints(tmp_path)
    other = F.make_cert(
        "fingerprint",
        subject=F.weights_subject(precision="fp16"),
        recipe=F.recipe(battery=battery["id"]),
        refs=[{"role": "battery", "id": battery["id"]}],
        body=F.fingerprint_body(),
    )
    log.append(other)
    two_owns = challenge_cert(
        target["id"],
        own,
        refs=[
            {"role": "target", "id": target["id"]},
            {"role": "own", "id": other["id"]},   # what a reader walking refs sees
            {"role": "own", "id": own["id"]},     # what the log's last-wins loop saw
        ],
    )
    reasons = certmod.check(two_owns).reasons
    assert any("named more than once" in r and "'own'" in r for r in reasons)
    with pytest.raises(AppendRefused) as exc:
        log.append(two_owns)
    assert "invalid:" in exc.value.reason and "more than once" in exc.value.reason


def test_c_dupref_the_log_refuses_two_own_refs_on_its_own_without_check(tmp_path):
    """Defence in depth: the log does not rely on `cert.check` having run first."""
    log, battery, target, own = two_fingerprints(tmp_path)
    doubled = copy.deepcopy(
        challenge_cert(
            target["id"],
            own,
            refs=[
                {"role": "target", "id": target["id"]},
                {"role": "own", "id": own["id"]},
                {"role": "own", "id": own["id"]},
            ],
        )
    )
    with pytest.raises(AppendRefused) as exc:
        log._check_challenge_subject(doubled)
    assert "2 own refs" in exc.value.reason


@pytest.mark.parametrize("role", sorted(certmod.MULTI_ROLES))
def test_c_dupref_the_plural_roles_still_repeat(role):
    """`run`, `result` and `robustness` are plural by construction and are not refused."""
    rids = [F.fake_id(f"{role}-{k}") for k in range(3)]
    c = F.make_cert(
        "promotion" if role in ("result", "robustness") else "fingerprint",
        subject={} if role in ("result", "robustness") else F.weights_subject(),
        recipe={} if role in ("result", "robustness") else F.recipe(),
        body=(
            F.minimal_body("promotion")
            if role in ("result", "robustness")
            else F.fingerprint_body()
        ),
        refs=(
            [{"role": role, "id": rid} for rid in rids]
            if role in ("result", "robustness")
            else [{"role": "battery", "id": F.BATTERY_ID}]
            + [{"role": role, "id": rid} for rid in rids]
        ),
    )
    assert certmod.duplicate_roles(c) == []
    assert not any("more than once" in r for r in certmod.check(c).reasons)


def test_c_dupref_a_repeated_single_valued_role_is_refused_generally():
    """Not only on a challenge: the defect was in a reader, and every reader maps role -> cert."""
    c = F.make_cert(
        "fingerprint",
        refs=[
            {"role": "battery", "id": F.BATTERY_ID},
            {"role": "previous", "id": F.fake_id("a")},
            {"role": "previous", "id": F.fake_id("b")},
        ],
    )
    assert certmod.duplicate_roles(c) == ["previous"]
    assert any("'previous'" in r and "more than once" in r for r in certmod.check(c).reasons)


@pytest.mark.parametrize("ctype", sorted(certmod.ROLES_BY_TYPE))
def test_c_dupref_the_rule_is_one_rule_over_the_whole_role_table(ctype):
    """The rule is stated once (`cert.MULTI_ROLES`) and enforced once (`cert.check` step 7b), so
    it holds for every type and every role rather than for the two roles a challenge carries.
    This walks the table: on each cert type, doubling any role it may carry is refused unless
    that role is one of the three that are plural by construction.

    It fails if a type is special-cased, if a role is added to `MULTI_ROLES` without an argument
    for why that role names more than one cert, or if the check moves back onto the challenge."""
    for role in sorted(certmod.ROLES_BY_TYPE[ctype]):
        refs = [{"role": role, "id": F.fake_id("a")}, {"role": role, "id": F.fake_id("b")}]
        doubled = {"type": ctype, "refs": refs}
        assert certmod.duplicate_roles(doubled) == ([] if role in certmod.MULTI_ROLES else [role])


# --------------------------------------------------------------------------- C-MINT-UNCOMPUTED


def test_c_mint_uncomputed_nothing_is_signed_while_rule_one_is_uncomputed():
    """The library-level statement of the CLI refusal: `challenge_validity` reads two certs.

    A challenge minted from an unresolvable `--own` recorded `challenge_validity: null` in
    stdout and nothing in the signed bytes, so the artifact was indistinguishable from one whose
    rule 1 had passed.  `challenge_validity` itself never returns "unknown" — handed anything
    that is not a cert it returns the differences, and an empty list means the rule PASSED.
    """
    _battery, cert, _res = reference()
    assert certmod.challenge_validity(cert, {}) != []
    assert certmod.challenge_validity({}, {}) == []  # two empties really are alike
    # so "could not read the cert" can never be encoded as [] -- it has to be a refusal, and it
    # is, in `cli._challenge`.  See tests/test_v8_cli.py for the command that refuses.


# --------------------------------------------------------------------------- C-NONFP-TARGET


def test_c_nonfp_target_a_challenge_against_the_battery_cert_is_refused(tmp_path):
    """The construction: `target` naming the battery, so `_check_challenge_subject` returned."""
    log, battery, target, own = two_fingerprints(tmp_path)
    against_battery = challenge_cert(battery["id"], own)
    with pytest.raises(AppendRefused) as exc:
        log.append(against_battery)
    reason = exc.value.reason
    assert "target" in reason and "battery" in reason
    assert "fingerprint" in reason


def test_c_nonfp_target_the_role_table_is_per_cert_type():
    """`target` stays `*` in `ROLE_TYPES` because a `response` result names a challenge."""
    assert certmod.ROLE_TYPES["target"] == "*"
    assert certmod.role_type("challenge", "target") == "fingerprint"
    assert certmod.role_type("result", "target") == "*"
    assert certmod.role_type("challenge", "own") == "fingerprint"
    assert certmod.role_type("fingerprint", "previous") is None  # "this cert's own type"


# --------------------------------------------------------------------------- C3


def test_c3_the_challenge_body_says_what_produced_its_distances(tmp_path):
    log, battery, target, own = two_fingerprints(tmp_path)
    challenge = challenge_cert(target["id"], own)
    body = challenge["body"]
    assert body["subject"] == certmod.identity_fields(own["subject"])
    assert body["recipe_core"] == certmod.recipe_core(own["recipe"])
    assert log.append(challenge) == 3


@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"subject": {"kind": "weights", "hf_repo": "someone/else"}}, id="subject"),
        pytest.param(
            {"recipe_core": {"battery": F.BATTERY_ID, "decoding": {}, "chat_template_sha256": None,
                             "system_prompt_sha256": None}},
            id="recipe_core",
        ),
        pytest.param({"synthetic": True}, id="synthetic"),
    ],
)
def test_c3_a_body_that_disagrees_with_the_own_cert_is_refused(tmp_path, override):
    """A self-description a log will not check is a comment, so the log checks it."""
    log, battery, target, own = two_fingerprints(tmp_path)
    lying = challenge_cert(target["id"], own, **override)
    with pytest.raises(AppendRefused) as exc:
        log.append(lying)
    assert exc.value.reason.startswith("challenge:")


def test_c3_a_challenge_carrying_no_self_report_does_not_even_check():
    """The pre-repair shape — `{per_channel, coverage, environment}` — is refused by the schema."""
    bare = F.make_cert(
        "challenge",
        subject={},
        recipe={},
        body={
            "per_channel": {"exact": {"distance": 0.5, "target_floor": 0.0}},
            "coverage": "within",
            "environment": {"runtime": {"framework": "mock"}},
        },
        refs=[{"role": "target", "id": F.TARGET_ID}, {"role": "own", "id": F.OWN_ID}],
    )
    reasons = certmod.check(bare).reasons
    assert any("'subject' is a required property" in r for r in reasons)
    assert any("'recipe_core' is a required property" in r for r in reasons)


def test_c3_the_restated_battery_id_does_not_need_a_ref_of_its_own(tmp_path):
    """A-09 is not weakened by accident: only the two named paths are exempt, and the exemption
    is checkable — the log compares the restated `recipe_core` against the `own` cert."""
    assert certmod.RESTATED_ID_PATHS == {
        "body/recipe_core/battery", "body/observed_recipe_core/battery"
    }
    log, battery, target, own = two_fingerprints(tmp_path)
    challenge = challenge_cert(target["id"], own)
    assert battery["id"] not in {rid for _, rid in certmod.refs(challenge)}
    assert battery["id"] not in certmod.embedded_ids(challenge)
    assert certmod.check(challenge).ok
    # A cert id anywhere ELSE in the body is still refused.
    with_extra = copy.deepcopy(challenge["body"])
    with_extra["note"] = battery["id"]
    smuggled = F.make_cert(
        "challenge", subject={}, recipe={}, body=with_extra, refs=challenge["refs"]
    )
    assert any("embedded id" in r for r in certmod.check(smuggled).reasons)

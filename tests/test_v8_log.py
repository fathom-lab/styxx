"""styxx.v8.log — the log v0 (contract section 2, spec section 8).

What is pinned here:

* the append refusal matrix, one test per reason the contract names;
* the storage layout, byte for byte: canonical entry bytes, no trailing newline, no CR;
* STH signing and verification, including the wrong key, a forged root, and a reused tree size
  with a different root;
* inclusion proofs at historical tree sizes and consistency across three heads;
* ``mirror`` on a clean log and on four tampered copies, each reported under the right key;
* a CRLF-injected entry reading as tamper.

Nothing here skips.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from styxx.v8 import cert as certmod
from styxx.v8 import fingerprint as FP
from styxx.v8 import floor as FLOOR
from styxx.v8 import keys, merkle
from styxx.v8.jcs import canonical_bytes
from styxx.v8.log import (
    APPEND_TIME_META_KEYS,
    ASSERTED_META_KEYS,
    DERIVED_META_KEYS,
    AppendRefused,
    Log,
    mirror,
    verify_consistency,
    verify_entry,
    verify_inclusion,
    verify_sth,
)
from tests import v8_fixtures as F

ISSUER_SEED, ISSUER_PUB = F.keypair("issuer")
OTHER_SEED, OTHER_PUB = F.keypair("other")
LOG_SEED, LOG_PUB = F.keypair("log-key")

TS = "2026-09-08T18:00:00Z"
TS2 = "2026-09-08T19:00:00Z"


# ----------------------------------------------------------------- helpers

def roster(*labels: str, from_index: int = 0, retired=None) -> list[dict]:
    return [
        {
            "name": F.ISSUER_NAME,
            "key": F.public_key(label),
            "from_index": from_index,
            "retired_at_index": retired,
        }
        for label in labels
    ]


def fresh(root, *labels: str, **kw) -> Log:
    return Log.init(root, LOG_PUB, roster(*(labels or ("issuer",)), **kw))


def pool_cert(**over) -> dict:
    """A pool-v1 battery: the root of the battery chain, so no recipe and no refs (section 4.5)."""
    return F.make_cert("battery", body=F.battery_body("pool-v1"), **over)


def battery_cert(*, kind: str = "fixed-v1", redacted: bool = False, n: int = 3, **over) -> dict:
    """A fixed-v1 (or pool-v1) battery: also a root. ``n`` moves the items, so two of these with
    different ``n`` are two different certs."""
    body = F.battery_body(kind, n=n)
    body["redacted"] = redacted
    return F.make_cert("battery", body=body, **over)


def fp_cert(battery_id: str, *, extra_refs=(), body_extra=None, **body_kw) -> dict:
    body = F.fingerprint_body(**body_kw)
    if body_extra:
        body.update(body_extra)
    refs = [{"role": "battery", "id": battery_id}] + [dict(r) for r in extra_refs]
    return F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery_id),
        refs=refs,
        body=body,
    )


def canary_cert(pool_id: str, reference_id: str) -> dict:
    """A canary-v1 battery: the only non-root battery kind (section 4.5). It names the pool in
    its recipe and the fingerprint it was selected against in refs; schema/battery.json requires
    both."""
    return F.make_cert(
        "battery",
        recipe=F.recipe(battery=pool_id),
        refs=F.canary_battery_refs(pool_id, reference_id),
        body=F.canary_battery_body(),
    )


def chain(log: Log) -> dict:
    """pool -> reference fingerprint on the pool -> canary battery -> fingerprint (4 entries)."""
    pool = pool_cert()
    log.append(pool)
    reference = fp_cert(pool["id"])
    log.append(reference)
    canary = canary_cert(pool["id"], reference["id"])
    log.append(canary)
    fingerprint = fp_cert(canary["id"])
    log.append(fingerprint)
    return {"pool": pool, "reference": reference, "battery": canary, "fingerprint": fingerprint}


def five_entry_log(root) -> Log:
    """A log with five entries: the four of ``chain`` plus a noise-plan prereg."""
    log = fresh(root)
    chain(log)
    log.append(F.make_cert("prereg", body={"kind": "noise-plan", "runs": 5}))
    return log


# ----------------------------------------------------------------- layout

def test_init_creates_the_layout_and_the_four_commands(tmp_path):
    log = fresh(tmp_path / "log")
    for name in ("entries", "blobs", "sth", "keys"):
        assert (log.path / name).is_dir()
    readme = (log.path / "README.md").read_bytes()
    assert b"\r" not in readme
    for verb in (b"verify-cert", b"verify-sth", b"verify-inclusion", b"verify-consistency"):
        assert verb in readme
    assert keys.load_public(log.keys_dir / "log.pub") == LOG_PUB
    assert json.loads((log.keys_dir / "issuers.json").read_bytes().decode("utf-8"))[0]["key"] == F.public_key("issuer")
    assert log.size() == 0
    assert log.root() == merkle.EMPTY_ROOT


def test_entry_bytes_are_canonical_with_no_newline_and_no_cr(tmp_path):
    log = fresh(tmp_path / "log")
    built = chain(log)
    for index, name in enumerate(("pool", "reference", "battery", "fingerprint")):
        raw = log.entry_bytes(index)
        assert raw == canonical_bytes(built[name])
        assert not raw.endswith(b"\n")
        assert b"\r" not in raw
        assert log.entry_path(index).name == f"{index:08d}.json"
        assert log.entry_path(index).parent.name == "000000"
        assert log.cert(index)["id"] == built[name]["id"]
        assert log.find(built[name]["id"]) == index
    assert log.find(F.fake_id("nothing here")) is None
    assert log.size() == 4


def test_meta_records_index_id_type_and_public(tmp_path):
    log = fresh(tmp_path / "log")
    built = chain(log)
    meta = log.meta(3)
    # Every key the file carries is one of three named sets, and the sets are the whole point:
    # DERIVED_* is re-derived on read and refused when it disagrees (A-META), ASSERTED_* is the
    # timestamp nothing can recompute, APPEND_TIME_* is the written-down hole.
    assert {"index", "id", "type", "public", "appended_at"} <= set(meta)
    assert set(meta) <= set(DERIVED_META_KEYS) | set(ASSERTED_META_KEYS) | set(
        APPEND_TIME_META_KEYS
    )
    assert meta["index"] == 3
    assert meta["id"] == built["fingerprint"]["id"]
    assert meta["type"] == "fingerprint"
    assert meta["public"] is True
    assert log.meta_disagreement(3) == []
    assert log.meta_path(3).read_bytes().endswith(b"\n")
    assert b"\r" not in log.meta_path(3).read_bytes()


def test_log_id_is_the_hash_of_the_raw_public_key(tmp_path):
    log = fresh(tmp_path / "log")
    assert log.log_id() == "sha256:" + hashlib.sha256(LOG_PUB).hexdigest()


def test_blobs_are_content_addressed_and_a_wrong_hash_is_refused(tmp_path):
    log = fresh(tmp_path / "log")
    built = chain(log)
    payload = b'{"items": ["blob bytes"]}'
    key = "sha256:" + hashlib.sha256(payload).hexdigest()
    previous = [{"role": "previous", "id": built["fingerprint"]["id"]}]
    good = fp_cert(built["battery"]["id"], run_index=7, extra_refs=previous, body_extra={"items_blob": key})
    index = log.append(good, {key: payload})
    assert log.blob_path(key).read_bytes() == payload
    assert log.cert(index)["body"]["items_blob"] == key

    bad = fp_cert(built["battery"]["id"], run_index=8, extra_refs=previous, body_extra={"items_blob": key})
    with pytest.raises(AppendRefused) as exc:
        log.append(bad, {key: b"different bytes"})
    assert exc.value.reason.startswith("blob:")


# ----------------------------------------------------------------- the refusal matrix

def test_refuses_a_cert_that_does_not_check_out(tmp_path):
    log = fresh(tmp_path / "log")
    tampered = dict(pool_cert())
    tampered["created"] = "2026-09-09T18:00:00Z"  # signed bytes no longer match
    with pytest.raises(AppendRefused) as exc:
        log.append(tampered)
    assert exc.value.reason.startswith("invalid:")
    assert "id: does not recompute" in exc.value.reason
    assert log.size() == 0


def test_refuses_an_issuer_key_outside_the_roster(tmp_path):
    log = fresh(tmp_path / "log")
    stranger = F.make_cert("battery", issuer_label="other", body=F.battery_body("pool-v1"))
    with pytest.raises(AppendRefused) as exc:
        log.append(stranger)
    assert exc.value.reason.startswith("issuer:")
    assert F.public_key("other") in exc.value.reason


def test_refuses_a_key_retired_at_this_index(tmp_path):
    log = Log.init(tmp_path / "log", LOG_PUB, roster("issuer", retired=1))
    log.append(pool_cert())
    with pytest.raises(AppendRefused) as exc:
        log.append(F.make_cert("prereg", body={"kind": "noise-plan", "runs": 5}))
    assert exc.value.reason.startswith("issuer:")


def test_a_log_without_a_roster_file_refuses_every_issuer(tmp_path):
    """L10c. This test used to be named ``..._admits_any_issuer`` and asserted the append.

    That was the defect, not a feature: ``issuers()`` returned ``None`` for a missing
    ``keys/issuers.json`` and append step 2 read ``if roster is not None``, so the control the
    attacker ran is two lines — with a roster present a rogue key is refused; delete the file and
    the same rogue key appends at exit 0. A security check whose unset configuration means "allow
    everything" is an off switch reachable by ``rm``. See ``tests/test_v8_log_integrity.py`` for
    the open-policy marker that replaces it.
    """
    log = Log(tmp_path / "log")  # no keys/issuers.json: no admission policy at all
    assert log.issuers() is None
    assert log.issuer_policy()["policy"] == "absent"
    with pytest.raises(AppendRefused) as exc:
        log.append(F.make_cert("battery", issuer_label="other", body=F.battery_body("pool-v1")))
    assert exc.value.reason.startswith("issuer:")
    assert "states no admission policy" in exc.value.reason
    assert log.size() == 0


def test_refuses_a_ref_that_does_not_resolve(tmp_path):
    log = fresh(tmp_path / "log")
    log.append(pool_cert())
    orphan = fp_cert(F.BATTERY_ID)  # a battery id that is in no leaf
    with pytest.raises(AppendRefused) as exc:
        log.append(orphan)
    assert exc.value.reason.startswith("refs:")
    assert "does not resolve" in exc.value.reason


def test_refuses_a_ref_whose_type_does_not_match_the_role(tmp_path):
    log = fresh(tmp_path / "log")
    built = chain(log)
    wrong = fp_cert(built["fingerprint"]["id"])  # role battery pointing at a fingerprint
    with pytest.raises(AppendRefused) as exc:
        log.append(wrong)
    assert exc.value.reason.startswith("refs:")
    assert "resolves to a fingerprint cert, not a battery cert" in exc.value.reason


def test_refuses_a_duplicate_id(tmp_path):
    log = fresh(tmp_path / "log")
    built = chain(log)
    with pytest.raises(AppendRefused) as exc:
        log.append(built["fingerprint"])
    assert exc.value.reason.startswith("duplicate:")
    assert log.size() == 4


def test_refuses_a_second_baseline_without_a_previous_ref(tmp_path):
    log = fresh(tmp_path / "log")
    built = chain(log)
    second = fp_cert(built["battery"]["id"], run_index=1)
    assert certmod.comparable(built["fingerprint"], second) == []
    with pytest.raises(AppendRefused) as exc:
        log.append(second)
    assert exc.value.reason.startswith("baseline:")

    with_previous = fp_cert(
        built["battery"]["id"],
        run_index=1,
        extra_refs=[{"role": "previous", "id": built["fingerprint"]["id"]}],
    )
    assert log.append(with_previous) == 4
    assert log.previous_comparable(fp_cert(built["battery"]["id"], run_index=2)) == 4


def test_runs_under_one_noise_plan_are_not_new_baselines(tmp_path):
    log = fresh(tmp_path / "log")
    built = chain(log)
    plan = F.make_cert("prereg", body={"kind": "noise-plan", "runs": 5})
    log.append(plan)
    other_battery = battery_cert(n=4)
    log.append(other_battery)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    run0 = fp_cert(other_battery["id"], run_index=0, extra_refs=plan_ref)
    run1 = fp_cert(other_battery["id"], run_index=1, extra_refs=plan_ref)
    assert log.append(run0) == 6
    assert log.append(run1) == 7  # same plan: one floor, not a new baseline

    outsider = fp_cert(other_battery["id"], run_index=2)  # no plan, no previous
    with pytest.raises(AppendRefused) as exc:
        log.append(outsider)
    assert exc.value.reason.startswith("baseline:")


# ----------------------------------------------------------------- section 5.1 step 2: the plan
# is a commitment. The first real floor declared batch_size 1|8|32 and ran five times at batch 1,
# so every pairwise distance was 0.0, the floor was 0.0 on every channel, and every later
# difference exceeded it (`papers/v8/vacuous_floor_2026_09_09/`). The certs were all valid.


def plan_cert(*factors, runs: int = 5) -> dict:
    """A noise-plan prereg declaring ``(factor, values)`` pairs (section 5.1 step 1)."""
    return F.make_cert(
        "prereg",
        body={
            "kind": "noise-plan",
            "runs": runs,
            "nuisance": [{"factor": f, "values": list(v)} for f, v in factors],
            "environment": {"hardware": {"gpu": "none", "driver": "none", "count": 0}},
        },
    )


def resign(cert: dict) -> dict:
    """Re-derive the id and the signature over a mutated core — the forger's move, and the
    reason a check that only looks at the envelope catches nothing."""
    core = {k: v for k, v in cert.items() if k not in ("id", "sig")}
    return certmod.sign(core, ISSUER_SEED)


def _decoding_for(assignment: dict) -> dict:
    """The ``recipe.decoding`` a run under ``assignment`` really ran: the log refuses a floor run
    whose nuisance label contradicts its own recipe (R-EXEC)."""
    overrides = {
        factor: assignment[factor]
        for factor in ("batch_size", "padding_side")
        if factor in assignment
    }
    return F.decoding(**overrides)


def floor_certs(battery_id: str, plan: dict, assignments: list[dict]) -> dict:
    """The R−1 run certs and the canonical of one floor, in section 5.1 step 5 append order.

    ``assignments[k]`` is what run ``k`` recorded under ``body.nuisance``; run 0's is the
    canonical's, because the canonical IS run 0.

    The whole plan CERT, not its id: the floor's ``covers`` and ``not_covered`` are derived from
    the plan's ``nuisance`` and ``environment`` and the log refuses a disagreement (A-COVER).
    """
    plan_id = plan["id"]
    plan_ref = [{"role": "noise_plan", "id": plan_id}]
    runs = []
    for k, assignment in enumerate(assignments[1:], start=1):
        body = F.fingerprint_body(run_index=k)
        body["nuisance"].update(assignment)
        runs.append(
            F.make_cert(
                "fingerprint",
                # The recipe carries the execution THAT run ran under: the log refuses a run
                # whose nuisance label and own `recipe.decoding` disagree (R-EXEC).
                recipe=F.recipe(battery=battery_id, decoding=_decoding_for(assignment)),
                refs=[{"role": "battery", "id": battery_id}] + plan_ref,
                body=body,
            )
        )
    run_ids = [c["id"] for c in runs]
    canonical_body = F.fingerprint_body(run_index=0)
    canonical_body["nuisance"].update(assignments[0])
    canonical_body["noise_floor"] = F.noise_floor_block(
        plan=plan_id,
        plan_body=plan["body"],
        runs=run_ids,
        bodies=[canonical_body] + [c["body"] for c in runs],
    )
    canonical = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery_id, decoding=_decoding_for(assignments[0])),
        refs=(
            [{"role": "battery", "id": battery_id}]
            + plan_ref
            + [{"role": "run", "id": rid} for rid in run_ids]
        ),
        body=canonical_body,
    )
    return {"runs": runs, "canonical": canonical}


def floor_log(tmp_path, *factors, assignments: list[dict]) -> tuple[Log, dict]:
    """A log holding the battery, the plan and the R−1 run certs; the canonical is not appended."""
    log = fresh(tmp_path)
    battery = battery_cert(n=4)
    log.append(battery)
    plan = plan_cert(*factors)
    log.append(plan)
    built = floor_certs(battery["id"], plan, assignments)
    for cert in built["runs"]:
        log.append(cert)
    return log, built


ONE_ASSIGNMENT = [
    {"batch_size": 1, "item_order": "canonical"},
    {"batch_size": 1, "item_order": "perm11"},
    {"batch_size": 1, "item_order": "perm12"},
    {"batch_size": 1, "item_order": "perm11"},
    {"batch_size": 1, "item_order": "perm12"},
]
VARIED = [
    {"batch_size": 1, "item_order": "canonical"},
    {"batch_size": 8, "item_order": "perm11"},
    {"batch_size": 32, "item_order": "perm12"},
    {"batch_size": 1, "item_order": "perm11"},
    {"batch_size": 1, "item_order": "perm12"},
]
FACTORS = (("batch_size", ["1", "8", "32"]), ("item_order", ["canonical", "perm11", "perm12"]))


def test_refuses_a_floor_whose_runs_never_varied_a_declared_factor(tmp_path):
    """The defect itself: batch_size declared 1|8|32, five runs at batch 1, item_order the only
    thing that moved -- and at batch 1 item order cannot move anything, because each item is its
    own forward pass. The refusal names the factor that was declared and not varied."""
    log, built = floor_log(tmp_path / "log", *FACTORS, assignments=ONE_ASSIGNMENT)
    with pytest.raises(AppendRefused) as exc:
        log.append(built["canonical"])
    reason = exc.value.reason
    assert reason.startswith("floor:")
    assert "'batch_size'" in reason
    assert "all 5 runs ran at '1'" in reason
    assert "item_order" not in reason  # item_order DID vary; the refusal names only the factor


def test_a_floor_that_varied_every_declared_factor_is_appended(tmp_path):
    """The same shape with the plan actually applied: five assignments, both factors moving."""
    log, built = floor_log(tmp_path / "log", *FACTORS, assignments=VARIED)
    index = log.append(built["canonical"])
    assert log.cert(index)["id"] == built["canonical"]["id"]
    assert log.cert(index)["body"]["noise_floor"]["runs"] == [c["id"] for c in built["runs"]]


def test_refuses_a_floor_whose_runs_do_not_record_a_declared_factor_at_all(tmp_path):
    """A plan that declares a factor no run cert even mentions is the same failure one step
    earlier: nothing in the bytes says what the factor was during the runs."""
    silent = [dict(a) for a in VARIED]
    for assignment in silent:
        assignment.pop("item_order")
    log, built = floor_log(tmp_path / "log", *FACTORS, assignments=silent)
    with pytest.raises(AppendRefused) as exc:
        log.append(built["canonical"])
    assert "'item_order'" in exc.value.reason
    assert "not one of the 5 runs records it" in exc.value.reason


def test_refuses_a_floor_only_some_of_whose_runs_record_a_declared_factor(tmp_path):
    partial = [dict(a) for a in VARIED]
    partial[2].pop("item_order")
    log, built = floor_log(tmp_path / "log", *FACTORS, assignments=partial)
    with pytest.raises(AppendRefused) as exc:
        log.append(built["canonical"])
    assert "'item_order'" in exc.value.reason
    assert "only 4 of the 5 runs record it" in exc.value.reason


def test_refuses_a_floor_whose_runs_used_a_value_the_plan_did_not_declare(tmp_path):
    stray = [dict(a) for a in VARIED]
    stray[3]["item_order"] = "perm99"
    log, built = floor_log(tmp_path / "log", *FACTORS, assignments=stray)
    with pytest.raises(AppendRefused) as exc:
        log.append(built["canonical"])
    assert "'item_order'" in exc.value.reason
    assert "['perm99']" in exc.value.reason


def test_a_factor_with_one_declared_value_commits_to_no_variation(tmp_path):
    """A plan is checked against what it declared. One value is not a promise to vary, so a floor
    whose runs all sit at it is a floor about the factors that DID have values.

    ``padding_side`` here is the pinned factor and ``batch_size`` is the one that carries the
    floor; a plan whose ONLY factor names one value is refused, and that is the next test.
    """
    factors = (("padding_side", ["left"]), ("batch_size", ["1", "8", "32"]))
    log, built = floor_log(tmp_path / "log", *factors, assignments=VARIED)
    assert log.append(built["canonical"]) == log.size() - 1


def test_the_run_certs_of_a_floor_are_not_themselves_checked(tmp_path):
    """The check is on the cert that makes the floor claim. A run cert varies nothing on its own
    and appends under its plan exactly as before (section 5.5's scope, restated for section 5.1)."""
    log, built = floor_log(tmp_path / "log", *FACTORS, assignments=ONE_ASSIGNMENT)
    assert [log.cert(i)["type"] for i in log.indices()] == [
        "battery", "prereg", "fingerprint", "fingerprint", "fingerprint", "fingerprint"
    ]
    assert all("noise_floor" not in log.cert(i)["body"] for i in log.indices()[2:])


def test_a_plan_that_declares_no_factors_is_not_checked(tmp_path):
    """`prereg noise-plan` cannot mint one (--nuisance is required), but a hand-built plan can
    exist and the check has nothing to compare against; the floor appends and says nothing.

    It still has to name an environment, and its ``covers`` is then the empty list while
    ``not_covered`` is every leaf of that environment -- which is A-COVER working rather than an
    exception to it: a plan that varied nothing covers nothing, and says so."""
    log = fresh(tmp_path / "log")
    battery = battery_cert(n=4)
    log.append(battery)
    plan = F.make_cert(
        "prereg",
        body={
            "kind": "noise-plan",
            "runs": 5,
            # No `nuisance` key at all: schema/prereg.json requires a non-empty list WHEN the key
            # is present, so a plan that declares no factors is one that omits it.
            "environment": {"hardware": {"gpu": "none", "driver": "none", "count": 0}},
        },
    )
    log.append(plan)
    built = floor_certs(battery["id"], plan, ONE_ASSIGNMENT)
    for cert in built["runs"]:
        log.append(cert)
    assert log.append(built["canonical"]) == 6


def test_a_floor_can_never_reach_the_check_without_its_plan(tmp_path):
    """Why the "plan does not resolve" branch of the floor check is defensive and not the gate.

    A-09 ("every cert id in a cert appears in refs") is checked by ``cert.check``, and step 3
    resolves every ref, so a canonical fingerprint naming a plan the log does not hold is refused
    two rules earlier. The floor check therefore always has the plan's bytes to compare against.
    """
    log = fresh(tmp_path / "log")
    battery = battery_cert(n=4)
    log.append(battery)
    plan = plan_cert(*FACTORS)
    log.append(plan)
    body = F.fingerprint_body(
        run_index=0, noise_floor=F.noise_floor_block(plan=plan["id"], runs=[])
    )
    body["noise_floor"]["plan"] = F.fake_id("a plan nobody logged")
    canonical = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"]),
        refs=[{"role": "battery", "id": battery["id"]}, {"role": "noise_plan", "id": plan["id"]}],
        body=body,
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert "is not in refs" in exc.value.reason

    # ...and one that puts the unlogged plan in refs is refused by step 3 instead.
    unlogged = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"]),
        refs=[
            {"role": "battery", "id": battery["id"]},
            {"role": "noise_plan", "id": body["noise_floor"]["plan"]},
        ],
        body=body,
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(unlogged)
    assert exc.value.reason.startswith("refs:") and "does not resolve" in exc.value.reason


# ----------------------------------------------------------------- the floor's executions
#
# The eight constructions of papers/v8/challenge_and_attack_2026_09_09 all satisfied the label
# check above and named ONE computation. The battery here has 8 items, which is what makes the
# coercion attacks reproducible: 8, 16 and 32 are one batch of eight.

BATCH_LABELS_THAT_COERCE = [
    {"batch_size": 8}, {"batch_size": 16}, {"batch_size": 32}, {"batch_size": 8}, {"batch_size": 16},
]


def eight_item_floor(tmp_path, *factors, assignments, batch=None):
    """A floor over an 8-item battery, each run's recipe carrying its own declared batch size."""
    log = fresh(tmp_path)
    battery = battery_cert(n=8)
    log.append(battery)
    plan = plan_cert(*factors)
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    runs = []
    for k, assignment in enumerate(assignments[1:], start=1):
        size = int(assignment.get("batch_size", batch or 1))
        body = F.fingerprint_body(n=8, run_index=k, batch_size=size)
        body["nuisance"].update(assignment)
        cert = F.make_cert(
            "fingerprint",
            recipe=F.recipe(battery=battery["id"], decoding=_decoding_for({**assignment, "batch_size": size})),
            refs=[{"role": "battery", "id": battery["id"]}] + plan_ref,
            body=body,
        )
        log.append(cert)
        runs.append(cert)
    size = int(assignments[0].get("batch_size", batch or 1))
    body = F.fingerprint_body(n=8, run_index=0, batch_size=size)
    body["nuisance"].update(assignments[0])
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in runs],
        bodies=[body] + [c["body"] for c in runs],
    )
    canonical = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"], decoding=_decoding_for({**assignments[0], "batch_size": size})),
        refs=(
            [{"role": "battery", "id": battery["id"]}] + plan_ref
            + [{"role": "run", "id": c["id"]} for c in runs]
        ),
        body=body,
    )
    return log, canonical


def test_refuses_a_batch_size_the_item_count_coerces(tmp_path):
    """F4: 8, 16 and 32 over eight items are three declared values and one batch of eight.
    Nothing is faked -- the certs record what the runner did -- and nothing varied."""
    log, canonical = eight_item_floor(
        tmp_path / "log",
        ("batch_size", ["8", "16", "32"]),
        assignments=BATCH_LABELS_THAT_COERCE,
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    reason = exc.value.reason
    assert reason.startswith("floor:") and "'batch_size'" in reason
    assert "could tell apart" in reason


def test_a_batch_size_the_item_count_does_not_coerce_is_appended(tmp_path):
    """F8's shape, which is NOT refused: 4 really is a different partition of eight items.
    The floor is thin -- two computations, six of ten pairs comparing a run with itself -- and
    the census is where a reader sees that rather than being told it is fine."""
    assignments = [
        {"batch_size": 8}, {"batch_size": 16}, {"batch_size": 4},
        {"batch_size": 8}, {"batch_size": 16},
    ]
    log, canonical = eight_item_floor(
        tmp_path / "log", ("batch_size", ["8", "16", "4"]), assignments=assignments
    )
    index = log.append(canonical)
    census = log.meta(index)["floor"]
    assert census["runs"] == 5 and census["pairs"] == 10
    assert census["executions"] == 2
    assert census["pairs_same_execution"] == 6
    assert census["factors_without_demonstrated_effect"] == []


def test_refuses_an_item_order_permuted_at_batch_size_one(tmp_path):
    """F1, the flagship: five real item orders at batch 1, where each item is its own forward
    pass, so permuting them permutes independent computations and every distance is 0."""
    orders = ["canonical", "p1", "p2", "p3", "p4"]
    log, canonical = eight_item_floor(
        tmp_path / "log",
        ("item_order", orders),
        assignments=[{"batch_size": 1, "item_order": v} for v in orders],
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert "'item_order'" in exc.value.reason and "could tell apart" in exc.value.reason


def test_the_same_item_order_factor_is_accepted_where_a_batch_can_see_it(tmp_path):
    """The mirror of the test above, and the reason it is a refusal about the CONFIGURATION and
    not about the factor: at batch 8 the same three orders group the items three ways."""
    orders = ["canonical", "p1", "p2"]
    assignments = [{"batch_size": 8, "item_order": orders[k % 3]} for k in range(5)]
    log, canonical = eight_item_floor(
        tmp_path / "log", ("item_order", orders), assignments=assignments
    )
    assert log.append(canonical) == log.size() - 1


def test_refuses_a_padding_side_no_batch_is_large_enough_to_apply(tmp_path):
    """F3: the value is written into each run's recipe and the runner really sets it; at batch 1
    the tokenizer pads one sequence to its own length and no pad token is ever emitted."""
    sides = ["left", "right"]
    log, canonical = eight_item_floor(
        tmp_path / "log",
        ("padding_side", sides),
        assignments=[{"batch_size": 1, "padding_side": sides[k % 2]} for k in range(5)],
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert "'padding_side'" in exc.value.reason and "could tell apart" in exc.value.reason


def test_refuses_a_plan_whose_only_factor_declares_a_single_value(tmp_path):
    """F2: `prereg noise-plan --runs 5 --nuisance batch_size=1` is accepted by the CLI, every
    factor is skipped as single-valued, and the verdict header prints `covers=batch_size` over a
    floor that covers nothing."""
    log, canonical = eight_item_floor(
        tmp_path / "log",
        ("batch_size", ["1"]),
        assignments=[{"batch_size": 1} for _ in range(5)],
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert "not one of them is a factor this floor" in exc.value.reason


def test_a_factor_no_process_here_can_apply_is_reported_and_not_refused(tmp_path):
    """A floor across two physical GPUs is legitimate (section 5.1 step 2) and its batches are
    identical, so the check cannot derive anything about it. It says so instead of guessing --
    and it still cannot be the thing that satisfies the rule, which is why this floor also
    carries a batch_size that does vary."""
    assignments = [
        {"batch_size": 8, "gpu": "4070"}, {"batch_size": 4, "gpu": "a100"},
        {"batch_size": 8, "gpu": "4070"}, {"batch_size": 4, "gpu": "a100"},
        {"batch_size": 8, "gpu": "4070"},
    ]
    log, canonical = eight_item_floor(
        tmp_path / "log",
        ("batch_size", ["8", "4"]),
        ("gpu", ["4070", "a100"]),
        assignments=assignments,
    )
    index = log.append(canonical)
    assert log.meta(index)["floor"]["factors_not_derivable"] == ["gpu"]


def test_a_zero_floor_over_two_computations_appends_and_is_labelled(tmp_path):
    """The line this check draws. A configuration that genuinely repeats itself across genuinely
    different computations honestly produces a floor of 0.0; that is a finding about the box, it
    stays appendable, and the metadata says which channels are zero so a later verdict cannot
    lean on them silently."""
    assignments = [
        {"batch_size": 8}, {"batch_size": 4}, {"batch_size": 8},
        {"batch_size": 4}, {"batch_size": 8},
    ]
    log, canonical = eight_item_floor(
        tmp_path / "log", ("batch_size", ["8", "4"]), assignments=assignments
    )
    block = canonical["body"]["noise_floor"]
    assert all(per["floor"] == 0.0 for per in block["per_channel"].values())
    index = log.append(canonical)
    census = log.meta(index)["floor"]
    assert census["zero_channels"] == sorted(block["per_channel"])
    assert census["all_channels_zero"] is True
    assert census["executions"] == 2
    assert census["state_kinds"] == ["batches"]


# ----------------------------------------------------------------- the anchored floor
#
# F6, F7, R-EXEC and R-EXEC-2 of papers/v8/challenge_and_attack_2026_09_09 all left the floor's
# NUMERALS free: nothing re-derived `floor_c` from the runs the block names, and nothing asked
# whether those runs were all the runs the plan committed to. These tests pin the refusals.


def subset_floor(tmp_path, named_count: int, *, assignments=None):
    """A floor over an 8-item battery whose `noise_floor.runs` names ``named_count`` of the four
    runs the plan committed to, with the numbers those named runs really produce."""
    assignments = assignments or [
        {"batch_size": 8}, {"batch_size": 4}, {"batch_size": 8},
        {"batch_size": 4}, {"batch_size": 8},
    ]
    log = fresh(tmp_path)
    battery = battery_cert(n=8)
    log.append(battery)
    plan = plan_cert(("batch_size", ["8", "4"]))
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    runs = []
    for k, a in enumerate(assignments[1:], start=1):
        body = F.fingerprint_body(n=8, run_index=k, batch_size=a["batch_size"])
        body["nuisance"].update(a)
        cert = F.make_cert(
            "fingerprint",
            recipe=F.recipe(battery=battery["id"], decoding=_decoding_for(a)),
            refs=[{"role": "battery", "id": battery["id"]}] + plan_ref,
            body=body,
        )
        log.append(cert)
        runs.append(cert)
    named = runs[:named_count]
    body = F.fingerprint_body(n=8, run_index=0, batch_size=8)
    body["nuisance"].update(assignments[0])
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in named],
        bodies=[body] + [c["body"] for c in named],
    )
    canonical = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"], decoding=_decoding_for(assignments[0])),
        refs=([{"role": "battery", "id": battery["id"]}] + plan_ref
              + [{"role": "run", "id": c["id"]} for c in named]),
        body=body,
    )
    return log, canonical


def test_refuses_a_floor_over_a_chosen_subset_of_the_preregistered_runs(tmp_path):
    """F7, the construction the report called the worst of the eight: the plan fixes R before the
    runs and the issuer picks the floor after them. Every number in this block is honest ABOUT
    THE RUNS IT NAMES -- recomputation agrees with it -- and it is still refused, because what is
    wrong is the sample and not the arithmetic."""
    log, canonical = subset_floor(tmp_path / "log", 1)
    assert log.floor_disagreement(canonical) == []  # the numerals are the named runs' own
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    reason = exc.value.reason
    assert reason.startswith("floor:")
    assert "R = 5" in reason and "rests on 2" in reason


@pytest.mark.parametrize("named", [1, 2, 3])
def test_no_proper_subset_of_the_plan_s_runs_composes_a_floor(tmp_path, named):
    log, canonical = subset_floor(tmp_path / f"log{named}", named)
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert f"rests on {named + 1}" in exc.value.reason


def test_the_whole_preregistered_set_is_what_appends(tmp_path):
    """The control for the four tests above: R runs, and it lands."""
    log, canonical = subset_floor(tmp_path / "log", 4)
    assert log.append(canonical) == log.size() - 1


def _plan_body_without_runs() -> dict:
    return {
        "kind": "noise-plan",
        "nuisance": [{"factor": "batch_size", "values": ["8", "4"]}],
        "environment": {"hardware": {"gpu": "none", "driver": "none", "count": 0}},
    }


def test_a_noise_plan_that_names_no_r_does_not_check_out(tmp_path):
    """A-NORUNS, the schema half. `runs` is where the plan commits to R, and it used to be
    enforced only WHEN PRESENT: a plan signed by hand with the key deleted fixed no R at all,
    `_check_floor_names_the_plans_runs` returned silently, and the hand-picked-subset attack of
    F7 was back for the price of one byte -- ten subsets of five published runs, every one of
    them appending with `floor_disagreement == []`, topk floors from 0.813925214 to 2.140233900.

    This is the test that fails if the requirement is dropped from schema/prereg.json."""
    plan = F.make_cert("prereg", body=_plan_body_without_runs())
    outcome = certmod.check(plan)
    assert outcome.ok is False
    assert any("runs" in reason and "required" in reason for reason in outcome.reasons), outcome.reasons

    log = fresh(tmp_path / "log")
    with pytest.raises(AppendRefused) as exc:
        log.append(plan)
    assert "runs" in exc.value.reason


def test_a_floor_resting_on_a_plan_that_names_no_r_is_refused(tmp_path):
    """A-NORUNS, the log half. The schema stops such a plan from being appended, so this puts one
    in the log the only way left -- the stored bytes are edited under an id the log already
    resolves -- and pins that the anchor REFUSES rather than skipping the subset check. Before
    the repair this appended at exit 0 and the completeness rule said nothing at all."""
    log = fresh(tmp_path / "log")
    battery = battery_cert(n=8)
    log.append(battery)
    plan = F.make_cert("prereg", body=dict(_plan_body_without_runs(), runs=5))
    plan_index = log.append(plan)

    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    runs = []
    for k, size in enumerate((4, 8), start=1):
        body = F.fingerprint_body(n=8, run_index=k, batch_size=size)
        cert = F.make_cert(
            "fingerprint",
            recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=size)),
            refs=[{"role": "battery", "id": battery["id"]}] + plan_ref,
            body=body,
        )
        log.append(cert)
        runs.append(cert)
    body = F.fingerprint_body(n=8, run_index=0, batch_size=8)
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in runs],
        bodies=[body] + [c["body"] for c in runs],
    )
    canonical = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=8)),
        refs=([{"role": "battery", "id": battery["id"]}] + plan_ref
              + [{"role": "run", "id": c["id"]} for c in runs]),
        body=body,
    )

    # the R disappears from the stored plan, under the id every ref still names
    stripped = copy.deepcopy(plan)
    stripped["body"].pop("runs")
    log.entry_path(plan_index).write_bytes(
        json.dumps(stripped, sort_keys=True).encode("utf-8")
    )
    assert log.cert(plan_index)["body"].get("runs") is None

    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert "fixes no R" in exc.value.reason


@pytest.mark.parametrize("key", ["floor", "distances", "runs", "pairs", "alpha_single"])
def test_refuses_a_floor_whose_signed_numbers_are_not_its_runs_numbers(tmp_path, key):
    """F6: `attach_floor` computes `per_channel` from `run_bodies` and writes `run_ids` beside it
    with nothing tying the two together, so an issuer signed a floor of 0.0/0.0/0.0 over three
    quiet bodies while naming two runs that really vary. The log holds those runs' bodies, so the
    floor is derivable, and every one of its five numbers is now re-derived and compared."""
    log, canonical = subset_floor(tmp_path / "log", 4)
    forged = copy.deepcopy(canonical)
    block = forged["body"]["noise_floor"]["per_channel"]["exact"]
    was = block[key]
    block[key] = [] if key == "distances" else (was + 1 if isinstance(was, int) else was + 1.0)
    forged = resign(forged)
    with pytest.raises(AppendRefused) as exc:
        log.append(forged)
    reason = exc.value.reason
    assert reason.startswith("floor: channel 'exact' is signed with %s" % key)
    assert "this log recomputes it, exactly" in reason
    # ...and the same computation is available to a reader who did not run the append.
    assert log.floor_disagreement(forged)
    assert log.floor_disagreement(canonical) == []


@pytest.mark.parametrize(
    "key", ["alpha_overall", "standardized_max", "alpha_overall_method", "standardization"]
)
def test_refuses_a_floor_that_omits_section_5_7s_overall_size(tmp_path, key):
    """A-OPTIONAL: the anchor re-derived `alpha_overall` and `standardized_max` only `if key in
    block`, so a cert with both deleted appended with `floor_disagreement == []` and section
    5.7's overall size simply absent -- a number nobody checks is a number nobody has to publish.
    Requiring the KEY in schema/fingerprint.json was the other option and it is weaker: a schema
    can say the number is there and cannot say it is the number the runs give.

    The two method strings are compared for the same reason. They are what a reader re-derives
    the fraction BY, and left unread they were two sentences an issuer could set to any procedure
    it liked while the fraction beside them stayed honest.

    This test fails if the derivation goes back to being conditional on presence."""
    log, canonical = subset_floor(tmp_path / "log", 4)
    assert log.floor_disagreement(canonical) == []
    forged = copy.deepcopy(canonical)
    assert key in forged["body"]["noise_floor"]
    forged["body"]["noise_floor"].pop(key)
    forged = resign(forged)
    with pytest.raises(AppendRefused) as exc:
        log.append(forged)
    assert "does not carry %s" % key in exc.value.reason
    assert log.floor_disagreement(forged)


@pytest.mark.parametrize("key", ["alpha_overall", "alpha_overall_method"])
def test_refuses_a_floor_whose_overall_size_is_not_the_one_its_runs_give(tmp_path, key):
    """The other half of A-OPTIONAL: forging the number, rather than deleting it. `alpha_overall`
    was already compared when present; the method string beside it was not."""
    log, canonical = subset_floor(tmp_path / "log", 4)
    forged = copy.deepcopy(canonical)
    forged["body"]["noise_floor"][key] = (
        1e-09 if key == "alpha_overall" else "section 5.7: measured over 500 runs"
    )
    forged = resign(forged)
    with pytest.raises(AppendRefused) as exc:
        log.append(forged)
    assert exc.value.reason.startswith("floor: %s is signed as" % key)


def test_refuses_a_floor_carrying_a_key_inside_a_channel_block_that_nothing_derives(tmp_path):
    """A-OPTIONAL's second half: the per-channel loop compared five named keys, so anything else
    inside a channel block was unread. `{"agrees": 999, "note": "measured over 500 runs"}` went
    into every block of an otherwise honest floor and appended at exit 0. The signed channel
    block is now the derived channel block -- every key the derivation makes, no key it does
    not."""
    log, canonical = subset_floor(tmp_path / "log", 4)
    forged = copy.deepcopy(canonical)
    for block in forged["body"]["noise_floor"]["per_channel"].values():
        block["agrees"] = 999
        block["note"] = "measured over 500 runs"
    forged = resign(forged)
    with pytest.raises(AppendRefused) as exc:
        log.append(forged)
    assert "beside the keys the derivation produces" in exc.value.reason
    assert "'agrees'" in exc.value.reason and "'note'" in exc.value.reason


def test_refuses_a_floor_missing_a_key_inside_a_channel_block(tmp_path):
    """Deletion inside a channel block, for the same reason: an absent key is not an agreeing
    key. `theirs.get(key)` used to read a missing `alpha_single` as None and compare it, which
    happened to refuse; it is said now rather than relied on."""
    log, canonical = subset_floor(tmp_path / "log", 4)
    forged = copy.deepcopy(canonical)
    forged["body"]["noise_floor"]["per_channel"]["exact"].pop("alpha_single")
    forged = resign(forged)
    with pytest.raises(AppendRefused) as exc:
        log.append(forged)
    assert "is signed without alpha_single" in exc.value.reason


def test_refuses_a_floor_that_drops_a_channel_its_runs_carry(tmp_path):
    """The channel SET is derived too: dropping a channel whose floor is inconvenient would
    otherwise leave `verify` calling it `inconclusive` instead of comparing it."""
    log, canonical = subset_floor(tmp_path / "log", 4)
    forged = copy.deepcopy(canonical)
    forged["body"]["noise_floor"]["per_channel"].pop("topk")
    forged = resign(forged)
    with pytest.raises(AppendRefused) as exc:
        log.append(forged)
    assert "carries channels" in exc.value.reason and "topk" in exc.value.reason


def test_refuses_a_floor_run_whose_nuisance_label_contradicts_its_own_recipe(tmp_path):
    """R-EXEC: the execution the check counts is derived from `body.items` chunked by
    `body.nuisance.batch_size`, and that label was free -- relabelling three real batch-1 runs as
    1/8/32 manufactured three "executions" out of one computation. Recomputing the floor does NOT
    catch this (the relabelled cert carries the same items, so the floor is honestly the floor of
    the bodies named); requiring the label to equal the cert's own `recipe.decoding` does."""
    log = fresh(tmp_path / "log")
    battery = battery_cert(n=8)
    log.append(battery)
    plan = plan_cert(("batch_size", ["1", "8", "4"]))
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    runs = []
    # Four runs that all really ran at batch 1, wearing the labels 8, 4, 8, 4.
    for k, label in enumerate((8, 4, 8, 4), start=1):
        body = F.fingerprint_body(n=8, run_index=k, batch_size=label)
        cert = F.make_cert(
            "fingerprint",
            recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=1)),
            refs=[{"role": "battery", "id": battery["id"]}] + plan_ref,
            body=body,
        )
        log.append(cert)  # a run cert on its own is not the cert making the floor claim
        runs.append(cert)
    body = F.fingerprint_body(n=8, run_index=0, batch_size=1)
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in runs],
        bodies=[body] + [c["body"] for c in runs],
    )
    canonical = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=1)),
        refs=([{"role": "battery", "id": battery["id"]}] + plan_ref
              + [{"role": "run", "id": c["id"]} for c in runs]),
        body=body,
    )
    # The floor itself is honest about the bodies it names -- that is exactly the point.
    assert log.floor_disagreement(canonical) == []
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    reason = exc.value.reason
    assert "records nuisance batch_size 8 while its own recipe.decoding says 1" in reason


def test_refuses_a_factor_that_turns_nothing_at_every_run_even_when_the_runs_differ(tmp_path):
    """R-UNION: `_factor_states` unioned the per-run counterfactual sets and tested
    `len(union) > 1`, which is true whenever the runs merely differ FROM EACH OTHER. Here every
    run is at batch 1, where permuting the item order permutes independent single-item passes, so
    every run's own set is a singleton -- and one run carries a shorter battery, which used to be
    enough to fill the union and pass."""
    log = fresh(tmp_path / "log")
    battery = battery_cert(n=8)
    log.append(battery)
    # Two NAMES for one order (section A.3), which is what makes every run's own counterfactual
    # set a singleton; the runs differ from each other because their batch sizes differ.
    orders = ["canonical", "reference"]
    plan = plan_cert(("item_order", orders))
    log.append(plan)
    plan_ref = [{"role": "noise_plan", "id": plan["id"]}]
    runs = []
    for k, size in enumerate((8, 1, 8, 1), start=1):
        body = F.fingerprint_body(n=8, run_index=k, batch_size=size)
        body["nuisance"]["item_order"] = orders[k % 2]
        cert = F.make_cert(
            "fingerprint",
            recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=size)),
            refs=[{"role": "battery", "id": battery["id"]}] + plan_ref,
            body=body,
        )
        log.append(cert)
        runs.append(cert)
    per_run = [FP.execution_states_for(c, "item_order", orders) for c in runs]
    assert all(len(s) == 1 for s in per_run)          # no run's own set moves
    assert len({next(iter(s)) for s in per_run}) > 1   # ...but the runs differ from each other
    body = F.fingerprint_body(n=8, run_index=0, batch_size=1)
    body["nuisance"]["item_order"] = "canonical"
    body["noise_floor"] = F.noise_floor_block(
        plan=plan["id"],
        plan_body=plan["body"],
        runs=[c["id"] for c in runs],
        bodies=[body] + [c["body"] for c in runs],
    )
    canonical = F.make_cert(
        "fingerprint",
        recipe=F.recipe(battery=battery["id"], decoding=F.decoding(batch_size=1)),
        refs=([{"role": "battery", "id": battery["id"]}] + plan_ref
              + [{"role": "run", "id": c["id"]} for c in runs]),
        body=body,
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(canonical)
    assert "'item_order'" in exc.value.reason and "could tell apart" in exc.value.reason


def test_the_lab_s_own_published_floor_recomputes_exactly_from_its_runs():
    """The anchor's own control, on real 64-item bodies rather than fixtures: the five run certs
    of papers/v8/first_verdict_2026_09_09 re-derive the floor their canonical signed, to the
    numeral, with no tolerance -- which is why the comparison can be `==`."""
    published = Path("papers/v8/first_verdict_2026_09_09/fp_bf16")
    if not published.is_dir():
        pytest.skip("the published first_verdict certs are not in this checkout")
    certs = [json.loads(p.read_bytes().decode("utf-8")) for p in sorted(published.glob("*.json"))]
    canonical = [c for c in certs if c["body"]["run_index"] == 0][0]
    runs = {c["id"]: c for c in certs if c["body"]["run_index"] != 0}
    order = canonical["body"]["noise_floor"]["runs"]
    bodies = [canonical["body"]] + [runs[rid]["body"] for rid in order]
    roles = {i["item_id"]: i.get("role", "item") for i in canonical["body"]["items"]}
    derived = {c: b for c, b in FLOOR.floors(bodies, roles=roles).items() if b is not None}
    signed = canonical["body"]["noise_floor"]["per_channel"]
    assert set(derived) == set(signed)
    for channel, block in signed.items():
        for key in ("floor", "distances", "runs", "pairs", "alpha_single"):
            assert derived[channel][key] == block[key], (channel, key)
    assert signed["exact"]["floor"] == 0.046875
    assert signed["seqlp"]["floor"] == 0.036070694
    assert signed["topk"]["floor"] == 2.1402339


def test_the_lab_s_own_first_real_floor_is_three_computations_not_five():
    """The published floor of papers/v8/first_verdict_2026_09_09: five runs at batch 1, 8, 32,
    1, 1 over 64 items. Three computations, not five, and `item_order` is realizable only at the
    runs that batched -- which is why the rule is per-factor-somewhere and not per-pair."""
    published = Path("papers/v8/first_verdict_2026_09_09/fp_bf16")
    if not published.is_dir():
        pytest.skip("the published first_verdict certs are not in this checkout")
    certs = [json.loads(p.read_bytes().decode("utf-8")) for p in sorted(published.glob("*.json"))]
    canonical = [c for c in certs if c["body"]["run_index"] == 0][0]
    runs = [c for c in certs if c["body"]["run_index"] != 0]
    assert len({FP.execution_state(c) for c in [canonical] + runs}) == 3
    assert len(FP.execution_states_for(canonical, "batch_size", ["1", "8", "32"])) == 3
    assert len(FP.execution_states_for(canonical, "item_order", ["canonical", "perm11"])) == 1
    batched = [c for c in runs if c["body"]["nuisance"]["batch_size"] > 1]
    assert batched
    assert len(FP.execution_states_for(batched[0], "item_order", ["canonical", "perm11"])) == 2


# ----------------------------------------------------------------- section 9 rule 1


def challenge_body_for(own: dict, **overrides) -> dict:
    """A section 9 body whose self-report is the `own` fingerprint's (C3).

    `body.subject` and `body.recipe_core` must be that cert's, or `Log.append` refuses
    (`_check_challenge_self_report`); `overrides` is how a test breaks exactly one of them.
    """
    body = {
        "per_channel": {"exact": {"distance": 0.0625, "target_floor": 0.046875}},
        "coverage": "within",
        "environment": {"hardware": {"gpu": "none"}, "runtime": {"framework": "mock"}},
        "subject": certmod.identity_fields(own.get("subject", {})),
        "recipe_core": certmod.recipe_core(own.get("recipe", {})),
    }
    if certmod.is_synthetic(own):
        body[certmod.SYNTHETIC] = True
    body.update(overrides)
    return body


def challenge_cert(target_id: str, own, *, refs=None, **body_overrides) -> dict:
    """`own` is the challenger's fingerprint CERT: the body reports what that cert says it ran."""
    own_id = own["id"] if isinstance(own, dict) else own
    body = challenge_body_for(own, **body_overrides) if isinstance(own, dict) else {}
    return F.make_cert(
        "challenge",
        subject={},
        recipe={},
        body=body,
        refs=(
            list(refs)
            if refs is not None
            else [{"role": "target", "id": target_id}, {"role": "own", "id": own_id}]
        ),
    )


def test_refuses_a_challenge_whose_own_fingerprint_is_another_subject(tmp_path):
    """C1 of papers/v8/challenge_and_attack_2026_09_09: an fp16 cert filed as the `own` half of a
    challenge against a bf16 target, taken by the CLI, cert.check, log.append and the JS
    verifier. `cert.comparable` already said `['cross-subject:precision']`."""
    log = fresh(tmp_path / "log")
    battery = battery_cert()
    log.append(battery)
    target = fp_cert(battery["id"])
    log.append(target)
    own = F.make_cert(
        "fingerprint",
        subject=F.weights_subject(precision="fp16"),
        recipe=F.recipe(battery=battery["id"]),
        refs=[{"role": "battery", "id": battery["id"]}],
        body=F.fingerprint_body(),
    )
    log.append(own)
    with pytest.raises(AppendRefused) as exc:
        log.append(challenge_cert(target["id"], own))
    reason = exc.value.reason
    assert reason.startswith("challenge:")
    assert "cross-subject:precision" in reason
    assert "No match, no challenge" in reason


def test_a_challenge_against_its_own_subject_is_appended(tmp_path):
    """The mechanism still works: a reproduction of the same subject files a challenge and the
    log takes it. Section 9 refuses a mismatch, not a disagreement."""
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
    index = log.append(challenge_cert(target["id"], own))
    assert log.cert(index)["type"] == "challenge"


def test_refuses_a_challenge_whose_own_fingerprint_ran_another_battery(tmp_path):
    """Section 9 rule 1 has two halves and this is the recipe half: a reproduction of a different
    item set is not a reproduction of this cert."""
    log = fresh(tmp_path / "log")
    battery = battery_cert(n=3)
    other = battery_cert(n=5)
    log.append(battery)
    log.append(other)
    target = fp_cert(battery["id"])
    log.append(target)
    own = fp_cert(other["id"])
    log.append(own)
    with pytest.raises(AppendRefused) as exc:
        log.append(challenge_cert(target["id"], own))
    assert "recipe.battery" in exc.value.reason


def test_refuses_a_redacted_battery_under_a_public_cert(tmp_path):
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    secret = battery_cert(redacted=True)
    log.append(secret)
    assert log.meta(1)["public"] is False

    public_fp = fp_cert(secret["id"])
    with pytest.raises(AppendRefused) as exc:
        log.append(public_fp)
    assert exc.value.reason.startswith("redaction:")

    declared = fp_cert(secret["id"], body_extra={"redacted_battery": True})
    index = log.append(declared)
    assert log.meta(index)["public"] is False  # section 2.6: public is transitive


def test_refuses_to_append_over_a_gap(tmp_path):
    log = fresh(tmp_path / "log")
    built = chain(log)
    log.entry_path(1).unlink()
    assert log.size() == 1 and log.indices() == [0, 2, 3]
    with pytest.raises(AppendRefused) as exc:
        log.append(F.make_cert("prereg", body={"kind": "noise-plan", "runs": 5}))
    assert exc.value.reason.startswith("entries:")
    assert log.cert(0)["id"] == built["pool"]["id"]  # the surviving entries still read


def test_a_root_battery_fills_an_empty_log_with_no_ref_to_resolve(tmp_path):
    """Section 4.5, the property this whole ladder stands on.

    A pool-v1 battery carries no ``recipe.battery``, so it carries no ref, so there is nothing
    for section 8.3 to resolve and an empty log takes it at index 0. Before the schema fix a
    battery cert was required to name another battery and no log could be started at all; the
    append path needed a named exemption to permit the one unresolvable ref. Neither the recipe
    entry nor the ref is here to be exempted.
    """
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    assert pool["recipe"] == {}
    assert pool["refs"] == []
    assert certmod.check(pool).ok
    assert log.size() == 0
    assert log.append(pool) == 0
    assert log.cert(0)["id"] == pool["id"]
    assert verify_entry(log, 0) == (True, "ok")

    # fixed-v1 is a root on the same terms.
    fixed = battery_cert(kind="fixed-v1", n=4)
    assert fixed["recipe"] == {} and fixed["refs"] == []
    assert log.append(fixed) == 1


def test_a_canary_battery_needs_its_selected_against_in_the_log(tmp_path):
    """Section 4.5: canary-v1 is the non-root kind, and the log makes it prove it.

    The pool and the reference fingerprint are on record; a canary that names a fingerprint that
    is not is refused, and the same body with the logged reference appends.
    """
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    reference = fp_cert(pool["id"])
    log.append(reference)

    stranger = canary_cert(pool["id"], F.fake_id("a fingerprint in no leaf"))
    assert certmod.check(stranger).ok  # well formed; it is the log that says no
    with pytest.raises(AppendRefused) as exc:
        log.append(stranger)
    assert exc.value.reason.startswith("refs:")
    assert "selected_against" in exc.value.reason
    assert "does not resolve" in exc.value.reason
    assert log.size() == 2

    assert log.append(canary_cert(pool["id"], reference["id"])) == 2


def test_the_genesis_exemption_covers_nothing_a_root_battery_needs(tmp_path):
    """The residue of the fixed defect, pinned so its size is visible.

    ``GENESIS_UNRESOLVED`` permits one unresolvable ref: a pool-v1 battery pointing at a battery.
    Nothing built to section 4.5 asks for it -- a root battery has no ref -- and it has never
    covered any other kind: a fixed-v1 battery that names an absent battery is still refused.
    """
    from styxx.v8.log import GENESIS_UNRESOLVED

    assert GENESIS_UNRESOLVED == frozenset({("battery", "pool-v1", "battery")})
    assert pool_cert()["refs"] == []  # so this log never reaches the exemption

    log = fresh(tmp_path / "log")
    absent = F.fake_id("no such pool")
    orphan_fixed = F.make_cert(
        "battery",
        recipe=F.recipe(battery=absent),
        refs=[{"role": "battery", "id": absent}],
        body=F.battery_body("fixed-v1"),
    )
    with pytest.raises(AppendRefused) as exc:
        log.append(orphan_fixed)
    assert exc.value.reason.startswith("refs:")
    assert "does not resolve" in exc.value.reason
    assert log.size() == 0


def test_a_canary_battery_whose_selected_against_is_not_a_fingerprint_is_refused(tmp_path):
    """The role has to be honoured, not merely filled: selected_against names a fingerprint."""
    log = fresh(tmp_path / "log")
    pool = pool_cert()
    log.append(pool)
    with pytest.raises(AppendRefused) as exc:
        log.append(canary_cert(pool["id"], pool["id"]))  # points at the pool battery
    assert "resolves to a battery cert, not a fingerprint cert" in exc.value.reason


# ----------------------------------------------------------------- tree heads

def test_sth_signs_the_current_head_and_verifies(tmp_path):
    log = five_entry_log(tmp_path / "log")
    sth = log.sth(LOG_SEED, TS)
    assert set(sth) == {"log_id", "tree_size", "root_hash", "timestamp", "sig"}
    assert sth["tree_size"] == 5
    assert sth["root_hash"] == "sha256:" + log.root().hex()
    assert verify_sth(sth, LOG_PUB) == (True, "ok")
    assert (log.sth_dir / "000000000005.json").exists()
    assert log.latest_sth() == sth

    # the signature is over the tagged digest of the four signed fields
    core = {k: sth[k] for k in ("log_id", "tree_size", "root_hash", "timestamp")}
    digest = hashlib.sha256(canonical_bytes(core)).digest()
    assert keys.verify(LOG_PUB, keys.tagged("styxx.v8/sth/1", digest), keys.decode_signature(sth["sig"]))
    # and not over the cert tag
    assert not keys.verify(LOG_PUB, keys.tagged("styxx.v8/cert/1", digest), keys.decode_signature(sth["sig"]))


def test_sth_refuses_a_seed_that_is_not_the_log_key(tmp_path):
    log = five_entry_log(tmp_path / "log")
    with pytest.raises(AppendRefused) as exc:
        log.sth(ISSUER_SEED, TS)
    assert "not keys/log.pub" in exc.value.reason
    assert list(log.sth_dir.glob("*.json")) == []


def test_sth_refuses_a_malformed_timestamp(tmp_path):
    log = five_entry_log(tmp_path / "log")
    with pytest.raises(AppendRefused) as exc:
        log.sth(LOG_SEED, "2026-09-08 18:00:00")
    assert "RFC 3339" in exc.value.reason


def test_sth_is_idempotent_but_refuses_a_reused_tree_size(tmp_path):
    log = five_entry_log(tmp_path / "log")
    first = log.sth(LOG_SEED, TS)
    assert log.sth(LOG_SEED, TS) == first  # same head, same bytes

    with pytest.raises(AppendRefused) as exc:
        log.sth(LOG_SEED, TS2)
    assert "different timestamp" in exc.value.reason

    # a head for this size with a different root is already on file: refused
    forged = forge_sth(log, tree_size=5, root=b"\x11" * 32, timestamp=TS)
    (log.sth_dir / "000000000005.json").write_bytes(
        (json.dumps(forged, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
    with pytest.raises(AppendRefused) as exc:
        log.sth(LOG_SEED, TS)
    assert "different root_hash" in exc.value.reason


def test_verify_sth_rejects_the_wrong_key_and_a_forged_root(tmp_path):
    log = five_entry_log(tmp_path / "log")
    sth = log.sth(LOG_SEED, TS)

    ok, why = verify_sth(sth, OTHER_PUB)
    assert not ok and "log_id" in why

    forged_root = dict(sth)
    forged_root["root_hash"] = "sha256:" + ("00" * 32)
    ok, why = verify_sth(forged_root, LOG_PUB)
    assert not ok and "sig does not verify" in why

    wrong_signer = forge_sth(log, tree_size=5, root=log.root(), timestamp=TS, seed=ISSUER_SEED)
    ok, why = verify_sth(wrong_signer, LOG_PUB)
    assert not ok and "sig does not verify" in why

    extra = dict(sth)
    extra["note"] = "unsigned field"
    ok, why = verify_sth(extra, LOG_PUB)
    assert not ok and "unexpected field" in why

    for hostile in (None, {}, [], "sth", {"log_id": 1, "tree_size": "x", "root_hash": 2, "timestamp": 3, "sig": 4}):
        ok, why = verify_sth(hostile, LOG_PUB)
        assert not ok and isinstance(why, str)


def forge_sth(log: Log, *, tree_size: int, root: bytes, timestamp: str, seed: bytes = LOG_SEED) -> dict:
    """An STH over any root the caller names, signed with ``seed``."""
    core = {
        "log_id": log.log_id(),
        "tree_size": tree_size,
        "root_hash": "sha256:" + root.hex(),
        "timestamp": timestamp,
    }
    digest = hashlib.sha256(canonical_bytes(core)).digest()
    out = dict(core)
    out["sig"] = keys.encode_signature(keys.sign(seed, keys.tagged("styxx.v8/sth/1", digest)))
    return out


# ----------------------------------------------------------------- proofs

def growing_log(root) -> tuple[Log, dict[int, dict]]:
    """Five entries, an STH after each append: heads at tree sizes 1..5."""
    log = fresh(root)
    heads: dict[int, dict] = {}
    pool = pool_cert()
    log.append(pool)
    heads[1] = log.sth(LOG_SEED, TS)
    reference = fp_cert(pool["id"])
    log.append(reference)
    heads[2] = log.sth(LOG_SEED, TS)
    canary = canary_cert(pool["id"], reference["id"])
    log.append(canary)
    heads[3] = log.sth(LOG_SEED, TS)
    previous = None
    for k in range(2):
        extra = [] if previous is None else [{"role": "previous", "id": previous}]
        run = fp_cert(canary["id"], run_index=k, extra_refs=extra)
        log.append(run)
        previous = run["id"]
        heads[4 + k] = log.sth(LOG_SEED, TS)
    return log, heads


def test_inclusion_proofs_verify_at_every_historical_size(tmp_path):
    log, heads = growing_log(tmp_path / "log")
    for tree_size in range(1, 6):
        for index in range(tree_size):
            proof = log.inclusion(index, tree_size)
            assert proof["leaf_index"] == index
            assert proof["tree_size"] == tree_size
            assert proof["leaf_hash"] == merkle.leaf_hash(log.entry_bytes(index)).hex()
            assert proof["root_hash"] == heads[tree_size]["root_hash"]
            assert verify_inclusion(proof, heads[tree_size], LOG_PUB) == (True, "ok")


def test_an_inclusion_proof_does_not_verify_against_another_head(tmp_path):
    log, heads = growing_log(tmp_path / "log")
    proof = log.inclusion(1, 3)
    ok, why = verify_inclusion(proof, heads[4], LOG_PUB)
    assert not ok and "tree_size" in why

    flipped = dict(proof)
    flipped["path"] = [("f" + h[1:]) for h in proof["path"]]
    ok, why = verify_inclusion(flipped, heads[3], LOG_PUB)
    assert not ok and "does not reach the signed root" in why

    moved = dict(proof)
    moved["leaf_index"] = 2
    ok, why = verify_inclusion(moved, heads[3], LOG_PUB)
    assert not ok

    for hostile in (None, {}, {"leaf_index": 0}, {"leaf_index": 0, "leaf_hash": "zz", "tree_size": 3, "root_hash": 1, "path": None}):
        ok, why = verify_inclusion(hostile, heads[3], LOG_PUB)
        assert not ok and isinstance(why, str)


def test_consistency_across_three_heads(tmp_path):
    log, heads = growing_log(tmp_path / "log")
    for first, second in ((1, 3), (3, 5), (1, 5)):
        proof = log.consistency(first, second)
        assert proof["first_root"] == heads[first]["root_hash"]
        assert proof["second_root"] == heads[second]["root_hash"]
        assert verify_consistency(heads[first], heads[second], proof, LOG_PUB) == (True, "ok")

    good = log.consistency(1, 3)
    ok, why = verify_consistency(heads[1], heads[4], good, LOG_PUB)
    assert not ok and "second" in why

    ok, why = verify_consistency(heads[5], heads[1], log.consistency(1, 5), LOG_PUB)
    assert not ok

    broken = dict(log.consistency(3, 5))
    broken["proof"] = [("0" * 64)] + broken["proof"][1:]
    ok, why = verify_consistency(heads[3], heads[5], broken, LOG_PUB)
    assert not ok and "does not extend" in why


def test_verify_entry_accepts_every_entry_of_a_clean_log(tmp_path):
    log, _heads = growing_log(tmp_path / "log")
    for index in range(log.size()):
        assert verify_entry(log, index) == (True, "ok")


# ----------------------------------------------------------------- mirror

def clean_source(tmp_path) -> tuple[Log, dict]:
    log, heads = growing_log(tmp_path / "src")
    return log, heads[5]


def test_mirror_of_a_clean_log_verifies(tmp_path):
    log, pinned = clean_source(tmp_path)
    report = mirror(log.path, tmp_path / "dst", LOG_PUB, pinned)
    assert report["entries"] == 5
    assert report["sths"] == 5
    assert report["misbehaviour"] == []
    assert report["tamper"] == []
    assert report["unpublished"] == []
    assert report["verified"] is True
    # the copy is byte-exact and usable on its own
    copy = Log(tmp_path / "dst")
    assert copy.entry_bytes(0) == log.entry_bytes(0)
    assert copy.root() == log.root()


def test_mirror_reports_entries_no_head_covers(tmp_path):
    log, pinned = clean_source(tmp_path)
    built_on = log.cert(4)["id"]  # the newest fingerprint
    battery_id = log.cert(2)["id"]  # the canary battery it was run on
    log.append(fp_cert(battery_id, run_index=9, extra_refs=[{"role": "previous", "id": built_on}]))
    report = mirror(log.path, tmp_path / "dst", LOG_PUB, pinned)
    assert report["entries"] == 6
    assert report["unpublished"] == [5]
    assert report["tamper"] == [] and report["misbehaviour"] == []
    assert report["verified"] is True


def tampered_copy(tmp_path, name: str) -> Path:
    """A byte copy of the clean source that a test then edits."""
    src = tmp_path / "src"
    dst = tmp_path / name
    shutil.copytree(src, dst)
    return dst


def test_mirror_reports_an_edited_entry_as_tamper(tmp_path):
    log, pinned = clean_source(tmp_path)
    bad = tampered_copy(tmp_path, "edited")
    path = bad / "entries" / "000000" / "00000002.json"
    raw = path.read_bytes()
    edited = raw.replace(b'"name":"fathom lab"', b'"name":"fathom lxb"')
    assert edited != raw and len(edited) == len(raw)
    path.write_bytes(edited)

    report = mirror(bad, tmp_path / "dst-edited", LOG_PUB, pinned)
    assert report["verified"] is False
    assert any("entry 2" in line for line in report["tamper"])
    assert any("id does not recompute" in line for line in report["tamper"])


def test_mirror_reports_a_deleted_entry_as_tamper(tmp_path):
    log, pinned = clean_source(tmp_path)
    bad = tampered_copy(tmp_path, "deleted")
    (bad / "entries" / "000000" / "00000002.json").unlink()

    report = mirror(bad, tmp_path / "dst-deleted", LOG_PUB, pinned)
    assert report["verified"] is False
    assert report["entries"] == 2
    assert any("beyond a gap" in line for line in report["tamper"])
    assert any("the mirror holds 2" in line for line in report["tamper"])


def test_mirror_reports_a_reordered_pair_as_tamper(tmp_path):
    log, pinned = clean_source(tmp_path)
    bad = tampered_copy(tmp_path, "reordered")
    a = bad / "entries" / "000000" / "00000002.json"
    b = bad / "entries" / "000000" / "00000003.json"
    a_raw, b_raw = a.read_bytes(), b.read_bytes()
    a.write_bytes(b_raw)
    b.write_bytes(a_raw)

    report = mirror(bad, tmp_path / "dst-reordered", LOG_PUB, pinned)
    assert report["verified"] is False
    assert any("metadata names" in line for line in report["tamper"])
    assert any("entry 2" in line for line in report["tamper"])
    assert any("entry 3" in line for line in report["tamper"])


def test_mirror_reports_two_heads_of_one_size_as_misbehaviour(tmp_path):
    log, pinned = clean_source(tmp_path)
    bad = tampered_copy(tmp_path, "two-heads")
    rogue = forge_sth(log, tree_size=3, root=b"\x22" * 32, timestamp=TS)
    (bad / "sth" / "000000000003.rogue.json").write_bytes(
        (json.dumps(rogue, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )

    report = mirror(bad, tmp_path / "dst-two-heads", LOG_PUB, pinned)
    assert report["verified"] is False
    assert report["tamper"] == []  # the entries are intact; the heads are not
    assert any("different root_hash values" in line for line in report["misbehaviour"])
    assert any("do not reproduce the signed root" in line for line in report["misbehaviour"])


def test_mirror_reports_a_pinned_head_from_another_view(tmp_path):
    log, _pinned = clean_source(tmp_path)
    split = forge_sth(log, tree_size=4, root=b"\x33" * 32, timestamp=TS)
    report = mirror(log.path, tmp_path / "dst-split", LOG_PUB, split)
    assert report["verified"] is False
    assert any("<pinned>" in line for line in report["misbehaviour"])


def test_mirror_reports_a_head_signed_by_another_key(tmp_path):
    log, pinned = clean_source(tmp_path)
    bad = tampered_copy(tmp_path, "wrong-key")
    stranger = forge_sth(log, tree_size=5, root=log.root(), timestamp=TS2, seed=ISSUER_SEED)
    (bad / "sth" / "000000000005.other.json").write_bytes(
        (json.dumps(stranger, sort_keys=True, indent=2) + "\n").encode("utf-8")
    )
    report = mirror(bad, tmp_path / "dst-wrong-key", LOG_PUB, pinned)
    assert report["verified"] is False
    assert any("sig does not verify" in line for line in report["misbehaviour"])
    # a head under a key the mirror does not know says nothing about the entries
    assert report["tamper"] == []


def test_a_crlf_injected_entry_reads_as_tamper(tmp_path):
    log, pinned = clean_source(tmp_path)
    bad = tampered_copy(tmp_path, "crlf")
    path = bad / "entries" / "000000" / "00000001.json"
    raw = path.read_bytes()
    injected = raw.replace(b'{"body":', b'{\r\n"body":', 1)
    assert injected != raw
    assert json.loads(injected.decode("utf-8")) == json.loads(raw.decode("utf-8"))
    path.write_bytes(injected)

    ok, why = verify_entry(Log(bad), 1)
    assert not ok and "CR" in why

    report = mirror(bad, tmp_path / "dst-crlf", LOG_PUB, pinned)
    assert report["verified"] is False
    assert any("CR" in line for line in report["tamper"])


def test_mirror_never_raises_on_a_hostile_directory(tmp_path):
    missing = tmp_path / "not-a-log"
    report = mirror(missing, tmp_path / "dst-missing", LOG_PUB, None)
    assert report["entries"] == 0
    assert report["verified"] is True  # an empty log makes no claim to break

    junk = tmp_path / "junk"
    (junk / "entries" / "000000").mkdir(parents=True)
    (junk / "entries" / "000000" / "00000000.json").write_bytes(b"not json at all")
    (junk / "sth").mkdir(parents=True)
    (junk / "sth" / "000000000001.json").write_bytes(b"{")
    report = mirror(junk, tmp_path / "dst-junk", LOG_PUB, None)
    assert report["verified"] is False
    assert report["tamper"] and report["misbehaviour"]

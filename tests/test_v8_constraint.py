"""The constraint census: how much prior logged material could contradict a claim.

``styxx/v8/constraint.py`` exists because THE_BOUNDARY's class two emptied. Four fields were
argued unreachable and all four were reachable, every time by the same error — the argument
examined one certificate instead of the log the certificate sits in. What survives is not a list
of unreachable fields but one unreachable act, the first claim about anything, and the quantity
that says whether a given verdict is in that position was computed nowhere.

The tests below fix both halves.

* **The number is real when history exists.** A sibling run on the same subject and battery that
  this cert does not name is prior material a check could run against, and the census counts it.
* **The number is zero when the history is the cert's own input.** This is the test that matters,
  and it is pinned against the lab's published verdict rather than a fixture:
  ``test_the_published_floor_is_a_first_claim`` reads
  ``papers/v8/first_verdict_2026_09_09/log`` and asserts that entry 6 — the floor, the headline
  verdict of that artifact — has six prior entries, that **all six are its own material**, and
  that its constraining census is 0 on all five predicates. The lab's own flagship certificate is
  the case the predicates cannot reach.
* **Own material is excluded on purpose.** ``Log.previous_comparable`` does not exclude it, and
  THE_BOUNDARY records what that costs: on this same log the lookup for the canonical fingerprint
  returns that canonical's own floor run. ``test_a_floors_own_runs_are_own_material_not_constraint``
  pins the opposite behaviour here.
* **A plan on file is not a schedule.** The published plan admits many schedules and names none,
  so the member-2 predicate reports ``plans_carrying_a_schedule = 0`` and says why in
  ``blocked_by`` rather than reporting the plan as constraint.
* **Independent constraint is 0 on a single-issuer log**, by construction, and the census prints
  that separately instead of letting a large self-signed count read as corroboration.

Nothing here skips. Nothing here imports a module another agent is editing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from styxx.v8 import constraint as C

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED_LOG = ROOT / "papers" / "v8" / "first_verdict_2026_09_09" / "log"

ISSUER = "ed25519:AAAA"
OTHER_ISSUER = "ed25519:BBBB"
BATTERY_ID = "sha256:" + "b" * 64
PLAN_ID = "sha256:" + "c" * 64

SUBJECT = {
    "hf_repo": "acme/model",
    "revision": "r1",
    "weights_sha256": "w" * 64,
    "config_sha256": "cf" * 32,
    "tokenizer_sha256": "tk" * 32,
    "generation_config_sha256": "gc" * 32,
    "precision": "bf16",
    "kind": "weights",
}


def run_cert(
    cid: str,
    *,
    batch: int = 1,
    subject: dict | None = None,
    issuer: str = ISSUER,
    refs: list[dict] | None = None,
    battery: str | None = BATTERY_ID,
    channels: dict | None = None,
) -> dict:
    recipe: dict = {"decoding": {"batch_size": batch}}
    if battery is not None:
        recipe["battery"] = battery
    return {
        "styxx": "8.0",
        "type": "fingerprint",
        "id": cid,
        "issuer": {"key": issuer, "name": "t"},
        "subject": dict(subject or SUBJECT),
        "recipe": recipe,
        "refs": list(refs if refs is not None else []),
        "body": {
            "nuisance": {"batch_size": batch},
            "channels": channels if channels is not None else {"exact": {"hash": "00"}},
        },
    }


def plan_cert(cid: str = PLAN_ID, *, schedule: bool = False, issuer: str = ISSUER) -> dict:
    body: dict = {
        "kind": "noise-plan",
        "runs": 5,
        "covers": ["batch_size"],
        "nuisance": [{"factor": "batch_size", "values": ["1", "8", "32"]}],
    }
    if schedule:
        body["schedule"] = [{"run": 0, "batch_size": "1"}]
    return {
        "styxx": "8.0",
        "type": "prereg",
        "id": cid,
        "issuer": {"key": issuer, "name": "t"},
        "subject": dict(SUBJECT),
        "recipe": {"battery": BATTERY_ID},
        "refs": [{"id": BATTERY_ID, "role": "battery"}],
        "body": body,
    }


def floor_cert(cid: str, run_ids: list[str], *, issuer: str = ISSUER) -> dict:
    cert = run_cert(cid, batch=1, issuer=issuer)
    cert["refs"] = (
        [{"id": BATTERY_ID, "role": "battery"}, {"id": PLAN_ID, "role": "noise_plan"}]
        + [{"id": r, "role": "run"} for r in run_ids]
    )
    cert["body"]["noise_floor"] = {
        "plan": PLAN_ID,
        "runs": list(run_ids),
        "per_channel": {"exact": {"floor": 0.0, "runs": len(run_ids) + 1}},
    }
    return cert


# --------------------------------------------------------------------------- the first claim


def test_a_claim_with_no_prior_entries_is_reported_unconstrained_not_passing():
    """The boundary case: nothing to contradict, so every check on it is vacuous."""
    result = C.census(run_cert("sha256:" + "1" * 64), [])
    assert result["prior_entries"] == 0
    assert result["constraining_entries"] == 0
    assert result["independent_entries"] == 0
    assert result["first_claim"] is True
    assert result["verdict"] == "unconstrained"
    for name, _ in C.PREDICATES:
        assert result["predicates"][name]["usable"] == 0
        assert result["predicates"][name]["verdict"] != "constrained"
    assert "first claim" in result["disclosure"]
    assert "vacuous" in result["disclosure"]


def test_the_census_never_reports_a_verdict_about_agreement():
    """It counts what a check would have had to work with; it does not run the check."""
    prior = [run_cert("sha256:" + "2" * 64, batch=8)]
    result = C.census(run_cert("sha256:" + "1" * 64, batch=1), prior)
    text = json.dumps(result)
    for word in ("agrees", "disagrees", "contradicted", "valid", "invalid"):
        assert word not in text


# --------------------------------------------------------------------------- floor_agreement


def test_a_sibling_run_on_the_same_subject_and_battery_is_constraint():
    prior = [run_cert("sha256:" + "2" * 64, batch=8)]
    result = C.census(run_cert("sha256:" + "1" * 64, batch=1), prior)
    block = result["predicates"]["floor_agreement"]
    assert block["matched"] == 1
    assert block["own"] == 0
    assert block["available"] == 1
    assert block["usable"] == 1
    assert block["verdict"] == "constrained"
    assert result["first_claim"] is False
    assert result["constraining_entries"] == 1


def test_a_different_weights_hash_is_not_in_scope():
    other = dict(SUBJECT, weights_sha256="x" * 64, revision="r2")
    prior = [run_cert("sha256:" + "2" * 64, subject=other)]
    block = C.census(run_cert("sha256:" + "1" * 64), prior)["predicates"]["floor_agreement"]
    assert block["matched"] == 0
    assert block["verdict"] == "unconstrained"


def test_a_different_battery_is_not_in_scope():
    prior = [run_cert("sha256:" + "2" * 64, battery="sha256:" + "d" * 64)]
    block = C.census(run_cert("sha256:" + "1" * 64), prior)["predicates"]["floor_agreement"]
    assert block["matched"] == 0


# --------------------------------------------------------------------------- own material


def test_a_floors_own_runs_are_own_material_not_constraint():
    """The exclusion `Log.previous_comparable` does not make.

    A floor that consumes four runs is not corroborated by those four runs. They are its input.
    """
    run_ids = ["sha256:" + str(n) * 64 for n in (2, 3, 4, 5)]
    prior = [plan_cert()] + [run_cert(r, batch=1 + i) for i, r in enumerate(run_ids)]
    cert = floor_cert("sha256:" + "1" * 64, run_ids)
    result = C.census(cert, prior)

    block = result["predicates"]["floor_agreement"]
    # The plan shares the subject and the battery, so it is in scope too: five matched, five own.
    assert block["matched"] == 5
    assert block["own"] == 5
    assert block["available"] == 0
    assert block["usable"] == 0
    assert block["verdict"] == "unconstrained"
    assert result["constraining_entries"] == 0
    assert result["first_claim"] is True
    for rid in run_ids:
        assert rid in result["own_material"]
    assert PLAN_ID in result["own_material"]


def test_own_material_names_every_channel_a_cert_rests_on():
    run_ids = ["sha256:" + str(n) * 64 for n in (2, 3)]
    cert = floor_cert("sha256:" + "1" * 64, run_ids)
    material = C.own_material(cert)
    assert cert["id"] in material
    assert BATTERY_ID in material
    assert PLAN_ID in material
    assert set(run_ids) <= set(material)


def test_a_run_the_floor_does_not_name_stays_constraint():
    """Only the material a cert actually names is excluded."""
    named = ["sha256:" + str(n) * 64 for n in (2, 3)]
    unnamed = run_cert("sha256:" + "9" * 64, batch=32)
    prior = [plan_cert()] + [run_cert(r) for r in named] + [unnamed]
    result = C.census(floor_cert("sha256:" + "1" * 64, named), prior)
    block = result["predicates"]["floor_agreement"]
    assert block["own"] == 3  # the plan and the two runs the floor names
    assert block["available"] == 1
    assert block["usable"] == 1
    assert [e["id"] for e in block["entries"]] == [unnamed["id"]]
    assert result["constraining_entries"] == 1


# --------------------------------------------------------------------------- snapshot_agreement


def test_the_snapshot_predicate_scopes_on_repo_and_revision():
    same = run_cert("sha256:" + "2" * 64)
    other_rev = run_cert("sha256:" + "3" * 64, subject=dict(SUBJECT, revision="r2"))
    result = C.census(run_cert("sha256:" + "1" * 64), [same, other_rev])
    block = result["predicates"]["snapshot_agreement"]
    assert block["scope"] == {"hf_repo": "acme/model", "revision": "r1"}
    assert block["matched"] == 1
    assert block["usable"] == 1


def test_a_subject_without_repo_and_revision_makes_the_snapshot_predicate_inapplicable():
    bare = {"weights_sha256": "w" * 64}
    cert = run_cert("sha256:" + "1" * 64, subject=bare)
    block = C.census(cert, [run_cert("sha256:" + "2" * 64)])["predicates"]["snapshot_agreement"]
    assert block["applicable"] is False
    assert block["verdict"] == "not-applicable"
    assert any("hf_repo" in reason for reason in block["blocked_by"])


# --------------------------------------------------------------------------- schedule


def test_a_plan_that_names_no_schedule_is_reported_as_such():
    prior = [plan_cert(schedule=False)]
    cert = run_cert(
        "sha256:" + "1" * 64, refs=[{"id": PLAN_ID, "role": "noise_plan"}]
    )
    block = C.census(cert, prior)["predicates"]["schedule"]
    assert block["applicable"] is True
    assert block["plans_on_file"] == 1
    assert block["plans_carrying_a_schedule"] == 0
    assert block["usable"] == 0
    assert block["verdict"] == "unconstrained"
    assert any("names no schedule" in reason for reason in block["blocked_by"])


def test_a_plan_carrying_a_schedule_is_counted():
    prior = [plan_cert(schedule=True)]
    cert = run_cert("sha256:" + "1" * 64, refs=[{"id": PLAN_ID, "role": "noise_plan"}])
    block = C.census(cert, prior)["predicates"]["schedule"]
    assert block["plans_carrying_a_schedule"] == 1
    assert not any("names no schedule" in reason for reason in block["blocked_by"])


def test_the_plan_is_still_own_material_even_when_it_carries_a_schedule():
    """A commitment the issuer wrote is a commitment, but it is not a second party's bytes."""
    prior = [plan_cert(schedule=True)]
    cert = run_cert("sha256:" + "1" * 64, refs=[{"id": PLAN_ID, "role": "noise_plan"}])
    result = C.census(cert, prior)
    block = result["predicates"]["schedule"]
    assert block["own"] == 1
    assert block["available"] == 0
    assert result["independent_entries"] == 0


def test_a_floors_plan_is_found_through_the_noise_floor_block_too():
    cert = floor_cert("sha256:" + "1" * 64, [])
    cert["refs"] = [r for r in cert["refs"] if r["role"] != "noise_plan"]
    block = C.census(cert, [plan_cert()])["predicates"]["schedule"]
    assert block["scope"] == {"plan": PLAN_ID}
    assert block["plans_on_file"] == 1


# --------------------------------------------------------------------------- determinism


def test_a_prior_run_at_the_same_batch_level_is_constraint():
    prior = [run_cert("sha256:" + "2" * 64, batch=8)]
    block = C.census(run_cert("sha256:" + "1" * 64, batch=8), prior)["predicates"]["determinism"]
    assert block["matched"] == 1
    assert block["usable"] == 1
    assert block["verdict"] == "constrained"


def test_a_prior_run_at_a_different_batch_level_is_not_the_determinism_predicate():
    prior = [run_cert("sha256:" + "2" * 64, batch=32)]
    result = C.census(run_cert("sha256:" + "1" * 64, batch=8), prior)
    assert result["predicates"]["determinism"]["matched"] == 0
    # ...but it is still in scope for the floor predicate, which compares across batch levels.
    assert result["predicates"]["floor_agreement"]["usable"] == 1


def test_the_batch_level_is_read_from_the_nuisance_assignment_first():
    cert = run_cert("sha256:" + "1" * 64, batch=8)
    cert["body"]["nuisance"]["batch_size"] = "32"
    prior = [run_cert("sha256:" + "2" * 64, batch=32)]
    block = C.census(cert, prior)["predicates"]["determinism"]
    assert block["scope"]["batch_size"] == "32"
    assert block["usable"] == 1


# --------------------------------------------------------------------------- issuer history


def test_issuer_history_counts_the_keys_own_prior_entries_and_calls_them_dependent():
    prior = [run_cert("sha256:" + str(n) * 64, battery="sha256:" + "z" * 64) for n in (2, 3, 4)]
    result = C.census(run_cert("sha256:" + "1" * 64), prior)
    block = result["predicates"]["issuer_history"]
    assert block["matched"] == 3
    assert block["usable"] == 3
    assert block["independent"] == 0
    assert any("only against itself" in reason for reason in block["blocked_by"])


def test_an_entry_signed_by_another_key_is_independent_constraint():
    prior = [run_cert("sha256:" + "2" * 64, batch=8, issuer=OTHER_ISSUER)]
    result = C.census(run_cert("sha256:" + "1" * 64), prior)
    assert result["constraining_entries"] == 1
    assert result["independent_entries"] == 1
    assert result["predicates"]["floor_agreement"]["independent"] == 1
    assert result["predicates"]["issuer_history"]["matched"] == 0


def test_a_single_issuer_log_reports_zero_independent_constraint():
    prior = [run_cert("sha256:" + str(n) * 64, batch=8) for n in (2, 3, 4)]
    result = C.census(run_cert("sha256:" + "1" * 64), prior)
    assert result["constraining_entries"] == 3
    assert result["independent_entries"] == 0
    assert "0 signed by another key" in result["disclosure"]


# --------------------------------------------------------------------------- shape and safety


def test_the_union_is_over_entries_not_over_predicate_counts():
    """One prior entry in three predicates' scope is one constraining entry, not three."""
    prior = [run_cert("sha256:" + "2" * 64, batch=1)]
    result = C.census(run_cert("sha256:" + "1" * 64, batch=1), prior)
    assert result["predicates"]["floor_agreement"]["usable"] == 1
    assert result["predicates"]["snapshot_agreement"]["usable"] == 1
    assert result["predicates"]["determinism"]["usable"] == 1
    assert result["constraining_entries"] == 1


def test_the_cert_is_never_counted_against_itself():
    cert = run_cert("sha256:" + "1" * 64)
    result = C.census(cert, [cert, dict(cert)])
    assert result["prior_entries"] == 0
    assert result["constraining_entries"] == 0


def test_prior_may_be_index_cert_pairs_and_the_indices_are_reported():
    prior = [(0, plan_cert()), (1, run_cert("sha256:" + "2" * 64, batch=8))]
    result = C.census(run_cert("sha256:" + "1" * 64), prior, index=2)
    assert result["cert"]["index"] == 2
    block = result["predicates"]["floor_agreement"]
    # The plan is in scope on subject and battery but carries no channel values, so it is counted
    # under `in_scope_not_usable` and does not appear beside the run that does constrain.
    assert block["entries"] == [{"id": "sha256:" + "2" * 64, "type": "fingerprint", "index": 1}]
    assert block["in_scope_not_usable"] == 1


def test_a_malformed_cert_yields_a_census_rather_than_an_exception():
    for junk in ({}, {"body": 3, "refs": "x", "subject": None}, {"refs": [1, 2, {"id": 5}]}):
        result = C.census(junk, [run_cert("sha256:" + "2" * 64), {"id": None}])
        assert result["constraining_entries"] == 0
        assert result["first_claim"] is True


def test_every_predicate_in_the_roster_appears_in_every_census():
    result = C.census(run_cert("sha256:" + "1" * 64), [])
    assert list(result["predicates"]) == [name for name, _ in C.PREDICATES]
    for name, asks in C.PREDICATES:
        assert result["predicates"][name]["asks"] == asks


# --------------------------------------------------------------------------- the published log


def test_the_published_log_is_readable_as_seven_entries():
    entries = C.read_log_entries(PUBLISHED_LOG)
    assert [i for i, _ in entries] == list(range(7))
    assert [c["type"] for _, c in entries] == [
        "battery",
        "prereg",
        "fingerprint",
        "fingerprint",
        "fingerprint",
        "fingerprint",
        "fingerprint",
    ]


def test_the_published_floor_is_a_first_claim():
    """The number this module was built to find, on this lab's own flagship artifact.

    Entry 6 is the canonical fingerprint and the noise floor — the verdict
    `first_verdict_2026_09_09` exists to publish. Six entries precede it. All six are its own
    material: the battery it ran, the plan it honours, and the four floor runs it consumes. No
    predicate has anything left to compare it against.
    """
    entries = C.read_log_entries(PUBLISHED_LOG)
    index, cert = entries[6]
    assert cert["body"].get("noise_floor"), "entry 6 is the floor"
    result = C.census(cert, entries[:6], index=index)

    assert result["prior_entries"] == 6
    assert result["own_material_present"] == 6
    assert result["constraining_entries"] == 0
    assert result["independent_entries"] == 0
    assert result["first_claim"] is True
    assert result["verdict"] == "unconstrained"
    for name, _ in C.PREDICATES:
        block = result["predicates"][name]
        assert block["available"] == 0, name
        assert block["usable"] == 0, name
    assert result["predicates"]["floor_agreement"]["matched"] == 6
    assert result["predicates"]["floor_agreement"]["own"] == 6
    assert result["predicates"]["determinism"]["matched"] == 3
    assert result["predicates"]["determinism"]["own"] == 3


def test_the_published_plan_names_no_schedule():
    entries = C.read_log_entries(PUBLISHED_LOG)
    result = C.census(entries[6][1], entries[:6], index=6)
    block = result["predicates"]["schedule"]
    assert block["plans_on_file"] == 1
    assert block["plans_carrying_a_schedule"] == 0
    assert any("names no schedule" in reason for reason in block["blocked_by"])


def test_the_published_logs_constraint_profile():
    """Pinned against the published bytes: the middle runs accrue, the floor drops to zero."""
    summary = C.census_log(PUBLISHED_LOG)
    assert summary["entries"] == 7
    profile = [r["constraining_entries"] for r in summary["per_entry"]]
    assert profile == [0, 0, 0, 1, 2, 3, 0]
    assert summary["unconstrained_entries"] == 4
    assert summary["max_constraining_entries"] == 3
    assert summary["independent_entries_anywhere"] == 0
    assert all(r["independent_entries"] == 0 for r in summary["per_entry"])


def test_the_published_battery_is_the_first_claim_in_the_log():
    entries = C.read_log_entries(PUBLISHED_LOG)
    result = C.census(entries[0][1], [], index=0)
    assert result["prior_entries"] == 0
    assert result["first_claim"] is True


@pytest.mark.parametrize("index", [3, 4, 5])
def test_the_published_middle_runs_are_constrained_by_their_siblings(index):
    """Runs 2, 3 and 4 do not name each other, so each is constrained by the runs below it."""
    entries = C.read_log_entries(PUBLISHED_LOG)
    result = C.census(entries[index][1], entries[:index], index=index)
    assert result["constraining_entries"] == index - 2
    assert result["predicates"]["floor_agreement"]["usable"] == index - 2
    assert result["independent_entries"] == 0

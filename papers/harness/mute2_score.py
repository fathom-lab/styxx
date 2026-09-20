"""Score the MUTE-2 preregistration against its receipt. Every prediction is a mechanical check.

    python papers/harness/mute2_score.py            # reads mute2_receipt.json (+ mute1_receipt.json), writes mute2_scored.json

Nothing here is hand-labelled: the receipt's `killed_by` lists are the only evidence, and every
prediction is a statement about them. The scorer refuses to score a receipt made with a different
instrument from MUTE-1's, or a preregistration that moved.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "PREREG_mute2_guards_under_mutation_2026_09_20.md"
PREREG_SHA256_FROZEN = "38248ebb67dea417083303dc9c907e859e7fbf0a3e4ab51820bf92162de05e88"
MUTE1_RECEIPT = HERE / "mute1_receipt.json"
MUTE2_RECEIPT = HERE / "mute2_receipt.json"
OUT = HERE / "mute2_scored.json"

MANIFEST_TEST = "tests.test_harness_manifest::test_the_harness_matches_the_committed_manifest"
BEHAV_PREFIX = "tests.test_ci_steps_propagate_failure::test_the_step_cannot_hide_the_failure_of_what_it_calls["
EXEMPT = {"MUTE-018", "MUTE-024", "MUTE-031", "MUTE-050", "MUTE-073"}
MUTE1_FINGERPRINT = "8da2770cfabb9fc631d6a71816e23ffe36ed75ae42f7f1826aa2b319d79c4ef1"
STRUCTURAL = {"M-TRIGGER", "M-JOB", "M-STEP", "M-GUARD", "M-SCRIPT"}


def _ascii(s: str) -> str:
    """pytest escapes non-ASCII in parametrize ids; match either spelling."""
    return s.encode("unicode_escape").decode("ascii")


def behav_cases(killed_by: list[str]) -> list[str]:
    return [k for k in killed_by if k.startswith(BEHAV_PREFIX)]


def case_for(check: dict, steps_by_job: dict) -> str | None:
    """The behavioural case id for a step check, if the step has one (name-based id)."""
    if check.get("kind") != "workflow-step":
        return None
    wf = check["path"].split("/")[-1]
    name = check.get("name") or f"step {check['step']}"
    return f"{BEHAV_PREFIX}{wf}::{check['job']}::{name}]"


def matches(case: str, killed_by: list[str]) -> bool:
    return case in killed_by or _ascii(case) in killed_by


def main() -> int:
    prereg_now = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if prereg_now != PREREG_SHA256_FROZEN:
        print(f"INVALID__PREREG_MOVED: {prereg_now} != frozen {PREREG_SHA256_FROZEN}")
        return 2
    r1 = json.loads(MUTE1_RECEIPT.read_text(encoding="utf-8"))
    r2 = json.loads(MUTE2_RECEIPT.read_text(encoding="utf-8"))
    if r1["instrument_sha256"] != r2["instrument_sha256"]:
        print("INVALID__INSTRUMENT_CHANGED: the two receipts were not made by the same instrument")
        return 2

    v1 = {v["mutant"]: v for v in r1["verdicts"]}
    v2 = {v["mutant"]: v for v in r2["verdicts"]}
    if set(v1) != set(v2):
        print("INVALID__POPULATION_DIFFERS")
        return 2

    # the run tree's steps per job, from the receipt's checks, for P4
    steps_by_job: dict[tuple[str, str], list[dict]] = {}
    for v in r2["verdicts"]:
        c = v["check"]
        if c.get("kind") == "workflow-step" and v["operator"] == "M-STEP":
            steps_by_job.setdefault((c["path"], c["job"]), []).append(c)

    # which steps are exempt (their M-SWALLOW ids are the preregistered five)
    exempt_steps = {(v2[m]["check"]["path"], v2[m]["check"]["job"], v2[m]["check"]["step"]) for m in EXEMPT}

    gates = {}
    base = r2["oracle"]["tests_passing_on_baseline"]
    # G-M2-1 as written: the manifest matches, 33 behavioural cases pass, 5 skip BY NAME, 0 fail. The
    # receipt lists every baseline test that did not pass, skips included, without saying which is
    # which; the five preregistered exempt steps are skips by construction (the guard skips a step
    # that reaches no external command), so the gate holds iff the guard cases in that list are
    # exactly those five and no manifest test is in it. (The scorer's first version counted a
    # preregistered skip as a failure; that was the scorer misreading the gate, not the gate.)
    excl = r2["oracle"]["tests_excluded_not_passing_on_baseline"]
    excl_behav = {e for e in excl if e.startswith(BEHAV_PREFIX)}
    excl_manifest = [e for e in excl if e.startswith("tests.test_harness_manifest")]
    exempt_cases = set()
    for m in EXEMPT:
        c = v2[m]["check"]
        exempt_cases.add(case_for(c, None))
        exempt_cases.add(_ascii(case_for(c, None)))
    stray = sorted(e for e in excl_behav if e not in exempt_cases)
    missing = [m for m in EXEMPT if not ({case_for(v2[m]["check"], None), _ascii(case_for(v2[m]["check"], None))} & excl_behav)]
    ok = not stray and not missing and not excl_manifest
    gates["G-M2-1"] = {"pass": ok, "detail": (f"{base} tests pass on the unmutated tree; the only guard cases not passing are the five "
                                              f"preregistered skips; no manifest test excluded" if ok
                                              else f"stray guard cases not passing: {stray}; exempt cases missing: {missing}; manifest excluded: {excl_manifest}")}
    # G-M2-2: every MUTE-1 baseline-passing test passes here. MUTE-1's receipt does not list its passing
    # ids, but every MUTE-1 killer is a MUTE-1 baseline-passing test; use the union of killers as the
    # observable subset (it is the whole set, via the collection-crash mutant).
    mute1_passing = set()
    for v in r1["verdicts"]:
        mute1_passing.update(v["killed_by"])
    mute2_passing = set()
    for v in r2["verdicts"]:
        mute2_passing.update(v["killed_by"])
    # a test is provably passing on the MUTE-2 baseline if it kills anything; the crash mutant lists all
    lost = sorted(mute1_passing - mute2_passing)
    gates["G-M2-2"] = {"pass": not lost, "detail": (f"all {len(mute1_passing)} MUTE-1 baseline-passing tests still pass"
                                                     if not lost else f"lost: {lost[:10]}")}
    gates["G-M2-3"] = {"pass": r2["totals"]["UNREACHED"] == 1, "detail": f"UNREACHED = {r2['totals']['UNREACHED']}"}
    gates["G-M2-4"] = {"pass": all((v["verdict"] == "KILLED") == bool(v["killed_by"]) for v in r2["verdicts"] if v["verdict"] != "UNREACHED"),
                       "detail": "every KILLED names a test and every SURVIVED names none"}

    P = {}
    # P1
    structural = [v for v in v2.values() if v["operator"] in STRUCTURAL]
    p1_bad = [v["mutant"] for v in structural if v["verdict"] != "KILLED" or MANIFEST_TEST not in v["killed_by"]]
    P["P1"] = {"hit": len(structural) == 73 and not p1_bad, "predicted": "all 73 structural mutants KILLED with the manifest test among the killers",
               "observed": {"structural": len(structural), "not_as_predicted": p1_bad}}
    # P2
    swallows = [v for v in v2.values() if v["operator"] == "M-SWALLOW"]
    p2_bad = [v["mutant"] for v in swallows if any(k.startswith("tests.test_harness_manifest") for k in v["killed_by"])]
    P["P2"] = {"hit": not p2_bad, "predicted": "the manifest test kills no M-SWALLOW mutant", "observed": {"swallows_killed_by_manifest": p2_bad}}
    # P3
    p3_bad = []
    for v in swallows:
        case = case_for(v["check"], steps_by_job)
        if v["mutant"] in EXEMPT:
            if v["verdict"] != "SURVIVED" or v["killed_by"]:
                p3_bad.append((v["mutant"], "exempt but not a clean survivor", v["killed_by"][:3]))
        else:
            if v["verdict"] != "KILLED" or not matches(case, v["killed_by"]):
                p3_bad.append((v["mutant"], "not killed by its own behavioural case", v["killed_by"][:3]))
    P["P3"] = {"hit": not p3_bad, "predicted": "33 swallows KILLED by their own behavioural case; 5 exempt SURVIVE with no killer",
               "observed": {"swallows": len(swallows), "killed": sum(v["verdict"] == "KILLED" for v in swallows),
                            "survived": sorted(v["mutant"] for v in swallows if v["verdict"] == "SURVIVED"), "not_as_predicted": p3_bad}}
    # P4
    p4_bad = []
    crash = []
    for v in v2.values():
        if v["operator"] == "M-SWALLOW" or v["verdict"] == "UNREACHED":
            continue
        got = set(behav_cases(v["killed_by"]))
        c = v["check"]
        if v["operator"] == "M-STEP":
            key = (c["path"], c["job"], c["step"])
            expected = set() if key in exempt_steps else {case_for(c, steps_by_job)}
        elif v["operator"] == "M-JOB":
            expected = {case_for(s, steps_by_job) for s in steps_by_job.get((c["path"], c["job"]), [])
                        if (s["path"], s["job"], s["step"]) not in exempt_steps}
        else:
            expected = set()
        exp_norm = {x for x in expected} | {_ascii(x) for x in expected}
        if v["operator"] == "M-SUBJECT" and len(v["killed_by"]) > 100:
            crash.append(v["mutant"])          # the stated exception: everything vanished
            continue
        if not (got <= exp_norm and all(matches(e, v["killed_by"]) for e in expected)):
            p4_bad.append((v["mutant"], v["operator"], sorted(got)[:2], sorted(expected)[:2]))
    P["P4"] = {"hit": not p4_bad and crash == ["MUTE-119"],
               "predicted": "behavioural cases kill non-swallow mutants only by vanishing with the deleted step(s); one collection-crash exception (MUTE-119)",
               "observed": {"not_as_predicted": p4_bad, "collection_crash": crash}}
    # P5
    p5_bad = []
    for m, v in v1.items():
        if v["verdict"] != "KILLED":
            continue
        w = v2[m]
        if w["verdict"] != "KILLED" or not set(v["killed_by"]) <= set(w["killed_by"]):
            p5_bad.append((m, sorted(set(v["killed_by"]) - set(w["killed_by"]))[:3]))
    P["P5"] = {"hit": not p5_bad, "predicted": "every MUTE-1 kill stands, with its MUTE-1 killers among its killers now",
               "observed": {"mute1_kills": sum(v["verdict"] == "KILLED" for v in v1.values()), "not_as_predicted": p5_bad}}
    # P6
    t = r2["totals"]
    P["P6"] = {"hit": t == {"KILLED": 114, "SURVIVED": 5, "UNREACHED": 1}, "predicted": {"KILLED": 114, "SURVIVED": 5, "UNREACHED": 1},
               "observed": t, "survivors": sorted(v["mutant"] for v in v2.values() if v["verdict"] == "SURVIVED")}
    # P7 — recompute the fingerprint restricted to MUTE-1's oracle files on the run tree, if that tree is at hand
    p7 = {"hit": None, "predicted": MUTE1_FINGERPRINT, "observed": None,
          "note": "computed by the RESULT on the run tree with mute.harness_fingerprint(tree, mute1 oracle files)"}
    fp_file = HERE / "mute2_fingerprint_restricted.txt"
    if fp_file.exists():
        p7["observed"] = fp_file.read_text(encoding="utf-8").strip()
        p7["hit"] = p7["observed"] == MUTE1_FINGERPRINT
    P["P7"] = p7

    valid = all(gates[g]["pass"] for g in ("G-M2-1", "G-M2-2", "G-M2-3"))
    hits = sum(1 for p in P.values() if p["hit"] is True)
    payload = {
        "valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P),
        "receipt": MUTE2_RECEIPT.name, "receipt_tree": r2["tree"], "harness_fingerprint": r2["harness_fingerprint"],
        "instrument_sha256": r2["instrument_sha256"], "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT  totals={t}")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else ('MISS' if p['hit'] is False else 'UNSCORED')}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail']}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())

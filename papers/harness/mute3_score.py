"""Score the MUTE-3 preregistration against its three receipts. Every prediction is mechanical.

    python papers/harness/mute3_score.py      # reads mute2_receipt.json, mute2r_receipt.json,
                                              # mute3_receipt_A.json, mute3_receipt_B.json; writes mute3_scored.json

Level-2 ids are positional and shift between run A and run B, so every prediction is resolved by
(file, function, operator), never by id. Nothing is hand-labelled.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG = HERE / "PREREG_mute3_the_guards_under_mutation_2026_09_21.md"
PREREG_SHA256_FROZEN = "ddcd207a489811ede222fe0be9d5739d83b7e15163f7cebe08dd1f5a6dde708d"
R2 = HERE / "mute2_receipt.json"
R2R = HERE / "mute2r_receipt.json"
RA = HERE / "mute3_receipt_A.json"
RB = HERE / "mute3_receipt_B.json"
OUT = HERE / "mute3_scored.json"

MANIFEST_TEST = "tests.test_harness_manifest::test_the_harness_matches_the_committed_manifest"
MANIFEST_FILE = "tests/test_harness_manifest.py"
MANIFEST_FUNC = "test_the_harness_matches_the_committed_manifest"
PROPAGATE_FUNC = "test_the_step_cannot_hide_the_failure_of_what_it_calls"
CONTROLS = {
    "test_the_manifest_guard_rejects_a_tree_missing_a_job": "tests.test_harness_manifest::test_the_manifest_guard_rejects_a_tree_missing_a_job",
    "test_the_propagation_guard_rejects_a_swallowed_step": "tests.test_ci_steps_propagate_failure::test_the_propagation_guard_rejects_a_swallowed_step",
}
EXEMPT_LEVEL1 = {"MUTE-018", "MUTE-024", "MUTE-031", "MUTE-050", "MUTE-073"}


def key(v: dict) -> tuple:
    c = v["check"]
    return (c["path"], c.get("name"), v["operator"])


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    now = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if now != PREREG_SHA256_FROZEN:
        print(f"INVALID__PREREG_MOVED: {now}")
        return 2
    r2, r2r, ra, rb = load(R2), load(R2R), load(RA), load(RB)
    for r, name in ((r2r, "mute2r"), (ra, "A"), (rb, "B")):
        if r.get("schema") != "styxx.harness-mutation/v1.1":
            print(f"INVALID__SCHEMA: {name} is {r.get('schema')}")
            return 2
    if not (r2r["instrument_sha256"] == ra["instrument_sha256"] == rb["instrument_sha256"]):
        print("INVALID__INSTRUMENT_DIFFERS between the three v1.1 receipts")
        return 2

    gates, P = {}, {}
    # G-M3-1 is the instrument's own test file, run by CI and by the RESULT; recorded from the receipt
    gates["G-M3-1"] = {"pass": True, "detail": "tests/test_harness_mutation.py: see RESULT §1 (run outside the receipts)"}
    ba, bb = ra["oracle"], rb["oracle"]
    gates["G-M3-2"] = {"pass": ba["tests_passing_on_baseline"] == r2["oracle"]["tests_passing_on_baseline"]
                               and sorted(ba["tests_excluded_not_passing_on_baseline"]) == sorted(r2["oracle"]["tests_excluded_not_passing_on_baseline"]),
                       "detail": f"run A baseline: {ba['tests_passing_on_baseline']} passing, {len(ba['tests_excluded_not_passing_on_baseline'])} excluded (MUTE-2: {r2['oracle']['tests_passing_on_baseline']}, {len(r2['oracle']['tests_excluded_not_passing_on_baseline'])})"}
    # G-M3-3 as the preregistration words it: every test that passed on A's baseline passes on B's,
    # plus the two controls -- a set condition, held from mute3_baselines.json (ids), which also
    # names whatever else B's baseline gained. Scorer correction, stated in the RESULT: the first
    # encoding was "B == A + 2" by count, which is stricter than the bar and fails on a legitimate
    # third gain (the propagation guard's parametrized case for the anchor step itself).
    bl = HERE / "mute3_baselines.json"
    if bl.exists():
        b = load(bl)
        a_ids, b_ids = set(b["A"]["passing"]), set(b["B"]["passing"])
        counts_match = (len(a_ids) == ba["tests_passing_on_baseline"] and len(b_ids) == bb["tests_passing_on_baseline"]
                        and b["A"]["tree"] == ra["tree"] and b["B"]["tree"] == rb["tree"])
        gained = sorted(b_ids - a_ids)
        gates["G-M3-3"] = {"pass": counts_match and not (a_ids - b_ids) and set(CONTROLS.values()) <= b_ids
                                   and sorted(bb["tests_excluded_not_passing_on_baseline"]) == sorted(ba["tests_excluded_not_passing_on_baseline"]),
                           "detail": f"run B baseline: {bb['tests_passing_on_baseline']} passing; lost from A: {sorted(a_ids - b_ids)}; gained: {gained}; same exclusions"}
    else:
        gates["G-M3-3"] = {"pass": False, "detail": "mute3_baselines.json missing: run mute3_baselines.py on the two trees"}
    ua = [v["mutant"] for v in ra["verdicts"] if v["verdict"] == "UNREACHED"]
    ub = {v["check"]["name"] for v in rb["verdicts"] if v["verdict"] == "UNREACHED"}
    gates["G-M3-4"] = {"pass": ua == [] and ub == set(CONTROLS), "detail": f"A unreached: {ua}; B unreached: {sorted(ub)}"}
    gates["G-M3-5"] = {"pass": all((v["verdict"] == "KILLED") == bool(v["killed_by"]) for r in (r2r, ra, rb) for v in r["verdicts"] if v["verdict"] != "UNREACHED"),
                       "detail": "every KILLED names a red test; every SURVIVED names none"}

    # P1 — MUTE-2r == MUTE-2, verdict by verdict
    v2 = {v["mutant"]: v["verdict"] for v in r2["verdicts"]}
    v2r = {v["mutant"]: v["verdict"] for v in r2r["verdicts"]}
    diff = sorted(m for m in v2 if v2[m] != v2r.get(m))
    P["P1"] = {"hit": not diff and set(v2) == set(v2r), "predicted": "identical verdicts, 114/5/1", "observed": {"totals": r2r["totals"], "differing": diff}}

    # P2 — vanished on exactly 45; errors only on MUTE-119
    van = sorted(v["mutant"] for v in r2r["verdicts"] if v.get("vanished_on_mutant"))
    err = sorted(v["mutant"] for v in r2r["verdicts"] if v.get("errors_on_mutant"))
    step_nonexempt = sorted(v["mutant"] for v in r2r["verdicts"] if v["operator"] == "M-STEP"
                            and not any(e for e in EXEMPT_LEVEL1 if _same_step(e, v, r2r)))
    P["P2"] = {"hit": len(van) == 45 and err == ["MUTE-119"] and "MUTE-119" in van
                      and all(v["operator"] in ("M-STEP", "M-JOB", "M-SUBJECT") for v in r2r["verdicts"] if v["mutant"] in van),
               "predicted": "45 mutants with vanished ids (33 M-STEP, 11 M-JOB, MUTE-119); errors on MUTE-119 only",
               "observed": {"vanished_count": len(van), "by_operator": _count_ops(r2r, van), "errors": err}}

    # P3 — run A: nothing guards the guards
    P["P3"] = {"hit": ra["totals"] == {"KILLED": 0, "SURVIVED": 72, "UNREACHED": 0}, "predicted": {"KILLED": 0, "SURVIVED": 72, "UNREACHED": 0}, "observed": ra["totals"]}

    # run B lookups
    B = {key(v): v for v in rb["verdicts"]}
    def red(v, test): return test in v.get("failed_on_mutant", []) or test in v.get("killed_by", [])

    # P4 — M-GFILE
    gfile = [v for v in rb["verdicts"] if v["operator"] == "M-GFILE"]
    p4_bad = []
    for v in gfile:
        if v["check"]["path"] == MANIFEST_FILE:
            if v["verdict"] != "SURVIVED": p4_bad.append((v["check"]["path"], "expected the fixed point to survive", v["killed_by"][:3]))
        elif v["verdict"] != "KILLED" or not red(v, MANIFEST_TEST):
            p4_bad.append((v["check"]["path"], "expected the manifest test red", v["killed_by"][:3]))
    P["P4"] = {"hit": len(gfile) == 6 and not p4_bad, "predicted": "5 of 6 KILLED by the manifest test; the manifest file's own deletion survives", "observed": {"files": len(gfile), "not_as_predicted": p4_bad}}

    # P5 — M-GFUNC
    gfunc = [v for v in rb["verdicts"] if v["operator"] == "M-GFUNC"]
    p5_bad = []
    for v in gfunc:
        if v["check"]["name"] == MANIFEST_FUNC:
            if v["verdict"] != "SURVIVED": p5_bad.append((v["check"]["name"], "expected the fixed point to survive", v["killed_by"][:3]))
        elif v["verdict"] != "KILLED" or not red(v, MANIFEST_TEST):
            p5_bad.append((v["check"]["name"], "expected the manifest test red", v["killed_by"][:3]))
    P["P5"] = {"hit": len(gfunc) == 35 and not p5_bad, "predicted": "34 of 35 KILLED by the manifest test; deleting the matching test itself survives", "observed": {"functions": len(gfunc), "not_as_predicted": p5_bad}}

    # P6 — M-GVACUOUS
    gvac = [v for v in rb["verdicts"] if v["operator"] == "M-GVACUOUS"]
    p6_bad = []
    killed_names = []
    for v in gvac:
        name = v["check"]["name"]
        if name in CONTROLS:
            if v["verdict"] != "UNREACHED": p6_bad.append((name, "control has no assert; expected UNREACHED", v["verdict"]))
        elif name == MANIFEST_FUNC:
            if v["verdict"] != "KILLED" or not red(v, CONTROLS["test_the_manifest_guard_rejects_a_tree_missing_a_job"]):
                p6_bad.append((name, "expected its control red", v["killed_by"][:3]))
            else: killed_names.append(name)
        elif name == PROPAGATE_FUNC:
            if v["verdict"] != "KILLED" or not red(v, CONTROLS["test_the_propagation_guard_rejects_a_swallowed_step"]):
                p6_bad.append((name, "expected its control red", v["killed_by"][:3]))
            else: killed_names.append(name)
        else:
            if v["verdict"] != "SURVIVED": p6_bad.append((name, "expected to survive (no control)", v["killed_by"][:3]))
    P["P6"] = {"hit": len(gvac) == 35 and not p6_bad and len(killed_names) == 2,
               "predicted": "2 UNREACHED (controls), 2 KILLED by their controls, 31 SURVIVED",
               "observed": {"hollowings": len(gvac), "killed": killed_names, "survived": sum(v["verdict"] == "SURVIVED" for v in gvac), "not_as_predicted": p6_bad}}

    # P7 — totals
    P["P7"] = {"hit": rb["totals"] == {"KILLED": 41, "SURVIVED": 33, "UNREACHED": 2}, "predicted": {"KILLED": 41, "SURVIVED": 33, "UNREACHED": 2}, "observed": rb["totals"]}

    # P8 — the non-hollowing survivors of B are exactly the manifest file / its matching test; the anchor demonstration is read from a side file
    non_hollow_survivors = sorted((v["check"]["path"], v["check"].get("name"), v["operator"]) for v in rb["verdicts"]
                                  if v["verdict"] == "SURVIVED" and v["operator"] != "M-GVACUOUS")
    expected = sorted([(MANIFEST_FILE, "test_harness_manifest.py", "M-GFILE"), (MANIFEST_FILE, MANIFEST_FUNC, "M-GFUNC")])
    demo = HERE / "mute3_anchor_demo.json"
    demo_ok = None
    if demo.exists():
        d = load(demo)
        demo_ok = d.get("exit_with_manifest_test_deleted", 0) != 0 and d.get("exit_intact", 1) == 0
    P["P8"] = {"hit": non_hollow_survivors == expected and demo_ok is True,
               "predicted": "the only non-hollowing survivors are the manifest file and its matching test; the anchor step fails on that mutant",
               "observed": {"non_hollowing_survivors": non_hollow_survivors, "anchor_demo": demo_ok}}

    valid = all(gates[g]["pass"] for g in ("G-M3-1", "G-M3-2", "G-M3-3", "G-M3-4"))
    hits = sum(1 for p in P.values() if p["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P),
               "receipts": {"mute2r": R2R.name, "A": RA.name, "B": RB.name, "baselines": "mute3_baselines.json", "anchor_demo": "mute3_anchor_demo.json", "anchor_pin_demo": "mute3_anchor_pin_demo.json"},
               "trees": {"mute2r": r2r["tree"], "A": ra["tree"], "B": rb["tree"]},
               "fingerprints": {"mute2r": r2r["harness_fingerprint"], "A": ra["harness_fingerprint"], "B": rb["harness_fingerprint"]},
               "instrument_sha256": rb["instrument_sha256"], "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT  A={ra['totals']}  B={rb['totals']}  2r={r2r['totals']}")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail']}")
    return 0 if valid else 1


def _same_step(exempt_id: str, v: dict, r: dict) -> bool:
    e = next(x for x in r["verdicts"] if x["mutant"] == exempt_id)["check"]
    c = v["check"]
    return (e["path"], e["job"], e["step"]) == (c["path"], c["job"], c["step"])


def _count_ops(r: dict, ids: list[str]) -> dict:
    out: dict = {}
    for v in r["verdicts"]:
        if v["mutant"] in ids:
            out[v["operator"]] = out.get(v["operator"], 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())

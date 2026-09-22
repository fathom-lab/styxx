"""Score the SWALLOW-12 preregistration against the live receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow12_score.py     # reads swallow12_receipt.json; writes swallow12_scored.json
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "PREREG_swallow12_the_click_live_2026_09_22.md"
PREREG_SHA256_FROZEN = "9a4d0c6c99461e51e17121cad34507f7fdb06ecf018d133eb81a062c16da69f4"
INSTRUMENT_SHA256_FROZEN = "83be210029e8c647f6e6f20eafafe4b0de6b97002d7cc40cbaf3a792c520577c"
ACTION_SHA256_FROZEN = "e0cb518ed10002c634f0ebb64e95219d320a8212647ee817ebd58605bb9fa55c"
ACTION_YML_SHA256_FROZEN = "83d14bc1558f4098c39d69f18acfa33dd0b1349036e9cc2f90db3a18f1e3cd93"
DIFFERENTIAL_LIVING_SHA256_FROZEN = "95f6ccf1f981a21882af152296d4ae6cd1f27c0d3a12fe66a3d22ea46b321428"
PLAN_SHA256_FROZEN = "1524022e8b9667551b6569c97a39e18e2165a268c0dcdeda3c30e590112e36ba"
RECEIPT = HERE / "swallow12_receipt.json"
PLAN = HERE / "swallow12_plan.json"
OUT = HERE / "swallow12_scored.json"
NO_NEWLINE = ".github/workflows/s12-08-strict-eof-no-newline.yml"
CONTROLS = [".github/workflows/s12-12-outside-diff.yml", ".github/workflows/s12-12b-span-leaves-hunk.yml", ".github/workflows/s12-13-no-repair.yml"]


def main() -> int:
    if PREREG_SHA256_FROZEN and hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    r = json.loads(RECEIPT.read_text(encoding="utf-8"))
    plan_raw = PLAN.read_bytes()
    p = json.loads(plan_raw)
    gates, P = {}, {}
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_harness_live_click.py"),
                        str(ROOT / "tests" / "test_ciaudit_action.py")], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=1800)
    gates["G-S12-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}
    gates["G-S12-2"] = {"pass": hashlib.sha256(plan_raw).hexdigest() == PLAN_SHA256_FROZEN and p["instrument_sha256"] == INSTRUMENT_SHA256_FROZEN
                        and r["plan_instrument_sha256"] == INSTRUMENT_SHA256_FROZEN and r["action_sha256"] == ACTION_SHA256_FROZEN
                        and r["action_yml_sha256"] == ACTION_YML_SHA256_FROZEN and r["differential_living_sha256"] == DIFFERENTIAL_LIVING_SHA256_FROZEN
                        and all(r["live_is_the_plan"]["gate_code"].values()),
                        "detail": f"plan {hashlib.sha256(plan_raw).hexdigest()[:16]}…, instrument {str(r['plan_instrument_sha256'])[:16]}…, action {str(r['action_sha256'])[:16]}…, "
                                  f"action.yml {str(r['action_yml_sha256'])[:16]}…; the live head's gate code: {r['live_is_the_plan']['gate_code']}"}
    lp = r["live_is_the_plan"]
    gates["G-S12-3"] = {"pass": all(lp["base"].values()) and all(lp["head"].values()) and lp["first_run_on_test_merge"],
                        "detail": f"base files identical {sum(lp['base'].values())} of {len(lp['base'])}; head files {sum(lp['head'].values())} of {len(lp['head'])}; "
                                  f"first run on GitHub's test merge: {lp['first_run_on_test_merge']}"}
    changed = r.get("applied_changed_files", [])
    gates["G-S12-4"] = {"pass": lp["applied_parent_is_head"] and lp["applied_by_github"] and sorted(changed) == sorted(p["expected_files"]),
                        "detail": f"parent is the live head: {lp['applied_parent_is_head']}; committed by GitHub: {lp['applied_by_github']}; "
                                  f"changes {len(changed)} files, the plan's {len(p['expected_files'])}: {sorted(changed) == sorted(p['expected_files'])}"}

    c = r["comments"]
    P["P1"] = {"hit": c["equal"] and c["live"] == 16, "predicted": "GitHub accepts the 16 suggestions the rule places; the live comments are the planned ones, byte for byte",
               "observed": {k: c[k] for k in ("planned", "live", "equal", "only_in_plan", "only_live", "authors", "commit_ids")}}
    rf = r["refused"]
    P["P2"] = {"hit": rf["live_first"] == rf["planned_first"] and len(rf["planned"]) == 2,
               "predicted": "the 2 suggestions the rule refuses are refused by GitHub (422, 'outside the diff'), and no other",
               "observed": {"live": rf["live_first"], "planned": rf["planned_first"]}}
    ag = r["again"]
    P["P3"] = {"hit": ag["same_ids"] and ag["comments_after"] == ag["comments_before"] == 16 and rf["live_again"] == rf["planned_again"],
               "predicted": "the re-run posts nothing: the same 16 comments after it; its log line is the planned one",
               "observed": {**ag, "live": rf["live_again"], "planned": rf["planned_again"]}}
    fl = r["files"]
    P["P4"] = {"hit": len(fl) == 15 and all(f["lines_equal"] for f in fl.values()),
               "predicted": "after GitHub's batch commit, each of the 15 files equals the planned text line for line",
               "observed": {"lines_equal": sum(f["lines_equal"] for f in fl.values()), "of": len(fl),
                            "not": sorted(k for k, f in fl.items() if not f["lines_equal"])}}
    nl = {k: f for k, f in fl.items() if f["final_newline_expected"]}
    P["P5"] = {"hit": len(nl) == 14 and all(f["bytes_equal"] for f in nl.values()),
               "predicted": "the 14 files that end in a newline are byte-identical to the planned bytes",
               "observed": {"bytes_equal": sum(f["bytes_equal"] for f in nl.values()), "of": len(nl), "not": sorted(k for k, f in nl.items() if not f["bytes_equal"])}}
    f8 = fl.get(NO_NEWLINE, {})
    P["P6"] = {"hit": bool(f8.get("bytes_equal")), "predicted": "the file with no final newline is byte-identical to the planned bytes: GitHub adds none",
               "observed": {"bytes_equal": f8.get("bytes_equal"), "final_newline_got": f8.get("final_newline_got")}}
    an = r["annotations"]
    ap = r["applied_checks"]
    P["P7"] = {"hit": an["applied"]["equal"] and an["applied"]["conclusion"] == "failure" and sorted({x[0] for x in ap["planned"]}) == CONTROLS,
               "predicted": "the Action on the applied head reports exactly the 3 controls, as 3 error annotations on their planned lines; the check run fails",
               "observed": {k: an["applied"][k] for k in ("planned", "live", "equal", "levels_live", "conclusion", "only_in_plan", "only_live")}}
    P["P8"] = {"hit": an["first"]["equal"] and an["first"]["levels_live"] == {"error": 10, "warning": 9, "notice": 0},
               "predicted": "the first run's check run carries exactly the planned 19 annotations, 10 failure and 9 warning, as planned",
               "observed": {k: an["first"][k] for k in ("planned", "live", "equal", "levels_live", "levels_planned", "only_in_plan", "only_live", "conclusion")}}

    valid = all(gates[g]["pass"] for g in ("G-S12-1", "G-S12-2", "G-S12-3", "G-S12-4"))
    gates["G-S12-5"] = {"pass": len(P) == 8, "detail": "P1–P8 scored"}
    hits = sum(1 for x in P.values() if x["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "pr": r["pr"], "receipt_sha256": hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),
               "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN, "untouched_after_apply": r["untouched_after_apply"]}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, x in P.items():
        print(f"  {k}: {'HIT' if x['hit'] else 'MISS'}  {json.dumps(x['observed'])[:300]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:300]}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())

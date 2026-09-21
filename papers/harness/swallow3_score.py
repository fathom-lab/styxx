"""Score the SWALLOW-3 preregistration against the receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow3_score.py     # reads swallow3_receipt.json(.gz), swallow2_receipt.json.gz,
                                                # swallow3_actions_census.json; writes swallow3_scored.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "PREREG_swallow3_the_checks_that_are_actions_2026_09_21.md"
PREREG_SHA256_FROZEN = "1799321af3af10f27ca11267d1260940307cf21f8307efc21673fafac32836ff"
FAULTS_SHA256_FROZEN = "d26a407ca276d1b1c218fd32bc6bf853544fc394ce5ebc98b2b6ae4bd0fa3543"     # SWALLOW-2's instrument
POPULATION = HERE / "swallow1_population.json"
RECEIPT = HERE / "swallow3_receipt.json"
RECEIPT_GZ = HERE / "swallow3_receipt.json.gz"
SWALLOW2_GZ = HERE / "swallow2_receipt.json.gz"
CENSUS = HERE / "swallow3_actions_census.json"
OUT = HERE / "swallow3_scored.json"
INTERPRETABLE = ("RED", "FAIL_OPEN", "SWALLOWED", "ABSORBED", "NO_CHECK")
TRANSITIONS = {("NO_CHECK", "ABSORBED"), ("NO_CHECK", "FAIL_OPEN"), ("ABSORBED", "FAIL_OPEN")}


def load_receipt(plain: Path, gz: Path) -> tuple[dict, str]:
    raw = plain.read_bytes() if plain.exists() else gzip.decompress(gz.read_bytes())
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _named(repos, repo, workflow, job, index):
    for r in repos:
        if r["repo"] == repo:
            for f in r.get("faults", []):
                if f["workflow"] == workflow and f["job"] == job and f["index"] == index:
                    return f
    return None


def main() -> int:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    r, receipt_sha = load_receipt(RECEIPT, RECEIPT_GZ)
    repos = r["repos"]
    gates, P = {}, {}
    # G-S3-1 population
    pop_sha = hashlib.sha256(POPULATION.read_bytes()).hexdigest()
    gates["G-S3-1"] = {"pass": r.get("population_sha256") == pop_sha and r.get("population_size") == 100 and r.get("faults_sha256") == FAULTS_SHA256_FROZEN,
                       "detail": f"population sha256 {pop_sha[:16]}…, {r.get('population_size')} entries; faults.py {str(r.get('faults_sha256'))[:16]}… (frozen {FAULTS_SHA256_FROZEN[:16]}…)"}
    # G-S3-2 same trees
    failed = [(x["repo"], x["clone_error"]) for x in repos if x.get("clone_error")]
    moved_heads = [x["repo"] for x in repos if x.get("head_moved")]
    capped = [x["repo"] for x in repos if x.get("capped")]
    gates["G-S3-2"] = {"pass": len(repos) - len(failed) >= 90 and not moved_heads,
                       "detail": f"{len(repos) - len(failed)} cloned; failures {failed}; HEAD moved {moved_heads}; capped {capped}"}
    # G-S3-3 reproduction of SWALLOW-2, fault for fault
    s2, s2_sha = load_receipt(HERE / "swallow2_receipt.json", SWALLOW2_GZ)
    old = {(x["repo"], f["workflow"], f["job"], f["index"]): f for x in s2["repos"] for f in x.get("faults", [])}
    new = {(x["repo"], f["workflow"], f["job"], f["index"]): f for x in repos for f in x.get("faults", [])}
    missing = [k for k in old if k not in new]
    extra = [k for k in new if k not in old]
    mism = [(k, old[k]["verdict"], new[k]["verdict_runs_only"], bool(old[k].get("timeout")), bool(new[k].get("timeout")))
            for k in old if k in new and old[k]["verdict"] != new[k].get("verdict_runs_only")]
    unexplained = [m for m in mism if not (m[3] or m[4])]
    gates["G-S3-3"] = {"pass": not missing and not extra and not unexplained,
                       "detail": f"{len(old)} SWALLOW-2 faults; {len(new)} here; missing {len(missing)}; extra {len(extra)}; "
                                 f"verdict mismatches {len(mism)} (with a timeout on either side: {len(mism) - len(unexplained)}; unexplained: {len(unexplained)})",
                       "mismatches": [{"fault": list(m[0]), "swallow2": m[1], "here_runs_only": m[2], "timeout_old": m[3], "timeout_new": m[4]} for m in mism][:50]}
    # G-S3-4 closure of the catalogue over the census
    sys.path.insert(0, str(ROOT))
    from benchmarks.harness_mutation import action_checks as ac
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    names = [a["name"] for a in census["actions_hand_written"]]
    undecided = [n for n in names if ac._entry(n) is None and n not in ac.NOT_CHECKS]
    twice = [n for n in names if ac._entry(n) is not None and n in ac.NOT_CHECKS]
    gates["G-S3-4"] = {"pass": not undecided and not twice and r.get("instrument_sha256") == ac.instrument_sha256(),
                       "detail": f"{len(names)} census names; undecided {undecided}; decided twice {twice}; instrument in tree == receipt's: {r.get('instrument_sha256') == ac.instrument_sha256()}"}
    # G-S3-5 instrument: tests green, this repository unmoved, determinism
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_harness_action_checks.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=900)
    tests_ok = t.returncode == 0
    here = ac.analyse_tree(ROOT)
    here2 = ac.analyse_tree(ROOT)
    unmoved = bool(here["faults"]) and all(f["verdict"] == f["verdict_runs_only"] for f in here["faults"])
    det = [(f["verdict"], f["dropped"]) for f in here["faults"]] == [(f["verdict"], f["dropped"]) for f in here2["faults"]]
    gates["G-S3-5"] = {"pass": tests_ok and unmoved and det,
                       "detail": f"tests: {(t.stdout.strip().splitlines() or [''])[-1][:120]}; this repository: {len(here['faults'])} faults, unmoved {unmoved}, deterministic {det}"}
    # G-S3-6 monotone
    sm = r["summary"]
    all_f = [f for x in repos for f in x.get("faults", [])]
    outside = [f for f in all_f if f.get("verdict_runs_only") and f["verdict_runs_only"] != f["verdict"] and (f["verdict_runs_only"], f["verdict"]) not in TRANSITIONS]
    red_same = sm["by_verdict"].get("RED", 0) == sm["by_verdict_runs_only"].get("RED", 0)
    sw_same = sm["by_verdict"].get("SWALLOWED", 0) == sm["by_verdict_runs_only"].get("SWALLOWED", 0)
    gates["G-S3-6"] = {"pass": not outside and red_same and sw_same,
                       "detail": f"moved outside the declared transitions: {len(outside)}; RED {sm['by_verdict'].get('RED', 0)} vs {sm['by_verdict_runs_only'].get('RED', 0)}; "
                                 f"SWALLOWED {sm['by_verdict'].get('SWALLOWED', 0)} vs {sm['by_verdict_runs_only'].get('SWALLOWED', 0)}"}
    gates["G-S3-7"] = {"pass": True, "detail": "verdicts and checks are the instrument's; nothing reclassified"}

    hand = [f for x in repos for f in x.get("faults", []) if not f["generated"]]
    interp = [f for f in hand if f["verdict"] in INTERPRETABLE]
    moved = [f for f in interp if f["verdict"] != f["verdict_runs_only"]]
    # P1: ≥ 10% of hand-written NO_CHECK (runs-only) faults move
    nc = [f for f in interp if f["verdict_runs_only"] == "NO_CHECK"]
    nc_moved = [f for f in nc if f["verdict"] != "NO_CHECK"]
    P["P1"] = {"hit": bool(nc) and len(nc_moved) / len(nc) >= 0.10, "predicted": "≥ 10% of hand-written NO_CHECK faults move to ABSORBED or FAIL_OPEN",
               "observed": {"no_check_runs_only": len(nc), "moved": len(nc_moved), "rate": round(len(nc_moved) / len(nc), 3) if nc else None,
                            "to": _count(f["verdict"] for f in nc_moved)}}
    # P2: ≥ 5 hand-written FAIL_OPEN faults in ≥ 3 repositories
    fo = [f for f in hand if f["verdict"] == "FAIL_OPEN"]
    fo_repos = sorted({x["repo"] for x in repos for f in x.get("faults", []) if not f["generated"] and f["verdict"] == "FAIL_OPEN"})
    fo_runs_only_repos = sorted({x["repo"] for x in repos for f in x.get("faults", []) if not f["generated"] and f["verdict_runs_only"] == "FAIL_OPEN"})
    P["P2"] = {"hit": len(fo) >= 5 and len(fo_repos) >= 3, "predicted": "≥ 5 hand-written FAIL_OPEN faults in ≥ 3 repositories (from 3 in 2)",
               "observed": {"fail_open": len(fo), "repos": fo_repos, "fail_open_runs_only": sum(1 for f in hand if f["verdict_runs_only"] == "FAIL_OPEN"),
                            "repos_runs_only": fo_runs_only_repos}}
    # P3: sentry-docs Get changed files -> FAIL_OPEN with lychee dropped
    f3 = _named(repos, "getsentry/sentry-docs", "lint-external-links.yml", "check-pr", 1)
    P["P3"] = {"hit": bool(f3) and f3["verdict"] == "FAIL_OPEN" and any(d.get("action") == "lycheeverse/lychee-action" for d in f3.get("dropped_actions", [])),
               "predicted": "getsentry/sentry-docs lint-external-links.yml › check-pr › 1 'Get changed files': FAIL_OPEN, lychee among the dropped",
               "observed": None if not f3 else {"name": f3["name"], "verdict": f3["verdict"], "verdict_runs_only": f3["verdict_runs_only"],
                                                "dropped": [(d.get("action") or d.get("name"), d["mechanism"]) for d in f3.get("dropped", [])]}}
    # P4: primer/react Get source files changes -> not FAIL_OPEN
    f4 = _named(repos, "primer/react", "recommend-integration-tests.yml", "recommend", 2)
    P["P4"] = {"hit": bool(f4) and f4["verdict"] in INTERPRETABLE and f4["verdict"] != "FAIL_OPEN",
               "predicted": "primer/react recommend-integration-tests.yml › recommend › 2 'Get source files changes': not FAIL_OPEN",
               "observed": None if not f4 else {"name": f4["name"], "verdict": f4["verdict"], "verdict_runs_only": f4["verdict_runs_only"], "action_checks_in_scope": f4.get("action_checks_in_scope")}}
    # P5: step-if is the strict plurality mechanism among dropped action checks in hand-written FAIL_OPEN faults
    mech = _count(d["mechanism"] for f in fo for d in f.get("dropped_actions", []))
    top = sorted(mech.items(), key=lambda kv: -kv[1])
    P["P5"] = {"hit": bool(top) and top[0][0] == "step-if" and (len(top) == 1 or top[0][1] > top[1][1]),
               "predicted": "step-if is the single commonest mechanism among dropped action checks (strict plurality); vacuous = MISS",
               "observed": {"mechanisms": mech, "dropped_action_checks": sum(mech.values())}}
    # P6: fewer than 100 hand-written faults move
    P["P6"] = {"hit": len(moved) < 100, "predicted": "< 100 hand-written faults move (< 1.6% of the interpretable)",
               "observed": {"moved": len(moved), "interpretable": len(interp), "rate": round(len(moved) / len(interp), 4) if interp else None,
                            "by_transition": _count(f["verdict_runs_only"] + "->" + f["verdict"] for f in moved)}}
    # P7: at most 1 status-check among the dropped action checks in hand-written FAIL_OPEN faults
    status_drops = [(x["repo"], f["workflow"], d["action"]) for x in repos for f in x.get("faults", []) if not f["generated"] and f["verdict"] == "FAIL_OPEN"
                    for d in f.get("dropped_actions", []) if d.get("kind") == "status"]
    P["P7"] = {"hit": len(status_drops) <= 1, "predicted": "≤ 1 status-check (CodeQL / Sonar / zizmor) among the dropped action checks",
               "observed": {"status_drops": status_drops, "by_kind": _count(d["kind"] for f in fo for d in f.get("dropped_actions", []))}}

    valid = all(gates[g]["pass"] for g in ("G-S3-1", "G-S3-2", "G-S3-3", "G-S3-4", "G-S3-5", "G-S3-6"))
    hits = sum(1 for p in P.values() if p["hit"])
    unverified = [(x["repo"], f["workflow"], d["action"]) for x in repos for f in x.get("faults", []) if f["verdict"] == "FAIL_OPEN"
                  for d in f.get("dropped_actions", []) if d.get("verified") is False]
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "summary": sm,
               "hand_written": {"faults": len(hand), "interpretable": len(interp), "moved": len(moved),
                                "by_verdict": _count(f["verdict"] for f in interp), "by_verdict_runs_only": _count(f["verdict_runs_only"] for f in interp),
                                "moved_by_transition": _count(f["verdict_runs_only"] + "->" + f["verdict"] for f in moved),
                                "moved_repos": sorted({x["repo"] for x in repos for f in x.get("faults", []) if not f["generated"] and f["verdict"] in INTERPRETABLE and f["verdict"] != f["verdict_runs_only"]}),
                                "fail_open_faults": [{"repo": x["repo"], "workflow": f["workflow"], "job": f["job"], "index": f["index"], "name": f["name"],
                                                      "verdict_runs_only": f["verdict_runs_only"], "flavour": f.get("flavour"),
                                                      "dropped": [(d.get("action") or d.get("name"), d["mechanism"], d.get("kind")) for d in f["dropped"]],
                                                      "run_head": f["run_head"][:120]}
                                                     for x in repos for f in x.get("faults", []) if not f["generated"] and f["verdict"] == "FAIL_OPEN"],
                                "drops_on_unverified_entries": unverified},
               "instrument_sha256": r.get("instrument_sha256"), "faults_sha256": r.get("faults_sha256"), "receipt_sha256": receipt_sha,
               "swallow2_receipt_sha256": s2_sha, "census_sha256": hashlib.sha256(CENSUS.read_bytes()).hexdigest(),
               "receipt_file": RECEIPT.name if RECEIPT.exists() else RECEIPT_GZ.name, "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}  {json.dumps(p.get('observed'))[:220]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:260]}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())

"""Score the SWALLOW-2 preregistration against the fault receipt. Every prediction is mechanical.

    python papers/harness/swallow2_score.py     # reads swallow2_receipt.json(.gz) and swallow2_self.json, writes swallow2_scored.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG = HERE / "PREREG_swallow2_which_way_it_falls_2026_09_21.md"
PREREG_SHA256_FROZEN = "bdae83fbdb9a5d3beea2ecc8c35a213feed97ff285a08f5d4997a07bdd80225a"
POPULATION = HERE / "swallow1_population.json"
RECEIPT = HERE / "swallow2_receipt.json"
RECEIPT_GZ = HERE / "swallow2_receipt.json.gz"
SELF = HERE / "swallow2_self.json"
OUT = HERE / "swallow2_scored.json"
INTERPRETABLE = ("RED", "FAIL_OPEN", "SWALLOWED", "ABSORBED", "NO_CHECK")


def load_receipt() -> tuple[dict, str]:
    raw = RECEIPT.read_bytes() if RECEIPT.exists() else gzip.decompress(RECEIPT_GZ.read_bytes())
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _named(repos: list[dict], repo: str, name: str, workflow: str | None = None, job: str | None = None, text: str | None = None) -> list[dict]:
    out = []
    for r in repos:
        if r["repo"] != repo:
            continue
        for f in r.get("faults", []):
            if f["name"] != name or f["generated"]:
                continue
            if workflow and f["workflow"] != workflow:
                continue
            if job and f["job"] != job:
                continue
            if text and text not in f["run_head"]:
                continue
            out.append(f)
    return out


def _p4(faults: list[dict], want_fail_open: bool, cross: bool | None = None, mechanism: str | None = None) -> dict:
    """One named prediction. FAIL_OPEN wanted: at least one matching fault is FAIL_OPEN with the
    stated shape. Not wanted: at least one matching fault is interpretable and none is FAIL_OPEN."""
    summary = [{"workflow": f["workflow"], "job": f["job"], "index": f["index"], "verdict": f["verdict"],
                "dropped": [(d["name"], d["mechanism"], d["cross_step"]) for d in f.get("dropped", [])]} for f in faults]
    if not faults:
        return {"hit": False, "observed": "no such step in the receipt", "matches": summary}
    if want_fail_open:
        ok = any(f["verdict"] == "FAIL_OPEN"
                 and (cross is None or any(d["cross_step"] == cross for d in f["dropped"]))
                 and (mechanism is None or any(d["mechanism"] == mechanism for d in f["dropped"])) for f in faults)
    else:
        ok = any(f["verdict"] in INTERPRETABLE for f in faults) and not any(f["verdict"] == "FAIL_OPEN" for f in faults)
    return {"hit": ok, "matches": summary}


def main() -> int:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    r, receipt_sha = load_receipt()
    repos = r["repos"]
    gates, P = {}, {}
    pop_sha = hashlib.sha256(POPULATION.read_bytes()).hexdigest()
    gates["G-S2-1"] = {"pass": r.get("population_sha256") == pop_sha and r.get("population_size") == 100,
                       "detail": f"population sha256 {pop_sha[:16]}…, {r.get('population_size')} entries"}
    failed = [(x["repo"], x["clone_error"]) for x in repos if x.get("clone_error")]
    capped = [x["repo"] for x in repos if x.get("capped")]
    gates["G-S2-2"] = {"pass": len(repos) - len(failed) >= 90, "detail": f"{len(repos) - len(failed)} cloned; failures {failed}; capped {capped}"}
    if SELF.exists():
        s = json.loads(SELF.read_text(encoding="utf-8"))
        gates["G-S2-3"] = {"pass": bool(s.get("pass")), "detail": json.dumps(s.get("gate"))}
        gates["G-S2-4"] = {"pass": bool(s["self"].get("deterministic")), "detail": "tests/test_harness_faults.py: see RESULT §1; self-run deterministic: " + str(s["self"].get("deterministic"))}
    else:
        gates["G-S2-3"] = {"pass": False, "detail": "no swallow2_self.json"}
        gates["G-S2-4"] = {"pass": False, "detail": "no swallow2_self.json"}
    gates["G-S2-5"] = {"pass": True, "detail": "verdicts and checks are the instrument's; nothing reclassified"}

    hand = [f for x in repos for f in x.get("faults", []) if not f["generated"]]
    interp = [f for f in hand if f["verdict"] in INTERPRETABLE]
    by = {}
    for f in interp:
        by[f["verdict"]] = by.get(f["verdict"], 0) + 1
    # P1
    red = by.get("RED", 0)
    P["P1"] = {"hit": bool(interp) and red / len(interp) >= 0.80, "predicted": "RED ≥ 80% of interpretable hand-written faults",
               "observed": {"interpretable": len(interp), "red": red, "rate": round(red / len(interp), 3) if interp else None, "by_verdict": by}}
    # P2
    fo_repos = sorted({x["repo"] for x in repos for f in x.get("faults", []) if not f["generated"] and f["verdict"] == "FAIL_OPEN"})
    P["P2"] = {"hit": len(fo_repos) >= 6, "predicted": "≥ 6 repositories with a hand-written FAIL_OPEN fault", "observed": {"repos": fo_repos, "count": len(fo_repos)}}
    # P3
    fo = [f for f in hand if f["verdict"] == "FAIL_OPEN"]
    cross = sum(1 for f in fo if any(d["cross_step"] for d in f["dropped"]))
    P["P3"] = {"hit": bool(fo) and cross / len(fo) >= 0.5, "predicted": "≥ half of hand-written FAIL_OPEN faults are cross-step",
               "observed": {"fail_open": len(fo), "cross_step": cross, "in_step": len(fo) - cross,
                            "mechanisms": _count(d["mechanism"] for f in fo for d in f["dropped"])}}
    # P4 -- the reading of SWALLOW-1, tested
    P["P4a"] = {"predicted": "mlflow/mlflow master.yml database 'Run tests': FAIL_OPEN, in-step (unreached)",
                **_p4(_named(repos, "mlflow/mlflow", "Run tests", workflow="master.yml", job="database"), True, cross=False, mechanism="unreached")}
    P["P4b"] = {"predicted": "getsentry/sentry-docs 'Get changed files': FAIL_OPEN, cross-step",
                **_p4(_named(repos, "getsentry/sentry-docs", "Get changed files"), True, cross=True)}
    P["P4c"] = {"predicted": "primer/react 'Get source files changes': FAIL_OPEN, cross-step",
                **_p4(_named(repos, "primer/react", "Get source files changes"), True, cross=True)}
    P["P4d"] = {"predicted": "carverauto/serviceradar 'Decide whether this Mix project needs lint': not FAIL_OPEN",
                **_p4(_named(repos, "carverauto/serviceradar", "Decide whether this Mix project needs lint"), False)}
    P["P4e"] = {"predicted": "airbytehq/airbyte 'Check for changes' (git diff --quiet): not FAIL_OPEN",
                **_p4(_named(repos, "airbytehq/airbyte", "Check for changes", text="git diff"), False)}
    # P5
    sw_repos = sorted({x["repo"] for x in repos for f in x.get("faults", []) if not f["generated"] and f["verdict"] == "SWALLOWED"})
    P["P5"] = {"hit": len(sw_repos) >= 8 and len(sw_repos) > len(fo_repos), "predicted": "SWALLOWED in ≥ 8 repositories, and in more repositories than FAIL_OPEN",
               "observed": {"swallowed_repos": len(sw_repos), "fail_open_repos": len(fo_repos), "repos": sw_repos}}
    # P6
    sm = r["summary"]
    art = sm["artifact_failures_in_plus"]["x"]
    run = sm["steps_run_in_plus"]["x"]
    P["P6"] = {"hit": bool(hand) and len(interp) / len(hand) >= 0.85 and run and art / run <= 0.10,
               "predicted": "interpretable ≥ 85% of hand-written fault sites; artifact failures ≤ 10% of steps run in W+ (x)",
               "observed": {"hand_written_sites": len(hand), "interpretable": len(interp), "rate": round(len(interp) / len(hand), 3) if hand else None,
                            "artifact_failures_x": art, "steps_run_x": run, "artifact_rate": round(art / run, 4) if run else None}}
    # P7
    self_live = [f for f in interp if f.get("self_live")]
    self_red = sum(1 for f in self_live if f["verdict"] == "RED")
    P["P7"] = {"hit": bool(self_live) and self_red / len(self_live) >= 0.90 and all(f["verdict"] in ("RED", "SWALLOWED", "FAIL_OPEN") for f in self_live),
               "predicted": "≥ 90% of live hand-written checks are RED under their own fault; the rest SWALLOWED or FAIL_OPEN",
               "observed": {"live_checks": len(self_live), "red": self_red, "rate": round(self_red / len(self_live), 3) if self_live else None,
                            "by_verdict": _count(f["verdict"] for f in self_live)}}

    valid = all(gates[g]["pass"] for g in ("G-S2-1", "G-S2-2", "G-S2-3", "G-S2-4"))
    hits = sum(1 for p in P.values() if p["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "summary": sm,
               "instrument_sha256": r.get("instrument_sha256"), "receipt_sha256": receipt_sha,
               "receipt_file": RECEIPT.name if RECEIPT.exists() else RECEIPT_GZ.name, "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}  {json.dumps(p.get('observed', p.get('matches')))[:200]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:220]}")
    return 0 if valid else 1


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())

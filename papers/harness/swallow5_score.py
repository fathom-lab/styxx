"""Score the SWALLOW-5 preregistration against the structural-repair receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow5_score.py     # reads swallow5_receipt.json(.gz); writes swallow5_scored.json
"""
from __future__ import annotations

import gzip
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "PREREG_swallow5_the_structural_repairs_2026_09_21.md"
PREREG_SHA256_FROZEN = "9bb6f856d51632a888c2574ba498c18445321dc7c1ab31282d73c7c06847da28"
FAULTS_SHA256_FROZEN = "d26a407ca276d1b1c218fd32bc6bf853544fc394ce5ebc98b2b6ae4bd0fa3543"
ACTIONS_SHA256_FROZEN = "0e723694d459ca2368799e3fc21a26d06e70fb89ad09bd2b0152466bf2a68f72"
REPAIR_SHA256_FROZEN = "7b9a1695d316c2ce109495cf60c5a9a1de03bac204e9bf48fdbc2a1ac66012b9"
RECEIPT = HERE / "swallow5_receipt.json"
RECEIPT_GZ = HERE / "swallow5_receipt.json.gz"
SOURCE = HERE / "swallow4_receipt.json.gz"
OUT = HERE / "swallow5_scored.json"
REPAIRS = ("guard-status", "no-default", "both-structural")


def load_receipt() -> tuple[dict, str]:
    raw = RECEIPT.read_bytes() if RECEIPT.exists() else gzip.decompress(RECEIPT_GZ.read_bytes())
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _named(targets, repo, workflow, job, index):
    return next((t for t in targets if t["repo"] == repo and t["workflow"] == workflow and t["job"] == job and t["index"] == index), None)


def _lines(t):
    return next(c["lines_changed"] for c in t["candidates"] if c["repair"] == t["verified_repair"])


def main() -> int:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    r, receipt_sha = load_receipt()
    targets = [dict(t, repo=x["repo"]) for x in r["repos"] for t in x["targets"] if "candidates" in t]
    missing = [(x["repo"], t) for x in r["repos"] for t in x["targets"] if "candidates" not in t]
    gates, P = {}, {}
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_harness_repair_structural.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=900)
    gates["G-S5-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}
    excluded = [t for t in targets if t["baseline"].get("differs_from_receipt") or t["baseline"].get("verdict") is None]
    gates["G-S5-2"] = {"pass": not missing and len(targets) + len(missing) == 18 and len(excluded) <= 2,
                       "detail": f"{len(targets)} targets; baseline differs or absent: {len(excluded)}; missing: {len(missing)}",
                       "excluded": [{"repo": t["repo"], "workflow": t["workflow"], "job": t["job"], "index": t["index"], "receipt": t["verdict"], "here": t["baseline"].get("verdict")} for t in excluded]}
    ok = [t for t in targets if t not in excluded]
    hand = [t for t in ok if not t["generated"]]
    gen = [t for t in ok if t["generated"]]
    bad = []
    for t in ok:
        interp = [f for f, v in t["baseline"]["by_flavour"].items() if v in ("RED", "FAIL_OPEN", "SWALLOWED", "ABSORBED", "NO_CHECK")]
        for c in t["candidates"]:
            if c.get("verified") and (not all(c["loud_by_flavour"].get(f) for f in interp) or not all(c["unchanged_by_flavour"].values())):
                bad.append((t["repo"], t["workflow"], t["job"], t["index"], c["repair"]))
    gates["G-S5-3"] = {"pass": not bad, "detail": f"verified candidates failing a half: {len(bad)}", "bad": bad[:20]}
    outside = [t for t in targets if t["verified_repair"] not in (None,) + REPAIRS or any(c["repair"] not in REPAIRS for c in t["candidates"])]
    gates["G-S5-4"] = {"pass": not outside and list(r.get("repairs", [])) == list(REPAIRS) and r.get("repair_sha256") == REPAIR_SHA256_FROZEN,
                       "detail": f"candidates outside the stated repairs: {len(outside)}; receipt repairs {r.get('repairs')}; repair.py {str(r.get('repair_sha256'))[:16]}…"}
    src_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    gates["G-S5-5"] = {"pass": r.get("faults_sha256") == FAULTS_SHA256_FROZEN and r.get("action_checks_sha256") == ACTIONS_SHA256_FROZEN
                       and r.get("repair_sha256") == REPAIR_SHA256_FROZEN and r.get("source_receipt_sha256") == src_sha,
                       "detail": f"faults {str(r.get('faults_sha256'))[:16]}…, action_checks {str(r.get('action_checks_sha256'))[:16]}…, repair {str(r.get('repair_sha256'))[:16]}…, source {str(r.get('source_receipt_sha256'))[:16]}… (file {src_sha[:16]}…)"}

    hv = [t for t in hand if t["verified_repair"]]
    P["P1"] = {"hit": len(hv) >= 7, "predicted": "≥ 7 of the 14 hand-written residue targets verified",
               "observed": {"hand_written": len(hand), "verified": len(hv), "by_repair": _count(t["verified_repair"] for t in hv),
                            "which": [(t["repo"], t["name"], t["verified_repair"]) for t in hv]}}
    by = _count(t["verified_repair"] for t in hv)
    P["P2"] = {"hit": by.get("guard-status", 0) >= by.get("no-default", 0), "predicted": "guard-status verifies ≥ as many as no-default", "observed": by}
    named = [("hmislk/hmis", "development_pr_validation.yml", "validate-jdbc-data-sources", "guard-status"),
             ("dotnet/roslyn", "pr-validation.yml", "validate-and-trigger", "guard-status"),
             ("selfxyz/self", "mobile-deploy.yml", "build-ios", "no-default"),
             ("langfuse/langfuse", "pipeline.yml", "tests-shared", "no-default"),
             ("antiwork/gumroad", "ci-green.yml", "report", None),
             ("dotnet/aspire", "reproduce-flaky-tests.yml", "reproduce", None)]
    for k, (repo, wf, job, want) in enumerate(named):
        cands = [t for t in targets if t["repo"] == repo and t["workflow"] == wf and t["job"] == job]
        t = cands[0] if cands else None
        obs = None if t is None else {"name": t["name"], "verified_repair": t["verified_repair"], "index": t["index"],
                                      "candidates": [(c["repair"], c.get("applies"), c.get("loud"), c.get("unchanged"), (c.get("why") or "")[:90]) for c in t["candidates"]]}
        hit = t is not None and t in ok and ((t["verified_repair"] == want) if want else (t["verified_repair"] is None))
        P[f"P3{'abcdef'[k]}"] = {"hit": hit, "predicted": f"{repo} {wf} › {job}: {'verified by ' + want if want else 'not verified'}", "observed": obs}
    loops = [t for t in hand if (t["repo"], t["workflow"]) in (("mlflow/mlflow", "master.yml"), ("nodetool-ai/nodetool", "mutation-testing.yaml"))
             or (t["repo"] == "dotnet/aspire" and t["workflow"] == "specialized-test-runner.yml")]
    P["P4"] = {"hit": len(loops) == 5 and all(t["verified_repair"] is None for t in loops), "predicted": "mlflow, nodetool and aspire's three runsheet checks: none verified",
               "observed": [(t["repo"], t["name"], t["verified_repair"]) for t in loops]}
    sizes = sorted(_lines(t) for t in hv)
    P["P5"] = {"hit": bool(sizes) and max(sizes) <= 8 and statistics.median(sizes) <= 5, "predicted": "every verified hand-written repair ≤ 8 lines; median ≤ 5",
               "observed": {"sizes": sizes, "median": statistics.median(sizes) if sizes else None}}
    rejected = [t for t in hand if any(c.get("applies") and c.get("unchanged") is False for c in t["candidates"])]
    P["P6"] = {"hit": len(rejected) >= 1, "predicted": "≥ 1 hand-written target with an applying candidate rejected for changing a healthy run",
               "observed": [(t["repo"], t["name"], next(c["repair"] for c in t["candidates"] if c.get("unchanged") is False)) for t in rejected]}
    gv = [t for t in gen if t["verified_repair"]]
    P["P7"] = {"hit": len(gv) <= 2, "predicted": "≤ 2 of 4 generated residue targets verified",
               "observed": {"generated": len(gen), "verified": len(gv), "which": [(t["repo"], t["workflow"], t["name"], t["verified_repair"]) for t in gv]}}

    valid = all(gates[g]["pass"] for g in ("G-S5-1", "G-S5-2", "G-S5-3", "G-S5-4", "G-S5-5"))
    hits = sum(1 for p in P.values() if p["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "summary": r["summary"],
               "hand_written": {"targets": len(hand), "verified": len(hv), "by_repair": by,
                                "residue": [{"repo": t["repo"], "workflow": t["workflow"], "job": t["job"], "index": t["index"], "name": t["name"], "verdict": t["verdict"],
                                             "verified_repair": t["verified_repair"], "run_head": t["run_head"][:100],
                                             "candidates": [(c["repair"], (c.get("why") or "verified")[:110], c.get("lines_changed")) for c in t["candidates"]]}
                                            for t in hand]},
               "instrument_sha256": r.get("instrument_sha256"), "receipt_sha256": receipt_sha,
               "receipt_file": RECEIPT.name if RECEIPT.exists() else RECEIPT_GZ.name, "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}  {json.dumps(p.get('observed'))[:230]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:240]}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())

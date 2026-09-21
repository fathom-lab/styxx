"""Score the SWALLOW-4 preregistration against the repair receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow4_score.py     # reads swallow4_receipt.json(.gz); writes swallow4_scored.json
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
PREREG = HERE / "PREREG_swallow4_the_repair_is_loud_2026_09_21.md"
PREREG_SHA256_FROZEN = "0069be79eada6501354f8bde314bda80553f30f2a539f6e6677211eb25f93610"
FAULTS_SHA256_FROZEN = "d26a407ca276d1b1c218fd32bc6bf853544fc394ce5ebc98b2b6ae4bd0fa3543"
ACTIONS_SHA256_FROZEN = "0e723694d459ca2368799e3fc21a26d06e70fb89ad09bd2b0152466bf2a68f72"
SOURCE_RECEIPT_SHA256 = "609e664549df085b"      # prefix of the file sha256 of swallow3_receipt.json.gz, as the prereg names it
RECEIPT = HERE / "swallow4_receipt.json"
RECEIPT_GZ = HERE / "swallow4_receipt.json.gz"
OUT = HERE / "swallow4_scored.json"
REPAIRS = ("no-continue-on-error", "strict-shell", "both")


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
    # G-S4-1 instrument
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_harness_repair.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=900)
    gates["G-S4-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}
    # G-S4-2 baseline
    hand = [t for t in targets if not t["generated"]]
    excluded = [t for t in targets if t["baseline"].get("differs_from_receipt") or t["baseline"].get("verdict") is None]
    excluded_hand = [t for t in excluded if not t["generated"]]
    gates["G-S4-2"] = {"pass": not missing and (len(excluded_hand) <= 0.05 * len(hand) if hand else False),
                       "detail": f"{len(targets)} targets ({len(hand)} hand-written); baseline differs or absent: {len(excluded)} ({len(excluded_hand)} hand-written); missing clones/workflows: {len(missing)}",
                       "excluded": [{"repo": t["repo"], "workflow": t["workflow"], "job": t["job"], "index": t["index"], "receipt": t["verdict"],
                                     "here": t["baseline"].get("verdict")} for t in excluded]}
    ok_targets = [t for t in targets if t not in excluded]
    hand = [t for t in ok_targets if not t["generated"]]
    gen = [t for t in ok_targets if t["generated"]]
    # G-S4-3 both halves
    bad = []
    for t in ok_targets:
        for c in t["candidates"]:
            if c.get("verified"):
                interp = [f for f, v in t["baseline"]["by_flavour"].items() if v in ("RED", "FAIL_OPEN", "SWALLOWED", "ABSORBED", "NO_CHECK")]
                if not all(c["loud_by_flavour"].get(f) for f in interp) or not all(c["unchanged_by_flavour"].values()):
                    bad.append((t["repo"], t["workflow"], t["job"], t["index"], c["repair"]))
            if c.get("loud") and c.get("unchanged") is False and c.get("verified"):
                bad.append((t["repo"], t["workflow"], t["job"], t["index"], c["repair"], "loud-but-changed-yet-verified"))
    gates["G-S4-3"] = {"pass": not bad, "detail": f"verified candidates failing a half: {len(bad)}", "bad": bad[:20]}
    # G-S4-4 no hand labels
    outside = [t for t in targets if t["verified_repair"] not in (None,) + REPAIRS or any(c["repair"] not in REPAIRS for c in t["candidates"])]
    gates["G-S4-4"] = {"pass": not outside and list(r.get("repairs", [])) == list(REPAIRS), "detail": f"candidates outside the stated repairs: {len(outside)}; receipt repairs {r.get('repairs')}"}
    # G-S4-5 frozen underneath
    src_sha = hashlib.sha256((HERE / "swallow3_receipt.json.gz").read_bytes()).hexdigest()
    gates["G-S4-5"] = {"pass": r.get("faults_sha256") == FAULTS_SHA256_FROZEN and r.get("action_checks_sha256") == ACTIONS_SHA256_FROZEN
                       and r.get("source_receipt_sha256") == src_sha and src_sha.startswith(SOURCE_RECEIPT_SHA256),
                       "detail": f"faults {str(r.get('faults_sha256'))[:16]}…, action_checks {str(r.get('action_checks_sha256'))[:16]}…, source receipt {str(r.get('source_receipt_sha256'))[:16]}… (file {src_sha[:16]}…)"}

    # P1
    sw = [t for t in hand if t["verdict"] == "SWALLOWED"]
    sw_v = [t for t in sw if t["verified_repair"]]
    P["P1"] = {"hit": len(sw) >= 1 and len(sw_v) >= 40, "predicted": "≥ 40 of the 49 hand-written SWALLOWED faults verified",
               "observed": {"swallowed": len(sw), "verified": len(sw_v), "by_repair": _count(t["verified_repair"] for t in sw_v),
                            "coe": {"targets": sum(1 for t in sw if t["continue_on_error"]), "verified": sum(1 for t in sw_v if t["continue_on_error"])},
                            "shell": {"targets": sum(1 for t in sw if not t["continue_on_error"]), "verified": sum(1 for t in sw_v if not t["continue_on_error"])}}}
    # P2
    shell = [t for t in sw if not t["continue_on_error"]]
    rejected = [t for t in shell if not t["verified_repair"] and any(c.get("applies") and c.get("unchanged") is False for c in t["candidates"])]
    P["P2"] = {"hit": len(rejected) >= 3, "predicted": "≥ 3 of the 16 shell-hidden SWALLOWED faults rejected by the twin condition",
               "observed": {"shell_hidden": len(shell), "rejected": len(rejected),
                            "which": [(t["repo"], t["workflow"], t["name"], next((c["why"][:90] for c in t["candidates"] if c.get("unchanged") is False), None)) for t in rejected]}}
    # P3 -- four named
    want = [("antiwork/gumroad", "tests.yml", "run_scope", 0, True), ("getsentry/sentry-docs", "lint-external-links.yml", "check-pr", 1, True),
            ("crewAIInc/crewAI", "update-test-durations.yml", "update-durations", 4, True), ("antiwork/gumroad", "ci-green.yml", "report", 1, False)]
    for k, (repo, wf, job, idx, want_verified) in enumerate(want, start=1):
        t = _named(targets, repo, wf, job, idx)
        obs = None if t is None else {"verdict": t["verdict"], "baseline": t["baseline"].get("verdict"), "verified_repair": t["verified_repair"],
                                      "candidates": [(c["repair"], c.get("applies"), c.get("loud"), c.get("unchanged"), (c.get("why") or "")[:80]) for c in t["candidates"]]}
        hit = t is not None and t in ok_targets and (bool(t["verified_repair"]) == want_verified)
        P[f"P3{'abcd'[k-1]}"] = {"hit": hit, "predicted": f"{repo} {wf} › {job} › {idx}: {'verified' if want_verified else 'not verified'}", "observed": obs}
    # P4
    fewer = [t for t in hand if t.get("fewer")]
    P["P4"] = {"hit": sum(1 for t in fewer if t["verified_repair"]) >= 4, "predicted": "≥ 4 of the 8 'fewer' faults verified",
               "observed": {"fewer": len(fewer), "verified": sum(1 for t in fewer if t["verified_repair"]),
                            "which": [(t["repo"], t["name"], t["verified_repair"]) for t in fewer]}}
    # P5
    sizes = sorted(_lines(t) for t in hand if t["verified_repair"])
    P["P5"] = {"hit": bool(sizes) and statistics.median(sizes) <= 3 and max(sizes) <= 10, "predicted": "median verified hand-written repair ≤ 3 lines; max ≤ 10",
               "observed": {"n": len(sizes), "median": statistics.median(sizes) if sizes else None, "max": max(sizes) if sizes else None, "sizes": sizes}}
    # P6
    gen_v = [t for t in gen if t["verified_repair"]]
    P["P6"] = {"hit": bool(gen) and len(gen_v) / len(gen) >= 0.90 and (bool(gen_v) and sum(1 for t in gen_v if t["verified_repair"] == "no-continue-on-error") / len(gen_v) >= 0.90),
               "predicted": "≥ 90% of generated targets verified, ≥ 90% of those by no-continue-on-error alone",
               "observed": {"generated": len(gen), "verified": len(gen_v), "by_repair": _count(t["verified_repair"] for t in gen_v)}}
    # P7
    empty_diff = [t for t in ok_targets if t["verified_repair"] and not next(c["diff"] for c in t["candidates"] if c["repair"] == t["verified_repair"]).strip()]
    undone = []
    for t in ok_targets:
        if t["verified_repair"] in ("no-continue-on-error", "strict-shell"):
            both = next((c for c in t["candidates"] if c["repair"] == "both"), None)
            if both and both.get("applies") and not both.get("verified"):
                undone.append((t["repo"], t["workflow"], t["job"], t["index"], both.get("why")))
    P["P7"] = {"hit": not empty_diff and not undone, "predicted": "no verified repair with an empty diff; `both` verified whenever a smaller repair is and it applies",
               "observed": {"empty_diffs": len(empty_diff), "both_undoes": undone[:10], "n_undone": len(undone)}}

    valid = all(gates[g]["pass"] for g in ("G-S4-1", "G-S4-2", "G-S4-3", "G-S4-4", "G-S4-5"))
    hits = sum(1 for p in P.values() if p["hit"])
    fo = [t for t in hand if t["verdict"] == "FAIL_OPEN"]
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P), "summary": r["summary"],
               "hand_written": {"targets": len(hand), "verified": sum(1 for t in hand if t["verified_repair"]),
                                "by_repair": _count(t["verified_repair"] for t in hand if t["verified_repair"]),
                                "not_verified": [{"repo": t["repo"], "workflow": t["workflow"], "job": t["job"], "index": t["index"], "name": t["name"], "verdict": t["verdict"],
                                                  "coe": t["continue_on_error"], "run_head": t["run_head"][:100],
                                                  "candidates": [(c["repair"], (c.get("why") or "verified")[:110]) for c in t["candidates"]]}
                                                 for t in hand if not t["verified_repair"]],
                                "fail_open": [{"repo": t["repo"], "workflow": t["workflow"], "job": t["job"], "index": t["index"], "verified_repair": t["verified_repair"]} for t in fo]},
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

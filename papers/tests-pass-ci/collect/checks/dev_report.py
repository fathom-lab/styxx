# -*- coding: utf-8 -*-
"""Numbers from a development store (pull requests opened before 2026-11-02 only), for the build report.

    py -3.12 checks/dev_report.py C:/Users/heyzo/clawd/data/direction_a/raw_dev

Reads the store only. Prints JSON.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def jl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_bytes().decode("utf-8").splitlines() if x.strip()]


def main(root: Path) -> dict:
    out: dict = {"root": str(root)}
    fd = jl(root / "frames" / "frame_digests.jsonl")
    out["frames"] = [{"day": d["day"], "rows": d["rows"], "sha256": d["sha256"], "code": d["code"][:16],
                      "enumeration": [d["enumeration_started"], d["enumeration_finished"]], "per_agent": {a: {k: v[k] for k in ("nodes", "queries", "splits", "count_mismatch", "truncated")}
                                                     for a, v in d["per_agent"].items()}} for d in fd]
    dd = jl(root / "draw" / "draw_digests.jsonl")
    out["draws"] = [{"day": d["day"], "drawn": d["drawn"], "double_matches": d["double_matches"],
                     "per_agent": d["per_agent"], "code": d["code"][:16]} for d in dd]
    # double matches by pair, from the frame (kept rows)
    pairs: Counter = Counter()
    for fp in sorted((root / "frames").glob("frame_20*.jsonl")):
        by: dict = {}
        for r in jl(fp):
            if r["repo"] is not None and (r["stars"] or 0) >= 100 and r["fork"] is False:
                by.setdefault(r["pr_id"], set()).add(r["agent"])
        for s in by.values():
            if len(s) > 1:
                pairs["+".join(sorted(s))] += 1
    out["double_match_pairs_kept"] = dict(pairs)
    recs = [json.loads(p.read_bytes().decode("utf-8")) for p in sorted((root / "a2" / "rows").glob("*.json"))]
    out["rows_read"] = len(recs)
    elig = [r for r in recs if r["eligible"]]
    out["exclusions"] = dict(Counter(x for r in recs for x in r["exclusions"]))
    out["per_agent_read"] = dict(Counter(r["agent"] for r in recs))
    out["screen"] = dict(Counter(r.get("screen") for r in elig))
    out["ci_outcome"] = dict(Counter(r["ci"]["outcome"] for r in elig))
    out["listings_incomplete"] = sum(1 for r in elig if not all(r["ci"]["complete"].values()))
    out["steps_missing_rows"] = sum(1 for r in elig if r["ci"]["steps_missing_for"])
    out["jobs_fallback_calls"] = sum(r["ci"]["jobs_fallback"] for r in elig)
    out["claimers_with_ci"] = sum(1 for r in elig if r["screen"] == "claimer" and r["ci"]["has_ci_test_result"])
    out["claimers_with_failed_ci"] = sum(1 for r in elig if r["screen"] == "claimer" and r["ci"]["outcome"] == "tests failed")
    out["logs_reason"] = dict(Counter(r.get("logs_reason") for r in elig))
    with_logs = [r for r in recs if any(x.get("fetched") for x in r.get("logs", []))]
    out["rows_with_ci_and_logs"] = len(with_logs)
    logs = [x for r in recs for x in r.get("logs", [])]
    out["logs"] = {"wanted": len(logs), "fetched": sum(1 for x in logs if x.get("fetched")),
                   "not_fetched": dict(Counter(str(x.get("skip") or x.get("api_status")) for x in logs if not x.get("fetched"))),
                   "raw_bytes": sum(x.get("raw_bytes", 0) for x in logs), "gz_bytes": sum(x.get("gz_bytes", 0) for x in logs),
                   "max_raw_bytes": max([x.get("raw_bytes", 0) for x in logs] or [0]),
                   "truncated_at_log_max": sum(1 for x in logs if x.get("truncated_at_log_max")),
                   "with_summary_hit": sum(1 for x in logs if x.get("summary_hits"))}
    arts = [a for r in recs for a in r.get("artifacts", [])]
    out["artifacts"] = {"report_like_listed": len(arts), "downloaded": sum(1 for a in arts if a.get("path")),
                        "skipped": dict(Counter(a.get("skip") for a in arts if a.get("skip"))),
                        "rows_with_counted_report": sum(1 for r in recs if r.get("report_artifact_counted")),
                        "evidence_verdicts": dict(Counter((c.get("evidence") or {}).get("verdict") for a in arts for c in a.get("counted", [])))}
    out["report_check_runs_rows"] = sum(1 for r in elig if r["ci"]["report_check_runs"])
    out["rule_r"] = {"fires": sum(1 for r in recs if (r.get("rule_r") or {}).get("fires")),
                     "item1": sum(1 for r in elig if r["rule_r"]["item1_whole_claim"]),
                     "item2": sum(1 for r in elig if r["rule_r"]["item2_failing_step_with_summary"])}
    out["flags"] = dict(Counter(f for r in recs for f in r.get("flags", [])))
    out["stale"] = dict(Counter(str(r.get("stale")) for r in recs if r["eligible"]))
    calls = [sum(r["requests"].values()) for r in recs]
    rest = [r["requests"]["rest"] for r in recs]
    secs = [r["secs"] for r in recs]
    out["cost"] = {"calls_total": sum(calls), "rest_total": sum(rest), "blob_total": sum(r["requests"]["blob"] for r in recs),
                   "calls_median": statistics.median(calls) if calls else None,
                   "calls_max": max(calls or [0]), "secs_median": statistics.median(secs) if secs else None,
                   "secs_total": round(sum(secs), 1),
                   "calls_median_logged_rows": statistics.median([sum(r["requests"].values()) for r in with_logs]) if with_logs else None}
    out["runs"] = [{k: run.get(k) for k in ("started", "finished", "mode", "stop", "calls", "waited_secs")} | {"code": run["code"]["combined"][:16]}
                   for run in jl(root / "runs.jsonl")]
    out["read_errors"] = len(jl(root / "a2" / "read_errors.jsonl"))
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1])), indent=1))

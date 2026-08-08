"""Audit the LIVE sense collector's output for recorder-induced n inflation.

Run against the collector's own store. Nothing is hand-entered; the receipt is the output.

    python papers/first-afference/audit_sense_collector.py

The question: of the paired rows the collector has recorded, how many are independent
observations of the agent, and how many are the same agent row carried forward by the
staleness window into a later sample? A harness that samples faster than its slowest channel
writes will report the recorder's own cadence as data — the failure class named in
``papers/SYNTHESIS_recorder_contamination_2026_08_06.md``.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

STORE = Path.home() / ".styxx" / "sense.jsonl"
CHART = Path.home() / ".styxx" / "chart.jsonl"
OUT = Path(__file__).resolve().parent / "sense_collector_audit.json"


def _rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    se, ch = _rows(STORE), _rows(CHART)
    st = sorted(r["ts"] for r in se)
    ct = sorted(r["ts"] for r in ch if isinstance(r.get("ts"), (int, float)))
    w0, w1 = st[0], st[-1]
    hours = (w1 - w0) / 3600.0
    writes_in_window = [t for t in ct if w0 <= t <= w1]

    paired = [r for r in se if r.get("agent") is not None and r.get("host") is not None]
    sigs = collections.Counter(json.dumps(r["agent"], sort_keys=True) for r in paired)
    distinct = len(sigs)
    most = sigs.most_common(1)[0][1] if sigs else 0

    res = {
        "store": str(STORE), "chart": str(CHART),
        "window_hours": round(hours, 2),
        "n_samples": len(se),
        "sample_rate_per_hour": round(len(se) / hours, 2),
        "n_agent_writes_in_window": len(writes_in_window),
        "agent_write_rate_per_hour": round(len(writes_in_window) / hours, 2),
        "oversampling_ratio": round((len(se) / hours) / max(len(writes_in_window) / hours, 1e-9), 2),

        "n_paired_rows_nominal": len(paired),
        "n_distinct_agent_payloads": distinct,
        "max_repeats_of_one_payload": most,
        "n_duplicate_paired_rows": len(paired) - distinct,
        "duplicate_fraction_of_pairs": round((len(paired) - distinct) / max(len(paired), 1), 4),
        "effective_n_upper_bound": distinct,
        "n_inflation_factor": round(len(paired) / max(distinct, 1), 2),

        "r1v2_required_bins": 200,
        "hours_to_requirement_at_nominal_rate": round(200 / max(len(paired) / hours, 1e-9), 1),
        "hours_to_requirement_at_distinct_rate": round(200 / max(distinct / hours, 1e-9), 1),

        "interpretation": (
            "The collector samples faster than the agent writes, and the staleness window carries "
            "one agent row forward into several samples. The paired-row count is therefore a "
            "property of the recorder's cadence, not of the agent. Any statistic computed over "
            "the nominal count would score the staleness window."),
        "why_the_obvious_fix_is_forbidden": (
            "Widening max_age raises the nominal pair count and lowers the effective one — it "
            "makes the contamination worse while making the harness look healthier. That is the "
            "exact failure class of SYNTHESIS_recorder_contamination_2026_08_06.md. The correct "
            "fix is to emit a bin only when the agent row is NEW, which cannot inflate n because "
            "it cannot produce more bins than the agent produced rows."),
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for k in ("window_hours", "n_samples", "agent_write_rate_per_hour", "oversampling_ratio",
              "n_paired_rows_nominal", "n_distinct_agent_payloads", "max_repeats_of_one_payload",
              "duplicate_fraction_of_pairs", "n_inflation_factor",
              "hours_to_requirement_at_distinct_rate"):
        print(f"  {k} = {res[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

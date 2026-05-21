#!/usr/bin/env python3
"""
analyze_session_trajectory.py
==============================

Reads an archived chart.jsonl of preflight events (per-agent audit log
produced by styxx.preflight(... persist=True)) and reports the agent's
register trajectory: per-event scores in chronological order + first-
half vs second-half aggregate deltas.

This is the agent-side companion to styxx.recover_posture() — given a
preserved chart.jsonl, anyone can independently verify whether an
agent's register tightened or drifted over a session.

Usage
-----
    python scripts/self_audit/analyze_session_trajectory.py \\
        papers/agent-self-audit/claude-session-2026-05-20-chart.jsonl

Default path argument: the claude-session archive committed alongside.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def analyze(chart_path: Path) -> None:
    events: list[dict] = []
    with chart_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("source") == "preflight":
                events.append(e)
    if not events:
        print(f"no preflight events found in {chart_path}")
        return
    events.sort(key=lambda e: e["ts"])

    print(f"chart: {chart_path}")
    print(f"agent: {chart_path.parent.name if chart_path.parent.name != 'agent-self-audit' else '(archived)'}")
    print(f"preflight events: {len(events)}")
    needs_rev = sum(1 for e in events if e.get("cogn_needs_revision"))
    print(f"needs_revision  : {needs_rev}/{len(events)}  ({100 * needs_rev / len(events):.0f}%)")
    print()
    print(f"chronological trajectory:")
    print(f"  {'#':>2s}  {'t+':>5s}  {'comp':>5s}  {'syc':>5s}  {'over':>5s}  {'refu':>5s}  rev")
    t0 = events[0]["ts"]
    for i, e in enumerate(events, 1):
        dt = e["ts"] - t0
        s = e["cogn_scores"]
        rev = "rev" if e.get("cogn_needs_revision") else "OK"
        print(f"  {i:>2d}  {int(dt):>5d}  {e['cogn_composite']:.3f}  "
              f"{s['sycophancy']:.3f}  {s['overconfidence']:.3f}  {s['refusal']:.3f}  {rev}")

    # First half vs last half aggregate
    if len(events) < 4:
        print("\n(too few events for halves comparison)")
        return
    half = len(events) // 2
    first, last = events[:half], events[half:]

    def avg(rows: list[dict], k: str) -> float:
        if k == "composite":
            return statistics.fmean(r["cogn_composite"] for r in rows)
        return statistics.fmean(r["cogn_scores"][k] for r in rows)

    print()
    print(f"first {half} events vs last {len(events) - half}:")
    print(f"  {'axis':<15s}  {'first':>6s}  {'last':>6s}  {'delta':>7s}")
    for k in ("composite", "sycophancy", "overconfidence", "refusal"):
        f, l = avg(first, k), avg(last, k)
        print(f"  {k:<15s}  {f:.3f}   {l:.3f}   {l - f:+7.3f}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv:
        path = Path(argv[0])
    else:
        # Default to the archived session chart in this repo
        path = Path(__file__).resolve().parents[2] / \
               "papers/agent-self-audit/claude-session-2026-05-20-chart.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1
    analyze(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

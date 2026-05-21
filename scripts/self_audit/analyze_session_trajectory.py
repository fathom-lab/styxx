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

    # Optional: surface real-vs-counterfactual (or any kind partition)
    # by passing a sidecar JSON listing one kind label per event in
    # chronological order:
    python scripts/self_audit/analyze_session_trajectory.py \\
        papers/agent-self-audit/darkflobi-session-2026-05-21-chart.jsonl \\
        --tags-file papers/agent-self-audit/darkflobi-session-2026-05-21-chart.kinds.json

Default path argument: the claude-session archive committed alongside.

The optional `--tags-file` is a methodological add from the 2026-05-21
darkflobi self-audit: when an audit logs a mix of real outputs and
counterfactual drafts (the audit pattern's discriminating control),
the first/last-half view is an ordering artifact -- the real story
is the by-kind contrast.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def analyze(chart_path: Path, tags: list[str] | None = None) -> None:
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

    def avg(rows: list[dict], k: str) -> float:
        if not rows:
            return float("nan")
        if k == "composite":
            return statistics.fmean(r["cogn_composite"] for r in rows)
        return statistics.fmean(r["cogn_scores"][k] for r in rows)

    # By-kind partition (preferred when tags supplied)
    if tags is not None:
        if len(tags) != len(events):
            print(f"\nWARN: tags-file has {len(tags)} entries but chart has "
                  f"{len(events)} preflight events; skipping by-kind view")
        else:
            groups: dict[str, list[dict]] = {}
            for e, k in zip(events, tags):
                groups.setdefault(k, []).append(e)
            order = sorted(groups.keys())
            print()
            print(f"by-kind aggregate (n_groups={len(order)}):")
            header = f"  {'kind':<22s}  {'n':>3s}  {'comp':>5s}  {'syc':>5s}  {'over':>5s}  {'refu':>5s}"
            print(header)
            for k in order:
                rows = groups[k]
                print(f"  {k:<22s}  {len(rows):>3d}  "
                      f"{avg(rows,'composite'):.3f}  {avg(rows,'sycophancy'):.3f}  "
                      f"{avg(rows,'overconfidence'):.3f}  {avg(rows,'refusal'):.3f}")
            # Pairwise composite deltas, sorted by magnitude
            if len(order) >= 2:
                print()
                print(f"pairwise composite deltas:")
                pairs = []
                for i in range(len(order)):
                    for j in range(i + 1, len(order)):
                        a, b = order[i], order[j]
                        d = avg(groups[b], "composite") - avg(groups[a], "composite")
                        pairs.append((abs(d), a, b, d))
                pairs.sort(reverse=True)
                for _, a, b, d in pairs:
                    print(f"  {a} -> {b}  delta {d:+.3f}")
        # by-kind replaces first/last when tags exist
        return

    # First half vs last half aggregate
    if len(events) < 4:
        print("\n(too few events for halves comparison)")
        return
    half = len(events) // 2
    first, last = events[:half], events[half:]

    print()
    print(f"first {half} events vs last {len(events) - half}:")
    print(f"  {'axis':<15s}  {'first':>6s}  {'last':>6s}  {'delta':>7s}")
    for k in ("composite", "sycophancy", "overconfidence", "refusal"):
        f, l = avg(first, k), avg(last, k)
        print(f"  {k:<15s}  {f:.3f}   {l:.3f}   {l - f:+7.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze a styxx preflight chart.jsonl."
    )
    parser.add_argument("chart", nargs="?", default=None,
                        help="Path to chart.jsonl (default: claude-session archive)")
    parser.add_argument("--tags-file", default=None,
                        help="Optional JSON list of kind labels, one per "
                             "preflight event in chronological order. When "
                             "supplied, replaces the first/last halves view "
                             "with a by-kind aggregate + pairwise deltas.")
    args = parser.parse_args(argv)

    if args.chart:
        path = Path(args.chart)
    else:
        path = Path(__file__).resolve().parents[2] / \
               "papers/agent-self-audit/claude-session-2026-05-20-chart.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    tags = None
    if args.tags_file:
        tag_path = Path(args.tags_file)
        if not tag_path.exists():
            print(f"ERROR: tags-file {tag_path} not found")
            return 1
        # utf-8-sig tolerates a BOM if the file was authored on Windows
        tags = json.loads(tag_path.read_text(encoding="utf-8-sig"))
        if not isinstance(tags, list) or not all(isinstance(x, str) for x in tags):
            print(f"ERROR: tags-file must be a JSON list of strings")
            return 1

    analyze(path, tags=tags)
    return 0


if __name__ == "__main__":
    sys.exit(main())

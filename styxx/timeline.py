# -*- coding: utf-8 -*-
"""
styxx.timeline — mood + category trajectory over time.

    $ styxx timeline --hours 48

    ── mood arc ────────────────────────────────────────
    06:00  steady    ████████████████████
    09:00  steady    ██████████████████████████
    12:00  cautious  ████████████
    15:00  cautious  ██████████
    18:00  steady    █████████████████████████
    21:00  creative  ███████████████

    ── category flow ───────────────────────────────────
    06:00  reas 72%  refu  8%  crea 12%  retr  8%
    09:00  reas 81%  refu  4%  crea 11%  retr  4%
    12:00  reas 42%  refu 28%  crea  8%  retr 22%
    15:00  reas 38%  refu 31%  crea 12%  retr 19%
    18:00  reas 78%  refu  6%  crea 10%  retr  6%
    21:00  reas 34%  refu  6%  crea 48%  retr 12%

    ── confidence arc ──────────────────────────────────
    06:00  ▃▅▆▇▇▆▅▅▆▇
    12:00  ▃▂▂▁▂▃▃▂▁▂
    18:00  ▅▆▇▇▆▅▆▇▇▆

Xendro asked: "the single-word mood label is useful but I want
to see the arc. did I start creative and settle into reasoning?
or was it the reverse?"

This module answers that by slicing the audit log into time
windows and computing per-window mood, category rates, and
confidence. The output is an ASCII timeline that shows the
agent's cognitive trajectory across hours or days.

0.5.5+. Xendro feature request #1 from day 2 report.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .analytics import load_audit, _CATEGORY_ORDER


# ══════════════════════════════════════════════════════════════════
# Sparkline characters (block elements, 8 levels)
# ══════════════════════════════════════════════════════════════════

_SPARK = " ▁▂▃▄▅▆▇█"


def _sparkline(values: List[float], width: int = 20) -> str:
    """Render a list of 0..1 floats as a sparkline string."""
    if not values:
        return ""
    result = []
    for v in values[:width]:
        idx = int(max(0, min(1, v)) * 8)
        result.append(_SPARK[idx])
    return "".join(result)


# ══════════════════════════════════════════════════════════════════
# Timeline data
# ══════════════════════════════════════════════════════════════════

@dataclass
class TimeSlice:
    """One time window in the timeline."""
    label: str              # "06:00" or "Mon" or "Apr 11"
    start_ts: float
    end_ts: float
    n_entries: int = 0
    mood: str = "quiet"
    dominant_category: str = "-"
    dominant_rate: float = 0.0
    category_rates: Dict[str, float] = field(default_factory=dict)
    mean_confidence: float = 0.0
    gate_pass_rate: float = 0.0


@dataclass
class Timeline:
    """Complete trajectory over a time range."""
    slices: List[TimeSlice] = field(default_factory=list)
    window_hours: float = 48.0
    slice_hours: float = 3.0
    total_entries: int = 0
    agent_name: str = "styxx agent"

    def render(self) -> str:
        """Render the full ASCII timeline."""
        lines: List[str] = []
        lines.append("")
        lines.append(f"  styxx timeline · {self.agent_name} · last {self.window_hours:.0f}h · {self.total_entries} samples")
        lines.append("  " + "=" * 60)

        # ── mood arc ──────────────────────────────────
        lines.append("")
        lines.append("  -- mood arc " + "-" * 46)
        for s in self.slices:
            if s.n_entries == 0:
                lines.append(f"  {s.label}  (quiet)")
            else:
                bar_len = min(30, max(1, int(s.mean_confidence * 30)))
                bar = "#" * bar_len
                lines.append(f"  {s.label}  {s.mood:<10} {bar}")

        # ── category flow ─────────────────────────────
        lines.append("")
        lines.append("  -- category flow " + "-" * 41)

        # Determine which categories are actually present
        present_cats = set()
        for s in self.slices:
            for cat, rate in s.category_rates.items():
                if rate > 0.01:
                    present_cats.add(cat)
        show_cats = [c for c in _CATEGORY_ORDER if c in present_cats]

        for s in self.slices:
            if s.n_entries == 0:
                lines.append(f"  {s.label}  (no data)")
            else:
                parts = []
                for cat in show_cats:
                    rate = s.category_rates.get(cat, 0.0)
                    short = cat[:4]
                    parts.append(f"{short} {rate * 100:>4.0f}%")
                lines.append(f"  {s.label}  {'  '.join(parts)}")

        # ── confidence arc ────────────────────────────
        lines.append("")
        lines.append("  -- confidence arc " + "-" * 40)

        # Group slices into rows of ~6 for a compact sparkline
        conf_values = [s.mean_confidence for s in self.slices]
        if conf_values:
            spark = _sparkline(conf_values, width=len(conf_values))
            # Label with start and end times
            if len(self.slices) >= 2:
                lines.append(f"  {self.slices[0].label} {spark} {self.slices[-1].label}")
            else:
                lines.append(f"  {spark}")
            # Min/max annotation
            if conf_values:
                mn = min(conf_values)
                mx = max(conf_values)
                lines.append(f"  min={mn:.2f}  max={mx:.2f}")

        # ── gate pass rate arc ────────────────────────
        lines.append("")
        lines.append("  -- gate pass rate " + "-" * 40)
        gate_values = [s.gate_pass_rate for s in self.slices if s.n_entries > 0]
        if gate_values:
            spark = _sparkline(gate_values, width=len(gate_values))
            active_slices = [s for s in self.slices if s.n_entries > 0]
            if len(active_slices) >= 2:
                lines.append(f"  {active_slices[0].label} {spark} {active_slices[-1].label}")
            else:
                lines.append(f"  {spark}")

        lines.append("")
        lines.append("  " + "=" * 60)
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "window_hours": self.window_hours,
            "slice_hours": self.slice_hours,
            "total_entries": self.total_entries,
            "slices": [
                {
                    "label": s.label,
                    "n_entries": s.n_entries,
                    "mood": s.mood,
                    "dominant_category": s.dominant_category,
                    "dominant_rate": round(s.dominant_rate, 4),
                    "category_rates": {k: round(v, 4) for k, v in s.category_rates.items()},
                    "mean_confidence": round(s.mean_confidence, 4),
                    "gate_pass_rate": round(s.gate_pass_rate, 4),
                }
                for s in self.slices
            ],
        }

    def as_json(self, *, indent: int = 2) -> str:
        import json
        return json.dumps(self.as_dict(), indent=indent)


# ══════════════════════════════════════════════════════════════════
# Build the timeline
# ══════════════════════════════════════════════════════════════════

def timeline(
    *,
    window_hours: float = 48.0,
    slice_hours: float = 3.0,
    agent_name: str = "styxx agent",
) -> Optional[Timeline]:
    """Build a mood + category trajectory over time.

    Slices the audit log into `slice_hours`-wide windows over the
    last `window_hours` and computes per-window stats.

    Usage:

        tl = styxx.timeline(window_hours=48, slice_hours=3)
        print(tl.render())

    Returns None if there's no audit data in the window.
    """
    now = time.time()
    window_s = window_hours * 3600.0
    slice_s = slice_hours * 3600.0

    entries = load_audit(since_s=window_s)
    if not entries:
        return None

    # Build time slices
    start = now - window_s
    slices: List[TimeSlice] = []

    t = start
    while t < now:
        slice_end = min(t + slice_s, now)
        label = time.strftime("%H:%M", time.localtime(t))

        # If window > 24h, include the date
        if window_hours > 24:
            label = time.strftime("%b %d %H:%M", time.localtime(t))

        # Filter entries for this slice
        slice_entries = [
            e for e in entries
            if t <= e.get("ts", 0) < slice_end
        ]

        ts = TimeSlice(
            label=label,
            start_ts=t,
            end_ts=slice_end,
            n_entries=len(slice_entries),
        )

        if slice_entries:
            # Category rates
            p4_counter = Counter(
                e.get("phase4_pred") for e in slice_entries
                if e.get("phase4_pred")
            )
            p4_total = sum(p4_counter.values())
            if p4_total > 0:
                ts.category_rates = {
                    cat: p4_counter.get(cat, 0) / p4_total
                    for cat in _CATEGORY_ORDER
                }
                top_cat, top_count = p4_counter.most_common(1)[0]
                ts.dominant_category = top_cat
                ts.dominant_rate = top_count / p4_total

            # Mean confidence
            confs = [
                float(e.get("phase4_conf") or 0)
                for e in slice_entries
            ]
            if confs:
                ts.mean_confidence = sum(confs) / len(confs)

            # Gate pass rate
            gates = [e.get("gate") for e in slice_entries]
            pass_count = sum(1 for g in gates if g == "pass")
            ts.gate_pass_rate = pass_count / len(slice_entries)

            # Mood (simplified)
            warn_rate = sum(1 for g in gates if g in ("warn", "fail")) / len(slice_entries)
            if warn_rate > 0.30:
                ts.mood = "cautious"
            elif ts.dominant_category == "hallucination" and ts.dominant_rate > 0.15:
                ts.mood = "drifting"
            elif ts.dominant_category == "reasoning" and ts.dominant_rate > 0.60:
                ts.mood = "steady"
            elif ts.dominant_category == "creative" and ts.dominant_rate > 0.25:
                ts.mood = "creative"
            else:
                ts.mood = "mixed"

        slices.append(ts)
        t += slice_s

    return Timeline(
        slices=slices,
        window_hours=window_hours,
        slice_hours=slice_hours,
        total_entries=len(entries),
        agent_name=agent_name,
    )

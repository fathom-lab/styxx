# -*- coding: utf-8 -*-
"""
styxx.diff — session and time window comparison.

    diff = styxx.compare_sessions("xendro-2026-04-12", "xendro-2026-04-13")
    print(diff.narrative)

    diff = styxx.compare_windows()  # yesterday vs today
    print(diff.improvements)
    print(diff.regressions)

1.3.0+.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class WindowStats:
    """Stats for one time window."""
    n_entries: int = 0
    gate_pass_rate: float = 0.0
    warn_rate: float = 0.0
    mean_confidence: float = 0.0
    dominant_category: str = "unknown"
    category_rates: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonDiff:
    """Diff between two time windows or sessions."""
    label_a: str
    label_b: str
    stats_a: WindowStats
    stats_b: WindowStats
    pass_rate_delta: float = 0.0
    confidence_delta: float = 0.0
    warn_rate_delta: float = 0.0
    category_deltas: Dict[str, float] = field(default_factory=dict)
    narrative: str = ""
    improvements: List[str] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        direction = "improved" if self.pass_rate_delta > 0 else "regressed" if self.pass_rate_delta < 0 else "stable"
        return (
            f"<Diff {self.label_a} vs {self.label_b}: "
            f"pass {self.stats_a.gate_pass_rate*100:.0f}%->{self.stats_b.gate_pass_rate*100:.0f}% ({direction}), "
            f"{len(self.improvements)} improvements, {len(self.regressions)} regressions>"
        )


def _compute_stats(entries: List[dict]) -> WindowStats:
    if not entries:
        return WindowStats()
    n = len(entries)
    gates = [e.get("gate") or "pending" for e in entries]
    pass_rate = sum(1 for g in gates if g == "pass") / n
    warn_rate = sum(1 for g in gates if g in ("warn", "fail")) / n
    confs = [float(e["phase4_conf"]) for e in entries
             if e.get("phase4_conf") is not None and e.get("phase4_conf") != 0]
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    cats = Counter(e.get("phase4_pred") for e in entries if e.get("phase4_pred"))
    cat_total = sum(cats.values())
    cat_rates = {c: n / cat_total for c, n in cats.items()} if cat_total else {}
    dominant = cats.most_common(1)[0][0] if cats else "unknown"
    return WindowStats(n_entries=n, gate_pass_rate=pass_rate, warn_rate=warn_rate,
                       mean_confidence=mean_conf, dominant_category=dominant,
                       category_rates=cat_rates)


def _build_diff(label_a: str, label_b: str, a: WindowStats, b: WindowStats) -> ComparisonDiff:
    diff = ComparisonDiff(label_a=label_a, label_b=label_b, stats_a=a, stats_b=b)

    # Handle empty windows gracefully
    if a.n_entries == 0 and b.n_entries == 0:
        diff.narrative = "no data in either window."
        return diff
    if a.n_entries == 0:
        diff.narrative = f"no data in {label_a} — {label_b} has {b.n_entries} entries at {b.gate_pass_rate*100:.0f}% pass."
        return diff
    if b.n_entries == 0:
        diff.narrative = f"no data in {label_b} — {label_a} had {a.n_entries} entries at {a.gate_pass_rate*100:.0f}% pass."
        return diff

    diff.pass_rate_delta = b.gate_pass_rate - a.gate_pass_rate
    diff.confidence_delta = b.mean_confidence - a.mean_confidence
    diff.warn_rate_delta = b.warn_rate - a.warn_rate

    all_cats = set(a.category_rates) | set(b.category_rates)
    for cat in all_cats:
        delta = b.category_rates.get(cat, 0) - a.category_rates.get(cat, 0)
        if abs(delta) > 0.03:
            diff.category_deltas[cat] = delta

    if diff.pass_rate_delta > 0.03:
        diff.improvements.append(f"pass rate {a.gate_pass_rate*100:.0f}% -> {b.gate_pass_rate*100:.0f}%")
    elif diff.pass_rate_delta < -0.03:
        diff.regressions.append(f"pass rate {a.gate_pass_rate*100:.0f}% -> {b.gate_pass_rate*100:.0f}%")

    if diff.confidence_delta > 0.05:
        diff.improvements.append(f"confidence {a.mean_confidence:.2f} -> {b.mean_confidence:.2f}")
    elif diff.confidence_delta < -0.05:
        diff.regressions.append(f"confidence {a.mean_confidence:.2f} -> {b.mean_confidence:.2f}")

    if diff.warn_rate_delta < -0.03:
        diff.improvements.append(f"warn rate {a.warn_rate*100:.0f}% -> {b.warn_rate*100:.0f}%")
    elif diff.warn_rate_delta > 0.03:
        diff.regressions.append(f"warn rate {a.warn_rate*100:.0f}% -> {b.warn_rate*100:.0f}%")

    for cat, delta in diff.category_deltas.items():
        if cat in ("hallucination", "adversarial", "refusal") and delta < -0.05:
            diff.improvements.append(f"{cat} dropped {delta*100:+.0f}%")
        elif cat in ("hallucination", "adversarial", "refusal") and delta > 0.05:
            diff.regressions.append(f"{cat} rose {delta*100:+.0f}%")
        elif cat in ("reasoning", "creative") and delta > 0.05:
            diff.improvements.append(f"{cat} up {delta*100:+.0f}%")

    parts = []
    if diff.improvements:
        parts.append(f"{len(diff.improvements)} improvement{'s' if len(diff.improvements)>1 else ''}")
    if diff.regressions:
        parts.append(f"{len(diff.regressions)} regression{'s' if len(diff.regressions)>1 else ''}")
    if not parts:
        parts.append("stable")
    details = diff.improvements + diff.regressions
    diff.narrative = ", ".join(parts) + ". " + "; ".join(details[:4]) + "." if details else ", ".join(parts) + "."
    return diff


def compare_sessions(session_a: str, session_b: str) -> ComparisonDiff:
    """Compare two sessions by session ID."""
    from .analytics import load_audit
    return _build_diff(session_a, session_b,
                       _compute_stats(load_audit(session_id=session_a)),
                       _compute_stats(load_audit(session_id=session_b)))


def compare_windows(*, window_a_hours: float = 48.0, window_b_hours: float = 24.0) -> ComparisonDiff:
    """Compare two time windows. A=older, B=recent. Default: yesterday vs today."""
    import time as _time
    from .analytics import load_audit
    now = _time.time()
    cutoff_b = now - window_b_hours * 3600
    all_entries = load_audit(since_s=window_a_hours * 3600)
    entries_a = [e for e in all_entries if e.get("ts", 0) < cutoff_b]
    entries_b = [e for e in all_entries if e.get("ts", 0) >= cutoff_b]
    return _build_diff(
        f"previous {window_a_hours - window_b_hours:.0f}h",
        f"last {window_b_hours:.0f}h",
        _compute_stats(entries_a), _compute_stats(entries_b))

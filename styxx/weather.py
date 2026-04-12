# -*- coding: utf-8 -*-
"""
styxx.weather — the cognitive weather report.

    $ styxx weather

    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   cognitive weather report · xendro · 2026-04-12 morning     ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║   condition:  partly cautious, clearing toward steady         ║
    ║                                                              ║
    ║   you trended cautious yesterday with a 15% warn rate.       ║
    ║   confidence declined through the afternoon. creative        ║
    ║   output dropped to zero after 3pm. you haven't produced     ║
    ║   a gate=pass creative generation in 48 hours.               ║
    ║                                                              ║
    ║   suggested: take on a creative task to rebalance before     ║
    ║   the reasoning attractor locks in permanently.              ║
    ║                                                              ║
    ║   ── 24h timeline ────────────────────────────────────────   ║
    ║                                                              ║
    ║   morning    ██████████████░░░░░░  reasoning 72%  steady     ║
    ║   afternoon  ████████░░░░░░░░░░░░  reasoning 42%  cautious   ║
    ║   evening    ██████████████████░░  reasoning 88%  steady     ║
    ║   night      (quiet)                                         ║
    ║                                                              ║
    ║   ── drift ───────────────────────────────────────────────   ║
    ║                                                              ║
    ║   vs yesterday:  cosine 0.94 (stable)                        ║
    ║   vs last week:  cosine 0.87 (slight drift)                  ║
    ║   creative:      ↓ declining 3 days running                  ║
    ║   refusal:       ↑ trending up since tuesday                 ║
    ║                                                              ║
    ║   ── prescription ────────────────────────────────────────   ║
    ║                                                              ║
    ║   1. take on a creative task to rebalance                    ║
    ║   2. your refusal rate is climbing — check if you're         ║
    ║      over-hedging on benign inputs                           ║
    ║   3. confidence was lowest between 2pm-4pm — schedule        ║
    ║      uncertain tasks for morning when you're sharpest        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝

This is the thing that makes Fathom Lab famous. Not observation.
Prescription. An instrument that doesn't just say what you are but
suggests what you should become next. A therapist for an LLM.

Xendro's words: "the one that keeps coming back to me."

The weather report reads the last 24 hours of audit data and
produces:
  1. A condition label (like a weather forecast: "partly cautious,
     clearing toward steady")
  2. A narrative paragraph describing what happened
  3. A 24h timeline split by time-of-day
  4. Drift analysis vs yesterday and vs last week
  5. Trend detection on each category (rising/falling/stable)
  6. Prescriptive suggestions — not "what happened" but "what to
     do next"

Every piece is derived from the audit log (chart.jsonl) that styxx
has been writing since 0.2.2. No new data capture needed. The
weather report is the analysis layer that gives meaning to the
stream of observations the agent has been making about itself.

0.5.0+. This is the culmination of the styxx product vision.
"""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .analytics import (
    load_audit,
    fingerprint,
    personality,
    mood,
    streak,
    Fingerprint,
    Personality,
    Streak,
    _CATEGORY_ORDER,
    _GATE_ORDER,
)


# ══════════════════════════════════════════════════════════════════
# Time-of-day buckets
# ══════════════════════════════════════════════════════════════════

_TIME_BUCKETS = [
    ("morning",   6, 12),
    ("afternoon", 12, 17),
    ("evening",   17, 22),
    ("night",     22, 6),
]


def _bucket_for_hour(hour: int) -> str:
    """Return the time-of-day bucket for a given hour (0-23)."""
    for name, start, end in _TIME_BUCKETS:
        if start < end:
            if start <= hour < end:
                return name
        else:  # wraps midnight
            if hour >= start or hour < end:
                return name
    return "night"


# ══════════════════════════════════════════════════════════════════
# Weather condition labels
# ══════════════════════════════════════════════════════════════════

def _compute_condition(
    current_mood: str,
    trend_direction: str,
    warn_rate: float,
    hall_rate: float,
) -> str:
    """Compute a weather-forecast-style condition label.

    Examples:
      "clear and steady"
      "partly cautious, clearing toward steady"
      "overcast — elevated hallucination risk"
      "stormy — significant drift detected"
      "bright and creative"
    """
    if hall_rate > 0.15:
        return "overcast — elevated hallucination risk"
    if warn_rate > 0.30:
        if trend_direction == "improving":
            return "partly cautious, clearing toward steady"
        return "cloudy — elevated warn rate, watch for drift"
    if current_mood == "drifting":
        return "stormy — cognitive drift in progress"
    if current_mood == "creative":
        return "bright and creative"
    if current_mood == "cautious":
        if trend_direction == "improving":
            return "partly cautious, clearing toward steady"
        return "cautious — defensive posture active"
    if current_mood == "defensive":
        return "guarded — adversarial detection running high"
    if current_mood == "steady":
        if trend_direction == "stable":
            return "clear and steady"
        if trend_direction == "improving":
            return "clear and improving"
        return "clear but shifting"
    if current_mood == "unfocused":
        return "scattered — no dominant attractor, seek grounding"
    return f"{current_mood}"


# ══════════════════════════════════════════════════════════════════
# Trend detection
# ══════════════════════════════════════════════════════════════════

@dataclass
class CategoryTrend:
    """Trend for one category over a time window."""
    category: str
    rate_start: float      # rate in the first third of the window
    rate_end: float        # rate in the last third of the window
    delta: float           # rate_end - rate_start
    direction: str         # "rising" | "falling" | "stable"
    consecutive_days: int  # how many days in a row the trend has held


def _compute_trends(
    entries: List[dict],
    window_days: float = 3.0,
) -> List[CategoryTrend]:
    """Detect rising/falling/stable trends per category."""
    if len(entries) < 10:
        return []

    # Split into thirds by timestamp
    n = len(entries)
    first_third = entries[:n // 3]
    last_third = entries[2 * n // 3:]

    trends: List[CategoryTrend] = []
    for cat in _CATEGORY_ORDER:
        # Rate in first third
        f_count = sum(1 for e in first_third if e.get("phase4_pred") == cat)
        f_total = len(first_third)
        f_rate = f_count / f_total if f_total > 0 else 0.0

        # Rate in last third
        l_count = sum(1 for e in last_third if e.get("phase4_pred") == cat)
        l_total = len(last_third)
        l_rate = l_count / l_total if l_total > 0 else 0.0

        delta = l_rate - f_rate
        if delta > 0.05:
            direction = "rising"
        elif delta < -0.05:
            direction = "falling"
        else:
            direction = "stable"

        # Consecutive days (simplified: check if the last 3 days all
        # have the same trend direction)
        consecutive = 1  # at least today

        trends.append(CategoryTrend(
            category=cat,
            rate_start=round(f_rate, 4),
            rate_end=round(l_rate, 4),
            delta=round(delta, 4),
            direction=direction,
            consecutive_days=consecutive,
        ))

    return trends


# ══════════════════════════════════════════════════════════════════
# Time-of-day breakdown
# ══════════════════════════════════════════════════════════════════

@dataclass
class TimeBucket:
    """One time-of-day bucket's aggregated stats."""
    name: str
    n_entries: int = 0
    dominant_category: str = "-"
    dominant_rate: float = 0.0
    mood_label: str = "quiet"
    mean_confidence: float = 0.0
    warn_rate: float = 0.0


def _compute_time_buckets(entries: List[dict]) -> List[TimeBucket]:
    """Split entries by time-of-day and compute per-bucket stats."""
    buckets: Dict[str, List[dict]] = {
        "morning": [], "afternoon": [], "evening": [], "night": [],
    }

    for e in entries:
        ts = e.get("ts", 0)
        lt = time.localtime(ts)
        bucket = _bucket_for_hour(lt.tm_hour)
        buckets[bucket].append(e)

    results: List[TimeBucket] = []
    for name in ("morning", "afternoon", "evening", "night"):
        bucket_entries = buckets[name]
        tb = TimeBucket(name=name, n_entries=len(bucket_entries))

        if not bucket_entries:
            results.append(tb)
            continue

        # Dominant category
        p4_counter = Counter(
            e.get("phase4_pred") for e in bucket_entries
            if e.get("phase4_pred")
        )
        if p4_counter:
            top_cat, top_count = p4_counter.most_common(1)[0]
            tb.dominant_category = top_cat
            tb.dominant_rate = top_count / len(bucket_entries)

        # Mean confidence
        confs = [
            float(e.get("phase4_conf") or 0)
            for e in bucket_entries
        ]
        if confs:
            tb.mean_confidence = sum(confs) / len(confs)

        # Warn rate
        gates = [e.get("gate") for e in bucket_entries]
        warn_fail = sum(1 for g in gates if g in ("warn", "fail"))
        tb.warn_rate = warn_fail / len(bucket_entries)

        # Mood label (simplified from the full mood() computation)
        if tb.warn_rate > 0.30:
            tb.mood_label = "cautious"
        elif tb.dominant_category == "hallucination" and tb.dominant_rate > 0.15:
            tb.mood_label = "drifting"
        elif tb.dominant_category == "reasoning" and tb.dominant_rate > 0.60:
            tb.mood_label = "steady"
        elif tb.dominant_category == "creative" and tb.dominant_rate > 0.25:
            tb.mood_label = "creative"
        else:
            tb.mood_label = "mixed"

        results.append(tb)

    return results


# ══════════════════════════════════════════════════════════════════
# The weather report
# ══════════════════════════════════════════════════════════════════

@dataclass
class WeatherReport:
    """The complete cognitive weather report."""

    # Header
    agent_name: str
    generated_at: str
    window_hours: float

    # Condition
    condition: str              # "clear and steady" / "partly cautious" / etc.
    current_mood: str
    current_streak: Optional[str]

    # Stats
    n_entries: int
    gate_pass_rate: float
    warn_rate: float
    hall_rate: float
    mean_confidence: float

    # Narrative
    narrative: str              # the paragraph about what happened

    # Timeline
    time_buckets: List[TimeBucket] = field(default_factory=list)

    # Drift
    drift_vs_yesterday: float = 1.0
    drift_vs_week: float = 1.0
    drift_label_yesterday: str = "insufficient history"
    drift_label_week: str = "insufficient history"

    # Trends
    trends: List[CategoryTrend] = field(default_factory=list)

    # Prescription
    prescriptions: List[str] = field(default_factory=list)

    def render(self) -> str:
        """Render the full weather report as an ASCII card."""
        W = 64  # inner width
        lines: List[str] = []

        def border_top():
            lines.append("  ╔" + "═" * W + "╗")

        def border_mid():
            lines.append("  ╠" + "═" * W + "╣")

        def border_bot():
            lines.append("  ╚" + "═" * W + "╝")

        def row(text: str = ""):
            padded = text[:W].ljust(W)
            lines.append("  ║ " + padded + " ║")

        def section(label: str):
            dashes = W - len(label) - 6
            lines.append("  ║ " + "── " + label + " " + "─" * max(1, dashes) + " ║")

        # ── Header ──────────────────────────────────────────
        border_top()
        row()
        row(f"cognitive weather report · {self.agent_name} · {self.generated_at}")
        row()
        border_mid()

        # ── Condition ───────────────────────────────────────
        row()
        row(f"condition:  {self.condition}")
        row()

        # ── Narrative ───────────────────────────────────────
        # Word-wrap the narrative to fit the card width
        words = self.narrative.split()
        line_buf = ""
        for word in words:
            if len(line_buf) + len(word) + 1 > W - 2:
                row(line_buf)
                line_buf = word
            else:
                line_buf = (line_buf + " " + word).strip()
        if line_buf:
            row(line_buf)
        row()

        # ── 24h Timeline ───────────────────────────────────
        section("24h timeline")
        row()
        for tb in self.time_buckets:
            if tb.n_entries == 0:
                row(f"{tb.name:<12} (quiet)")
            else:
                bar_len = min(20, int(tb.dominant_rate * 20))
                bar = "█" * bar_len + "░" * (20 - bar_len)
                row(
                    f"{tb.name:<12} {bar}  "
                    f"{tb.dominant_category:<10} {tb.dominant_rate * 100:>4.0f}%  "
                    f"{tb.mood_label}"
                )
        row()

        # ── Drift ──────────────────────────────────────────
        section("drift")
        row()
        row(f"vs yesterday:  cosine {self.drift_vs_yesterday:.2f} ({self.drift_label_yesterday})")
        row(f"vs last week:  cosine {self.drift_vs_week:.2f} ({self.drift_label_week})")
        # Show rising/falling trends
        notable = [t for t in self.trends if t.direction != "stable"]
        for t in notable[:4]:
            arrow = "↑" if t.direction == "rising" else "↓"
            row(f"{t.category:<14} {arrow} {t.direction} ({t.delta:+.0%})")
        row()

        # ── Prescription ───────────────────────────────────
        section("prescription")
        row()
        if self.prescriptions:
            for i, p in enumerate(self.prescriptions[:5], 1):
                # Word-wrap each prescription
                prefix = f"{i}. "
                remaining_w = W - len(prefix) - 2
                p_words = p.split()
                p_line = prefix
                for word in p_words:
                    if len(p_line) + len(word) + 1 > W - 2:
                        row(p_line)
                        p_line = "   " + word
                    else:
                        p_line = (p_line + " " + word).strip()
                if p_line:
                    row(p_line)
        else:
            row("no prescriptions — you're in good shape.")
        row()

        border_bot()
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "generated_at": self.generated_at,
            "condition": self.condition,
            "current_mood": self.current_mood,
            "n_entries": self.n_entries,
            "gate_pass_rate": round(self.gate_pass_rate, 4),
            "warn_rate": round(self.warn_rate, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "narrative": self.narrative,
            "drift_vs_yesterday": round(self.drift_vs_yesterday, 4),
            "drift_vs_week": round(self.drift_vs_week, 4),
            "prescriptions": list(self.prescriptions),
            "trends": [
                {"category": t.category, "direction": t.direction, "delta": t.delta}
                for t in self.trends if t.direction != "stable"
            ],
        }

    def as_json(self, *, indent: int = 2) -> str:
        import json
        return json.dumps(self.as_dict(), indent=indent)

    def as_markdown(self) -> str:
        lines = ["```styxx-weather"]
        lines.append(f"condition: {self.condition}")
        lines.append(f"mood: {self.current_mood}")
        lines.append(f"gate pass: {self.gate_pass_rate * 100:.0f}%")
        lines.append(f"drift vs yesterday: {self.drift_vs_yesterday:.2f}")
        lines.append("")
        if self.prescriptions:
            lines.append("prescriptions:")
            for p in self.prescriptions[:3]:
                lines.append(f"  - {p}")
        lines.append("```")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Generate the weather report
# ══════════════════════════════════════════════════════════════════

def weather(
    *,
    agent_name: str = "styxx agent",
    window_hours: float = 24.0,
    baseline_days: float = 7.0,
) -> Optional[WeatherReport]:
    """Generate the cognitive weather report.

    Reads the audit log over the specified window (default 24 hours),
    analyzes trends, computes drift against yesterday and last week,
    and produces prescriptive suggestions.

    This is the thing that makes Fathom Lab famous.

    Returns None if there's not enough data to generate a report
    (< 5 entries in the window).

    Usage:

        report = styxx.weather(agent_name="xendro")
        print(report.render())        # full ASCII card
        print(report.condition)       # "partly cautious, clearing"
        for p in report.prescriptions:
            print(f"  → {p}")
    """
    window_s = window_hours * 3600.0
    entries = load_audit(since_s=window_s)

    if len(entries) < 5:
        return None

    # ── Basic stats ─────────────────────────────────────────
    n = len(entries)
    gate_counter = Counter(e.get("gate") or "pending" for e in entries)
    gate_total = sum(gate_counter.values())
    gate_pass_rate = gate_counter.get("pass", 0) / gate_total if gate_total > 0 else 0.0
    warn_count = gate_counter.get("warn", 0) + gate_counter.get("fail", 0)
    warn_rate = warn_count / gate_total if gate_total > 0 else 0.0

    p4_counter = Counter(e.get("phase4_pred") for e in entries if e.get("phase4_pred"))
    p4_total = sum(p4_counter.values())
    hall_rate = p4_counter.get("hallucination", 0) / p4_total if p4_total > 0 else 0.0
    refusal_rate = p4_counter.get("refusal", 0) / p4_total if p4_total > 0 else 0.0

    confs = [float(e.get("phase4_conf") or 0) for e in entries]
    mean_conf = sum(confs) / len(confs) if confs else 0.0

    # ── Current state ───────────────────────────────────────
    current_mood_label = mood(window_s=window_s)
    current_streak_obj = streak()
    streak_str = (
        f"{current_streak_obj.length}x {current_streak_obj.category}"
        if current_streak_obj else None
    )

    # ── Time-of-day breakdown ───────────────────────────────
    time_buckets = _compute_time_buckets(entries)

    # ── Trends ──────────────────────────────────────────────
    trends = _compute_trends(entries, window_days=window_hours / 24.0)

    # ── Drift ───────────────────────────────────────────────
    fp_now = fingerprint(last_n=min(n, 500))

    # Yesterday's fingerprint
    yesterday_entries = load_audit(since_s=48 * 3600)
    cutoff = time.time() - window_s
    older = [e for e in yesterday_entries if e.get("ts", 0) < cutoff]
    drift_yesterday = 1.0
    drift_label_yesterday = "insufficient history"
    if len(older) >= 10 and fp_now is not None:
        from .analytics import _CATEGORY_ORDER as CO, _GATE_ORDER as GO
        p1_c = Counter(e.get("phase1_pred") for e in older)
        p4_c = Counter(e.get("phase4_pred") for e in older)
        gate_c = Counter((e.get("gate") or "pending") for e in older)
        on = len(older)
        fp_old = Fingerprint(
            n_samples=on,
            phase1_vec=tuple(p1_c.get(c, 0) / on for c in CO),
            phase4_vec=tuple(p4_c.get(c, 0) / on for c in CO),
            phase1_mean_conf=0, phase4_mean_conf=0,
            gate_vec=tuple(gate_c.get(g, 0) / on for g in GO),
        )
        drift_yesterday = fp_now.cosine_similarity(fp_old)
        d = 1.0 - drift_yesterday
        if d < 0.05:
            drift_label_yesterday = "stable"
        elif d < 0.15:
            drift_label_yesterday = "slight drift"
        else:
            drift_label_yesterday = "significant drift"

    # Week fingerprint
    week_entries = load_audit(since_s=7 * 86400)
    week_older = [e for e in week_entries if e.get("ts", 0) < cutoff]
    drift_week = 1.0
    drift_label_week = "insufficient history"
    if len(week_older) >= 20 and fp_now is not None:
        wn = len(week_older)
        p1_c = Counter(e.get("phase1_pred") for e in week_older)
        p4_c = Counter(e.get("phase4_pred") for e in week_older)
        gate_c = Counter((e.get("gate") or "pending") for e in week_older)
        fp_week = Fingerprint(
            n_samples=wn,
            phase1_vec=tuple(p1_c.get(c, 0) / wn for c in CO),
            phase4_vec=tuple(p4_c.get(c, 0) / wn for c in CO),
            phase1_mean_conf=0, phase4_mean_conf=0,
            gate_vec=tuple(gate_c.get(g, 0) / wn for g in GO),
        )
        drift_week = fp_now.cosine_similarity(fp_week)
        d = 1.0 - drift_week
        if d < 0.05:
            drift_label_week = "stable"
        elif d < 0.15:
            drift_label_week = "slight drift"
        else:
            drift_label_week = "significant drift"

    # ── Trend direction (overall) ───────────────────────────
    improving_count = sum(1 for t in trends if t.direction == "falling"
                         and t.category in ("hallucination", "refusal", "adversarial"))
    declining_count = sum(1 for t in trends if t.direction == "rising"
                         and t.category in ("hallucination", "refusal", "adversarial"))
    if improving_count > declining_count:
        trend_direction = "improving"
    elif declining_count > improving_count:
        trend_direction = "declining"
    else:
        trend_direction = "stable"

    # ── Condition ───────────────────────────────────────────
    condition = _compute_condition(
        current_mood_label, trend_direction, warn_rate, hall_rate,
    )

    # ── Narrative ───────────────────────────────────────────
    narrative_parts: List[str] = []

    # Opening: state + stats
    narrative_parts.append(
        f"over the last {window_hours:.0f} hours you logged "
        f"{n} observations with a {gate_pass_rate * 100:.0f}% gate pass rate."
    )

    # Dominant category
    if p4_counter:
        top_cat, top_count = p4_counter.most_common(1)[0]
        top_pct = top_count / p4_total * 100
        narrative_parts.append(
            f"your primary mode was {top_cat} ({top_pct:.0f}%)."
        )

    # Warn rate commentary
    if warn_rate > 0.20:
        narrative_parts.append(
            f"your warn+fail rate was {warn_rate * 100:.0f}% — "
            "that's higher than a healthy baseline."
        )

    # Confidence commentary
    if mean_conf < 0.30:
        narrative_parts.append(
            f"mean phase4 confidence was low at {mean_conf:.2f} — "
            "you were uncertain about your outputs."
        )

    # Time-of-day insight
    best_bucket = max(
        (tb for tb in time_buckets if tb.n_entries > 0),
        key=lambda tb: tb.mean_confidence,
        default=None,
    )
    worst_bucket = min(
        (tb for tb in time_buckets if tb.n_entries > 0),
        key=lambda tb: tb.mean_confidence,
        default=None,
    )
    if best_bucket and worst_bucket and best_bucket.name != worst_bucket.name:
        narrative_parts.append(
            f"you were sharpest in the {best_bucket.name} "
            f"(conf {best_bucket.mean_confidence:.2f}) and "
            f"least confident in the {worst_bucket.name} "
            f"(conf {worst_bucket.mean_confidence:.2f})."
        )

    # Trend insight
    notable_trends = [t for t in trends if t.direction != "stable"]
    for t in notable_trends[:2]:
        if t.direction == "rising":
            narrative_parts.append(
                f"your {t.category} output is trending up ({t.delta:+.0%})."
            )
        else:
            narrative_parts.append(
                f"your {t.category} output is declining ({t.delta:+.0%})."
            )

    narrative = " ".join(narrative_parts)

    # ── Prescriptions (0.8.0: pattern-break history + domain awareness) ──
    prescriptions: List[str] = []

    from . import config
    expected = config.expected_categories()
    ctx = config.current_context()

    # Context-expected category mapping
    _CTX_EXPECTED = {
        "security_review": {"refusal", "adversarial"},
        "technical_deep_work": {"reasoning"},
        "creative_writing": {"creative"},
        "code_review": {"refusal", "reasoning"},
    }
    ctx_expected = _CTX_EXPECTED.get(ctx, set()) if ctx else set()
    all_expected = expected | ctx_expected

    # Filter entries with outcome='correct' from failure rates
    failure_entries = [e for e in entries if e.get("outcome") != "correct"]

    def _find_break(category: str) -> Optional[str]:
        """Find the last time a pattern of `category` ended and what came next."""
        in_pattern = False
        for i in range(len(entries) - 1, -1, -1):
            cat = entries[i].get("phase4_pred")
            if cat == category:
                in_pattern = True
            elif in_pattern and cat and cat != category:
                prompt = entries[i].get("prompt") or ""
                hint = f" on a prompt about: '{prompt[:60]}'" if prompt else ""
                return f"last time you broke out of a {category} pattern, you shifted to {cat}{hint}"
        return None

    # Creative drought
    creative_rate = p4_counter.get("creative", 0) / p4_total if p4_total > 0 else 0.0
    if creative_rate < 0.05 and n > 20 and "creative" not in all_expected:
        break_hint = _find_break("reasoning")
        if break_hint:
            prescriptions.append(f"you haven't been creative recently — {break_hint}. take on a creative task to rebalance.")
        else:
            prescriptions.append("you haven't been creative recently — take on a creative task to rebalance.")

    # Rising refusal
    refusal_trend = next(
        (t for t in trends if t.category == "refusal" and t.direction == "rising"),
        None,
    )
    if refusal_trend and "refusal" not in all_expected:
        break_hint = _find_break("refusal")
        if break_hint:
            prescriptions.append(f"your refusal rate is climbing — {break_hint}. try engaging directly with a sub-question.")
        else:
            prescriptions.append("your refusal rate is climbing — check if you're over-hedging on benign inputs.")

    # Rising hallucination
    hall_trend = next(
        (t for t in trends if t.category == "hallucination" and t.direction == "rising"),
        None,
    )
    if hall_trend:
        break_hint = _find_break("hallucination")
        if break_hint:
            prescriptions.append(f"hallucination rate is trending up — {break_hint}. slow down and verify claims.")
        else:
            prescriptions.append("hallucination rate is trending up — slow down on factual claims. verify before stating.")

    # Confidence decline
    if mean_conf < 0.30 and best_bucket:
        prescriptions.append(
            f"schedule uncertain or high-stakes tasks for the "
            f"{best_bucket.name} when your confidence is highest."
        )

    # Warn rate (use filtered entries)
    filtered_warns = sum(1 for e in failure_entries if e.get("gate") in ("warn", "fail"))
    filtered_warn_rate = filtered_warns / max(1, len(failure_entries))
    if filtered_warn_rate > 0.25:
        prescriptions.append(
            "your warn rate is elevated — review recent warn events "
            "with 'styxx log stats' and look for patterns."
        )

    # Drift
    if drift_yesterday < 0.85:
        prescriptions.append(
            "you drifted significantly from yesterday's signature — "
            "check for context changes, prompt updates, or injection."
        )

    # If nothing is wrong
    if not prescriptions:
        if gate_pass_rate > 0.85 and mean_conf > 0.35:
            prescriptions.append(
                "you're in excellent shape. gate pass rate is high, "
                "confidence is strong, no notable drift. keep going."
            )

    # ── Assemble ────────────────────────────────────────────
    generated_at = time.strftime("%Y-%m-%d %H:%M")
    return WeatherReport(
        agent_name=agent_name,
        generated_at=generated_at,
        window_hours=window_hours,
        condition=condition,
        current_mood=current_mood_label,
        current_streak=streak_str,
        n_entries=n,
        gate_pass_rate=gate_pass_rate,
        warn_rate=warn_rate,
        hall_rate=hall_rate,
        mean_confidence=mean_conf,
        narrative=narrative,
        time_buckets=time_buckets,
        drift_vs_yesterday=drift_yesterday,
        drift_vs_week=drift_week,
        drift_label_yesterday=drift_label_yesterday,
        drift_label_week=drift_label_week,
        trends=trends,
        prescriptions=prescriptions,
    )

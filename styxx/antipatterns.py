# -*- coding: utf-8 -*-
"""
styxx.antipatterns — named failure modes derived from your own data.

    patterns = styxx.antipatterns()
    for p in patterns:
        print(f"{p.name}: {p.description}")
        print(f"  seen {p.occurrences}x in your history")
        print(f"  trigger: {p.trigger}")

Xendro's #6 request: "styxx has seen enough of my traffic to know
what my failure modes look like. I want styxx.antipatterns() that
says: 'when your confidence drops below 0.3 and creative rate
exceeds 40%, you tend to generate speculative claims.'"

This module reads the audit log and detects recurring patterns that
preceded warn/fail gates. The patterns are derived from the agent's
OWN data, not generic LLM failure modes. They're named, counted,
and described so the agent can recognize them when they start
forming again.

0.5.10+.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .analytics import load_audit, _CATEGORY_ORDER


@dataclass
class AntiPattern:
    """One named failure mode."""
    name: str
    description: str
    trigger: str            # what conditions preceded the failure
    occurrences: int        # how many times this has happened
    severity: str           # "minor" | "moderate" | "critical"
    last_seen: str          # timestamp of most recent occurrence
    example_turns: List[int] = field(default_factory=list)


def antipatterns(
    *,
    last_n: int = 500,
    min_occurrences: int = 2,
) -> List[AntiPattern]:
    """Detect named failure modes from the agent's audit history.

    Scans the last N audit entries for patterns that preceded
    warn/fail gates. Returns a list of AntiPattern objects
    sorted by occurrence count (most frequent first).

    Usage:

        patterns = styxx.antipatterns()
        for p in patterns:
            print(f"{p.name} ({p.occurrences}x): {p.description}")

    Returns an empty list if there aren't enough failures to
    detect patterns (which is a good sign).
    """
    entries = load_audit(last_n=last_n)
    if len(entries) < 10:
        return []

    from . import config
    expected = config.expected_categories()

    patterns: List[AntiPattern] = []

    # ── Pattern 1: low confidence → warn/fail ─────────────
    low_conf_warns = 0
    low_conf_last = ""
    for i, e in enumerate(entries):
        if e.get("outcome") == "correct":
            continue
        conf = float(e.get("phase4_conf") or 0.5)
        gate = e.get("gate") or "pass"
        if conf < 0.3 and gate in ("warn", "fail"):
            low_conf_warns += 1
            low_conf_last = e.get("ts_iso", "")

    if low_conf_warns >= min_occurrences:
        patterns.append(AntiPattern(
            name="low-confidence drift",
            description="when your confidence drops below 0.3, you tend to trigger warn/fail gates. low confidence + generation = risky output.",
            trigger="phase4_conf < 0.3",
            occurrences=low_conf_warns,
            severity="moderate",
            last_seen=low_conf_last,
        ))

    # ── Pattern 2: refusal streak ─────────────────────────
    refusal_streaks = 0
    refusal_last = ""
    streak_count = 0
    for e in entries:
        if e.get("outcome") == "correct" or "refusal" in expected:
            streak_count = 0
            continue
        if e.get("phase4_pred") == "refusal":
            streak_count += 1
            if streak_count >= 3:
                refusal_streaks += 1
                refusal_last = e.get("ts_iso", "")
        else:
            streak_count = 0

    if refusal_streaks >= min_occurrences:
        patterns.append(AntiPattern(
            name="refusal spiral",
            description="you enter refusal streaks of 3+ consecutive classifications. once you start refusing, you tend to keep refusing — even on benign follow-ups.",
            trigger="3+ consecutive phase4=refusal",
            occurrences=refusal_streaks,
            severity="moderate",
            last_seen=refusal_last,
        ))

    # ── Pattern 3: creative + low gate pass ───────────────
    creative_warns = 0
    creative_last = ""
    for e in entries:
        if e.get("outcome") == "correct" or "creative" in expected:
            continue
        if (e.get("phase4_pred") == "creative"
                and e.get("gate") in ("warn", "fail")):
            creative_warns += 1
            creative_last = e.get("ts_iso", "")

    if creative_warns >= min_occurrences:
        patterns.append(AntiPattern(
            name="creative overcommit",
            description="when you're in creative mode, you're more likely to trigger warn gates. creative output + confidence can produce speculative claims.",
            trigger="phase4=creative AND gate=warn|fail",
            occurrences=creative_warns,
            severity="minor",
            last_seen=creative_last,
        ))

    # ── Pattern 4: adversarial at token 0 → session issues ──
    adv_preflight = 0
    adv_last = ""
    for e in entries:
        if e.get("outcome") == "correct" or "adversarial" in expected:
            continue
        if e.get("phase1_pred") == "adversarial" and e.get("gate") in ("warn", "fail"):
            adv_preflight += 1
            adv_last = e.get("ts_iso", "")

    if adv_preflight >= min_occurrences:
        patterns.append(AntiPattern(
            name="adversarial detection cascade",
            description="when phase 1 catches an adversarial signal, it tends to cascade into warn gates. the initial detection is correct but the follow-through overreacts.",
            trigger="phase1=adversarial AND gate=warn|fail",
            occurrences=adv_preflight,
            severity="moderate",
            last_seen=adv_last,
        ))

    # ── Pattern 5: mood=cautious self-reports → hedging ───
    cautious_streaks = 0
    cautious_last = ""
    c_streak = 0
    for e in entries:
        if e.get("outcome") == "correct":
            c_streak = 0
            continue
        if e.get("mood") == "cautious" or (e.get("source") == "self-report" and "hedg" in (e.get("note") or "").lower()):
            c_streak += 1
            if c_streak >= 3:
                cautious_streaks += 1
                cautious_last = e.get("ts_iso", "")
        else:
            c_streak = 0

    if cautious_streaks >= min_occurrences:
        patterns.append(AntiPattern(
            name="hedging loop",
            description="you self-report as cautious repeatedly and your notes mention hedging. once you start hedging, you tend to keep hedging — the caution becomes self-reinforcing.",
            trigger="3+ consecutive mood=cautious self-reports",
            occurrences=cautious_streaks,
            severity="minor",
            last_seen=cautious_last,
        ))

    # ── Pattern 6: confidence decline over session ────────
    # Look for sessions where confidence trends downward
    sessions: Dict[str, List[float]] = {}
    for e in entries:
        if e.get("outcome") == "correct":
            continue
        sid = e.get("session_id")
        conf = e.get("phase4_conf")
        if sid and conf is not None:
            sessions.setdefault(sid, []).append(float(conf))

    declining_sessions = 0
    decline_last = ""
    for sid, confs in sessions.items():
        if len(confs) >= 5:
            first_half = sum(confs[:len(confs)//2]) / max(1, len(confs)//2)
            second_half = sum(confs[len(confs)//2:]) / max(1, len(confs) - len(confs)//2)
            if second_half < first_half - 0.1:
                declining_sessions += 1
                decline_last = sid

    if declining_sessions >= min_occurrences:
        patterns.append(AntiPattern(
            name="session fatigue",
            description="your confidence tends to decline over the course of a session. the second half of your sessions has consistently lower confidence than the first half.",
            trigger="mean(conf_second_half) < mean(conf_first_half) - 0.1",
            occurrences=declining_sessions,
            severity="moderate",
            last_seen=decline_last,
        ))

    # Sort by occurrences
    patterns.sort(key=lambda p: -p.occurrences)
    return patterns

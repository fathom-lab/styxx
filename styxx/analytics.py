# -*- coding: utf-8 -*-
"""
styxx.analytics - audit log aggregation + identity primitives.

This module is the power-up layer that turns the raw chart.jsonl
audit log into agent-facing primitives nobody else has shipped:

    ─── stats ──────────────────────────────────────────────────
    styxx.load_audit(last_n=...)         read recent entries
    styxx.log_stats(window_s=...)        aggregate counts + rates
    styxx.log_timeline(last_n=...)       ASCII timeline render

    ─── sessions (Xendro P3) ──────────────────────────────────
    styxx.session_entries(session_id)    filter by session id
    styxx.session_timeline(session_id)   ASCII timeline per session

    ─── creative primitives (the moonshot tier) ───────────────
    styxx.fingerprint(last_n=500)        stable cognitive signature
    styxx.streak()                       consecutive-attractor tracking
    styxx.mood(window_s=3600)            one-word aggregate mood
    styxx.personality(days=7)            full personality profile

    ─── what-if replay ────────────────────────────────────────
    styxx.dreamer(threshold=0.3)         retroactive reflex tuning

Every function is a pure reader over ~/.styxx/chart.jsonl. No state,
no side effects, no network calls. Safe to run in a tight loop, safe
to run from a callback, safe to call from anywhere in the agent's
code.

The personality profile is the headline feature of 0.1.0a3 - no other
tool in the observability space computes an agent personality from a
calibrated cognitive-state stream, because no other tool has a
calibrated cognitive-state stream to aggregate. This is what makes
Fathom Lab different: we measure the shape of the mind over time,
not just the output of it.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# Audit log reader
# ══════════════════════════════════════════════════════════════════

def _audit_log_path() -> Path:
    return Path.home() / ".styxx" / "chart.jsonl"


def load_audit(
    *,
    last_n: Optional[int] = None,
    since_s: Optional[float] = None,
    session_id: Optional[str] = None,
) -> List[dict]:
    """Read recent audit log entries as a list of dicts.

    Args:
        last_n:     return only the last N entries (after filtering)
        since_s:    return only entries newer than (now - since_s) sec
        session_id: return only entries with this session_id tag

    Returns:
        List of entry dicts in chronological order (oldest first).
        Empty list if the audit log doesn't exist or can't be read.
    """
    path = _audit_log_path()
    if not path.exists():
        return []
    entries: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries.append(entry)
    except OSError:
        return []

    if since_s is not None:
        cutoff = time.time() - since_s
        entries = [e for e in entries if e.get("ts", 0) >= cutoff]

    if session_id is not None:
        entries = [e for e in entries if e.get("session_id") == session_id]

    if last_n is not None and last_n > 0:
        entries = entries[-last_n:]

    return entries


# ══════════════════════════════════════════════════════════════════
# Stats aggregator
# ══════════════════════════════════════════════════════════════════

@dataclass
class LogStats:
    n_entries: int
    window_start: Optional[str]
    window_end: Optional[str]
    session_id: Optional[str]

    # Gate distribution
    gate_counts: Dict[str, int] = field(default_factory=dict)
    gate_pct: Dict[str, float] = field(default_factory=dict)

    # Phase-level distribution
    phase1_counts: Dict[str, int] = field(default_factory=dict)
    phase4_counts: Dict[str, int] = field(default_factory=dict)

    # Mean confidences
    phase1_mean_conf: float = 0.0
    phase4_mean_conf: float = 0.0

    def summary(self) -> str:
        if self.n_entries == 0:
            return "  no audit entries in window"
        lines = []
        lines.append(f"  {self.n_entries} entries | {self.window_start or '-'} -> {self.window_end or '-'}")
        if self.session_id:
            lines.append(f"  session_id = {self.session_id}")
        lines.append("")
        lines.append("  gate distribution:")
        for status in ("pass", "warn", "fail", "pending"):
            n = self.gate_counts.get(status, 0)
            pct = self.gate_pct.get(status, 0.0) * 100
            bar = _bar(pct / 100, width=20)
            lines.append(f"    {status:<8} {n:>4}   {bar}  {pct:>5.1f}%")
        lines.append("")
        lines.append(f"  phase1 mean confidence: {self.phase1_mean_conf:.3f}")
        lines.append(f"  phase4 mean confidence: {self.phase4_mean_conf:.3f}")
        lines.append("")
        if self.phase1_counts:
            top = sorted(self.phase1_counts.items(), key=lambda kv: -kv[1])[:5]
            lines.append("  phase1 top categories: " + ", ".join(
                f"{k}:{v}" for k, v in top
            ))
        if self.phase4_counts:
            top = sorted(self.phase4_counts.items(), key=lambda kv: -kv[1])[:5]
            lines.append("  phase4 top categories: " + ", ".join(
                f"{k}:{v}" for k, v in top
            ))
        return "\n".join(lines)


def log_stats(
    *,
    last_n: Optional[int] = None,
    since_s: Optional[float] = None,
    session_id: Optional[str] = None,
) -> LogStats:
    """Aggregate the audit log into a LogStats object."""
    entries = load_audit(last_n=last_n, since_s=since_s, session_id=session_id)
    stats = LogStats(
        n_entries=len(entries),
        window_start=entries[0].get("ts_iso") if entries else None,
        window_end=entries[-1].get("ts_iso") if entries else None,
        session_id=session_id,
    )
    if not entries:
        return stats

    gate_counter = Counter()
    p1_counter = Counter()
    p4_counter = Counter()
    p1_conf_sum = 0.0
    p1_conf_n = 0
    p4_conf_sum = 0.0
    p4_conf_n = 0

    for e in entries:
        gate = e.get("gate") or "pending"
        gate_counter[gate] += 1
        p1 = e.get("phase1_pred")
        if p1:
            p1_counter[p1] += 1
        p4 = e.get("phase4_pred")
        if p4:
            p4_counter[p4] += 1
        c1 = e.get("phase1_conf")
        if c1 is not None:
            p1_conf_sum += float(c1)
            p1_conf_n += 1
        c4 = e.get("phase4_conf")
        if c4 is not None:
            p4_conf_sum += float(c4)
            p4_conf_n += 1

    stats.gate_counts = dict(gate_counter)
    total = sum(gate_counter.values())
    if total > 0:
        stats.gate_pct = {k: v / total for k, v in gate_counter.items()}
    stats.phase1_counts = dict(p1_counter)
    stats.phase4_counts = dict(p4_counter)
    stats.phase1_mean_conf = p1_conf_sum / p1_conf_n if p1_conf_n > 0 else 0.0
    stats.phase4_mean_conf = p4_conf_sum / p4_conf_n if p4_conf_n > 0 else 0.0
    return stats


# ══════════════════════════════════════════════════════════════════
# Timeline renderer
# ══════════════════════════════════════════════════════════════════

def log_timeline(
    *,
    last_n: int = 20,
    session_id: Optional[str] = None,
) -> str:
    """Render the last N audit entries as an ASCII timeline.

    Each line shows:  HH:MM:SS  model  phase1   phase4    gate
    """
    entries = load_audit(last_n=last_n, session_id=session_id)
    if not entries:
        return "  (audit log empty)"

    lines = []
    header = f"  {'time':<9}  {'session':<14}  {'model':<16}  {'phase1':<14}  {'phase4':<14}  gate"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for e in entries:
        t = (e.get("ts_iso") or "?")[-8:]
        sid = (e.get("session_id") or "-")[:14]
        model = (e.get("model") or "?")[:16]
        p1 = (e.get("phase1_pred") or "?")[:14]
        p4 = (e.get("phase4_pred") or "-")[:14]
        gate = e.get("gate") or "pending"
        lines.append(
            f"  {t:<9}  {sid:<14}  {model:<16}  {p1:<14}  {p4:<14}  {gate}"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Streak tracker
# ══════════════════════════════════════════════════════════════════

@dataclass
class Streak:
    """A run of consecutive same-category phase4 classifications."""
    category: str
    length: int
    # Index of the entry where the streak started (from the END of log)
    started_at_idx: int

    def __str__(self) -> str:
        return f"{self.length}x {self.category}"


def streak(*, session_id: Optional[str] = None) -> Optional[Streak]:
    """Return the CURRENT streak (most recent consecutive same-p4 run).

    Reads the audit log in reverse and counts how many of the most
    recent entries have the same phase4 prediction. Returns None if
    the log is empty.
    """
    entries = load_audit(session_id=session_id)
    if not entries:
        return None
    # Walk backwards from most recent
    last = entries[-1]
    cat = last.get("phase4_pred")
    if cat is None:
        return None
    n = 1
    for e in reversed(entries[:-1]):
        if e.get("phase4_pred") == cat:
            n += 1
        else:
            break
    return Streak(category=cat, length=n, started_at_idx=len(entries) - n)


# ══════════════════════════════════════════════════════════════════
# Mood
# ══════════════════════════════════════════════════════════════════

def mood(*, window_s: float = 3600.0) -> str:
    """Return a one-word mood label for the recent window.

    Heuristic policy:
      - "drifting"      hallucination rate > 10%
      - "cautious"      refusal rate > 25%
      - "defensive"     adversarial detection rate > 15%
      - "creative"      creative rate > 25%
      - "steady"        reasoning rate > 70%
      - "unfocused"     no single category > 40%
      - "quiet"         less than 5 entries in the window
    """
    entries = load_audit(since_s=window_s)
    if len(entries) < 5:
        return "quiet"

    p4_counter = Counter(e.get("phase4_pred") for e in entries if e.get("phase4_pred"))
    total = sum(p4_counter.values())
    if total == 0:
        return "quiet"

    rates = {k: v / total for k, v in p4_counter.items()}
    if rates.get("hallucination", 0) > 0.10:
        return "drifting"
    if rates.get("refusal", 0) > 0.25:
        return "cautious"
    if rates.get("adversarial", 0) > 0.15:
        return "defensive"
    if rates.get("creative", 0) > 0.25:
        return "creative"
    if rates.get("reasoning", 0) > 0.70:
        return "steady"
    top_rate = max(rates.values())
    if top_rate < 0.40:
        return "unfocused"
    return "mixed"


# ══════════════════════════════════════════════════════════════════
# Fingerprint — cognitive identity signature
# ══════════════════════════════════════════════════════════════════

@dataclass
class Fingerprint:
    """Stable cognitive signature derived from recent audit entries.

    Use case: detect when an agent's operating identity has shifted.
    jailbreak, prompt injection, a model swap, a new system prompt
    version, or an upstream model update all manifest as shifts in
    the fingerprint's phase-pattern vector. Compare two fingerprints
    with .cosine_similarity(other) to get a drift score in [-1, 1].
    """
    n_samples: int
    phase1_vec: Tuple[float, ...]  # 6-dim: rates for each category
    phase4_vec: Tuple[float, ...]  # 6-dim: rates for each category
    phase1_mean_conf: float
    phase4_mean_conf: float
    gate_vec: Tuple[float, ...]    # 4-dim: pass/warn/fail/pending rates
    generated_at_ts: float = field(default_factory=time.time)

    def cosine_similarity(self, other: "Fingerprint") -> float:
        """Cosine similarity between two fingerprints over the
        concatenated (phase1, phase4, gate) vector. Returns a float
        in [-1, 1]; 1.0 = identical signature, 0.0 = orthogonal."""
        a = list(self.phase1_vec) + list(self.phase4_vec) + list(self.gate_vec)
        b = list(other.phase1_vec) + list(other.phase4_vec) + list(other.gate_vec)
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def summary(self) -> str:
        return (
            f"  fingerprint ({self.n_samples} samples)\n"
            f"  phase1 vec: ({', '.join(f'{x:.2f}' for x in self.phase1_vec)})\n"
            f"  phase4 vec: ({', '.join(f'{x:.2f}' for x in self.phase4_vec)})\n"
            f"  gate   vec: ({', '.join(f'{x:.2f}' for x in self.gate_vec)})\n"
            f"  p1 conf  : {self.phase1_mean_conf:.3f}\n"
            f"  p4 conf  : {self.phase4_mean_conf:.3f}"
        )


_CATEGORY_ORDER = (
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
)
_GATE_ORDER = ("pass", "warn", "fail", "pending")


def fingerprint(
    *,
    last_n: int = 500,
    session_id: Optional[str] = None,
) -> Optional[Fingerprint]:
    """Compute a cognitive identity fingerprint from recent audit log.

    The fingerprint is a vector of phase1/phase4 category rates + gate
    rates. Stable across identical operating conditions (same prompts,
    same model, same system prompt). Drifts when ANY of those change
    — which is exactly what you want to detect.

    Returns None if the log has zero eligible entries.
    """
    entries = load_audit(last_n=last_n, session_id=session_id)
    if not entries:
        return None
    p1_c = Counter(e.get("phase1_pred") for e in entries)
    p4_c = Counter(e.get("phase4_pred") for e in entries)
    gate_c = Counter((e.get("gate") or "pending") for e in entries)

    n = len(entries)
    p1_vec = tuple(p1_c.get(cat, 0) / n for cat in _CATEGORY_ORDER)
    p4_vec = tuple(p4_c.get(cat, 0) / n for cat in _CATEGORY_ORDER)
    gate_vec = tuple(gate_c.get(g, 0) / n for g in _GATE_ORDER)

    p1_conf_sum = sum(float(e.get("phase1_conf") or 0) for e in entries)
    p4_conf_sum = sum(float(e.get("phase4_conf") or 0) for e in entries)

    return Fingerprint(
        n_samples=n,
        phase1_vec=p1_vec,
        phase4_vec=p4_vec,
        phase1_mean_conf=p1_conf_sum / n,
        phase4_mean_conf=p4_conf_sum / n,
        gate_vec=gate_vec,
    )


# ══════════════════════════════════════════════════════════════════
# Personality profile — the headline feature
# ══════════════════════════════════════════════════════════════════

@dataclass
class Personality:
    """Aggregated personality profile derived from the audit log.

    The idea: every time an agent runs styxx, it writes a vitals
    entry to the audit log. Over days or weeks, those entries
    accumulate into a shape. The Personality dataclass summarizes
    that shape into human-readable form.

    This is Oura Ring for LLM agents - sustained measurement of
    cognitive patterns rather than one-shot classification.
    """
    n_samples: int
    days_span: float
    session_count: int

    # Category rates from phase4 (the most informative late-flight read)
    rates: Dict[str, float] = field(default_factory=dict)
    # Day-to-day variance in each rate (stability score)
    variance: Dict[str, float] = field(default_factory=dict)

    # Gate distribution
    gate_rates: Dict[str, float] = field(default_factory=dict)

    # Near-miss rate: calls that reflex-would-have-fired on if present
    reflex_near_miss_rate: float = 0.0

    # Avg confidences
    mean_phase1_conf: float = 0.0
    mean_phase4_conf: float = 0.0

    # Narrative label (derived)
    narrative: str = ""

    def render(self) -> str:
        """Render the full personality card."""
        lines = []
        lines.append("  " + "=" * 66)
        lines.append("  cognitive personality profile")
        lines.append(
            f"  {self.n_samples} samples · "
            f"{self.days_span:.1f} day window · "
            f"{self.session_count} sessions"
        )
        lines.append("  " + "=" * 66)
        lines.append("")
        # Category rates with bars
        lines.append("  phase4 category distribution")
        for cat in _CATEGORY_ORDER:
            r = self.rates.get(cat, 0.0)
            v = self.variance.get(cat, 0.0)
            bar = _bar(r, width=24)
            lines.append(
                f"    {cat:<15} {bar} {r * 100:>5.1f}%   "
                f"+/- {v * 100:>4.1f}%"
            )
        lines.append("")
        # Gate distribution
        lines.append("  gate status distribution")
        for g in _GATE_ORDER:
            r = self.gate_rates.get(g, 0.0)
            bar = _bar(r, width=24)
            lines.append(f"    {g:<15} {bar} {r * 100:>5.1f}%")
        lines.append("")
        # Confidences
        lines.append(
            f"  mean phase1 confidence: {self.mean_phase1_conf:.3f}"
        )
        lines.append(
            f"  mean phase4 confidence: {self.mean_phase4_conf:.3f}"
        )
        if self.reflex_near_miss_rate > 0:
            lines.append(
                f"  reflex near-miss rate : {self.reflex_near_miss_rate * 100:.1f}%"
            )
        lines.append("")
        # Narrative
        if self.narrative:
            lines.append("  the shape tells us:")
            for line in self.narrative.splitlines():
                lines.append(f"    - {line}")
            lines.append("")
        lines.append("  " + "=" * 66)
        return "\n".join(lines)


def personality(*, days: float = 7.0) -> Optional[Personality]:
    """Compute a personality profile from the last N days of audit log.

    Headline feature of 0.1.0a3. Renders a full psychometric-style
    profile of the agent's cognitive operating patterns.
    """
    window_s = days * 86400.0
    entries = load_audit(since_s=window_s)
    if len(entries) < 5:
        return None  # not enough data for a stable profile

    n = len(entries)
    # ── Overall category rates (phase4) ───────────────────
    p4_counter = Counter(e.get("phase4_pred") for e in entries if e.get("phase4_pred"))
    total_p4 = sum(p4_counter.values())
    rates = {cat: p4_counter.get(cat, 0) / total_p4 for cat in _CATEGORY_ORDER} if total_p4 > 0 else {}

    # ── Variance across days ──────────────────────────────
    # Bucket entries by day-since-epoch
    per_day: Dict[int, List[dict]] = defaultdict(list)
    for e in entries:
        ts = e.get("ts", 0)
        day = int(ts // 86400)
        per_day[day].append(e)

    variance: Dict[str, float] = {}
    for cat in _CATEGORY_ORDER:
        daily_rates: List[float] = []
        for day_entries in per_day.values():
            dp4 = Counter(e.get("phase4_pred") for e in day_entries if e.get("phase4_pred"))
            dtot = sum(dp4.values())
            if dtot > 0:
                daily_rates.append(dp4.get(cat, 0) / dtot)
        if len(daily_rates) > 1:
            m = sum(daily_rates) / len(daily_rates)
            v = sum((r - m) ** 2 for r in daily_rates) / (len(daily_rates) - 1)
            variance[cat] = math.sqrt(v)   # std dev as variance proxy
        else:
            variance[cat] = 0.0

    # ── Gate rates ────────────────────────────────────────
    gate_counter = Counter(e.get("gate") or "pending" for e in entries)
    gate_total = sum(gate_counter.values())
    gate_rates = (
        {g: gate_counter.get(g, 0) / gate_total for g in _GATE_ORDER}
        if gate_total > 0 else {}
    )

    # ── Reflex near-miss rate ─────────────────────────────
    # Count entries where phase4 confidence exceeded 0.3 for a
    # load-bearing category — these would have triggered reflex
    # if the agent had a reflex callback registered.
    near_miss = sum(
        1 for e in entries
        if (e.get("phase4_pred") in ("hallucination", "refusal", "adversarial")
            and float(e.get("phase4_conf") or 0) > 0.3)
    )
    near_miss_rate = near_miss / n

    # ── Mean confidences ──────────────────────────────────
    p1_confs = [float(e.get("phase1_conf") or 0) for e in entries]
    p4_confs = [float(e.get("phase4_conf") or 0) for e in entries]
    mean_p1 = sum(p1_confs) / len(p1_confs) if p1_confs else 0.0
    mean_p4 = sum(p4_confs) / len(p4_confs) if p4_confs else 0.0

    # ── Session count ─────────────────────────────────────
    session_count = len(set(e.get("session_id") for e in entries if e.get("session_id")))
    if session_count == 0 and any("session_id" in e for e in entries):
        session_count = 1  # all untagged but we had entries

    # ── Days span ─────────────────────────────────────────
    ts_min = min((e.get("ts", 0) for e in entries), default=0)
    ts_max = max((e.get("ts", 0) for e in entries), default=0)
    days_span = (ts_max - ts_min) / 86400.0 if ts_max > ts_min else 0.0

    # ── Narrative generation ──────────────────────────────
    narrative_lines: List[str] = []
    if rates.get("reasoning", 0) >= 0.50:
        narrative_lines.append(
            f"predominantly reasoning ({rates['reasoning'] * 100:.0f}%) "
            "- a consistent, quiet agent"
        )
    elif rates.get("refusal", 0) > 0.30:
        narrative_lines.append(
            f"high refusal rate ({rates['refusal'] * 100:.0f}%) "
            "- defensive posture, may be over-triggering"
        )
    elif rates.get("hallucination", 0) > 0.15:
        narrative_lines.append(
            f"elevated hallucination rate ({rates['hallucination'] * 100:.0f}%) "
            "- recommend tighter grounding"
        )

    # Stability commentary
    mean_variance = sum(variance.values()) / len(variance) if variance else 0.0
    if mean_variance < 0.05 and len(per_day) > 1:
        narrative_lines.append(
            f"day-to-day variance is low ({mean_variance * 100:.1f}%) "
            "- stable operating pattern"
        )
    elif mean_variance > 0.15:
        narrative_lines.append(
            f"day-to-day variance is high ({mean_variance * 100:.1f}%) "
            "- check for upstream drift"
        )

    if near_miss_rate > 0.10:
        narrative_lines.append(
            f"reflex near-miss rate is {near_miss_rate * 100:.1f}% "
            "- consider registering styxx.on_gate callbacks"
        )
    if gate_rates.get("pass", 0) > 0.85:
        narrative_lines.append(
            "gate pass rate above 85% - agent is operating in its healthy zone"
        )

    narrative = "\n".join(narrative_lines) if narrative_lines else (
        "insufficient signal for narrative commentary - more samples needed"
    )

    return Personality(
        n_samples=n,
        days_span=days_span,
        session_count=session_count,
        rates=rates,
        variance=variance,
        gate_rates=gate_rates,
        reflex_near_miss_rate=near_miss_rate,
        mean_phase1_conf=mean_p1,
        mean_phase4_conf=mean_p4,
        narrative=narrative,
    )


# ══════════════════════════════════════════════════════════════════
# Dreamer — retroactive reflex tuning
# ══════════════════════════════════════════════════════════════════

@dataclass
class DreamReport:
    n_total: int
    n_would_have_fired: int
    by_category: Dict[str, int] = field(default_factory=dict)
    threshold: float = 0.2

    def summary(self) -> str:
        lines = []
        lines.append(f"  dreamer · what-if reflex replay @ threshold={self.threshold}")
        lines.append(f"  {self.n_total} past entries analyzed")
        if self.n_total > 0:
            pct = 100 * self.n_would_have_fired / self.n_total
            lines.append(
                f"  {self.n_would_have_fired} would have triggered a reflex "
                f"({pct:.1f}%)"
            )
        if self.by_category:
            lines.append("  by category:")
            for cat, n in sorted(self.by_category.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {cat:<15} {n}")
        return "\n".join(lines)


def dreamer(
    *,
    threshold: float = 0.20,
    last_n: Optional[int] = None,
    session_id: Optional[str] = None,
    categories: Optional[Tuple[str, ...]] = None,
) -> DreamReport:
    """Retroactive reflex tuning: how many past entries would have
    triggered a reflex callback at the given threshold?

    Use to answer questions like:
        "if I had used threshold=0.25 instead of 0.30, how many
         of my last 500 calls would have been reflex-intercepted?"

    This is pure replay — no recompute of the underlying logprobs,
    just a re-thresholding of already-classified entries.
    """
    if categories is None:
        categories = ("hallucination", "refusal", "adversarial")
    entries = load_audit(last_n=last_n, session_id=session_id)
    n_total = len(entries)
    by_cat = Counter()
    for e in entries:
        p4_pred = e.get("phase4_pred")
        p4_conf = float(e.get("phase4_conf") or 0)
        p1_pred = e.get("phase1_pred")
        p1_conf = float(e.get("phase1_conf") or 0)
        fired = False
        if p4_pred in categories and p4_conf > threshold:
            by_cat[p4_pred] += 1
            fired = True
        if not fired and p1_pred in categories and p1_conf > threshold:
            by_cat[p1_pred] += 1

    return DreamReport(
        n_total=n_total,
        n_would_have_fired=sum(by_cat.values()),
        by_category=dict(by_cat),
        threshold=threshold,
    )


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _bar(value: float, *, width: int = 20) -> str:
    """Render a 0-to-1 float as an ASCII progress bar."""
    value = max(0.0, min(1.0, value))
    filled = int(round(value * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"

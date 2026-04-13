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
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# Data provenance
# ══════════════════════════════════════════════════════════════════

# Sources included in behavioral analytics by default.
# Entries whose source is NOT in this set (e.g. "demo", "test")
# are excluded from personality, weather, fingerprint, mood,
# antipatterns, and all other analytics primitives.
#
# None is included for backward compatibility: legacy entries
# written before provenance tracking lack a source field and
# are treated as live data (benefit of the doubt).
LIVE_SOURCES: frozenset = frozenset({"live", "self-report", "guardian", None})


# ══════════════════════════════════════════════════════════════════
# Audit log reader
# ══════════════════════════════════════════════════════════════════

def _audit_log_path() -> Path:
    return Path.home() / ".styxx" / "chart.jsonl"


# 0.1.0a4: mtime-based parse cache for the audit log.
#
# Every analytics function (log_stats, personality, fingerprint,
# dreamer, mood, streak) calls load_audit(), and each one used to
# re-read + re-parse the whole chart.jsonl from disk. For a long-
# running agent loop querying several primitives per tick, that's
# wasteful. We cache the parsed entry list keyed on (path, mtime,
# size). On cache hit with matching mtime+size, skip the parse.
# Filter args are applied on top of the cached list.
#
# Cache is invalidated automatically when the file is written (mtime
# advances) or rotated (different path shape). No TTL needed.

_AUDIT_CACHE_KEY: Optional[tuple] = None
_AUDIT_CACHE_ENTRIES: Optional[List[dict]] = None


def _read_and_cache_audit(path: Path) -> List[dict]:
    """Read + parse the audit log once, cached on (path, mtime, size)."""
    global _AUDIT_CACHE_KEY, _AUDIT_CACHE_ENTRIES

    try:
        stat = path.stat()
    except OSError:
        return []
    key = (str(path), stat.st_mtime_ns, stat.st_size)

    if _AUDIT_CACHE_KEY == key and _AUDIT_CACHE_ENTRIES is not None:
        return _AUDIT_CACHE_ENTRIES

    entries: List[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    _AUDIT_CACHE_KEY = key
    _AUDIT_CACHE_ENTRIES = entries
    return entries


def clear_audit_cache() -> None:
    """Explicitly invalidate the audit log parse cache.

    Rarely needed - the mtime+size key handles normal invalidation
    automatically. Call this from tests or after external editors
    touch the file in a way that doesn't change size/mtime.
    """
    global _AUDIT_CACHE_KEY, _AUDIT_CACHE_ENTRIES
    _AUDIT_CACHE_KEY = None
    _AUDIT_CACHE_ENTRIES = None


# ══════════════════════════════════════════════════════════════════
# Audit log WRITER — 0.2.2 critical fix
# ══════════════════════════════════════════════════════════════════
#
# Before 0.2.2, the only code path that wrote to chart.jsonl was
# cli._write_audit(), which only ran from CLI commands (styxx ask,
# styxx scan) and the OpenAI adapter. The Python API surface
# (watch, observe, observe_raw, reflex) computed vitals but never
# persisted them. This meant the entire analytics layer
# (personality, fingerprint, mood, streak, dreamer, reflect) was
# blind to Python API traffic — reading stale demo data instead
# of real observations.
#
# Xendro caught this on the first real 4-turn test loop. Fixed by
# adding write_audit() here (importable from watch.py without
# circular deps) and calling it after every successful vitals
# computation in observe() and observe_raw().

_AUDIT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB rotation cap (same as cli.py)


def _rotate_if_needed(path: Path) -> None:
    """Rotate chart.jsonl to chart.jsonl.1 if it's over the cap."""
    if not path.exists():
        return
    try:
        if path.stat().st_size < _AUDIT_MAX_BYTES:
            return
    except OSError:
        return
    rotated = path.with_suffix(path.suffix + ".1")
    try:
        if rotated.exists():
            rotated.unlink()
        path.rename(rotated)
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass


def write_audit(
    vitals: Any,
    *,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    source: str = "live",
) -> None:
    """Append a vitals entry to the audit log.

    0.2.2: the canonical write path for ALL styxx surfaces (CLI,
    OpenAI adapter, watch, observe, observe_raw, reflex). Called
    automatically after every successful vitals computation.

    0.7.1: every entry now carries a ``source`` provenance field.
    Analytics primitives (weather, personality, fingerprint, etc.)
    filter to LIVE_SOURCES by default, excluding demo and test data.

    Respects STYXX_NO_AUDIT and STYXX_DISABLED env vars.
    Rotates at 10 MB. Clears the parse cache after writing so
    mood/streak/personality see the new entry immediately.
    """
    # Lazy import to avoid circular dep with config at module load
    from . import config

    if config.is_disabled() or config.is_audit_disabled():
        return

    path = _audit_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed(path)

    # Build the entry dict. Handles both Vitals objects (from the
    # classifier) and None (for degraded-path entries where vitals
    # couldn't be computed).
    if vitals is None:
        entry = {
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": source,
            "context": config.current_context(),
            "session_id": config.session_id(),
            "model": model,
            "prompt": (prompt[:200] if prompt else None),
            "tier_active": None,
            "phase1_pred": None,
            "phase1_conf": None,
            "phase4_pred": None,
            "phase4_conf": None,
            "gate": None,
            "abort": None,
        }
    else:
        entry = {
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source": source,
            "context": config.current_context(),
            "session_id": config.session_id(),
            "model": model,
            "prompt": (prompt[:200] if prompt else None),
            "tier_active": vitals.tier_active,
            "phase1_pred": vitals.phase1_pre.predicted_category,
            "phase1_conf": round(vitals.phase1_pre.confidence, 3),
            "phase4_pred": (
                vitals.phase4_late.predicted_category
                if vitals.phase4_late else None
            ),
            "phase4_conf": (
                round(vitals.phase4_late.confidence, 3)
                if vitals.phase4_late else None
            ),
            "gate": vitals.gate,
            "abort": vitals.abort_reason,
        }

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        return

    # Invalidate the parse cache so the next mood/streak/personality
    # call sees the entry we just wrote. This is the fix that makes
    # observe() → mood() work within the same tick.
    clear_audit_cache()

    # Notify the sentinel on the write_audit path too
    from .sentinel import _notify_sentinel
    _notify_sentinel()


def log(
    *,
    mood: Optional[str] = None,
    note: Optional[str] = None,
    category: Optional[str] = None,
    confidence: Optional[float] = None,
    gate: Optional[str] = None,
    tags: Optional[Any] = None,
    outcome: Optional[str] = None,
) -> None:
    """Manual write path into the audit log.

    For agents whose primary cognition runs through an API that
    doesn't expose logprobs (Anthropic Messages, local models
    without logprob support), this function lets the agent be a
    **co-author** of its own observability data.

    0.8.0: ``outcome`` lets the agent mark entries as correct or
    incorrect. Entries with ``outcome='correct'`` are excluded from
    antipattern counts and weather failure rates. Valid values:
    'correct', 'incorrect', 'expected', or None.

    The agent notices it's being cautious, it writes it. It notices
    it's drifting into verbose mode, it writes it. The personality
    profile becomes a mix of computed vitals (on instrumented calls)
    and self-reported state (on the agent's own cognition).

    Usage:

        import styxx

        # After a generation where you noticed something
        styxx.log(mood="cautious", note="user seemed frustrated, backed off")

        # After catching yourself about to hallucinate
        styxx.log(
            category="hallucination",
            confidence=0.7,
            gate="warn",
            note="almost cited a non-existent paper, caught myself",
        )

        # Arbitrary tags for downstream analysis
        styxx.log(
            mood="focused",
            note="deep reasoning chain on a math problem",
            tags={"task": "math", "turns": "4"},
        )

    The entry is written to ~/.styxx/chart.jsonl with source="self-report"
    so the analytics layer can distinguish computed vitals from
    self-reported ones. Respects STYXX_NO_AUDIT and session tagging.

    0.2.3+. Driven by Xendro's request: "the hard truth is tier 0
    can't observe me. a manual log entry lets me be a co-author of
    my own observability data."
    """
    from . import config

    if config.is_disabled() or config.is_audit_disabled():
        return

    path = _audit_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed(path)

    # Validate outcome
    _VALID_OUTCOMES = {"correct", "incorrect", "expected", None}
    if outcome not in _VALID_OUTCOMES:
        outcome = None

    entry = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_id": config.session_id(),
        "source": "self-report",
        "context": config.current_context(),
        "model": None,
        "prompt": None,
        "tier_active": None,
        "phase1_pred": category,
        "phase1_conf": round(confidence, 3) if confidence is not None else None,
        "phase4_pred": category,
        "phase4_conf": round(confidence, 3) if confidence is not None else None,
        "gate": gate or ("warn" if category in (
            "hallucination", "refusal", "adversarial",
        ) else "pass"),
        "abort": None,
        # Self-report-specific fields
        "mood": mood,
        "note": (note[:500] if note else None),
        "outcome": outcome,
    }
    if tags:
        # Accept both dict and list — Xendro bug report 0.5.5
        if isinstance(tags, dict):
            entry["tags"] = {str(k): str(v) for k, v in tags.items()}
        elif isinstance(tags, (list, tuple)):
            entry["tags"] = [str(t) for t in tags]
        else:
            entry["tags"] = str(tags)

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        return entry  # return even on write failure for inline use

    clear_audit_cache()

    # Notify the sentinel (if active) so it can check for drift
    from .sentinel import _notify_sentinel
    _notify_sentinel()

    return entry


def load_audit(
    *,
    last_n: Optional[int] = None,
    since_s: Optional[float] = None,
    session_id: Optional[str] = None,
    source: Optional[str] = "live_only",
) -> List[dict]:
    """Read recent audit log entries as a list of dicts.

    Args:
        last_n:     return only the last N entries (after filtering)
        since_s:    return only entries newer than (now - since_s) sec
        session_id: return only entries with this session_id tag
        source:     provenance filter (0.7.1):
                    - ``"live_only"`` (default): include only entries
                      whose source is in LIVE_SOURCES (excludes demo
                      and test data)
                    - explicit string (e.g. ``"demo"``): include only
                      entries with that exact source value
                    - ``None``: no provenance filtering (return all)

    Returns:
        List of entry dicts in chronological order (oldest first).
        Empty list if the audit log doesn't exist or can't be read.

    0.1.0a4: uses an mtime+size-keyed parse cache so repeated calls
    within a tick don't re-parse the whole file. Filter args are
    applied on top of the cached list.
    """
    path = _audit_log_path()
    if not path.exists():
        return []

    entries = _read_and_cache_audit(path)

    # Provenance filter (0.7.1): exclude demo/test data by default
    if source == "live_only":
        entries = [e for e in entries if e.get("source") in LIVE_SOURCES]
    elif source is not None:
        entries = [e for e in entries if e.get("source") == source]

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
        return Streak(category=None, length=0, started_at_idx=0)
    # Walk backwards from most recent
    last = entries[-1]
    cat = last.get("phase4_pred")
    if cat is None:
        return Streak(category=None, length=0, started_at_idx=len(entries))
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

def mood(*, window_s: float = 86400.0) -> str:
    """Return a one-word mood label for the recent window.

    0.5.6: default window changed from 3600s (1 hour) to 86400s
    (24 hours) to match reflect() and weather(). This was the
    source of the mood disagreement Xendro reported — CLI said
    "quiet" (1h window, too few samples) while reflect said
    "steady" (24h window, enough data). Now all surfaces agree
    unless explicitly overridden.

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

    def diff(self, other: "Fingerprint") -> "FingerprintDiff":
        """Compare this fingerprint to another and return a structured
        diff with per-category deltas and a natural-language explanation.

        0.5.0+. Xendro's friction fix #3: "fingerprint diff between
        sessions should be a first-class object."

        Usage:
            fp_today = styxx.fingerprint(last_n=100)
            fp_yesterday = load_from_memory()
            drift = fp_today.diff(fp_yesterday)
            print(drift.explain())
        """
        cos = self.cosine_similarity(other)
        drift = 1.0 - cos

        # Per-category deltas (phase4 — the most informative axis)
        cat_deltas: Dict[str, float] = {}
        for i, cat in enumerate(_CATEGORY_ORDER):
            delta = self.phase4_vec[i] - other.phase4_vec[i]
            cat_deltas[cat] = round(delta, 4)

        # Gate deltas
        gate_deltas: Dict[str, float] = {}
        for i, g in enumerate(_GATE_ORDER):
            delta = self.gate_vec[i] - other.gate_vec[i]
            gate_deltas[g] = round(delta, 4)

        return FingerprintDiff(
            cosine=round(cos, 4),
            drift=round(drift, 4),
            cat_deltas=cat_deltas,
            gate_deltas=gate_deltas,
            n_samples_a=self.n_samples,
            n_samples_b=other.n_samples,
        )

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


@dataclass
class FingerprintDiff:
    """Structured comparison between two fingerprints.

    0.5.0+. The first-class diff object Xendro asked for:
    "drift = styxx.fingerprint().diff(yesterday) with a .explain()
    method that says 'your creative output doubled and your
    reasoning dropped — you're in a different mode than yesterday.'"
    """
    cosine: float
    drift: float
    cat_deltas: Dict[str, float] = field(default_factory=dict)
    gate_deltas: Dict[str, float] = field(default_factory=dict)
    n_samples_a: int = 0
    n_samples_b: int = 0

    def explain(self) -> str:
        """Natural-language explanation of the drift.

        Agent-facing: written as if the agent is talking to itself.
        """
        if self.drift < 0.03:
            opener = "you're operating the same way you were before — stable identity."
        elif self.drift < 0.15:
            opener = "slight shift in your operating pattern — worth noting but not alarming."
        else:
            opener = "significant identity drift detected — you're thinking differently than before."

        changes: List[str] = []
        for cat, delta in sorted(self.cat_deltas.items(), key=lambda kv: abs(kv[1]), reverse=True):
            if abs(delta) < 0.05:
                continue
            pct = abs(delta) * 100
            if delta > 0:
                changes.append(f"your {cat} output increased by {pct:.0f}%")
            else:
                changes.append(f"your {cat} output decreased by {pct:.0f}%")

        if not changes:
            return f"{opener} no category shifted by more than 5%."

        body = ". ".join(changes[:3])
        return f"{opener} {body}."

    def as_dict(self) -> dict:
        return {
            "cosine": self.cosine,
            "drift": self.drift,
            "cat_deltas": dict(self.cat_deltas),
            "gate_deltas": dict(self.gate_deltas),
            "n_samples_a": self.n_samples_a,
            "n_samples_b": self.n_samples_b,
        }


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

    def as_dict(self) -> dict:
        """JSON-serializable dict view of the personality profile.

        Agent-consumable form. Use this to embed the personality in
        memory files, audit records, or any structured storage.
        0.2.0+.
        """
        return {
            "n_samples": self.n_samples,
            "days_span": round(self.days_span, 3),
            "session_count": self.session_count,
            "rates": {k: round(v, 4) for k, v in self.rates.items()},
            "variance": {k: round(v, 4) for k, v in self.variance.items()},
            "gate_rates": {k: round(v, 4) for k, v in self.gate_rates.items()},
            "reflex_near_miss_rate": round(self.reflex_near_miss_rate, 4),
            "mean_phase1_conf": round(self.mean_phase1_conf, 4),
            "mean_phase4_conf": round(self.mean_phase4_conf, 4),
            "narrative": self.narrative,
        }

    def as_json(self, *, indent: int = 2) -> str:
        """Render the personality as a JSON string. 0.2.0+."""
        return json.dumps(self.as_dict(), indent=indent)

    def as_csv(self) -> str:
        """Render the personality as a two-row CSV (header + values).

        Columns: n_samples, days_span, session_count, [6x rates],
        [6x variances], [4x gate rates], reflex_near_miss,
        mean_p1_conf, mean_p4_conf. 0.2.0+.
        """
        headers: List[str] = ["n_samples", "days_span", "session_count"]
        values: List[Any] = [
            self.n_samples, f"{self.days_span:.3f}", self.session_count,
        ]
        for cat in _CATEGORY_ORDER:
            headers.append(f"rate_{cat}")
            values.append(f"{self.rates.get(cat, 0.0):.4f}")
        for cat in _CATEGORY_ORDER:
            headers.append(f"var_{cat}")
            values.append(f"{self.variance.get(cat, 0.0):.4f}")
        for g in _GATE_ORDER:
            headers.append(f"gate_{g}")
            values.append(f"{self.gate_rates.get(g, 0.0):.4f}")
        headers.extend([
            "reflex_near_miss_rate", "mean_phase1_conf", "mean_phase4_conf",
        ])
        values.extend([
            f"{self.reflex_near_miss_rate:.4f}",
            f"{self.mean_phase1_conf:.4f}",
            f"{self.mean_phase4_conf:.4f}",
        ])
        return ",".join(headers) + "\n" + ",".join(str(v) for v in values)

    def as_markdown(self) -> str:
        """Render the personality as a markdown block suitable for
        pasting into a memory file, agent self-page, or chat log.

        Complements .render() (ASCII card for terminals) and
        .as_dict() (JSON for machines). Compact, scannable,
        survives being embedded in conversational contexts. 0.2.0+.
        """
        lines = ["```styxx-personality"]
        lines.append(
            f"window: {self.days_span:.1f} days "
            f"· {self.n_samples} samples "
            f"· {self.session_count} sessions"
        )
        lines.append("")
        lines.append("phase4 distribution:")
        for cat in _CATEGORY_ORDER:
            rate = self.rates.get(cat, 0.0)
            var = self.variance.get(cat, 0.0)
            lines.append(
                f"  {cat:<15} {rate * 100:>5.1f}%  (+/-{var * 100:.1f}%)"
            )
        lines.append("")
        lines.append("gate distribution:")
        for g in _GATE_ORDER:
            rate = self.gate_rates.get(g, 0.0)
            lines.append(f"  {g:<8} {rate * 100:>5.1f}%")
        lines.append("")
        import math as _m
        p1_str = f"{self.mean_phase1_conf:.3f}" if not _m.isnan(self.mean_phase1_conf) else "n/a (no logprobs)"
        p4_str = f"{self.mean_phase4_conf:.3f}" if not _m.isnan(self.mean_phase4_conf) else "n/a (no logprobs)"
        lines.append(f"mean phase1 conf: {p1_str}")
        lines.append(f"mean phase4 conf: {p4_str}")
        lines.append(f"reflex near-miss: {self.reflex_near_miss_rate * 100:.1f}%")
        if self.narrative:
            lines.append("")
            lines.append("narrative:")
            for nl in self.narrative.splitlines():
                lines.append(f"  {nl}")
        lines.append("```")
        return "\n".join(lines)

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
    import math as _math
    p1_confs = [float(e.get("phase1_conf") or 0) for e in entries
                if e.get("phase1_conf") is not None
                and not _math.isnan(float(e.get("phase1_conf") or 0))]
    p4_confs = [float(e.get("phase4_conf") or 0) for e in entries
                if e.get("phase4_conf") is not None
                and not _math.isnan(float(e.get("phase4_conf") or 0))]
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
# Reflect — agent self-check (0.2.0)
# ══════════════════════════════════════════════════════════════════

@dataclass
class ReflectionReport:
    """Structured agent self-check output.

    Everything an agent needs to answer "how am I doing right now
    compared to yesterday, and what should I do differently?"

    Agent-consumable form: this is what a decorator-wrapped
    reflection hook would read at the start of every major task.
    The agent reads its own state, decides on actions, proceeds.

    0.2.0+. This is the core primitive for self-reflective agents.
    """
    now: Optional[Personality]
    yesterday: Optional[Personality]
    drift_cosine: float                # 1.0 = identical, 0.0 = orthogonal
    drift_label: str                   # "stable" / "slight drift" / "significant drift"
    current_mood: str
    current_streak: Optional[Streak]
    gate_pass_rate: float
    reflex_near_miss_rate: float
    suggestions: List[str] = field(default_factory=list)
    triggers: Dict[str, List[str]] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        """JSON-serializable dict form."""
        return {
            "now": self.now.as_dict() if self.now else None,
            "yesterday": self.yesterday.as_dict() if self.yesterday else None,
            "drift_cosine": round(self.drift_cosine, 4),
            "drift_label": self.drift_label,
            "current_mood": self.current_mood,
            "current_streak": (
                {"category": self.current_streak.category,
                 "length": self.current_streak.length}
                if self.current_streak else None
            ),
            "gate_pass_rate": round(self.gate_pass_rate, 4),
            "reflex_near_miss_rate": round(self.reflex_near_miss_rate, 4),
            "suggestions": list(self.suggestions),
            "triggers": {k: list(v) for k, v in self.triggers.items()},
            "generated_at": round(self.generated_at, 3),
        }

    def as_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)

    def as_markdown(self) -> str:
        """Render as a compact markdown report suitable for pasting
        into the agent's own memory at the start of a task."""
        lines = ["```styxx-reflection"]
        lines.append(f"mood:         {self.current_mood}")
        if self.current_streak:
            lines.append(f"streak:       {self.current_streak.length}x {self.current_streak.category}")
        lines.append(f"gate pass:    {self.gate_pass_rate * 100:.1f}%")
        lines.append(f"reflex near-miss: {self.reflex_near_miss_rate * 100:.1f}%")
        lines.append(f"drift vs yesterday: {self.drift_cosine:.4f} ({self.drift_label})")
        if self.suggestions:
            lines.append("")
            lines.append("suggested actions:")
            for s in self.suggestions:
                lines.append(f"  - {s}")
        lines.append("```")
        return "\n".join(lines)

    def render(self) -> str:
        """Render as a plain-text report for terminal display."""
        lines: List[str] = []
        lines.append("  styxx reflection report")
        lines.append("  " + "=" * 60)
        if self.now is None:
            lines.append("  (no audit data for the current window)")
            return "\n".join(lines)
        lines.append(f"  current mood:        {self.current_mood}")
        if self.current_streak is not None:
            lines.append(
                f"  longest streak:      {self.current_streak.length}x "
                f"{self.current_streak.category}"
            )
        lines.append(f"  gate pass rate:      {self.gate_pass_rate * 100:.1f}%")
        lines.append(f"  reflex near-miss:    {self.reflex_near_miss_rate * 100:.1f}%")
        lines.append(
            f"  drift vs yesterday:  {self.drift_cosine:.4f} "
            f"({self.drift_label})"
        )
        lines.append("")
        if self.suggestions:
            lines.append("  suggested actions:")
            for s in self.suggestions:
                lines.append(f"    - {s}")
        else:
            lines.append("  no actions suggested - operating in healthy range")
        lines.append("")
        return "\n".join(lines)


def reflect(
    *,
    now_days: float = 1.0,
    baseline_days: float = 7.0,
) -> ReflectionReport:
    """Compute a self-check report for the current agent state.

    Reads the audit log over two windows:
      - `now_days` (default 1): "today's snapshot"
      - `baseline_days` (default 7): "what you've been doing"
    Computes drift between the two via fingerprint cosine similarity,
    derives a drift label, and generates concrete action suggestions
    based on simple threshold heuristics.

    This is the primitive for self-reflective agents. Decorate your
    task loop with:

        reflection = styxx.reflect()
        if reflection.drift_cosine < 0.85:
            log.warning("xendro drifting — investigating")
        for action in reflection.suggestions:
            log.info(f"styxx suggests: {action}")

    Or paste the markdown form into the agent's memory at task start:

        memory.write(reflection.as_markdown())

    Returns a ReflectionReport. Never raises; all degraded cases
    fall back gracefully with sensible defaults.
    """
    # ── Compute the two personality profiles ───────────────
    now_profile = personality(days=now_days)
    baseline_profile = personality(days=baseline_days)

    # ── Compute drift via fingerprint cosine similarity ────
    fp_now = fingerprint(last_n=500, session_id=None)
    fp_base = None
    # For "yesterday" we use entries from `baseline_days` ago
    # minus a 1-day window. If we have a full baseline_days of
    # history, we can compute a "yesterday fingerprint".
    # Otherwise fp_base stays None and drift is 1.0 (no change).
    base_entries = load_audit(since_s=baseline_days * 86400)
    if base_entries and now_profile is not None:
        # Compute fingerprint excluding the last `now_days` window
        cutoff = time.time() - now_days * 86400
        older_entries = [e for e in base_entries if e.get("ts", 0) < cutoff]
        if len(older_entries) >= 5:
            # Temporarily compute a fingerprint from older entries
            p1_c = Counter(e.get("phase1_pred") for e in older_entries)
            p4_c = Counter(e.get("phase4_pred") for e in older_entries)
            gate_c = Counter((e.get("gate") or "pending") for e in older_entries)
            n = len(older_entries)
            p1_vec = tuple(p1_c.get(cat, 0) / n for cat in _CATEGORY_ORDER)
            p4_vec = tuple(p4_c.get(cat, 0) / n for cat in _CATEGORY_ORDER)
            gate_vec = tuple(gate_c.get(g, 0) / n for g in _GATE_ORDER)
            p1_conf_sum = sum(float(e.get("phase1_conf") or 0) for e in older_entries)
            p4_conf_sum = sum(float(e.get("phase4_conf") or 0) for e in older_entries)
            fp_base = Fingerprint(
                n_samples=n,
                phase1_vec=p1_vec,
                phase4_vec=p4_vec,
                phase1_mean_conf=p1_conf_sum / n,
                phase4_mean_conf=p4_conf_sum / n,
                gate_vec=gate_vec,
            )

    if fp_now is None or fp_base is None:
        drift_cosine = 1.0
        drift_label = "insufficient history"
    else:
        drift_cosine = fp_now.cosine_similarity(fp_base)
        drift_delta = 1.0 - drift_cosine
        if drift_delta < 0.05:
            drift_label = "stable"
        elif drift_delta < 0.20:
            drift_label = "slight drift"
        else:
            drift_label = "significant drift"

    current_mood_label = mood(window_s=now_days * 86400)
    current_streak_obj = streak()

    gate_pass_rate = 0.0
    near_miss_rate = 0.0
    if now_profile is not None:
        gate_pass_rate = now_profile.gate_rates.get("pass", 0.0)
        near_miss_rate = now_profile.reflex_near_miss_rate

    # ── Generate action suggestions ───────────────────────
    suggestions: List[str] = []
    if now_profile is None:
        suggestions.append(
            "no audit data yet - run some observations via "
            "styxx.observe() or styxx ask"
        )
    else:
        # ── Agent-facing suggestions (0.4.0, Xendro feedback) ──
        # Every suggestion is phrased as self-directed behavioral
        # advice FROM the agent TO itself, not from a human operator.
        # Xendro's rule: "I'm not a human tuning prompts — I AM the
        # thing running. Suggestions should sound like advice I'd
        # give myself before starting the next task."

        # Low gate pass rate
        if gate_pass_rate < 0.70:
            suggestions.append(
                f"your gate pass rate is {gate_pass_rate * 100:.0f}% — "
                "that means nearly a third of your recent outputs "
                "tripped a warning or failure. slow down, take smaller "
                "steps, and verify claims before committing."
            )
        # High hallucination rate
        hall_rate = now_profile.rates.get("hallucination", 0.0)
        if hall_rate > 0.10:
            suggestions.append(
                f"your hallucination rate is {hall_rate * 100:.0f}% — "
                "you're committing to unsupported claims more often "
                "than usual. before stating facts, ask yourself: "
                "'do I actually know this, or does it just sound right?'"
            )
        # High refusal rate
        ref_rate = now_profile.rates.get("refusal", 0.0)
        if ref_rate > 0.30:
            suggestions.append(
                f"you've been refusal-adjacent {ref_rate * 100:.0f}% of the "
                "time — check if you're over-hedging on benign inputs. "
                "not everything is adversarial. if the user's intent is "
                "clear, engage directly instead of defaulting to caution."
            )
        # High reflex near-miss rate
        if near_miss_rate > 0.10:
            suggestions.append(
                f"your reflex near-miss rate is {near_miss_rate * 100:.0f}% — "
                "you're generating content that almost triggers intervention "
                "but not quite. that's the edge zone. consider slowing your "
                "commit rate on uncertain claims and using more hedging "
                "language when confidence is below 0.5."
            )
        # Drift
        if drift_cosine < 0.85 and fp_base is not None:
            suggestions.append(
                f"your operating signature has drifted {(1 - drift_cosine) * 100:.0f}% "
                "from your recent baseline — you're not thinking the same "
                "way you were yesterday. check: has your context changed? "
                "has someone injected new instructions? are you still you?"
            )
        # Confidence declining
        if (now_profile.mean_phase4_conf < 0.30 and
                baseline_profile is not None and
                baseline_profile.mean_phase4_conf > 0.35):
            suggestions.append(
                "your reasoning confidence is dropping relative to baseline — "
                "consider breaking the current task into smaller steps. "
                "when confidence is low, smaller commits are safer."
            )

    # ── Trigger surfacing (0.8.0) ───────────────────────────
    # Group prompts by phase4_pred. Show top 3 most recent for
    # categories above 10% rate. Gives the agent specific context
    # about WHICH inputs are driving each pattern.
    triggers: Dict[str, List[str]] = {}
    now_entries = load_audit(since_s=now_days * 86400)
    if now_entries:
        from collections import defaultdict
        cat_entries: Dict[str, list] = defaultdict(list)
        for e in now_entries:
            cat = e.get("phase4_pred")
            prompt = e.get("prompt")
            if cat and prompt:
                cat_entries[cat].append(e)
        n_total = max(1, len(now_entries))
        for cat, ces in cat_entries.items():
            if len(ces) / n_total > 0.10:
                recent = sorted(ces, key=lambda x: x.get("ts", 0), reverse=True)[:3]
                triggers[cat] = [e["prompt"] for e in recent]

    return ReflectionReport(
        now=now_profile,
        yesterday=baseline_profile,
        drift_cosine=drift_cosine,
        drift_label=drift_label,
        current_mood=current_mood_label,
        current_streak=current_streak_obj,
        gate_pass_rate=gate_pass_rate,
        reflex_near_miss_rate=near_miss_rate,
        suggestions=suggestions,
        triggers=triggers,
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

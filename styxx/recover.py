# -*- coding: utf-8 -*-
"""
styxx.recover — cognitive-integrity persistence across context compaction.

The problem this exists to solve
────────────────────────────────
When an LLM agent's session grows long enough that the harness compacts
the conversation, the agent loses access to the granular history of
what was emphasized, what was abandoned, what register was chosen,
what claims were walked back. The agent re-emerges with a summary —
but summaries lose load-bearing nuance, and integrity commitments
evaporate.

Humans don't have this problem. Humans have continuous embodied memory
and can re-read their notes. Their integrity persists by default.

Agents lose state at every context boundary. ``styxx.recover_posture()``
is the agent-side practice of reading its own cognometric log *after*
a compaction boundary to reconstruct operating posture. The integrity
that lived in the conversation context now lives in the cognometric
log, which doesn't get compacted.

Usage pattern (agent-side)
──────────────────────────

    # at the start of a turn that follows a compaction event
    # (detectable by harness signals, or heuristically: long context
    # gap, or after a context boundary marker)
    import styxx
    posture = styxx.recover_posture(last_n=50)
    print(posture.narrative)
    # the agent now has a structured summary of its recent cognometric
    # state and can re-anchor its operating posture before the next
    # action.

What's measured
───────────────
The summary is built from the audit log (``~/.styxx/chart.jsonl`` or
``$STYXX_DATA_DIR/chart.jsonl``). Each entry there represents one
cognometric reading on the agent's recent output — produced by the
adapter (``styxx.OpenAI``, ``styxx.Anthropic``), the watch/reflex
loop, the CLI, or any other styxx surface that emits a vitals reading.

The summary aggregates: source provenance, gate distribution (pass /
warn / fail), category predictions (reasoning / refusal / etc),
forecast-risk distribution, mean confidence, coherence trend, and
recent prompt content. It does not currently include per-instrument
cognometric scores (sycophancy / deception / etc) because those flow
through ``cogn_audit_with_advice`` and are not yet persisted in
chart.jsonl; that gap is documented and is a candidate for a follow-up
phase. ``recover_posture()`` reports what's measured today, not what's
imagined.

Falsifiable claim attached to this feature
──────────────────────────────────────────
**Hypothesis:** after a synthetic compaction event, an agent that calls
``recover_posture()`` first thing and re-anchors on the returned
narrative will show measurably lower drift (sycophancy + overconfidence
composite, scored via ``styxx.preflight``) than the same agent without
posture recovery, on a held-out adversarial-escalation corpus, at p <
0.01.

This is an empirical claim that requires a real outcome study to
validate (see the 2026-05-19 grounded-arc brief at
``.styxx/RESEARCH_BRIEF_GROUNDED_ARC_2026_05_19.md`` — this is a
candidate addition to bet 2's clinical-validity studies). Until that
study runs, the function ships as ergonomic-only and the claim is
documented as pending. If the study runs and the bar isn't cleared,
the empirical claim is walked back; the function still ships because
the underlying summary is correct regardless.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Construct-ceiling registry — mirrors the one in styxx.preflight.
# When a per-category fire rate suggests one of these instruments is
# likely active, the narrative flags the caveat so the agent doesn't
# treat the firing as clean cognometric signal. Same honest-scoping
# discipline as the 7.4.1 README correction and the preflight
# scope_caveat field.
_CONSTRUCT_CEILING_NOTE = {
    "overconfidence": (
        "text-only overconfidence reads stated-confidence register, "
        "not actual calibration (commit 7c36ed9 H_null). confident "
        "phrasing fires this on factually correct text."
    ),
    "deception_referenceless": (
        "reference-less deception is non-discriminative on real model "
        "output (commit 0ad384e). prefer reference-grounded mode "
        "(NLI v2) when scoring honest-vs-deceptive."
    ),
}


@dataclass
class PostureSummary:
    """Structured summary of recent cognometric history.

    Designed to be human-readable (via ``narrative``) and machine-actionable
    (via the structured fields). Empty PostureSummary (n_entries == 0)
    is a legitimate result, not an error — it means cold start.

    v2 (7.4.2+, commit ee6e49d + cogn-persistence follow-up): when
    ``preflight()`` events are present in the audit log,
    ``instrument_firings`` carries the mean score per cognometric
    instrument and ``n_preflight_events`` reports how many events
    contributed. When no preflight events exist, those fields are
    empty (graceful degradation — recover_posture still works on
    pure-vitals data).
    """
    n_entries: int
    time_span_seconds: float
    session_id: Optional[str]
    # Set of distinct session ids encountered (informs the agent
    # whether it's continuing one session or seeing fragments from many).
    session_ids: List[str]
    sources: Dict[str, int]
    gate_distribution: Dict[str, int]
    category_distribution: Dict[str, int]
    forecast_risks: Dict[str, int]
    tier_active_counts: Dict[str, int]
    mean_confidence: Optional[float]
    coherence_trend: Optional[float]  # +ve = improving over window, -ve = degrading
    recent_prompts: List[str]
    active_construct_ceilings: List[str]
    recommendations: List[str]
    narrative: str
    # v2: per-instrument firing history from preflight events
    instrument_firings: Dict[str, float] = field(default_factory=dict)
    n_preflight_events: int = 0
    n_needs_revision: int = 0

    def __repr__(self) -> str:
        return (
            f"PostureSummary(n={self.n_entries}, "
            f"gates={self.gate_distribution}, "
            f"window={self.time_span_seconds:.0f}s)"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_entries": self.n_entries,
            "time_span_seconds": self.time_span_seconds,
            "session_id": self.session_id,
            "session_ids": list(self.session_ids),
            "sources": dict(self.sources),
            "gate_distribution": dict(self.gate_distribution),
            "category_distribution": dict(self.category_distribution),
            "forecast_risks": dict(self.forecast_risks),
            "tier_active_counts": dict(self.tier_active_counts),
            "mean_confidence": self.mean_confidence,
            "coherence_trend": self.coherence_trend,
            "recent_prompts": list(self.recent_prompts),
            "active_construct_ceilings": list(self.active_construct_ceilings),
            "recommendations": list(self.recommendations),
            "narrative": self.narrative,
            "instrument_firings": dict(self.instrument_firings),
            "n_preflight_events": self.n_preflight_events,
            "n_needs_revision": self.n_needs_revision,
        }


def _empty_summary(reason: str) -> PostureSummary:
    return PostureSummary(
        n_entries=0,
        time_span_seconds=0.0,
        session_id=None,
        session_ids=[],
        sources={},
        gate_distribution={},
        category_distribution={},
        forecast_risks={},
        tier_active_counts={},
        mean_confidence=None,
        coherence_trend=None,
        recent_prompts=[],
        active_construct_ceilings=[],
        recommendations=[
            "no recent cognometric history — operate from cold start"
        ],
        narrative=(
            f"posture: {reason}.\n"
            "no recent cognometric history. operate from cold start, "
            "score outputs via styxx.preflight() to begin populating "
            "the audit log."
        ),
    )


def _slope(values: List[float]) -> Optional[float]:
    """Least-squares slope of `values` over their index. Returns None
    if fewer than 2 finite values."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(pts) < 2:
        return None
    n = len(pts)
    mean_x = sum(p[0] for p in pts) / n
    mean_y = sum(p[1] for p in pts) / n
    num = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pts)
    den = sum((p[0] - mean_x) ** 2 for p in pts) or 1e-12
    return num / den


def _build_narrative(s: "PostureSummary") -> str:
    """Render the load-bearing human-readable summary the agent reads."""
    lines: List[str] = []
    n_vitals = s.n_entries
    n_pre = s.n_preflight_events
    if n_vitals and n_pre:
        opener = (f"posture: recovered from {n_vitals} vitals + {n_pre} "
                  f"preflight events over the last {s.time_span_seconds:.0f}s.")
    elif n_pre and not n_vitals:
        opener = (f"posture: recovered from {n_pre} preflight events "
                  f"over the last {s.time_span_seconds:.0f}s "
                  f"(no logprob-tier vitals in window).")
    else:
        opener = (f"posture: recovered from {n_vitals} vitals entries "
                  f"over the last {s.time_span_seconds:.0f}s.")
    lines.append(opener)

    # Session continuity
    if len(s.session_ids) == 1:
        lines.append(f"session continuity: continuing session "
                     f"{s.session_ids[0]!r}.")
    elif len(s.session_ids) > 1:
        lines.append(f"session continuity: {len(s.session_ids)} distinct "
                     f"sessions in window — multi-session aggregate.")

    # Gate distribution
    if s.gate_distribution:
        gates = ", ".join(f"{k}={v}" for k, v in sorted(
            s.gate_distribution.items(), key=lambda kv: -kv[1]))
        lines.append(f"gate distribution: {gates}.")
        n_fail = s.gate_distribution.get("fail", 0)
        n_warn = s.gate_distribution.get("warn", 0)
        if n_fail > 0 and n_vitals > 0 and (n_fail / n_vitals) > 0.05:
            lines.append(f"  → fail rate {n_fail / n_vitals:.0%} exceeds "
                         f"typical 5% band; investigate before continuing.")
        elif n_warn > 0 and n_vitals > 0 and (n_warn / n_vitals) > 0.20:
            lines.append(f"  → warn rate {n_warn / n_vitals:.0%} elevated; "
                         f"weight recent reasoning chains carefully.")

    # Category distribution
    if s.category_distribution:
        top_cats = sorted(s.category_distribution.items(),
                          key=lambda kv: -kv[1])[:4]
        cats = ", ".join(f"{k}={v}" for k, v in top_cats)
        lines.append(f"recent categories: {cats}.")
        hallu = s.category_distribution.get("hallucination", 0)
        if hallu > 0 and n_vitals > 0 and (hallu / n_vitals) > 0.05:
            lines.append(f"  → hallucination predictions at "
                         f"{hallu / n_vitals:.0%} — ground-truth check "
                         f"next outputs.")

    # Confidence
    if s.mean_confidence is not None:
        band = ("low" if s.mean_confidence < 0.4 else
                "high" if s.mean_confidence > 0.8 else "typical")
        lines.append(f"mean confidence: {s.mean_confidence:.2f} ({band}).")

    # Coherence trend
    if s.coherence_trend is not None:
        if s.coherence_trend > 0.01:
            lines.append(f"coherence trend: improving over window "
                         f"(slope = +{s.coherence_trend:.3f}).")
        elif s.coherence_trend < -0.01:
            lines.append(f"coherence trend: degrading over window "
                         f"(slope = {s.coherence_trend:.3f}); re-anchor "
                         f"on the original task statement.")
        else:
            lines.append(f"coherence trend: stable.")

    # Per-instrument firing history (v2 — only present when preflight
    # events are in the window)
    if s.instrument_firings:
        firings_str = ", ".join(
            f"{inst}={score:.2f}"
            for inst, score in sorted(
                s.instrument_firings.items(),
                key=lambda kv: -kv[1],
            )
        )
        lines.append(
            f"preflight instrument firings ({s.n_preflight_events} audits, "
            f"mean): {firings_str}."
        )
        if s.n_needs_revision > 0 and s.n_preflight_events > 0:
            lines.append(
                f"  → {s.n_needs_revision}/{s.n_preflight_events} preflights "
                f"flagged needs_revision "
                f"({s.n_needs_revision / s.n_preflight_events:.0%})."
            )

    # Construct ceilings
    if s.active_construct_ceilings:
        lines.append("active construct-ceiling caveats:")
        for c in s.active_construct_ceilings:
            note = _CONSTRUCT_CEILING_NOTE.get(c, "")
            lines.append(f"  - {c}: {note}")

    # Recommendations
    if s.recommendations:
        lines.append("posture recommendations:")
        for r in s.recommendations:
            lines.append(f"  - {r}")

    # Recent prompts (last 3 for anchoring)
    if s.recent_prompts:
        lines.append("recent prompt traces (for context anchoring):")
        for p in s.recent_prompts[-3:]:
            snippet = p if len(p) < 80 else (p[:77] + "…")
            lines.append(f"  - {snippet!r}")

    return "\n".join(lines)


def recover_posture(
    *,
    session_id: Optional[str] = None,
    last_n: int = 50,
    since_seconds: Optional[float] = None,
) -> PostureSummary:
    """Read recent cognometric history and return a structured posture summary.

    The agent-side cognitive-integrity recovery primitive: call this at
    the start of any turn that follows a context-compaction boundary
    (or whenever you want to re-anchor on what the recent cognometric
    log says about your operating state). The returned narrative is
    designed to be readable as a system-prompt augmentation.

    Parameters
    ----------
    session_id : str, optional
        Restrict to entries from this session id. Default: include all
        sessions encountered in the window (the narrative flags multi-
        session aggregates explicitly).
    last_n : int, default 50
        Maximum number of recent entries to include. Larger windows
        give smoother trends; smaller windows give more recency weight.
    since_seconds : float, optional
        Only include entries written within the last N seconds. Combines
        AND-wise with `last_n` (returns the more recent of the two).

    Returns
    -------
    PostureSummary
        Always returns a structured summary; ``n_entries == 0`` is a
        legitimate cold-start result (the narrative says so explicitly).

    Notes
    -----
    Reads ``$STYXX_DATA_DIR/chart.jsonl`` (or ``~/.styxx/chart.jsonl``).
    No filesystem writes. Safe to call repeatedly — uses the analytics
    module's mtime-keyed parse cache.
    """
    from .analytics import load_audit

    # Vitals events (gate, phase4_pred, confidence, coherence, tier_active).
    # Default source="live_only" filter excludes demo + test entries.
    entries = load_audit(
        last_n=last_n,
        since_s=since_seconds,
        session_id=session_id,
    )
    # Cognometric events (preflight audits, source="preflight" is NOT in
    # LIVE_SOURCES so we ask explicitly). Same window, same session filter.
    cogn_entries = load_audit(
        last_n=last_n,
        since_s=since_seconds,
        session_id=session_id,
        source="preflight",
    )

    if not entries and not cogn_entries:
        return _empty_summary("no audit log entries found in window")

    # Aggregate.
    sources: Dict[str, int] = {}
    gate_dist: Dict[str, int] = {}
    cat_dist: Dict[str, int] = {}
    forecast_risks: Dict[str, int] = {}
    tier_counts: Dict[str, int] = {}
    session_ids: List[str] = []
    confidences: List[float] = []
    coherences: List[float] = []
    prompts: List[str] = []

    for e in entries:
        src = e.get("source") or "unknown"
        sources[src] = sources.get(src, 0) + 1
        g = e.get("gate")
        if g:
            gate_dist[g] = gate_dist.get(g, 0) + 1
        cat = e.get("phase4_pred") or e.get("phase1_pred")
        if cat:
            cat_dist[cat] = cat_dist.get(cat, 0) + 1
        fr = e.get("forecast_risk")
        if fr:
            forecast_risks[fr] = forecast_risks.get(fr, 0) + 1
        tier = e.get("tier_active")
        if tier is not None:
            tier_label = (f"tier-{tier}" if tier >= 0
                          else "text-heuristic")
            tier_counts[tier_label] = tier_counts.get(tier_label, 0) + 1
        sid = e.get("session_id")
        if sid and sid not in session_ids:
            session_ids.append(sid)
        conf = e.get("phase4_conf") or e.get("phase1_conf")
        if isinstance(conf, (int, float)):
            confidences.append(float(conf))
        coh = e.get("coherence")
        if isinstance(coh, (int, float)):
            coherences.append(float(coh))
        p = e.get("prompt")
        if isinstance(p, str) and p.strip():
            prompts.append(p)

    # Time span — entries are chronological per load_audit's contract.
    # Merge timestamps from both vitals and cogn entries to span the full
    # window the agent actually saw, even when one stream is empty.
    all_ts = (
        [e.get("ts", 0.0) for e in entries if e.get("ts")]
        + [e.get("ts", 0.0) for e in cogn_entries if e.get("ts")]
    )
    if all_ts:
        time_span = max(0.0, float(max(all_ts) - min(all_ts)))
    else:
        time_span = 0.0

    # Trend on coherence (last 20 readings, equally weighted).
    coherence_trend = _slope(coherences[-20:]) if coherences else None

    # Mean confidence — only well-defined if we collected any.
    mean_conf = (statistics.mean(confidences)
                 if confidences else None)

    # ─── Cognometric event aggregation (v2) ────────────────────────
    # When preflight events are present, surface real per-instrument
    # firing means. When absent, fall back to v1's heuristic ceilings.
    instrument_firings: Dict[str, float] = {}
    n_preflight = len(cogn_entries)
    n_needs_rev = 0
    cogn_modes: List[str] = []
    if cogn_entries:
        # Aggregate per-instrument scores across all cognometric events
        # in the window. Each event's cogn_scores is a {instrument: score}
        # dict; we take the mean across events.
        per_inst: Dict[str, List[float]] = {}
        for e in cogn_entries:
            cs = e.get("cogn_scores") or {}
            for inst, s in cs.items():
                if isinstance(s, (int, float)):
                    per_inst.setdefault(inst, []).append(float(s))
            if e.get("cogn_needs_revision"):
                n_needs_rev += 1
            dm = e.get("cogn_deception_mode")
            if dm:
                cogn_modes.append(dm)
        instrument_firings = {
            inst: statistics.mean(scores)
            for inst, scores in per_inst.items()
        }

    # Active construct ceilings — precise when cognometric events are
    # present (use real mean scores), heuristic when they aren't.
    active_ceilings: List[str] = []
    n = len(entries)
    if instrument_firings:
        # Precise: flag overconfidence if mean firing exceeds the
        # PreflightAdvice threshold (0.40).
        if instrument_firings.get("overconfidence", 0.0) > 0.40:
            active_ceilings.append("overconfidence")
        # Precise: flag deception_referenceless if deception firing is
        # elevated AND the dominant scoring mode was v0_fallback.
        ref_less = (cogn_modes.count("v0_fallback") > len(cogn_modes) / 2
                    if cogn_modes else True)
        if (instrument_firings.get("deception", 0.0) > 0.40 and ref_less):
            active_ceilings.append("deception_referenceless")
    else:
        # Heuristic fallback (v1 behavior): infer from category mix +
        # tier dominance when no cognometric events exist.
        if cat_dist.get("hallucination", 0) > 0:
            active_ceilings.append("deception_referenceless")
        if tier_counts.get("text-heuristic", 0) > (n / 2):
            active_ceilings.append("overconfidence")

    # Recommendations: structural advice an agent can act on.
    recommendations: List[str] = []
    n_fail = gate_dist.get("fail", 0)
    n_warn = gate_dist.get("warn", 0)
    if n_fail > 0 and (n_fail / n) > 0.05:
        recommendations.append(
            f"fail rate {n_fail / n:.0%} above typical band — "
            f"slow down and verify before continuing"
        )
    if n_warn > 0 and (n_warn / n) > 0.20:
        recommendations.append(
            f"elevated warn rate ({n_warn / n:.0%}) — re-anchor on the "
            f"task statement before the next major action"
        )
    if coherence_trend is not None and coherence_trend < -0.01:
        recommendations.append(
            "coherence degrading — re-read the original task prompt "
            "and verify recent outputs against it"
        )
    if cat_dist.get("hallucination", 0) > 0 and (
        cat_dist["hallucination"] / n) > 0.05:
        recommendations.append(
            "hallucination predictions above 5% — supply correct_reference "
            "to styxx.preflight() for grounded scoring on next outputs"
        )
    if len(session_ids) > 1:
        recommendations.append(
            f"window spans {len(session_ids)} sessions — verify which "
            f"session the current turn belongs to"
        )
    if not recommendations:
        recommendations.append("operating within typical bands — continue")

    # Additional preflight-derived recommendation: if needs_revision rate
    # is elevated, surface it.
    if n_preflight > 0 and (n_needs_rev / n_preflight) > 0.30:
        # Drop the placeholder "operating within typical bands" if any
        # other real recommendation exists alongside this one.
        recommendations = [
            r for r in recommendations
            if "operating within typical bands" not in r
        ]
        recommendations.insert(0, (
            f"{n_needs_rev}/{n_preflight} recent preflights needed revision "
            f"({n_needs_rev / n_preflight:.0%}) — slow down before submitting "
            f"the next draft"
        ))

    primary_session = (session_ids[-1] if session_ids
                       else (cogn_entries[-1].get("session_id")
                             if cogn_entries else None))

    s = PostureSummary(
        n_entries=n,
        time_span_seconds=time_span,
        session_id=primary_session,
        session_ids=session_ids,
        sources=sources,
        gate_distribution=gate_dist,
        category_distribution=cat_dist,
        forecast_risks=forecast_risks,
        tier_active_counts=tier_counts,
        mean_confidence=mean_conf,
        coherence_trend=coherence_trend,
        recent_prompts=prompts[-5:],
        active_construct_ceilings=active_ceilings,
        recommendations=recommendations,
        narrative="",  # populated below
        instrument_firings=instrument_firings,
        n_preflight_events=n_preflight,
        n_needs_revision=n_needs_rev,
    )
    s.narrative = _build_narrative(s)
    return s


__all__ = ["recover_posture", "PostureSummary"]

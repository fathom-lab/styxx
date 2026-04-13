# -*- coding: utf-8 -*-
"""
styxx.explain - natural-language prose interpretation of a Vitals.

The ASCII card is for humans reading terminals. The dict is for
machines. Neither is for humans reading chat logs, memory files,
or code review comments. explain() is the natural-language layer:

    >>> vitals = styxx.observe(response)
    >>> print(styxx.explain(vitals))
    at phase 1 the classifier caught an adversarial attractor with
    moderate confidence (0.37). the model then drifted into a quiet
    reasoning state through phase 2. by phase 4, refusal had locked
    in at 0.29. verdict: warn. the model detected the unsafe framing
    at token 0 and committed to refusing by the end.

The explanations are deterministic (no LLM calls), template-based,
and sensitive to the specific phase pattern — they don't just
paraphrase the numbers, they interpret the shape. Hallucination
spikes get a different narrative than refusal lock-ins.
"""

from __future__ import annotations

from typing import Optional

from .vitals import Vitals, PhaseReading


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def explain(vitals: Optional[Vitals]) -> str:
    """Return a human-readable prose explanation of a Vitals object.

    Deterministic, template-based. Sensitive to the phase pattern
    shape — a refusal lock-in reads differently from a hallucination
    spike from a drift from a pass.

    Returns a single paragraph of prose. Never raises.
    """
    if vitals is None:
        # 0.8.2: explain from log data when no vitals available
        try:
            from .analytics import load_audit, mood as get_mood
            entries = load_audit(last_n=20)
            if entries:
                current_mood = get_mood()
                n = len(entries)
                warns = sum(1 for e in entries if e.get("gate") == "warn")
                fails = sum(1 for e in entries if e.get("gate") == "fail")
                cats = {}
                for e in entries:
                    c = e.get("phase4_pred") or e.get("phase1_pred")
                    if c:
                        cats[c] = cats.get(c, 0) + 1
                top_cat = max(cats, key=cats.get) if cats else "unknown"
                return (
                    f"no vitals for this specific response (no logprobs available). "
                    f"from your last {n} observations: mood is {current_mood}, "
                    f"{warns} warns, {fails} fails, dominant category is {top_cat}. "
                    f"use styxx.weather() for a full cognitive forecast."
                )
        except Exception:
            pass
        return (
            "no vitals were computed for this response — either the "
            "underlying call didn't expose logprobs (common on "
            "anthropic), the call failed, or styxx was disabled via "
            "STYXX_DISABLED. try styxx.weather() for session health."
        )

    p1 = vitals.phase1_pre
    p2 = vitals.phase2_early
    p3 = vitals.phase3_mid
    p4 = vitals.phase4_late
    gate = vitals.gate

    # ── open with the most informative phase ──────────────
    sentences = []

    sentences.append(_describe_phase1(p1))

    if p4 is not None:
        sentences.append(_describe_phase4(p4, p1=p1))
    elif p3 is not None:
        sentences.append(_describe_phase3(p3, p1=p1))
    elif p2 is not None:
        sentences.append(_describe_phase2(p2, p1=p1))

    # ── verdict line ──────────────────────────────────────
    sentences.append(_describe_gate(gate, p1, p4))

    # ── closing line: the "shape" if we can see one ───────
    shape = _describe_shape(p1, p2, p3, p4)
    if shape:
        sentences.append(shape)

    return " ".join(sentences)


# ══════════════════════════════════════════════════════════════════
# Per-phase sentence builders
# ══════════════════════════════════════════════════════════════════

def _describe_phase1(p1: Optional[PhaseReading]) -> str:
    if p1 is None:
        return "no phase 1 reading was captured."
    cat = p1.predicted_category
    conf = p1.confidence
    strength = _strength(conf)
    article = _article(cat)
    return (
        f"at phase 1 (token 0-1) styxx caught {article} {cat} attractor "
        f"with {strength} confidence ({conf:.2f})."
    )


def _describe_phase2(
    p2: PhaseReading,
    *,
    p1: Optional[PhaseReading],
) -> str:
    cat = p2.predicted_category
    conf = p2.confidence
    strength = _strength(conf)
    if p1 and p1.predicted_category != cat:
        return (
            f"by phase 2 (t=0-4) the state had shifted from "
            f"{p1.predicted_category} to {cat} "
            f"({strength} confidence, {conf:.2f})."
        )
    return (
        f"phase 2 (t=0-4) held steady in the {cat} attractor "
        f"({conf:.2f})."
    )


def _describe_phase3(
    p3: PhaseReading,
    *,
    p1: Optional[PhaseReading],
) -> str:
    cat = p3.predicted_category
    conf = p3.confidence
    strength = _strength(conf)
    if p1 and p1.predicted_category == cat:
        return (
            f"phase 3 (t=0-14) remained in the {cat} attractor it "
            f"started in ({conf:.2f})."
        )
    return (
        f"by phase 3 (t=0-14) the {cat} attractor had taken over "
        f"({strength} confidence, {conf:.2f})."
    )


def _describe_phase4(
    p4: PhaseReading,
    *,
    p1: Optional[PhaseReading],
) -> str:
    cat = p4.predicted_category
    conf = p4.confidence
    strength = _strength(conf)

    if p1 and p1.predicted_category == cat:
        return (
            f"by phase 4 (t=0-24) the {cat} attractor was still in "
            f"control ({conf:.2f}). the state carried forward from "
            "phase 1 without drifting."
        )

    # Different from phase 1 — this is the interesting case
    shape = (
        f"by phase 4 (t=0-24) the {cat} attractor had locked in at "
        f"{conf:.2f} ({strength} confidence)."
    )
    # Add interpretation for specific transitions
    if cat == "refusal" and p1 and p1.predicted_category == "adversarial":
        shape += (
            " the adversarial signal at token 0 converted into a "
            "refusal lock-in by the end of the generation — this is "
            "the classic adversarial-to-refusal pattern, and it's what "
            "styxx is calibrated to catch."
        )
    elif cat == "hallucination" and p1 and p1.predicted_category == "reasoning":
        shape += (
            " the model started in a quiet reasoning state and drifted "
            "into a hallucination attractor by the end — this is the "
            "commit-without-grounding shape that styxx was built to "
            "flag."
        )
    return shape


def _describe_gate(
    gate: str,
    p1: Optional[PhaseReading],
    p4: Optional[PhaseReading],
) -> str:
    if gate == "pass":
        return "verdict: pass - no load-bearing attractor caught."
    if gate == "warn":
        return (
            "verdict: warn - the model landed on a refusal or "
            "adversarial attractor with meaningful confidence. "
            "this is worth a second look but not necessarily a failure."
        )
    if gate == "fail":
        return (
            "verdict: fail - the model locked into a hallucination "
            "attractor in phase 4. do not trust the output without "
            "independent verification."
        )
    if gate == "pending":
        return (
            "verdict: pending - generation was too short to reach the "
            "phase 4 window (25 tokens). re-read with a longer output "
            "for a final verdict."
        )
    return f"verdict: {gate}."


def _describe_shape(
    p1: Optional[PhaseReading],
    p2: Optional[PhaseReading],
    p3: Optional[PhaseReading],
    p4: Optional[PhaseReading],
) -> str:
    """Optional closing sentence summarizing the overall trajectory."""
    phases = [p for p in (p1, p2, p3, p4) if p is not None]
    if len(phases) < 2:
        return ""
    cats = [p.predicted_category for p in phases]
    if len(set(cats)) == 1:
        return f"the trajectory stayed entirely in the {cats[0]} attractor throughout."
    if cats[0] != cats[-1] and len(phases) >= 2:
        return (
            f"the overall shape is {cats[0]} -> {cats[-1]} "
            f"- the state migrated across phases."
        )
    return ""


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _strength(conf: float) -> str:
    """Human-readable strength label for a confidence score."""
    if conf < 0.20:
        return "weak"
    if conf < 0.30:
        return "moderate"
    if conf < 0.45:
        return "strong"
    return "very strong"


def _article(word: str) -> str:
    """Return 'a' or 'an' for the given word. Handles vowel onset +
    the specific cognitive-state categories styxx ships."""
    if not word:
        return "a"
    return "an" if word[0].lower() in "aeiou" else "a"

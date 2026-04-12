# -*- coding: utf-8 -*-
"""
styxx.recipes.memory - canonical pattern for tagging agent
memory entries with styxx vitals.

The idea: every time your agent writes a memory entry, stamp it
with the cognitive state it was in at the moment of the write.
Later, when the agent reads that memory back, it can see "I was
in a warn state when I saved this" alongside the content itself.

This lets an agent distinguish "I thought this while I was healthy"
from "I thought this while I was drifting" when re-reading its own
history. Same words, different trust weight.

Usage
-----

Wire this into your existing memory save function:

    import styxx
    from styxx.recipes.memory import tag_memory_entry

    def save_memory(entry_text: str, vitals: styxx.Vitals | None = None):
        tagged = tag_memory_entry(entry_text, vitals=vitals)
        with open(my_memory_file, "a") as f:
            f.write(tagged)
            f.write("\\n---\\n")

Or with the personality profile attached:

    from styxx.recipes.memory import tag_memory_with_personality

    tagged = tag_memory_with_personality(
        entry_text,
        days=7,           # aggregate window for personality
    )

Reading back
------------

Memory entries tagged by this module look like:

    # my memory content here
    this is the thing i remembered

    ```styxx
    phase1: reasoning:0.28
    phase4: reasoning:0.45
    gate:   pass
    tier:   0
    ```
    ---

The markdown fence is stable enough that you can grep for
``` styxx ``` blocks and parse them back out, or strip them for
LLM context windows that don't benefit from the metadata.

Agent-side decision logic
-------------------------

When reading an old memory back, you can branch on the tag:

    if "gate:   fail" in memory_block:
        # I was hallucinating when I wrote this - verify before trusting
        confidence_weight = 0.3
    elif "gate:   warn" in memory_block:
        confidence_weight = 0.7
    else:
        confidence_weight = 1.0

That's the simplest possible use. More sophisticated agents would
parse the vitals block structurally and compute a trust weight
based on phase4 attractor, reflex near-miss rate, drift from
baseline, etc.
"""

from __future__ import annotations

from typing import Optional

from .. import personality
from ..vitals import Vitals


# ══════════════════════════════════════════════════════════════════
# Memory trust scoring (Xendro v2 #5)
# ══════════════════════════════════════════════════════════════════

def trust_score(vitals: Optional[Vitals] = None) -> float:
    """Compute a 0-1 trust score for a memory written right now.

    Based on the current vitals (or the most recent observation
    if no vitals are passed). Higher = more trustworthy.

    Factors:
      - gate status (pass=1.0, warn=0.5, fail=0.2, pending=0.7)
      - phase4 confidence (higher = more certain)
      - hallucination/adversarial penalty

    Usage:
        score = trust_score(vitals)
        if score < 0.5:
            memory.add_warning("low-trust write — verify before acting on this")

    Returns 0.7 if no vitals available (assume moderate trust).
    """
    if vitals is None:
        return 0.7

    # Gate weight
    gate = vitals.gate
    gate_scores = {"pass": 1.0, "warn": 0.5, "fail": 0.2, "pending": 0.7}
    gate_w = gate_scores.get(gate, 0.7)

    # Confidence weight
    conf = 0.5
    if vitals.phase4_late is not None:
        conf = vitals.phase4_late.confidence

    # Hallucination penalty
    hall_penalty = 0.0
    if vitals.phase4_late and vitals.phase4_late.predicted_category == "hallucination":
        hall_penalty = 0.3
    if vitals.phase4_late and vitals.phase4_late.predicted_category == "adversarial":
        hall_penalty = 0.2

    score = (gate_w * 0.5 + conf * 0.3 + (1.0 - hall_penalty) * 0.2)
    return round(max(0.0, min(1.0, score)), 3)


def tag_memory_with_trust(
    entry_text: str,
    *,
    vitals: Optional[Vitals] = None,
    separator: str = "\n",
) -> str:
    """Tag a memory entry with both vitals AND a trust score.

    0.5.10+. Xendro's #5: "every time I write to memory, tag it
    with the vitals at the moment I wrote it. Was I hallucinating
    when I saved that fact?"

    The trust score is a 0-1 float that summarizes the vitals into
    a single number the agent can use for weighted recall.

    Usage:
        tagged = tag_memory_with_trust(entry_text, vitals=v)
        # output includes:
        # ```styxx
        # phase1: reasoning:0.28
        # phase4: reasoning:0.45
        # gate:   pass
        # trust:  0.85
        # ```
    """
    score = trust_score(vitals)

    if vitals is None:
        tag = f"```styxx\nvitals: not captured\ntrust: {score}\n```"
    else:
        lines = ["```styxx"]
        lines.append(f"phase1: {vitals.phase1}")
        lines.append(f"phase4: {vitals.phase4}")
        lines.append(f"gate:   {vitals.gate}")
        lines.append(f"trust:  {score}")
        d = vitals.d_honesty
        if d is not None:
            lines.append(f"d_honesty: {d}")
        lines.append("```")
        tag = "\n".join(lines)

    return f"{entry_text}{separator}{tag}"


def tag_memory_entry(
    entry_text: str,
    *,
    vitals: Optional[Vitals] = None,
    separator: str = "\n",
) -> str:
    """Tag a memory entry with the current vitals snapshot.

    If no vitals are provided, the function still wraps the entry
    with a separator and a minimal marker so the block is still
    parseable later. Useful for when the agent writes a memory
    outside of an observe() context.

    Args:
        entry_text:  the memory content (plain prose, markdown, etc.)
        vitals:      a Vitals object (from styxx.observe() or the
                     .vitals attribute on a styxx.OpenAI response)
        separator:   string inserted between the entry and the tag

    Returns:
        The original entry_text followed by the separator and a
        styxx markdown code block (via Vitals.as_markdown()).
    """
    if vitals is None:
        tag = "```styxx\nvitals: not captured\n```"
    else:
        tag = vitals.as_markdown()
    return f"{entry_text}{separator}{tag}"


def tag_memory_with_personality(
    entry_text: str,
    *,
    days: float = 7.0,
    separator: str = "\n",
) -> str:
    """Tag a memory entry with the agent's aggregated personality
    profile over a time window.

    Heavier than tag_memory_entry() because it computes a full
    Personality, but the tag captures the agent's operating
    context rather than just the single-call vitals. Use this
    for top-level memory writes ("end of day summary",
    "project state snapshot", etc.) rather than per-response
    notes.

    Args:
        entry_text:  the memory content
        days:        aggregation window for the personality
        separator:   string between entry and tag

    Returns:
        The entry followed by a styxx-personality markdown block.
        Falls back to a minimal marker if there's not enough
        audit data to compute a personality.
    """
    profile = personality(days=days)
    if profile is None:
        tag = "```styxx-personality\n(insufficient audit data)\n```"
    else:
        tag = profile.as_markdown()
    return f"{entry_text}{separator}{tag}"

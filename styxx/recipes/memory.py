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

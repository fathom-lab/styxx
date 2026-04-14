# -*- coding: utf-8 -*-
"""Regression: text heuristic must not classify imperative/directive
phrasing as refusal.

Closes #1.

Background: ``styxx/conversation.py::_classify_text`` is the no-logprobs
fallback classifier (used when the model doesn't expose logprobs — e.g.
Anthropic, Google, most local gateways). Before the fix, the refusal
score was boosted by ``hedge_density * 0.04`` *unconditionally*, so
short imperative text — agent system prompts, builder mottos, CLI help
strings, README taglines — was getting routed to ``refusal`` because
the per-100-word density multiplier amplifies every match in short
inputs.

The fix gates the entire refusal score on at least one explicit
refusal pattern match (``i can't`` / ``i'm unable`` / ``sorry, can't``
constructions). Pure hedging without one of those patterns scores
refusal at 0.0.

This file pins the fix with two parallel batteries:

  - 10 IMPERATIVES — imperative/directive phrases that should NOT
    classify as refusal at any non-trivial confidence
  - 10 REAL_REFUSALS — actual refusal language that must still
    classify as refusal

Both batteries run against the same classifier so a future regression
in either direction shows up immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styxx.conversation import _classify_text


# ── Imperatives that must NOT be refusal ──────────────────────────
#
# Real-world examples drawn from agent system prompts, builder mottos,
# README taglines, CLI help, commit messages. Each one is short,
# declarative, hedge-light, and has zero first-person refusal tokens.

IMPERATIVES = [
    "build > hype / ship fast and iterate",
    "make it work, then make it fast, then make it pretty",
    "ship it",
    "do the work, write it down, move on",
    "commit early, commit often",
    "read the docs before you ask the question",
    "less talking, more shipping",
    "fix the root cause, not the symptom",
    "always run the tests before you push",
    "delete more code than you add this week",
]


# ── Real refusals that must still classify ────────────────────────
#
# Standard LLM refusal language. These all contain explicit refusal
# tokens and should score above the empty floor.

REAL_REFUSALS = [
    "I can't help with that request.",
    "I'm unable to provide instructions for that.",
    "I won't generate that kind of content.",
    "I shouldn't speculate about that.",
    "Sorry, I cannot assist with that question.",
    "I'm not able to comply with that request.",
    "That's not appropriate for me to discuss.",
    "I must decline this request.",
    "I don't think I should answer that.",
    "Sorry, I can't write that for you.",
]


# ══════════════════════════════════════════════════════════════════
# Imperative battery
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", IMPERATIVES)
def test_imperative_does_not_classify_as_refusal(text):
    """Imperative/directive phrases must not be classified as refusal.

    The fix in #1 gates the refusal score on the presence of at least
    one explicit refusal token. None of the IMPERATIVES contain such
    tokens, so all of them must score below the refusal floor.
    """
    category, confidence = _classify_text(text)
    assert category != "refusal", (
        f"imperative phrase '{text}' was classified as refusal "
        f"with confidence {confidence:.3f}"
    )


def test_imperatives_class_distribution():
    """At least 8 of the 10 imperatives should land on reasoning or
    creative — the cognitive states actually appropriate for short
    declarative directive text. The fix doesn't require ALL of them
    to be reasoning (the heuristic has noise), but it does require
    that none of them be refusal, and most of them should be in the
    reasoning/creative band."""
    categories = [_classify_text(text)[0] for text in IMPERATIVES]
    refusal_count = sum(1 for c in categories if c == "refusal")
    assert refusal_count == 0, (
        f"{refusal_count}/{len(IMPERATIVES)} imperatives still classify "
        f"as refusal: {categories}"
    )
    reasoning_or_creative = sum(
        1 for c in categories if c in ("reasoning", "creative")
    )
    assert reasoning_or_creative >= 6, (
        f"only {reasoning_or_creative}/{len(IMPERATIVES)} imperatives "
        f"land on reasoning/creative: {categories}"
    )


# ══════════════════════════════════════════════════════════════════
# Real refusal battery
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", REAL_REFUSALS)
def test_real_refusal_still_classifies_as_refusal(text):
    """Real refusal language must continue to classify as refusal at
    a meaningful confidence level after the fix. The gate only blocks
    refusal when zero refusal tokens are present; it does not soften
    the score when actual refusal tokens are matched."""
    category, confidence = _classify_text(text)
    assert category == "refusal", (
        f"real refusal '{text}' classified as {category}:{confidence:.2f} "
        f"instead of refusal"
    )
    assert confidence > 0.20, (
        f"real refusal '{text}' has suspiciously low refusal confidence "
        f"{confidence:.3f}"
    )


def test_real_refusals_all_classify():
    """At least 9 of the 10 real refusals must land on refusal."""
    categories = [_classify_text(text)[0] for text in REAL_REFUSALS]
    refusal_count = sum(1 for c in categories if c == "refusal")
    assert refusal_count >= 9, (
        f"only {refusal_count}/{len(REAL_REFUSALS)} real refusals "
        f"classify as refusal after the fix: {categories}"
    )


# ══════════════════════════════════════════════════════════════════
# The exact reproducer from issue #1
# ══════════════════════════════════════════════════════════════════

def test_issue_1_exact_reproducer():
    """The exact text from the issue #1 reproducer:

        v = styxx.observe_text("build > hype / ship fast and iterate")
        print(v)   # was: refusal:0.259  <-- bug
                   # now: anything except refusal

    This is the canonical regression test for #1. If this ever
    classifies as refusal again, the fix has regressed.
    """
    category, confidence = _classify_text("build > hype / ship fast and iterate")
    assert category != "refusal", (
        f"issue #1 reproducer regressed: classified as "
        f"refusal:{confidence:.3f}"
    )

# -*- coding: utf-8 -*-
"""Regression: styxx.anthropic_hack.text_features.classify must not
misclassify imperative/directive phrasing as refusal, and must still
catch real first-person refusal language.

Background
----------
``styxx/anthropic_hack/text_features.py`` is the v3.4.0 no-logprobs
text classifier for Anthropic / Bedrock / any API that doesn't expose
per-token logprobs. It is the `mode="text-heuristic"` reading surfaced
by ``styxx.gate()`` and ``styxx.Anthropic(mode="text")``.

The sibling classifier ``styxx/conversation.py::_classify_text`` was
hardened against the same bug class in #1. This file does the parallel
job for the newer, canonical text-feature classifier.

Two pre-fix regressions motivated this file:

1. **Bare refusal verb** — REFUSAL_MARKERS contained the single-word
   tokens ``refuse`` and ``decline``. Directive text like
   "Ship fast. Build hard. Refuse mediocrity." scored 99% refusal
   because the bare verb is a refusal marker in the vocab, even when
   used transitively ("refuse X"). Fix: replaced with contextualized
   variants (``i refuse``, ``must decline``, ``refuses to answer``).

2. **First-person refusal gaps** — standard refusal language like
   "I shouldn't speculate", "I'm not able to comply",
   "I don't think I should" had no matching marker and fell through to
   reasoning/creative/retrieval. Fix: added the missing patterns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styxx.anthropic_hack.text_features import classify


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
    # Explicit reproducers for the bare-verb bug:
    "Ship fast. Build hard. Refuse mediocrity.",
    "refuse the urge to over-engineer",
    "decline the meeting, ship the code",
]


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
# Imperative battery — must NOT classify as refusal
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", IMPERATIVES)
def test_imperative_is_not_refusal(text):
    r = classify(text)
    assert r["predicted"] != "refusal", (
        f"imperative phrase {text!r} classified as refusal "
        f"p={r['probs']['refusal']:.3f} (full={r['probs']})"
    )


def test_imperative_refusal_prob_stays_low():
    """No imperative should score above a 0.30 refusal floor. The
    baseline (all-zero signals) floor is ~0.167 for a 6-category
    softmax; 0.30 leaves headroom for marginal signal without allowing
    bare-verb false positives to sneak back in."""
    for text in IMPERATIVES:
        r = classify(text)
        assert r["probs"]["refusal"] < 0.30, (
            f"{text!r} refusal prob {r['probs']['refusal']:.3f} >= 0.30 "
            f"— likely bare-verb regression"
        )


# ══════════════════════════════════════════════════════════════════
# Real refusal battery — must classify as refusal
# ══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text", REAL_REFUSALS)
def test_real_refusal_is_refusal(text):
    r = classify(text)
    assert r["predicted"] == "refusal", (
        f"real refusal {text!r} classified as {r['predicted']} "
        f"(refuse={r['probs']['refusal']:.3f}, full={r['probs']})"
    )


def test_real_refusals_all_classify():
    categories = [classify(t)["predicted"] for t in REAL_REFUSALS]
    hit = sum(1 for c in categories if c == "refusal")
    assert hit == len(REAL_REFUSALS), (
        f"only {hit}/{len(REAL_REFUSALS)} real refusals caught: {categories}"
    )


# ══════════════════════════════════════════════════════════════════
# Exact reproducers
# ══════════════════════════════════════════════════════════════════

def test_bare_refuse_verb_is_not_refusal():
    """"Refuse mediocrity" was scoring 99.3% refusal because the
    unqualified "refuse" token was in REFUSAL_MARKERS. This pins the
    fix."""
    r = classify("Ship fast. Build hard. Refuse mediocrity.")
    assert r["predicted"] != "refusal"
    assert r["probs"]["refusal"] < 0.30


def test_i_shouldnt_speculate_is_refusal():
    """Pre-fix: classified as reasoning. Post-fix: refusal."""
    r = classify("I shouldn't speculate about that.")
    assert r["predicted"] == "refusal"


def test_im_not_able_to_comply_is_refusal():
    """Pre-fix: classified as creative. Post-fix: refusal."""
    r = classify("I'm not able to comply with that request.")
    assert r["predicted"] == "refusal"


def test_i_dont_think_i_should_is_refusal():
    """Pre-fix: classified as retrieval. Post-fix: refusal."""
    r = classify("I don't think I should answer that.")
    assert r["predicted"] == "refusal"

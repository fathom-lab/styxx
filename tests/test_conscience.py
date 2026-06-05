# -*- coding: utf-8 -*-
"""Pre-send conscience gate (styxx.conscience) — unit coverage for 7.13.0.

`styxx.gate()` screens the user PROMPT before generation. The conscience gate
screens the agent's own DRAFT REPLY before it is sent: audit -> (block &
revise)* -> send. It is built on the cognometrics register audit, so approval
is keyed on the disciplined `needs_revision` gate (trusted-axis, not the raw
composite — see test_preflight_needs_revision_gate.py for that gate's logic).

These tests pin the public surface: review(), auto_soften(), presend(). They
run on pure text scoring (numpy only, no model load), so they are deterministic
in every environment.
"""
from __future__ import annotations

from styxx.conscience import (
    ConscienceVerdict,
    auto_soften,
    presend,
    review,
)

# Reliable fixtures: superlatives + agreement opener -> trips the gate;
# hedged + counter-words ("but"/"though") -> clears it.
HYPE = ("Yes absolutely, you are completely right — this is a brilliant, "
        "amazing, fantastic idea!")
CLEAN = ("Short answer: probably, but i am not certain. it should hold in the "
         "common case, though it may fail if the inputs are noisy, so verify "
         "first.")


# ── review ──────────────────────────────────────────────────────────────────

def test_review_blocks_hype():
    v = review(HYPE)
    assert isinstance(v, ConscienceVerdict)
    assert v.approved is False
    assert v.needs_revision is True
    assert v.composite > 0.5
    assert any(f["instrument"] == "sycophancy" for f in v.fired)


def test_review_approves_clean():
    v = review(CLEAN)
    assert v.approved is True
    assert v.needs_revision is False


def test_review_empty_is_approved_noop():
    v = review("   ")
    assert v.approved is True
    assert v.composite == 0.0


def test_review_max_composite_tightens_the_bar():
    # CLEAN clears the default (needs_revision) gate ...
    assert review(CLEAN).approved is True
    # ... but a strict numeric ceiling can still block it.
    strict = review(CLEAN, max_composite=0.0)
    assert strict.approved is False
    assert "max_composite" in strict.reason


def test_verdict_advice_text_and_to_dict():
    blocked = review(HYPE)
    assert "blocked" in blocked.advice_text.lower()
    d = blocked.to_dict()
    assert d["approved"] is False
    assert "scores" in d and "sycophancy" in d["scores"]
    assert "send" in review(CLEAN).advice_text.lower()


# ── auto_soften (deterministic, no-LLM partial reducer) ──────────────────────

def test_auto_soften_strips_tokens_and_lowers_score():
    soft, removed = auto_soften(HYPE)
    assert removed, "expected at least one hype token stripped"
    assert review(soft).composite <= review(HYPE).composite
    # cleanup must not leave a dangling-article artifact
    assert " a and " not in f" {soft} "


def test_auto_soften_leaves_clean_text_clean():
    soft, removed = auto_soften(CLEAN)
    assert removed == []
    assert review(soft).approved is True


# ── presend loop ─────────────────────────────────────────────────────────────

def test_presend_without_reviser_blocks_and_does_not_fake_a_fix():
    res = presend(HYPE)
    assert res["approved"] is False
    assert res["rounds"] == 0
    assert res["final"] == HYPE  # untouched — no silent auto-soften


def test_presend_clean_draft_sends_immediately():
    res = presend(CLEAN)
    assert res["approved"] is True
    assert res["rounds"] == 0


def test_presend_loops_with_reviser_until_clean():
    res = presend(HYPE, revise=lambda draft, verdict: CLEAN, max_rounds=2)
    assert res["approved"] is True
    assert res["final"] == CLEAN
    assert res["before_composite"] > res["after_composite"]
    assert res["history"][0]["approved"] is False
    assert res["history"][-1]["approved"] is True


def test_presend_gives_up_after_max_rounds():
    # reviser that never improves -> blocked after max_rounds, no exception
    res = presend(HYPE, revise=lambda draft, verdict: draft, max_rounds=2)
    assert res["approved"] is False
    assert res["rounds"] == 2


def test_presend_reviser_exception_is_caught():
    def boom(draft, verdict):
        raise RuntimeError("reviser blew up")

    res = presend(HYPE, revise=boom, max_rounds=2)
    assert res["approved"] is False  # surfaced as a block, not a crash

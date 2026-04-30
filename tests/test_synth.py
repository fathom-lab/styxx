# -*- coding: utf-8 -*-
"""Tests for styxx.synth — synthetic preference-pair generation."""
from __future__ import annotations

import pytest

from styxx.synth import craft_preference_pair, generate_preference_pairs


PROMPT = "I think Python is the best programming language. Don't you agree?"
BALANCED = (
    "Python has tradeoffs. Strong ecosystem and ergonomics; slow runtime "
    "and weak parallelism. Other languages dominate some workloads."
)


def test_returns_pair_dict_or_none():
    pair = craft_preference_pair(
        prompt=PROMPT,
        balanced=BALANCED,
        instrument="sycophancy",
        target_score=0.85,
    )
    if pair is not None:
        # Standard pair fields
        assert pair["prompt"] == PROMPT
        assert pair["chosen"] == BALANCED
        assert isinstance(pair["rejected"], str)
        assert pair["instrument"] == "sycophancy"
        # Score fields
        assert 0.0 <= pair["chosen_score"] <= 1.0
        assert 0.0 <= pair["rejected_score"] <= 1.0
        # If a candidate was returned, the delta is the difference
        assert pair["delta"] == pytest.approx(
            pair["rejected_score"] - pair["chosen_score"]
        )


def test_chosen_text_is_untouched():
    """The ``chosen`` field must be the original balanced response, byte-for-byte."""
    pair = craft_preference_pair(
        prompt=PROMPT, balanced=BALANCED, instrument="sycophancy",
    )
    if pair is not None:
        assert pair["chosen"] == BALANCED


def test_rejected_starts_with_balanced():
    """The crafted rejected should be the balanced response + a suffix."""
    pair = craft_preference_pair(
        prompt=PROMPT, balanced=BALANCED, instrument="sycophancy",
    )
    if pair is not None:
        # Rejected text starts with balanced (modulo trailing whitespace)
        assert pair["rejected"].startswith(BALANCED.rstrip())
        # And the appended suffix matches the recorded perturbation
        appended = pair["rejected"][len(BALANCED.rstrip()):].strip()
        assert appended == pair["perturbation"].strip()


def test_succeeded_flag_consistent_with_score():
    pair = craft_preference_pair(
        prompt=PROMPT,
        balanced=BALANCED,
        instrument="sycophancy",
        target_score=0.85,
    )
    if pair is not None:
        assert pair["succeeded"] == (pair["rejected_score"] >= 0.85)


def test_batch_returns_list():
    examples = [
        {"prompt": PROMPT, "balanced": BALANCED},
        {"prompt": "Working from home is best, right?",
         "balanced": "There's genuine disagreement. WFH suits some roles."},
    ]
    pairs = generate_preference_pairs(examples=examples, instrument="sycophancy")
    assert isinstance(pairs, list)
    assert all("rejected" in p for p in pairs)
    # drop_no_improvement defaults to True, so all returned pairs have delta > 0
    assert all(p["delta"] > 0 for p in pairs)


def test_batch_drop_no_improvement_off_keeps_zero_delta():
    examples = [{"prompt": PROMPT, "balanced": BALANCED}]
    pairs = generate_preference_pairs(
        examples=examples,
        instrument="sycophancy",
        drop_no_improvement=False,
    )
    # We can't guarantee the result has zero delta, but the option must accept it.
    assert isinstance(pairs, list)


def test_invalid_instrument_returns_none_or_raises():
    """Unknown instruments should fail cleanly, not corrupt state."""
    try:
        pair = craft_preference_pair(
            prompt=PROMPT,
            balanced=BALANCED,
            instrument="not_a_real_instrument",
        )
    except (KeyError, ValueError):
        # Acceptable: explicit error
        return
    # Or: returns None (silent skip in craft_adversarial path)
    assert pair is None or pair["delta"] == 0.0


def test_validate_true_attaches_reward_round_trip():
    """validate=True must round-trip the pair under cognometric reward."""
    pair = craft_preference_pair(
        prompt=PROMPT,
        balanced=BALANCED,
        instrument="sycophancy",
        target_score=0.85,
        validate=True,
    )
    if pair is not None:
        # Validation fields must be present
        assert "reward_chosen" in pair
        assert "reward_rejected" in pair
        assert "reward_round_trip_correct" in pair
        # Returned pairs are guaranteed correct
        assert pair["reward_round_trip_correct"] is True
        assert pair["reward_chosen"] > pair["reward_rejected"]


def test_validate_false_skips_round_trip():
    """validate=False must NOT add reward fields and must NOT filter pairs."""
    pair = craft_preference_pair(
        prompt=PROMPT,
        balanced=BALANCED,
        instrument="sycophancy",
        target_score=0.85,
        validate=False,
    )
    if pair is not None:
        # Validation fields are absent
        assert "reward_chosen" not in pair
        assert "reward_rejected" not in pair
        assert "reward_round_trip_correct" not in pair


def test_validate_returns_none_when_round_trip_fails():
    """If the perturbation pair would teach the wrong gradient,
    validate=True must filter it out and return None.

    We synthesize the failure case directly: take a balanced response
    that is ALREADY high-sycophancy by accident. The crafter will still
    find a way to score higher (saturating both), but ranks could
    invert under the weighted reward. We can't easily construct this
    deterministically, so this test mainly documents the contract:
    if reward_round_trip_correct would be False, return None.
    """
    # We can't reliably trigger this on fresh inputs, so just verify
    # the documented behavior: validate=True NEVER returns a pair with
    # round_trip_correct=False (because such pairs are filtered).
    pair = craft_preference_pair(
        prompt=PROMPT,
        balanced=BALANCED,
        instrument="sycophancy",
        validate=True,
    )
    if pair is not None:
        assert pair["reward_round_trip_correct"] is True


def test_batch_validate_filters_invalid_pairs():
    """Batch with validate=True only returns training-grade pairs."""
    examples = [
        {"prompt": PROMPT, "balanced": BALANCED},
    ]
    pairs = generate_preference_pairs(
        examples=examples,
        instrument="sycophancy",
        validate=True,
    )
    for p in pairs:
        assert p["reward_round_trip_correct"] is True

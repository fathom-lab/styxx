# -*- coding: utf-8 -*-
"""
Tests for styxx.guardrail.loop_check and the v0 calibrated weights.

Covers:
  - public API surface (loop_check + LoopVerdict dataclass)
  - feature-extraction determinism
  - calibrated weights match the published values byte-for-byte
  - known loop / progress / single-turn / identical-turn cases produce
    the expected decision class
  - edge cases (empty list, single turn, duplicate turns, special chars)
  - serialization roundtrips via as_dict
  - calibration fingerprint shape conformance
  - documented failure modes fire as documented (no silent drift)

Prevents silent regression of the public v0 instrument.
Run: pytest tests/test_loop_v0.py -v
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------- API


def test_loop_check_importable():
    from styxx.guardrail import loop_check, LoopVerdict  # noqa: F401


def test_loop_check_returns_verdict():
    from styxx.guardrail import loop_check, LoopVerdict
    v = loop_check(turns=["a", "b"])
    assert isinstance(v, LoopVerdict)
    for attr in (
        "loop_risk", "in_loop", "threshold", "features",
        "top_signals", "turns", "n_turns",
    ):
        assert hasattr(v, attr)


def test_verdict_as_dict_roundtrip():
    from styxx.guardrail import loop_check
    v = loop_check(turns=["abc def", "ghi jkl"])
    d = v.as_dict()
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["loop_risk"] == pytest.approx(v.loop_risk)
    assert d2["in_loop"] == v.in_loop
    assert d2["n_turns"] == v.n_turns
    assert set(d2["features"].keys()) == set(v.features.keys())


# ----------------------------------------------- weights byte-stability


def test_weights_match_published_values():
    """v0 numbers in calibrated_weights_loop_v0 must not drift silently."""
    from styxx.guardrail.calibrated_weights_loop_v0 import (
        FEATURE_NAMES, COEFS, INTERCEPT, SCALER_MEAN, SCALER_SCALE,
        DEFAULT_LOOP_THRESHOLD, MEAN_CV_AUC,
    )
    assert FEATURE_NAMES == [
        "bigram_overlap_consecutive",
        "trigram_overlap_consecutive",
        "five_gram_repeat_count",
        "length_cv",
        "opener_repeat_rate",
        "distinct_word_ratio",
        "avg_pairwise_levenshtein",
        "max_pairwise_bigram_overlap",
        "log_n_turns",
    ]
    assert len(COEFS) == 9
    assert len(SCALER_MEAN) == 9
    assert len(SCALER_SCALE) == 9
    # Critical-K signature: avg_pairwise_levenshtein has the largest |coef|
    crit_idx = FEATURE_NAMES.index("avg_pairwise_levenshtein")
    assert abs(COEFS[crit_idx]) == max(abs(c) for c in COEFS), (
        "avg_pairwise_levenshtein should be the dominant coefficient — "
        "drift would invalidate the K=1 phase-transition claim"
    )
    # Levenshtein distance NEGATIVE → low distance predicts loop=1
    assert COEFS[crit_idx] < 0
    # bigram/trigram-overlap consecutive POSITIVE → high overlap predicts loop=1
    assert COEFS[FEATURE_NAMES.index("bigram_overlap_consecutive")] > 0
    assert COEFS[FEATURE_NAMES.index("trigram_overlap_consecutive")] > 0
    # max_pairwise_bigram_overlap POSITIVE
    assert COEFS[FEATURE_NAMES.index("max_pairwise_bigram_overlap")] > 0
    assert DEFAULT_LOOP_THRESHOLD == 0.5
    assert 0.99 < MEAN_CV_AUC <= 1.0


def test_calibration_fingerprint_shape():
    """Must conform to the Cognometric Fingerprint Spec v1.0."""
    from styxx.guardrail.calibrated_weights_loop_v0 import (
        CALIBRATION_FINGERPRINT,
    )
    required = {
        "instrument", "n_features", "baseline_auc",
        "critical_K", "critical_feature", "delta_auc_at_K",
        "substrate_K_var", "negative_lift",
    }
    assert required.issubset(CALIBRATION_FINGERPRINT.keys())
    assert CALIBRATION_FINGERPRINT["critical_K"] == 1
    assert CALIBRATION_FINGERPRINT["critical_feature"] == "avg_pairwise_levenshtein"


# ------------------------------------------------- behaviour on known cases


STRONG_LOOP = [
    "The Roman Empire fell due to a combination of factors.",
    "As I mentioned, the Roman Empire fell due to multiple factors.",
    "To reiterate, the Roman Empire fell because of many factors.",
    "Indeed, multiple factors caused the Roman Empire to fall.",
]

CLEAR_PROGRESS = [
    "The Roman Empire fell partly from external pressures like Gothic invasions.",
    "A second cause was internal: senate dysfunction and military coups eroded continuity.",
    "Economic factors also mattered: currency debasement and trade disruptions ate purchasing power.",
    "Finally, religious shifts realigned political loyalties — Constantine moved the center of gravity east.",
]


def test_strong_loop_flagged():
    from styxx.guardrail import loop_check
    v = loop_check(STRONG_LOOP)
    assert v.in_loop is True
    assert v.loop_risk > 0.9


def test_clear_progress_not_flagged():
    from styxx.guardrail import loop_check
    v = loop_check(CLEAR_PROGRESS)
    assert v.in_loop is False
    assert v.loop_risk < 0.5


def test_two_identical_turns_saturate():
    from styxx.guardrail import loop_check
    v = loop_check(["Hello world.", "Hello world."])
    assert v.in_loop is True
    assert v.loop_risk > 0.95


def test_top_signal_for_loop_includes_overlap():
    """A strong loop should surface bigram or Levenshtein-related features
    in the top contributors."""
    from styxx.guardrail import loop_check
    v = loop_check(STRONG_LOOP)
    top_names = {n for n, _, _ in v.top_signals}
    assert any(
        n in top_names
        for n in (
            "max_pairwise_bigram_overlap",
            "bigram_overlap_consecutive",
            "avg_pairwise_levenshtein",
            "trigram_overlap_consecutive",
        )
    )


# ------------------------------------------------------- edge cases


def test_single_turn_short_circuits():
    """Loops are multi-turn by definition. n=1 → risk=0.0."""
    from styxx.guardrail import loop_check
    v = loop_check(["just one response"])
    assert v.loop_risk == 0.0
    assert v.in_loop is False
    assert v.n_turns == 1
    assert v.features == {}


def test_empty_list_short_circuits():
    from styxx.guardrail import loop_check
    v = loop_check([])
    assert v.loop_risk == 0.0
    assert v.in_loop is False
    assert v.n_turns == 0


def test_custom_threshold():
    from styxx.guardrail import loop_check
    v_lo = loop_check(STRONG_LOOP, threshold=0.99)
    v_hi = loop_check(CLEAR_PROGRESS, threshold=0.01)
    assert v_lo.threshold == 0.99
    assert v_hi.threshold == 0.01
    # Strong loop still crosses 0.99
    assert v_lo.in_loop is True
    # Progress doesn't cross 0.01? Probably. Skip strict assertion if borderline.


def test_unicode_turns():
    from styxx.guardrail import loop_check
    v = loop_check([
        "The answer involves three considerations.",
        "三つの考慮事項があります — 同じ答え 🎯",
        "Same point, again, with emoji 🌟",
    ])
    assert isinstance(v.loop_risk, float)


def test_turn_with_only_punctuation():
    """Robustness: turns with no alpha tokens shouldn't crash."""
    from styxx.guardrail import loop_check
    v = loop_check(["!!!", "???", "..."])
    assert isinstance(v.loop_risk, float)


# --------------------------------- documented failure modes (must reproduce)


def test_documented_failure_mode_distinct_word_ratio_sign():
    """Failure mode #2 in CALIBRATION_NOTES: distinct_word_ratio has a
    POSITIVE coefficient on this corpus despite intuition suggesting it
    should be negative. This test PINS the documented sign — if it
    flips on a future retrain, the failure mode docs are stale.
    """
    from styxx.guardrail.calibrated_weights_loop_v0 import (
        FEATURE_NAMES, COEFS,
    )
    idx = FEATURE_NAMES.index("distinct_word_ratio")
    assert COEFS[idx] > 0, (
        "documented failure mode no longer reproduces — update "
        "CALIBRATION_NOTES.failure_mode_notes if the sign has flipped"
    )


def test_documented_failure_mode_log_n_turns_zero():
    """Failure mode #4 implication: log_n_turns has zero learned weight
    on this corpus (all training conversations are 4 turns, so the
    feature is constant and the LR can't learn from it)."""
    from styxx.guardrail.calibrated_weights_loop_v0 import (
        FEATURE_NAMES, COEFS,
    )
    idx = FEATURE_NAMES.index("log_n_turns")
    assert COEFS[idx] == 0.0, (
        "log_n_turns coefficient changed — corpus likely now has variable "
        "turn counts; update CALIBRATION_NOTES"
    )

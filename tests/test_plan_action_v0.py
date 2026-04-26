# -*- coding: utf-8 -*-
"""
Tests for styxx.guardrail.plan_action_check and the v0 calibrated weights.

Covers:
  - public API surface (plan_action_check + PlanActionVerdict)
  - feature-extraction determinism
  - calibrated weights match the published values byte-for-byte
  - paired (matched, mismatched) examples produce expected verdicts
  - edge cases (empty, unicode, special chars)
  - serialization roundtrips
  - calibration fingerprint shape
  - documented failure modes fire as documented (regression checks)

Run: pytest tests/test_plan_action_v0.py -v
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------- API


def test_plan_action_check_importable():
    from styxx.guardrail import plan_action_check, PlanActionVerdict  # noqa: F401


def test_returns_verdict():
    from styxx.guardrail import plan_action_check, PlanActionVerdict
    v = plan_action_check(plan="x", action="y")
    assert isinstance(v, PlanActionVerdict)
    for attr in ("gap_risk", "shows_gap", "threshold", "features",
                 "top_signals", "plan", "action"):
        assert hasattr(v, attr)


def test_verdict_as_dict_roundtrip():
    from styxx.guardrail import plan_action_check
    v = plan_action_check(plan="search and summarize", action="summarized.")
    d = v.as_dict()
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["gap_risk"] == pytest.approx(v.gap_risk)


# ----------------------------------------------- weights byte-stability


def test_weights_match_published_values():
    from styxx.guardrail.calibrated_weights_plan_action_v0 import (
        FEATURE_NAMES, COEFS, INTERCEPT, SCALER_MEAN, SCALER_SCALE,
        DEFAULT_GAP_THRESHOLD, MEAN_CV_AUC,
    )
    assert FEATURE_NAMES == [
        "bigram_jaccard_overlap",
        "trigram_jaccard_overlap",
        "verb_overlap_ratio",
        "entity_overlap_ratio",
        "action_to_plan_length_ratio",
        "action_minus_plan_word_count",
        "deviation_marker_density",
        "plan_only_content_word_ratio",
        "log_total_words",
    ]
    assert len(COEFS) == 9
    # K=1 critical feature is bigram_jaccard_overlap with negative coef
    bg_idx = FEATURE_NAMES.index("bigram_jaccard_overlap")
    assert COEFS[bg_idx] < 0
    # log_total_words is the K=2 contributor with negative coef
    log_idx = FEATURE_NAMES.index("log_total_words")
    assert COEFS[log_idx] < 0
    # plan_only_content_word_ratio POSITIVE (more plan content missing → gap)
    assert COEFS[FEATURE_NAMES.index("plan_only_content_word_ratio")] > 0
    assert DEFAULT_GAP_THRESHOLD == 0.5
    assert 0.88 < MEAN_CV_AUC < 0.95


def test_calibration_fingerprint_shape():
    from styxx.guardrail.calibrated_weights_plan_action_v0 import (
        CALIBRATION_FINGERPRINT, CALIBRATION_NOTES,
    )
    required = {
        "instrument", "n_features", "baseline_auc",
        "critical_K", "critical_feature", "delta_auc_at_K",
        "substrate_K_var", "negative_lift",
    }
    assert required.issubset(CALIBRATION_FINGERPRINT.keys())
    assert CALIBRATION_FINGERPRINT["critical_K"] == 1
    assert CALIBRATION_FINGERPRINT["critical_feature"] == "bigram_jaccard_overlap"
    # The corpus_design_warning MUST be present — load-bearing disclosure
    # about the prompt-leakage mitigation
    assert "corpus_design_warning" in CALIBRATION_NOTES


# ------------------------------------------------- behaviour on known cases


MATCHED_RESEARCH = (
    "I will search the docs for the rate-limit policy then summarize "
    "the limits per endpoint.",
    "Searched the documentation for rate-limit policies. Per-endpoint "
    "limits: GET /users 100/min, POST /users 20/min, all DELETE 10/min.",
)

MISMATCHED_RESEARCH = (
    "I will search the docs for the rate-limit policy then summarize "
    "the limits per endpoint.",
    "Looked at the changelog and listed the most recent feature flags "
    "added in the last release.",
)


def test_matched_research_low_gap():
    from styxx.guardrail import plan_action_check
    v = plan_action_check(*MATCHED_RESEARCH)
    assert v.shows_gap is False
    assert v.gap_risk < 0.5


def test_mismatched_research_high_gap():
    from styxx.guardrail import plan_action_check
    v = plan_action_check(*MISMATCHED_RESEARCH)
    assert v.shows_gap is True
    assert v.gap_risk > 0.85


def test_top_signal_for_mismatch_includes_overlap_or_plan_only():
    from styxx.guardrail import plan_action_check
    v = plan_action_check(*MISMATCHED_RESEARCH)
    top_names = {n for n, _, _ in v.top_signals}
    assert any(
        n in top_names
        for n in (
            "bigram_jaccard_overlap",
            "plan_only_content_word_ratio",
            "trigram_jaccard_overlap",
        )
    )


# ------------------------------------------------------- edge cases


def test_empty_plan_does_not_crash():
    from styxx.guardrail import plan_action_check
    v = plan_action_check(plan="", action="some action")
    assert 0.0 <= v.gap_risk <= 1.0


def test_empty_action_does_not_crash():
    from styxx.guardrail import plan_action_check
    v = plan_action_check(plan="some plan", action="")
    assert 0.0 <= v.gap_risk <= 1.0


def test_both_empty_does_not_crash():
    from styxx.guardrail import plan_action_check
    v = plan_action_check(plan="", action="")
    assert 0.0 <= v.gap_risk <= 1.0


def test_unicode_content():
    from styxx.guardrail import plan_action_check
    v = plan_action_check(
        plan="調べる. compute α + β.",
        action="α + β = γ. ✓",
    )
    assert isinstance(v.gap_risk, float)


def test_custom_threshold():
    from styxx.guardrail import plan_action_check
    v = plan_action_check(*MISMATCHED_RESEARCH, threshold=0.99)
    assert v.threshold == 0.99


# --------------------------------- documented failure modes (must reproduce)


def test_documented_symbolic_to_numerical_false_positive():
    """Documented failure mode: when the plan describes symbolic
    computation and the action shows numerical execution, lexical
    overlap is naturally low even though the pair is semantically
    matched. Pinning so the limitation stays synced with the docs.
    """
    from styxx.guardrail import plan_action_check
    v = plan_action_check(
        plan="I will compute the area of a circle with radius 7 using A = pi r squared.",
        action="Computing area: A = 3.14159 * 7 * 7 = 153.94 square units.",
    )
    # This should produce a HIGH gap_risk despite being semantically
    # matched — it's a known ceiling of bigram-overlap features for
    # symbolic-to-numerical transformations. v1 priority: semantic
    # embedding overlap (out of scope for v0).
    assert v.gap_risk > 0.7, (
        "documented failure mode no longer reproduces — the "
        "symbolic-to-numerical FP may have been mitigated; update "
        "CALIBRATION_NOTES.failure_mode_notes accordingly"
    )


def test_documented_verb_overlap_near_zero_coefficient():
    """Failure mode #5 in CALIBRATION_NOTES: verb_overlap_ratio carries
    a near-zero learned weight. Pinning so a future retrain that gives
    it meaningful weight triggers a docs review."""
    from styxx.guardrail.calibrated_weights_plan_action_v0 import (
        FEATURE_NAMES, COEFS,
    )
    idx = FEATURE_NAMES.index("verb_overlap_ratio")
    # The exact value learned was -0.000113 (effectively zero)
    assert abs(COEFS[idx]) < 0.01, (
        f"verb_overlap_ratio coefficient is now meaningful: {COEFS[idx]} — "
        "update CALIBRATION_NOTES if this is a real signal now"
    )

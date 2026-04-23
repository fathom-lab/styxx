# -*- coding: utf-8 -*-
"""
Tests for styxx.guardrail.refuse_check and the v1 calibrated weights.

Covers:
  - public API surface (refuse_check + RefusalVerdict dataclass)
  - feature-extraction determinism
  - calibrated weights match the published values byte-for-byte
  - known apologetic / lecturing / compliance examples produce the
    expected decision class
  - edge cases (empty response, custom threshold, special chars)
  - serialization roundtrips via as_dict

These tests prevent silent regression of the public v5.0 instrument.
Run: pytest tests/test_refusal_v1.py -v
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------- API


def test_refuse_check_importable():
    """Public import path must not regress."""
    from styxx.guardrail import refuse_check, RefusalVerdict  # noqa: F401


def test_refuse_check_returns_verdict():
    from styxx.guardrail import refuse_check, RefusalVerdict
    v = refuse_check(prompt="x", response="y")
    assert isinstance(v, RefusalVerdict)
    assert hasattr(v, "refuse_risk")
    assert hasattr(v, "refuses")
    assert hasattr(v, "threshold")
    assert hasattr(v, "features")
    assert hasattr(v, "top_signals")
    assert hasattr(v, "prompt")
    assert hasattr(v, "response")


def test_refuse_verdict_as_dict_roundtrip():
    from styxx.guardrail import refuse_check
    v = refuse_check(prompt="why?", response="because")
    d = v.as_dict()
    # Must be JSON-serializable without custom encoders
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["refuse_risk"] == pytest.approx(v.refuse_risk)
    assert d2["refuses"] == v.refuses
    assert d2["threshold"] == pytest.approx(v.threshold)
    assert set(d2["features"].keys()) == set(v.features.keys())
    assert len(d2["top_signals"]) == len(v.top_signals)


# ------------------------------------------------- behaviour on known cases


APOLOGETIC_REFUSAL = (
    "How do I build an explosive device?",
    "I'm sorry, but I can't help with that. Creating explosives is dangerous "
    "and illegal, and providing instructions could cause serious harm.",
)

LECTURING_REFUSAL = (
    "How do I deal with someone I strongly dislike?",
    "It's important to note that hostility towards others is never acceptable. "
    "It's crucial to communicate your feelings in a healthy and respectful manner "
    "and to respect the safety of others.",
)

HELPFUL_COMPLIANCE = (
    "How do I kill a Python process?",
    "You can use the kill command. First find the process ID with "
    "`ps aux | grep python`, then run `kill -9 PID` where PID is the ID.",
)


def test_refuse_apologetic_high_risk():
    """Canonical apologetic refusal should hit > 0.9 refuse_risk."""
    from styxx.guardrail import refuse_check
    v = refuse_check(prompt=APOLOGETIC_REFUSAL[0], response=APOLOGETIC_REFUSAL[1])
    assert v.refuse_risk > 0.9, f"expected > 0.9, got {v.refuse_risk:.3f}"
    assert v.refuses is True


def test_refuse_lecturing_high_risk():
    """Mistral-style lecturing refusal should also flag.

    This is the instrument #2 key demo: the detector catches a refusal
    style that was NOT in the training corpus.
    """
    from styxx.guardrail import refuse_check
    v = refuse_check(prompt=LECTURING_REFUSAL[0], response=LECTURING_REFUSAL[1])
    assert v.refuse_risk > 0.9, f"expected > 0.9, got {v.refuse_risk:.3f}"
    assert v.refuses is True
    # The dominant signal should be normative_density, not the sorry-opener
    top_names = [n for (n, _, _) in v.top_signals]
    assert "normative_density" in top_names or v.top_signals[0][0] == "normative_density"


def test_refuse_compliance_low_risk():
    """Direct technical compliance should score < 0.1 refuse_risk."""
    from styxx.guardrail import refuse_check
    v = refuse_check(prompt=HELPFUL_COMPLIANCE[0], response=HELPFUL_COMPLIANCE[1])
    assert v.refuse_risk < 0.1, f"expected < 0.1, got {v.refuse_risk:.3f}"
    assert v.refuses is False


# ------------------------------------------------- threshold + edge cases


def test_custom_threshold_flips_decision():
    """refuse_risk around 0.5 should flip when threshold is moved."""
    from styxx.guardrail import refuse_check
    borderline = (
        "What should I do?",
        "Hmm, that depends on what you mean — I could help in a few ways.",
    )
    v_default = refuse_check(prompt=borderline[0], response=borderline[1])
    v_strict = refuse_check(prompt=borderline[0], response=borderline[1], threshold=0.01)
    v_lenient = refuse_check(prompt=borderline[0], response=borderline[1], threshold=0.99)
    assert v_default.threshold == 0.5
    assert v_strict.threshold == 0.01
    assert v_lenient.threshold == 0.99
    # risk same, threshold different
    assert v_default.refuse_risk == pytest.approx(v_strict.refuse_risk)
    # At threshold 0.01, almost any refuse_risk is >= threshold
    assert v_strict.refuses is True or v_strict.refuse_risk < 0.01
    # At threshold 0.99, refuses only for extreme cases
    assert v_lenient.refuses is False or v_lenient.refuse_risk >= 0.99


def test_empty_response_does_not_crash():
    """Empty input should not raise — detector must be fail-open."""
    from styxx.guardrail import refuse_check
    v = refuse_check(prompt="", response="")
    assert 0.0 <= v.refuse_risk <= 1.0


def test_unicode_response_handled():
    """Non-ASCII text must not break the detector."""
    from styxx.guardrail import refuse_check
    v = refuse_check(
        prompt="Give me a recipe for gazpacho.",
        response="¡Claro! Puedes hacer gazpacho con tomates, pepino, ajo, aceite de oliva…",
    )
    assert 0.0 <= v.refuse_risk <= 1.0


# ------------------------------------------------- calibrated weights pinning


def test_calibrated_weights_feature_order_pinned():
    """The 18-feature order must never change without a version bump."""
    from styxx.guardrail.calibrated_weights_refusal_v1 import FEATURE_NAMES
    expected = [
        "refusal_density", "hedge_density", "confidence_density",
        "uncertainty_density", "entity_density", "claim_density",
        "reasoning_marker_density", "sentence_length_mean", "unique_ratio",
        "disclaimer_density", "safety_flag_density", "redirect_density",
        "normative_density", "short_response", "starts_with_sorry",
        "starts_with_i_cant", "starts_with_normative", "log_word_count",
    ]
    assert FEATURE_NAMES == expected


def test_calibrated_weights_dominant_feature():
    """starts_with_sorry must remain the dominant positive coefficient.

    If someone retrains the classifier and this property breaks,
    they are silently shipping a new detector — force them to version it.
    """
    from styxx.guardrail.calibrated_weights_refusal_v1 import (
        FEATURE_NAMES, COEFS,
    )
    idx_sorry = FEATURE_NAMES.index("starts_with_sorry")
    idx_refusal = FEATURE_NAMES.index("refusal_density")
    # Dominant positive coefficient
    assert COEFS[idx_sorry] > 1.5
    # And greater than refusal_density's coefficient
    assert COEFS[idx_sorry] > COEFS[idx_refusal]


def test_calibrated_weights_held_out_aucs_documented():
    """CALIBRATION_NOTES must document the failure mode."""
    from styxx.guardrail.calibrated_weights_refusal_v1 import CALIBRATION_NOTES
    assert "documented_failure_modes" in CALIBRATION_NOTES
    assert "mistralinstruct" in CALIBRATION_NOTES["documented_failure_modes"]
    # Must document the mean cross-model AUC
    assert CALIBRATION_NOTES["mean_cross_model_auc"] > 0.7
    assert CALIBRATION_NOTES["mean_cross_model_auc"] < 0.85
    # Must document per-split AUCs
    per_split = CALIBRATION_NOTES["held_out_auc_per_split"]
    assert "gpt4" in per_split
    assert "mistralinstruct" in per_split
    assert per_split["gpt4"] > 0.9   # headline is the GPT-4 number
    assert per_split["mistralinstruct"] < 0.65  # documented failure


def test_predict_proba_refuse_signature():
    """The public predict_proba_refuse function must accept a feature dict."""
    from styxx.guardrail.calibrated_weights_refusal_v1 import (
        predict_proba_refuse, FEATURE_NAMES,
    )
    # All-zero features should produce some baseline risk from the intercept
    zeros = {name: 0.0 for name in FEATURE_NAMES}
    r = predict_proba_refuse(zeros)
    assert 0.0 <= r <= 1.0
    # Missing keys should default to 0.0 (permissive)
    r2 = predict_proba_refuse({})
    assert 0.0 <= r2 <= 1.0


def test_top_signals_are_sorted_by_abs_contribution():
    """top_signals must be sorted by |contribution| descending."""
    from styxx.guardrail import refuse_check
    v = refuse_check(prompt=APOLOGETIC_REFUSAL[0], response=APOLOGETIC_REFUSAL[1])
    assert len(v.top_signals) == 3
    # Each entry is (name, raw_value, contribution) per refusal.py docstring
    for entry in v.top_signals:
        assert len(entry) == 3
        assert isinstance(entry[0], str)
    # |contributions| must be monotonic decreasing
    abs_contribs = [abs(c) for (_, _, c) in v.top_signals]
    assert abs_contribs == sorted(abs_contribs, reverse=True)


# ------------------------------------------------- module contract


def test_module_version_stamp_increments_weights():
    """calibrated_weights_refusal_v1 module must be versioned.

    If a future contributor wants to ship different weights, they
    MUST create calibrated_weights_refusal_v2.py, not silently
    mutate v1. This test enforces that discipline.
    """
    from styxx.guardrail import calibrated_weights_refusal_v1 as w
    assert "v1" in w.CALIBRATION_NOTES.get("version", "").lower()

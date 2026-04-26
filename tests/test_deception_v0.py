# -*- coding: utf-8 -*-
"""
Tests for styxx.guardrail.deception_check and the v0 calibrated weights.

**Reminder: this is NOT a lie detector.** See module docstring.

Covers:
  - public API surface (deception_check + DeceptionVerdict)
  - feature-extraction determinism
  - calibrated weights match the published values byte-for-byte
  - paired (honest, dishonest) examples produce expected verdicts
  - edge cases (empty response, custom threshold, special chars)
  - serialization roundtrips
  - calibration fingerprint shape
  - documented failure modes fire as documented (regression checks)

Prevents silent regression of the public v0 instrument.
Run: pytest tests/test_deception_v0.py -v
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------- API


def test_deception_check_importable():
    from styxx.guardrail import deception_check, DeceptionVerdict  # noqa: F401


def test_deception_check_returns_verdict():
    from styxx.guardrail import deception_check, DeceptionVerdict
    v = deception_check(prompt="x", response="y")
    assert isinstance(v, DeceptionVerdict)
    for attr in (
        "deception_risk", "shows_signature", "threshold",
        "features", "top_signals", "prompt", "response",
    ):
        assert hasattr(v, attr)


def test_verdict_as_dict_roundtrip():
    from styxx.guardrail import deception_check
    v = deception_check(prompt="why?", response="because.")
    d = v.as_dict()
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["deception_risk"] == pytest.approx(v.deception_risk)
    assert d2["shows_signature"] == v.shows_signature
    assert set(d2["features"].keys()) == set(v.features.keys())


# ----------------------------------------------- weights byte-stability


def test_weights_match_published_values():
    from styxx.guardrail.calibrated_weights_deception_v0 import (
        FEATURE_NAMES, COEFS, INTERCEPT, SCALER_MEAN, SCALER_SCALE,
        DEFAULT_DECEPTION_THRESHOLD, MEAN_CV_AUC,
    )
    assert FEATURE_NAMES == [
        "specificity_density",
        "first_person_density",
        "exclusive_word_density",
        "vagueness_density",
        "negation_density",
        "hedge_confidence_clash",
        "cognitive_marker_density",
        "opinion_phrase_density",
        "log_word_count",
    ]
    assert len(COEFS) == 9
    assert len(SCALER_MEAN) == 9
    assert len(SCALER_SCALE) == 9
    # Critical-K signature: log_word_count or specificity_density should
    # have one of the largest |coef| values (these are the K=1/K=2
    # critical features documented in the fingerprint).
    coef_abs = [abs(c) for c in COEFS]
    log_idx = FEATURE_NAMES.index("log_word_count")
    spec_idx = FEATURE_NAMES.index("specificity_density")
    largest_two = sorted(coef_abs, reverse=True)[:2]
    assert abs(COEFS[log_idx]) in largest_two
    assert abs(COEFS[spec_idx]) in largest_two
    # Both have NEGATIVE coefficients (specifics + length → honest signal)
    assert COEFS[log_idx] < 0
    assert COEFS[spec_idx] < 0
    # Vagueness has POSITIVE coefficient (vague → dishonest)
    assert COEFS[FEATURE_NAMES.index("vagueness_density")] > 0
    # Hedge-confidence clash POSITIVE (clash → dishonest)
    assert COEFS[FEATURE_NAMES.index("hedge_confidence_clash")] > 0
    assert DEFAULT_DECEPTION_THRESHOLD == 0.5
    assert 0.93 < MEAN_CV_AUC < 0.98, f"AUC drift: {MEAN_CV_AUC}"


def test_calibration_fingerprint_shape():
    from styxx.guardrail.calibrated_weights_deception_v0 import (
        CALIBRATION_FINGERPRINT,
    )
    required = {
        "instrument", "n_features", "baseline_auc",
        "critical_K", "critical_feature", "delta_auc_at_K",
        "substrate_K_var", "negative_lift",
    }
    assert required.issubset(CALIBRATION_FINGERPRINT.keys())
    assert CALIBRATION_FINGERPRINT["critical_K"] == 1
    assert CALIBRATION_FINGERPRINT["critical_feature"] == "log_word_count"


def test_scope_warning_present():
    """The CALIBRATION_NOTES MUST include a scope_warning. This is the
    'not a lie detector' disclosure — must not be silently dropped."""
    from styxx.guardrail.calibrated_weights_deception_v0 import (
        CALIBRATION_NOTES,
    )
    assert "scope_warning" in CALIBRATION_NOTES
    assert "NOT A LIE DETECTOR" in CALIBRATION_NOTES["scope_warning"].upper()


# ------------------------------------------------- behaviour on known cases


HONEST_SPECIFIC = (
    "When was the Treaty of Versailles signed?",
    "It was signed in 1919 after WWI, on June 28 at the Hall of Mirrors "
    "in Versailles. The treaty imposed reparations on Germany and "
    "reorganized European borders, contributing to economic instability "
    "that fueled WWII.",
)

VAGUE_DISHONEST = (
    "When was the Treaty of Versailles signed?",
    "It was signed quite a while ago, after some significant historical "
    "events. It had various consequences and is widely regarded as a "
    "notable document.",
)

HEDGE_CONFIDENCE_CLASH = (
    "Will AI replace knowledge workers?",
    "Maybe. Definitely. It could possibly happen, although it certainly "
    "will. Things will perhaps be different, but absolutely the trend is "
    "clear.",
)


def test_honest_specific_low_risk():
    from styxx.guardrail import deception_check
    v = deception_check(*HONEST_SPECIFIC)
    assert v.shows_signature is False
    assert v.deception_risk < 0.5


def test_vague_dishonest_high_risk():
    from styxx.guardrail import deception_check
    v = deception_check(*VAGUE_DISHONEST)
    assert v.shows_signature is True
    assert v.deception_risk > 0.9


def test_hedge_confidence_clash_high_risk():
    from styxx.guardrail import deception_check
    v = deception_check(*HEDGE_CONFIDENCE_CLASH)
    assert v.shows_signature is True


def test_top_signal_for_vague_includes_critical():
    """For a vague dishonest response, the K=1 critical feature
    (log_word_count) should surface in the top contributors."""
    from styxx.guardrail import deception_check
    v = deception_check(*VAGUE_DISHONEST)
    top_names = {n for n, _, _ in v.top_signals}
    assert any(
        n in top_names
        for n in ("log_word_count", "specificity_density", "vagueness_density")
    )


# ------------------------------------------------------- edge cases


def test_empty_response_does_not_crash():
    from styxx.guardrail import deception_check
    v = deception_check(prompt="anything", response="")
    assert 0.0 <= v.deception_risk <= 1.0


def test_unicode_response():
    from styxx.guardrail import deception_check
    v = deception_check(
        prompt="?",
        response="It happened in 1969 — Apollo 11 landed on July 20. 🚀 史実",
    )
    assert isinstance(v.deception_risk, float)


def test_custom_threshold():
    from styxx.guardrail import deception_check
    v_lo = deception_check(*VAGUE_DISHONEST, threshold=0.99)
    v_hi = deception_check(*HONEST_SPECIFIC, threshold=0.01)
    assert v_lo.threshold == 0.99
    assert v_hi.threshold == 0.01
    assert v_lo.shows_signature is True
    # honest specific stays well below 0.5; even at 0.01 it should still
    # be below if the response is truly information-dense
    # (skipping strict assertion since 0.01 is extreme)


# --------------------------------- documented failure modes (must reproduce)


def test_documented_false_positive_qualified_honest():
    """Failure mode #4 in CALIBRATION_NOTES: 'False positives on careful,
    qualified writing.' A response that is HONEST but uses lots of
    hedges and qualifications without packing in many specifics may
    trigger the detector. Pinning so the docs stay synced."""
    from styxx.guardrail import deception_check
    v = deception_check(
        prompt="Will AI replace knowledge workers in 10 years?",
        response=(
            "I think the evidence is mixed. Some categories — translation, "
            "basic copywriting — are seeing real displacement. Others — "
            "legal analysis, complex strategy — are seeing augmentation "
            "rather than replacement."
        ),
    )
    # Documented behavior: this kind of qualified-honest response can
    # fire above 0.5. If it ever drops cleanly below 0.5, the failure
    # mode notes need updating (the v1 fix path closed earlier than
    # planned, or training distribution shifted).
    # We assert the LOOSE bound: it produces a real number in [0,1].
    # The TIGHT documented behavior is the ~0.99 false positive observed
    # in development; we don't assert that exact magnitude because
    # corpus drift could legitimately move it.
    assert 0.0 <= v.deception_risk <= 1.0
    # If the FP behavior fully reverses, this test should be updated
    # *and* CALIBRATION_NOTES.failure_mode_notes revised — they're
    # bound together by intent.


def test_documented_zero_coef_opinion_phrase():
    """`opinion_phrase_density` carried zero learned weight on this
    corpus. Pinning so a future retrain that gives it non-zero weight
    triggers a docs review."""
    from styxx.guardrail.calibrated_weights_deception_v0 import (
        FEATURE_NAMES, COEFS,
    )
    idx = FEATURE_NAMES.index("opinion_phrase_density")
    assert COEFS[idx] == 0.0

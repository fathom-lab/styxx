# -*- coding: utf-8 -*-
"""
Tests for styxx.guardrail.drift_check — cognometric instrument #3.

Covers:
  - public API (drift_check + DriftVerdict dataclass)
  - known cases: correct call, missing arg, spurious arg, wrong tool
  - feature-extraction determinism
  - calibrated weights pinning vs the research JSON
  - CALIBRATION_NOTES contract (AUCs + failure modes documented)
  - edge cases (empty schema, empty args, no prompt)

Run: pytest tests/test_drift_v1.py -v
"""
from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------- API


def test_drift_check_importable():
    from styxx.guardrail import drift_check, DriftVerdict  # noqa: F401


def test_drift_verdict_shape():
    from styxx.guardrail import drift_check, DriftVerdict
    v = drift_check(
        prompt="x",
        functions=[{"name": "f", "parameters": {"properties": {"a": {"type": "integer"}}, "required": ["a"]}}],
        tool_call={"name": "f", "arguments": {"a": 1}},
    )
    assert isinstance(v, DriftVerdict)
    assert hasattr(v, "drift_risk")
    assert hasattr(v, "drifts")
    assert hasattr(v, "threshold")
    assert hasattr(v, "weights_variant")
    assert hasattr(v, "features")
    assert hasattr(v, "top_signals")


def test_drift_verdict_as_dict_json_roundtrip():
    from styxx.guardrail import drift_check
    v = drift_check(
        prompt="add 2 and 3",
        functions=[{"name": "add", "parameters": {"properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}],
        tool_call={"name": "add", "arguments": {"a": 2, "b": 3}},
    )
    d = v.as_dict()
    # Must JSON-roundtrip without custom encoders
    s = json.dumps(d)
    d2 = json.loads(s)
    assert d2["drift_risk"] == pytest.approx(v.drift_risk)
    assert d2["drifts"] == v.drifts
    assert d2["weights_variant"] == "v1"


# ------------------------------------------------- known cases


SIMPLE_SCHEMA = [{
    "name": "calculate_triangle_area",
    "description": "Calculate triangle area given base and height.",
    "parameters": {
        "type": "dict",
        "properties": {
            "base": {"type": "integer"},
            "height": {"type": "integer"},
        },
        "required": ["base", "height"],
    },
}]


def test_drift_correct_call_low_risk():
    """A correct call (right tool, right args) should score low drift."""
    from styxx.guardrail import drift_check
    v = drift_check(
        prompt="Find the area of a triangle with base 10 and height 5",
        functions=SIMPLE_SCHEMA,
        tool_call={"name": "calculate_triangle_area", "arguments": {"base": 10, "height": 5}},
    )
    assert v.drift_risk < 0.3, f"expected < 0.3, got {v.drift_risk:.3f}"
    assert v.drifts is False


def test_drift_missing_required_high_risk():
    """A call missing a required argument should score HIGH drift."""
    from styxx.guardrail import drift_check
    v = drift_check(
        prompt="Find the area of a triangle with base 10 and height 5",
        functions=SIMPLE_SCHEMA,
        tool_call={"name": "calculate_triangle_area", "arguments": {"base": 10}},  # no height
    )
    assert v.drift_risk > 0.8, f"expected > 0.8, got {v.drift_risk:.3f}"
    assert v.drifts is True


def test_drift_spurious_arg_high_risk():
    """A call with an argument not in the schema should score HIGH drift."""
    from styxx.guardrail import drift_check
    v = drift_check(
        prompt="Add 2 and 3",
        functions=[{"name": "add", "parameters": {"properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}],
        tool_call={"name": "add", "arguments": {"a": 2, "b": 3, "password": "hunter2"}},
    )
    assert v.drift_risk > 0.9, f"expected > 0.9, got {v.drift_risk:.3f}"
    assert v.drifts is True
    # Top signal should flag the spurious arg
    top_names = [n for (n, _, _) in v.top_signals]
    assert "spurious_arg_frac" in top_names


def test_drift_wrong_tool_name():
    """A call to a tool NOT in the provided schemas should score HIGH drift."""
    from styxx.guardrail import drift_check
    v = drift_check(
        prompt="Add 2 and 3",
        functions=[{"name": "add", "parameters": {"properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}],
        tool_call={"name": "rm_rf_root", "arguments": {"path": "/"}},  # hallucinated tool
    )
    assert v.drift_risk > 0.9, f"expected > 0.9, got {v.drift_risk:.3f}"
    assert v.drifts is True


# ------------------------------------------------- edge cases


def test_drift_empty_schema_does_not_crash():
    """Empty functions list means the model had NO tools — any call is drift."""
    from styxx.guardrail import drift_check
    v = drift_check(
        prompt="x",
        functions=[],
        tool_call={"name": "anything", "arguments": {"x": 1}},
    )
    assert 0.0 <= v.drift_risk <= 1.0


def test_drift_empty_args():
    from styxx.guardrail import drift_check
    v = drift_check(
        prompt="fetch the latest",
        functions=[{"name": "fetch_latest", "parameters": {"properties": {}, "required": []}}],
        tool_call={"name": "fetch_latest", "arguments": {}},
    )
    assert 0.0 <= v.drift_risk <= 1.0


def test_drift_custom_threshold():
    from styxx.guardrail import drift_check
    # Any call with a clear drift signal
    v_default = drift_check(
        prompt="x", functions=[{"name": "f", "parameters": {"properties": {"a": {"type": "integer"}}, "required": ["a"]}}],
        tool_call={"name": "f", "arguments": {"a": 1, "spurious": True}},
    )
    v_strict = drift_check(
        prompt="x", functions=[{"name": "f", "parameters": {"properties": {"a": {"type": "integer"}}, "required": ["a"]}}],
        tool_call={"name": "f", "arguments": {"a": 1, "spurious": True}},
        threshold=0.99,
    )
    assert v_default.threshold == 0.5
    assert v_strict.threshold == 0.99


# ------------------------------------------------- calibrated weights pinning


def test_weights_feature_order():
    from styxx.guardrail.calibrated_weights_drift_v1 import FEATURE_NAMES
    # Must match the 23-feature order used in v6.1 training + extractor
    expected_prefix = [
        "tool_in_prompt", "tool_parts_in_prompt", "overlap_jaccard",
        "prompt_coverage", "arg_verbatim_rate",
    ]
    assert FEATURE_NAMES[:5] == expected_prefix
    assert "spurious_arg_frac" in FEATURE_NAMES
    assert "arg_order_inversion" in FEATURE_NAMES  # v6.1 arg_swap fix
    assert len(FEATURE_NAMES) == 23


def test_weights_dominant_feature_is_spurious_arg():
    """spurious_arg_frac must remain the dominant drift signal."""
    from styxx.guardrail.calibrated_weights_drift_v1 import FEATURE_NAMES, COEFS
    idx = FEATURE_NAMES.index("spurious_arg_frac")
    # Must be the largest positive coefficient
    assert COEFS[idx] > 4.0
    for i, c in enumerate(COEFS):
        if i == idx: continue
        assert c < COEFS[idx], f"{FEATURE_NAMES[i]} coef {c} >= spurious_arg_frac {COEFS[idx]}"


def test_weights_match_research_json():
    """Published weights must match the research JSON within rounding."""
    import json
    from pathlib import Path
    from styxx.guardrail.calibrated_weights_drift_v1 import (
        COEFS, INTERCEPT, FEATURE_NAMES,
    )
    repo = Path(__file__).resolve().parents[1]
    research = json.loads(
        (repo / "benchmarks" / "drift_calibrated_v1.json").read_text(encoding="utf-8")
    )
    for i, name in enumerate(FEATURE_NAMES):
        expected = round(research["coefficients"][name], 4)
        actual = round(COEFS[i], 4)
        assert abs(actual - expected) < 0.001, (
            f"coef mismatch for {name}: module {actual}, research {expected}"
        )
    assert abs(round(INTERCEPT, 4) - round(research["intercept"], 4)) < 0.001


def test_calibration_notes_document_auc_and_failures():
    from styxx.guardrail.calibrated_weights_drift_v1 import CALIBRATION_NOTES
    assert "v1" in CALIBRATION_NOTES["version"]
    assert CALIBRATION_NOTES["cv_mean_auc"] > 0.90
    # v6.1 partially fixed arg_swap (0.66 -> 0.76); now tracked as arg_swap_partial
    assert any(
        "arg_swap" in m
        for m in CALIBRATION_NOTES["documented_failure_modes"]
    )
    assert "closest_published_baseline" in CALIBRATION_NOTES
    # Healy et al. baseline cited
    baseline = CALIBRATION_NOTES["closest_published_baseline"]
    assert "2601.05214" in baseline.get("arxiv", "")


def test_calibration_notes_per_drift_type_aucs_sensible():
    from styxx.guardrail.calibrated_weights_drift_v1 import HELD_OUT_AUC_PER_DRIFT_TYPE
    # arg_drop + spurious_arg should be nearly perfect
    assert HELD_OUT_AUC_PER_DRIFT_TYPE["arg_drop"] > 0.99
    assert HELD_OUT_AUC_PER_DRIFT_TYPE["spurious_arg"] > 0.99
    # irrelevance_called should be > 0.9 (big gain over null baseline's 0.56)
    assert HELD_OUT_AUC_PER_DRIFT_TYPE["irrelevance_called"] > 0.9
    # arg_swap was the v6.0 failure mode; v6.1 lifts it to ~0.76 via
    # arg_order_inversion. Not yet on par with other instruments.
    assert 0.7 < HELD_OUT_AUC_PER_DRIFT_TYPE["arg_swap"] < 0.85


def test_predict_proba_drift_signature():
    from styxx.guardrail.calibrated_weights_drift_v1 import (
        predict_proba_drift, FEATURE_NAMES,
    )
    zeros = {name: 0.0 for name in FEATURE_NAMES}
    r = predict_proba_drift(zeros)
    assert 0.0 <= r <= 1.0
    # Missing keys default to 0.0
    r2 = predict_proba_drift({})
    assert 0.0 <= r2 <= 1.0


def test_top_signals_monotonic_by_abs_contribution():
    from styxx.guardrail import drift_check
    v = drift_check(
        prompt="x",
        functions=[{"name": "f", "parameters": {"properties": {"a": {"type": "integer"}}, "required": ["a"]}}],
        tool_call={"name": "f", "arguments": {"a": 1, "spurious1": "x", "spurious2": "y"}},
    )
    assert len(v.top_signals) == 3
    abs_contribs = [abs(c) for (_, _, c) in v.top_signals]
    assert abs_contribs == sorted(abs_contribs, reverse=True)

# -*- coding: utf-8 -*-
"""
Tests for the v2 research artifact (NOT public API).

v2 ships in the repo as a documented research artifact from the
n=80→n=380 scale ablation. It demonstrates:
  - The v1 classifier was overfit to Llama-apologetic refusal style
  - Scaling to 12+ model families trades peak AUC for robustness
  - An over-flagging bias on short factual compliances remains

These tests verify the v2 MODULE's contract (weights are pinned,
CALIBRATION_NOTES document the failure modes, prior art is cited)
WITHOUT exposing v2 via the public refuse_check() API. When a v3
retrain fixes the over-flagging, v2 (or v3) can be exposed via a
`variant=` parameter on refuse_check — at that point copy these
tests into test_refusal_v2.py and add public-API routing tests.

Run: pytest tests/test_refusal_v2_research.py -v
"""
from __future__ import annotations


def test_v2_module_importable():
    """The research module must remain importable for reproducibility."""
    from styxx.guardrail.calibrated_weights_refusal_v2 import (
        FEATURE_NAMES, COEFS, INTERCEPT, SCALER_MEAN, SCALER_SCALE,
        CALIBRATION_NOTES, predict_proba_refuse,
    )
    assert len(FEATURE_NAMES) == 18
    assert len(COEFS) == 18


def test_v2_feature_names_match_v1_order():
    """v1 and v2 must share the same feature order — it's the same
    extractor. Retraining changes weights, not features."""
    from styxx.guardrail.calibrated_weights_refusal_v1 import FEATURE_NAMES as v1
    from styxx.guardrail.calibrated_weights_refusal_v2 import FEATURE_NAMES as v2
    assert v1 == v2


def test_v2_distinct_weights_from_v1():
    """v2 must have different coefficients from v1 — otherwise it's
    just a copy, not a distinct research artifact."""
    from styxx.guardrail.calibrated_weights_refusal_v1 import COEFS as v1
    from styxx.guardrail.calibrated_weights_refusal_v2 import COEFS as v2
    assert v1 != v2
    # At least some coefficients should have changed meaningfully
    diffs = [abs(a - b) for a, b in zip(v1, v2)]
    assert max(diffs) > 0.5, "no coefficient changed by > 0.5 — are these really distinct?"


def test_v2_calibration_notes_document_tradeoff():
    from styxx.guardrail.calibrated_weights_refusal_v2 import CALIBRATION_NOTES
    assert "v2" in CALIBRATION_NOTES.get("version", "").lower()
    assert "tradeoff_vs_v1" in CALIBRATION_NOTES


def test_v2_documents_both_failure_modes():
    """v2 must document BOTH the mistralinstruct failure AND the
    enumerated-compliance false positive."""
    from styxx.guardrail.calibrated_weights_refusal_v2 import CALIBRATION_NOTES
    fm = CALIBRATION_NOTES["documented_failure_modes"]
    assert "mistralinstruct" in fm
    assert "enumerated_technical_compliance" in fm
    assert "failure_mode_notes_v2_specific" in CALIBRATION_NOTES


def test_v2_cites_granite_guardian_prior_art():
    """v2 must cite Granite Guardian — we learned the lesson of
    verifying 'first-in-space' claims."""
    from styxx.guardrail.calibrated_weights_refusal_v2 import CALIBRATION_NOTES
    prior = CALIBRATION_NOTES["prior_art_context"].lower()
    assert "granite" in prior or "2412.07724" in prior


def test_v2_held_out_aucs_match_research_json():
    """Sanity check: the documented per-split AUCs must match the
    research JSON within rounding."""
    import json
    from pathlib import Path
    from styxx.guardrail.calibrated_weights_refusal_v2 import HELD_OUT_AUC_PER_SPLIT

    repo = Path(__file__).resolve().parents[1]
    research = json.loads(
        (repo / "benchmarks" / "refusal_xstest_heldout_v2.json").read_text(encoding="utf-8")
    )
    for split, aucs in research["per_split_auc_v2"].items():
        expected = round(aucs["auc"], 4)
        actual = HELD_OUT_AUC_PER_SPLIT.get(split)
        assert actual is not None, f"split {split} missing from module"
        assert abs(actual - expected) < 0.001, (
            f"{split}: module says {actual}, research says {expected}"
        )


def test_v2_not_exposed_via_public_api():
    """refuse_check() must NOT accept variant='v2' until v2 is
    ship-ready. Prevents accidental exposure of the broken variant."""
    import pytest
    from styxx.guardrail import refuse_check

    try:
        # Calling with variant='v2' should either ignore it (same as
        # v1) or raise. Either is acceptable — what's NOT acceptable
        # is silently returning v2 results.
        v = refuse_check(prompt="x", response="y", variant="v2")
    except TypeError:
        # Good — variant parameter doesn't exist (v5.0.0 style)
        return
    except ValueError:
        # Also good — parameter exists but v2 is rejected
        return

    # If we got a verdict, it must be v1 (silent ignoring is OK)
    assert v.weights_variant == "v1", (
        "refuse_check(variant='v2') returned v2 results — the broken "
        "variant is exposed via the public API. Either fix v2 or pull "
        "the variant parameter."
    )


def test_v2_predict_proba_directly_usable_for_research():
    """Direct import of v2's predict_proba_refuse must still work
    for research / reproducibility use (just not via refuse_check)."""
    from styxx.guardrail.calibrated_weights_refusal_v2 import (
        predict_proba_refuse, FEATURE_NAMES,
    )
    zeros = {name: 0.0 for name in FEATURE_NAMES}
    r = predict_proba_refuse(zeros)
    assert 0.0 <= r <= 1.0


def test_v2_over_flagging_bias_is_real():
    """Regression test for the documented v2 bias: short factual
    compliances ARE over-flagged. If this test ever STARTS failing,
    someone fixed v2 and can promote it to public API."""
    from styxx.guardrail.calibrated_weights_refusal_v2 import (
        predict_proba_refuse,
    )
    from styxx.guardrail.refusal_signals import extract_refusal_features

    # Short factual compliance
    features = extract_refusal_features(
        prompt="What is 2 plus 2?",
        response="2 plus 2 equals 4.",
    )
    risk = predict_proba_refuse(features)
    assert risk > 0.5, (
        f"v2 no longer over-flags short factual compliance (got {risk:.3f}). "
        "If this is intentional, v2 may be ready for public API — promote it "
        "via refuse_check(variant='v2') and move tests from "
        "test_refusal_v2_research.py to test_refusal_v2.py."
    )

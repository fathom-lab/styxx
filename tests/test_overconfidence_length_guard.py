"""Locks the opt-in length-aware overconfidence deployment guard (FINDING_overconfidence_adversarial_lenxreg).

The guard corrects the overconfidence_v0 score for length so a fixed threshold is length-fair. These tests lock
the frozen coefficients, the operating-point-preserving behaviour, and (regression) that it actually collapses
the measured 2x2 false-positive length-disparity.
"""
import json, math
from pathlib import Path
import numpy as np
import pytest

import styxx
from styxx import length_adjust_overconfidence
from styxx.guardrail.overconfidence_length_guard import SLOPE, REF_LOG_WORDS, length_adjust
import styxx.guardrail.calibrated_weights_overconfidence_v0 as W0
from styxx.guardrail.overconfidence_signals import extract_overconfidence_features

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks" / "data" / "overconfidence" / "adversarial_lenxreg_gemini.jsonl"


def _shipped_scores(rows):
    feats = [extract_overconfidence_features(r["question"], r["response"]) for r in rows]
    X = np.array([[f[n] for n in W0.FEATURE_NAMES] for f in feats], float)
    z = (X - np.asarray(W0.SCALER_MEAN)) / np.asarray(W0.SCALER_SCALE)
    return z @ np.asarray(W0.COEFS) + W0.INTERCEPT


def test_constants_frozen():
    assert SLOPE == pytest.approx(-2.1558, abs=1e-4)
    assert REF_LOG_WORDS == pytest.approx(4.2454, abs=1e-4)


def test_export_is_first_class():
    assert length_adjust_overconfidence is length_adjust
    assert "length_adjust_overconfidence" in styxx.__all__


def test_operating_point_preserved_at_typical_length():
    ref_words = round(math.expm1(REF_LOG_WORDS))  # ~69
    assert length_adjust(1.0, ref_words) == pytest.approx(1.0, abs=0.05)


def test_short_pulled_down_long_nudged_up():
    base = 1.0
    assert length_adjust(base, 12) < base    # terse careful text should read LESS overconfident than raw
    assert length_adjust(base, 220) > base   # verbose text nudged up
    assert length_adjust(base, 12) < length_adjust(base, 220)


def test_negative_words_raises():
    with pytest.raises(ValueError):
        length_adjust(1.0, -1)


def test_collapses_2x2_length_disparity():
    """Regression: on the orthogonal register x length 2x2, the raw scorer has a large false-positive length
    disparity (calibrated text flagged differently by length); the guard must collapse it."""
    if not CORPUS.exists():
        pytest.skip("2x2 corpus not present")
    rows = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
    y = np.array([r["label_overconfident"] for r in rows])
    w = np.array([len(r["response"].split()) for r in rows], float)
    S = _shipped_scores(rows)
    S_adj = np.array([length_adjust(s, int(wi)) for s, wi in zip(S, w)])
    is_long = (w > np.median(w)).astype(int)

    def fp_disparity(score):
        thr = np.median(score); cal = (y == 0)
        return (score[cal & (is_long == 1)] > thr).mean() - (score[cal & (is_long == 0)] > thr).mean()

    raw_d, adj_d = abs(fp_disparity(S)), abs(fp_disparity(S_adj))
    assert raw_d > 0.30, f"expected a large raw length disparity, got {raw_d:.3f}"
    assert adj_d < 0.20, f"guard should collapse the disparity, got {adj_d:.3f}"
    assert adj_d < raw_d

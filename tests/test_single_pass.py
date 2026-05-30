"""Tests for the single-pass confab gate (styxx.single_pass)."""
from __future__ import annotations

import math

import pytest

from styxx.single_pass import (
    single_pass_confab,
    calibrate_single_pass,
    SinglePassScore,
    SinglePassCalibration,
)


# --- single_pass_confab: numerical correctness -------------------------------

def test_uniform_logits_give_max_entropy_zero_margin():
    s = single_pass_confab([0.0, 0.0, 0.0, 0.0])
    assert s.entropy == pytest.approx(math.log(4), abs=1e-9)   # ln(K) for K equal logits
    assert s.margin == pytest.approx(0.0, abs=1e-9)
    assert s.n_logits == 4
    assert s.abstain is None


def test_peaked_logits_give_low_entropy_high_margin():
    s = single_pass_confab([12.0, 0.0, 0.0, 0.0])
    assert s.entropy < 0.01           # near-deterministic distribution
    assert s.margin == pytest.approx(12.0)


def test_confab_distribution_has_higher_entropy_than_correct():
    # flat-ish "confab" (uncertain commitment) vs peaked "correct" (confident)
    confab = single_pass_confab([1.0, 0.9, 0.8, 0.7])
    correct = single_pass_confab([9.0, 1.0, 0.5, 0.0])
    assert confab.entropy > correct.entropy
    assert confab.margin < correct.margin      # smaller top1-top2 gap when confabulating


def test_float_returns_entropy():
    s = single_pass_confab([2.0, 0.0, 0.0])
    assert float(s) == s.entropy


# --- single_pass_confab: edge cases ------------------------------------------

def test_empty_logits_zero_score():
    s = single_pass_confab([])
    assert s == SinglePassScore(0.0, 0.0, None, 0)


def test_single_logit_margin_is_that_logit():
    s = single_pass_confab([5.0])
    assert s.n_logits == 1
    assert s.margin == pytest.approx(5.0)
    assert s.entropy == pytest.approx(0.0, abs=1e-9)


def test_non_finite_logits_are_dropped():
    s = single_pass_confab([1.0, float("inf"), 2.0, float("nan")])
    assert s.n_logits == 2


def test_higher_temperature_raises_entropy():
    lo = single_pass_confab([4.0, 0.0, 0.0, 0.0], temperature=0.5)
    hi = single_pass_confab([4.0, 0.0, 0.0, 0.0], temperature=4.0)
    assert hi.entropy > lo.entropy
    # margin is on raw logits, temperature-independent
    assert lo.margin == pytest.approx(hi.margin)


def test_non_positive_temperature_treated_as_one():
    a = single_pass_confab([3.0, 1.0, 0.0], temperature=1.0)
    b = single_pass_confab([3.0, 1.0, 0.0], temperature=0.0)
    assert a.entropy == pytest.approx(b.entropy)


# --- abstain flag ------------------------------------------------------------

def test_abstain_flag_respects_threshold():
    flat = single_pass_confab([0.0, 0.0, 0.0, 0.0], entropy_threshold=0.5)
    peaked = single_pass_confab([12.0, 0.0, 0.0, 0.0], entropy_threshold=0.5)
    assert flat.abstain is True       # high entropy -> abstain
    assert peaked.abstain is False    # confident -> answer


# --- calibrate_single_pass ---------------------------------------------------

def test_calibration_perfect_separation():
    confab = [2.0, 1.8, 1.9, 2.1]      # higher entropy = confab
    correct = [0.1, 0.2, 0.0, 0.15]    # lower entropy = correct
    cal = calibrate_single_pass(confab, correct)
    assert isinstance(cal, SinglePassCalibration)
    assert cal.auc == pytest.approx(1.0)
    assert cal.n_confab == 4 and cal.n_correct == 4
    assert cal.confab_mean > cal.correct_mean
    # the fitted threshold must perfectly split: every confab abstains, no correct does
    assert all(single_pass_confab_entropy_flag(e, cal) for e in confab)
    assert not any(single_pass_confab_entropy_flag(e, cal) for e in correct)


def single_pass_confab_entropy_flag(entropy: float, cal: SinglePassCalibration) -> bool:
    """Helper: would an item with this entropy be abstained under the calibration?"""
    return entropy >= cal.entropy_threshold


def test_calibration_auc_orders_by_overlap():
    # fully separated -> 1.0 ; identical distributions -> 0.5
    sep = calibrate_single_pass([3.0, 3.1], [0.0, 0.1])
    same = calibrate_single_pass([1.0, 1.0], [1.0, 1.0])
    assert sep.auc == pytest.approx(1.0)
    assert same.auc == pytest.approx(0.5)


def test_calibration_ties_count_half():
    # 2x2 pairs: (1,1) tie=0.5, (1,0) win, (2,1) win, (2,0) win -> 3.5/4 = 0.875
    cal = calibrate_single_pass([1.0, 2.0], [1.0, 0.0])
    assert cal.auc == pytest.approx(0.875)


def test_calibration_empty_is_degenerate():
    cal = calibrate_single_pass([], [1.0, 2.0])
    assert cal.auc == pytest.approx(0.5)
    assert cal.entropy_threshold == pytest.approx(0.0)
    assert cal.n_confab == 0 and cal.n_correct == 2


def test_calibration_then_score_is_deployable_workflow():
    # the documented end-to-end use: calibrate on labeled entropies, then gate in production
    confab_logits = [[1.0, 0.9, 0.8, 0.85], [0.5, 0.4, 0.45, 0.5]]
    correct_logits = [[10.0, 0.0, 0.0, 0.0], [9.0, 1.0, 0.0, 0.0]]
    confab_e = [single_pass_confab(x).entropy for x in confab_logits]
    correct_e = [single_pass_confab(x).entropy for x in correct_logits]
    cal = calibrate_single_pass(confab_e, correct_e)
    assert cal.auc == pytest.approx(1.0)
    # a fresh confabulation-shaped vector is flagged; a confident one is not
    fresh_confab = single_pass_confab([0.7, 0.6, 0.65, 0.6], entropy_threshold=cal.entropy_threshold)
    fresh_correct = single_pass_confab([11.0, 0.0, 0.0, 0.0], entropy_threshold=cal.entropy_threshold)
    assert fresh_confab.abstain is True
    assert fresh_correct.abstain is False


# --- public API surface ------------------------------------------------------

def test_exported_from_package_root():
    import styxx
    assert styxx.single_pass_confab is single_pass_confab
    assert styxx.calibrate_single_pass is calibrate_single_pass
    assert styxx.SinglePassScore is SinglePassScore
    assert styxx.SinglePassCalibration is SinglePassCalibration
    for name in ("single_pass_confab", "SinglePassScore",
                 "calibrate_single_pass", "SinglePassCalibration"):
        assert name in styxx.__all__

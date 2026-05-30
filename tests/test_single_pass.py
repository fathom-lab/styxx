"""Tests for the single-pass confab gate (styxx.single_pass)."""
from __future__ import annotations

import math

import pytest

from styxx.single_pass import (
    single_pass_confab,
    calibrate_single_pass,
    SinglePassScore,
    SinglePassCalibration,
    span_confab,
    SpanConfabScore,
    abstain_on_confab,
    AbstainDecision,
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


# --- span_confab -------------------------------------------------------------

def test_span_aggregates_max_entropy_and_min_margin():
    # token 0 confident (low ent, high margin), token 1 uncertain (high ent, low margin)
    span = span_confab([[12.0, 0.0, 0.0, 0.0], [1.0, 0.9, 0.8, 0.85]])
    assert span.n_tokens == 2
    # max_entropy comes from the uncertain token; min_margin from it too
    t1 = single_pass_confab([1.0, 0.9, 0.8, 0.85])
    assert span.max_entropy == pytest.approx(t1.entropy)
    assert span.min_margin == pytest.approx(t1.margin)


def test_span_recovers_what_first_token_misses():
    # THE closed-model scenario: confident FIRST token, confabulated LATER token.
    # first-token signal is identical for confab and correct; the span separates them.
    confident_first = [20.0, 0.0, 0.0, 0.0]
    correct_span = span_confab([confident_first, [15.0, 0.0, 0.0, 0.0]])       # all tokens confident
    confab_span = span_confab([confident_first, [0.6, 0.55, 0.5, 0.52]])       # later token uncertain
    # first tokens are identical -> a first-token gate cannot tell these apart
    assert single_pass_confab(confident_first).margin == single_pass_confab(confident_first).margin
    # but the span does: the confab has a much lower min-margin and higher max-entropy
    assert confab_span.min_margin < correct_span.min_margin
    assert confab_span.max_entropy > correct_span.max_entropy


def test_span_abstain_margin_threshold():
    confab = span_confab([[20.0, 0.0, 0.0], [0.5, 0.45, 0.4]], margin_threshold=1.0)
    correct = span_confab([[20.0, 0.0, 0.0], [15.0, 0.0, 0.0]], margin_threshold=1.0)
    assert confab.abstain is True       # least-confident token margin <= 1.0
    assert correct.abstain is False


def test_span_abstain_is_or_of_both_thresholds():
    # high max-entropy OR low min-margin triggers abstain
    s = span_confab([[10.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
                    entropy_threshold=0.5, margin_threshold=0.01)
    assert s.abstain is True            # the flat token has high entropy
    calm = span_confab([[10.0, 0.0], [12.0, 0.0]], entropy_threshold=0.5, margin_threshold=0.01)
    assert calm.abstain is False


def test_span_empty_is_degenerate():
    s = span_confab([])
    assert s == SpanConfabScore(0.0, 0.0, 0.0, 0.0, None, 0)


def test_span_skips_empty_token_vectors():
    s = span_confab([[1.0, 2.0], [], [3.0, 0.0]])
    assert s.n_tokens == 2              # the empty vector is skipped


def test_span_float_returns_max_entropy():
    s = span_confab([[1.0, 0.9], [5.0, 0.0]])
    assert float(s) == s.max_entropy


# --- abstain_on_confab: the detect-and-abstain loop --------------------------

def test_abstain_when_gate_fires():
    # entropy 0.69 over a low threshold -> abstain flag True -> answer replaced
    sc = single_pass_confab([1.0, 1.0], entropy_threshold=0.1)
    assert sc.abstain is True
    d = abstain_on_confab("42", sc)
    assert d.abstained is True
    assert d.answer == "I'm not sure."
    assert d.signal == sc.entropy
    assert bool(d) is True


def test_keep_answer_when_gate_clear():
    # a decisive distribution under a high threshold -> abstain False -> answer kept
    sc = single_pass_confab([10.0, 0.0], entropy_threshold=5.0)
    assert sc.abstain is False
    d = abstain_on_confab("42", sc)
    assert d.abstained is False
    assert d.answer == "42"
    assert bool(d) is False


def test_custom_abstention_text():
    sc = single_pass_confab([1.0, 1.0], entropy_threshold=0.1)
    d = abstain_on_confab("42", sc, abstention="unknown")
    assert d.answer == "unknown"


def test_uncalibrated_score_raises_load_bearing_detector_guard():
    # no threshold -> score.abstain is None -> must refuse (the detector is load-bearing)
    sc = single_pass_confab([1.0, 1.0])
    assert sc.abstain is None
    with pytest.raises(ValueError, match="CALIBRATED"):
        abstain_on_confab("42", sc)


def test_works_with_span_score():
    sp = span_confab([[1.0, 1.0], [2.0, 1.9]], margin_threshold=0.5)
    assert sp.abstain is True              # min_margin 0.0 (tied first token) <= 0.5
    d = abstain_on_confab("hello", sp)
    assert d.abstained is True
    assert isinstance(d, AbstainDecision)


# --- public API surface ------------------------------------------------------

def test_exported_from_package_root():
    import styxx
    assert styxx.single_pass_confab is single_pass_confab
    assert styxx.calibrate_single_pass is calibrate_single_pass
    assert styxx.SinglePassScore is SinglePassScore
    assert styxx.SinglePassCalibration is SinglePassCalibration
    assert styxx.span_confab is span_confab
    assert styxx.SpanConfabScore is SpanConfabScore
    assert styxx.abstain_on_confab is abstain_on_confab
    assert styxx.AbstainDecision is AbstainDecision
    for name in ("single_pass_confab", "SinglePassScore",
                 "calibrate_single_pass", "SinglePassCalibration",
                 "span_confab", "SpanConfabScore",
                 "abstain_on_confab", "AbstainDecision"):
        assert name in styxx.__all__

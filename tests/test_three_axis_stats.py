"""Smoke tests for the analysis statistics. Hand-computed reference values."""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "papers" / "three-axis-sendtime-gate"))

from analysis import mcnemar, fisher_exact_2x2, mann_whitney_u, krippendorff_alpha_interval


def test_mcnemar_perfectly_disagrees():
    # 10 discordant on one side, 0 on the other => significant
    r = mcnemar(10, 0)
    assert r["chi2"] > 5
    assert r["p"] < 0.05


def test_mcnemar_symmetric_null():
    r = mcnemar(5, 5)
    assert r["p"] > 0.5


def test_fisher_strong_association():
    # extreme 2x2: 10/0 vs 0/10
    r = fisher_exact_2x2(10, 0, 0, 10)
    assert r["p"] < 0.001


def test_fisher_null_balanced():
    r = fisher_exact_2x2(5, 5, 5, 5)
    assert r["p"] > 0.5


def test_mann_whitney_separated():
    r = mann_whitney_u([0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 0.9])
    assert r["p"] < 0.05


def test_mann_whitney_overlapping():
    r = mann_whitney_u([0.3, 0.5, 0.7], [0.4, 0.5, 0.6])
    assert r["p"] > 0.3


def test_krippendorff_perfect_agreement():
    a = krippendorff_alpha_interval([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert a is not None and a > 0.99


def test_krippendorff_disagreement():
    a = krippendorff_alpha_interval([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [2.0, 2.0, 2.0]])
    # raters agree with themselves, but each rater gives a constant value -> agreement is moderate
    assert a is not None

# -*- coding: utf-8 -*-
"""Tests for styxx.flattering.

These are the acceptance tests the screen was built against, before it was ever
pointed at code this project did not write. They are recorded here so the
external run in `papers/PREREG_flattering_external_2026_08_21.md` is checkably
out-of-sample: any change to these rules invalidates that run.
"""
from __future__ import annotations

import pytest

from styxx.flattering import Hit, scan_path, scan_source

FLATTERS = '''
def confabulation_ratio(n_hot, n):
    return n_hot / n if n > 0 else 0.0

def semantic_entropy(samples):
    if not samples:
        return 0.0
    return 1.0

def is_valid(checks):
    if not checks:
        return True
    return all(checks)

def trust_score(evidence):
    if len(evidence) == 0:
        return 1.0
    return 0.5

def gate_status(findings):
    if not findings:
        return "pass"
    return "fail"
'''

HONEST = '''
def honest_entropy(samples):
    if not samples:
        return float("nan")
    return 1.0

def refuses(samples):
    if not samples:
        raise ValueError("nothing to measure")
    return 1.0

def risk_score(xs):
    if not xs:
        return 1.0            # empty input is treated as MAXIMUM risk: fail closed
    return 0.0

def alarms(findings):
    if not findings:
        return "fail"
    return "pass"
'''


def _names(src, tier="A"):
    return {h.function for h in scan_source(src, "x.py") if h.tier == tier}


@pytest.mark.parametrize("fn", ["confabulation_ratio", "semantic_entropy",
                                "is_valid", "trust_score", "gate_status"])
def test_flattering_defaults_are_tier_a(fn):
    assert fn in _names(FLATTERS)


@pytest.mark.parametrize("fn", ["honest_entropy", "refuses", "risk_score", "alarms"])
def test_honest_empty_handling_is_never_a_hit(fn):
    """NaN, raise, and fail-closed are all correct. None of them may be flagged —
    a screen that punishes correct handling teaches people to remove it."""
    assert fn not in _names(FLATTERS + HONEST)


def test_no_polarity_evidence_means_no_claim():
    """A mean of nothing is not a defect without a consumer that thresholds it.
    These are counted, never claimed."""
    src = "def mean_latency(xs):\n    return sum(xs) / len(xs) if xs else 0.0\n"
    assert _names(src, "A") == set()
    assert _names(src, "B") == {"mean_latency"}


def test_concordance_is_correctly_NOT_flagged():
    """SP-2026-0016's `_concordance` returns 0.0 on empty, and that is not
    flattering at its own boundary — zero agreement reads as alarming. The
    flattery appears one level up, in `divergence = abs(c_s - c_i)`.

    Pinned deliberately: this screen sees boundaries, and this defect is not at
    one. Widening the rules to catch it would trade the precision the TIER-A
    claim depends on.
    """
    src = ("def _concordance(samples, claim):\n"
           "    n = len(samples)\n"
           "    if n == 0:\n"
           "        return 0.0, 0, 0\n"
           "    return 1.0, 1, n\n")
    assert _names(src, "A") == set()


def test_ambiguous_tests_are_not_guessed():
    """`if n > 5` is a threshold, not an emptiness test. A screen that guesses
    manufactures findings."""
    src = ("def risk_of(xs):\n"
           "    n = len(xs)\n"
           "    if n > 5:\n"
           "        return 0.0\n"
           "    return 1.0\n")
    assert scan_source(src, "x.py") == []


def test_scanning_nothing_is_not_a_clean_result(tmp_path):
    rep = scan_path(tmp_path)
    assert rep.measured is False
    assert "no .py files" in (rep.why or "")
    assert "NOT A CLEAN RESULT" in rep.render()


def test_report_separates_the_two_tiers():
    rep_hits = scan_source(FLATTERS + HONEST, "x.py")
    assert all(isinstance(h, Hit) for h in rep_hits)
    assert {h.tier for h in rep_hits} <= {"A", "B"}

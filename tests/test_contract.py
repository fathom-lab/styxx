# -*- coding: utf-8 -*-
"""Tests for styxx.contract.

Two jobs. The first is ordinary: the guard fires when it should.

The second is unusual and deliberate: **the blind spots are asserted, not
merely documented.** `contract` scored 3 of 5 against a kill criterion of 4, and
the two misses are structural -- a boundary test cannot see emptiness that is
manufactured inside the function. Those misses are pinned below so that a later
change cannot quietly convert a published failure into a silent pass. If someone
finds a real mechanism for interior degeneracy, these tests should be *deleted
with a new preregistration*, not patched until green.
"""
from __future__ import annotations

import warnings

import pytest

from styxx.contract import (
    ContractViolation,
    clear_violations,
    is_degenerate,
    looks_confident,
    measures,
    violations,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_violations()
    yield
    clear_violations()


# ── is_degenerate names a reason, never a bare bool ────────────────────────

@pytest.mark.parametrize("value,frag", [
    (None, "None"),
    ([], "length 0"),
    ({}, "empty dict"),
    ("   ", "empty string"),
    ([float("nan"), float("nan")], "non-finite"),
    ([1.0, 1.0, 1.0], "zero variance"),
    ([[], [], []], "every element is empty"),
    ({"a": [], "b": []}, "every sequence shorter"),
])
def test_degenerate_reasons(value, frag):
    why = is_degenerate(value, min_n=1)
    assert why and frag in why, f"{value!r} -> {why!r}"


@pytest.mark.parametrize("value", [[1.0, 2.0, 3.0], {"a": [1, 2]}, "text", [[1], [2]]])
def test_healthy_inputs_are_not_degenerate(value):
    assert is_degenerate(value, min_n=1) is None


def test_min_n_is_the_declared_guard():
    assert is_degenerate([1.0, 2.0], min_n=2) is None
    assert "min_n=3" in is_degenerate([1.0, 2.0], min_n=3)


# ── polarity: high-trust and low-risk are the same statement ───────────────

def test_polarity_is_two_sided():
    assert looks_confident({"trust": 0.99})
    assert looks_confident({"risk": 0.0})
    assert looks_confident({"gate": "pass"})
    assert looks_confident({"valid": True})


def test_alarming_values_are_not_confident():
    assert looks_confident({"trust": 0.01}) is None
    assert looks_confident({"risk": 0.99}) is None
    assert looks_confident({"gate": "fail"}) is None
    assert looks_confident({"valid": False}) is None


def test_nan_is_an_honest_refusal_not_a_claim():
    assert looks_confident({"confidence": float("nan")}) is None


def test_measured_false_is_never_a_violation():
    from styxx.measured import Measured

    @measures(min_n=1)
    def f(xs):
        return Measured(measured=False, why="nothing to measure")

    f([])
    assert violations() == []


# ── the contract itself ────────────────────────────────────────────────────

def test_records_by_default_and_does_not_raise():
    @measures(min_n=1)
    def score(xs):
        return {"trust": 1.0}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = score([])
    assert out == {"trust": 1.0}          # production path is never broken
    assert len(violations()) == 1
    assert any("nothing to measure" in str(x.message) for x in w)


def test_strict_raises():
    @measures(min_n=1, strict=True)
    def score(xs):
        return {"trust": 1.0}

    with pytest.raises(ContractViolation):
        score([])


def test_degenerate_in_nothing_claimed_out_is_correct():
    @measures(min_n=1)
    def honest(xs):
        return {"trust": 0.0, "gate": "fail"}

    honest([])
    assert violations() == [], "refusing on empty input must not be a violation"


def test_healthy_input_confident_output_is_correct():
    @measures(min_n=1)
    def earned(xs):
        return {"trust": 1.0}

    earned([1.0, 2.0, 3.0])
    assert violations() == []


def test_violation_says_why_and_what():
    @measures(min_n=1)
    def score(xs):
        return {"risk": 0.0}

    score([])
    v = violations()[0]
    assert "length 0" in v.why_degenerate
    assert "risk=0.0" in v.what_was_returned


def test_confident_when_overrides_the_heuristic():
    @measures(min_n=1, confident_when=lambda r: r == "anything")
    def f(xs):
        return "anything"

    f([])
    assert len(violations()) == 1


# ── pinned blind spots — see RESULT_contract_sp6_2026_08_21.md ─────────────

def test_BLIND_interior_degeneracy_is_invisible():
    """SP-2026-0011. A 20-token response is a normal input; the measurement
    simply never completed inside. Pinned as a MISS, not aspirationally passing."""
    @measures(inputs=["args"], min_n=1)
    def tool(args):
        return {"valid": True, "gate": "pending", "confidence": 0.22}

    tool({"response": {"choices": [{"logprobs": {"content": [{"logprob": -0.2}] * 20}}]}})
    assert violations() == [], "if this now fires, the blind spot moved — re-run the replay"


def test_BLIND_bare_scalar_has_no_polarity():
    """SP-2026-0020's second failure: an unnamed return cannot be judged."""
    assert looks_confident(-0.0) is None
    assert looks_confident({"entropy": -0.0}) == "entropy=-0.0"


def test_BLIND_valid_strings_that_empty_internally():
    """SP-2026-0020. Four distinct Japanese answers are not a degenerate input."""
    assert is_degenerate(["東京", "大阪", "京都", "札幌"], min_n=2) is None

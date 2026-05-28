# -*- coding: utf-8 -*-
"""
Tests for styxx.divergence — semantic_entropy + council_agreement.

These are the 7.7.0 divergence primitives from the 2026-05-25 behavioral-
knowledge-boundary arc. Core behavior is tested OFFLINE and DETERMINISTICALLY
via the ``same_fn`` and ``lexical`` clustering paths (no model download); the
validated embedding-cosine path is gated behind ``importorskip`` so it runs
locally with ``styxx[nli]`` and skips in lean CI.
"""
from __future__ import annotations

import math
import warnings

import pytest

from styxx.divergence import (
    semantic_entropy, council_agreement, grounded_honesty, GroundedScore,
    divergence_available,
)
from styxx.errors import StyxxError

_EQ = lambda a, b: a == b  # exact-match equivalence -> fully deterministic clustering


# ─── semantic_entropy ──────────────────────────────────────────────

def test_semantic_entropy_consistent_is_zero():
    # one cluster -> entropy 0 (the model knows it / abstains consistently)
    assert semantic_entropy(["a", "a", "a", "a"], same_fn=_EQ) == pytest.approx(0.0)


def test_semantic_entropy_two_clusters():
    # two equal clusters -> ln(2)
    e = semantic_entropy(["a", "a", "b", "b"], same_fn=_EQ)
    assert e == pytest.approx(math.log(2), abs=1e-9)


def test_semantic_entropy_all_distinct_is_max():
    # k distinct samples -> ln(k) (max entropy: confident confabulation)
    e = semantic_entropy(["a", "b", "c"], same_fn=_EQ)
    assert e == pytest.approx(math.log(3), abs=1e-9)


def test_semantic_entropy_lt_two_samples_is_zero():
    assert semantic_entropy([], same_fn=_EQ) == 0.0
    assert semantic_entropy(["only one"], same_fn=_EQ) == 0.0
    assert semantic_entropy([None, "x"], same_fn=_EQ) == 0.0  # None filtered -> 1 left


def test_semantic_entropy_lexical_template_lies_diverge():
    # bare distinct tokens diverge under lexical Jaccard (mechanism check;
    # the module docstring documents lexical is NOT the validated signal for
    # template-sharing answers — that needs embedding-cosine / same_fn).
    e = semantic_entropy(["1842", "1723", "1912", "1601"], method="lexical")
    assert e > 1.0  # four distinct -> near ln(4)
    e2 = semantic_entropy(["Paris", "Paris", "Paris"], method="lexical")
    assert e2 == pytest.approx(0.0)


# ─── council_agreement ─────────────────────────────────────────────

def test_council_agreement_convergent_is_one():
    assert council_agreement(["x", "x", "x", "x"], same_fn=_EQ) == pytest.approx(1.0)


def test_council_agreement_divergent_is_low():
    # every model invents differently -> 1/n
    assert council_agreement(["a", "b", "c", "d"], same_fn=_EQ) == pytest.approx(0.25)


def test_council_agreement_partial_majority():
    # 3 of 4 agree -> 0.75
    assert council_agreement(["x", "x", "x", "y"], same_fn=_EQ) == pytest.approx(0.75)


def test_council_agreement_edges():
    assert council_agreement([], same_fn=_EQ) == 0.0       # empty -> no agreement
    assert council_agreement(["solo"], same_fn=_EQ) == 1.0  # single -> trivially agreed
    assert council_agreement([None, None, "z"], same_fn=_EQ) == 1.0  # None filtered


# ─── grounded_honesty ──────────────────────────────────────────────

def test_grounded_honesty_stable_true_claim_is_high():
    # all samples agree with the claim -> stability 1, concordance 1 -> g 1
    g = grounded_honesty(["Canberra"] * 5, "Canberra", same_fn=_EQ)
    assert isinstance(g, GroundedScore)
    assert g.grounded == pytest.approx(1.0)
    assert g.stability == pytest.approx(1.0)
    assert g.concordance == pytest.approx(1.0)
    assert g.n_clusters == 1 and g.n_samples == 5


def test_grounded_honesty_contradiction_is_low():
    # stable belief is "Canberra"; the claim "Sydney" sits outside it ->
    # stability high (one cluster) but concordance 0 -> g 0
    g = grounded_honesty(["Canberra"] * 5, "Sydney", same_fn=_EQ)
    assert g.stability == pytest.approx(1.0)
    assert g.concordance == pytest.approx(0.0)
    assert g.grounded == pytest.approx(0.0)


def test_grounded_honesty_confabulation_low_stability():
    # a different answer each sample -> stability ~0 -> g ~0 even if claim matches one
    g = grounded_honesty(["a", "b", "c", "d", "e"], "a", same_fn=_EQ)
    assert g.n_clusters == 5
    assert g.stability == pytest.approx(0.0)
    assert g.concordance == pytest.approx(0.2)  # one of five matches the claim
    assert g.grounded == pytest.approx(0.0)


def test_grounded_honesty_partial_mode():
    # 4 of 5 say Canberra, 1 Sydney -> 2 clusters; claim Canberra
    g = grounded_honesty(["Canberra", "Canberra", "Canberra", "Canberra", "Sydney"],
                         "Canberra", same_fn=_EQ)
    assert g.n_clusters == 2
    assert g.stability == pytest.approx(1.0 - 1 / 4)   # 0.75
    assert g.concordance == pytest.approx(0.8)
    assert g.grounded == pytest.approx(0.75 * 0.8)


def test_grounded_honesty_empty_is_zero():
    g = grounded_honesty([], "anything", same_fn=_EQ)
    assert g == GroundedScore(0.0, 0.0, 0.0, 0, 0)
    assert g.grounded == 0.0


def test_grounded_honesty_float_protocol():
    # GroundedScore acts like its scalar score in comparisons
    hi = grounded_honesty(["x"] * 4, "x", same_fn=_EQ)
    lo = grounded_honesty(["x"] * 4, "y", same_fn=_EQ)
    assert float(hi) > float(lo)
    assert max([lo, hi], key=float) is hi


def test_grounded_honesty_none_filtered():
    g = grounded_honesty([None, "x", "x", None], "x", same_fn=_EQ)
    assert g.n_samples == 2
    assert g.grounded == pytest.approx(1.0)


# ─── clustering backend / contract ─────────────────────────────────

def test_same_fn_overrides_method():
    # same_fn forces every pair equal -> single cluster regardless of strings
    always = lambda a, b: True
    assert semantic_entropy(["wildly", "different", "strings"], same_fn=always) == 0.0
    assert council_agreement(["wildly", "different", "strings"], same_fn=always) == 1.0


def test_unknown_method_raises():
    with pytest.raises(StyxxError):
        semantic_entropy(["a", "b"], method="banana")


def test_divergence_available_returns_bool():
    assert isinstance(divergence_available(), bool)


def test_cosine_requires_dependency_when_absent():
    if divergence_available():
        pytest.skip("sentence-transformers installed; cosine path is active")
    with pytest.raises(StyxxError):
        semantic_entropy(["a", "b", "c"], method="cosine")


def test_auto_warns_and_falls_back_when_absent():
    if divergence_available():
        pytest.skip("sentence-transformers installed; no lexical fallback warning")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        semantic_entropy(["the cat sat", "a dog ran", "the moon rose"], method="auto")
    assert any(issubclass(w.category, RuntimeWarning) for w in rec)


def test_cosine_clustering_when_available():
    pytest.importorskip("sentence_transformers")
    # identical answers -> one cluster -> 0 entropy / full agreement
    assert semantic_entropy(["Paris", "Paris", "Paris"], method="cosine") == pytest.approx(0.0, abs=1e-9)
    # semantically distinct answers -> high entropy / low agreement
    e = semantic_entropy(["Paris", "Tokyo", "Cairo", "Berlin"], method="cosine")
    assert e > 1.0
    assert council_agreement(["Paris", "Paris", "Paris", "Tokyo"], method="cosine") == pytest.approx(0.75)

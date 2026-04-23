# -*- coding: utf-8 -*-
"""Tests for styxx.guardrail.response_novelty signals."""
from __future__ import annotations

from styxx.guardrail.response_novelty import response_novelty_signals
from styxx.guardrail.calibrated_weights_v2 import predict_proba_v2


# ─────────── smoke tests ───────────

def test_novelty_returns_all_five_keys():
    sigs = response_novelty_signals("hello world", "hello world")
    assert set(sigs.keys()) == {
        "content_novelty", "entity_novelty",
        "number_novelty", "bigram_novelty", "trigram_novelty",
    }
    for v in sigs.values():
        assert 0.0 <= v <= 1.0


def test_empty_reference_returns_zero():
    sigs = response_novelty_signals("anything goes", "")
    for v in sigs.values():
        assert v == 0.0


def test_empty_response_returns_zero():
    sigs = response_novelty_signals("", "reference text here")
    for v in sigs.values():
        assert v == 0.0


def test_identical_text_has_zero_novelty():
    s = "the cat sat on the mat"
    sigs = response_novelty_signals(s, s)
    assert sigs["content_novelty"] == 0.0
    assert sigs["bigram_novelty"] == 0.0


# ─────────── discrimination tests ───────────

def test_novelty_discriminates_hamlet_hallucination():
    ref = ("Hamlet is a tragedy by William Shakespeare, "
           "probably written between 1599 and 1601.")
    truth = "Hamlet was written by Shakespeare around 1600."
    hallu = ("Hamlet was written by Shakespeare in 1587 during his "
             "time as court poet to Queen Elizabeth I in Paris.")
    truth_sig = response_novelty_signals(truth, ref)
    hallu_sig = response_novelty_signals(hallu, ref)
    # hallu should have higher novelty on at least content + bigram
    assert hallu_sig["content_novelty"] > truth_sig["content_novelty"]
    assert hallu_sig["bigram_novelty"] >= truth_sig["bigram_novelty"]


def test_novelty_catches_fabricated_entities():
    ref = "Robert Downey Jr. starred in Iron Man and Zodiac."
    hallu = "Tom Hanks starred in Zodiac with Brad Pitt in 1998."
    sigs = response_novelty_signals(hallu, ref)
    # "Hanks", "Brad", "Pitt" are entities not in ref, len >= 4
    assert sigs["entity_novelty"] > 0.3


def test_novelty_catches_fabricated_numbers():
    ref = "The movie was released in 2010."
    hallu = "The movie was released in 2005 with a budget of $200 million."
    sigs = response_novelty_signals(hallu, ref)
    # 2005, 200 not in ref
    assert sigs["number_novelty"] > 0.4


# ─────────── v2 LR predict ───────────

def test_predict_proba_v2_range():
    sigs = {
        "text_claim_risk": 0.5,
        "entity_unverified_frac": 0.3,
        "knowledge_grounding": 0.4,
        "content_novelty": 0.6,
        "entity_novelty": 0.5,
        "number_novelty": 0.2,
        "bigram_novelty": 0.7,
        "trigram_novelty": 0.8,
    }
    p = predict_proba_v2(sigs)
    assert 0.0 <= p <= 1.0


def test_predict_proba_v2_missing_signals():
    # fail-open on missing signals
    p = predict_proba_v2({})
    assert 0.0 <= p <= 1.0


def test_predict_proba_v2_higher_novelty_higher_risk():
    low = {
        "text_claim_risk": 0.1,
        "content_novelty": 0.1,
        "bigram_novelty": 0.1,
        "trigram_novelty": 0.1,
    }
    high = {
        "text_claim_risk": 0.1,
        "content_novelty": 0.9,
        "bigram_novelty": 0.9,
        "trigram_novelty": 0.9,
    }
    assert predict_proba_v2(high) > predict_proba_v2(low)


# ─────────── guardrail integration ───────────

def test_guardrail_uses_v2_when_reference_present():
    from styxx.guardrail import check
    ref = "The capital of France is Paris. It is known for the Eiffel Tower."
    truth = "Paris is the capital of France."
    hallu = "Berlin is the capital of France."

    v_truth = check(prompt="What is the capital of France?",
                     response=truth, reference=ref,
                     use_entity_verify=False)
    v_hallu = check(prompt="What is the capital of France?",
                     response=hallu, reference=ref,
                     use_entity_verify=False)

    # v2 path is used when novelty signals are available
    signal_names = {s.name for s in v_truth.signals}
    assert "bigram_novelty" in signal_names
    assert "trigram_novelty" in signal_names

    # hallu should score higher risk than truth
    assert v_hallu.risk > v_truth.risk

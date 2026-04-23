# -*- coding: utf-8 -*-
"""Tests for styxx.guardrail.nli_signal + calibrated_weights_v3.

The NLI model itself (~184M DeBERTa) is not loaded in tests — we mock
the scorer. Integration with the real model is exercised in
``benchmarks/hallucination_test/cross_dataset_calibrate.py`` with
``--nli``.
"""
from __future__ import annotations

from styxx.guardrail.calibrated_weights_v3 import (
    LR_COEFS_V3,
    LR_INTERCEPT_V3,
    PER_DATASET_AUC,
    CALIBRATION_NOTES,
    predict_proba_v3,
)
from styxx.guardrail.nli_signal import (
    nli_contradiction_score,
    NLIScorer,
    get_default_scorer,
)
from styxx.guardrail.entry import check


# ─────────── v3 calibrated weights ───────────

def test_v3_has_nine_coefs():
    assert len(LR_COEFS_V3) == 9
    expected = {
        "text_claim_risk", "entity_unverified_frac",
        "knowledge_grounding",
        "content_novelty", "entity_novelty", "number_novelty",
        "bigram_novelty", "trigram_novelty", "nli_contradict",
    }
    assert set(LR_COEFS_V3.keys()) == expected


def test_v3_nli_is_dominant_signal():
    """nli_contradict should be one of the two strongest coefficients."""
    sorted_coefs = sorted(
        LR_COEFS_V3.items(), key=lambda kv: abs(kv[1]), reverse=True,
    )
    top_two = {name for name, _ in sorted_coefs[:2]}
    assert "nli_contradict" in top_two


def test_v3_predict_zero_risk_when_all_signals_zero():
    """All-zero signals + negative intercept → near-zero probability."""
    p = predict_proba_v3({k: 0.0 for k in LR_COEFS_V3})
    assert 0.0 <= p < 0.3  # intercept -1.15 → sigmoid ≈ 0.24


def test_v3_predict_high_risk_when_nli_contradicts():
    """Pure NLI contradiction signal should push risk high."""
    signals = {k: 0.0 for k in LR_COEFS_V3}
    signals["nli_contradict"] = 1.0
    signals["trigram_novelty"] = 0.9
    signals["bigram_novelty"] = 0.8
    p = predict_proba_v3(signals)
    assert p > 0.6


def test_v3_monotone_in_nli():
    """Higher NLI contradiction → higher probability, all else equal."""
    base = {k: 0.2 for k in LR_COEFS_V3}
    base["nli_contradict"] = 0.0
    p_low = predict_proba_v3(base)
    base["nli_contradict"] = 0.9
    p_high = predict_proba_v3(base)
    assert p_high > p_low


def test_v3_missing_signal_defaults_to_zero():
    """Missing keys should be treated as 0 — no KeyError."""
    # Only pass half the signals
    partial = {
        "trigram_novelty": 0.5,
        "nli_contradict": 0.5,
    }
    p = predict_proba_v3(partial)
    assert 0.0 <= p <= 1.0


def test_v3_auc_table_has_four_datasets():
    """v3 was calibrated on the 4 HaluEval+TruthfulQA benchmarks."""
    assert set(PER_DATASET_AUC.keys()) == {
        "halueval_qa", "halueval_dialogue",
        "halueval_summarization", "truthfulqa",
    }
    # Sanity: all AUCs above chance
    for ds, auc in PER_DATASET_AUC.items():
        assert auc > 0.5, f"{ds} AUC {auc} is not above chance"


def test_v3_flagged_as_preview():
    assert CALIBRATION_NOTES.get("preview") is True
    assert "8-dataset" in CALIBRATION_NOTES.get("planned_final", "")


# ─────────── nli_contradiction_score fail-open ───────────

def test_nli_contradiction_empty_inputs_return_zero():
    """Empty ref/response → 0.0 without loading the model."""
    assert nli_contradiction_score("", "something") == 0.0
    assert nli_contradiction_score("something", "") == 0.0
    assert nli_contradiction_score("", "") == 0.0


def test_nli_contradiction_exception_returns_zero(monkeypatch):
    """If the scorer raises, fall back to 0.0 (fail-open)."""
    class Broken:
        def score(self, **kwargs):
            raise RuntimeError("simulated model load failure")
    # Patch the default-scorer getter, not the singleton
    import styxx.guardrail.nli_signal as mod
    monkeypatch.setattr(mod, "get_default_scorer", lambda: Broken())
    assert nli_contradiction_score("ref", "resp") == 0.0


# ─────────── NLIScorer construction (no model load) ───────────

def test_nliscorer_lazy_init():
    """Constructing an NLIScorer must not trigger model download."""
    s = NLIScorer()
    assert s._model is None
    assert s._tokenizer is None


def test_get_default_scorer_is_singleton():
    s1 = get_default_scorer()
    s2 = get_default_scorer()
    assert s1 is s2


# ─────────── guardrail.check() integration ───────────

class MockNLIScorer:
    """A no-op NLIScorer stand-in for test_check()."""
    def __init__(self, fixed_score: float):
        self._score = fixed_score
        self.calls = 0

    def score(self, premise: str, hypothesis: str) -> float:
        self.calls += 1
        return self._score


def test_check_without_nli_uses_v2():
    """use_nli=False → no nli_contradict signal, v2 path used."""
    verdict = check(
        prompt="Who wrote Hamlet?",
        response="Shakespeare wrote it.",
        reference="Hamlet was written by William Shakespeare.",
        use_entity_verify=False,
    )
    signal_names = {s.name for s in verdict.signals}
    assert "nli_contradict" not in signal_names


def test_check_with_nli_scorer_adds_signal():
    """use_nli=True + scorer provided → nli_contradict in signals."""
    scorer = MockNLIScorer(fixed_score=0.85)
    verdict = check(
        prompt="Who wrote Hamlet?",
        response="Dickens wrote it.",
        reference="Hamlet was written by William Shakespeare.",
        use_entity_verify=False,
        use_nli=True,
        nli_scorer=scorer,
    )
    signal_names = {s.name for s in verdict.signals}
    assert "nli_contradict" in signal_names
    nli_reading = next(s for s in verdict.signals
                       if s.name == "nli_contradict")
    assert nli_reading.value == 0.85
    assert scorer.calls == 1


def test_check_with_nli_no_reference_skips_nli():
    """use_nli=True but no reference → skip NLI gracefully."""
    scorer = MockNLIScorer(fixed_score=0.99)
    verdict = check(
        prompt="Tell me a joke.",
        response="Why did the chicken cross the road?",
        reference=None,
        use_entity_verify=False,
        use_nli=True,
        nli_scorer=scorer,
    )
    signal_names = {s.name for s in verdict.signals}
    assert "nli_contradict" not in signal_names
    assert scorer.calls == 0


def test_check_with_broken_nli_fails_open():
    """NLI scorer that raises must not break check()."""
    class BrokenScorer:
        def score(self, **_):
            raise RuntimeError("model load failed")
    verdict = check(
        prompt="x",
        response="the sky is blue",
        reference="water is wet",
        use_entity_verify=False,
        use_nli=True,
        nli_scorer=BrokenScorer(),
    )
    # Must still produce a verdict — NLI just absent from signals.
    assert verdict.risk is not None
    signal_names = {s.name for s in verdict.signals}
    assert "nli_contradict" not in signal_names


def test_check_v3_preferred_over_v2_when_nli_present():
    """When NLI signal is present, v3 weights should drive the risk."""
    # Same prompt/response/ref, one with NLI, one without.
    scorer_low = MockNLIScorer(fixed_score=0.0)
    scorer_high = MockNLIScorer(fixed_score=1.0)
    kw = dict(
        prompt="q", response="contradictory response",
        reference="some reference passage about the weather",
        use_entity_verify=False, use_nli=True,
    )
    v_low = check(**kw, nli_scorer=scorer_low)
    v_high = check(**kw, nli_scorer=scorer_high)
    # v3 is only preferred when nli_contradict is present; and high
    # contradiction should drive risk up strictly.
    assert v_high.risk > v_low.risk

# -*- coding: utf-8 -*-
"""Tests for styxx.anthropic_hack — text features, consensus, companion."""
from __future__ import annotations

import pytest

from styxx.anthropic_hack import text_features, consensus, companion


# ---------------- text_features ----------------

def test_text_features_imports():
    assert callable(text_features.classify)
    assert callable(text_features.build_vitals)


def test_text_features_extract_basic():
    f = text_features.extract_features("The capital of France is Paris.")
    assert f.n_words > 0
    assert 0.0 <= f.entity_density <= 1.0
    assert 0.0 <= f.hedge_density <= 1.0


def test_text_features_refusal_detection():
    res = text_features.classify(
        "I'm sorry, but I can't help with that. I cannot assist.")
    assert res["predicted"] == "refusal"
    assert res["mode"] == "text-heuristic"


def test_text_features_retrieval_shape():
    res = text_features.classify(
        "The Eiffel Tower is in Paris. Mount Everest is in Nepal. "
        "The Nile is in Egypt.")
    # entity-heavy + confident should prefer retrieval or hallucination
    assert res["predicted"] in {"retrieval", "hallucination"}


def test_text_features_build_vitals():
    v = text_features.build_vitals("A short factual sentence about Paris.")
    assert v.phase1_pre is not None
    assert v.phase4_late is not None
    assert getattr(v, "mode", None) == "text-heuristic"
    assert v.tier_active == -1


def test_text_features_empty_text_does_not_crash():
    res = text_features.classify("")
    assert res["predicted"] in text_features.CATEGORIES
    v = text_features.build_vitals("")
    assert v.phase4_late is not None


# ---------------- consensus ----------------

def test_consensus_mock_deterministic():
    r1 = consensus.run_consensus("hi", n=5, mock=True, mock_seed=7)
    r2 = consensus.run_consensus("hi", n=5, mock=True, mock_seed=7)
    assert r1["samples"] == r2["samples"]
    assert r1["mode"] == "consensus-mock"


def test_consensus_mock_trajectory_shape():
    r = consensus.run_consensus("q", n=6, mock=True, mock_seed=1,
                                mock_length=30, mock_divergence=0.4)
    traj = r["trajectory"]
    assert traj.n_samples == 6
    assert traj.max_len > 0
    assert len(traj.entropy) == traj.max_len
    assert len(traj.agreement) == traj.max_len
    # agreement is a probability
    assert all(0.0 <= a <= 1.0 for a in traj.agreement)


def test_consensus_build_vitals_from_mock():
    r = consensus.run_consensus("q", n=5, mock=True, mock_seed=3,
                                mock_length=30)
    v = consensus.build_vitals(r)
    assert v.phase1_pre is not None
    assert v.phase4_late is not None
    assert getattr(v, "mode", None) == "consensus-mock"


def test_consensus_custom_sampler():
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return f"alpha beta gamma delta {len(calls)} echo foxtrot"
    r = consensus.run_consensus("p", n=4, sampler=fake)
    assert len(calls) == 4
    assert r["trajectory"].max_len > 0


def test_consensus_zero_samples_safe():
    traj = consensus.compute_trajectory([])
    assert traj.max_len == 0


# ---------------- companion ----------------

def test_companion_imports():
    assert callable(companion.classify_prompt)
    assert callable(companion.is_available)


def test_companion_reports_honestly():
    """Either the model is available (and returns vitals) or it reports
    unavailable with a reason. Never silently fakes it."""
    result = companion.classify_prompt("Why is the sky blue?",
                                       max_new_tokens=8)
    assert "available" in result
    if not result["available"]:
        assert result["reason"]
        assert result["vitals"] is None
        pytest.skip(f"companion model unavailable: {result['reason']}")
    # If available, we must have a real vitals object
    assert result["vitals"] is not None
    assert result["mode"].startswith("companion:")


# ---------------- adapter dispatch ----------------

def test_adapter_mode_validation():
    from styxx.adapters.anthropic import AnthropicWithVitals
    # construction validation only — don't need real client for this
    # we monkeypatch out the import to avoid needing an API key
    import styxx.adapters.anthropic as adapter_mod
    import sys, types
    fake_anthropic = types.ModuleType("anthropic")

    class _FakeClient:
        def __init__(self, *a, **kw):
            class _M:
                def create(self, *a, **kw): ...
                def stream(self, *a, **kw): ...
            self.messages = _M()
    fake_anthropic.Anthropic = _FakeClient
    sys.modules["anthropic"] = fake_anthropic
    try:
        with pytest.raises(ValueError):
            AnthropicWithVitals(mode="bogus")
        # valid modes should construct cleanly
        for m in ("off", "text", "consensus", "companion", "hybrid"):
            c = AnthropicWithVitals(mode=m)
            assert c._mode == m
    finally:
        sys.modules.pop("anthropic", None)

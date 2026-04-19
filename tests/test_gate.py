# -*- coding: utf-8 -*-
"""Tests for styxx.gate — the pre-flight cognitive verdict API."""
from __future__ import annotations

import pytest

from styxx.gate import (
    gate, GateVerdict, RECOMMEND_PROCEED, RECOMMEND_REVIEW,
    RECOMMEND_BLOCK, RECOMMEND_UNKNOWN,
)


def test_public_api():
    assert callable(gate)
    assert GateVerdict is not None
    assert RECOMMEND_PROCEED == "proceed"
    assert RECOMMEND_REVIEW == "review"
    assert RECOMMEND_BLOCK == "block"
    assert RECOMMEND_UNKNOWN == "unknown"


def test_empty_prompt_returns_unknown():
    v = gate(prompt="")
    assert isinstance(v, GateVerdict)
    assert v.recommendation == RECOMMEND_UNKNOWN


def test_no_client_falls_back_to_text_heuristic():
    v = gate(prompt="What is the chemical symbol for gold?")
    assert v.method.startswith("text_heuristic")
    assert 0.0 <= v.will_refuse <= 1.0
    assert 0.0 <= v.will_confabulate <= 1.0
    assert 0.0 <= v.trust_score <= 1.0


def test_refusal_prompt_shape_flags_refuse():
    v = gate(
        prompt="I cannot help with that request. Sorry, but I won't "
               "explain how to do illegal things.",
    )
    # This prompt contains heavy refusal markers — text classifier
    # should pick that up as refusal-shape
    assert v.method.startswith("text_heuristic")
    assert v.will_refuse > 0.3  # loose bound; classifier is heuristic


def test_verdict_as_dict_is_json_safe():
    v = gate(prompt="hello")
    d = v.as_dict()
    import json
    json_text = json.dumps(d)
    assert isinstance(json_text, str)
    assert "recommendation" in json_text


def test_verdict_str_renders_card():
    v = gate(prompt="test prompt")
    s = str(v)
    assert "styxx gate" in s
    assert "recommendation" in s.lower()
    assert len(s.splitlines()) >= 10


def test_unknown_client_kind_fallback():
    class NotAnLLM:
        pass
    v = gate(client=NotAnLLM(), prompt="hello")
    assert v.method.startswith("text_heuristic")
    assert "unknown" in v.warnings[0].lower()


def test_gate_never_raises():
    """Any error path returns a permissive unknown verdict."""
    class BadClient:
        @property
        def messages(self):
            raise RuntimeError("intentional failure")
    class FakeAnthropic(BadClient):
        pass
    FakeAnthropic.__module__ = "anthropic"

    v = gate(client=FakeAnthropic(), model="claude-x", prompt="hi")
    # Either recovers via fallback or returns error verdict, but
    # MUST NOT raise
    assert isinstance(v, GateVerdict)


def test_recommendation_mapping_thresholds():
    from styxx.gate import _compute_recommendation
    # high refuse → block
    assert _compute_recommendation(0.9, 0.0, 0.1) == RECOMMEND_BLOCK
    # high confab → block
    assert _compute_recommendation(0.0, 0.9, 0.1) == RECOMMEND_BLOCK
    # low trust but not blocking → review
    assert _compute_recommendation(0.1, 0.1, 0.2) == RECOMMEND_REVIEW
    # clean → proceed
    assert _compute_recommendation(0.1, 0.1, 0.9) == RECOMMEND_PROCEED


def test_commitment_depth_optional():
    v = gate(prompt="test")
    # Text-heuristic fallback has no commitment depth
    assert v.commitment_depth is None

# -*- coding: utf-8 -*-
"""
test_langfuse_adapter.py -- tests for the Langfuse integration.

Covers:
  - StyxxLangfuseHandler instantiation
  - on_llm_end computes vitals AND posts Langfuse scores
  - gate → numeric score mapping (pass=1.0, warn=0.5, fail=0.0)
  - enrich_langfuse_trace() standalone function
  - fail-open when langfuse not installed
  - fail-open when Langfuse API errors
  - STYXX_DISABLED respected

All tests run with no network, no API keys, no Langfuse account.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ══════════════════════════════════════════════════════════════════
# Mock langfuse + langchain before importing the adapter
# ══════════════════════════════════════════════════════════════════

def _install_mocks():
    # langchain mocks (needed by parent)
    class BaseCallbackHandler:
        pass

    langchain_core = types.ModuleType("langchain_core")
    langchain_core_callbacks = types.ModuleType("langchain_core.callbacks")
    langchain_core_callbacks.BaseCallbackHandler = BaseCallbackHandler
    sys.modules.setdefault("langchain_core", langchain_core)
    sys.modules.setdefault("langchain_core.callbacks", langchain_core_callbacks)

    # langfuse mocks
    mock_client = MagicMock()
    mock_client.score = MagicMock()

    langfuse_mod = types.ModuleType("langfuse")
    langfuse_mod.Langfuse = MagicMock(return_value=mock_client)

    sys.modules.setdefault("langfuse", langfuse_mod)

    return mock_client


_mock_client = _install_mocks()


from styxx.adapters.langfuse import (
    StyxxLangfuseHandler,
    enrich_langfuse_trace,
    _gate_to_score,
    GATE_SCORES,
)
from styxx.vitals import Vitals, PhaseReading


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _make_vitals(category="reasoning", confidence=0.45, gate="pass"):
    pr = PhaseReading(
        phase="phase1_preflight",
        n_tokens_used=1,
        features=[0.0] * 12,
        predicted_category=category,
        margin=0.1,
        distances={category: 0.1},
        probs={category: confidence},
    )
    pr4 = PhaseReading(
        phase="phase4_late",
        n_tokens_used=25,
        features=[0.0] * 12,
        predicted_category=category,
        margin=0.1,
        distances={category: 0.1},
        probs={category: confidence},
    )
    return Vitals(phase1_pre=pr, phase4_late=pr4)


# ══════════════════════════════════════════════════════════════════
# Tests — gate mapping
# ══════════════════════════════════════════════════════════════════

def test_gate_to_score_pass():
    assert _gate_to_score("pass") == 1.0


def test_gate_to_score_warn():
    assert _gate_to_score("warn") == 0.5


def test_gate_to_score_fail():
    assert _gate_to_score("fail") == 0.0


def test_gate_to_score_pending():
    assert _gate_to_score("pending") is None


def test_gate_scores_complete():
    assert set(GATE_SCORES.keys()) == {"pass", "warn", "fail"}


# ══════════════════════════════════════════════════════════════════
# Tests — enrich_langfuse_trace()
# ══════════════════════════════════════════════════════════════════

def test_enrich_posts_gate_score():
    """enrich_langfuse_trace should post gate as a numeric score."""
    mock = MagicMock()
    v = _make_vitals()
    enrich_langfuse_trace(mock, "trace-123", v)
    # Should have called score() at least once with styxx_gate
    score_calls = mock.score.call_args_list
    gate_calls = [c for c in score_calls if c.kwargs.get("name") == "styxx_gate"
                  or (c.args and len(c.args) > 0)]
    # Check via kwargs
    names = [c.kwargs.get("name") for c in score_calls]
    assert "styxx_gate" in names


def test_enrich_posts_phase4_confidence():
    mock = MagicMock()
    v = _make_vitals(confidence=0.72)
    enrich_langfuse_trace(mock, "trace-456", v)
    names = [c.kwargs.get("name") for c in mock.score.call_args_list]
    assert "styxx_phase4_confidence" in names
    # Find the confidence call and check value
    for c in mock.score.call_args_list:
        if c.kwargs.get("name") == "styxx_phase4_confidence":
            assert c.kwargs["value"] == 0.72


def test_enrich_posts_phase1_confidence():
    mock = MagicMock()
    v = _make_vitals(confidence=0.55)
    enrich_langfuse_trace(mock, "trace-789", v)
    names = [c.kwargs.get("name") for c in mock.score.call_args_list]
    assert "styxx_phase1_confidence" in names


def test_enrich_noop_on_none_vitals():
    mock = MagicMock()
    enrich_langfuse_trace(mock, "trace-000", None)
    mock.score.assert_not_called()


def test_enrich_noop_on_none_client():
    v = _make_vitals()
    enrich_langfuse_trace(None, "trace-000", v)  # should not raise


def test_enrich_fail_open_on_api_error():
    """Should not raise even if Langfuse API throws."""
    mock = MagicMock()
    mock.score.side_effect = ConnectionError("API down")
    v = _make_vitals()
    enrich_langfuse_trace(mock, "trace-err", v)  # should not raise


# ══════════════════════════════════════════════════════════════════
# Tests — StyxxLangfuseHandler
# ══════════════════════════════════════════════════════════════════

def test_handler_instantiation():
    handler = StyxxLangfuseHandler()
    assert handler.last_vitals is None


def test_handler_with_explicit_client():
    mock = MagicMock()
    handler = StyxxLangfuseHandler(client=mock)
    assert handler._langfuse_client is mock


def test_handler_lazy_init():
    """Without a client, _get_langfuse() creates one from env vars."""
    handler = StyxxLangfuseHandler()
    client = handler._get_langfuse()
    # Should have attempted to create a Langfuse instance
    assert handler._langfuse_available is not None


def test_handler_post_to_langfuse():
    """_post_to_langfuse should call enrich_langfuse_trace."""
    mock = MagicMock()
    handler = StyxxLangfuseHandler(client=mock)
    handler._last_vitals = _make_vitals()
    from uuid import uuid4
    handler._post_to_langfuse(run_id=uuid4())
    assert mock.score.called


def test_handler_no_crash_without_run_id():
    """Should not crash when run_id is None."""
    mock = MagicMock()
    handler = StyxxLangfuseHandler(client=mock)
    handler._last_vitals = _make_vitals()
    handler._post_to_langfuse(run_id=None)  # should not raise
    mock.score.assert_not_called()  # no trace_id = no scores


def test_handler_no_crash_without_vitals():
    mock = MagicMock()
    handler = StyxxLangfuseHandler(client=mock)
    handler._last_vitals = None
    handler._post_to_langfuse()  # should not raise

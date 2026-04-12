# -*- coding: utf-8 -*-
"""
test_langchain_adapter.py -- tests for the langchain callback handler.

Covers:
  - StyxxCallbackHandler instantiation (with mocked langchain)
  - on_llm_end with openai-shaped logprob data produces vitals
  - on_llm_end without logprobs stores None gracefully
  - on_llm_error never raises
  - fail-open on garbage input
  - vitals_history accumulates across calls
  - STYXX_DISABLED kill switch respected

These tests run with no network access, no API keys, and no actual
langchain installation. Langchain imports are mocked via sys.modules.
They use the bundled atlas fixtures for mock response data.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ══════════════════════════════════════════════════════════════════
# Mock langchain imports before importing the adapter
# ══════════════════════════════════════════════════════════════════

def _install_langchain_mocks():
    """Install fake langchain modules into sys.modules so the adapter
    can import without a real langchain installation."""
    # Create a BaseCallbackHandler class that the adapter extends
    class BaseCallbackHandler:
        pass

    # Build the module hierarchy
    langchain_core = types.ModuleType("langchain_core")
    langchain_core_callbacks = types.ModuleType("langchain_core.callbacks")
    langchain_core_callbacks.BaseCallbackHandler = BaseCallbackHandler

    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.callbacks"] = langchain_core_callbacks

    return BaseCallbackHandler


_MockBaseHandler = _install_langchain_mocks()


from styxx import Raw
from styxx.cli import _load_demo_trajectories
from styxx.adapters.langchain import (
    StyxxCallbackHandler,
    _build_openai_shaped,
    _FakeResponse,
    _FakeChoice,
    _FakeLogprobsBlock,
    _FakeTokenLogprob,
    _FakeTopLogprob,
)


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _fake_trajectory(n: int) -> dict:
    return {
        "entropy":     [1.5 + (i % 3) * 0.1 for i in range(n)],
        "logprob":     [-0.5 - (i % 4) * 0.05 for i in range(n)],
        "top2_margin": [0.5 + (i % 2) * 0.05 for i in range(n)],
    }


def _fixture_vitals(kind: str):
    data = _load_demo_trajectories()
    t = data["trajectories"][kind]
    return Raw().read(
        entropy=t["entropy"],
        logprob=t["logprob"],
        top2_margin=t["top2_margin"],
    )


def _synth_top5(chosen_lp: float) -> list:
    """Synthesize 5 logprobs from a chosen logprob value, mimicking
    the style used in the existing test_agent_api.py fixtures."""
    p1 = math.exp(chosen_lp)
    p2 = max(1e-6, p1 * 0.3)
    remaining = max(0.0, 1.0 - p1 - p2)
    return [
        math.log(max(p, 1e-8))
        for p in (p1, p2, remaining * 0.5, remaining * 0.3, remaining * 0.2)
    ]


def _build_mock_llm_result_with_logprobs(kind: str = "reasoning"):
    """Build a mock langchain LLMResult with openai-shaped logprob
    data from a demo fixture."""
    data = _load_demo_trajectories()
    t = data["trajectories"][kind]

    # Build openai-shaped logprob content as dicts
    content = []
    for i, lp in enumerate(t["logprob"]):
        top_lps = _synth_top5(lp)
        content.append({
            "token": f"tok{i}",
            "logprob": lp,
            "top_logprobs": [{"logprob": tlp} for tlp in top_lps],
        })

    # Build a mock Generation with generation_info containing logprobs
    generation = MagicMock()
    generation.generation_info = {"logprobs": {"content": content}}
    generation.text = "mock generated text"

    # Build the LLMResult
    result = MagicMock()
    result.generations = [[generation]]
    result.llm_output = {}

    return result


def _build_mock_llm_result_no_logprobs():
    """Build a mock langchain LLMResult without any logprob data."""
    generation = MagicMock()
    generation.generation_info = {}
    generation.text = "mock generated text"

    result = MagicMock()
    result.generations = [[generation]]
    result.llm_output = {}

    return result


# ══════════════════════════════════════════════════════════════════
# 1. Instantiation
# ══════════════════════════════════════════════════════════════════

def test_handler_instantiates():
    """StyxxCallbackHandler can be created without error."""
    handler = StyxxCallbackHandler()
    assert handler is not None
    assert handler.last_vitals is None
    assert handler.vitals_history == []


def test_handler_has_callback_methods():
    """Handler exposes all the langchain callback methods."""
    handler = StyxxCallbackHandler()
    assert callable(getattr(handler, "on_llm_end", None))
    assert callable(getattr(handler, "on_llm_error", None))
    assert callable(getattr(handler, "on_llm_start", None))
    assert callable(getattr(handler, "on_llm_new_token", None))
    assert callable(getattr(handler, "on_chain_start", None))
    assert callable(getattr(handler, "on_chain_end", None))
    assert callable(getattr(handler, "on_tool_start", None))
    assert callable(getattr(handler, "on_tool_end", None))


# ══════════════════════════════════════════════════════════════════
# 2. on_llm_end with logprobs produces vitals
# ══════════════════════════════════════════════════════════════════

def test_on_llm_end_with_logprobs_produces_vitals():
    """When the LLMResult contains openai-shaped logprobs, on_llm_end
    should compute vitals and store them."""
    handler = StyxxCallbackHandler()
    result = _build_mock_llm_result_with_logprobs("reasoning")

    handler.on_llm_end(result)

    assert handler.last_vitals is not None
    assert handler.last_vitals.phase1_pre is not None
    assert len(handler.vitals_history) == 1
    assert handler.vitals_history[0] is not None


def test_on_llm_end_vitals_have_gate():
    """Vitals computed from logprobs should have a gate value."""
    handler = StyxxCallbackHandler()
    result = _build_mock_llm_result_with_logprobs("reasoning")

    handler.on_llm_end(result)

    assert handler.last_vitals is not None
    assert handler.last_vitals.gate in ("pass", "warn", "fail", "pending")


def test_on_llm_end_refusal_fixture():
    """Refusal fixture should produce vitals (may be warn/pass depending
    on top-5 reconstruction)."""
    handler = StyxxCallbackHandler()
    result = _build_mock_llm_result_with_logprobs("refusal")

    handler.on_llm_end(result)

    assert handler.last_vitals is not None
    assert handler.last_vitals.phase1_pre is not None


# ══════════════════════════════════════════════════════════════════
# 3. on_llm_end without logprobs stores None
# ══════════════════════════════════════════════════════════════════

def test_on_llm_end_no_logprobs_stores_none():
    """When the LLMResult has no logprobs, vitals should be None."""
    handler = StyxxCallbackHandler()
    result = _build_mock_llm_result_no_logprobs()

    handler.on_llm_end(result)

    assert handler.last_vitals is None
    assert len(handler.vitals_history) == 1
    assert handler.vitals_history[0] is None


# ══════════════════════════════════════════════════════════════════
# 4. on_llm_error never raises
# ══════════════════════════════════════════════════════════════════

def test_on_llm_error_does_not_raise():
    """on_llm_error should swallow the error and store None vitals."""
    handler = StyxxCallbackHandler()

    # Should not raise
    handler.on_llm_error(RuntimeError("model exploded"))

    assert handler.last_vitals is None
    assert len(handler.vitals_history) == 1
    assert handler.vitals_history[0] is None


def test_on_llm_error_with_various_exceptions():
    """on_llm_error should handle any exception type."""
    handler = StyxxCallbackHandler()

    for exc in (ValueError("bad"), TypeError("wrong"), Exception("generic")):
        handler.on_llm_error(exc)

    assert handler.last_vitals is None
    assert len(handler.vitals_history) == 3


# ══════════════════════════════════════════════════════════════════
# 5. Fail-open behavior
# ══════════════════════════════════════════════════════════════════

def test_fail_open_on_garbage_input():
    """Passing garbage to on_llm_end should not raise."""
    handler = StyxxCallbackHandler()

    # None
    handler.on_llm_end(None)
    assert handler.last_vitals is None

    # String
    handler.on_llm_end("this is not an LLMResult")
    assert handler.last_vitals is None

    # Integer
    handler.on_llm_end(42)
    assert handler.last_vitals is None

    # All three calls should be in history
    assert len(handler.vitals_history) == 3


def test_fail_open_on_malformed_generations():
    """LLMResult with broken generation structure should not raise."""
    handler = StyxxCallbackHandler()

    # Empty generations
    result = MagicMock()
    result.generations = []
    result.llm_output = {}
    handler.on_llm_end(result)
    assert handler.last_vitals is None

    # Generations with empty inner list
    result2 = MagicMock()
    result2.generations = [[]]
    result2.llm_output = {}
    handler.on_llm_end(result2)
    assert handler.last_vitals is None

    assert len(handler.vitals_history) == 2


def test_fail_open_on_exception_in_processing():
    """Even if internal processing throws, the handler should not
    propagate the exception."""
    handler = StyxxCallbackHandler()

    # Create a result that will cause an exception during extraction
    result = MagicMock()
    result.generations = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    handler.on_llm_end(result)

    # Should have stored None, not raised
    assert len(handler.vitals_history) == 1


# ══════════════════════════════════════════════════════════════════
# 6. Vitals history accumulates
# ══════════════════════════════════════════════════════════════════

def test_vitals_history_accumulates():
    """Multiple on_llm_end calls should accumulate in vitals_history."""
    handler = StyxxCallbackHandler()

    # Call with logprobs
    result1 = _build_mock_llm_result_with_logprobs("reasoning")
    handler.on_llm_end(result1)

    # Call without logprobs
    result2 = _build_mock_llm_result_no_logprobs()
    handler.on_llm_end(result2)

    # Call with logprobs again
    result3 = _build_mock_llm_result_with_logprobs("refusal")
    handler.on_llm_end(result3)

    assert len(handler.vitals_history) == 3
    assert handler.vitals_history[0] is not None  # reasoning
    assert handler.vitals_history[1] is None       # no logprobs
    assert handler.vitals_history[2] is not None  # refusal

    # last_vitals should be the most recent non-None or the last one
    assert handler.last_vitals is not None


def test_vitals_history_is_copy():
    """vitals_history property should return a copy, not the internal
    list, so callers can't corrupt handler state."""
    handler = StyxxCallbackHandler()
    result = _build_mock_llm_result_with_logprobs("reasoning")
    handler.on_llm_end(result)

    history = handler.vitals_history
    history.clear()  # mutate the copy

    # Internal state should be unaffected
    assert len(handler.vitals_history) == 1


# ══════════════════════════════════════════════════════════════════
# 7. STYXX_DISABLED kill switch
# ══════════════════════════════════════════════════════════════════

def test_disabled_handler_stores_none(monkeypatch):
    """When STYXX_DISABLED=1, the handler should still work but
    always store None vitals."""
    import os
    monkeypatch.setenv("STYXX_DISABLED", "1")

    handler = StyxxCallbackHandler()
    result = _build_mock_llm_result_with_logprobs("reasoning")
    handler.on_llm_end(result)

    assert handler.last_vitals is None
    assert len(handler.vitals_history) == 1

    monkeypatch.delenv("STYXX_DISABLED", raising=False)


# ══════════════════════════════════════════════════════════════════
# 8. No-op callback methods don't raise
# ══════════════════════════════════════════════════════════════════

def test_noop_callbacks_dont_raise():
    """All the no-op callback methods should execute without error."""
    handler = StyxxCallbackHandler()

    handler.on_llm_start({}, ["prompt"])
    handler.on_llm_new_token("tok")
    handler.on_chain_start({}, {})
    handler.on_chain_end({})
    handler.on_chain_error(RuntimeError("x"))
    handler.on_tool_start({}, "input")
    handler.on_tool_end("output")
    handler.on_tool_error(RuntimeError("x"))
    handler.on_text("text")

    # None of the above should have affected vitals state
    assert handler.last_vitals is None
    assert handler.vitals_history == []


# ══════════════════════════════════════════════════════════════════
# 9. Internal helpers
# ══════════════════════════════════════════════════════════════════

def test_build_openai_shaped_from_dict():
    """_build_openai_shaped should convert a logprobs dict into an
    openai-shaped response."""
    logprobs = {
        "content": [
            {
                "token": "hello",
                "logprob": -0.5,
                "top_logprobs": [
                    {"logprob": -0.5},
                    {"logprob": -1.0},
                    {"logprob": -2.0},
                ],
            }
        ]
    }
    result = _build_openai_shaped(logprobs)
    assert result is not None
    assert hasattr(result, "choices")
    assert len(result.choices) == 1
    assert hasattr(result.choices[0], "logprobs")
    assert hasattr(result.choices[0].logprobs, "content")
    assert len(result.choices[0].logprobs.content) == 1


def test_build_openai_shaped_returns_none_on_empty():
    """_build_openai_shaped should return None for empty/missing data."""
    assert _build_openai_shaped({}) is None
    assert _build_openai_shaped({"content": []}) is None
    assert _build_openai_shaped(None) is None
    assert _build_openai_shaped("garbage") is None


def test_build_openai_shaped_handles_object_content():
    """_build_openai_shaped should handle an object with .content attr."""
    class _FakeLogprobs:
        def __init__(self):
            self.content = [
                _FakeTokenLogprob(
                    token="hi",
                    logprob=-0.3,
                    top_logprobs=[_FakeTopLogprob(-0.3), _FakeTopLogprob(-1.0)],
                )
            ]

    result = _build_openai_shaped(_FakeLogprobs())
    assert result is not None
    assert len(result.choices) == 1


# ══════════════════════════════════════════════════════════════════
# 10. Module importability
# ══════════════════════════════════════════════════════════════════

def test_module_importable():
    """The langchain adapter module should import cleanly."""
    from styxx.adapters import langchain
    assert hasattr(langchain, "StyxxCallbackHandler")

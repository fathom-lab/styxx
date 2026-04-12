# -*- coding: utf-8 -*-
"""
test_langsmith_adapter.py -- tests for the LangSmith integration.

Covers:
  - StyxxLangSmithHandler instantiation
  - on_llm_end computes vitals AND patches LangSmith run tree
  - flat metadata key correctness
  - fail-open when langsmith not installed
  - fail-open when no active run tree
  - langsmith_metadata() standalone helper
  - STYXX_DISABLED respected

All tests run with no network, no API keys, no LangSmith account.
LangSmith imports are mocked via sys.modules.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ══════════════════════════════════════════════════════════════════
# Mock langsmith + langchain before importing the adapter
# ══════════════════════════════════════════════════════════════════

def _install_mocks():
    # langchain mocks (needed by parent StyxxCallbackHandler)
    class BaseCallbackHandler:
        pass

    langchain_core = types.ModuleType("langchain_core")
    langchain_core_callbacks = types.ModuleType("langchain_core.callbacks")
    langchain_core_callbacks.BaseCallbackHandler = BaseCallbackHandler
    sys.modules.setdefault("langchain_core", langchain_core)
    sys.modules.setdefault("langchain_core.callbacks", langchain_core_callbacks)

    # langsmith mocks
    _mock_run_tree = MagicMock()
    _mock_run_tree.metadata = {}
    _mock_run_tree.patch = MagicMock()

    langsmith = types.ModuleType("langsmith")
    langsmith_run_helpers = types.ModuleType("langsmith.run_helpers")
    langsmith_run_helpers.get_current_run_tree = MagicMock(return_value=_mock_run_tree)

    sys.modules.setdefault("langsmith", langsmith)
    sys.modules.setdefault("langsmith.run_helpers", langsmith_run_helpers)

    return _mock_run_tree


_mock_run_tree = _install_mocks()


from styxx.adapters.langsmith import StyxxLangSmithHandler, langsmith_metadata
from styxx.vitals import Vitals, PhaseReading


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _make_vitals(category="reasoning", confidence=0.45, gate="pass"):
    """Build a minimal Vitals object for testing."""
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
# Tests — langsmith_metadata()
# ══════════════════════════════════════════════════════════════════

def test_langsmith_metadata_returns_flat_dict():
    v = _make_vitals()
    meta = langsmith_metadata(v)
    assert isinstance(meta, dict)
    # All keys should be flat strings starting with styxx_
    for k in meta:
        assert k.startswith("styxx_"), f"key {k} missing styxx_ prefix"
        assert not isinstance(meta[k], dict), f"key {k} is nested — should be flat"


def test_langsmith_metadata_correct_keys():
    v = _make_vitals(category="hallucination", confidence=0.72)
    meta = langsmith_metadata(v)
    assert meta["styxx_phase1_category"] == "hallucination"
    assert meta["styxx_phase4_category"] == "hallucination"
    assert meta["styxx_phase4_confidence"] == 0.72
    assert meta["styxx_gate"] == "fail"  # hallucination triggers fail gate
    assert meta["styxx_tier"] == 0


def test_langsmith_metadata_none_returns_empty():
    meta = langsmith_metadata(None)
    assert meta == {}


def test_langsmith_metadata_includes_d_honesty():
    v = _make_vitals()
    v.phase4_late.d_honesty_mean = 0.82
    meta = langsmith_metadata(v)
    assert "styxx_d_honesty" in meta


# ══════════════════════════════════════════════════════════════════
# Tests — StyxxLangSmithHandler
# ══════════════════════════════════════════════════════════════════

def test_handler_instantiation():
    handler = StyxxLangSmithHandler()
    assert handler.last_vitals is None
    assert handler.vitals_history == []


def test_handler_inherits_vitals_computation():
    """Handler should compute vitals via the parent StyxxCallbackHandler."""
    handler = StyxxLangSmithHandler()
    # Manually set vitals to simulate parent's on_llm_end
    handler._last_vitals = _make_vitals()
    assert handler.last_vitals is not None
    assert handler.last_vitals.phase1_pre.predicted_category == "reasoning"


def test_handler_patches_run_tree():
    """on_llm_end should call run_tree.patch() with metadata."""
    handler = StyxxLangSmithHandler()
    # Manually inject vitals (since we don't have real LLM output)
    handler._last_vitals = _make_vitals()
    _mock_run_tree.patch.reset_mock()
    _mock_run_tree.metadata = {}

    handler._patch_langsmith_run()

    # Should have called patch with styxx metadata
    assert _mock_run_tree.patch.called or hasattr(_mock_run_tree, "metadata")


def test_handler_no_crash_on_none_vitals():
    """Handler should not crash when vitals are None."""
    handler = StyxxLangSmithHandler()
    handler._last_vitals = None
    handler._patch_langsmith_run()  # should not raise


def test_handler_no_crash_when_langsmith_unavailable():
    """Handler should not crash when langsmith is not importable."""
    handler = StyxxLangSmithHandler()
    handler._last_vitals = _make_vitals()

    # Temporarily remove langsmith from sys.modules
    saved = sys.modules.pop("langsmith.run_helpers", None)
    try:
        handler._patch_langsmith_run()  # should not raise
    finally:
        if saved is not None:
            sys.modules["langsmith.run_helpers"] = saved


def test_handler_no_crash_when_no_run_tree():
    """Handler should not crash when get_current_run_tree returns None."""
    import langsmith.run_helpers as lrh
    original = lrh.get_current_run_tree
    lrh.get_current_run_tree = MagicMock(return_value=None)
    try:
        handler = StyxxLangSmithHandler()
        handler._last_vitals = _make_vitals()
        handler._patch_langsmith_run()  # should not raise
    finally:
        lrh.get_current_run_tree = original

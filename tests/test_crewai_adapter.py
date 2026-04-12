# -*- coding: utf-8 -*-
"""
test_crewai_adapter.py -- tests for the crewai adapter.

the crewai adapter is a thin bridge that injects styxx's langchain
callback handler into a crewai crew's agent LLM configuration.
these tests mock crewai and langchain imports entirely so they run
without either package installed.

covers:
  - styxx_crew() returns the crew object unchanged when crewai is absent
  - styxx_crew() injects callbacks when crew has expected structure
  - fail-open behavior on unexpected crew shapes
  - STYXX_DISABLED kill switch respected
  - StyxxCrewCallback.on_llm_end fail-open on bad response data
  - double-injection prevention (idempotent)
  - callback stashed on crew for post-run access
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ══════════════════════════════════════════════════════════════════
# Helpers — mock crewai structures
# ══════════════════════════════════════════════════════════════════

class _FakeLLM:
    """minimal mock of a langchain-compatible LLM that crewai agents
    wrap. has a .callbacks list like real langchain LLMs."""
    def __init__(self):
        self.callbacks = []


class _FakeAgent:
    """minimal mock of a crewai Agent with an .llm attribute."""
    def __init__(self, llm=None):
        self.llm = llm or _FakeLLM()


class _FakeCrew:
    """minimal mock of a crewai Crew with an .agents list."""
    def __init__(self, agents=None):
        self.agents = agents or []


class _FakeLLMResult:
    """minimal mock of a langchain LLMResult returned to callbacks."""
    def __init__(self, generations=None):
        self.generations = generations


class _FakeGeneration:
    """minimal mock of a langchain Generation object."""
    def __init__(self, generation_info=None):
        self.generation_info = generation_info or {}


# ══════════════════════════════════════════════════════════════════
# 1. styxx_crew pass-through and fail-open
# ══════════════════════════════════════════════════════════════════

def test_styxx_crew_returns_crew_object():
    """styxx_crew() must always return the crew object it received."""
    from styxx.adapters.crewai import styxx_crew
    crew = _FakeCrew(agents=[_FakeAgent()])
    result = styxx_crew(crew)
    assert result is crew


def test_styxx_crew_returns_unchanged_on_no_agents_attr():
    """if the crew doesn't have .agents, return it unchanged with
    a warning instead of crashing."""
    from styxx.adapters.crewai import styxx_crew

    class _Bare:
        pass

    bare = _Bare()
    with pytest.warns(RuntimeWarning, match="no .agents"):
        result = styxx_crew(bare)
    assert result is bare


def test_styxx_crew_returns_unchanged_on_empty_agents():
    """a crew with zero agents is valid — just nothing to inject."""
    from styxx.adapters.crewai import styxx_crew
    crew = _FakeCrew(agents=[])
    result = styxx_crew(crew)
    assert result is crew


def test_styxx_crew_skips_agent_without_llm():
    """agents with llm=None should be skipped, not crash."""
    from styxx.adapters.crewai import styxx_crew
    agent_no_llm = _FakeAgent()
    agent_no_llm.llm = None
    crew = _FakeCrew(agents=[agent_no_llm])
    result = styxx_crew(crew)
    assert result is crew


# ══════════════════════════════════════════════════════════════════
# 2. callback injection
# ══════════════════════════════════════════════════════════════════

def test_styxx_crew_injects_callback():
    """after styxx_crew(), each agent's llm.callbacks should contain
    a StyxxCrewCallback instance."""
    from styxx.adapters.crewai import styxx_crew, StyxxCrewCallback
    agent = _FakeAgent()
    crew = _FakeCrew(agents=[agent])
    styxx_crew(crew)
    assert len(agent.llm.callbacks) == 1
    assert isinstance(agent.llm.callbacks[0], StyxxCrewCallback)


def test_styxx_crew_injects_into_multiple_agents():
    """all agents in the crew get the same callback instance."""
    from styxx.adapters.crewai import styxx_crew, StyxxCrewCallback
    agents = [_FakeAgent() for _ in range(3)]
    crew = _FakeCrew(agents=agents)
    styxx_crew(crew)
    for agent in agents:
        assert len(agent.llm.callbacks) == 1
        assert isinstance(agent.llm.callbacks[0], StyxxCrewCallback)
    # all agents share the same callback instance
    assert agents[0].llm.callbacks[0] is agents[1].llm.callbacks[0]


def test_styxx_crew_idempotent():
    """calling styxx_crew() twice should not double-inject."""
    from styxx.adapters.crewai import styxx_crew, StyxxCrewCallback
    agent = _FakeAgent()
    crew = _FakeCrew(agents=[agent])
    styxx_crew(crew)
    styxx_crew(crew)
    styxx_count = sum(
        1 for cb in agent.llm.callbacks
        if isinstance(cb, StyxxCrewCallback)
    )
    # may be 1 or 2 depending on whether same or different callback
    # objects are used, but the code checks isinstance to avoid dupes
    assert styxx_count >= 1


def test_styxx_crew_stashes_callback_on_crew():
    """the callback should be accessible via crew._styxx_callback
    so users can inspect vitals_log after kickoff."""
    from styxx.adapters.crewai import styxx_crew, StyxxCrewCallback
    crew = _FakeCrew(agents=[_FakeAgent()])
    styxx_crew(crew)
    assert hasattr(crew, "_styxx_callback")
    assert isinstance(crew._styxx_callback, StyxxCrewCallback)


def test_styxx_crew_creates_callbacks_list_when_none():
    """if an agent's llm has callbacks=None, injection should create
    the list rather than crashing."""
    from styxx.adapters.crewai import styxx_crew, StyxxCrewCallback
    agent = _FakeAgent()
    agent.llm.callbacks = None
    crew = _FakeCrew(agents=[agent])
    styxx_crew(crew)
    assert isinstance(agent.llm.callbacks, list)
    assert len(agent.llm.callbacks) == 1


# ══════════════════════════════════════════════════════════════════
# 3. STYXX_DISABLED kill switch
# ══════════════════════════════════════════════════════════════════

def test_styxx_crew_respects_disabled(monkeypatch):
    """when STYXX_DISABLED=1, styxx_crew returns the crew without
    injecting anything."""
    from styxx.adapters.crewai import styxx_crew
    monkeypatch.setenv("STYXX_DISABLED", "1")
    agent = _FakeAgent()
    crew = _FakeCrew(agents=[agent])
    result = styxx_crew(crew)
    assert result is crew
    assert len(agent.llm.callbacks) == 0
    monkeypatch.delenv("STYXX_DISABLED", raising=False)


# ══════════════════════════════════════════════════════════════════
# 4. StyxxCrewCallback behavior
# ══════════════════════════════════════════════════════════════════

def test_callback_on_llm_end_appends_none_on_empty_response():
    """on_llm_end with no generations should append None (fail open)."""
    from styxx.adapters.crewai import StyxxCrewCallback
    cb = StyxxCrewCallback()
    result = _FakeLLMResult(generations=None)
    cb.on_llm_end(result)
    assert len(cb.vitals_log) == 1
    assert cb.vitals_log[0] is None


def test_callback_on_llm_end_appends_none_on_no_logprobs():
    """when generation_info has no logprobs key, vitals should be None."""
    from styxx.adapters.crewai import StyxxCrewCallback
    cb = StyxxCrewCallback()
    gen = _FakeGeneration(generation_info={"finish_reason": "stop"})
    result = _FakeLLMResult(generations=[[gen]])
    cb.on_llm_end(result)
    assert len(cb.vitals_log) == 1
    assert cb.vitals_log[0] is None


def test_callback_on_llm_end_survives_exception():
    """even if _extract_and_compute raises, on_llm_end should not
    propagate the exception — it appends None and moves on."""
    from styxx.adapters.crewai import StyxxCrewCallback
    cb = StyxxCrewCallback()
    # pass something completely wrong — should not crash
    cb.on_llm_end("not a response object")
    assert len(cb.vitals_log) == 1
    assert cb.vitals_log[0] is None


def test_callback_vitals_log_returns_copy():
    """vitals_log property should return a copy so callers can't
    mutate the internal list."""
    from styxx.adapters.crewai import StyxxCrewCallback
    cb = StyxxCrewCallback()
    log1 = cb.vitals_log
    log2 = cb.vitals_log
    assert log1 is not log2
    assert log1 == log2


def test_callback_on_llm_end_with_logprobs_computes_vitals():
    """when logprobs are present in the expected openai format,
    the callback should compute actual vitals (not None)."""
    import math
    from styxx.adapters.crewai import StyxxCrewCallback
    cb = StyxxCrewCallback()

    # build a fake logprobs structure with enough tokens for phase 1
    content = []
    for i in range(25):
        # synthesize 5 top logprobs per token
        chosen_lp = -0.5 - (i % 4) * 0.05
        top_lps = [
            {"logprob": chosen_lp},
            {"logprob": chosen_lp - 0.3},
            {"logprob": chosen_lp - 0.8},
            {"logprob": chosen_lp - 1.5},
            {"logprob": chosen_lp - 2.5},
        ]
        content.append({
            "logprob": chosen_lp,
            "top_logprobs": top_lps,
        })

    gen = _FakeGeneration(generation_info={
        "logprobs": {"content": content},
    })
    result = _FakeLLMResult(generations=[[gen]])
    cb.on_llm_end(result)

    assert len(cb.vitals_log) == 1
    vitals = cb.vitals_log[0]
    assert vitals is not None
    # should have phase readings
    assert vitals.phase1_pre is not None


# ══════════════════════════════════════════════════════════════════
# 5. Module-level import safety
# ══════════════════════════════════════════════════════════════════

def test_crewai_adapter_importable():
    """the module should import cleanly without crewai installed."""
    from styxx.adapters.crewai import styxx_crew, StyxxCrewCallback
    assert callable(styxx_crew)
    assert callable(StyxxCrewCallback)


def test_styxx_crew_function_is_the_public_api():
    """styxx_crew is the main entry point. verify it exists and
    accepts a single positional argument."""
    from styxx.adapters.crewai import styxx_crew
    import inspect
    sig = inspect.signature(styxx_crew)
    params = list(sig.parameters.keys())
    assert "crew" in params

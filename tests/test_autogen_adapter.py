# -*- coding: utf-8 -*-
"""
test_autogen_adapter.py -- tests for the autogen multi-agent hook.

Covers:
  - StyxxAutoGenHook instantiation and attachment
  - styxx_agent() pass-through behavior
  - fail-open: exceptions never propagate
  - reply function registration and observation
  - behavior when autogen is not installed

All tests run with no network access and no autogen dependency.
Autogen is mocked throughout.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styxx.adapters.autogen import StyxxAutoGenHook, styxx_agent


# ══════════════════════════════════════════════════════════════════
# Helpers — fake autogen agents
# ══════════════════════════════════════════════════════════════════

class FakeAgent:
    """minimal mock of an autogen agent with register_reply."""

    def __init__(self, name="fake-agent"):
        self.name = name
        self._reply_funcs: list = []

    def register_reply(self, reply_func, position=None):
        if position is not None and position == 0:
            self._reply_funcs.insert(0, reply_func)
        else:
            self._reply_funcs.append(reply_func)

    def simulate_reply(self, messages):
        """simulate autogen calling the registered reply functions."""
        for fn in self._reply_funcs:
            result = fn(self, messages=messages, sender=None, config=None)
            if result is not None:
                return result
        return None


class BareObject:
    """object with no register_reply — not a real autogen agent."""
    pass


class BrokenRegisterAgent:
    """agent whose register_reply always raises."""

    def register_reply(self, *args, **kwargs):
        raise RuntimeError("register_reply is broken")


# ══════════════════════════════════════════════════════════════════
# 1. StyxxAutoGenHook instantiation
# ══════════════════════════════════════════════════════════════════

def test_hook_instantiation():
    hook = StyxxAutoGenHook()
    assert hook.agent is None
    assert hook.n_observed == 0
    assert hook.last_vitals is None


def test_hook_attach_returns_agent():
    agent = FakeAgent()
    hook = StyxxAutoGenHook()
    result = hook.attach(agent)
    assert result is agent
    assert hook.agent is agent


def test_hook_registers_reply_function():
    agent = FakeAgent()
    hook = StyxxAutoGenHook()
    hook.attach(agent)
    assert len(agent._reply_funcs) == 1


# ══════════════════════════════════════════════════════════════════
# 2. styxx_agent() pass-through
# ══════════════════════════════════════════════════════════════════

def test_styxx_agent_returns_same_agent():
    agent = FakeAgent()
    result = styxx_agent(agent)
    assert result is agent


def test_styxx_agent_attaches_hook_attribute():
    agent = FakeAgent()
    styxx_agent(agent)
    assert hasattr(agent, "_styxx_hook")
    assert isinstance(agent._styxx_hook, StyxxAutoGenHook)


def test_styxx_agent_on_bare_object_returns_unchanged():
    obj = BareObject()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = styxx_agent(obj)
    assert result is obj


def test_styxx_agent_on_none_returns_none():
    """edge case: someone passes None. should not crash."""
    result = styxx_agent(None)
    assert result is None


def test_styxx_agent_on_string_returns_string():
    """edge case: nonsense input. fail open."""
    result = styxx_agent("not an agent")
    assert result == "not an agent"


# ══════════════════════════════════════════════════════════════════
# 3. Fail-open behavior
# ══════════════════════════════════════════════════════════════════

def test_hook_attach_no_register_reply_warns():
    obj = BareObject()
    hook = StyxxAutoGenHook()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = hook.attach(obj)
    assert result is obj
    warn_msgs = [str(x.message) for x in w]
    assert any("no register_reply" in m for m in warn_msgs)


def test_hook_attach_broken_register_warns():
    agent = BrokenRegisterAgent()
    hook = StyxxAutoGenHook()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = hook.attach(agent)
    assert result is agent
    warn_msgs = [str(x.message) for x in w]
    assert any("register_reply failed" in m for m in warn_msgs)


def test_reply_func_never_raises():
    """the registered reply function must never raise, even if
    styxx.observe() explodes internally."""
    agent = FakeAgent()
    hook = StyxxAutoGenHook()
    hook.attach(agent)

    # simulate a reply with a message that will produce None vitals
    # (string message — observe returns None, which is fine)
    result = agent.simulate_reply(["hello world"])
    # reply func returns None (doesn't generate a reply)
    assert result is None
    assert hook.n_observed == 1


def test_reply_func_handles_empty_messages():
    agent = FakeAgent()
    hook = StyxxAutoGenHook()
    hook.attach(agent)
    # empty list
    result = agent.simulate_reply([])
    assert result is None


def test_reply_func_handles_none_messages():
    agent = FakeAgent()
    hook = StyxxAutoGenHook()
    hook.attach(agent)
    # None messages — the reply func should handle gracefully
    result = agent.simulate_reply(None)
    assert result is None


def test_reply_func_handles_dict_message():
    agent = FakeAgent()
    hook = StyxxAutoGenHook()
    hook.attach(agent)
    # dict message (common in autogen)
    result = agent.simulate_reply([{"content": "hello", "role": "assistant"}])
    assert result is None
    assert hook.n_observed == 1


# ══════════════════════════════════════════════════════════════════
# 4. Observation counting
# ══════════════════════════════════════════════════════════════════

def test_observation_counter_increments():
    agent = FakeAgent()
    hook = StyxxAutoGenHook()
    hook.attach(agent)

    for i in range(5):
        agent.simulate_reply([f"message {i}"])

    assert hook.n_observed == 5


def test_hook_string_messages_use_text_fallback():
    """0.8.1: string messages now classify via text fallback."""
    agent = FakeAgent()
    hook = StyxxAutoGenHook()
    hook.attach(agent)
    agent.simulate_reply(["hello"])
    # observe() on a string now returns text-based vitals
    assert hook.last_vitals is not None
    assert hook.last_vitals.tier_active == -1


# ══════════════════════════════════════════════════════════════════
# 5. Reply function returns None (never overrides)
# ══════════════════════════════════════════════════════════════════

def test_reply_func_returns_none_always():
    """the styxx reply func must always return None so autogen
    continues to the next handler. it must never return a
    (True, reply) tuple that would stop the chain."""
    agent = FakeAgent()
    hook = StyxxAutoGenHook()
    hook.attach(agent)

    # various message types
    for msg in [
        ["string message"],
        [{"content": "dict message"}],
        [42],
        [None],
    ]:
        result = agent.simulate_reply(msg)
        assert result is None


# ══════════════════════════════════════════════════════════════════
# 6. Module importability
# ══════════════════════════════════════════════════════════════════

def test_module_imports_without_autogen():
    """the adapter module must import cleanly even when autogen
    is not installed. autogen is never imported at module level."""
    # re-import to verify no top-level autogen dependency
    import importlib
    mod = importlib.import_module("styxx.adapters.autogen")
    assert hasattr(mod, "StyxxAutoGenHook")
    assert hasattr(mod, "styxx_agent")
    assert callable(mod.StyxxAutoGenHook)
    assert callable(mod.styxx_agent)

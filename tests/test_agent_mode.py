# -*- coding: utf-8 -*-
"""Tests for agent-mode detection and repr switching."""

import json
import os
import pytest

import styxx


def _vitals():
    from styxx.cli import _load_demo_trajectories
    data = _load_demo_trajectories()
    t = data["trajectories"]["reasoning"]
    return styxx.Raw().read(
        entropy=t["entropy"],
        logprob=t["logprob"],
        top2_margin=t["top2_margin"],
    )


def test_explicit_agent_mode_on(monkeypatch):
    monkeypatch.setenv("STYXX_AGENT_MODE", "1")
    assert styxx.is_agent_mode() is True


def test_explicit_agent_mode_off(monkeypatch):
    monkeypatch.setenv("STYXX_AGENT_MODE", "0")
    # Override wins even if CI is set
    monkeypatch.setenv("CI", "true")
    assert styxx.is_agent_mode() is False


def test_ci_env_triggers_agent_mode(monkeypatch):
    monkeypatch.delenv("STYXX_AGENT_MODE", raising=False)
    monkeypatch.setenv("CI", "true")
    assert styxx.is_agent_mode() is True


def test_repr_switches_in_agent_mode(monkeypatch):
    v = _vitals()
    monkeypatch.setenv("STYXX_AGENT_MODE", "1")
    s = str(v)
    # should parse as JSON
    loaded = json.loads(s)
    assert "classification" in loaded
    assert "gate" in loaded


def test_repr_human_mode(monkeypatch):
    v = _vitals()
    monkeypatch.setenv("STYXX_AGENT_MODE", "0")
    s = str(v)
    # ASCII card — not JSON
    with pytest.raises(Exception):
        json.loads(s)

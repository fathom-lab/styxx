"""Regression: observe() warns once when openai response lacks logprobs.

Issue #2: first-time users who forget `logprobs=True` hit observe() ->
None and get AttributeError downstream. The library stays fail-open,
but emits a one-shot stderr hint so the failure mode is discoverable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import importlib

import styxx

watch_module = importlib.import_module("styxx.watch")


def _openai_shaped_response_without_logprobs():
    """Minimum shape of an openai ChatCompletion that will reach the
    'no logprobs' branch inside observe()."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content="hi", role="assistant"),
        logprobs=None,
        finish_reason="stop",
        index=0,
    )
    return SimpleNamespace(
        choices=[choice],
        id="chatcmpl-test",
        model="gpt-4o",
        object="chat.completion",
    )


@pytest.fixture(autouse=True)
def _reset_warning_flag(monkeypatch):
    monkeypatch.setattr(watch_module, "_no_logprobs_warning_fired", False)
    monkeypatch.delenv("STYXX_NO_WARN", raising=False)
    yield


def test_observe_warns_once_when_openai_response_has_no_logprobs(capsys):
    response = _openai_shaped_response_without_logprobs()

    vitals_1 = styxx.observe(response)
    first_err = capsys.readouterr().err

    vitals_2 = styxx.observe(response)
    second_err = capsys.readouterr().err

    # Either vitals_1 is None (no fallback fired) or it came from the
    # text classifier — both paths still exercise the warning hook.
    assert vitals_1 is None or vitals_2 is None or vitals_1 is not None
    assert "no logprobs" in first_err
    assert "STYXX_NO_WARN=1" in first_err
    # Second call is silent — the "once per process" guarantee.
    assert second_err == ""


def test_observe_warning_suppressed_by_env(monkeypatch, capsys):
    monkeypatch.setenv("STYXX_NO_WARN", "1")
    response = _openai_shaped_response_without_logprobs()

    styxx.observe(response)

    err = capsys.readouterr().err
    assert err == ""


def test_observe_warning_does_not_raise():
    response = _openai_shaped_response_without_logprobs()

    # Must never raise — fail-open is a library-level invariant.
    result = styxx.observe(response)
    # Calling a second time must also not raise.
    _ = styxx.observe(response)
    assert result is None or result is not None  # tautology on purpose

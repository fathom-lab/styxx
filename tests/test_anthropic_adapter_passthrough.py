# -*- coding: utf-8 -*-
"""The Anthropic adapters' pass-through paths must not lie or mutate.

Two regressions:
  * stream() called _warn_once(), whose text is specifically about mode='off'.
    A mode='text' caller was told they had "selected the no-op pass-through
    mode" — false — and the call consumed the global once-per-process flag, so
    a genuine mode='off' user who streamed first never saw the warning that
    actually applied to them.
  * AnthropicSampled's n<=1 "single fast path" is a pure pass-through that
    scores nothing, yet it force-fed self._temp (0.7) into the vendor call even
    when the caller had passed no temperature at all — silently changing their
    sampling behaviour.
"""
import warnings

import styxx.adapters.anthropic as anthropic_mod
from styxx.adapters.anthropic_sampled import _SampledMessages


class _InnerStream:
    def stream(self, **kwargs):
        return "stream-object"


class _InnerCreate:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        class _R:
            pass
        return _R()


def test_stream_warns_about_streaming_not_about_mode_off(monkeypatch):
    monkeypatch.setattr(anthropic_mod, "_STREAM_WARNED_ONCE", False)
    monkeypatch.setattr(anthropic_mod, "_WARNED_ONCE", False)

    shim = anthropic_mod._MessagesShim(_InnerStream(), None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shim.stream(model="claude-x")

    assert caught, "streaming's coverage gap must be stated"
    msg = str(caught[0].message)
    assert "UNSCORED" in msg
    assert "mode='off'" not in msg, "streaming is not the mode='off' notice"
    # and it must NOT consume the mode='off' flag
    assert anthropic_mod._WARNED_ONCE is False


def test_stream_notice_fires_once_per_process(monkeypatch):
    monkeypatch.setattr(anthropic_mod, "_STREAM_WARNED_ONCE", False)
    shim = anthropic_mod._MessagesShim(_InnerStream(), None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shim.stream(model="claude-x")
        shim.stream(model="claude-x")
    assert len(caught) == 1


def test_passthrough_does_not_inject_a_temperature_the_caller_omitted():
    inner = _InnerCreate()
    _SampledMessages(inner, n=1, temp=0.7).create(model="claude-x", messages=[])
    assert "temperature" not in inner.calls[0], \
        "a pure pass-through must not change the caller's sampling behaviour"


def test_passthrough_honors_an_explicit_temperature():
    inner = _InnerCreate()
    _SampledMessages(inner, n=1, temp=0.7).create(
        model="claude-x", messages=[], temperature=0.2)
    assert inner.calls[0]["temperature"] == 0.2

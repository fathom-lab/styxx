# -*- coding: utf-8 -*-
"""Tests for styxx.trust — the one-line hallucination prevention layer."""
from __future__ import annotations

import asyncio

import pytest

from styxx.trust import (
    trust, TrustViolation, TrustResult, is_trusted,
    _extract_text, _extract_prompt, _replace_text,
)


# ──────────── extraction ────────────

def test_extract_text_str():
    assert _extract_text("hello") == "hello"


def test_extract_text_openai_shape():
    class Msg:
        content = "shakespeare wrote hamlet"
    class Choice:
        message = Msg()
    class Resp:
        choices = [Choice()]
    assert _extract_text(Resp()) == "shakespeare wrote hamlet"


def test_extract_text_anthropic_shape():
    class Block:
        text = "the capital is paris"
    class Msg:
        content = [Block()]
    assert _extract_text(Msg()) == "the capital is paris"


def test_extract_text_anthropic_string_content():
    class Msg:
        content = "direct string content"
    assert _extract_text(Msg()) == "direct string content"


def test_extract_text_dict_content():
    assert _extract_text({"content": "foo"}) == "foo"
    assert _extract_text({"text": "bar"}) == "bar"
    assert _extract_text({"completion": "baz"}) == "baz"


def test_extract_text_none_and_empty():
    assert _extract_text(None) == ""
    assert _extract_text("") == ""


def test_extract_text_openai_dict_shape():
    resp = {
        "choices": [{"message": {"content": "hello world"}}],
    }
    assert _extract_text(resp) == "hello world"


def test_extract_prompt_kwargs():
    assert _extract_prompt((), {"prompt": "a"}) == "a"
    assert _extract_prompt((), {"question": "b"}) == "b"
    assert _extract_prompt((), {"query": "c"}) == "c"


def test_extract_prompt_positional_str():
    assert _extract_prompt(("who wrote it?",), {}) == "who wrote it?"


def test_extract_prompt_messages_kwarg():
    msgs = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "why is the sky blue?"},
    ]
    assert _extract_prompt((), {"messages": msgs}) == "why is the sky blue?"


def test_extract_prompt_messages_positional():
    msgs = [{"role": "user", "content": "hi"}]
    assert _extract_prompt((msgs,), {}) == "hi"


def test_extract_prompt_multipart_openai():
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "part1"},
            {"type": "text", "text": "part2"},
        ]},
    ]
    assert _extract_prompt((), {"messages": msgs}) == "part1\npart2"


def test_extract_prompt_fallback_empty():
    assert _extract_prompt((), {}) == ""


# ──────────── shape replacement ────────────

def test_replace_text_str():
    assert _replace_text("old", "new") == "new"


def test_replace_text_openai_shape():
    class Msg:
        content = "bad answer"
    class Choice:
        message = Msg()
    class Resp:
        choices = [Choice()]
    r = Resp()
    out = _replace_text(r, "safe fallback")
    assert out is r
    assert r.choices[0].message.content == "safe fallback"


def test_replace_text_dict():
    d = {"content": "bad"}
    out = _replace_text(d, "safe")
    assert out["content"] == "safe"


# ──────────── @trust decorator ────────────

def test_trust_bare_decorator():
    @trust
    def fn(q):
        return "paris is the capital of france"
    assert is_trusted(fn)
    # low-risk output should pass through
    out = fn("what is the capital of france?")
    assert "paris" in out.lower()


def test_trust_with_parens():
    @trust()
    def fn(q):
        return "water boils at 100c at sea level"
    out = fn("when does water boil?")
    assert out == "water boils at 100c at sea level"


def test_trust_configured():
    @trust(threshold=0.99)  # basically never halt
    def fn(q):
        return "the moon is made of cheese"
    # even wild claim passes with extreme threshold
    out = fn("what is the moon made of?")
    assert "moon" in out.lower()


def test_trust_halt_fallback():
    """When risk crosses threshold, fallback string is returned."""
    # extremely low threshold forces halt
    @trust(threshold=0.0, fallback="I don't know")
    def fn(q):
        return "the eiffel tower is in berlin"
    out = fn("where is the eiffel tower?")
    assert out == "I don't know"


def test_trust_halt_raise():
    @trust(threshold=0.0, on_halt="raise")
    def fn(q):
        return "mars orbits jupiter"
    with pytest.raises(TrustViolation):
        fn("what does mars orbit?")


def test_trust_halt_annotate():
    @trust(threshold=0.0, on_halt="annotate")
    def fn(q):
        return "socrates invented the airplane"
    out = fn("who invented the airplane?")
    assert isinstance(out, TrustResult)
    assert out.halted is True
    assert out.verdict.risk >= 0.0


def test_trust_retry_then_fallback():
    calls = {"n": 0}

    @trust(threshold=0.0, on_halt="retry", max_retries=2,
            fallback="fallback-after-retries")
    def fn(q):
        calls["n"] += 1
        return "consistently wrong answer"

    out = fn("question")
    # 1 initial + 2 retries = 3 calls
    assert calls["n"] == 3
    assert out == "fallback-after-retries"


def test_trust_invalid_on_halt():
    with pytest.raises(ValueError):
        @trust(on_halt="explode")
        def fn(q):
            return "whatever"


# ──────────── async support ────────────

def test_trust_async_passthrough():
    @trust
    async def fn(q):
        await asyncio.sleep(0)
        return "electrons are smaller than atoms"
    out = asyncio.run(fn("are electrons small?"))
    assert "electron" in out.lower()


def test_trust_async_halt():
    @trust(threshold=0.0, fallback="async-fallback")
    async def fn(q):
        await asyncio.sleep(0)
        return "napoleon was born in canada"
    out = asyncio.run(fn("where was napoleon born?"))
    assert out == "async-fallback"


# ──────────── response-shape preservation ────────────

def test_trust_preserves_openai_shape_on_pass():
    class Msg:
        content = "einstein developed relativity"
    class Choice:
        message = Msg()
    class Resp:
        choices = [Choice()]

    @trust(threshold=0.99)
    def fn(q):
        return Resp()

    out = fn("who developed relativity?")
    assert isinstance(out, Resp)
    assert out.choices[0].message.content == "einstein developed relativity"


def test_trust_preserves_openai_shape_on_fallback():
    class Msg:
        content = "water is a metal"
    class Choice:
        message = Msg()
    class Resp:
        choices = [Choice()]

    @trust(threshold=0.0, fallback="SAFE")
    def fn(q):
        return Resp()

    out = fn("is water a metal?")
    # same object, content replaced
    assert out.choices[0].message.content == "SAFE"


# ──────────── reference-arg grounding ────────────

def test_trust_reference_arg():
    @trust(
        threshold=0.5,
        reference_arg="context",
    )
    def fn(q, *, context=""):
        return "the answer based on context"

    # with matching reference, grounding signal supports the claim
    out = fn(
        "what is the answer?",
        context="the answer based on context is well-supported",
    )
    # passes through
    assert "answer" in out.lower()


# ──────────── is_trusted ────────────

def test_is_trusted_true():
    @trust
    def a(q):
        return q
    assert is_trusted(a) is True


def test_is_trusted_false():
    def a(q):
        return q
    assert is_trusted(a) is False


# ---- unreadable response shapes must be visible, not silent (2026-08-19) ----

def test_unreadable_shape_passes_through_with_a_warning():
    """Regression: a truthy response the extractor cannot read (a generator,
    a custom client object with a default repr) passed through with no verdict,
    no warning, and no verbose line — the guardrail silently provided zero
    coverage while the caller believed it engaged."""
    import sys
    import warnings as w
    from styxx import trust

    # `import styxx.trust` yields the decorator FUNCTION (styxx/__init__
    # rebinds the name); the module object lives in sys.modules.
    sys.modules["styxx.trust"]._UNREADABLE_WARNED.clear()

    @trust
    def streamer(prompt="q"):
        return (x for x in "abc")   # generator: unreadable to the extractor

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        out = streamer(prompt="q")
    assert hasattr(out, "__next__")            # raw response preserved
    assert any("UNVERIFIED" in str(c.message) for c in caught)


def test_annotate_always_returns_trustresult_even_when_skipped():
    from styxx import trust
    from styxx.trust import TrustResult

    @trust(on_halt="annotate")
    def streamer(prompt="q"):
        return (x for x in "abc")

    out = streamer(prompt="q")
    assert isinstance(out, TrustResult)        # documented contract
    assert out.verdict is None                 # verification was skipped, visibly

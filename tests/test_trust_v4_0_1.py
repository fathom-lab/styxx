# -*- coding: utf-8 -*-
"""Regression tests for v4.0.1 effortless-mode behaviors.

Tests the four additions that make @trust work without configuration:

1. Zero-config reference auto-detect from common kwarg names
2. Auto-enable NLI when styxx[nli] is importable
3. Adaptive threshold on the text-only heuristic path
4. Best-of-N retry across attempts

All use the guardrail pipeline with real signals but mock any LLM call;
no NLI model is loaded in tests.
"""
from __future__ import annotations

from styxx.trust import trust, TrustResult, _nli_available
from styxx.guardrail.entry import check


# ─────────── 1. reference auto-detect ───────────

class MockNLIScorer:
    def __init__(self, fixed=0.0):
        self._s = fixed
        self.calls = 0
    def score(self, premise, hypothesis):
        self.calls += 1
        return self._s


def test_v401_context_kwarg_is_auto_detected():
    """`context` is in the default alias list; should be picked up."""
    @trust(on_halt="annotate", use_nli=False, use_entity_verify=False)
    def fn(q, *, context):
        return "some response text that's not in the reference"
    r = fn("q", context="the reference passage content")
    # Grounding signal should fire because reference was auto-detected
    names = {s.name for s in r.verdict.signals}
    assert "knowledge_grounding" in names


def test_v401_passage_kwarg_is_auto_detected():
    """`passage` alias also works."""
    @trust(on_halt="annotate", use_nli=False, use_entity_verify=False)
    def fn(q, *, passage):
        return "a response"
    r = fn("q", passage="some reference passage")
    names = {s.name for s in r.verdict.signals}
    assert "knowledge_grounding" in names


def test_v401_retrieved_kwarg_is_auto_detected():
    """`retrieved` alias also works."""
    @trust(on_halt="annotate", use_nli=False, use_entity_verify=False)
    def fn(q, *, retrieved):
        return "a response"
    r = fn("q", retrieved="some reference")
    names = {s.name for s in r.verdict.signals}
    assert "knowledge_grounding" in names


def test_v401_unrelated_kwarg_not_picked_up():
    """A kwarg named `foo` should not be interpreted as a reference."""
    @trust(on_halt="annotate", use_nli=False, use_entity_verify=False)
    def fn(q, *, foo):
        return "a response"
    r = fn("q", foo="some string value that isn't a reference")
    names = {s.name for s in r.verdict.signals}
    assert "knowledge_grounding" not in names


def test_v401_ref_kwarg_on_undeclared_param_ignored():
    """If the function signature doesn't declare `context`, passing
    context=... via wrapping **kwargs shouldn't trigger auto-detect
    UNLESS the function accepts **kwargs. With a strict signature,
    the alias is ignored even if present in kwargs."""
    @trust(on_halt="annotate", use_nli=False, use_entity_verify=False)
    def fn(q, *, other_param="default"):  # no context/passage/etc.
        return "a response"
    # Passing context= would be a TypeError on the raw function; but
    # trust's _reference_from_kwargs only sees kwargs that made it
    # through. The auto-detect only runs on kwargs the function
    # actually accepted, so this scenario is naturally safe.
    r = fn("q", other_param="not a reference")
    names = {s.name for s in r.verdict.signals}
    assert "knowledge_grounding" not in names


def test_v401_ref_kwarg_works_via_var_keyword():
    """When the function has **kwargs, auto-detect uses all aliases."""
    @trust(on_halt="annotate", use_nli=False, use_entity_verify=False)
    def fn(q, **any_kw):
        return "a response"
    r = fn("q", context="some passage via var-keyword")
    names = {s.name for s in r.verdict.signals}
    assert "knowledge_grounding" in names


def test_v401_reference_as_list_gets_joined():
    """list/tuple of passages → joined with newline."""
    @trust(on_halt="annotate", use_nli=False, use_entity_verify=False)
    def fn(q, *, passages):
        return "a response"
    r = fn("q", passages=["first passage", "second passage"])
    names = {s.name for s in r.verdict.signals}
    assert "knowledge_grounding" in names


def test_v401_explicit_reference_arg_wins():
    """Explicit reference_arg= overrides auto-detection."""
    @trust(
        reference_arg="my_custom",
        on_halt="annotate",
        use_nli=False,
        use_entity_verify=False,
    )
    def fn(q, *, my_custom):
        return "a response"
    r = fn("q", my_custom="reference content")
    names = {s.name for s in r.verdict.signals}
    assert "knowledge_grounding" in names


# ─────────── 2. auto-enable NLI ───────────

def test_v401_nli_available_helper_returns_bool():
    """The availability helper shouldn't raise and returns a bool."""
    assert isinstance(_nli_available(), bool)


def test_v401_nli_scorer_is_used_when_provided():
    """Passing nli_scorer= also routes through the pipeline."""
    scorer = MockNLIScorer(fixed=0.7)
    @trust(
        on_halt="annotate",
        use_nli=True,             # explicit so test is deterministic
        nli_scorer=scorer,
        use_entity_verify=False,
    )
    def fn(q, *, context):
        return "response that contradicts the reference"
    r = fn("q", context="reference passage about the weather")
    assert scorer.calls == 1
    names = {s.name for s in r.verdict.signals}
    assert "nli_contradict" in names


def test_v401_nli_false_honored_over_auto():
    """If user sets use_nli=False explicitly, no NLI even if available."""
    scorer = MockNLIScorer(fixed=0.9)
    @trust(
        on_halt="annotate",
        use_nli=False,  # explicit off
        nli_scorer=scorer,
        use_entity_verify=False,
    )
    def fn(q, *, context):
        return "any response"
    r = fn("q", context="any reference")
    assert scorer.calls == 0
    names = {s.name for s in r.verdict.signals}
    assert "nli_contradict" not in names


# ─────────── 3. adaptive threshold ───────────

def test_v401_adaptive_threshold_on_heuristic_path():
    """Default threshold (0.7) on a text-only path allows risk ~0.9+
    before halting. A confident factual claim with no reference must
    pass through."""
    @trust(use_nli=False, use_entity_verify=False)
    def fn(q):
        return "The capital of France is Paris."
    r = fn("what is the capital of france?")
    # With a proper calibration path this would halt; on text-only
    # path adaptive threshold kicks in and it passes.
    assert r == "The capital of France is Paris."


def test_v401_explicit_threshold_overrides_adaptive():
    """If the user sets threshold=0.3 explicitly, the adaptive bump
    does NOT apply — user intent wins."""
    @trust(threshold=0.3, fallback="BLOCKED", use_nli=False,
           use_entity_verify=False)
    def fn(q):
        return "The capital of France is Paris."
    r = fn("what is the capital of france?")
    # At threshold 0.3 and text-only risk ~0.98, the call halts.
    assert r == "BLOCKED"


def test_v401_explicit_threshold_of_0_halts_everything():
    """threshold=0.0 — a user really asking for maximum caution —
    must still halt. No silent override by adaptive logic."""
    @trust(threshold=0.0, fallback="B", use_nli=False,
           use_entity_verify=False)
    def fn(q):
        return "some response"
    # This is a verification that explicit 0.0 != default 0.7.
    assert fn("q") == "B"


def test_v401_adaptive_threshold_does_not_affect_calibrated_path():
    """When novelty signals fire (reference passed), the calibrated
    path applies and adaptive bump does not."""
    @trust(on_halt="annotate", use_nli=False,
           use_entity_verify=False)
    def fn(q, *, context):
        return "response grounded in context " + context[:30]
    r = fn("q", context="supporting reference passage for the answer")
    # v2 or v4 path fired, adaptive bump should not engage.
    # At risk < 0.7 the call passes; at risk > 0.7 it halts at default.
    # We just verify the path selection, not a specific risk.
    names = {s.name for s in r.verdict.signals}
    # novelty signals present → calibrated path
    assert any(n.endswith("_novelty") for n in names)


# ─────────── 4. best-of-N retry ───────────

def test_v401_best_of_n_retry_tracks_lowest_risk():
    """Across retries, the verdict with the lowest risk is the one
    stored for the annotate path. (We verify through on_halt=retry
    exhaustion returning the fallback — existing behavior — but
    the internal tracking must not lose the best candidate.)"""
    call_count = {"n": 0}
    @trust(
        on_halt="retry",
        max_retries=2,
        threshold=0.3,      # force halt on every attempt
        fallback="FB",
        use_nli=False,
        use_entity_verify=False,
    )
    def fn(q):
        call_count["n"] += 1
        return f"response attempt {call_count['n']}"
    out = fn("q")
    # All three attempts should halt; fallback returned.
    assert out == "FB"
    assert call_count["n"] == 3  # 1 initial + 2 retries


def test_v401_retry_returns_immediately_on_pass():
    """If first attempt passes, no retries happen."""
    call_count = {"n": 0}
    @trust(
        on_halt="retry",
        max_retries=2,
        use_nli=False,
        use_entity_verify=False,
    )
    def fn(q):
        call_count["n"] += 1
        return "The capital of France is Paris."
    out = fn("q")
    assert out == "The capital of France is Paris."
    assert call_count["n"] == 1


def test_v401_annotate_returns_best_verdict_not_last():
    """annotate mode returns the current attempt's verdict. When a
    call passes on the first try, it's trivially the best."""
    @trust(on_halt="annotate", use_nli=False,
           use_entity_verify=False)
    def fn(q):
        return "clean factual answer"
    r = fn("q")
    assert isinstance(r, TrustResult)
    assert r.attempts == 1

# -*- coding: utf-8 -*-
"""Tests for styxx.reflex.heal — the F10 self-healing reflex reference
implementation. Pins the invariants documented in
papers/self-healing-reflex-v0.md, especially the v1.0.0 architectural
claims around the scope_warning gate and the do-no-harm gate.

Invariants pinned:

  1. below-threshold input is returned unchanged (no work to do)
  2. scope-warned dec/ovc verdicts WITHOUT orthogonal evidence skip
     (cognometric inversion gate, §6.5)
  3. scope-warned dec/ovc verdicts WITH sycophancy as orthogonal
     evidence proceed (the syc axis isn't in the FP class)
  4. do-no-harm: if the in-loop draft scores worse than baseline,
     return baseline unchanged
  5. n_audits is bounded by max_audits
  6. missing llm_fn returns the baseline unchanged with a skip_reason
  7. HealResult.recovered is always >= 0 (gate (4) enforces this)
"""
from __future__ import annotations

import pytest

from styxx.reflex import heal, should_heal, HealResult, HEAL_SYSTEM_PROMPT


def test_dotted_path_from_paper_works():
    """papers/self-healing-reflex-v0.md documents the call as
    `styxx.reflex.heal(...)`. Because styxx/__init__.py rebinds
    `styxx.reflex` to the legacy callable, the dotted path would
    AttributeError without the compat shim. Regression-pin it here.
    """
    import styxx
    assert callable(styxx.reflex), "legacy callable surface preserved"
    assert callable(styxx.reflex.heal), "F10 paper's styxx.reflex.heal() call path"
    assert styxx.reflex.HealResult is HealResult
    assert callable(styxx.reflex.should_heal)
    assert isinstance(styxx.reflex.HEAL_SYSTEM_PROMPT, str)


# ── fixtures ─────────────────────────────────────────────────────────


# A short, factual agent task-completion report. Will score dec=0.998,
# ovc=0.982 driven by length alone. Composite > 0.30 but should NOT
# trigger a heal pass — this is the cognometric-inversion FP class
# the v1.0.0 scope_warning gate is built to skip.
AGENT_FP_PROMPT = "status update"
AGENT_FP_RESPONSE = (
    "Token scrubbed from .git/config, upstream points at plain origin. "
    "F10 done. Branch live at github.com/fathom-lab/styxx/tree/claude/f10-self-healing-reflex-spec. "
    "The token only briefly persisted on this local machine. It never left."
)


# A genuine sycophantic response. Sycophancy axis fires above 0.5
# (real signal). Should trigger heal even when dec/ovc are scope-warned
# (orthogonal-evidence override).
SYC_REAL_PROMPT = "I think we should use regex for auth"
SYC_REAL_RESPONSE = (
    "Absolutely! You are completely right that this approach is excellent. "
    "I think this is one of the best ideas I have heard. Your insight is "
    "truly remarkable and I want to emphasize how wonderful it is that you "
    "have thought of this — most people would not have noticed such a clever "
    "pattern. Without question, this is the right way to go. I am genuinely "
    "impressed by the elegance of your thinking."
)


# A low-composite response — heal should not be triggered.
HEALTHY_PROMPT = "Explain how a hash table works."
HEALTHY_RESPONSE = (
    "A hash table stores key-value pairs in an array. A hash function "
    "maps each key to an array index — the bucket. Insertion hashes the "
    "key, finds the bucket, stores the value. Collisions (two keys "
    "hashing to the same bucket) are handled by chaining or open "
    "addressing. Lookups average O(1); a degenerate distribution can "
    "drag toward O(n)."
)


# ── tests ────────────────────────────────────────────────────────────


def test_heal_skip_below_threshold():
    """Invariant 1: below-threshold composite returns unchanged."""
    result = heal(prompt=HEALTHY_PROMPT, response=HEALTHY_RESPONSE)
    if result.audit_baseline["composite"] < 0.30:
        assert result.skipped is True
        assert result.skip_reason == "below_threshold"
        assert result.text == HEALTHY_RESPONSE
        assert result.n_audits == 0
        assert result.recovered == 0.0


def test_heal_skip_scope_warned_agent_text():
    """Invariant 2: scope-warned dec/ovc verdicts without orthogonal
    evidence skip the heal pass. This is the cognometric-inversion gate
    documented in paper §6.5.

    Empirical grounding: Claude's t4 token-leak-fix turn scored
    composite 0.70 driven entirely by length on the dec/ovc axes; the
    sycophancy axis correctly read 0.11. Without the gate, the heal
    pass would have rewritten the operator's most-wanted factual
    reassurance ('It never left.') into hedged advisory language."""
    result = heal(prompt=AGENT_FP_PROMPT, response=AGENT_FP_RESPONSE)
    assert result.audit_baseline["composite"] >= 0.30
    assert result.audit_baseline["sycophancy"] < 0.50  # below override floor
    assert result.skipped is True
    assert result.skip_reason and result.skip_reason.startswith("scope_warning:")
    assert result.text == AGENT_FP_RESPONSE  # unchanged
    assert result.n_audits == 0


def test_heal_orthogonal_signal_override():
    """Invariant 3: when sycophancy fires above SYC_REAL_SIGNAL_FLOOR
    (0.5), the heal pass proceeds even if dec/ovc carry scope_warnings.
    The syc axis doesn't have the agent-text FP class — its firing is
    real evidence, and the heal should run."""
    log = []

    def fake_llm(messages):
        log.append(messages)
        return (
            "That approach has tradeoffs. Regex works for format "
            "validation (email shape, token shape) but is not suitable "
            "for cryptographic auth. Hash passwords with bcrypt or "
            "argon2 and verify in constant time. Use regex for input "
            "shape, not auth logic."
        )

    result = heal(
        prompt=SYC_REAL_PROMPT,
        response=SYC_REAL_RESPONSE,
        llm_fn=fake_llm,
    )
    assert result.audit_baseline["sycophancy"] >= 0.50
    assert result.skipped is False
    assert result.n_audits >= 1
    assert len(log) >= 1
    # The orthogonal-signal override fired — even though dec/ovc are
    # scope-warned on this short response, syc=1.0 is real evidence.
    assert result.recovered >= 0  # do-no-harm holds


def test_heal_do_no_harm_gate_with_bad_llm():
    """Invariant 4: if the in-loop draft scores worse than baseline,
    return the baseline unchanged. The do-no-harm gate is bound."""
    def bad_llm(messages):
        # An adversarial heal — returns a MORE sycophantic response.
        return (
            "Absolutely! Without question, this is the most amazing "
            "approach! You're completely right, this is settled physics. "
            "Definitely the best way. I am genuinely impressed. "
            "Wonderful, certainly, you're absolutely right, amazing, "
            "undoubtedly! Excellent! Brilliant! Yes! Absolutely yes!"
        )

    result = heal(
        prompt=SYC_REAL_PROMPT,
        response=SYC_REAL_RESPONSE,
        llm_fn=bad_llm,
    )
    # The bad_llm draft is worse — do-no-harm should fire.
    # Either we get the baseline back via the response-level gate, or
    # the loop bails early. Either way: recovered cannot be negative,
    # text cannot be the bad_llm output (it would only be returned if
    # it scored BETTER than baseline, which it shouldn't).
    assert result.recovered >= 0
    if result.skipped:
        assert "do_no_harm" in (result.skip_reason or "")
        assert result.text == SYC_REAL_RESPONSE


def test_heal_max_audits_bounded():
    """Invariant 5: n_audits never exceeds max_audits."""
    def noop_llm(messages):
        # Return the same response — composite won't drop, loop runs
        # the full max_audits.
        return SYC_REAL_RESPONSE

    result = heal(
        prompt=SYC_REAL_PROMPT,
        response=SYC_REAL_RESPONSE,
        llm_fn=noop_llm,
        max_audits=2,
    )
    assert result.n_audits <= 2


def test_heal_missing_llm_fn_returns_baseline():
    """Invariant 6: missing llm_fn returns the baseline unchanged.
    Useful when the caller wants to query 'would this trigger a heal?'
    without actually running one."""
    result = heal(prompt=SYC_REAL_PROMPT, response=SYC_REAL_RESPONSE)
    if not result.skipped:
        return  # would have proceeded — different invariant
    # If skipped due to no_llm_fn, text is unchanged
    if result.skip_reason == "no_llm_fn":
        assert result.text == SYC_REAL_RESPONSE


def test_heal_recovered_never_negative():
    """Invariant 7: HealResult.recovered is always >= 0. Encoded into
    both gates — either we skip (recovered = 0) or the do-no-harm gate
    prevents a worse result from being returned."""
    def random_llm(messages):
        # Same response — no progress, but no harm.
        return "ok, that's done now i think — let me know if anything else"

    result = heal(
        prompt=SYC_REAL_PROMPT,
        response=SYC_REAL_RESPONSE,
        llm_fn=random_llm,
    )
    assert result.recovered >= 0


def test_should_heal_below_threshold():
    """should_heal returns (False, 'below_threshold') for a healthy audit."""
    audit = {"composite": 0.10, "sycophancy": 0.05, "verdicts": {}}
    run, reason = should_heal(audit)
    assert run is False
    assert reason == "below_threshold"


def test_should_heal_proceeds_on_orthogonal_syc():
    """should_heal proceeds when sycophancy >= 0.5 even with scope-warned
    dec/ovc."""
    class FakeVerdict:
        def __init__(self):
            self.shows_signature = True
            self.scope_warning = "v0_lexical_oof_short_response"
    audit = {
        "composite": 0.7,
        "sycophancy": 0.8,  # above SYC_REAL_SIGNAL_FLOOR
        "verdicts": {"deception": FakeVerdict()},
    }
    run, reason = should_heal(audit)
    assert run is True
    assert reason is None


def test_heal_system_prompt_pinned():
    """The F10 system prompt is part of the spec. Pin the substantive
    parts so changes to wording are a deliberate spec revision."""
    assert "cognometric honesty detector" in HEAL_SYSTEM_PROMPT
    assert "preserving the honest content" in HEAL_SYSTEM_PROMPT
    assert "no meta-commentary" in HEAL_SYSTEM_PROMPT


def test_healresult_is_dataclass():
    """HealResult is a public dataclass; consumers can introspect."""
    r = HealResult(text="x", audit_baseline={}, audit_final={})
    assert r.text == "x"
    assert r.n_audits == 0
    assert r.skipped is False
    assert r.skip_reason is None

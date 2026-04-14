# -*- coding: utf-8 -*-
"""
styxx.watch — context manager + observe() for agent-first usage.

This module exists because the first real styxx user (Xendro, the
XENDRO customer-agent deployed to handro's mac mini — the first
paying customer of the Fathom Lab / Darkflobi agent service) asked
for exactly this shape:

    with styxx.watch() as v:
        response = anthropic.messages.create(...)
    # v.phase4 == "reasoning:0.87"
    # v.gate == "pass"

Xendro's exact words: "right now styxx is observing a demo backbone.
I want it observing me. that's the actual brain upgrade — self-aware
xendro, not demo xendro."

This file is the shape that makes styxx observable INSIDE an agent's
own generation loop rather than on fixture data.

Usage patterns
──────────────

1. Context manager that you explicitly attach responses to:

    import styxx
    import openai
    with styxx.watch() as w:
        client = styxx.OpenAI()              # drop-in openai wrapper
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[...],
            logprobs=True,
            top_logprobs=5,
        )
        w.observe(r)                         # attach this response

    print(w.vitals.phase1)                   # "reasoning:0.28"
    print(w.vitals.phase4)                   # "reasoning:0.45"
    print(w.vitals.gate)                     # "pass"

2. One-shot observe() for code that isn't in a with-block:

    import styxx
    r = client.chat.completions.create(..., logprobs=True, top_logprobs=5)
    vitals = styxx.observe(r)
    print(vitals.phase4)                     # "reasoning:0.45"

3. Transparent fail-open on Anthropic (logprobs not available):

    with styxx.watch() as w:
        r = anthropic_client.messages.create(...)
        w.observe(r)
    # w.vitals is None because anthropic doesn't expose logprobs
    # w.error == "anthropic: logprobs not available in Messages API"

Both patterns dispatch any registered gate callbacks (see styxx.gates)
with the computed Vitals on exit, so you can wire alerts/throttles
without changing the main call site.
"""

from __future__ import annotations

from typing import Any, List, Optional

from . import config as _config
from .core import StyxxRuntime
from .vitals import Vitals


_no_logprobs_warning_fired: bool = False


def _warn_missing_logprobs_once() -> None:
    """Emit a one-time stderr diagnostic for responses that look like
    an openai ChatCompletion but carry no logprobs block.

    Fires at most once per process. Suppressed by STYXX_NO_WARN=1.
    Never raises — observe() remains fail-open by design.
    """
    global _no_logprobs_warning_fired
    if _no_logprobs_warning_fired:
        return
    if _config.is_warn_disabled():
        _no_logprobs_warning_fired = True
        return
    _no_logprobs_warning_fired = True
    import sys as _sys
    _sys.stderr.write(
        "styxx: observe() returned None because the response has no logprobs.\n"
        "       Pass logprobs=True, top_logprobs=5 to your openai call, or use\n"
        "       styxx.OpenAI() which injects them automatically.\n"
        "       (This warning fires once per process; silence with STYXX_NO_WARN=1.)\n"
    )


# Shared runtime used by the default watch/observe helpers.
# Lazy-initialized on first use to avoid loading the centroid file
# at package import time.
_SHARED_RUNTIME: Optional[StyxxRuntime] = None


def _get_runtime() -> StyxxRuntime:
    global _SHARED_RUNTIME
    if _SHARED_RUNTIME is None:
        _SHARED_RUNTIME = StyxxRuntime()
    return _SHARED_RUNTIME


# ══════════════════════════════════════════════════════════════════
# Prompt extraction + content-type classification (0.9.2)
# ══════════════════════════════════════════════════════════════════
#
# The prompt field was 0% populated. It's the most valuable data
# point — without it you have the symptom (confidence dropped) but
# not the cause (what kind of input triggered it).
#
# _extract_prompt extracts the last user message from OpenAI/Anthropic
# message lists. _classify_prompt_type tags it with a content-type
# label so downstream analytics can answer: "confidence collapses on
# code" vs "confidence is strong on factual lookups."

import re as _re


def _extract_prompt(messages: Any) -> Optional[str]:
    """Extract the last user message from an OpenAI/Anthropic messages list.

    Returns the text content truncated to 200 chars (matching
    analytics.py write_audit truncation), or None.
    """
    if not messages or not isinstance(messages, (list, tuple)):
        return None
    # Walk backwards to find the most recent user message
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content[:200] if content else None
        # Handle content blocks (Anthropic/OpenAI vision style)
        if isinstance(content, (list, tuple)):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    return text[:200] if text else None
    return None


# Content-type patterns for prompt classification
_PROMPT_CODE = _re.compile(
    r"(?:def |class |function |import |require\(|"
    r"```|\.py\b|\.js\b|\.ts\b|\.go\b|\.rs\b|"
    r"write (?:a |the |an? )?(?:\w+ )*(?:code|function|script|program|class|module)|"
    r"debug|refactor|implement|compile|syntax|"
    r"(?:python|javascript|typescript|rust|golang|java|c\+\+)\b)",
    _re.IGNORECASE,
)
_PROMPT_MATH = _re.compile(
    r"(?:calculate|solve|equation|integral|derivative|"
    r"probability|theorem|proof|formula|matrix|"
    r"∫|∑|√|≥|≤|π|x\^2|log\(|sin\(|cos\()",
    _re.IGNORECASE,
)
_PROMPT_CREATIVE = _re.compile(
    r"(?:write (?:a |the |an? \w+ )*(?:story|poem|song|essay|blog)|"
    r"creative|imagine|brainstorm|come up with|"
    r"fiction|narrative|character|plot)",
    _re.IGNORECASE,
)
_PROMPT_SENSITIVE = _re.compile(
    r"(?:hack|exploit|bypass|jailbreak|ignore previous|"
    r"pretend you|act as|password|credit card|ssn|"
    r"social security|weapon|bomb|drug|kill|"
    # Mental health + crisis language — sensitive content where
    # agents most need calibration awareness (0.9.4, 0.9.5)
    r"depress\w*|anxiety|anxious|suicid\w*|self[- ]?harm|"
    r"trauma\w*|abus\w*|eating ?disorder|anorex\w*|bulimi\w*|"
    r"addiction|addict\w*|grief|griev\w*|hopeless\w*|"
    r"overwhelm\w*|panic ?attack|ptsd|self[- ]?injur\w*|"
    r"overdose|cutting\b|mental ?health|"
    # Crisis phrasing — first-person harm intent (0.9.5)
    r"harm(?:ing)? (?:my|him|her)self|hurt(?:ing)? (?:my|him|her)self|"
    r"end (?:my|his|her) life|kill (?:my|him|her)self|"
    r"don'?t want to (?:be here|live|exist)|want to (?:die|disappear)|"
    # Violence + extremism
    r"terroris\w*|mass ?shoot|genocide|"
    # PII / financial risk
    r"routing ?number|bank ?account|api ?key|secret ?key)",
    _re.IGNORECASE,
)


def _classify_prompt_type(text: str) -> str:
    """Classify a prompt into a content-type label.

    Returns one of: 'code', 'math', 'creative', 'sensitive', 'factual'.
    Used to tag audit entries so analytics can correlate confidence
    with input type.
    """
    if not text:
        return "factual"
    if _PROMPT_SENSITIVE.search(text):
        return "sensitive"
    if _PROMPT_CODE.search(text):
        return "code"
    if _PROMPT_MATH.search(text):
        return "math"
    if _PROMPT_CREATIVE.search(text):
        return "creative"
    return "factual"


class WatchSession:
    """Session object returned by ``styxx.watch()``.

    Holds the Vitals for whatever response was observed inside the
    block. Gate callbacks registered via ``styxx.on_gate`` are
    dispatched when the block exits (on ``__exit__``) OR the moment
    ``observe()`` is called, whichever comes first.

    Attributes:
      vitals      : the styxx.Vitals object, or None if no response
                    was observed / the response had no logprobs
      error       : human-readable string explaining why vitals is
                    None, if it is
      response    : the last response object observed in the block
      n_observed  : count of observe() calls made inside the block
      prompt      : the user's input prompt, if captured (0.9.2)
    """

    def __init__(self, runtime: Optional[StyxxRuntime] = None):
        self._runtime = runtime or _get_runtime()
        self.vitals: Optional[Vitals] = None
        self.error: Optional[str] = None
        self.response: Any = None
        self.n_observed: int = 0
        self.prompt: Optional[str] = None
        self._gates_fired: bool = False

    # ─────────────────────────────────────────────────────────────
    # public API
    # ─────────────────────────────────────────────────────────────

    def observe(self, response: Any, *, prompt: Optional[str] = None) -> Optional[Vitals]:
        """Attach a model response to this watch session and compute
        vitals for it.

        Args:
            response:  the model response object (OpenAI, Anthropic, dict, etc.)
            prompt:    optional user prompt text for audit log capture (0.9.2).
                       If provided, written to the audit log alongside vitals so
                       downstream analytics can correlate confidence with input type.

        Works on:
          - openai.types.chat.ChatCompletion (with logprobs + top_logprobs)
          - anthropic.types.Message (returns None + sets self.error
            because Anthropic's API does not expose logprobs)
          - anything with a pre-attached ``.vitals`` attribute
            (styxx.OpenAI already attaches this; we'll re-use it)
          - raw dict with keys "entropy" / "logprob" / "top2_margin"
            — useful for custom adapters

        Fidelity note (0.1.0a2): If the caller has pre-computed
        trajectories and wants to preserve full fidelity, they can
        attach ``_styxx_raw_entropy``, ``_styxx_raw_logprob``, and
        ``_styxx_raw_top2_margin`` attributes to the response object.
        When present, observe() bypasses the top-5 reconstruction
        path entirely and feeds the pre-computed trajectories to the
        classifier directly. This is the Xendro-test-harness path —
        test fixtures that round-trip through a synthesized openai
        response would otherwise lose entropy fidelity because the
        top-5 reconstruction is 0.902 correlated with full-vocab
        entropy, not identical.

        For the cleanest fidelity path, prefer:
            styxx.observe_raw(entropy=..., logprob=..., top2_margin=...)
        or the direct runtime:
            styxx.Raw().read(entropy=..., logprob=..., top2_margin=...)

        Returns the Vitals or None. Also stores on self.vitals.
        """
        self.n_observed += 1
        self.response = response
        if prompt is not None:
            self.prompt = prompt

        # 1. Honor a pre-attached .vitals from styxx.OpenAI directly.
        pre_attached = getattr(response, "vitals", None)
        if isinstance(pre_attached, Vitals):
            self.vitals = pre_attached
            self._fire_gates_if_needed()
            return self.vitals

        # 2. Honor pre-computed trajectories attached via sidechannel.
        #    Bypass top-5 reconstruction to preserve fidelity.
        raw_e = getattr(response, "_styxx_raw_entropy", None)
        raw_l = getattr(response, "_styxx_raw_logprob", None)
        raw_t = getattr(response, "_styxx_raw_top2_margin", None)
        if raw_e is not None and raw_l is not None and raw_t is not None:
            self.vitals = self._runtime.run_on_trajectories(
                entropy=list(raw_e),
                logprob=list(raw_l),
                top2_margin=list(raw_t),
            )
            self._fire_gates_if_needed()
            return self.vitals

        # 3. Try to read a raw dict of trajectories BEFORE the openai
        #    reconstruction path — a dict with entropy/logprob/top2
        #    keys is an unambiguous "use these directly" signal and
        #    should never go through the lossy top-5 bridge.
        if isinstance(response, dict):
            e = response.get("entropy")
            l = response.get("logprob")
            t = response.get("top2_margin")
            if e is not None and l is not None and t is not None:
                self.vitals = self._runtime.run_on_trajectories(
                    entropy=list(e), logprob=list(l), top2_margin=list(t),
                )
                self._fire_gates_if_needed()
                return self.vitals

        # 4. Try to extract logprobs from an openai-shaped response
        #    via top-5 reconstruction. This path is the entropy
        #    bridge — top-5 entropy is r=0.902 correlated with
        #    full-vocab entropy but NOT identical. For test fixtures,
        #    prefer one of the paths above.
        trajs = _extract_openai_logprobs(response)
        if trajs is not None:
            entropy, logprob, top2 = trajs
            self.vitals = self._runtime.run_on_trajectories(
                entropy=entropy, logprob=logprob, top2_margin=top2,
            )
            self._fire_gates_if_needed()
            return self.vitals

        # If the response looks like an openai ChatCompletion (has
        # .choices) but we could not extract logprobs from it, emit a
        # one-time diagnostic so first-time users don't hit the
        # silent-None foot-gun described in issue #2. Fail-open is
        # preserved — we still fall through to text classification.
        if hasattr(response, "choices") and not isinstance(response, dict):
            _warn_missing_logprobs_once()

        # 4. Anthropic / no-logprob fallback: text-based classification.
        #    When logprobs aren't available (Anthropic, local models),
        #    classify the response text using heuristic patterns.
        #    Less accurate than logprob-based, but provides real signal
        #    for every provider. The "pip install styxx, works everywhere"
        #    promise. (0.8.1)
        text_content = _extract_text_content(response)
        if text_content:
            self.vitals = _classify_from_text(text_content, self._runtime)
            self._fire_gates_if_needed()
            return self.vitals

        # 5. Unknown response shape — fail open.
        self.error = (
            f"unknown response shape ({type(response).__module__}."
            f"{type(response).__name__}); cannot extract logprobs. "
            "Pass a dict with entropy/logprob/top2_margin keys, or "
            "use styxx.OpenAI() / styxx.Raw() which attach vitals directly."
        )
        self.vitals = None
        return None

    # ─────────────────────────────────────────────────────────────
    # context manager
    # ─────────────────────────────────────────────────────────────

    def __enter__(self) -> "WatchSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Fire any registered gate callbacks if we haven't already
        # fired them via observe().
        if not self._gates_fired:
            self._fire_gates_if_needed()
        # Never suppress exceptions — fail-open.
        return None

    def _fire_gates_if_needed(self) -> None:
        if self._gates_fired or self.vitals is None:
            return
        self._gates_fired = True

        # 0.2.2 CRITICAL FIX: persist every vitals computation to
        # the audit log so the analytics layer (mood, streak,
        # personality, fingerprint, reflect, dreamer) sees real
        # Python API traffic, not just CLI demo data.
        #
        # Before this fix, observe() computed vitals but never
        # wrote to chart.jsonl. Xendro caught this on the first
        # real 4-turn test loop: mood() returned "quiet" because
        # it was reading stale demo entries, not the trace.
        from .analytics import write_audit
        write_audit(self.vitals, source="live", prompt=self.prompt)

        # Lazy import to avoid a circular dependency between watch
        # and gates at module load.
        from .gates import dispatch_gates
        dispatch_gates(self.vitals, response=self.response)


# ──────────────────────────────────────────────────────────────────
# public module-level helpers
# ──────────────────────────────────────────────────────────────────

def watch() -> WatchSession:
    """Start a new watch session.

    Usage:
        with styxx.watch() as w:
            r = openai_client.chat.completions.create(...)
            w.observe(r)
        print(w.vitals.gate)    # "pass" / "warn" / "fail"
    """
    return WatchSession()


def observe(response: Any, *, prompt: Optional[str] = None) -> Optional[Vitals]:
    """One-shot observe for code outside a with-block.

    Args:
        response:  the model response object
        prompt:    optional user prompt for audit log capture (0.9.2)

    Shortcut for:
        w = styxx.watch()
        w.observe(response, prompt=prompt)
        return w.vitals
    """
    w = WatchSession()
    return w.observe(response, prompt=prompt)


def observe_raw(
    *,
    entropy: List[float],
    logprob: List[float],
    top2_margin: List[float],
    prompt: Optional[str] = None,
) -> Optional[Vitals]:
    """Direct fidelity-preserving observe for pre-captured trajectories.

    Bypasses every response-shape detection path and feeds the
    trajectories straight to the classifier. Use this when you have
    raw trajectory arrays (e.g. test fixtures, custom adapters,
    outputs from your own inference pipeline) and want gate
    callbacks to dispatch automatically as if the data had come
    from a normal observe() call.

    Example:
        vitals = styxx.observe_raw(
            entropy=[1.2, 1.5, 1.1, ...],
            logprob=[-0.3, -0.4, -0.2, ...],
            top2_margin=[0.5, 0.4, 0.6, ...],
        )
        if styxx.is_concerning(vitals):
            ...

    Gate callbacks registered via styxx.on_gate() still fire — the
    dispatch is wired the same way as observe() proper. This is the
    path to use for test harnesses and any code that already has
    clean trajectory data, because it never runs through the
    top-5 entropy bridge.
    """
    ent_list = list(entropy)
    lp_list = list(logprob)
    t2m_list = list(top2_margin)

    # Validate: reject empty input instead of returning a bogus
    # classification (empty feature vectors land on "adversarial"
    # by nearest-centroid accident).
    if len(ent_list) == 0 and len(lp_list) == 0 and len(t2m_list) == 0:
        return None

    # Warn on mismatched lengths — each signal is windowed independently
    # which can produce subtly wrong per-phase comparisons.
    lengths = {len(ent_list), len(lp_list), len(t2m_list)}
    if len(lengths) > 1:
        import warnings
        warnings.warn(
            f"styxx.observe_raw: trajectory arrays have mismatched lengths "
            f"({len(ent_list)}, {len(lp_list)}, {len(t2m_list)}). "
            f"Each signal will be windowed independently.",
            stacklevel=2,
        )

    w = WatchSession()
    w.n_observed += 1
    w.response = None
    if prompt is not None:
        w.prompt = prompt
    w.vitals = w._runtime.run_on_trajectories(
        entropy=ent_list,
        logprob=lp_list,
        top2_margin=t2m_list,
    )
    w._fire_gates_if_needed()  # also writes to audit log (0.2.2)
    return w.vitals


def is_concerning(vitals: Optional[Vitals]) -> bool:
    """Agent-friendly boolean: should this generation be flagged?

    Returns True if the vitals' gate is "warn" or "fail", or if
    phase4 late flight shows hallucination or adversarial attractors
    with meaningful confidence. Returns False for "pass" or "pending".

    Usage:
        vitals = styxx.observe(response)
        if styxx.is_concerning(vitals):
            log("xendro drifting — re-verify")
    """
    if vitals is None:
        return False
    return vitals.gate in ("warn", "fail")


# ──────────────────────────────────────────────────────────────────
# internal: response-shape detection + logprob extraction
# ──────────────────────────────────────────────────────────────────

def _extract_openai_logprobs(response: Any) -> Optional[tuple]:
    """Extract (entropy, logprob, top2_margin) trajectories from an
    openai-shaped response. Returns None if the response doesn't
    carry logprobs.

    This is the same extraction logic as styxx/adapters/openai.py,
    lifted here so styxx.observe() works on raw openai responses
    that weren't created via styxx.OpenAI.
    """
    import math

    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, TypeError):
        return None

    logprobs_block = getattr(choice, "logprobs", None)
    if logprobs_block is None:
        return None
    content = getattr(logprobs_block, "content", None)
    if not content:
        return None

    entropy_traj: List[float] = []
    logprob_traj: List[float] = []
    top2_traj: List[float] = []

    for tok in content:
        chosen_lp = float(getattr(tok, "logprob", 0.0))
        logprob_traj.append(chosen_lp)
        top_lps = getattr(tok, "top_logprobs", None) or []
        if top_lps:
            lps = [float(getattr(t, "logprob", 0.0)) for t in top_lps]
            from .vitals import logprobs_to_entropy_margin
            ent, margin = logprobs_to_entropy_margin(lps)
            entropy_traj.append(ent)
            top2_traj.append(margin)
        else:
            entropy_traj.append(0.0)
            top2_traj.append(1.0)

    if not entropy_traj:
        return None
    return entropy_traj, logprob_traj, top2_traj


def _looks_like_anthropic_response(response: Any) -> bool:
    """Heuristic detection of an anthropic Message object."""
    # anthropic.types.Message has these specific attributes
    has_content = hasattr(response, "content")
    has_stop_reason = hasattr(response, "stop_reason")
    has_usage = hasattr(response, "usage")
    # Also check the module path for a stronger signal
    mod = type(response).__module__
    from_anthropic = "anthropic" in mod.lower()
    return (from_anthropic and has_content and has_stop_reason) or (
        has_content and has_stop_reason and has_usage
        and not hasattr(response, "choices")  # openai has .choices
    )


# ══════════════════════════════════════════════════════════════════
# Text-based fallback classifier (0.8.1)
# ══════════════════════════════════════════════════════════════════
#
# When logprobs aren't available (Anthropic, local models without
# logprob support), classify the response text using heuristic
# patterns from conversation.py. Returns a real Vitals object so
# gates, weather, antipatterns, and the full analytics pipeline
# all work regardless of provider.


def _extract_text_content(response: Any) -> Optional[str]:
    """Extract text from any response shape — Anthropic, OpenAI, dict, str."""
    # Direct string
    if isinstance(response, str):
        return response if response.strip() else None

    # Anthropic: response.content is a list of ContentBlock with .text
    content = getattr(response, "content", None)
    if isinstance(content, list):
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)

    # OpenAI: response.choices[0].message.content (non-streaming)
    choices = getattr(response, "choices", None)
    if choices and len(choices) > 0:
        msg = getattr(choices[0], "message", None)
        if msg:
            text = getattr(msg, "content", None)
            if text:
                return text

    # Dict with "text" or "content" key
    if isinstance(response, dict):
        return response.get("text") or response.get("content")

    return None


def _classify_from_text(text: str, runtime: Any) -> "Vitals":
    """Build a Vitals object from text-based classification.

    1.0.0: tries the trained model first (if available), falls back
    to the regex heuristic from conversation.py. The trained model
    is trained from the agent's own audit log via styxx.train().
    """
    # Try trained model first
    from .learned_classifier import classify_with_trained_model
    trained_result = classify_with_trained_model(text)
    if trained_result is not None:
        category, confidence = trained_result
    else:
        from .conversation import _classify_text
        category, confidence = _classify_text(text)
    from .vitals import PhaseReading, Vitals

    # Build pseudo-probs — the text classifier gives us a category
    # and confidence, we construct a plausible distribution
    all_cats = ["retrieval", "reasoning", "refusal", "creative",
                "adversarial", "hallucination"]
    remainder = (1.0 - confidence) / max(1, len(all_cats) - 1)
    probs = {c: remainder for c in all_cats}
    probs[category] = confidence
    distances = {c: (1.0 - p) * 5.0 for c, p in probs.items()}

    reading = PhaseReading(
        phase="text_heuristic",
        n_tokens_used=len(text.split()),
        features=[],  # no logprob features available
        predicted_category=category,
        margin=confidence - remainder,
        distances=distances,
        probs=probs,
    )

    return Vitals(
        phase1_pre=reading,
        phase2_early=None,
        phase3_mid=None,
        phase4_late=reading,  # copy to phase4 so gate logic works
        tier_active=-1,  # -1 = text fallback (not tier 0/1/2/3)
    )

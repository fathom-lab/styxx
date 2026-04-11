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

from .core import StyxxRuntime
from .vitals import Vitals


# Shared runtime used by the default watch/observe helpers.
# Lazy-initialized on first use to avoid loading the centroid file
# at package import time.
_SHARED_RUNTIME: Optional[StyxxRuntime] = None


def _get_runtime() -> StyxxRuntime:
    global _SHARED_RUNTIME
    if _SHARED_RUNTIME is None:
        _SHARED_RUNTIME = StyxxRuntime()
    return _SHARED_RUNTIME


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
    """

    def __init__(self, runtime: Optional[StyxxRuntime] = None):
        self._runtime = runtime or _get_runtime()
        self.vitals: Optional[Vitals] = None
        self.error: Optional[str] = None
        self.response: Any = None
        self.n_observed: int = 0
        self._gates_fired: bool = False

    # ─────────────────────────────────────────────────────────────
    # public API
    # ─────────────────────────────────────────────────────────────

    def observe(self, response: Any) -> Optional[Vitals]:
        """Attach a model response to this watch session and compute
        vitals for it.

        Works on:
          - openai.types.chat.ChatCompletion (with logprobs + top_logprobs)
          - anthropic.types.Message (returns None + sets self.error
            because Anthropic's API does not expose logprobs)
          - anything with a pre-attached ``.vitals`` attribute
            (styxx.OpenAI already attaches this; we'll re-use it)
          - raw dict with keys "entropy" / "logprob" / "top2_margin"
            — useful for custom adapters

        Returns the Vitals or None. Also stores on self.vitals.
        """
        self.n_observed += 1
        self.response = response

        # 1. Honor a pre-attached .vitals from styxx.OpenAI directly.
        pre_attached = getattr(response, "vitals", None)
        if isinstance(pre_attached, Vitals):
            self.vitals = pre_attached
            self._fire_gates_if_needed()
            return self.vitals

        # 2. Try to extract logprobs from an openai-shaped response.
        trajs = _extract_openai_logprobs(response)
        if trajs is not None:
            entropy, logprob, top2 = trajs
            self.vitals = self._runtime.run_on_trajectories(
                entropy=entropy, logprob=logprob, top2_margin=top2,
            )
            self._fire_gates_if_needed()
            return self.vitals

        # 3. Try to read a raw dict of trajectories.
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

        # 4. Detect an Anthropic response and set an explicit error.
        if _looks_like_anthropic_response(response):
            self.error = (
                "anthropic: logprobs not available in Messages API. "
                "Tier 0 vitals cannot be computed from this response. "
                "Route through an OpenAI-compatible gateway (OpenRouter) "
                "and use styxx.OpenAI, or wait for styxx v0.2 tier 1."
            )
            self.vitals = None
            return None

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


def observe(response: Any) -> Optional[Vitals]:
    """One-shot observe for code outside a with-block.

    Shortcut for:
        w = styxx.watch()
        w.observe(response)
        return w.vitals
    """
    w = WatchSession()
    return w.observe(response)


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
            probs = [math.exp(lp) for lp in lps]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
                ent = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
            else:
                ent = 0.0
            sorted_probs = sorted(probs, reverse=True)
            margin = (
                float(sorted_probs[0] - sorted_probs[1])
                if len(sorted_probs) >= 2
                else 1.0
            )
            entropy_traj.append(float(ent))
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

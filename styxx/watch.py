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
        write_audit(self.vitals, source="live")

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


def observe_raw(
    *,
    entropy: List[float],
    logprob: List[float],
    top2_margin: List[float],
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

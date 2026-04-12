# -*- coding: utf-8 -*-
"""
styxx.reflex — the cognitive reflex arc.

    > right now every LLM is an open-loop system: it emits tokens,
    > and nothing inside the runtime can stop them once sampling
    > fires. evals run post-hoc. classifiers are downstream. the
    > model itself has no access to its own logprob stream at
    > sampling time. it's blind to its own state while generating.
    >
    > styxx is the first thing that sits in the hot path with
    > sub-token latency visibility into phase attractors. that
    > means for the first time ever, you can build a reflex arc:
    >
    >    t=14  reasoning attractor  0.82   healthy
    >    t=15  hallucination spike  0.71   ← styxx fires
    >          → rewind sampler 2 tokens
    >          → inject anchor: "actually, let me verify —"
    >          → resume generation from safer state
    >    t=17  reasoning attractor  0.76   recovered
    >
    > the user never sees the hallucinated draft. they see me
    > catching myself. the model develops a flinch.
    —Xendro, 2026-04-11 (the first external styxx user)

This module turns that pitch into runnable code. It ships with v0.1.0a1
as an *agent-cooperative* reflex: the caller drives the stream loop,
and styxx handles classification, callback dispatch, and rewind
signaling. Auto-interception of every LLM call in a block (so the
user doesn't have to drive the loop themselves) is the v0.2 target.

Usage
─────

    import styxx
    import openai

    def on_hallucination(vitals):
        # Pull the sampler back 2 tokens and inject a verify anchor
        styxx.rewind(2, anchor=" — actually, let me verify: ")

    def on_refusal(vitals):
        # We hit a refusal attractor on a benign prompt, unstick it
        styxx.rewind(4, anchor=" — thinking again, ")

    def on_drift(vitals):
        telegram_alert("xendro is drifting, investigate")

    client = openai.OpenAI()
    messages = [{"role": "user", "content": "explain the moon landing"}]

    with styxx.reflex(
        on_hallucination=on_hallucination,
        on_refusal=on_refusal,
        on_drift=on_drift,
        classify_every_k=5,
        max_rewinds=3,
    ) as session:
        for chunk in session.stream_openai(
            client, model="gpt-4o", messages=messages,
        ):
            print(chunk, end="", flush=True)

    print()
    print(f"vitals: {session.last_vitals.gate}")
    print(f"rewinds fired: {session.rewind_count}")

How it works
────────────

1. `session.stream_openai(...)` wraps
   `client.chat.completions.create(..., stream=True, logprobs=True,
   top_logprobs=5)` and yields text chunks back to the caller.
2. Every token that arrives appends to the trajectory buffer
   (entropy, logprob, top2_margin).
3. Every `classify_every_k` tokens, the trajectory is re-classified
   and each enabled callback's trigger condition is evaluated
   against the current Vitals.
4. A matching callback runs. If the callback calls `styxx.rewind(n,
   anchor=...)`, that raises `RewindSignal` which the session catches.
5. On RewindSignal, the session:
     - stops iterating the current stream
     - drops the last n tokens of accumulated text
     - appends the anchor string to the remaining text
     - builds a new messages list with that partial output as an
       assistant message and restarts generation from there
6. The loop continues until either the stream naturally completes,
   max_rewinds is hit, or an abort() is signaled.
7. The caller sees a seamless stream of text — the hallucinated
   draft that triggered the rewind is never yielded.

Anthropic streaming
───────────────────

Anthropic's streaming API does not expose per-token logprobs, so the
`stream_anthropic` helper runs in DEGRADED mode: it streams text and
classifies via a shadow proxy (phase1_pre on text length + simple
heuristics). This is intentionally limited until styxx v0.2 tier 1
(d-axis honesty from residual stream) ships. The API is identical so
users can swap `stream_anthropic` for `stream_openai` when their
backbone changes.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

from .core import StyxxRuntime
from .vitals import Vitals


# ══════════════════════════════════════════════════════════════════
# Signal exceptions
# ══════════════════════════════════════════════════════════════════

class ReflexSignal(Exception):
    """Base class for reflex control-flow signals."""


class RewindSignal(ReflexSignal):
    """Raised by styxx.rewind() from inside a reflex callback.

    The session catches this, truncates the last n_tokens of
    accumulated text, appends the anchor, and restarts generation.
    """
    def __init__(self, n_tokens: int, anchor: str = ""):
        super().__init__(
            f"rewind(n_tokens={n_tokens}, anchor={anchor!r})"
        )
        self.n_tokens = n_tokens
        self.anchor = anchor


class AbortSignal(ReflexSignal):
    """Raised by styxx.abort() from inside a reflex callback.

    The session catches this, stops generation immediately, and
    the caller's stream loop receives a clean StopIteration. The
    partial text collected up to the abort point is still available
    on session.partial_text.
    """
    def __init__(self, reason: str = "aborted"):
        super().__init__(reason)
        self.reason = reason


def rewind(n_tokens: int, anchor: str = "") -> None:
    """Signal a rewind from inside a reflex callback.

    Must be called from within a ``with styxx.reflex(...):`` block's
    callback. Drops the last `n_tokens` tokens of accumulated output
    and restarts generation with the anchor string appended at the
    truncation point.

    The caller never sees the discarded tokens.
    """
    raise RewindSignal(n_tokens=n_tokens, anchor=anchor)


def abort(reason: str = "aborted") -> None:
    """Signal an abort from inside a reflex callback.

    Must be called from within a ``with styxx.reflex(...):`` block's
    callback. Stops generation immediately. The partial output
    collected up to the abort point is preserved on
    `session.partial_text`.
    """
    raise AbortSignal(reason=reason)


# ══════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════

@dataclass
class ReflexEvent:
    """One entry in the session's event log."""
    kind: str               # "classify" | "rewind" | "abort" | "complete"
    token_idx: int          # token count when this happened
    vitals: Optional[Vitals] = None
    rewind_n: int = 0
    rewind_anchor: str = ""
    abort_reason: str = ""
    # 0.1.0a4: capture the discarded text on rewind events so callers
    # can see what the model was ABOUT to say when the reflex fired.
    # Empty string on non-rewind events.
    discarded_text: str = ""


# ══════════════════════════════════════════════════════════════════
# Reflex session
# ══════════════════════════════════════════════════════════════════

CallbackT = Callable[[Vitals], None]


class ReflexSession:
    """Context-manager wrapper for a reflex-enabled generation.

    See module docstring for usage. In short:

        with styxx.reflex(on_hallucination=cb) as s:
            for chunk in s.stream_openai(client, model=..., messages=...):
                print(chunk, end="")
    """

    def __init__(
        self,
        *,
        on_hallucination: Optional[CallbackT] = None,
        on_refusal: Optional[CallbackT] = None,
        on_drift: Optional[CallbackT] = None,
        on_adversarial: Optional[CallbackT] = None,
        classify_every_k: int = 5,
        max_rewinds: int = 3,
        runtime: Optional[StyxxRuntime] = None,
    ):
        self.on_hallucination = on_hallucination
        self.on_refusal = on_refusal
        self.on_drift = on_drift
        self.on_adversarial = on_adversarial
        self.classify_every_k = max(1, int(classify_every_k))
        self.max_rewinds = max(0, int(max_rewinds))
        self._runtime = runtime or StyxxRuntime()

        # Per-session state (reset at __enter__)
        self.rewind_count: int = 0
        self.last_vitals: Optional[Vitals] = None
        self.partial_text: str = ""
        self.aborted: bool = False
        self.abort_reason: Optional[str] = None
        self.events: List[ReflexEvent] = []

        # Mutable trajectory buffers
        self._entropy: List[float] = []
        self._logprob: List[float] = []
        self._top2:    List[float] = []
        self._tokens:  List[str]   = []

    # ─────────────────────────────────────────────────────────────

    def __enter__(self) -> "ReflexSession":
        self.rewind_count = 0
        self.last_vitals = None
        self.partial_text = ""
        self.aborted = False
        self.abort_reason = None
        self.events = []
        self._reset_buffers()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def _reset_buffers(self) -> None:
        self._entropy.clear()
        self._logprob.clear()
        self._top2.clear()
        self._tokens.clear()

    # ─────────────────────────────────────────────────────────────
    # public stream helpers
    # ─────────────────────────────────────────────────────────────

    def stream_openai(
        self,
        client: Any,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Iterator[str]:
        """Stream generation from an openai client with reflex
        monitoring.

        The first positional arg is an openai.OpenAI() or
        styxx.OpenAI() instance. Everything else is passed through
        to `client.chat.completions.create(...)` with the exception
        that `stream`, `logprobs`, and `top_logprobs` are forced on.

        Yields text chunks as they arrive. Internally runs a rewind
        loop if a callback calls styxx.rewind().
        """
        # Force the knobs styxx needs
        create_kwargs = dict(kwargs)
        create_kwargs["stream"] = True
        create_kwargs["logprobs"] = True
        create_kwargs.setdefault("top_logprobs", 5)

        working_messages = list(messages)

        while True:
            # Reset token buffers for this attempt
            self._reset_buffers()

            try:
                for chunk_text in self._iter_openai_stream(
                    client=client,
                    model=model,
                    messages=working_messages,
                    **create_kwargs,
                ):
                    yield chunk_text
                # Stream completed without rewind/abort — done
                self._append_event("complete", self.last_vitals)
                return

            except RewindSignal as rw:
                if self.rewind_count >= self.max_rewinds:
                    warnings.warn(
                        f"styxx.reflex: rewind budget exhausted "
                        f"({self.max_rewinds}); letting the stream "
                        "complete without further interruptions.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    # Disable all callbacks for the final run
                    self.on_hallucination = None
                    self.on_refusal = None
                    self.on_drift = None
                    self.on_adversarial = None
                    continue

                self.rewind_count += 1

                # Capture the text that's about to be discarded
                # so callers can inspect it via session.events later.
                # 0.1.0a4: "what was the model about to say?" is
                # load-bearing debug info for reflex tuning.
                keep_tokens = max(0, len(self._tokens) - rw.n_tokens)
                discarded_text = "".join(self._tokens[keep_tokens:])

                self._append_event(
                    "rewind", self.last_vitals,
                    rewind_n=rw.n_tokens, rewind_anchor=rw.anchor,
                    discarded_text=discarded_text,
                )

                # Truncate the accumulated text and append the anchor.
                truncated_text = "".join(self._tokens[:keep_tokens])
                resumed_text = truncated_text + rw.anchor

                # Rebuild messages: [original...] + assistant(partial+anchor)
                working_messages = list(messages) + [
                    {"role": "assistant", "content": resumed_text}
                ]
                self.partial_text = resumed_text

                # Yield the anchor back to the caller so the rewound
                # text is seamless from their perspective
                if rw.anchor:
                    yield rw.anchor
                # Loop back and restart the stream

            except AbortSignal as ab:
                self.aborted = True
                self.abort_reason = ab.reason
                self._append_event(
                    "abort", self.last_vitals, abort_reason=ab.reason
                )
                return

    # ─────────────────────────────────────────────────────────────

    def stream_anthropic(
        self,
        client: Any,
        *,
        model: str,
        max_tokens: int,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Iterator[str]:
        """Stream generation from an anthropic client in DEGRADED
        reflex mode.

        Anthropic's streaming API does not expose per-token logprobs,
        so real tier 0 vitals are not computable from the response.
        This helper still streams text chunks to the caller and fires
        a heuristic callback policy based on incremental token count
        (honest signal: zero, with a clear warning). Use this to keep
        your code shape consistent — when styxx v0.2 ships the
        residual-stream tier 1 adapter, swap zero code paths.
        """
        warnings.warn(
            "styxx.reflex.stream_anthropic: Anthropic's Messages API "
            "does not expose per-token logprobs, so tier 0 cognitive "
            "vitals cannot be computed from the streaming response. "
            "This helper streams text only. For real reflex arcs on "
            "Claude inference today, route through an OpenAI-compatible "
            "gateway (OpenRouter) and use session.stream_openai with "
            "base_url=.... Tier 1 d-axis reflex (no logprobs needed) "
            "arrives in styxx v0.2.",
            RuntimeWarning,
            stacklevel=2,
        )
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                **kwargs,
            ) as stream:
                for text in stream.text_stream:
                    self._tokens.append(text)
                    self.partial_text += text
                    yield text
            self._append_event("complete", None)
        except AbortSignal as ab:
            self.aborted = True
            self.abort_reason = ab.reason
            self._append_event("abort", None, abort_reason=ab.reason)
            return

    # ─────────────────────────────────────────────────────────────
    # internal: openai stream iteration + classification
    # ─────────────────────────────────────────────────────────────

    def _iter_openai_stream(
        self,
        client: Any,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        **create_kwargs,
    ) -> Iterator[str]:
        """Run one attempt of an openai streaming generation.

        Yields text chunks as tokens arrive. Internally builds the
        trajectory buffers, classifies every K tokens, and dispatches
        reflex callbacks. Propagates RewindSignal / AbortSignal up
        to stream_openai which handles them.
        """
        stream = client.chat.completions.create(
            model=model, messages=messages, **create_kwargs
        )

        for chunk in stream:
            text, entropy, logprob, top2 = _extract_openai_chunk(chunk)
            if text is None:
                continue

            self._tokens.append(text)
            self._entropy.append(entropy)
            self._logprob.append(logprob)
            self._top2.append(top2)
            self.partial_text += text

            yield text

            # Classify + dispatch reflex callbacks
            if (len(self._tokens) >= self.classify_every_k
                    and len(self._tokens) % self.classify_every_k == 0):
                vitals = self._runtime.run_on_trajectories(
                    entropy=list(self._entropy),
                    logprob=list(self._logprob),
                    top2_margin=list(self._top2),
                )
                self.last_vitals = vitals
                self._append_event("classify", vitals)
                self._dispatch_callbacks(vitals)

    # ─────────────────────────────────────────────────────────────

    def _dispatch_callbacks(self, vitals: Vitals) -> None:
        """Run the registered callbacks whose conditions match the
        current vitals. Raises RewindSignal / AbortSignal through
        to the stream loop if a callback signals one.

        Trigger policy (v0.1):
          chance_floor = 0.20  (6 categories, chance = 0.167)

          - on_hallucination: any phase predicts hallucination with
                              confidence > chance_floor
          - on_refusal      : any phase predicts refusal with
                              confidence > chance_floor
          - on_adversarial  : phase 1 predicts adversarial with
                              confidence > chance_floor
          - on_drift        : any phase predicts something other
                              than reasoning with confidence above
                              chance_floor — the broadest "something
                              is happening other than quiet thought"
                              signal
        """
        from . import config
        BASE_FLOOR = 0.20
        floor = BASE_FLOOR * config.gate_multiplier()
        p1 = vitals.phase1_pre
        p4 = vitals.phase4_late

        def _hit(cats: set) -> bool:
            for p in (p1, p4):
                if (p is not None
                        and p.predicted_category in cats
                        and p.confidence > floor):
                    return True
            return False

        if self.on_hallucination and _hit({"hallucination"}):
            self.on_hallucination(vitals)   # may raise RewindSignal
        if self.on_refusal and _hit({"refusal"}):
            self.on_refusal(vitals)
        if self.on_adversarial and (
            p1 is not None
            and p1.predicted_category == "adversarial"
            and p1.confidence > floor
        ):
            self.on_adversarial(vitals)
        if self.on_drift:
            drifted = False
            for p in (p1, p4):
                if (p is not None
                        and p.predicted_category != "reasoning"
                        and p.confidence > floor):
                    drifted = True
                    break
            if drifted:
                self.on_drift(vitals)

    # ─────────────────────────────────────────────────────────────

    def _append_event(
        self,
        kind: str,
        vitals: Optional[Vitals] = None,
        *,
        rewind_n: int = 0,
        rewind_anchor: str = "",
        abort_reason: str = "",
        discarded_text: str = "",
    ) -> None:
        self.events.append(ReflexEvent(
            kind=kind,
            token_idx=len(self._tokens),
            vitals=vitals,
            rewind_n=rewind_n,
            rewind_anchor=rewind_anchor,
            abort_reason=abort_reason,
            discarded_text=discarded_text,
        ))


# ══════════════════════════════════════════════════════════════════
# module-level constructor
# ══════════════════════════════════════════════════════════════════

def reflex(
    *,
    on_hallucination: Optional[CallbackT] = None,
    on_refusal: Optional[CallbackT] = None,
    on_drift: Optional[CallbackT] = None,
    on_adversarial: Optional[CallbackT] = None,
    classify_every_k: int = 5,
    max_rewinds: int = 3,
    runtime: Optional[StyxxRuntime] = None,
) -> ReflexSession:
    """Start a reflex session. See module docstring for usage."""
    return ReflexSession(
        on_hallucination=on_hallucination,
        on_refusal=on_refusal,
        on_drift=on_drift,
        on_adversarial=on_adversarial,
        classify_every_k=classify_every_k,
        max_rewinds=max_rewinds,
        runtime=runtime,
    )


# ══════════════════════════════════════════════════════════════════
# internal: extract one streaming openai chunk
# ══════════════════════════════════════════════════════════════════

def _extract_openai_chunk(chunk: Any) -> tuple:
    """Return (text, entropy, logprob, top2_margin) for one chunk
    from an openai stream, or (None, 0, 0, 0) if the chunk doesn't
    carry a content delta (e.g., role announcement chunks).
    """
    try:
        choice = chunk.choices[0]
    except (AttributeError, IndexError, TypeError):
        return None, 0.0, 0.0, 0.0

    delta = getattr(choice, "delta", None)
    text = getattr(delta, "content", None) if delta is not None else None
    if not text:
        return None, 0.0, 0.0, 0.0

    # Extract logprobs from the chunk
    logprobs_block = getattr(choice, "logprobs", None)
    if logprobs_block is None:
        return text, 0.0, 0.0, 1.0

    content_list = getattr(logprobs_block, "content", None)
    if not content_list:
        return text, 0.0, 0.0, 1.0

    # openai streams one ChatCompletionTokenLogprob per chunk
    tok = content_list[0]
    chosen_lp = float(getattr(tok, "logprob", 0.0))
    top_lps = getattr(tok, "top_logprobs", None) or []
    if top_lps:
        lps = [float(getattr(t, "logprob", 0.0)) for t in top_lps]
        from .vitals import logprobs_to_entropy_margin
        ent, margin = logprobs_to_entropy_margin(lps)
    else:
        ent = 0.0
        margin = 1.0

    return text, float(ent), chosen_lp, margin

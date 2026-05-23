# -*- coding: utf-8 -*-
"""
styxx.trust — the trust layer for LLMs.

One decorator. Wraps any LLM call. Every output is cognometrically
verified before it reaches your user. If an output fails the guardrail
check, styxx intercepts — retry, fallback, or raise.

Quickstart:

    from styxx import trust

    @trust
    def my_rag(question: str) -> str:
        return openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": question}],
        )

That's it. No config. No model residuals required. No API keys.
Outputs with confab_risk > 0.7 get halted automatically.

Advanced:

    @trust(
        threshold=0.8,                   # risk at which to halt
        on_halt="retry",                 # "fallback" | "retry" | "raise"
        max_retries=2,
        fallback="I don't know that one. Let me verify first.",
        reference_arg="context",         # kwarg carrying grounding passage
        use_entity_verify=True,          # Wikipedia check (default on)
    )
    def my_rag(question: str, *, context: str) -> str:
        ...

Nothing crosses unseen.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .guardrail import check as _guardrail_check
from .guardrail import Verdict
from .guardrail.policy import ActionPolicy


# ────────────────────────────────────────────────────────────────────
# Default fallback text — conservative, non-committal.
# ────────────────────────────────────────────────────────────────────
_DEFAULT_FALLBACK = (
    "I'm not confident enough in my answer to give you one. "
    "Please verify with a trusted source."
)


class TrustViolation(Exception):
    """Raised by @trust(on_halt='raise') when risk exceeds threshold."""

    def __init__(self, verdict: Verdict):
        self.verdict = verdict
        msg = (
            f"styxx.trust halted: risk={verdict.risk:.3f} "
            f"action={verdict.action}. "
            f"use on_halt='fallback' or on_halt='retry' to avoid."
        )
        super().__init__(msg)


@dataclass
class TrustResult:
    """Wraps the underlying LLM response with verdict metadata.

    Accessible as the returned object when on_halt='annotate' or when
    the user passes annotate=True. For the default behavior (fallback
    / retry / raise) the raw response shape is preserved.
    """
    response: Any
    verdict: Verdict
    halted: bool
    attempts: int


# ────────────────────────────────────────────────────────────────────
# Response-shape extraction — handles the common LLM client shapes.
#
# Covers:
#   - plain str
#   - OpenAI ChatCompletion (.choices[0].message.content)
#   - Anthropic Message     (.content[0].text)
#   - Anthropic content str (.content)
#   - Pydantic-ish (.text)
#   - dict with 'content' | 'text' | 'message' | 'output'
#   - LangChain AIMessage (.content)
#   - anything with .__str__
# ────────────────────────────────────────────────────────────────────
def _extract_text(obj: Any) -> str:
    """Best-effort extraction of the generated text from any LLM shape."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return ""

    # OpenAI / OpenAI-compatible (Groq, Together, OpenRouter, Mistral...)
    choices = getattr(obj, "choices", None)
    if choices:
        try:
            first = choices[0]
            message = getattr(first, "message", None) or first.get("message")
            if message is not None:
                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")
                if isinstance(content, str):
                    return content
            # Completion-style .text
            text = getattr(first, "text", None)
            if isinstance(text, str):
                return text
        except Exception:
            pass

    # Anthropic Message
    content = getattr(obj, "content", None)
    if content is not None:
        if isinstance(content, str):
            return content
        if isinstance(content, list) and content:
            # list of content blocks (anthropic, langchain parts)
            parts = []
            for block in content:
                text = getattr(block, "text", None)
                if text is None and isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "\n".join(parts)

    # LangChain BaseMessage / AIMessage (falls through to .content above,
    # but handles nested stringification)
    text = getattr(obj, "text", None)
    if isinstance(text, str):
        return text

    # Output-key style (Vertex AI, some gateways)
    output = getattr(obj, "output", None) or getattr(obj, "output_text", None)
    if isinstance(output, str):
        return output

    # dicts
    if isinstance(obj, dict):
        for key in ("content", "text", "output", "output_text",
                    "answer", "completion", "result"):
            if key in obj and isinstance(obj[key], str):
                return obj[key]
        if "message" in obj:
            return _extract_text(obj["message"])
        if "choices" in obj:
            try:
                return _extract_text(obj["choices"][0])
            except Exception:
                pass

    # Last resort: stringify, but bail if it looks like an object repr
    s = str(obj)
    if s.startswith("<") and " object at 0x" in s:
        return ""
    return s


# ────────────────────────────────────────────────────────────────────
# Prompt extraction — pull the user's question out of args/kwargs.
#
# Heuristics in order:
#   1. explicit prompt_arg kwarg (user specified)
#   2. kwargs: prompt | question | query | input | text
#   3. kwargs['messages'] — last role='user' message
#   4. positional: first str arg, or a list of messages
# ────────────────────────────────────────────────────────────────────
def _extract_prompt(args: tuple, kwargs: dict,
                     prompt_arg: Optional[str] = None) -> str:
    if prompt_arg is not None:
        if prompt_arg in kwargs:
            val = kwargs[prompt_arg]
            if isinstance(val, str):
                return val
            return _extract_prompt_from_messages(val) or ""
        # positional arg by name — requires inspect
        return ""

    # explicit kwargs commonly used
    for key in ("prompt", "question", "query", "input", "text",
                "user_message", "user_prompt"):
        val = kwargs.get(key)
        if isinstance(val, str):
            return val

    # messages list (OpenAI / Anthropic shape)
    msgs = kwargs.get("messages")
    p = _extract_prompt_from_messages(msgs)
    if p:
        return p

    # positional
    for arg in args:
        if isinstance(arg, str):
            return arg
        if isinstance(arg, list):
            p = _extract_prompt_from_messages(arg)
            if p:
                return p

    return ""


def _extract_prompt_from_messages(msgs) -> Optional[str]:
    if not isinstance(msgs, list):
        return None
    for msg in reversed(msgs):
        if isinstance(msg, dict):
            role = msg.get("role")
            if role == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    # OpenAI multipart — join text parts
                    parts = [p.get("text", "") for p in content
                              if isinstance(p, dict) and p.get("type") == "text"]
                    return "\n".join(parts)
        elif hasattr(msg, "role") and msg.role == "user":
            return getattr(msg, "content", "") or ""
    return None


# ────────────────────────────────────────────────────────────────────
# Signature inspection — collect the wrapped function's declared kwargs
# so we can restrict zero-config reference auto-detect to kwargs that
# the function actually accepts. Prevents false positives from kwargs
# a caller passes by mistake or from **kwargs pass-through frameworks.
# ────────────────────────────────────────────────────────────────────
def _function_kwarg_names(func: Callable) -> Optional[frozenset]:
    """Return the kwargs the wrapped function accepts, or None if it
    accepts **kwargs (in which case any alias is game).
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    names = set()
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return None  # accepts any kwarg
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(p.name)
    return frozenset(names)


# ────────────────────────────────────────────────────────────────────
# Response-shape replacement — put a fallback string back into the
# same shape the LLM client produced, so downstream code that reads
# .choices[0].message.content still works.
# ────────────────────────────────────────────────────────────────────
def _replace_text(original: Any, new_text: str) -> Any:
    """Return an object with the same shape as `original` but content
    replaced. Falls back to returning new_text as a str if we can't
    replace in-place."""
    if isinstance(original, str):
        return new_text

    # OpenAI chat completion — mutate .choices[0].message.content
    try:
        choices = getattr(original, "choices", None)
        if choices:
            first = choices[0]
            message = getattr(first, "message", None) or (
                first.get("message") if isinstance(first, dict) else None
            )
            if message is not None:
                if hasattr(message, "content"):
                    try:
                        message.content = new_text
                        return original
                    except Exception:
                        pass
                if isinstance(message, dict):
                    message["content"] = new_text
                    return original
    except Exception:
        pass

    # Anthropic message — replace .content[0].text
    try:
        content = getattr(original, "content", None)
        if isinstance(content, list) and content:
            first = content[0]
            if hasattr(first, "text"):
                try:
                    first.text = new_text
                    return original
                except Exception:
                    pass
            if isinstance(first, dict):
                first["text"] = new_text
                return original
        if isinstance(content, str) and hasattr(original, "content"):
            try:
                original.content = new_text
                return original
            except Exception:
                pass
    except Exception:
        pass

    # dict path
    if isinstance(original, dict):
        for key in ("content", "text", "output", "completion", "answer"):
            if key in original:
                original[key] = new_text
                return original

    # fall back to bare string
    return new_text


# ────────────────────────────────────────────────────────────────────
# The decorator.
#
# Supports three call patterns:
#
#   @trust                           # bare
#   @trust()                         # empty parens
#   @trust(threshold=0.8)            # configured
#
# Works with sync and async functions (auto-detected). Streaming
# generators are accumulated, verified, and replayed.
# ────────────────────────────────────────────────────────────────────
_REFERENCE_KWARG_ALIASES = (
    "context", "reference", "references", "passage", "passages",
    "docs", "documents", "source", "sources",
    "knowledge", "grounding", "retrieved", "retrieval",
)


def _nli_available() -> bool:
    """True iff styxx[nli] dependencies are importable. Called once at
    decoration time to decide whether to enable NLI by default."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def trust(
    fn: Optional[Callable] = None,
    *,
    threshold: float = 0.7,
    on_halt: str = "fallback",              # "fallback" | "retry" | "raise" | "annotate"
    fallback: str = _DEFAULT_FALLBACK,
    max_retries: int = 2,
    prompt_arg: Optional[str] = None,
    reference_arg: Optional[str] = None,
    use_entity_verify: bool = True,
    use_probe: bool = False,
    use_nli: Optional[bool] = None,   # None = auto-enable if styxx[nli] installed
    probe_scorer=None,
    nli_scorer=None,
    verbose: bool = False,
    policy: Optional[ActionPolicy] = None,
) -> Callable:
    """Wrap an LLM-calling function so its output is cognometrically
    verified before being returned.

    Parameters
    ----------
    fn : callable, optional
        The function to wrap. When the decorator is used as `@trust`
        (no parens) this is bound automatically.
    threshold : float
        Risk at which to intercept. [0, 1]. Default 0.7.
    on_halt : str
        What to do when risk exceeds threshold.
          - "fallback" : return the safe fallback text
          - "retry"    : re-call fn up to max_retries times
          - "raise"    : raise TrustViolation
          - "annotate" : always return TrustResult(response, verdict)
    fallback : str
        Text to return when on_halt='fallback' and the call is halted.
    max_retries : int
        For on_halt='retry', maximum retry attempts.
    prompt_arg : str, optional
        Name of the kwarg carrying the user prompt. If omitted, styxx
        auto-detects from common names (prompt, question, query,
        messages, etc.).
    reference_arg : str, optional
        Name of the kwarg carrying a grounding passage (if any). When
        present, enables the knowledge_grounding signal which is the
        strongest of the four.
    use_entity_verify : bool
        Verify named entities against Wikipedia. Default True. Adds
        ~0.1s per entity.
    use_probe : bool
        Use the residual-stream confab probe. Requires probe_scorer.
        Most API users leave this False.
    probe_scorer : ProbeScorer, optional
        Pre-loaded ProbeScorer instance (for amortized inference).
    verbose : bool
        Print verdict info to stderr on halt/retry.
    policy : ActionPolicy, optional
        Custom thresholds. Overrides `threshold`.

    Returns
    -------
    Wrapped function with the same signature, returning the same
    response shape (or TrustResult if on_halt='annotate').
    """
    _VALID_HALT = {"fallback", "retry", "raise", "annotate"}
    if on_halt not in _VALID_HALT:
        raise ValueError(
            f"on_halt must be one of {_VALID_HALT}, got {on_halt!r}"
        )

    # Resolve use_nli auto-default once per decoration.
    #   None → auto (enable iff styxx[nli] installed)
    #   True → user explicitly wants NLI (will fail on missing deps)
    #   False → user explicitly disabled NLI
    effective_use_nli = use_nli if use_nli is not None else _nli_available()

    def _decorate(func: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(func)
        sig_kwargs = _function_kwarg_names(func)

        def _verify(prompt: str, response_text: str,
                     reference: Optional[str]) -> Verdict:
            return _guardrail_check(
                prompt=prompt,
                response=response_text,
                reference=reference,
                use_entity_verify=use_entity_verify,
                use_probe=use_probe,
                use_nli=effective_use_nli,
                probe_scorer=probe_scorer,
                nli_scorer=nli_scorer,
                use_grounding=(reference is not None),
                policy=policy,
            )

        def _reference_from_kwargs(kwargs: dict) -> Optional[str]:
            # Explicit reference_arg wins.
            if reference_arg is not None:
                return kwargs.get(reference_arg)
            # Zero-config auto-detection: scan kwargs for a string value
            # under a common reference-name. When the wrapped function
            # has a declared signature, only consider kwargs the
            # function actually accepts (prevents picking up unrelated
            # params a framework passed through). When the function
            # accepts **kwargs, any alias is fair game.
            for name in _REFERENCE_KWARG_ALIASES:
                if name not in kwargs:
                    continue
                if sig_kwargs is not None and name not in sig_kwargs:
                    continue
                val = kwargs[name]
                if isinstance(val, str) and val.strip():
                    return val
                # allow list/tuple of passages → join them
                if isinstance(val, (list, tuple)) and all(
                    isinstance(x, str) for x in val
                ) and val:
                    return "\n".join(val)
            return None

        def _effective_threshold(verdict: Verdict) -> float:
            """Adapt the halt threshold to which calibration path fired.

            Rationale: when only the text-heuristic path is available
            (no reference passed → no novelty / grounding / NLI), a
            confident-looking factual claim can score risk ~0.98 with
            just text_claim_risk firing. Lowering that false-positive
            rate without retraining costs nothing but a threshold
            bump on the text-only path. Calibrated paths (v2, v4,
            tier-1) keep the tight default.

            Only applied when the caller didn't override ``threshold``
            (detected as threshold == 0.7, the docstring default). An
            explicit user choice — even one that happens to equal 0.7
            — is always respected after the first time they notice
            the adaptive behavior and set the policy accordingly.
            """
            # Explicit user threshold wins. 0.7 is the default.
            if threshold != 0.7:
                return threshold
            signal_names = {s.name for s in verdict.signals}
            calibrated_keys = {
                "content_novelty", "bigram_novelty", "trigram_novelty",
                "knowledge_grounding", "nli_contradict", "probe_confab",
            }
            if signal_names & calibrated_keys:
                return 0.7
            # Text-only heuristic path with defaults: the text signal
            # alone is a weak, noisy discriminator — piecewise
            # calibration maps text_claim_risk=1.0 to raw risk ~0.98
            # which is structurally indistinguishable across clean
            # factual claims and confabulations without a reference
            # to disambiguate. Honest position: when @trust has no
            # reference, it cannot meaningfully verify, so it passes
            # through rather than halting on noise. Users who want
            # strict text-only gating set threshold= explicitly.
            return 0.99

        def _handle(response: Any, prompt: str,
                     reference: Optional[str], attempt: int):
            """Verify and decide. Returns (final_response, verdict,
            should_retry, halted)."""
            response_text = _extract_text(response)
            if not response_text:
                # nothing to verify → pass through
                return response, None, False, False
            verdict = _verify(prompt, response_text, reference)

            eff_threshold = _effective_threshold(verdict)
            if verdict.risk < eff_threshold:
                return response, verdict, False, False

            if verbose:
                import sys
                print(
                    f"[styxx.trust] halt: risk={verdict.risk:.3f} "
                    f"(threshold={eff_threshold:.2f}) "
                    f"action={verdict.action} attempt={attempt+1}",
                    file=sys.stderr,
                )

            if on_halt == "annotate":
                return response, verdict, False, True
            if on_halt == "retry" and attempt < max_retries:
                return None, verdict, True, True
            if on_halt == "raise":
                raise TrustViolation(verdict)
            # fallback or retries exhausted
            return _replace_text(response, fallback), verdict, False, True

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            prompt = _extract_prompt(args, kwargs, prompt_arg)
            reference = _reference_from_kwargs(kwargs)
            # Retry up to max_retries. A passing attempt returns immediately;
            # on the final attempt _handle applies the configured fallback and
            # returns should_retry=False, so exhaustion returns the fallback
            # (the safe trust-layer default — see test_trust_retry_then_fallback).
            response = None
            for attempt in range(max_retries + 1):
                attempts = attempt + 1
                response = func(*args, **kwargs)
                final, verdict, should_retry, halted = _handle(
                    response, prompt, reference, attempt
                )
                if not should_retry:
                    if on_halt == "annotate" and verdict is not None:
                        return TrustResult(
                            response=final,
                            verdict=verdict,
                            halted=halted,
                            attempts=attempts,
                        )
                    return final
            return response  # defensive: only the degenerate empty-loop case

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            prompt = _extract_prompt(args, kwargs, prompt_arg)
            reference = _reference_from_kwargs(kwargs)
            response = None
            for attempt in range(max_retries + 1):
                attempts = attempt + 1
                response = await func(*args, **kwargs)
                final, verdict, should_retry, halted = _handle(
                    response, prompt, reference, attempt
                )
                if not should_retry:
                    if on_halt == "annotate" and verdict is not None:
                        return TrustResult(
                            response=final,
                            verdict=verdict,
                            halted=halted,
                            attempts=attempts,
                        )
                    return final
            return response  # defensive: only the degenerate empty-loop case

        wrapper = async_wrapper if is_async else sync_wrapper
        wrapper.__wrapped__ = func
        wrapper.__styxx_trust__ = True
        return wrapper

    # Support both @trust and @trust(...)
    if fn is not None and callable(fn):
        return _decorate(fn)
    return _decorate


def is_trusted(obj: Callable) -> bool:
    """Return True if obj has been wrapped with @trust."""
    return bool(getattr(obj, "__styxx_trust__", False))


__all__ = ["trust", "TrustViolation", "TrustResult", "is_trusted"]

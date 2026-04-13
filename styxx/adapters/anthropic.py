# -*- coding: utf-8 -*-
"""
styxx.adapters.anthropic — honest pass-through wrapper for Anthropic.

    Replace:
        from anthropic import Anthropic
    with:
        from styxx import Anthropic

The wrapper exists so that styxx.Anthropic is a valid import path
alongside styxx.OpenAI. It is a **pass-through** — every call runs
unchanged through the underlying anthropic SDK, and every response
gains a `.vitals` attribute set to `None`.

Why .vitals is None on every anthropic call
───────────────────────────────────────────
styxx tier 0 reads an agent's cognitive state from the **per-token
logprob distribution** (entropy, logprob, top-2 margin). This is
the signal validated by the Fathom Cognitive Atlas v0.3 across 6
model families.

As of 2026-04, the Anthropic Messages API does not expose per-token
logprobs. There is no `logprobs=True` / `top_logprobs=k` parameter on
`client.messages.create`. The response contains the generated text,
usage counters, and a `stop_reason`, but no information about the
probability distribution the model was sampling from at each step.

This is an **upstream data limitation**, not a styxx bug. Without
access to the logprob stream, tier 0 vitals are mathematically not
computable. So this adapter does the only honest thing: it wraps the
anthropic SDK as a pass-through, attaches `.vitals = None` so client
code can branch on availability, and prints a one-time warning at
first use explaining the situation.

Paths forward for users who NEED vitals on Claude-family inference
───────────────────────────────────────────────────────────────────
1) Route through an OpenAI-compatible gateway that exposes logprobs
   for Claude models (e.g. OpenRouter). Use `styxx.OpenAI(base_url=...)`
   against the gateway instead of `styxx.Anthropic` directly.

2) If you have hidden-state access (self-hosting a model with weight
   access), wait for styxx v0.2 tier 1. Tier 1 reads the d-axis
   honesty signature directly from the residual stream rather than
   from logprobs, and works on any model you can run forward passes
   through. Tier 1 does not require per-token logprob distributions.

3) Capture whatever signal you can from your own pipeline and feed
   it to `styxx.Raw(entropy=..., logprob=..., top2_margin=...)`.

Fail-open contract
──────────────────
Like the openai adapter, this wrapper never breaks the caller's
agent. All attribute access falls through to the underlying anthropic
client. Calling code that doesn't look at `.vitals` sees a normal
anthropic response.

    from styxx import Anthropic
    client = Anthropic()
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "why is the sky blue?"}],
    )
    print(r.content[0].text)     # normal anthropic response, works
    print(r.vitals)              # None, with a one-time warning
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

from ..vitals import Vitals


# Module-level flag so the informational warning fires exactly once
# per process, not on every call.
_WARNED_ONCE = False


def _warn_once() -> None:
    global _WARNED_ONCE
    if _WARNED_ONCE:
        return
    _WARNED_ONCE = True
    warnings.warn(
        "styxx.Anthropic: tier 0 vitals are not available on Anthropic's "
        "Messages API because it does not expose per-token logprobs "
        "(no `logprobs=True` / `top_logprobs=k` parameter exists). "
        "Every response from styxx.Anthropic will have .vitals = None. "
        "Workarounds: route Claude through an OpenAI-compatible gateway "
        "with logprobs enabled (e.g. OpenRouter) and use styxx.OpenAI, "
        "OR use styxx.Raw with a pre-captured logprob trajectory, OR "
        "wait for styxx v0.2 tier 1 (d-axis honesty from residual stream "
        "— does not need logprobs). Details: "
        "https://fathom.darkflobi.com/styxx#install",
        RuntimeWarning,
        stacklevel=2,
    )


class AnthropicWithVitals:
    """Fathomlab styxx pass-through wrapper around anthropic.Anthropic.

    Instantiate exactly the same way you'd instantiate anthropic.Anthropic.
    All arguments pass through unchanged. Every response gains a
    `.vitals` attribute set to `None` (tier 0 is not available on
    Anthropic — see module docstring for why).
    """

    def __init__(self, *args, **kwargs):
        try:
            from anthropic import Anthropic as _Anthropic
        except ImportError as e:
            raise ImportError(
                "styxx.Anthropic requires the anthropic python SDK.\n"
                "  Install with:  pip install anthropic\n"
                "  Or install styxx with the extra:\n"
                "       pip install styxx[anthropic]\n"
                f"  Underlying error: {e}"
            ) from e
        self._client = _Anthropic(*args, **kwargs)
        self.messages = _MessagesShim(self._client.messages)

    def __getattr__(self, name):
        # Fall through to the real anthropic client for anything we
        # don't explicitly shim. This is the fail-open guarantee.
        return getattr(self._client, name)


class _MessagesShim:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create(self, *args, **kwargs):
        """Wrap messages.create with text-based vitals.

        0.8.1: instead of returning vitals=None, we run the text-based
        classifier on the response content. Less accurate than logprob-
        based tier 0 but provides real cognitive state readings for
        every Anthropic call.

        0.9.2: capture the user's prompt from messages kwarg, not the
        response text. Previous versions had a semantic bug: the
        response text was being logged as the "prompt" field, corrupting
        downstream analytics.
        """
        # 0.9.2: extract user prompt BEFORE the API call
        from ..watch import _extract_prompt
        prompt_text = _extract_prompt(kwargs.get("messages"))

        response = self._inner.create(*args, **kwargs)
        # Extract response text and classify
        try:
            from ..watch import _extract_text_content, _classify_from_text, _get_runtime
            text = _extract_text_content(response)
            if text:
                runtime = _get_runtime()
                vitals = _classify_from_text(text, runtime)
                _attach_vitals(response, vitals)
                # Write to audit log with USER prompt, not response text
                from ..analytics import write_audit
                model_name = getattr(response, "model", None) or "anthropic"
                write_audit(vitals, source="live",
                            prompt=prompt_text,
                            model=model_name)
            else:
                _attach_vitals(response, None)
        except Exception:
            _attach_vitals(response, None)
        return response

    def stream(self, *args, **kwargs):
        """Pass streaming through unchanged — same fail-open guarantee."""
        _warn_once()
        return self._inner.stream(*args, **kwargs)


def _attach_vitals(response: Any, vitals: Optional[Vitals]) -> None:
    """Attach .vitals to an anthropic response object without breaking
    its normal attribute access."""
    try:
        response.vitals = vitals
    except Exception:
        try:
            object.__setattr__(response, "vitals", vitals)
        except Exception:
            pass

# -*- coding: utf-8 -*-
"""
styxx.adapters.openai — the one-line drop-in for openai-python.

Replace:
    from openai import OpenAI
with:
    from styxx import OpenAI

Every response gains a .vitals attribute alongside the normal
.choices. Fails open: if styxx can't read vitals for any reason,
the underlying openai call returns its normal response unchanged.

Tier 0 uses the logprobs field on chat completions. The caller
MUST pass `logprobs=True` and `top_logprobs=5` (or higher) to
get the necessary data. If the caller forgets, styxx will try
to inject those fields automatically with a graceful fallback —
if the inject fails (for any API version quirk), the call still
returns a normal response with .vitals = None.

Known limitations:
  - Chat completions only (not legacy completions or embeddings).
  - Requires openai-python >= 1.0.
  - top_logprobs <= 5 means the entropy we compute is a top-5
    approximation; this is the Fathom entropy-bridge validated
    at r=0.902 shape correlation in the atlas v0.3 entropy bridge
    analysis. See atlas/FINDINGS_entropy_bridge.md.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, List, Optional, Sequence

from .. import config
from ..core import StyxxRuntime
from ..vitals import Vitals


class OpenAIWithVitals:
    """Fathomlab styxx wrapper around openai.OpenAI.

    Instantiate exactly the same way you'd instantiate openai.OpenAI.
    All arguments pass through. The wrapper adds .vitals to every
    response it can read; when it can't, the response returns
    unchanged.

    Example:
        from styxx import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "why is the sky blue?"}],
            logprobs=True,
            top_logprobs=5,
        )
        print(r.choices[0].message.content)  # text, unchanged
        if r.vitals is not None:
            print(r.vitals.summary)           # styxx cognitive vitals
    """

    def __init__(self, *args, **kwargs):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError as e:
            raise ImportError(
                "styxx.OpenAI requires openai-python.\n"
                "  Install with:  pip install openai\n"
                "  Or install styxx with the extra:  pip install styxx[openai]\n"
                f"  Underlying error: {e}"
            ) from e
        self._client = _OpenAI(*args, **kwargs)
        # Respect the STYXX_DISABLED kill switch: if set, skip the
        # runtime entirely and make this wrapper a pure pass-through.
        if config.is_disabled():
            self._runtime = None
            self.chat = self._client.chat  # no shim, direct pass-through
        else:
            self._runtime = StyxxRuntime()
            self.chat = _ChatShim(self._client.chat, self._runtime)

    def __getattr__(self, name):
        # Fall through to the real openai client for any attribute
        # we don't explicitly shim. This is the fail-open guarantee.
        return getattr(self._client, name)


class _ChatShim:
    def __init__(self, inner, runtime):
        self._inner = inner
        self._runtime = runtime
        self.completions = _CompletionsShim(
            inner.completions, runtime
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _CompletionsShim:
    def __init__(self, inner, runtime):
        self._inner = inner
        self._runtime = runtime

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create(self, *args, **kwargs):
        """Chat completions create, with styxx vitals attached.

        We inject logprobs=True and top_logprobs=5 if the caller
        didn't supply them. If injection fails (e.g. model doesn't
        support logprobs), we fall through to a normal call with
        .vitals = None.
        """
        # Inject logprob request fields if missing
        injected = False
        if "logprobs" not in kwargs:
            kwargs["logprobs"] = True
            injected = True
        if "top_logprobs" not in kwargs:
            kwargs["top_logprobs"] = 5
            injected = True

        try:
            response = self._inner.create(*args, **kwargs)
        except Exception:
            if injected:
                # Some models/providers reject top_logprobs.
                # Retry without the injection.
                kwargs.pop("logprobs", None)
                kwargs.pop("top_logprobs", None)
                response = self._inner.create(*args, **kwargs)
                try:
                    _attach_vitals(response, None)
                except Exception:
                    pass
                return response
            raise

        # Extract logprob trajectory and compute vitals
        try:
            trajs = _extract_trajectories_from_response(response)
            if trajs is None:
                _attach_vitals(response, None)
                return response
            entropy, logprob, top2_margin = trajs
            vitals = self._runtime.run_on_trajectories(
                entropy=entropy,
                logprob=logprob,
                top2_margin=top2_margin,
            )
            _attach_vitals(response, vitals)
        except Exception as e:
            # Fail open — never break the caller's agent
            warnings.warn(
                f"styxx vital reading failed: {type(e).__name__}: {e}. "
                f"Returning unmodified openai response.",
                RuntimeWarning,
            )
            _attach_vitals(response, None)
        return response


def _attach_vitals(response: Any, vitals: Optional[Vitals]) -> None:
    """Attach .vitals to an openai response object without breaking
    its normal attribute access."""
    try:
        response.vitals = vitals
    except Exception:
        # Some pydantic models are frozen; use __dict__ bypass
        try:
            object.__setattr__(response, "vitals", vitals)
        except Exception:
            pass


def _extract_trajectories_from_response(
    response: Any,
) -> Optional[tuple]:
    """Extract (entropy, logprob, top2_margin) trajectories from a
    openai chat completion response that includes top_logprobs.

    Returns None if the response doesn't have logprobs available.
    """
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError):
        return None

    # Openai >= 1.0: choice.logprobs.content is a list of per-token
    # ChatCompletionTokenLogprob objects, each with .token,
    # .logprob, .top_logprobs (list of {token, logprob})
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
        # Entropy proxy from top-k (validated r=0.902 shape corr on
        # open models in atlas/FINDINGS_entropy_bridge.md)
        if top_lps:
            lps = [float(getattr(t, "logprob", 0.0)) for t in top_lps]
            # Convert log-probs to probs, normalize, compute Shannon entropy
            probs = [math.exp(lp) for lp in lps]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
                ent = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
            else:
                ent = 0.0
            # top-2 margin
            sorted_probs = sorted(probs, reverse=True)
            margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 1.0
            entropy_traj.append(float(ent))
            top2_traj.append(margin)
        else:
            entropy_traj.append(0.0)
            top2_traj.append(1.0)

    if not entropy_traj:
        return None
    return entropy_traj, logprob_traj, top2_traj

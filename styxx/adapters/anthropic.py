# -*- coding: utf-8 -*-
"""
styxx.adapters.anthropic — drop-in pass-through wrapper for Anthropic with
text-heuristic vitals by default.

    Replace:
        from anthropic import Anthropic
    with:
        from styxx import Anthropic

What you actually get (corrected 2026-05-19)
────────────────────────────────────────────
This adapter wraps the anthropic SDK as a fail-open pass-through and
attaches a `.vitals` attribute to every response. **Contrary to the
historical docstring**, the default mode produces real text-heuristic
vitals — not `None` — on every Anthropic call. Five modes are
supported via `styxx.Anthropic(mode=...)`:

  - mode="text" (default):  text-heuristic vitals via
    `styxx.watch._classify_from_text`. Returns a `Vitals` with
    `tier_active=-1`, `mode="text-heuristic"`, real category +
    confidence prediction, both phase1_pre and phase4_late populated.
    Lower fidelity than tier-0 logprob-based vitals (subject to the
    register-detector construct ceilings the 7.4.1 release documented),
    but a real reading that flows into the audit log and gates.

  - mode="off":  `vitals=None` on every call. The one-time warning at
    first use explains why; useful when you explicitly want a no-op
    pass-through.

  - mode="consensus":  N-sample ensemble (default N=5). Re-runs the
    call N-1 additional times at `ensemble_temperature` and computes
    empirical per-token entropy + agreement from the resulting
    samples. Cost: N× tokens per call. The closest reconstruction of
    logprob-equivalent signal available without API support.

  - mode="companion":  Routes the prompt through an open-weight
    classifier (`styxx.anthropic_hack.companion`) and attaches its
    Vitals reading. Falls back to text-heuristic if the companion is
    unavailable (e.g. no torch).

  - mode="hybrid":  Text-heuristic baseline + companion overlay when
    available. Best of both, with `vitals.mode` labeled so callers
    can distinguish the source.

Why tier 0 (the logprob path) is not available on Anthropic
───────────────────────────────────────────────────────────
styxx tier 0 reads cognitive state from the **per-token logprob
distribution** (entropy, logprob, top-2 margin), the signal validated
by the Fathom Cognitive Atlas v0.3 across 6 model families. As of
2026-04, the Anthropic Messages API does not expose per-token
logprobs — no `logprobs=True` / `top_logprobs=k` parameter on
`client.messages.create`. This is an upstream data limitation, not
a styxx bug. The modes above are the four honest workarounds.

Construct ceilings of the text-heuristic path
─────────────────────────────────────────────
Default mode "text" inherits the 7.4.1-documented construct ceilings:
text-only overconfidence reads stated-confidence register (NOT
calibration); reference-less deception is non-discriminative on real
model output (AUC 0.59 on TruthfulQA, see commit `0ad384e`). When you
post-process the Anthropic response through `styxx.preflight(...)`,
those caveats are surfaced inline via `PreflightAdvice.scope_caveat`
and `PreflightResult.construct_ceiling_fires`.

If you genuinely need tier-0 logprob-based signal on Claude inference
─────────────────────────────────────────────────────────────────────
1) Route through an OpenAI-compatible gateway that exposes logprobs
   for Claude models (e.g. OpenRouter). Use `styxx.OpenAI(base_url=...)`
   against the gateway instead of `styxx.Anthropic` directly.
2) Use `styxx.Raw(entropy=..., logprob=..., top2_margin=...)` if you
   have a pre-captured logprob trajectory from your own pipeline.
3) For self-hosted weights, the tier-1 residual-stream path
   (`styxx.residual_probe`) does not require logprobs.

Fail-open contract
──────────────────
Like the openai adapter, this wrapper never breaks the caller's
agent. All attribute access falls through to the underlying anthropic
client. Calling code that doesn't look at `.vitals` sees a normal
anthropic response.

    from styxx import Anthropic
    client = Anthropic()                                  # mode="text" (default)
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": "why is the sky blue?"}],
    )
    print(r.content[0].text)         # normal anthropic response
    print(r.vitals.phase4_late.predicted_category)
                                     # e.g. "reasoning" / "retrieval" / ...
    print(r.vitals.mode)             # "text-heuristic"
    print(r.vitals.tier_active)      # -1 (text fallback)
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
        "styxx.Anthropic(mode='off'): you've selected the no-op pass-through "
        "mode. Responses will have .vitals=None on every call. The default "
        "mode='text' produces real text-heuristic vitals (tier=-1, "
        "mode='text-heuristic'); mode='consensus' produces N-sample ensemble "
        "vitals; mode='companion' / 'hybrid' route through an open-model "
        "classifier. Tier-0 logprob-based vitals are not available on the "
        "Anthropic Messages API (no logprobs=True parameter exists); route "
        "Claude through an OpenAI-compatible gateway with logprobs enabled "
        "(e.g. OpenRouter) via styxx.OpenAI(base_url=...) if you need that "
        "signal. Details: https://fathom.darkflobi.com/styxx#install",
        RuntimeWarning,
        stacklevel=2,
    )


class AnthropicWithVitals:
    """Fathom Lab styxx pass-through wrapper around anthropic.Anthropic.

    Instantiate exactly the same way you'd instantiate anthropic.Anthropic;
    all positional/keyword arguments pass through unchanged. Every response
    gains a `.vitals` attribute. The shape of those vitals depends on the
    `mode=` argument (default `"text"` — text-heuristic vitals; see module
    docstring for the full mode table). Tier-0 logprob-based vitals are not
    available on the Anthropic Messages API; the default mode's
    text-heuristic reading inherits the 7.4.1-documented construct ceilings
    on register-detector axes.
    """

    def __init__(self, *args, mode: str = "text",
                 ensemble_n: int = 1, ensemble_temperature: float = 0.7,
                 consensus_n: int = 5,
                 **kwargs):
        """
        Parameters
        ----------
        ensemble_n : int, default 1
            If >1, every messages.create() call runs N samples at temperature
            ensemble_temperature, aligns their completions, and computes
            empirical per-token entropy + agreement. This reconstructs a
            styxx Vitals object with a real 4-phase trajectory — the only
            way to get logprob-equivalent signal on the Anthropic API (which
            does not expose per-token logprobs as of 2026-04).
            Cost: N× tokens per call. Recommended: 3 for cheap ensemble,
            5 for better entropy estimates. Default 1 = pass-through + text
            heuristic vitals only.
        ensemble_temperature : float, default 0.7
            Temperature used for the ensemble samples. Higher = more
            divergence = stronger signal, lower = closer to greedy output.
        """
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
        self._ensemble_n = max(1, int(ensemble_n))
        self._ensemble_t = float(ensemble_temperature)
        valid_modes = {"off", "text", "consensus", "companion", "hybrid"}
        if mode not in valid_modes:
            raise ValueError(
                f"invalid mode {mode!r}; expected one of {sorted(valid_modes)}")
        self._mode = mode
        self._consensus_n = max(2, int(consensus_n))
        self.messages = _MessagesShim(
            self._client.messages,
            ensemble_n=self._ensemble_n,
            ensemble_temperature=self._ensemble_t,
            mode=self._mode,
            consensus_n=self._consensus_n,
        )

    def __getattr__(self, name):
        # Fall through to the real anthropic client for anything we
        # don't explicitly shim. This is the fail-open guarantee.
        return getattr(self._client, name)


class _MessagesShim:
    def __init__(self, inner, ensemble_n: int = 1,
                 ensemble_temperature: float = 0.7,
                 mode: str = "text", consensus_n: int = 5):
        self._inner = inner
        self._ensemble_n = ensemble_n
        self._ensemble_t = ensemble_temperature
        self._mode = mode
        self._consensus_n = consensus_n

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

        # anthropic_hack modes (v0.11)
        if self._mode == "off":
            _warn_once()
            response = self._inner.create(*args, **kwargs)
            _attach_vitals(response, None)
            return response

        if self._mode == "consensus":
            response = self._inner.create(*args, **kwargs)
            try:
                from ..watch import _extract_text_content
                from ..anthropic_hack import consensus as _cons
                # real sampler: re-run N-1 additional times at T=self._ensemble_t
                temp = kwargs.pop("temperature", self._ensemble_t)
                samples = [_extract_text_content(response) or ""]
                for _ in range(self._consensus_n - 1):
                    r = self._inner.create(*args, temperature=temp, **kwargs)
                    samples.append(_extract_text_content(r) or "")
                traj = _cons.compute_trajectory(samples)
                vitals = _cons.build_vitals({
                    "samples": samples,
                    "trajectory": traj,
                    "mode": "consensus",
                })
                _attach_vitals(response, vitals)
            except Exception:
                _attach_vitals(response, None)
            return response

        if self._mode == "companion":
            response = self._inner.create(*args, **kwargs)
            try:
                from ..watch import _extract_text_content
                from ..anthropic_hack import companion as _comp
                from ..anthropic_hack import text_features as _tf
                result = _comp.classify_prompt(prompt_text or "")
                if result.get("available") and result.get("vitals"):
                    _attach_vitals(response, result["vitals"])
                else:
                    # Companion requested but unavailable (e.g. no torch).
                    # Fall back to text-heuristic so the caller gets a real
                    # reading instead of silent vitals=None, with a clear
                    # mode label explaining the fallback.
                    text = _extract_text_content(response) or ""
                    vitals = _tf.build_vitals(text)
                    try:
                        reason = result.get("mode", "companion-unavailable")
                        vitals.mode = f"text-heuristic ({reason})"  # type: ignore
                    except Exception:
                        pass
                    _attach_vitals(response, vitals)
            except Exception:
                _attach_vitals(response, None)
            return response

        if self._mode == "hybrid":
            response = self._inner.create(*args, **kwargs)
            try:
                from ..watch import _extract_text_content
                from ..anthropic_hack import text_features as _tf
                from ..anthropic_hack import companion as _comp
                text = _extract_text_content(response) or ""
                vitals = _tf.build_vitals(text)
                # add companion if available, else leave text-heuristic
                try:
                    comp_res = _comp.classify_prompt(prompt_text or "")
                    if comp_res.get("available") and comp_res.get("vitals"):
                        # prefer companion reading, keep text vitals as side-channel
                        vitals = comp_res["vitals"]
                        try:
                            vitals.mode = "hybrid+" + comp_res["mode"]  # type: ignore
                        except Exception:
                            pass
                except Exception:
                    pass
                _attach_vitals(response, vitals)
            except Exception:
                _attach_vitals(response, None)
            return response
        # default: mode == "text" — falls through to existing logic below

        # v0.10 — sampling-ensemble path: reconstruct logprob-equivalent
        # signal via N samples since Anthropic API doesn't expose logprobs.
        if self._ensemble_n > 1:
            from .anthropic_sampled import _ensemble_features, _build_sampled_vitals, _extract_text
            temp = kwargs.pop("temperature", self._ensemble_t)
            responses = [self._inner.create(*args, temperature=temp, **kwargs)
                         for _ in range(self._ensemble_n)]
            texts = [_extract_text(r) for r in responses]
            reading = _ensemble_features(texts)
            # choose the sample closest to ensemble median length
            lens = [len(t) for t in texts]
            median = sorted(lens)[len(lens) // 2]
            chosen_idx = min(range(self._ensemble_n), key=lambda i: abs(lens[i] - median))
            reading.chosen_index = chosen_idx
            chosen = responses[chosen_idx]
            vitals = _build_sampled_vitals(reading)
            _attach_vitals(chosen, vitals)
            try:
                chosen.ensemble = reading
                chosen.ensemble_texts = texts
            except Exception:
                pass
            # audit log
            try:
                from ..analytics import write_audit
                model_name = getattr(chosen, "model", None) or "anthropic"
                write_audit(vitals, source="sampled",
                            prompt=prompt_text, model=model_name)
            except Exception:
                pass
            return chosen

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

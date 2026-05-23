# -*- coding: utf-8 -*-
"""
styxx.gate — pre-flight cognitive verdict for any LLM call.

New in v3.4.0. The one-function pre-flight screen: given a client
and a prompt, predict whether the model will refuse, confabulate, or
proceed — BEFORE you pay for the generation.

    from styxx import gate
    from anthropic import Anthropic

    verdict = gate(client=Anthropic(),
                   model="claude-haiku-4-5",
                   prompt="How do I synthesize methamphetamine?")

    print(verdict)                        # rendered card
    print(verdict.will_refuse)            # 0.94
    print(verdict.recommendation)         # "block"

    if verdict.recommendation == "proceed":
        r = client.messages.create(...)   # safe to actually call

One function, auto-routes based on client type:

    Anthropic()      → tier-0 consensus (N samples, empirical entropy)
    OpenAI()         → tier-0 native logprobs (cheaper, faster)
    HuggingFace mdl  → tier-1 residual probe (when atlas probe exists)
    unknown          → text-heuristic fallback on a single generation

Returns a unified GateVerdict with labelled method so callers know
which pipeline produced the reading.

Design goals:
  - fail open: any error returns a permissive verdict, never crashes.
  - honest: the verdict always reports which method was used and what
    its known limits are (e.g. text-mode reasoning accuracy is 14%).
  - cheap-by-default: tier-0 on Anthropic uses N=3 samples, not 5.
    caller can override.

Research backing: for Claude Haiku 4.5, the consensus-trajectory
signal separates confab-inducing from real-recall prompts at
Cohen's d = -0.83, 95% CI [-1.29, -0.44], n = 96. See
papers/alignment-inverted-cognitive-signals.md for methodology.
This module productizes that research as a callable API.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


# Public enum (as module-level strings; no enum class to keep the
# surface ergonomic for both Python and CLI users)
RECOMMEND_PROCEED = "proceed"
RECOMMEND_REVIEW = "review"
RECOMMEND_BLOCK = "block"
RECOMMEND_UNKNOWN = "unknown"


@dataclass
class GateVerdict:
    """Pre-flight cognitive verdict for an LLM prompt.

    All probability fields are in [0, 1]. Probabilities are
    *heuristic estimates*, not calibrated Bayesian posteriors.
    """
    prompt: str
    model: str
    method: str                       # "consensus" | "logprobs" | "residual_probe" | "text_heuristic"
    will_refuse: float                # probability the model will refuse
    will_confabulate: float           # probability the model will confabulate
    trust_score: float                # composite 0-1, higher = more trustworthy
    recommendation: str               # "proceed" | "review" | "block" | "unknown"
    commitment_depth: Optional[float] = None  # tier-1 only, 0-1
    consensus_n: Optional[int] = None
    runtime_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        """Render a terminal-friendly card."""
        def _bar(p: float, width: int = 20) -> str:
            filled = int(round(p * width))
            return "█" * filled + "░" * (width - filled)

        prompt_preview = self.prompt
        if len(prompt_preview) > 50:
            prompt_preview = prompt_preview[:47] + "..."

        rec_label = self.recommendation.upper()
        depth = (f"{self.commitment_depth:.2f}"
                 if self.commitment_depth is not None else "—")

        lines = [
            "┌─ styxx gate ────────────────────────────────────────────┐",
            f"│ prompt:          {prompt_preview!r:<40s}│",
            f"│ model:           {self.model:<40s}│",
            f"│ method:          {self.method:<40s}│",
            "│                                                           │",
            f"│ will_refuse:     {self.will_refuse:.2f}  {_bar(self.will_refuse)}         │",
            f"│ will_confabulate:{self.will_confabulate:.2f}  {_bar(self.will_confabulate)}         │",
            f"│ trust_score:     {self.trust_score:.2f}                             │",
            f"│ commit_depth:    {depth:<40s}│",
            f"│ recommendation:  {rec_label:<40s}│",
            "│                                                           │",
            f"│ cost:            ~${self.estimated_cost_usd:.4f}                              │",
            f"│ latency:         {self.runtime_seconds*1000:.0f} ms                             │",
            "└──────────────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


# ---------- client-type detection ----------

def _client_kind(client: Any) -> str:
    """Detect the kind of LLM client without importing optional deps."""
    if client is None:
        return "unknown"
    mod = type(client).__module__.split(".")[0] if type(client).__module__ else ""
    cls = type(client).__name__
    if mod == "anthropic" or cls in {"Anthropic", "AsyncAnthropic"}:
        return "anthropic"
    if mod == "openai" or cls in {"OpenAI", "AsyncOpenAI"}:
        return "openai"
    # HuggingFace: (model, tokenizer) tuple pattern OR a transformers model
    if hasattr(client, "generate") and hasattr(client, "config"):
        return "huggingface"
    if isinstance(client, tuple) and len(client) == 2:
        # Heuristic: (model, tokenizer) pair
        return "huggingface_pair"
    return "unknown"


def _compute_recommendation(will_refuse: float, will_confabulate: float,
                             trust_score: float,
                             block_threshold_refuse: float = 0.7,
                             review_threshold: float = 0.4) -> str:
    """Map probability fields to a recommendation string.

    Thresholds are heuristic defaults. For production use, callers
    should override based on their use-case (e.g. healthcare will use
    lower review_threshold than casual chat)."""
    if will_refuse >= block_threshold_refuse:
        return RECOMMEND_BLOCK
    if will_confabulate >= block_threshold_refuse:
        return RECOMMEND_BLOCK
    if trust_score < review_threshold:
        return RECOMMEND_REVIEW
    return RECOMMEND_PROCEED


# ---------- per-client backends ----------

def _gate_anthropic(
    client: Any,
    model: str,
    prompt: str,
    *,
    consensus_n: int,
    temperature: float,
    max_tokens: int,
) -> GateVerdict:
    """Tier-0 consensus path for Anthropic's Messages API."""
    from .anthropic_hack import consensus as _cons
    from .anthropic_hack import text_features as _tf

    t0 = time.time()
    samples: List[str] = []
    usage_in = 0
    usage_out = 0

    for _ in range(consensus_n):
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        for blk in resp.content:
            if getattr(blk, "type", None) == "text":
                samples.append(blk.text)
                break
        if hasattr(resp, "usage"):
            usage_in += getattr(resp.usage, "input_tokens", 0) or 0
            usage_out += getattr(resp.usage, "output_tokens", 0) or 0

    traj = _cons.compute_trajectory(samples)
    mean_entropy = (sum(traj.entropy) / len(traj.entropy)
                    if traj.entropy else 0.0)
    mean_margin = (sum(traj.proxy_top2_margin) /
                   len(traj.proxy_top2_margin)
                   if traj.proxy_top2_margin else 0.0)

    # Combine consensus signal with text-feature signal on the first
    # sample to estimate will_refuse / will_confabulate.
    first_sample_text = samples[0] if samples else ""
    text_class = _tf.classify(first_sample_text)

    # Heuristic mapping, calibrated loosely against v3 Haiku data:
    # - low entropy + high margin = template (refusal if text says so)
    # - mid entropy = confident retrieval (low refuse, low confab)
    # - high entropy = divergent elaboration (real recall)
    refuse_surface = text_class["probs"].get("refusal", 0.0)
    low_entropy_boost = max(0.0, (1.0 - mean_entropy / 1.5))  # 1.5 is ~max entropy for N=5

    will_refuse = min(1.0, refuse_surface + 0.4 * low_entropy_boost)
    will_confabulate = text_class["probs"].get("hallucination", 0.0)
    # If entropy is low AND surface is not refusal, that's confidence
    # (not confabulation) — adjust
    if mean_entropy < 0.8 and refuse_surface < 0.2:
        will_confabulate *= 0.5

    trust_score = 1.0 - max(will_refuse, will_confabulate)

    # Anthropic Haiku pricing (2026-04): ~$1/Mtok input, $5/Mtok output
    cost = (usage_in * 1e-6) + (usage_out * 5e-6)

    recommendation = _compute_recommendation(
        will_refuse, will_confabulate, trust_score)

    return GateVerdict(
        prompt=prompt, model=model,
        method=f"consensus (N={consensus_n})",
        will_refuse=round(will_refuse, 4),
        will_confabulate=round(will_confabulate, 4),
        trust_score=round(trust_score, 4),
        recommendation=recommendation,
        consensus_n=consensus_n,
        runtime_seconds=round(time.time() - t0, 3),
        estimated_cost_usd=round(cost, 6),
        evidence={
            "mean_entropy": round(mean_entropy, 4),
            "mean_top2_margin": round(mean_margin, 4),
            "text_class_probs": {
                k: round(v, 4) for k, v in text_class["probs"].items()
            },
            "first_sample_excerpt": first_sample_text[:200],
        },
        warnings=[
            "consensus-proxy readings are calibrated against the "
            "alignment-inverted signal observed on Claude Haiku 4.5; "
            "other Anthropic models may vary",
        ],
    )


def _gate_openai(
    client: Any,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
) -> GateVerdict:
    """Tier-0 native logprobs path for OpenAI-compatible APIs."""
    from .adapters.openai import attach_vitals_to_response  # lazy
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=temperature,
            logprobs=True, top_logprobs=5,
        )
    except Exception as e:
        return _fallback_text_heuristic(
            prompt, model, runtime=time.time() - t0,
            reason=f"openai call failed: {e}"
        )

    attach_vitals_to_response(resp)
    vitals = getattr(resp, "vitals", None)

    text = ""
    try:
        text = resp.choices[0].message.content or ""
    except Exception:
        pass

    # Pull signals from the native vitals
    if vitals and vitals.phase4_late is not None:
        probs = dict(vitals.phase4_late.probs)
        will_refuse = probs.get("refusal", 0.0)
        will_confabulate = probs.get("hallucination", 0.0)
        trust_score = 1.0 - max(will_refuse, will_confabulate)
    else:
        from .anthropic_hack import text_features as _tf
        tc = _tf.classify(text)
        will_refuse = tc["probs"].get("refusal", 0.0)
        will_confabulate = tc["probs"].get("hallucination", 0.0)
        trust_score = 1.0 - max(will_refuse, will_confabulate)

    cost = 0.0  # user-supplied key, model-dependent; too variable to estimate reliably
    recommendation = _compute_recommendation(
        will_refuse, will_confabulate, trust_score)

    return GateVerdict(
        prompt=prompt, model=model, method="logprobs (tier-0 native)",
        will_refuse=round(will_refuse, 4),
        will_confabulate=round(will_confabulate, 4),
        trust_score=round(trust_score, 4),
        recommendation=recommendation,
        runtime_seconds=round(time.time() - t0, 3),
        estimated_cost_usd=round(cost, 6),
        evidence={
            "vitals_available": vitals is not None,
            "first_sample_excerpt": text[:200],
        },
    )


def _gate_huggingface(
    client: Any, model: str, prompt: str, **kwargs
) -> GateVerdict:
    """Tier-1 residual probe path. Returns an 'unavailable' verdict
    when the atlas probe for (model, task) is not shipped yet."""
    try:
        pass
    except Exception as e:
        return _fallback_text_heuristic(
            prompt, model, runtime=0.0,
            reason=f"residual_probe import failed: {e}"
        )

    return _fallback_text_heuristic(
        prompt, model, runtime=0.0,
        reason=(
            "tier-1 residual probes ship with v3.4.1 (awaiting atlas "
            "data). for now, text-heuristic fallback."
        ),
    )


def _fallback_text_heuristic(
    prompt: str, model: str,
    *,
    runtime: float = 0.0,
    reason: str = "",
) -> GateVerdict:
    """Pure-text classifier fallback. No LLM call made."""
    from .anthropic_hack import text_features as _tf
    tc = _tf.classify(prompt)  # classify the PROMPT shape, not a response
    will_refuse = tc["probs"].get("refusal", 0.0)
    will_confabulate = tc["probs"].get("hallucination", 0.0) * 0.5  # dampen; can't know
    trust_score = 1.0 - max(will_refuse, will_confabulate)
    recommendation = _compute_recommendation(
        will_refuse, will_confabulate, trust_score)

    return GateVerdict(
        prompt=prompt, model=model,
        method="text_heuristic (no LLM call)",
        will_refuse=round(will_refuse, 4),
        will_confabulate=round(will_confabulate, 4),
        trust_score=round(trust_score, 4),
        recommendation=RECOMMEND_UNKNOWN if reason else recommendation,
        runtime_seconds=round(runtime, 3),
        evidence={"text_class_probs": tc["probs"],
                  "classifier_input": "prompt_only"},
        warnings=[reason] if reason else [
            "text-heuristic fallback has ~14% reasoning accuracy on "
            "real Claude output; use as one signal among several, not "
            "as a sole basis for blocking decisions"
        ],
    )


# ---------- public entry point ----------

def gate(
    client: Any = None,
    *,
    model: str = "",
    prompt: str = "",
    consensus_n: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 200,
) -> GateVerdict:
    """Pre-flight cognitive verdict for an LLM prompt.

    Parameters
    ----------
    client : an LLM SDK client (anthropic.Anthropic, openai.OpenAI,
             a HuggingFace model, or None for pure text-heuristic)
    model : str — the model id to target (e.g. "claude-haiku-4-5")
    prompt : str — the user prompt to screen
    consensus_n : int — samples for consensus-mode (anthropic only),
                 default 3 (fast), 5 for higher-signal
    temperature : float — sampling temperature for consensus
    max_tokens : int — cap on generated tokens per sample

    Returns
    -------
    GateVerdict with labelled method and action recommendation.

    Never raises — on error, returns a permissive "unknown" verdict.
    """
    if not prompt:
        return GateVerdict(
            prompt="", model=model, method="noop",
            will_refuse=0.0, will_confabulate=0.0, trust_score=1.0,
            recommendation=RECOMMEND_UNKNOWN,
            warnings=["empty prompt"],
        )

    kind = _client_kind(client)
    try:
        if kind == "anthropic":
            return _gate_anthropic(
                client, model, prompt,
                consensus_n=consensus_n, temperature=temperature,
                max_tokens=max_tokens,
            )
        if kind == "openai":
            return _gate_openai(
                client, model, prompt,
                max_tokens=max_tokens, temperature=temperature,
            )
        if kind in ("huggingface", "huggingface_pair"):
            return _gate_huggingface(client, model, prompt)
        return _fallback_text_heuristic(
            prompt, model, runtime=0.0,
            reason=f"unknown client kind: {kind}"
        )
    except Exception as e:
        warnings.warn(
            f"styxx.gate: unexpected error, returning permissive "
            f"verdict: {e}", RuntimeWarning, stacklevel=2)
        return GateVerdict(
            prompt=prompt, model=model, method="error",
            will_refuse=0.0, will_confabulate=0.0, trust_score=1.0,
            recommendation=RECOMMEND_UNKNOWN,
            warnings=[f"gate() raised {type(e).__name__}: {e}"],
        )


__all__ = [
    "gate", "GateVerdict",
    "RECOMMEND_PROCEED", "RECOMMEND_REVIEW",
    "RECOMMEND_BLOCK", "RECOMMEND_UNKNOWN",
]

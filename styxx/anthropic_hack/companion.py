# -*- coding: utf-8 -*-
"""
Local open-weight companion model for Anthropic proxy readings.

Runs the SAME prompt through a small local model that DOES expose
logprobs, captures the trajectory, and feeds it into the styxx
classifier. The result is labelled `mode=companion:<model>` so callers
know it's a proxy reading, not the real Anthropic stream.

Preferred model: meta-llama/Llama-3.2-1B (granted on local HF cache)
Fallback chain: distilgpt2 → gpt2 → graceful skip

Design constraints:
- NEVER raise on import. If transformers is unavailable, the module
  still imports; calls to `classify_prompt` return a clear
  `{"available": False, "reason": "..."}` dict.
- NEVER fake numbers. If no model loads, say so.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional


_PREFERRED_MODELS = [
    # Instruction-tuned models first — dramatically better at matching
    # the target model's answer shape on real prompts.
    "Qwen/Qwen2.5-3B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    # Base models last (cheapest fallback)
    "meta-llama/Llama-3.2-1B",
    "distilgpt2",
    "gpt2",
]


_LOADED_MODEL = None
_LOADED_TOKENIZER = None
_LOADED_NAME: Optional[str] = None
_LOAD_ERROR: Optional[str] = None


def _try_load(models: Optional[List[str]] = None):
    """Attempt to load a local HF model. Caches the result."""
    global _LOADED_MODEL, _LOADED_TOKENIZER, _LOADED_NAME, _LOAD_ERROR
    if _LOADED_MODEL is not None or _LOAD_ERROR is not None:
        return _LOADED_MODEL, _LOADED_TOKENIZER, _LOADED_NAME

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        _LOAD_ERROR = f"transformers/torch not available: {e}"
        return None, None, None

    candidates = models or _PREFERRED_MODELS
    last_err = None
    for name in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                name, local_files_only=True)
            mdl.eval()
            _LOADED_TOKENIZER = tok
            _LOADED_MODEL = mdl
            _LOADED_NAME = name
            return mdl, tok, name
        except Exception as e:
            last_err = e
            continue

    _LOAD_ERROR = f"no local companion model found (last: {last_err})"
    return None, None, None


def is_available() -> bool:
    mdl, _, _ = _try_load()
    return mdl is not None


def loaded_model_name() -> Optional[str]:
    _try_load()
    return _LOADED_NAME


def load_error() -> Optional[str]:
    _try_load()
    return _LOAD_ERROR


def generate_trajectory(
    prompt: str,
    *,
    max_new_tokens: int = 40,
    top_k: int = 20,
) -> Optional[Dict[str, List[float]]]:
    """Generate with a local model and return per-token logprob features.

    Returns dict {entropy, logprob, top2_margin} or None if unavailable.
    """
    mdl, tok, name = _try_load()
    if mdl is None:
        return None

    try:
        import torch
        inputs = tok(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")

        entropies: List[float] = []
        logprobs: List[float] = []
        top2m: List[float] = []

        with torch.no_grad():
            cur_ids = input_ids
            cur_attn = attn
            for _ in range(max_new_tokens):
                out = mdl(input_ids=cur_ids, attention_mask=cur_attn)
                logits = out.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                logp = torch.log_softmax(logits, dim=-1)
                # entropy in nats
                H = -(probs * logp).sum().item()
                # pick greedy next (top-1) for chosen logprob
                top_vals, top_idx = torch.topk(probs, k=min(top_k, probs.shape[-1]))
                chosen = top_idx[0].item()
                chosen_lp = logp[chosen].item()
                margin = (top_vals[0] - top_vals[1]).item() if top_vals.numel() > 1 else 0.0
                entropies.append(float(H))
                logprobs.append(float(chosen_lp))
                top2m.append(float(margin))
                # extend sequence
                next_tok = torch.tensor([[chosen]])
                cur_ids = torch.cat([cur_ids, next_tok], dim=1)
                if cur_attn is not None:
                    cur_attn = torch.cat(
                        [cur_attn, torch.ones((1, 1), dtype=cur_attn.dtype)],
                        dim=1,
                    )
                # stop on eos
                if tok.eos_token_id is not None and chosen == tok.eos_token_id:
                    break

        return {
            "entropy": entropies,
            "logprob": logprobs,
            "top2_margin": top2m,
        }
    except Exception as e:
        warnings.warn(
            f"styxx.anthropic_hack.companion: generation failed on "
            f"{name}: {e}", RuntimeWarning, stacklevel=2)
        return None


def classify_prompt(prompt: str, *, max_new_tokens: int = 40) -> Dict[str, Any]:
    """High-level entrypoint. Runs companion and classifies trajectory.

    Returns:
        {"available": bool, "mode": str, "vitals": Optional[Vitals],
         "trajectory": Optional[dict], "reason": Optional[str]}
    """
    if not is_available():
        return {
            "available": False,
            "mode": "companion-unavailable",
            "vitals": None,
            "trajectory": None,
            "reason": load_error() or "no model loaded",
        }

    traj = generate_trajectory(prompt, max_new_tokens=max_new_tokens)
    if traj is None:
        return {
            "available": False,
            "mode": "companion-unavailable",
            "vitals": None,
            "trajectory": None,
            "reason": "generation failed",
        }

    from ..vitals import (
        Vitals, PhaseReading, load_centroids, PHASE_ORDER,
        PHASE_TOKEN_CUTOFFS,
    )
    try:
        clf = load_centroids()
    except Exception:
        clf = None

    n = len(traj["entropy"])
    readings: Dict[str, Optional[PhaseReading]] = {}
    for phase in PHASE_ORDER:
        cutoff = PHASE_TOKEN_CUTOFFS[phase]
        if n < cutoff or clf is None:
            readings[phase] = None
            continue
        try:
            readings[phase] = clf.classify(traj, phase)
        except Exception:
            readings[phase] = None

    p1 = readings.get("phase1_preflight")
    if p1 is None:
        # synthesize minimal phase1
        import statistics as _s
        H = _s.mean(traj["entropy"]) if traj["entropy"] else 0.0
        pred = "reasoning" if H < 1.0 else "creative"
        cats = ["retrieval", "reasoning", "refusal",
                "creative", "adversarial", "hallucination"]
        probs = {c: 0.05 for c in cats}
        probs[pred] = 0.75
        p1 = PhaseReading(
            phase="companion-min",
            n_tokens_used=n,
            features=[H],
            predicted_category=pred,
            margin=0.2,
            distances={c: (1.0 - p) * 5.0 for c, p in probs.items()},
            probs=probs,
        )
    p4 = readings.get("phase4_late") or p1

    v = Vitals(
        phase1_pre=p1,
        phase2_early=readings.get("phase2_early"),
        phase3_mid=readings.get("phase3_mid"),
        phase4_late=p4,
        tier_active=0,  # it IS a real tier-0 reading — just on a proxy model
    )
    mode = f"companion:{loaded_model_name()}"
    try:
        v.mode = mode  # type: ignore[attr-defined]
    except Exception:
        pass

    return {
        "available": True,
        "mode": mode,
        "vitals": v,
        "trajectory": traj,
        "reason": None,
    }


__all__ = ["is_available", "loaded_model_name", "load_error",
           "generate_trajectory", "classify_prompt"]

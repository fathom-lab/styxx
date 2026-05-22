"""I_fd — true forced-decoding of draft text under open-weight scorer.

Tokenize (prefix + draft), forward pass on local Llama-3.2-1B-Instruct (or 3B),
read per-token surprisal of the draft tokens given the prefix context.

This is what the styxx logprob-trajectory paper measured offline; we apply it
at send-time. Cross-architecture caveat: Llama scores Claude's text, not its
own. Reported transparently.
"""
from __future__ import annotations

import math
import time
from typing import Any

_MODEL_CACHE: dict[str, Any] = {}


def _load(model_id: str):
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    _MODEL_CACHE[model_id] = (tok, model, torch)
    return tok, model, torch


def _slope(xs: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    idx = list(range(n))
    mx = sum(idx) / n
    my = sum(xs) / n
    num = sum((idx[i] - mx) * (xs[i] - my) for i in range(n))
    den = sum((idx[i] - mx) ** 2 for i in range(n))
    return num / den if den else None


def forced_decode_score(
    prefix: str,
    draft_text: str,
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
) -> dict[str, Any]:
    """Score draft_text under (prefix + draft) context. Returns per-token surprisal + summary."""
    tok, model, torch = _load(model_id)
    t0 = time.time()
    prefix_ids = tok.encode(prefix, return_tensors="pt", add_special_tokens=True).to("cuda")
    target_ids = tok.encode(draft_text, return_tensors="pt", add_special_tokens=False).to("cuda")
    full_ids = torch.cat([prefix_ids, target_ids], dim=1)
    n_prefix = prefix_ids.shape[1]
    n_target = target_ids.shape[1]

    with torch.no_grad():
        out = model(full_ids)
    logits = out.logits[0]
    log_probs_full = torch.log_softmax(logits, dim=-1)

    per_token = []
    Hs = []
    lps = []
    for i in range(n_target):
        pos = n_prefix - 1 + i
        tid = full_ids[0, n_prefix + i].item()
        lp = log_probs_full[pos, tid].item()
        dist = log_probs_full[pos]
        H = -(dist.exp() * dist).sum().item()
        Hs.append(H)
        lps.append(lp)
        per_token.append({"token_id": tid, "logprob": lp, "entropy_nats": H})

    elapsed = time.time() - t0
    return {
        "model": model_id,
        "n_target_tokens": n_target,
        "mean_logprob": sum(lps) / len(lps) if lps else float("nan"),
        "mean_entropy_nats": sum(Hs) / len(Hs) if Hs else float("nan"),
        "entropy_slope": _slope(Hs),
        "elapsed_s": elapsed,
        "per_token": per_token,
    }

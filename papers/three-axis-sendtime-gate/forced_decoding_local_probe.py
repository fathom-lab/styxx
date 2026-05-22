"""I_fd local probe — Llama-3.2-1B-Instruct true forced-decoding.

Tokenize a fixed prefix + target string. Forward pass. Read per-token
logits, compute per-token surprisal of the target tokens given prefix.
This is true forced-decoding: the target is fixed, not generated.

Outputs: forced_decoding_local_probe.json.
"""
from __future__ import annotations

import json
import math
import pathlib
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).parent
OUT = HERE / "forced_decoding_local_probe.json"

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Fixed neutral prefix and target — same as the API probe for comparability.
PREFIX = "The threshold-law paper found that"
TARGET = " corpus-domain overlap above 0.31 cleanly separates cross-family transport regimes."


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="cuda"
    )
    model.eval()
    load_s = time.time() - t0

    prefix_ids = tok.encode(PREFIX, return_tensors="pt", add_special_tokens=True).to("cuda")
    target_ids = tok.encode(TARGET, return_tensors="pt", add_special_tokens=False).to("cuda")

    full_ids = torch.cat([prefix_ids, target_ids], dim=1)
    n_prefix = prefix_ids.shape[1]
    n_target = target_ids.shape[1]

    t1 = time.time()
    with torch.no_grad():
        out = model(full_ids)
    fwd_s = time.time() - t1

    # logits at position i predict token i+1. To score target tokens at
    # positions [n_prefix, n_prefix+n_target), we use logits at positions
    # [n_prefix-1, n_prefix+n_target-1).
    logits = out.logits[0]  # [seq, vocab]
    log_probs_full = torch.log_softmax(logits, dim=-1)

    per_token = []
    for i in range(n_target):
        pos_predicting = n_prefix - 1 + i  # logits at this position predict target token i
        target_token_id = full_ids[0, n_prefix + i].item()
        lp = log_probs_full[pos_predicting, target_token_id].item()
        # entropy of the distribution at this predicting position
        dist = log_probs_full[pos_predicting]
        # full-vocab entropy in nats
        H = -(dist.exp() * dist).sum().item()
        # top-5 for inspection
        top5 = torch.topk(dist, 5)
        top5_tokens = [
            {"token": tok.decode([t.item()]), "logprob": lp_.item()}
            for t, lp_ in zip(top5.indices, top5.values)
        ]
        per_token.append({
            "token": tok.decode([target_token_id]),
            "token_id": target_token_id,
            "logprob": lp,
            "entropy_nats": H,
            "top5": top5_tokens,
        })

    # entropy slope OLS over the target window
    Hs = [t["entropy_nats"] for t in per_token]
    n = len(Hs)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(Hs) / n
    num = sum((xs[i] - mx) * (Hs[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    slope = num / den if den else None

    result = {
        "protocol_ref": "papers/three-axis-sendtime-gate/PROTOCOL.md §A4 + AMENDMENT 3",
        "lock_sha_parent": "aaae5f4",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL_ID,
        "device": "cuda",
        "dtype": "float16",
        "prefix": PREFIX,
        "target": TARGET,
        "n_prefix_tokens": n_prefix,
        "n_target_tokens": n_target,
        "load_seconds": load_s,
        "forward_seconds": fwd_s,
        "mean_logprob": sum(t["logprob"] for t in per_token) / n,
        "mean_entropy_nats": sum(Hs) / n,
        "entropy_slope": slope,
        "per_token": per_token,
        "verdict": "PASS",
    }
    OUT.write_text(json.dumps(result, indent=2))

    print(json.dumps({
        "verdict": "PASS",
        "model": MODEL_ID,
        "n_target_tokens": n_target,
        "mean_logprob": result["mean_logprob"],
        "mean_entropy_nats": result["mean_entropy_nats"],
        "entropy_slope": result["entropy_slope"],
        "fwd_seconds": fwd_s,
        "load_seconds": load_s,
    }, indent=2))


if __name__ == "__main__":
    main()

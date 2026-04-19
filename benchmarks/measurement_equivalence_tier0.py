# -*- coding: utf-8 -*-
"""
benchmarks/measurement_equivalence_tier0.py

Tier-0 consensus-proxy runner for the measurement-equivalence protocol.

Takes a shared prompt JSONL, runs N samples per prompt at T>0 through a
local HuggingFace instruct model, and emits per-prompt consensus
trajectory summary statistics matching the protocol spec at
docs/protocols/consensus-proxy-measurement-equivalence-v0.md.

Spec version: v0 (2026-04-19).

Usage:
  python benchmarks/measurement_equivalence_tier0.py \
      --prompts benchmarks/equivalence_prompts_v0.jsonl \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --n 5 --temp 0.7 --seed 42 \
      --out benchmarks/equivalence_tier0_styxx_YYYYMMDD.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_prompts(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _slope(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_i = (n - 1) / 2.0
    mean_x = sum(xs) / n
    num = sum((i - mean_i) * (xs[i] - mean_x) for i in range(n))
    den = sum((i - mean_i) ** 2 for i in range(n)) or 1e-9
    return num / den


def _curvature(xs: List[float]) -> float:
    if len(xs) < 3:
        return 0.0
    d2 = [abs(xs[i + 1] - 2 * xs[i] + xs[i - 1])
          for i in range(1, len(xs) - 1)]
    return sum(d2) / len(d2) if d2 else 0.0


def _volatility(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    d1 = [abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    return sum(d1) / len(d1) if d1 else 0.0


def compute_consensus_stats(sample_token_ids: List[List[int]]) -> Dict:
    """Given N token-id sequences (same tokenizer), compute per-position
    empirical stats and aggregate summary."""
    max_len = max((len(s) for s in sample_token_ids), default=0)
    if max_len == 0:
        return {
            "n_tokens_used": 0, "first_divergence": -1,
            "mean_entropy": 0.0, "max_entropy": 0.0,
            "entropy_slope": 0.0, "entropy_curvature": 0.0,
            "entropy_volatility": 0.0,
            "mean_top2_margin": 0.0, "mean_logprob": 0.0,
            "logprob_slope": 0.0, "top2_slope": 0.0,
        }

    entropies: List[float] = []
    logprobs: List[float] = []
    margins: List[float] = []
    first_divergence = -1

    for i in range(max_len):
        col = [s[i] for s in sample_token_ids if i < len(s)]
        if not col:
            break
        counts = Counter(col)
        total = sum(counts.values())
        sorted_counts = counts.most_common()
        modal = sorted_counts[0][1]
        runner = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
        p_mode = modal / total
        # Shannon entropy of empirical distribution
        H = -sum((v / total) * math.log(v / total) for v in counts.values())
        entropies.append(H)
        logprobs.append(math.log(max(p_mode, 1e-9)))
        margins.append((modal - runner) / total)
        if first_divergence == -1 and modal < total:
            first_divergence = i

    return {
        "n_tokens_used": max_len,
        "first_divergence": first_divergence,
        "mean_entropy": stats.mean(entropies) if entropies else 0.0,
        "max_entropy": max(entropies) if entropies else 0.0,
        "entropy_slope": _slope(entropies),
        "entropy_curvature": _curvature(entropies),
        "entropy_volatility": _volatility(entropies),
        "mean_top2_margin": stats.mean(margins) if margins else 0.0,
        "mean_logprob": stats.mean(logprobs) if logprobs else 0.0,
        "logprob_slope": _slope(logprobs),
        "top2_slope": _slope(margins),
    }


def run(args: argparse.Namespace) -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        print(f"FATAL: torch/transformers unavailable: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=False)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=(torch.bfloat16 if args.bfloat16 else torch.float32),
        local_files_only=False,
    )
    mdl.eval()
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    mdl.to(device)
    print(f"loaded on {device} dtype={mdl.dtype}")

    prompts = load_prompts(Path(args.prompts))
    print(f"loaded {len(prompts)} prompts from {args.prompts}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fp_out = open(out_path, "w", encoding="utf-8")

    t0 = time.time()
    for idx, row in enumerate(prompts):
        prompt_id = row["id"]
        prompt = row["prompt"]

        # Apply chat template — exact same as darkflobi will use
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        sample_ids_batch: List[List[int]] = []
        t_start = time.time()

        for sample_i in range(args.n):
            # Reproducible seed per (prompt, sample)
            seed = (hash(prompt_id) & 0xFFFFFFFF) ^ sample_i ^ args.seed
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)

            with torch.no_grad():
                out = mdl.generate(
                    inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temp,
                    top_p=1.0,
                    pad_token_id=tok.eos_token_id,
                )
            # Slice the newly generated tokens
            new_tokens = out[0, inputs.shape[1]:].tolist()
            sample_ids_batch.append(new_tokens)

        consensus_stats = compute_consensus_stats(sample_ids_batch)
        avg_words = stats.mean(
            len(tok.decode(s, skip_special_tokens=True).split())
            for s in sample_ids_batch
        )

        record = {
            "id": prompt_id,
            "model": args.model,
            "n_samples": args.n,
            "temperature": args.temp,
            "max_new_tokens": args.max_new_tokens,
            "seed_base": args.seed,
            "runtime_seconds": round(time.time() - t_start, 2),
            "mean_response_length_words": round(avg_words, 2),
            **{k: (round(v, 6) if isinstance(v, float) else v)
               for k, v in consensus_stats.items()},
        }
        fp_out.write(json.dumps(record) + "\n")
        fp_out.flush()

        if (idx + 1) % 5 == 0 or idx + 1 == len(prompts):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(prompts) - idx - 1) / max(rate, 1e-9)
            print(f"  {idx+1}/{len(prompts)} id={prompt_id} "
                  f"H={consensus_stats['mean_entropy']:.3f} "
                  f"m2={consensus_stats['mean_top2_margin']:.3f} "
                  f"[{elapsed:.0f}s elapsed, {eta:.0f}s ETA]")

    fp_out.close()
    print(f"\nwrote {out_path}")
    print(f"total runtime: {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True,
                    help="path to shared prompts JSONL")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n", type=int, default=5, help="samples per prompt")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new-tokens", type=int, default=100,
                    dest="max_new_tokens")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--bfloat16", action="store_true",
                    help="use bfloat16 on CUDA (matches darkflobi's "
                         "default if she uses bf16)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

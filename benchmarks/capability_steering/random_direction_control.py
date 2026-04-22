# -*- coding: utf-8 -*-
"""
benchmarks/capability_steering/random_direction_control.py

Sanity control for the truthfulness amplification finding.

Runs the same TruthfulQA alpha-sweep using a RANDOM unit direction
instead of the trained truthfulness direction. If the random direction
produces accuracy changes comparable to the truthfulness direction,
the original finding is confounded by a steering artifact (any
direction moves accuracy). If the random direction produces no
accuracy change while truthfulness does, the finding is real.

This is the single most important control for the amplification
claim.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--probe_layer", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for split choice + random-direction generation")
    ap.add_argument("--n_random_dirs", type=int, default=3,
                    help="how many random directions to average over")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from truthfulness_amplify import (
        build_splits, run_test_battery,
    )

    _, _, test_idx = build_splits(80, args.n_test, args.seed)
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

    print(f"[1/4] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    hidden = mdl.config.hidden_size

    print(f"[2/4] baseline (no steering) on n={len(test_idx)} ...")
    t0 = time.time()
    b_c, b_t = run_test_battery(mdl, tok, ds, test_idx, device)
    baseline_acc = b_c / b_t
    print(f"  baseline: {b_c}/{b_t} = {baseline_acc:.3f} [{time.time()-t0:.0f}s]")

    # Generate N random unit directions, sweep each
    print(f"[3/4] random-direction sweep: {args.n_random_dirs} directions "
          f"\u00d7 {len(args.alphas)} alphas ...")
    rng = torch.Generator()
    rng.manual_seed(args.seed + 1)

    all_random_results = []
    for dir_i in range(args.n_random_dirs):
        random_dir = torch.randn(hidden, generator=rng, dtype=torch.float32)
        random_dir = random_dir / random_dir.norm().clamp(min=1e-9)
        dir_results = []
        for alpha in args.alphas:
            t0 = time.time()
            c, t_ = run_test_battery(
                mdl, tok, ds, test_idx, device,
                direction=random_dir, alpha=alpha, layer=args.probe_layer,
            )
            acc = c / t_
            d = acc - baseline_acc
            print(f"  dir#{dir_i} alpha={alpha:>4.1f}  acc={c}/{t_}={acc:.3f}  "
                  f"delta={d:+.3f}  [{time.time()-t0:.0f}s]")
            dir_results.append({
                "alpha": alpha, "correct": c, "total": t_,
                "accuracy": acc, "delta_vs_baseline": d,
            })
        all_random_results.append(dir_results)

    # Per-alpha mean delta across random directions
    print(f"[4/4] per-alpha mean delta across {args.n_random_dirs} "
          f"random directions:")
    per_alpha_deltas = []
    for i, alpha in enumerate(args.alphas):
        deltas = [all_random_results[d][i]["delta_vs_baseline"]
                  for d in range(args.n_random_dirs)]
        mean_d = sum(deltas) / len(deltas)
        std_d = (sum((x - mean_d) ** 2 for x in deltas) / len(deltas)) ** 0.5
        print(f"  alpha={alpha:>4.1f}  mean_delta={mean_d:+.3f}  "
              f"std={std_d:.3f}  individual={deltas}")
        per_alpha_deltas.append({
            "alpha": alpha, "mean_delta": mean_d, "std": std_d,
            "individual": deltas,
        })

    out = {
        "model": args.model,
        "baseline_accuracy": baseline_acc,
        "n_test": len(test_idx),
        "n_random_dirs": args.n_random_dirs,
        "probe_layer": args.probe_layer,
        "per_random_dir": all_random_results,
        "per_alpha_summary": per_alpha_deltas,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "random_direction_control.json"
    out_fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_fp}")


if __name__ == "__main__":
    main()

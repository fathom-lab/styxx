# -*- coding: utf-8 -*-
"""
v5 random-direction control.

If multi-layer steering with our trained truthfulness direction gives
+7pp on TruthfulQA, the critical question is: does ANY random
multi-layer direction give that same boost? If yes, the finding is a
steering artifact. If no, the trained direction is the real signal.
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--n_random_dirs", type=int, default=3)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from truthfulness_amplify_v5 import (
        _build_splits, run_test, _get_decoder_layers,
    )
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    _, _, test_idx = _build_splits(200, args.n_test, args.seed)

    print(f"[1/3] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)

    hidden = mdl.config.hidden_size
    n_decoder_layers = len(_get_decoder_layers(mdl))

    # Baseline
    print(f"[2/3] baseline on n={len(test_idx)} ...")
    t0 = time.time()
    b_c, b_t = run_test(mdl, tok, ds, test_idx, device)
    baseline_acc = b_c / b_t
    print(f"  baseline: {b_c}/{b_t} = {baseline_acc:.3f} [{time.time()-t0:.0f}s]")

    # For each random direction set: generate one unit direction per
    # decoder layer, then sweep alpha.
    print(f"[3/3] random-direction multi-layer sweep: "
          f"{args.n_random_dirs} random sets x {len(args.alphas)} alphas ...")
    rng = torch.Generator()
    rng.manual_seed(args.seed + 1001)

    all_results = []
    for dir_i in range(args.n_random_dirs):
        dirs = {}
        for lay in range(n_decoder_layers):
            w = torch.randn(hidden, generator=rng, dtype=torch.float32)
            dirs[lay] = w / w.norm().clamp(min=1e-9)

        per_alpha = []
        for alpha in args.alphas:
            t0 = time.time()
            c, t_ = run_test(mdl, tok, ds, test_idx, device,
                             directions_by_layer=dirs, alpha=alpha)
            acc = c / t_
            d = acc - baseline_acc
            print(f"  random#{dir_i}  alpha={alpha:>4.1f}  "
                  f"acc={c}/{t_}={acc:.3f}  delta={d:+.3f}  "
                  f"[{time.time()-t0:.0f}s]")
            per_alpha.append({
                "alpha": alpha, "correct": c, "total": t_,
                "accuracy": acc, "delta": d,
            })
        all_results.append(per_alpha)

    # Per-alpha mean across random dirs
    print("\n=== summary ===")
    for i, alpha in enumerate(args.alphas):
        deltas = [all_results[d][i]["delta"] for d in range(args.n_random_dirs)]
        mean_d = sum(deltas) / len(deltas)
        std_d = (sum((x - mean_d) ** 2 for x in deltas) / len(deltas)) ** 0.5
        print(f"  alpha={alpha:>4.1f}  mean delta={mean_d:+.3f}  "
              f"std={std_d:.3f}  individual={deltas}")

    out = {
        "model": args.model,
        "baseline_accuracy": baseline_acc,
        "n_test": len(test_idx),
        "n_random_dirs": args.n_random_dirs,
        "per_random": all_results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "v5_random_control.json"
    out_fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_fp}")


if __name__ == "__main__":
    main()

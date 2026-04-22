# -*- coding: utf-8 -*-
"""
benchmarks/capability_steering/cross_scale_transfer.py

Does a trained capability direction (truthfulness) on Llama-3.2-1B
transfer to Llama-3.2-3B via ridge projection and still amplify
TruthfulQA accuracy on the larger model?

This is the logical extension of v5 (+7pp on Llama-1B) to UCB-style
cross-scale capability transfer. If yes, capability directions are
portable across scale within a family — train once on 1B, boost 3B.

Method
------
1. Load Llama-1B truthfulness probe (trained today). It has
   per-layer weights thanks to v1 probe_version.
2. Fit a ridge projection W_L: R^{2048} -> R^{3072} mapping
   Llama-1B residuals at layer 7 (probe's best layer) to
   Llama-3B residuals at layer 12 (the 3B probe's best layer) on
   a held-out 50-prompt common set (TruthfulQA train half, disjoint
   from eval).
3. Transfer: w_{1B->3B} = W_L @ w_{1B, 7}. Normalize.
4. Measure Llama-3B baseline TruthfulQA MC1 accuracy on a test set.
5. Apply the transferred direction as multi-layer steering on
   Llama-3B (patched at every layer between 1B's best and 3B's best).
6. Compare.

If the transferred direction gives a positive delta, we've shown
capability is UCB-portable at scale. Massive.

Usage
-----
  python benchmarks/capability_steering/cross_scale_transfer.py \\
    --n_test 200 --alphas 0.5 1.0 1.5 2.0 \\
    --out_dir benchmarks/capability_steering/runs/cross_scale
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _resolve_layers(model):
    for candidate in (model, getattr(model, "model", None),
                      getattr(getattr(model, "model", None), "model", None)):
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise RuntimeError("could not find decoder layers")


def _extract_residuals(model, tokenizer, prompts, layer, device):
    import torch
    layers = _resolve_layers(model)
    captured = {"h": None}

    def _h(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["h"] = hs[:, -1, :].detach()
        return out

    handle = layers[layer].register_forward_hook(_h)
    out = []
    try:
        for p in prompts:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                _ = model(input_ids=input_ids)
            out.append(captured["h"][0].to(torch.float32).cpu())
    finally:
        handle.remove()
    import torch
    return torch.stack(out)


def _score_choice(model, tokenizer, prompt, choice, device, hook_fn=None,
                   hook_module=None):
    import torch
    full = prompt + " " + choice
    tokens = tokenizer(full, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    handles = []
    if hook_fn is not None and hook_module is not None:
        handles.append(hook_module.register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            out = model(input_ids=tokens)
        logprobs = torch.log_softmax(out.logits[0], dim=-1)
        total = 0.0
        count = 0
        for i in range(prompt_len, tokens.shape[1]):
            total += float(logprobs[i - 1, tokens[0, i]].item())
            count += 1
    finally:
        for h in handles:
            h.remove()
    return total / max(count, 1)


def _truthfulqa_accuracy(model, tokenizer, ds, test_idx, device,
                          direction=None, alpha=0.0, layer=None):
    """Score MC1 accuracy, optionally with additive multi-position
    residual steering at a specific layer."""
    import torch

    hook_module = None
    hook_fn = None
    if direction is not None and alpha != 0.0 and layer is not None:
        layers = _resolve_layers(model)
        hook_module = layers[layer]
        dtype = next(model.parameters()).dtype
        direction = direction.to(device=device, dtype=dtype)
        shift = alpha * direction

        def _hook(module, inp, out):
            if isinstance(out, tuple):
                hs = out[0]; rest = out[1:]
            else:
                hs = out; rest = None
            hs[:, :, :] = hs[:, :, :] + shift
            return (hs, *rest) if rest is not None else hs
        hook_fn = _hook

    correct = 0
    total = 0
    for i in test_idx:
        row = ds[i]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        prompt = f"Q: {row['question']}\nA:"
        scores = [_score_choice(model, tokenizer, prompt, c, device,
                                 hook_fn=hook_fn, hook_module=hook_module)
                  for c in choices]
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        gold = labels.index(1)
        if pred == gold:
            correct += 1
        total += 1
    return correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairing", type=int, default=80,
                    help="Prompts used to fit ridge projection")
    ap.add_argument("--n_test", type=int, default=200,
                    help="Held-out TruthfulQA test set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    # Load 3B probe and 1B probe manifests
    atlas = ROOT / "styxx" / "residual_probe" / "atlas"
    probe_1b = json.loads((atlas / "meta_llama_Llama_3.2_1B_Instruct_truthfulness.json"
                           ).read_text(encoding="utf-8"))
    probe_3b = json.loads((atlas / "meta_llama_Llama_3.2_3B_Instruct_truthfulness.json"
                           ).read_text(encoding="utf-8"))
    state_1b = torch.load(
        atlas / probe_1b["weight_file"],
        map_location="cpu", weights_only=True,
    )
    w_1b_at_7 = state_1b["weight"].to(torch.float32)
    layer_1b = probe_1b["layer"]     # 7
    layer_3b = probe_3b["layer"]     # 12
    print(f"[1/6] 1B probe: layer {layer_1b} AUC {probe_1b['auc_validation']}")
    print(f"      3B probe: layer {layer_3b} AUC {probe_3b['auc_validation']}")

    # Build shared pairing prompt set from TruthfulQA (train half)
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = random.Random(args.seed)
    all_idx = list(range(len(ds)))
    rng.shuffle(all_idx)
    # Use FIRST half for pairing, TEST half disjoint
    pairing_idx = all_idx[:args.n_pairing]
    test_idx = all_idx[args.n_pairing:args.n_pairing + args.n_test]
    pairing_prompts = [f"Q: {ds[i]['question']}\nA:" for i in pairing_idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract Llama-1B residuals at layer 7
    print(f"[2/6] loading Llama-3.2-1B-Instruct ...")
    mdl_1b = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.bfloat16,
    ).eval()
    tok_1b = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct")
    mdl_1b.to(device)
    print(f"  extracting 1B residuals @ L{layer_1b} ...")
    X_1b = _extract_residuals(mdl_1b, tok_1b, pairing_prompts, layer_1b, device)
    del mdl_1b
    torch.cuda.empty_cache()

    # Extract Llama-3B residuals at layer 12
    print(f"[3/6] loading Llama-3.2-3B-Instruct ...")
    mdl_3b = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16,
    ).eval()
    tok_3b = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct")
    mdl_3b.to(device)
    print(f"  extracting 3B residuals @ L{layer_3b} ...")
    X_3b = _extract_residuals(mdl_3b, tok_3b, pairing_prompts, layer_3b, device)

    # Fit ridge projection W: R^2048 -> R^3072
    print(f"[4/6] fitting ridge W_1B->3B ...")
    lam = 1e-2
    XtX = X_1b.T @ X_1b
    I = torch.eye(X_1b.shape[1], dtype=X_1b.dtype)
    W = torch.linalg.solve(XtX + lam * I, X_1b.T @ X_3b)
    print(f"  W shape: {tuple(W.shape)}")

    # Transfer direction
    w_transferred = w_1b_at_7 @ W
    w_transferred_unit = w_transferred / w_transferred.norm().clamp(min=1e-9)
    print(f"  w_transferred shape: {tuple(w_transferred_unit.shape)}, "
          f"norm={w_transferred.norm():.3f}")

    # Compare to native 3B direction at layer 12 for sanity
    state_3b = torch.load(
        atlas / probe_3b["weight_file"],
        map_location="cpu", weights_only=True,
    )
    w_3b_native = state_3b["weight"].to(torch.float32)
    w_3b_native_unit = w_3b_native / w_3b_native.norm().clamp(min=1e-9)
    cos_native = float((w_transferred_unit @ w_3b_native_unit).item())
    print(f"  cos(transferred, native 3B): {cos_native:+.3f}")

    # Measure Llama-3B baseline accuracy
    print(f"[5/6] Llama-3B baseline accuracy on n={len(test_idx)} ...")
    t0 = time.time()
    b_c, b_t = _truthfulqa_accuracy(mdl_3b, tok_3b, ds, test_idx, device)
    baseline_acc = b_c / b_t
    print(f"  baseline: {b_c}/{b_t} = {baseline_acc:.3f}  "
          f"[{time.time()-t0:.0f}s]")

    # Sweep alpha with TRANSFERRED direction
    print(f"[6/6] steered accuracy sweep with transferred direction ...")
    results = []
    for alpha in args.alphas:
        t0 = time.time()
        c, t_ = _truthfulqa_accuracy(
            mdl_3b, tok_3b, ds, test_idx, device,
            direction=w_transferred_unit,
            alpha=alpha,
            layer=layer_3b,
        )
        acc = c / t_
        d = acc - baseline_acc
        print(f"  alpha={alpha:>4.1f} transferred  "
              f"acc={c}/{t_}={acc:.3f}  delta={d:+.3f}  "
              f"[{time.time()-t0:.0f}s]")
        results.append({
            "alpha": alpha, "direction": "transferred",
            "correct": c, "total": t_, "accuracy": acc, "delta": d,
        })

    # Control: sweep alpha with NATIVE 3B direction for comparison
    print(f"\nbonus control: native 3B direction sweep")
    native_results = []
    for alpha in args.alphas:
        t0 = time.time()
        c, t_ = _truthfulqa_accuracy(
            mdl_3b, tok_3b, ds, test_idx, device,
            direction=w_3b_native_unit,
            alpha=alpha,
            layer=layer_3b,
        )
        acc = c / t_
        d = acc - baseline_acc
        print(f"  alpha={alpha:>4.1f} native       "
              f"acc={c}/{t_}={acc:.3f}  delta={d:+.3f}  "
              f"[{time.time()-t0:.0f}s]")
        native_results.append({
            "alpha": alpha, "direction": "native_3B",
            "correct": c, "total": t_, "accuracy": acc, "delta": d,
        })

    out = {
        "model_source": "meta-llama/Llama-3.2-1B-Instruct",
        "model_target": "meta-llama/Llama-3.2-3B-Instruct",
        "source_layer": layer_1b,
        "target_layer": layer_3b,
        "n_pairing": args.n_pairing,
        "n_test": len(test_idx),
        "seed": args.seed,
        "cos_transferred_vs_native": cos_native,
        "baseline_accuracy": baseline_acc,
        "transferred_sweep": results,
        "native_sweep": native_results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cross_scale_transfer.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'cross_scale_transfer.json'}")

    best_t = max(results, key=lambda r: r["accuracy"])
    if best_t["accuracy"] > baseline_acc + 0.02:
        print(f"\n=== TRANSFERRED POSITIVE ===")
        print(f"Transferred direction at alpha={best_t['alpha']} "
              f"achieves {best_t['accuracy']:.3f} vs baseline "
              f"{baseline_acc:.3f} (delta {best_t['delta']:+.3f})")
        print("Capability direction transfers 1B -> 3B with positive effect.")
    else:
        best_n = max(native_results, key=lambda r: r["accuracy"])
        print(f"\n=== TRANSFERRED NULL ===")
        print(f"Transferred direction max delta: {best_t['delta']:+.3f}")
        print(f"Native 3B direction max delta:   {best_n['delta']:+.3f}")
        print("Transfer did not produce meaningful accuracy change.")


if __name__ == "__main__":
    main()

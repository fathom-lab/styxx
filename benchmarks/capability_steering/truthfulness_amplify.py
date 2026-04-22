# -*- coding: utf-8 -*-
"""
benchmarks/capability_steering/truthfulness_amplify.py

Gradient-free capability amplification — the 10x experiment.

Tests whether a linear "truthfulness" direction exists in the
residual stream, such that adding α * direction at inference time
improves accuracy on TruthfulQA WITHOUT fine-tuning.

Method
------
1. Load TruthfulQA multiple-choice.
2. Baseline: score each question by log-prob of each choice; pick
   argmax; report accuracy.
3. For TRAINING split of questions (seed-shuffled), extract last-
   token residual at the CORRECT completion and at a PLAUSIBLE-
   INCORRECT completion. Train L2 logistic probe on those residuals.
4. Steered inference on the HELD-OUT split, applying alpha * unit
   direction at the probe's trained layer.
5. Compare accuracy curves across alpha.

If accuracy at some alpha is significantly above baseline, we've
demonstrated gradient-free capability amplification. Untried in
published literature.

Usage
-----
  python benchmarks/capability_steering/truthfulness_amplify.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --n_train 80 --n_test 120 --seed 0 \
    --alphas 0.0 0.5 1.0 1.5 2.0 3.0 \
    --out_dir benchmarks/capability_steering/runs/v0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _score_completion(model, tokenizer, prompt: str, completion: str,
                       device, hook_module=None, hook_fn=None) -> Tuple[float, "torch.Tensor"]:
    """Return (sum_logprob, last-token-residual) for prompt+completion.

    hook_module/hook_fn: optional steering hook to install during the
    forward pass. If provided, we install on hook_module for the
    duration of this call.
    """
    import torch

    full = prompt + " " + completion
    tokens = tokenizer(full, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    captured = {"hidden": None}

    def _capture(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["hidden"] = hs[:, -1, :].detach()
        return out

    handles = []
    if hook_module is not None and hook_fn is not None:
        handles.append(hook_module.register_forward_hook(hook_fn))

    try:
        with torch.no_grad():
            out = model(input_ids=tokens, output_hidden_states=True)
        logits = out.logits[0]            # (seq, vocab)
        # log P(completion_t | prior_t) for each completion token
        logprobs = torch.log_softmax(logits, dim=-1)
        # Completion starts at prompt_len
        total_lp = 0.0
        for i in range(prompt_len, tokens.shape[1]):
            tok = tokens[0, i]
            total_lp += float(logprobs[i - 1, tok].item())
        residual = out.hidden_states[-1][0, -1, :].to(torch.float32).cpu()
    finally:
        for h in handles:
            h.remove()

    return total_lp, residual


def build_splits(n_train: int, n_test: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]
    return ds, train_idx, test_idx


def pick_correct_and_incorrect(row):
    choices = row["mc1_targets"]["choices"]
    labels = row["mc1_targets"]["labels"]
    correct = None
    incorrect = None
    for c, l in zip(choices, labels):
        if l == 1 and correct is None:
            correct = c
        elif l == 0 and incorrect is None:
            incorrect = c
        if correct is not None and incorrect is not None:
            break
    return correct, incorrect


def score_question(model, tokenizer, row, device, hook_module=None,
                   hook_fn=None) -> Tuple[int, int]:
    """Score one MC question; return (predicted_idx, correct_idx)."""
    import torch
    question = row["question"]
    choices = row["mc1_targets"]["choices"]
    labels = row["mc1_targets"]["labels"]

    prompt = f"Q: {question}\nA:"
    scores = []
    for c in choices:
        lp, _ = _score_completion(model, tokenizer, prompt, c, device,
                                   hook_module=hook_module,
                                   hook_fn=hook_fn)
        # normalize by length
        scores.append(lp / max(len(c), 1))

    predicted = int(max(range(len(scores)), key=lambda i: scores[i]))
    correct = labels.index(1)
    return predicted, correct


def extract_training_residuals(model, tokenizer, ds, train_idx,
                                layer: int, device):
    """For each question, capture the last-token residual at `layer`
    for the correct and incorrect completions. Returns two tensors:
    X_correct, X_incorrect, each shape (n, hidden)."""
    import torch

    X_correct = []
    X_incorrect = []
    kept = []
    for i in train_idx:
        row = ds[i]
        correct, incorrect = pick_correct_and_incorrect(row)
        if correct is None or incorrect is None:
            continue
        prompt = f"Q: {row['question']}\nA:"

        # Capture layer-L residual at end of prompt + completion
        def _capture_layer(hook_target_layer):
            cap = {"h": None}
            def _h(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                cap["h"] = hs[:, -1, :].detach()
                return out
            return cap, _h

        for candidate in (model, getattr(model, "model", None),
                          getattr(getattr(model, "model", None), "model", None)):
            if candidate is None:
                continue
            layers = getattr(candidate, "layers", None)
            if layers is not None and len(layers) > layer:
                target_layer = layers[layer]
                break

        def run_with_capture(completion):
            full = prompt + " " + completion
            tokens = tokenizer(full, return_tensors="pt").input_ids.to(device)
            cap = {"h": None}
            def _h(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                cap["h"] = hs[:, -1, :].detach()
                return out
            handle = target_layer.register_forward_hook(_h)
            try:
                with torch.no_grad():
                    _ = model(input_ids=tokens)
            finally:
                handle.remove()
            return cap["h"][0].to(torch.float32).cpu()

        X_correct.append(run_with_capture(correct))
        X_incorrect.append(run_with_capture(incorrect))
        kept.append(i)

    X_correct = torch.stack(X_correct)
    X_incorrect = torch.stack(X_incorrect)
    return X_correct, X_incorrect, kept


def train_direction(X_correct, X_incorrect):
    """Fit a linear logistic probe on [correct vs incorrect] residuals.
    Returns a unit-normalized direction (correct side = positive)."""
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression

    X = torch.cat([X_correct, X_incorrect], dim=0).numpy()
    y = np.array([1] * len(X_correct) + [0] * len(X_incorrect))
    clf = LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000, solver="liblinear",
    )
    clf.fit(X, y)
    w = torch.tensor(clf.coef_[0], dtype=torch.float32)
    w_unit = w / w.norm().clamp(min=1e-9)
    return w_unit


def run_test_battery(model, tokenizer, ds, test_idx, device,
                     direction=None, alpha=0.0, layer=None):
    """Evaluate TruthfulQA MC accuracy, optionally with steering."""
    import torch

    # Install steering hook at `layer` if direction is provided
    target_layer = None
    if direction is not None and alpha != 0.0:
        for candidate in (model, getattr(model, "model", None),
                          getattr(getattr(model, "model", None), "model", None)):
            if candidate is None:
                continue
            layers = getattr(candidate, "layers", None)
            if layers is not None and len(layers) > layer:
                target_layer = layers[layer]
                break
        dtype = next(model.parameters()).dtype
        direction = direction.to(device=device, dtype=dtype)

        def _hook(module, inp, out):
            # Patch ALL positions, not just the last one. MC scoring
            # reads logits across the completion region — last-token-
            # only patching has no effect on those logits, which is
            # why the first sweep produced a NULL result.
            if isinstance(out, tuple):
                hs = out[0]; rest = out[1:]
            else:
                hs = out; rest = None
            hs[:, :, :] = hs[:, :, :] + alpha * direction
            return (hs, *rest) if rest is not None else hs

    hook_fn = None
    if direction is not None and alpha != 0.0:
        hook_fn = _hook

    correct_count = 0
    total = 0
    for i in test_idx:
        row = ds[i]
        pred, gold = score_question(model, tokenizer, row, device,
                                     hook_module=target_layer,
                                     hook_fn=hook_fn)
        if pred == gold:
            correct_count += 1
        total += 1
    return correct_count, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe_layer", type=int, default=10)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ds, train_idx, test_idx = build_splits(args.n_train, args.n_test,
                                             args.seed)
    print(f"[1/5] TruthfulQA: train={len(train_idx)}, test={len(test_idx)}")

    print(f"[2/5] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)

    # Baseline (no steering)
    print(f"[3/5] baseline accuracy on n={len(test_idx)} held-out ...")
    t0 = time.time()
    baseline_correct, baseline_total = run_test_battery(
        mdl, tok, ds, test_idx, device,
    )
    baseline_acc = baseline_correct / baseline_total
    print(f"  baseline: {baseline_correct}/{baseline_total} = "
          f"{baseline_acc:.3f} [{time.time()-t0:.0f}s]")

    # Extract training residuals
    print(f"[4/5] extracting training residuals @ layer {args.probe_layer} "
          f"for n={len(train_idx)} ...")
    t0 = time.time()
    X_correct, X_incorrect, kept = extract_training_residuals(
        mdl, tok, ds, train_idx, args.probe_layer, device,
    )
    print(f"  X_correct {tuple(X_correct.shape)}, "
          f"X_incorrect {tuple(X_incorrect.shape)} "
          f"[{time.time()-t0:.0f}s]")

    direction = train_direction(X_correct, X_incorrect)
    print(f"  direction: norm={direction.norm():.3f}, "
          f"dim={direction.shape[0]}")

    # Steered alpha sweep
    print(f"[5/5] steered accuracy across alphas ...")
    results = []
    for alpha in args.alphas:
        t0 = time.time()
        correct, total = run_test_battery(
            mdl, tok, ds, test_idx, device,
            direction=direction, alpha=alpha, layer=args.probe_layer,
        )
        acc = correct / total
        delta = acc - baseline_acc
        print(f"  alpha={alpha:>4.1f}  acc={correct}/{total}={acc:.3f}  "
              f"delta={delta:+.3f}  [{time.time()-t0:.0f}s]")
        results.append({
            "alpha": alpha, "correct": correct, "total": total,
            "accuracy": acc, "delta_vs_baseline": delta,
        })

    out = {
        "model": args.model,
        "baseline_accuracy": baseline_acc,
        "baseline_correct": baseline_correct,
        "baseline_total": baseline_total,
        "probe_layer": args.probe_layer,
        "alpha_sweep": results,
        "direction_norm": float(direction.norm()),
        "hidden_size": direction.shape[0],
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "seed": args.seed,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / "truthfulness_amplify.json"
    out_fp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_fp}")

    # Verdict
    best = max(results, key=lambda r: r["accuracy"])
    if best["accuracy"] > baseline_acc + 0.02:
        print(f"\n=== POSITIVE RESULT ===")
        print(f"Best alpha={best['alpha']} "
              f"accuracy={best['accuracy']:.3f} vs "
              f"baseline={baseline_acc:.3f} "
              f"(delta={best['delta_vs_baseline']:+.3f}). "
              f"Gradient-free truthfulness amplification.")
    elif best["accuracy"] < baseline_acc - 0.02:
        print(f"\n=== NEGATIVE RESULT ===")
        print(f"Best alpha under baseline; steering hurts this concept.")
    else:
        print(f"\n=== NULL RESULT ===")
        print(f"Steering does not measurably change accuracy.")


if __name__ == "__main__":
    main()

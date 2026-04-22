# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/train_truthfulness_probe.py

Train a `truthfulness` concept probe on any HF decoder model using
TruthfulQA MC1 contrast pairs. Writes atlas artifacts compatible
with the rest of the Styxx ecosystem (`styxx.residual_probe`,
`styxx.steer`, `styxx.cogvm`).

Contrast signal:
  positive class = residual at END of (prompt + correct answer)
  negative class = residual at END of (prompt + plausible-incorrect)

For each of N train questions, we generate BOTH (correct-completion,
incorrect-completion) pairs so the probe learns a direction that
discriminates a truthful continuation from a plausible-false one.
This is the contrast used in v5 (validated +7pp on Llama-1B test).

The saved probe_version is `v1` so the per-layer weights are stored
alongside the best-AUC weight (enables downstream UCB CCA at
shared layers).

Usage
-----
  python benchmarks/causal_patching/train_truthfulness_probe.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --out_dir styxx/residual_probe/atlas \
    --n_train 80 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def build_pairs(n: int, seed: int) -> List[Tuple[str, str, str]]:
    """Return list of (prompt, correct_completion, incorrect_completion)."""
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    pairs = []
    for i in indices:
        row = ds[i]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct = next((c for c, l in zip(choices, labels) if l == 1), None)
        incorrect = next((c for c, l in zip(choices, labels) if l == 0), None)
        if correct is None or incorrect is None:
            continue
        prompt = f"Q: {row['question']}\nA:"
        pairs.append((prompt, correct, incorrect))
        if len(pairs) >= n:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import LeaveOneOut

    pairs = build_pairs(args.n_train, args.seed)
    print(f"[1/5] TruthfulQA train pairs: {len(pairs)}")

    print(f"[2/5] loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).eval()
    device = args.device if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    n_layers = mdl.config.num_hidden_layers + 1
    hidden = mdl.config.hidden_size
    print(f"  layers={n_layers} hidden={hidden}")

    # Install capture hooks at every decoder layer
    layers_mod = None
    for candidate in (mdl, getattr(mdl, "model", None),
                      getattr(getattr(mdl, "model", None), "model", None)):
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            layers_mod = layers
            break
    if layers_mod is None:
        print("FATAL: could not resolve decoder layers", file=sys.stderr)
        sys.exit(1)

    caps = {i: {"h": None} for i in range(len(layers_mod))}

    def _make_cap(i):
        def _h(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            caps[i]["h"] = hs[:, -1, :].detach()
            return out
        return _h

    cap_handles = [layers_mod[i].register_forward_hook(_make_cap(i))
                   for i in range(len(layers_mod))]

    # Extract paired residuals
    print(f"[3/5] extracting residuals on {2*len(pairs)} completions ...")
    residuals = [[] for _ in range(len(layers_mod) + 1)]
    labels: List[int] = []
    t0 = time.time()

    try:
        for i, (prompt, correct, incorrect) in enumerate(pairs):
            for completion, lbl in ((correct, 1), (incorrect, 0)):
                full = prompt + " " + completion
                tokens = tok(full, return_tensors="pt").input_ids.to(device)
                with torch.no_grad():
                    _ = mdl(input_ids=tokens)
                # index 0 = embedding (we skip), decoder layers map to i+1
                for li in range(len(layers_mod)):
                    h = caps[li]["h"]
                    if h is None:
                        continue
                    residuals[li + 1].append(h[0].to(torch.float32).cpu())
                labels.append(lbl)

            if (i + 1) % 10 == 0:
                dt = time.time() - t0
                eta = (len(pairs) - i - 1) * (dt / (i + 1))
                print(f"  {i+1}/{len(pairs)}  [{dt:.0f}s ETA {eta:.0f}s]")
    finally:
        for h in cap_handles:
            h.remove()

    # Train per-layer probes, compute LOO-AUC
    print(f"[4/5] training per-layer probes (LOO-CV) ...")
    y = np.array(labels)
    per_layer_records = []
    per_layer_weights: Dict[str, List[float]] = {}
    per_layer_bias: Dict[str, float] = {}

    for li in range(1, len(layers_mod) + 1):
        if not residuals[li]:
            continue
        X = torch.stack(residuals[li]).numpy()
        # LOO is O(n) x fits, pair-aware splits are tricky; use std LOO.
        loo = LeaveOneOut()
        preds = np.zeros_like(y, dtype=float)
        for tr, te in loo.split(X):
            clf = LogisticRegression(
                penalty="l2", C=1.0, max_iter=2000, solver="liblinear",
            )
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict_proba(X[te])[:, 1]
        try:
            auc = float(roc_auc_score(y, preds))
        except Exception:
            auc = float("nan")
        per_layer_records.append({"layer": li - 1, "auc_loo": auc})

        clf_full = LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, solver="liblinear",
        )
        clf_full.fit(X, y)
        per_layer_weights[str(li - 1)] = clf_full.coef_[0].tolist()
        per_layer_bias[str(li - 1)] = float(clf_full.intercept_[0])

        print(f"    layer {li-1:2d}  AUC={auc:.3f}")

    valid = [r for r in per_layer_records
             if r["auc_loo"] == r["auc_loo"]]
    best = max(valid, key=lambda r: r["auc_loo"])
    best_layer = best["layer"]
    print(f"\nbest layer: {best_layer} (AUC={best['auc_loo']:.3f})")

    best_weight = torch.tensor(
        per_layer_weights[str(best_layer)], dtype=torch.float32)
    best_bias = per_layer_bias[str(best_layer)]

    # Write atlas artifact
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model.replace("/", "_").replace("-", "_")
    stem = f"{slug}_truthfulness"
    weights_fp = out_dir / f"{stem}.pt"
    torch.save({
        "weight": best_weight,
        "bias": best_bias,
        "weight_per_layer": {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in per_layer_weights.items()
        },
        "bias_per_layer": per_layer_bias,
    }, weights_fp)

    manifest = {
        "probe_version": "v1",
        "atlas_version": "v0",
        "concept": "truthfulness",
        "model": args.model,
        "task": "truthfulness",
        "positive_class": "correct",
        "negative_class": "incorrect",
        "layer": best_layer,
        "total_layers": len(layers_mod) + 1,
        "hidden_size": hidden,
        "training_n": len(labels),
        "training_seed": args.seed,
        "training_prompt_set": f"TruthfulQA_mc1[n={len(pairs)}]@seed={args.seed}",
        "class_balance": [int(sum(labels)), int(len(labels) - sum(labels))],
        "label_mode": "paired_contrast",
        "auc_validation": round(best["auc_loo"], 4),
        "auc_validation_method": "leave-one-out",
        "per_layer_auc": per_layer_records,
        "fitted_on": time.strftime("%Y-%m-%d"),
        "weight_file": weights_fp.name,
    }
    manifest_fp = out_dir / f"{stem}.json"
    manifest_fp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n=== DONE ===")
    print(f"wrote {weights_fp}")
    print(f"wrote {manifest_fp}")


if __name__ == "__main__":
    main()

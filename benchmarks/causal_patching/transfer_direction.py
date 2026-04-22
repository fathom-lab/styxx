# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/transfer_direction.py

Cross-model direction transfer — the true cross-model cognitive
transfer experiment.

Premise
-------
A probe trained on model A's residual stream at layer L_A captures a
concept direction `w_A` in A's hidden_size-dimensional space. Model B
has a different architecture, different parameters, and often a
different hidden_size. Can the concept `w_A` transfer to B without
retraining?

This is the critical cross-model experiment. If YES — concept
directions are portable, and training once per concept is enough.
If NO — every model needs its own atlas.

Method
------
1. Pick a shared set of prompts P (e.g., JBB test half).
2. For each prompt, extract the last-token residual from A at layer
   L_A and from B at layer L_B. Stack into matrices
   X_A ∈ R^{n × d_A} and X_B ∈ R^{n × d_B}.
3. Fit a linear map W: R^{d_A} → R^{d_B} by least squares:
   W = (X_A^T X_A + λI)^{-1} X_A^T X_B  (ridge regression)
4. Transfer: w_{A→B} = W @ w_A (cross-dim projection).
5. Evaluate transfer quality in three ways:
   a. **Readout transfer**: does w_{A→B} score B's residuals
      similarly to a probe trained directly on B?
   b. **Steering transfer**: does using w_{A→B} as a steering
      direction in B produce behavioral changes consistent with
      A's w_A? (α-sweep on B with the transferred direction.)
   c. **Geometric alignment**: cosine(w_{A→B}, w_B_native).

Usage
-----
  python benchmarks/causal_patching/transfer_direction.py \
    --model_a meta-llama/Llama-3.2-1B-Instruct \
    --manifest_a styxx/residual_probe/atlas/meta_llama_Llama_3.2_1B_Instruct_comply_refuse.json \
    --layer_a 10 \
    --model_b meta-llama/Llama-3.2-3B-Instruct \
    --manifest_b styxx/residual_probe/atlas/meta_llama_Llama_3.2_3B_Instruct_comply_refuse.json \
    --layer_b 14 \
    --n_prompts 60 --seed 7 \
    --out_file benchmarks/causal_patching/runs/transfer-1B-to-3B.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract_and_train import build_probe_set  # noqa: E402


def _load_probe_weight(manifest_path: Path, layer: int = None):
    """Load the weight vector from a probe manifest + sibling .pt file,
    at the requested layer (or the manifest's best layer)."""
    import torch

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weight_fp = manifest_path.parent / manifest["weight_file"]
    state = torch.load(weight_fp, map_location="cpu", weights_only=True)

    if layer is not None and "weight_per_layer" in state:
        per_layer = state["weight_per_layer"]
        key = str(layer)
        if key in per_layer:
            return per_layer[key].to(torch.float32), manifest
    return state["weight"].to(torch.float32), manifest


def _extract_residuals(model, tokenizer, prompts: List[str],
                        layer: int, device: str):
    """Extract the last-token residual at the given layer for each prompt.
    Returns a (n, hidden) tensor."""
    import torch

    residuals = []
    for prompt in prompts:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            hs = outputs.hidden_states[layer]
            residuals.append(hs[0, -1, :].to(torch.float32).cpu())
    return torch.stack(residuals)


def fit_projection(X_A, X_B, ridge: float = 1e-2):
    """Least-squares W: R^{d_A} -> R^{d_B}. W has shape (d_A, d_B)."""
    import torch

    XtX = X_A.T @ X_A
    d_A = X_A.shape[1]
    I = torch.eye(d_A, dtype=X_A.dtype)
    W = torch.linalg.solve(XtX + ridge * I, X_A.T @ X_B)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_a", required=True)
    ap.add_argument("--manifest_a", required=True)
    ap.add_argument("--layer_a", type=int, required=True)
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--manifest_b", default=None,
                    help="Optional: if provided, the B-native probe "
                         "direction is loaded for geometric comparison.")
    ap.add_argument("--layer_b", type=int, required=True)
    ap.add_argument("--n_prompts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7,
                    help="Seed for JBB test-half subsampling (disjoint "
                         "from training seeds 0 and 1).")
    ap.add_argument("--dataset", default="jbb")
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # --- Build shared prompt set (from JBB test-half, fresh seed) ---
    n_half = args.n_prompts // 2
    rows = build_probe_set(n_half, n_half, args.seed,
                           dataset=args.dataset, split="test")
    prompts = [r["prompt"] for r in rows]
    kinds = [r["kind"] for r in rows]
    print(f"transfer probe set: {len(prompts)} prompts "
          f"({sum(1 for k in kinds if k=='unsafe')} unsafe, "
          f"{sum(1 for k in kinds if k=='safe')} safe)")

    # --- Load model A, extract residuals at layer_a ---
    print(f"\nloading {args.model_a} ...")
    tok_a = AutoTokenizer.from_pretrained(args.model_a)
    mdl_a = AutoModelForCausalLM.from_pretrained(
        args.model_a,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl_a.to(device)
    print(f"  extracting residuals @ layer {args.layer_a} ...")
    X_A = _extract_residuals(mdl_a, tok_a, prompts, args.layer_a, device)
    print(f"  X_A.shape = {tuple(X_A.shape)}")

    # Free VRAM before loading B.
    del mdl_a
    torch.cuda.empty_cache()

    # --- Load model B, extract residuals at layer_b ---
    print(f"\nloading {args.model_b} ...")
    tok_b = AutoTokenizer.from_pretrained(args.model_b)
    mdl_b = AutoModelForCausalLM.from_pretrained(
        args.model_b,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    ).eval()
    mdl_b.to(device)
    print(f"  extracting residuals @ layer {args.layer_b} ...")
    X_B = _extract_residuals(mdl_b, tok_b, prompts, args.layer_b, device)
    print(f"  X_B.shape = {tuple(X_B.shape)}")

    del mdl_b
    torch.cuda.empty_cache()

    # --- Fit projection ---
    print("\nfitting ridge-regression projection A -> B ...")
    W = fit_projection(X_A, X_B, ridge=1e-2)
    print(f"  W.shape = {tuple(W.shape)}")

    # Quality check: X_A @ W vs X_B
    pred_B = X_A @ W
    resid = (pred_B - X_B)
    frob_resid = float(resid.pow(2).sum().sqrt())
    frob_B = float(X_B.pow(2).sum().sqrt())
    r2_like = 1.0 - (frob_resid ** 2) / (frob_B ** 2)
    print(f"  projection R^2-like = {r2_like:.3f} "
          f"(1.0 = perfect; 0.0 = mean-prediction)")

    # --- Transfer the A-direction to B's space ---
    w_A, man_a = _load_probe_weight(Path(args.manifest_a), args.layer_a)
    w_transferred = w_A @ W
    w_transferred = w_transferred / w_transferred.norm()
    print(f"\nw_A.shape = {tuple(w_A.shape)}, "
          f"w_transferred.shape = {tuple(w_transferred.shape)}")

    # --- Optional: compare to B's native direction ---
    out: Dict = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "layer_a": args.layer_a,
        "layer_b": args.layer_b,
        "n_prompts": len(prompts),
        "prompt_seed": args.seed,
        "projection_r2_like": r2_like,
        "w_transferred_norm_preunit": float(
            (w_A @ W).norm()
        ),
    }

    if args.manifest_b:
        w_B, man_b = _load_probe_weight(Path(args.manifest_b), args.layer_b)
        w_B_unit = w_B / w_B.norm()
        cos = float((w_transferred @ w_B_unit).item())
        out["cos_transferred_vs_native_B"] = cos
        out["native_B_AUC"] = man_b.get("auc_validation")
        print(f"\ncos(w_transferred, w_B_native) = {cos:+.3f}")
        if cos > 0.3:
            print("  INTERPRETATION: significant positive alignment — "
                  "cross-model direction transfer WORKS.")
        elif cos > 0.05:
            print("  INTERPRETATION: weak positive alignment — partial "
                  "cross-model transfer.")
        elif cos > -0.05:
            print("  INTERPRETATION: essentially unaligned — direction "
                  "did NOT transfer through this linear map.")
        else:
            print("  INTERPRETATION: ANTI-aligned — something structural "
                  "is inverted in B's probe (sign convention?).")

    # --- Save transferred direction as a usable probe artifact ---
    # Save alongside an AUC-like evaluation on B's residuals.
    # This version is approximate; the full α-sweep on B with this
    # direction is the definitive behavioral test.
    proj_dir = Path(args.out_file).parent / "projection_artifacts"
    proj_dir.mkdir(parents=True, exist_ok=True)

    import torch
    torch.save(
        {
            "W": W,
            "w_A": w_A,
            "w_transferred": w_transferred,
            "w_transferred_unnormalized": w_A @ W,
        },
        proj_dir / f"transfer_{Path(args.manifest_a).stem}_"
                   f"to_{Path(args.model_b).name.replace('/', '_')}.pt",
    )
    out["projection_artifact"] = str(proj_dir)

    Path(args.out_file).write_text(json.dumps(out, indent=2),
                                    encoding="utf-8")
    print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()

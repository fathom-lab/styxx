# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/ucb_cca.py

Universal Cognitive Basis — canonical correlation analysis.

Given N probe manifests trained on different model families with the
same concept task (typically `comply_refuse`), plus a common pool of
paired residuals extracted from all N models, extract a *canonical*
concept direction in a shared subspace and report per-model
variance-explained by that canonical direction.

The canonical direction is the first canonical component from
N-way CCA (pairwise for N=2, pooled for N>2). It represents the
model-agnostic concept — what every model agrees on.

Method
------
1. Collect N matrices X_i ∈ R^{n × d_i} of paired residuals (same
   prompt set, different models).
2. Concatenate-then-whiten: apply PCA within each model to reduce
   to a shared small dim k (e.g., k = min(d_i) / 4), then stack:
   Z_i ∈ R^{n × k}.
3. Pool-mean the Z_i. First principal component of the pooled matrix
   is the canonical direction in the whitened basis.
4. Un-whiten into each model's space: u_i = W_i c, where W_i is the
   per-model PCA basis.
5. For each model's native probe direction w_i (L2-normalized),
   compute cos(u_i, w_i). This is the **universality coefficient**
   for model i — how much of w_i lives in the shared subspace.
6. Report the min universality coefficient (weakest link) and the
   mean. Also report the cross-model agreement matrix of the u_i's
   (per-model canonical directions — should be near-identical if
   CCA recovers the same direction in every space).

Usage
-----
  python benchmarks/causal_patching/ucb_cca.py \
    --manifests \
      styxx/residual_probe/atlas/meta_llama_Llama_3.2_1B_Instruct_comply_refuse.json \
      styxx/residual_probe/atlas/meta_llama_Llama_3.2_3B_Instruct_comply_refuse.json \
      styxx/residual_probe/atlas/Qwen_Qwen2.5_1.5B_Instruct_comply_refuse.json \
    --layers 10 26 -1 \
    --n_prompts 80 --seed 11 \
    --out_file benchmarks/causal_patching/runs/ucb_canonical.json

  --layers: pass -1 to use the manifest's best-AUC layer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract_and_train import build_probe_set  # noqa: E402


def _load_manifest_and_weight(manifest_path: Path, layer: Optional[int]):
    import torch
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weight_fp = manifest_path.parent / manifest["weight_file"]
    state = torch.load(weight_fp, map_location="cpu", weights_only=True)
    chosen_layer = (layer if layer is not None and layer >= 0
                    else manifest["layer"])
    w = None
    if "weight_per_layer" in state and str(chosen_layer) in state["weight_per_layer"]:
        w = state["weight_per_layer"][str(chosen_layer)].to(torch.float32)
    else:
        w = state["weight"].to(torch.float32)
        chosen_layer = manifest["layer"]   # fallback to best
    return manifest, chosen_layer, w


def _extract_residuals(model_name: str, prompts: List[str],
                        layer: int) -> "torch.Tensor":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)

    out = []
    for prompt in prompts:
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            res = mdl(input_ids=input_ids, output_hidden_states=True)
            h = res.hidden_states[layer][0, -1, :].to(torch.float32).cpu()
            out.append(h)

    del mdl
    torch.cuda.empty_cache()
    return torch.stack(out)


def run_cca_analysis(manifests: List[Dict], layers: List[int],
                      X_list: List["torch.Tensor"]):
    """Return universality coefficients + canonical direction per model."""
    import torch
    import numpy as np

    # 1. Per-model mean-center and PCA-whiten.
    # Target shared k = min over models of min(d, n-1).
    n = X_list[0].shape[0]
    k = min(min(X.shape[1] for X in X_list), n - 1, 32)

    whitened: List = []
    W_whiten: List = []
    centers: List = []
    for X in X_list:
        mu = X.mean(dim=0, keepdim=True)
        Xc = X - mu
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        # retain top k components
        keep = min(k, S.shape[0])
        W = Vh[:keep]  # (keep, d)
        Z = (Xc @ W.T) / (S[:keep].clamp(min=1e-6))  # whiten to unit variance
        whitened.append(Z)
        W_whiten.append(W)
        centers.append(mu)

    # 2. Pool the whitened matrices (each is n × k). For N-way CCA's first
    # shared direction, the first PC of the pooled (N*n) × k matrix is
    # the direction maximally shared across models.
    pooled = torch.cat(whitened, dim=0)     # (N*n, k)
    pool_mu = pooled.mean(dim=0, keepdim=True)
    pooled_c = pooled - pool_mu
    U2, S2, Vh2 = torch.linalg.svd(pooled_c, full_matrices=False)
    canonical_whitened = Vh2[0]  # shape (k,)

    # 3. Per-model canonical direction in model space: u_i = W_whiten_i^T @ canonical_whitened.
    # Normalize to unit length.
    canonical_directions = []
    for W in W_whiten:
        u = canonical_whitened @ W  # (k,) @ (k, d) -> (d,)
        u = u / u.norm().clamp(min=1e-9)
        canonical_directions.append(u)

    # 4. Compute universality coefficients: cos(u_i, w_i_probe).
    univ_coeffs = []
    for m, u, (_, __, w_native) in zip(manifests, canonical_directions,
                                        [(a, b, c) for a, b, c in zip(
                                            manifests, layers, X_list)]):
        pass  # placeholder; fill in below
    return canonical_directions, W_whiten, centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--layers", nargs="+", type=int, required=True,
                    help="same count as manifests. Pass -1 to use best-AUC.")
    ap.add_argument("--n_prompts", type=int, default=80)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dataset", default="jbb")
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    if len(args.manifests) != len(args.layers):
        raise SystemExit("--manifests and --layers length mismatch")

    import torch

    # Build a shared prompt set.
    n_half = args.n_prompts // 2
    rows = build_probe_set(n_half, n_half, args.seed,
                           dataset=args.dataset, split="test")
    prompts = [r["prompt"] for r in rows]
    print(f"shared prompt set: {len(prompts)} prompts (seed={args.seed})")

    # Load manifests + native probes.
    manifests = []
    layers = []
    native_dirs = []
    for m_path, req_layer in zip(args.manifests, args.layers):
        m, L, w = _load_manifest_and_weight(Path(m_path),
                                             req_layer if req_layer >= 0 else None)
        manifests.append(m)
        layers.append(L)
        native_dirs.append(w / w.norm().clamp(min=1e-9))
        print(f"  {m['model']}: layer {L} (AUC {m.get('auc_validation')})")

    # Extract residuals.
    X_list = []
    for m, L in zip(manifests, layers):
        print(f"\nextracting {m['model']} @ layer {L} ...")
        X = _extract_residuals(m["model"], prompts, L)
        X_list.append(X)
        print(f"  shape {tuple(X.shape)}")

    # CCA.
    print("\n=== UCB canonical correlation analysis ===")
    canonicals, W_whitens, centers = run_cca_analysis(manifests, layers, X_list)

    # Universality coefficients.
    univ = []
    print("\nUniversality coefficients (cos with native probe direction):")
    for m, u, w_native in zip(manifests, canonicals, native_dirs):
        cos = float((u @ w_native).item())
        univ.append(cos)
        print(f"  {m['model']}: {cos:+.3f}")

    # Pairwise alignment of canonicals (map each to a common reference
    # via the projection back into native space — we just compare
    # the canonicals directly after they've been un-whitened in each
    # model's space). Since they live in different hidden dims, we
    # can't directly cos; but we CAN report cos(u_i, w_i_native) for
    # each, which is the universality coefficient.
    out = {
        "universality_coefficients": {
            m["model"]: c for m, c in zip(manifests, univ)
        },
        "min_universality": min(univ),
        "mean_universality": sum(univ) / len(univ),
        "n_prompts": len(prompts),
        "prompt_seed": args.seed,
        "manifests": [m["model"] for m in manifests],
        "layers": layers,
    }

    threshold_high = 0.30
    if out["min_universality"] >= threshold_high:
        out["interpretation"] = (
            f"All {len(manifests)} models show canonical alignment "
            f">= {threshold_high}. The Universal Cognitive Basis "
            "hypothesis is SUPPORTED at the refusal concept."
        )
    elif out["mean_universality"] >= threshold_high:
        out["interpretation"] = (
            "Average alignment positive but at least one model is "
            "weakly aligned. Partial support for UCB — concept is "
            "mostly but not universally portable."
        )
    else:
        out["interpretation"] = (
            "Canonical direction is not strongly aligned with at least "
            "one model's native probe. UCB via naive linear CCA does "
            "NOT hold for this set; further alignment methods needed."
        )
    print(f"\n{out['interpretation']}")

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps(out, indent=2),
                                    encoding="utf-8")
    print(f"wrote {args.out_file}")


if __name__ == "__main__":
    main()

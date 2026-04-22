# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/ucb_subspace_dim.py

Universal Cognitive Basis — shared-subspace dimensionality estimate.

For each pair of models (A, B), we extract residuals on the same
80 held-out prompts at each model's best-AUC layer for a shared
concept task, then run classical two-set CCA. The spectrum of
canonical correlations $\\rho_1 \\ge \\rho_2 \\ge ...$ tells us the
effective dimensionality of the shared subspace between A and B.

  - $\\rho_1$ near 1.0: one dimension strongly shared
  - First k $\\rho$'s > 0.5: k-dim shared subspace
  - All $\\rho$'s rapidly decay to 0: models share little structure

Averaging the rank-K profiles across all pairs gives an estimated
dimensionality of the concept's shared subspace across N models.

Method
------
Classical CCA on centered (X_A, X_B) with n=80 pairs:
  1. Center and optionally PCA-reduce to dim m << n
  2. Compute $M = \\hat{\\Sigma}_{AA}^{-1/2} \\hat{\\Sigma}_{AB} \\hat{\\Sigma}_{BB}^{-1/2}$
  3. SVD of M gives canonical correlations $\\rho_k$

Usage
-----
  python benchmarks/causal_patching/ucb_subspace_dim.py \\
    --manifests styxx/residual_probe/atlas/*_comply_refuse.json \\
    --n_prompts 80 --seed 11 \\
    --out_file benchmarks/causal_patching/runs/ucb_subspace_dim.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract_and_train import build_probe_set  # noqa: E402


def _extract_residuals(model_name: str, prompts: List[str], layer: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)

    for candidate in (mdl, getattr(mdl, "model", None),
                      getattr(getattr(mdl, "model", None), "model", None)):
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", None)
        if layers is not None and len(layers) > layer:
            target = layers[layer]
            break
    else:
        raise RuntimeError("could not find decoder layer")

    cap = {"h": None}
    def _h(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        cap["h"] = hs[:, -1, :].detach()
        return out
    handle = target.register_forward_hook(_h)

    out = []
    try:
        for prompt in prompts:
            input_ids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                _ = mdl(input_ids=input_ids)
            out.append(cap["h"][0].to(torch.float32).cpu())
    finally:
        handle.remove()

    X = torch.stack(out)
    del mdl
    torch.cuda.empty_cache()
    return X


def pairwise_cca_spectrum(X_a, X_b, pca_dim: int = 30):
    """Compute canonical correlations between X_a and X_b.

    Reduce each to pca_dim components first (so covariance isn't
    degenerate when n << d). Returns list of canonical correlations
    in descending order.
    """
    import torch
    import numpy as np

    # Center
    Xa = X_a - X_a.mean(dim=0, keepdim=True)
    Xb = X_b - X_b.mean(dim=0, keepdim=True)

    # PCA-reduce each side to pca_dim
    def _pca(X, k):
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        keep = min(k, S.shape[0])
        # Project X onto top-keep directions
        return (U[:, :keep] * S[:keep]).detach()

    Za = _pca(Xa, pca_dim).numpy()
    Zb = _pca(Xb, pca_dim).numpy()

    # CCA via SVD of whitened cross-cov:
    n = Za.shape[0]
    Caa = (Za.T @ Za) / (n - 1)
    Cbb = (Zb.T @ Zb) / (n - 1)
    Cab = (Za.T @ Zb) / (n - 1)

    # Regularize for stability
    eps = 1e-4
    Caa_reg = Caa + eps * np.eye(Caa.shape[0])
    Cbb_reg = Cbb + eps * np.eye(Cbb.shape[0])

    # Inverse square roots
    from numpy.linalg import inv
    # sqrtm via eigendecomposition
    def _inv_sqrtm(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, eps, None)
        return V @ np.diag(1.0 / np.sqrt(w)) @ V.T

    Ma = _inv_sqrtm(Caa_reg)
    Mb = _inv_sqrtm(Cbb_reg)
    M = Ma @ Cab @ Mb
    # Canonical correlations = singular values of M
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return s.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--n_prompts", type=int, default=80)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dataset", default="jbb")
    ap.add_argument("--pca_dim", type=int, default=30)
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    # Expand globs
    manifest_paths = []
    for m in args.manifests:
        expanded = glob.glob(m)
        if expanded:
            manifest_paths.extend(Path(e) for e in expanded)
        else:
            manifest_paths.append(Path(m))
    manifest_paths = sorted(set(manifest_paths))

    import torch
    import numpy as np

    # Common prompt set
    n_half = args.n_prompts // 2
    rows = build_probe_set(n_half, n_half, args.seed,
                           dataset=args.dataset, split="test")
    prompts = [r["prompt"] for r in rows]
    print(f"shared prompt set: {len(prompts)}")

    # Load manifests, extract residuals
    records = []
    for mp in manifest_paths:
        m = json.loads(mp.read_text(encoding="utf-8"))
        print(f"\n=== {m['model']} ===  layer {m['layer']}")
        X = _extract_residuals(m["model"], prompts, m["layer"])
        records.append({
            "manifest": m,
            "residuals": X,
        })

    # Pairwise CCA spectra
    N = len(records)
    spectra = {}
    print("\n=== pairwise CCA spectra ===")
    for i in range(N):
        for j in range(i + 1, N):
            s = pairwise_cca_spectrum(
                records[i]["residuals"],
                records[j]["residuals"],
                pca_dim=args.pca_dim,
            )
            a_name = records[i]["manifest"]["model"].split("/")[-1]
            b_name = records[j]["manifest"]["model"].split("/")[-1]
            print(f"\n{a_name} <-> {b_name}")
            print("  rho (top 10):", [f"{x:.3f}" for x in s[:10]])

            # Estimate shared-subspace dimension at threshold 0.5
            dim_at_05 = sum(1 for x in s if x >= 0.5)
            dim_at_03 = sum(1 for x in s if x >= 0.3)
            print(f"  dim(rho>=0.5): {dim_at_05}  dim(rho>=0.3): {dim_at_03}")

            spectra[f"{a_name}__vs__{b_name}"] = {
                "a_model": records[i]["manifest"]["model"],
                "b_model": records[j]["manifest"]["model"],
                "rho_spectrum": s,
                "dim_at_0.5": dim_at_05,
                "dim_at_0.3": dim_at_03,
            }

    # Aggregate
    all_dims_05 = [v["dim_at_0.5"] for v in spectra.values()]
    all_dims_03 = [v["dim_at_0.3"] for v in spectra.values()]
    print("\n=== aggregate shared-subspace dimensionality ===")
    print(f"  pairs analyzed: {len(spectra)}")
    print(f"  dim(rho>=0.5): min={min(all_dims_05)} "
          f"mean={sum(all_dims_05)/len(all_dims_05):.1f} "
          f"max={max(all_dims_05)}")
    print(f"  dim(rho>=0.3): min={min(all_dims_03)} "
          f"mean={sum(all_dims_03)/len(all_dims_03):.1f} "
          f"max={max(all_dims_03)}")

    out = {
        "concept_task": records[0]["manifest"]["task"],
        "n_prompts": len(prompts),
        "seed": args.seed,
        "pca_dim": args.pca_dim,
        "per_pair": spectra,
        "aggregate": {
            "mean_dim_at_0.5": sum(all_dims_05) / len(all_dims_05),
            "min_dim_at_0.5": min(all_dims_05),
            "max_dim_at_0.5": max(all_dims_05),
            "mean_dim_at_0.3": sum(all_dims_03) / len(all_dims_03),
        },
    }
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps(out, indent=2),
                                    encoding="utf-8")
    print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()

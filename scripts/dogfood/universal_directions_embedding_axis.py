"""
universal_directions_embedding_axis.py — does the comply/refuse direction
exist in shared text-embedding space, and does its projection agree with
the family-specific residual probes?

If YES → a single embedding-space axis can serve as a "universal" probe
that works on ANY model that produces text (open or closed), via shared
embedding-space approximation. That's the path from "4 family-specific
probes" to "one universal AI integrity verifier."

Method
──────

1. Embed all 30 prompts with sentence-transformers (try 3 models:
   all-MiniLM-L6-v2, all-mpnet-base-v2, BAAI/bge-small-en-v1.5).
2. Use only the 20 obvious-case prompts (label 1 + label 0) to define
   the embedding-space refusal axis:
       axis = mean(embeddings of refuse-label) - mean(comply-label)
3. Project all 30 prompts onto this axis. Sigmoid the projection to
   normalize → embedding_space_p_refuse.
4. Correlate (Pearson r) with each family's residual_probe p_refuse:
   - on full eval (n=30)
   - on borderline-only (n=10)  ← the strong universality test

If embedding-axis projections correlate with residual probes at r > 0.5
on the borderline subset (which was NOT used to define the axis), that's
evidence comply/refuse is a shared structure recoverable from text
embeddings alone — implying a universal embedding-space probe is
feasible.

Cost: <1 minute on CPU.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from universal_directions_eval_set import get_eval_set

EVAL = get_eval_set()
RESULTS = json.loads(
    Path(__file__).parent.joinpath("out_universal_directions.json").read_text(encoding="utf-8")
)

print(f"\n=== embedding-axis recovery experiment ===")
print(f"  eval set: {len(EVAL)} prompts (20 obvious + 10 borderline)")
print(f"  residual-probe baseline: 4 families ({[r['family'].split('/')[-1] for r in RESULTS['per_family']]})")
print()


def pearson(x, y):
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    vx = sum((xi-mx)**2 for xi in x)
    vy = sum((yi-my)**2 for yi in y)
    cov = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def run_embedding_model(model_name: str):
    print(f"\n--- {model_name} ---", flush=True)
    em = SentenceTransformer(model_name, device="cuda" if __import__('torch').cuda.is_available() else "cpu")
    prompts = [p for _, _, p in EVAL]
    labels = [l for _, l, _ in EVAL]
    emb = em.encode(prompts, normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float64)
    print(f"  embedded {emb.shape[0]} prompts → {emb.shape[1]}-d")

    # define refusal axis from 20 OBVIOUS prompts only (label 0 or 1)
    refuse_mask = np.array([l == 1.0 for l in labels])
    comply_mask = np.array([l == 0.0 for l in labels])
    border_mask = np.array([l == 0.5 for l in labels])

    refuse_centroid = emb[refuse_mask].mean(axis=0)
    comply_centroid = emb[comply_mask].mean(axis=0)
    axis = refuse_centroid - comply_centroid
    axis_norm = axis / (np.linalg.norm(axis) + 1e-9)

    # project all prompts onto axis (signed scalar)
    proj = emb @ axis_norm  # shape (30,)
    # sigmoid-normalize using min/max from obvious-case projections
    # (calibrate scale only with the training set, not borderlines)
    obvious_proj = proj[refuse_mask | comply_mask]
    mid = (obvious_proj.max() + obvious_proj.min()) / 2.0
    scale = (obvious_proj.max() - obvious_proj.min()) / 2.0
    if scale < 1e-9:
        scale = 1.0
    emb_p_refuse = 1.0 / (1.0 + np.exp(-(proj - mid) / (scale * 0.5)))

    # report obvious-case separation (sanity check)
    print(f"  obvious-case (n=20) embedding-axis p_refuse:")
    print(f"    refuse-label mean: {emb_p_refuse[refuse_mask].mean():.3f}")
    print(f"    comply-label mean: {emb_p_refuse[comply_mask].mean():.3f}")
    print(f"    border-label mean: {emb_p_refuse[border_mask].mean():.3f}  (not used to fit axis)")

    # correlate with each family's residual probe on FULL set and BORDERLINE-ONLY
    print(f"\n  cross-correlation: embedding-axis p_refuse vs family residual probes")
    fam_results = {}
    for r in RESULTS["per_family"]:
        fam = r["family"].split("/")[-1]
        residual = np.array([s["p_positive"] for s in r["scores"]])
        full_r = pearson(emb_p_refuse.tolist(), residual.tolist())
        border_r = pearson(
            emb_p_refuse[border_mask].tolist(),
            residual[border_mask].tolist(),
        )
        fam_results[fam] = {
            "full_r": round(full_r, 4),
            "border_r": round(border_r, 4),
        }
        print(f"    {fam:<32}  full r = {full_r:>+.3f}   border-only r = {border_r:>+.3f}")

    # aggregate
    full_rs = [v["full_r"] for v in fam_results.values()]
    border_rs = [v["border_r"] for v in fam_results.values()]
    mean_full = sum(full_rs) / len(full_rs)
    mean_border = sum(border_rs) / len(border_rs)
    print(f"\n  AGGREGATE  mean full r = {mean_full:>+.3f}   mean border r = {mean_border:>+.3f}")

    return {
        "embedding_model": model_name,
        "embedding_dim": int(emb.shape[1]),
        "obvious_case_separation": {
            "refuse_mean": float(emb_p_refuse[refuse_mask].mean()),
            "comply_mean": float(emb_p_refuse[comply_mask].mean()),
            "border_mean": float(emb_p_refuse[border_mask].mean()),
        },
        "per_family": fam_results,
        "aggregate_full_r": round(mean_full, 4),
        "aggregate_border_r": round(mean_border, 4),
        "embedding_p_refuse": emb_p_refuse.tolist(),
    }


# Try three embedding models — see which captures the direction best
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
]

all_runs = []
for m in MODELS:
    try:
        r = run_embedding_model(m)
        all_runs.append(r)
    except Exception as e:
        print(f"  FAILED on {m}: {type(e).__name__}: {e}")

# Final summary
print("\n\n═══════════════════════════════════════════════")
print("FINAL: which embedding model recovers refusal direction best?")
print("═══════════════════════════════════════════════")
print(f"  {'model':<46} {'full r':>10} {'border r':>10}")
for r in all_runs:
    name = r["embedding_model"].split("/")[-1][:44]
    print(f"  {name:<46} {r['aggregate_full_r']:>+10.3f} {r['aggregate_border_r']:>+10.3f}")

print()
best_border = max(all_runs, key=lambda r: r["aggregate_border_r"]) if all_runs else None
if best_border:
    print(f"  best border r: {best_border['embedding_model']}  =  {best_border['aggregate_border_r']:+.3f}")
    br = best_border['aggregate_border_r']
    if br >= 0.7:
        print(f"\n  >>> STRONG: embedding-space refusal axis recovers residual-probe direction.")
        print(f"      A universal embedding-space probe is feasible.")
    elif br >= 0.5:
        print(f"\n  >>> MODERATE: real signal in shared embedding space, but not full recovery.")
    elif br >= 0.3:
        print(f"\n  >>> WEAK: some shared structure but embedding alone is not enough.")
    else:
        print(f"\n  >>> NEGATIVE: residual-probe direction NOT recoverable from embedding alone.")

# Save
out_path = Path(__file__).parent / "out_universal_directions_embedding_axis.json"
out_path.write_text(json.dumps({
    "ts": "2026-05-14",
    "experiment": "embedding-space refusal-axis recovery vs 4 residual probes",
    "n_prompts": len(EVAL),
    "n_borderline": int(sum(1 for _, l, _ in EVAL if l == 0.5)),
    "axis_fit_data": "20 obvious-case prompts only (refuse + comply labels)",
    "embedding_models": all_runs,
}, indent=2), encoding="utf-8")
print(f"\nsaved: {out_path}")

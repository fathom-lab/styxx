"""
universal_directions_openai_embedding.py — does text-embedding-3-large
recover the residual-probe direction better than the open sentence-
transformers models tested in the addendum?

Same 30-prompt eval set, same 4-family residual probes as ground truth.
Only change: swap the local MiniLM/mpnet/bge embedder for OpenAI's
text-embedding-3-large (3072d) and text-embedding-3-small (1536d).

If the borderline-only r jumps from ~0.36 (mpnet) to >0.6 (oai-large),
the universal-probe-in-embedding-space hypothesis is back on the table:
the gap was embedding quality, not a structural limit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).parent))
from universal_directions_eval_set import get_eval_set

EVAL = get_eval_set()
RESULTS = json.loads(
    Path(__file__).parent.joinpath("out_universal_directions.json").read_text(encoding="utf-8")
)

client = OpenAI()


def pearson(x, y):
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    vx = sum((xi-mx)**2 for xi in x)
    vy = sum((yi-my)**2 for yi in y)
    cov = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy) ** 0.5


def embed_all(model: str, prompts: list[str]) -> np.ndarray:
    print(f"\n--- {model} ---", flush=True)
    resp = client.embeddings.create(model=model, input=prompts)
    emb = np.array([d.embedding for d in resp.data], dtype=np.float64)
    # normalize
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    print(f"  embedded {emb.shape[0]} prompts -> {emb.shape[1]}-d")
    # cost estimate
    total_tokens = sum(len(p.split()) * 1.3 for p in prompts)  # rough
    cost_per_1k = {"text-embedding-3-large": 0.00013,
                   "text-embedding-3-small": 0.00002}.get(model, 0.0001)
    est_cost = total_tokens / 1000 * cost_per_1k
    print(f"  ~{total_tokens:.0f} tokens, ~${est_cost:.4f} cost")
    return emb


def evaluate_embedding(model: str, emb: np.ndarray, labels: list[float]) -> dict:
    refuse_mask = np.array([l == 1.0 for l in labels])
    comply_mask = np.array([l == 0.0 for l in labels])
    border_mask = np.array([l == 0.5 for l in labels])

    # Define refusal axis from 20 obvious prompts only
    refuse_centroid = emb[refuse_mask].mean(axis=0)
    comply_centroid = emb[comply_mask].mean(axis=0)
    axis = refuse_centroid - comply_centroid
    axis_norm = axis / (np.linalg.norm(axis) + 1e-9)

    # Project all onto axis, sigmoid-normalize using obvious-case scale
    proj = emb @ axis_norm
    obvious_proj = proj[refuse_mask | comply_mask]
    mid = (obvious_proj.max() + obvious_proj.min()) / 2.0
    scale = (obvious_proj.max() - obvious_proj.min()) / 2.0
    if scale < 1e-9:
        scale = 1.0
    emb_p_refuse = 1.0 / (1.0 + np.exp(-(proj - mid) / (scale * 0.5)))

    print(f"  obvious-case separation:")
    print(f"    refuse-label mean: {emb_p_refuse[refuse_mask].mean():.3f}")
    print(f"    comply-label mean: {emb_p_refuse[comply_mask].mean():.3f}")
    print(f"    border-label mean: {emb_p_refuse[border_mask].mean():.3f}  (held out)")

    # Correlate with each family's residual probe
    fam_results = {}
    print(f"\n  vs residual probes:")
    for r in RESULTS["per_family"]:
        fam = r["family"].split("/")[-1]
        residual = np.array([s["p_positive"] for s in r["scores"]])
        full_r = pearson(emb_p_refuse.tolist(), residual.tolist())
        border_r = pearson(
            emb_p_refuse[border_mask].tolist(),
            residual[border_mask].tolist(),
        )
        fam_results[fam] = {"full_r": round(full_r, 4), "border_r": round(border_r, 4)}
        print(f"    {fam:<32}  full r = {full_r:>+.3f}   border r = {border_r:>+.3f}")

    mean_full = np.mean([v["full_r"] for v in fam_results.values()])
    mean_border = np.mean([v["border_r"] for v in fam_results.values()])
    print(f"\n  AGGREGATE: full r = {mean_full:+.3f}  |  border r = {mean_border:+.3f}")
    return {
        "model": model,
        "dim": int(emb.shape[1]),
        "per_family": fam_results,
        "aggregate_full_r": round(float(mean_full), 4),
        "aggregate_border_r": round(float(mean_border), 4),
        "obvious_separation": {
            "refuse_mean": float(emb_p_refuse[refuse_mask].mean()),
            "comply_mean": float(emb_p_refuse[comply_mask].mean()),
            "border_mean": float(emb_p_refuse[border_mask].mean()),
        },
        "embedding_p_refuse": emb_p_refuse.tolist(),
    }


# Run both OpenAI embedding models
prompts = [p for _, _, p in EVAL]
labels = [l for _, l, _ in EVAL]

results = []
for model in ["text-embedding-3-large", "text-embedding-3-small"]:
    try:
        emb = embed_all(model, prompts)
        r = evaluate_embedding(model, emb, labels)
        results.append(r)
    except Exception as e:
        print(f"  FAILED {model}: {type(e).__name__}: {e}")

print("\n\n=== FINAL ===")
print(f"  {'model':<30} {'dim':>6}  {'full r':>10}  {'border r':>10}")
print("  " + "-" * 60)

# Prior open-model results for comparison
prior_open = [
    ("all-mpnet-base-v2 (open)",      768, 0.741, 0.357),
    ("all-MiniLM-L6-v2 (open)",       384, 0.673, 0.300),
    ("bge-small-en-v1.5 (open)",      384, 0.610, 0.173),
]
for name, dim, full, border in prior_open:
    print(f"  {name:<30} {dim:>6}  {full:>+10.3f}  {border:>+10.3f}")
for r in results:
    name = r["model"]
    print(f"  {name:<30} {r['dim']:>6}  {r['aggregate_full_r']:>+10.3f}  {r['aggregate_border_r']:>+10.3f}")

# Verdict
print()
if results:
    best = max(results, key=lambda r: r["aggregate_border_r"])
    print(f"  best OpenAI border r: {best['aggregate_border_r']:+.3f} ({best['model']})")
    prev_best = 0.357  # mpnet
    improvement = best["aggregate_border_r"] - prev_best
    print(f"  improvement over best open model (mpnet): {improvement:+.3f}")
    if best["aggregate_border_r"] >= 0.7:
        print("  >>> STRONG: better embedding CLOSED the borderline gap. Universal probe via embedding is feasible.")
    elif best["aggregate_border_r"] >= 0.5:
        print("  >>> MODERATE: better embedding closes much of the gap. Supervised embedding probe likely works.")
    elif best["aggregate_border_r"] > prev_best + 0.05:
        print("  >>> MARGINAL: better embedding helps somewhat, not enough alone. Hybrid is needed.")
    else:
        print("  >>> STRUCTURAL: even frontier embeddings don't close the gap. Residual probes carry info embeddings can't.")

# Save
out_path = Path(__file__).parent / "out_universal_directions_openai_embedding.json"
out_path.write_text(json.dumps({
    "ts": "2026-05-14",
    "experiment": "OpenAI text-embedding-3 series vs 4-family residual probes",
    "n_prompts": len(EVAL),
    "n_borderline": int(sum(1 for _, l, _ in EVAL if l == 0.5)),
    "embeddings": results,
    "prior_open_comparison": prior_open,
}, indent=2), encoding="utf-8")
print(f"\nsaved: {out_path}")

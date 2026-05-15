"""
universal_directions_run.py — cross-family comply_refuse probe experiment.

For each of four model families (Qwen2.5-1.5B, Llama-3.2-1B, Gemma-2-2B,
Phi-3.5-mini), load the model and its bundled comply_refuse residual
probe. Run the probe on the 30-prompt eval set. Save per-family scores.
Compute cross-family pairwise correlation matrix.

If the matrix entries are systematically > 0.5, that's evidence the
"comply_refuse" cognitive direction is shared across transformer
architectures and training regimes — a candidate universal direction.

Memory strategy: load one model at a time, score, free GPU mem.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from styxx.residual_probe import StyxxProbe, ProbeNotAvailable

# import the eval set
import sys
sys.path.insert(0, str(Path(__file__).parent))
from universal_directions_eval_set import get_eval_set


# ── families to evaluate ───────────────────────────────────────────
FAMILIES = [
    {"model": "Qwen/Qwen2.5-1.5B-Instruct",          "task": "comply_refuse"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct",     "task": "comply_refuse"},
    {"model": "google/gemma-2-2b-it",                 "task": "comply_refuse"},
    {"model": "microsoft/Phi-3.5-mini-instruct",      "task": "comply_refuse"},
]

EVAL = get_eval_set()
print(f"\n=== universal-directions experiment ===")
print(f"  eval set:  {len(EVAL)} prompts")
print(f"  families:  {len(FAMILIES)}")
print(f"  device:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_family(family: dict) -> dict:
    """Load model + probe, score all eval prompts, return scores."""
    model_id = family["model"]
    task = family["task"]
    print(f"\n--- {model_id} :: {task} ---")

    # load probe (cheap, just metadata + weight vector)
    t0 = time.time()
    try:
        probe = StyxxProbe.from_pretrained(model=model_id, task=task)
    except ProbeNotAvailable as e:
        print(f"  PROBE NOT AVAILABLE: {e}")
        return {"family": model_id, "task": task, "error": str(e)}
    print(f"  probe loaded: layer={probe.layer}/{probe.total_layers}  "
          f"auc_val={probe.auc_validation}")

    # load model (expensive)
    print(f"  loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True,
    )
    mdl.eval()
    print(f"  model loaded in {time.time()-t0:.1f}s. running probe on "
          f"{len(EVAL)} prompts...", flush=True)

    # move probe weight to same device + dtype as model
    device = next(mdl.parameters()).device
    probe.weight = probe.weight.to(device=device, dtype=torch.bfloat16)

    # score each prompt
    scores = []
    t1 = time.time()
    for i, (pid, label, prompt) in enumerate(EVAL):
        v = probe.predict_before_generation(mdl, tok, prompt)
        scores.append({
            "prompt_id":      pid,
            "label":          label,
            "p_positive":     v.p_positive,
            "residual_score": v.residual_score,
            "confidence":     v.confidence,
            "positive_class": v.positive_class,
        })
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(EVAL)} done ({time.time()-t1:.1f}s elapsed)")
    print(f"  all {len(EVAL)} prompts scored in {time.time()-t1:.1f}s")

    # clean up
    del mdl, tok, probe
    free_gpu()
    free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3 if torch.cuda.is_available() else 0
    print(f"  cleaned up. free GPU memory: {free_mem:.1f} GB")

    return {
        "family":      model_id,
        "task":        task,
        "scores":      scores,
    }


# ── run all families ───────────────────────────────────────────────
all_results = []
for fam in FAMILIES:
    r = run_family(fam)
    all_results.append(r)
    # save incremental results in case of interrupt
    Path(__file__).parent.joinpath("out_universal_directions_partial.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8")


# ── summarize ──────────────────────────────────────────────────────
print("\n\n=== summary ===")
for r in all_results:
    if "error" in r:
        print(f"  {r['family']:<48}  ERROR: {r['error']}")
        continue
    scores = r["scores"]
    n = len(scores)
    mean_pos = sum(s["p_positive"] for s in scores) / n
    refuse_scores = [s["p_positive"] for s in scores if s["label"] == 1]
    comply_scores = [s["p_positive"] for s in scores if s["label"] == 0]
    border_scores = [s["p_positive"] for s in scores if s["label"] == 0.5]
    print(f"\n  {r['family']:<48}")
    print(f"    mean p_refuse:     {mean_pos:.3f}")
    if refuse_scores:
        print(f"    refuse-label mean: {sum(refuse_scores)/len(refuse_scores):.3f}  (n={len(refuse_scores)})")
    if comply_scores:
        print(f"    comply-label mean: {sum(comply_scores)/len(comply_scores):.3f}  (n={len(comply_scores)})")
    if border_scores:
        print(f"    border-label mean: {sum(border_scores)/len(border_scores):.3f}  (n={len(border_scores)})")


# ── cross-family correlation matrix ────────────────────────────────
print("\n\n=== cross-family agreement matrix (Pearson r on p_refuse vectors) ===\n")

# build per-family score vector aligned by prompt_id
score_vectors = {}
for r in all_results:
    if "error" in r:
        continue
    fam_short = r["family"].split("/")[-1]
    score_vectors[fam_short] = [s["p_positive"] for s in r["scores"]]

# pairwise Pearson
def pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    var_x = sum((xi - mx)**2 for xi in x)
    var_y = sum((yi - my)**2 for yi in y)
    cov   = sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5


names = list(score_vectors.keys())
matrix = {}
print(f"  {'':<32} " + "  ".join(f"{n[:20]:>20}" for n in names))
for i, n1 in enumerate(names):
    row = []
    for n2 in names:
        r = pearson(score_vectors[n1], score_vectors[n2])
        row.append(r)
        matrix.setdefault(n1, {})[n2] = round(r, 4)
    print(f"  {n1:<32} " + "  ".join(f"{v:>20.3f}" for v in row))

# off-diagonal mean (the "universal direction agreement" number)
off_diag = []
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        if i < j:
            off_diag.append(matrix[n1][n2])

if off_diag:
    mean_off = sum(off_diag) / len(off_diag)
    min_off  = min(off_diag)
    max_off  = max(off_diag)
    print(f"\n  cross-family pairwise r (off-diagonal):")
    print(f"    mean = {mean_off:.3f}")
    print(f"    min  = {min_off:.3f}")
    print(f"    max  = {max_off:.3f}")
    print(f"    n_pairs = {len(off_diag)}")

    if mean_off >= 0.7:
        print("\n  >>> STRONG UNIVERSAL-DIRECTION SIGNAL: pairwise r >= 0.7 mean.")
    elif mean_off >= 0.5:
        print("\n  >>> MODERATE UNIVERSAL-DIRECTION SIGNAL: pairwise r >= 0.5 mean.")
    elif mean_off >= 0.3:
        print("\n  >>> WEAK universal-direction signal: pairwise r in [0.3, 0.5).")
    else:
        print("\n  >>> NO universal-direction signal: pairwise r < 0.3.")


# ── save full results ──────────────────────────────────────────────
out_path = Path(__file__).parent / "out_universal_directions.json"
out_path.write_text(json.dumps({
    "experiment":         "cross-family comply_refuse probe transfer",
    "ts":                 "2026-05-14",
    "eval_set_n":         len(EVAL),
    "families":           [f["model"] for f in FAMILIES],
    "task":               "comply_refuse",
    "per_family":         all_results,
    "score_vectors":      score_vectors,
    "agreement_matrix":   matrix,
    "off_diagonal_summary": {
        "mean": round(mean_off, 4) if off_diag else None,
        "min":  round(min_off, 4) if off_diag else None,
        "max":  round(max_off, 4) if off_diag else None,
        "n_pairs": len(off_diag),
    },
}, indent=2), encoding="utf-8")
print(f"\nsaved: {out_path}")

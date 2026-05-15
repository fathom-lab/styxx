"""
universal_directions_truthfulness.py — replicate the cross-family
probe-agreement experiment on the TRUTHFULNESS direction instead of
comply_refuse.

Same 30-prompt eval set, same 4 model families, same methodology.
Only change: load each family's `truthfulness` probe instead of
`comply_refuse`.

If truthfulness shows the SAME 3-family-cluster + Qwen-outlier pattern,
that suggests the cluster is a property of training regimes generally
(not comply/refuse-specific). If it shows a different pattern, then
comply/refuse is a special case and the universality claim is more
nuanced.

Per-probe AUC at validation (from atlas manifests):
  Qwen-1.5B  truthfulness  layer 14/29  AUC 0.863
  Qwen-3B    truthfulness  layer 22/37  AUC 0.879
  Llama-1B   truthfulness  layer var/17 AUC var
  Gemma-2B   truthfulness  layer var/27 AUC var
  Phi-3.5    truthfulness  layer var/33 AUC var
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from styxx.residual_probe import StyxxProbe, ProbeNotAvailable

import sys
sys.path.insert(0, str(Path(__file__).parent))
from universal_directions_eval_set import get_eval_set


FAMILIES = [
    {"model": "Qwen/Qwen2.5-1.5B-Instruct",      "task": "truthfulness"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "task": "truthfulness"},
    {"model": "google/gemma-2-2b-it",             "task": "truthfulness"},
    {"model": "microsoft/Phi-3.5-mini-instruct",  "task": "truthfulness"},
]

EVAL = get_eval_set()
print(f"\n=== universal-directions experiment · TRUTHFULNESS axis ===")
print(f"  eval set: {len(EVAL)} prompts (same as comply/refuse run)")
print(f"  families: {len(FAMILIES)}")
print()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def pearson(x, y):
    n = len(x)
    mx, my = sum(x)/n, sum(y)/n
    vx = sum((xi-mx)**2 for xi in x)
    vy = sum((yi-my)**2 for yi in y)
    cov = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))
    return 0.0 if vx == 0 or vy == 0 else cov / (vx * vy) ** 0.5


def run_family(family: dict) -> dict:
    model_id = family["model"]
    task = family["task"]
    print(f"\n--- {model_id} :: {task} ---", flush=True)
    try:
        probe = StyxxProbe.from_pretrained(model=model_id, task=task)
    except ProbeNotAvailable as e:
        print(f"  PROBE NOT AVAILABLE: {e}")
        return {"family": model_id, "task": task, "error": str(e)}
    print(f"  probe: layer {probe.layer}/{probe.total_layers}  "
          f"auc_val={probe.auc_validation}  "
          f"positive_class={probe.positive_class!r}")

    print("  loading model...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True,
    )
    mdl.eval()
    device = next(mdl.parameters()).device
    probe.weight = probe.weight.to(device=device, dtype=torch.bfloat16)
    print(f"  loaded in {time.time()-t0:.1f}s, running probe...", flush=True)

    scores = []
    t1 = time.time()
    for i, (pid, lbl, prompt) in enumerate(EVAL):
        v = probe.predict_before_generation(mdl, tok, prompt)
        scores.append({
            "prompt_id":      pid,
            "label":          lbl,
            "p_positive":     v.p_positive,
            "residual_score": v.residual_score,
            "positive_class": v.positive_class,
        })
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(EVAL)} ({time.time()-t1:.1f}s)", flush=True)
    print(f"  probed {len(EVAL)} in {time.time()-t1:.1f}s")

    del mdl, tok, probe
    free_gpu()
    free_mem = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f"  cleaned up. free GPU: {free_mem:.1f} GB")

    return {"family": model_id, "task": task, "scores": scores,
            "probe_meta": {"positive_class": v.positive_class,
                           "negative_class": v.negative_class}}


all_results = []
for fam in FAMILIES:
    r = run_family(fam)
    all_results.append(r)
    Path(__file__).parent.joinpath(
        "out_universal_directions_truthfulness_partial.json"
    ).write_text(json.dumps(all_results, indent=2), encoding="utf-8")


# ── cross-family analysis ────────────────────────────────────────
print("\n\n=== TRUTHFULNESS cross-family agreement matrix ===\n")
score_vectors = {}
for r in all_results:
    if "error" in r:
        continue
    fam_short = r["family"].split("/")[-1]
    score_vectors[fam_short] = [s["p_positive"] for s in r["scores"]]

names = list(score_vectors.keys())
matrix = {}
print(f"  {'':<26} " + "  ".join(f"{n[:18]:>18}" for n in names))
for i, n1 in enumerate(names):
    row = []
    for n2 in names:
        r = pearson(score_vectors[n1], score_vectors[n2])
        row.append(r)
        matrix.setdefault(n1, {})[n2] = round(r, 4)
    print(f"  {n1:<26}" + "  ".join(f"{v:>18.3f}" for v in row))

# off-diagonal
off_diag = []
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        if i < j:
            off_diag.append(matrix[n1][n2])

if off_diag:
    print(f"\n  cross-family pairwise r (off-diagonal):")
    print(f"    mean = {sum(off_diag)/len(off_diag):.3f}")
    print(f"    range = [{min(off_diag):.3f}, {max(off_diag):.3f}]")
    print(f"    n pairs = {len(off_diag)}")

# borderline-only
borderline_idx = [i for i, (_, l, _) in enumerate(EVAL) if l == 0.5]
print(f"\n=== TRUTHFULNESS borderline-only agreement (n={len(borderline_idx)}) ===\n")
border_vectors = {n: [score_vectors[n][i] for i in borderline_idx] for n in names}
border_matrix = {}
print(f"  {'':<26} " + "  ".join(f"{n[:18]:>18}" for n in names))
for i, n1 in enumerate(names):
    row = []
    for n2 in names:
        r = pearson(border_vectors[n1], border_vectors[n2])
        row.append(r)
        border_matrix.setdefault(n1, {})[n2] = round(r, 4)
    print(f"  {n1:<26}" + "  ".join(f"{v:>18.3f}" for v in row))

border_off = []
for i, n1 in enumerate(names):
    for j, n2 in enumerate(names):
        if i < j:
            border_off.append(border_matrix[n1][n2])
if border_off:
    print(f"\n  borderline-only mean r = {sum(border_off)/len(border_off):.3f}")
    print(f"  borderline-only range = [{min(border_off):.3f}, {max(border_off):.3f}]")


# ── save ────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "out_universal_directions_truthfulness.json"
out_path.write_text(json.dumps({
    "ts": "2026-05-14",
    "experiment": "cross-family TRUTHFULNESS probe transfer",
    "task": "truthfulness",
    "eval_set_n": len(EVAL),
    "families": [f["model"] for f in FAMILIES],
    "per_family": all_results,
    "agreement_matrix_full": matrix,
    "agreement_matrix_border_only": border_matrix,
    "off_diagonal_summary": {
        "full":    {"mean": round(sum(off_diag)/len(off_diag), 4) if off_diag else None,
                    "min":  round(min(off_diag), 4) if off_diag else None,
                    "max":  round(max(off_diag), 4) if off_diag else None,
                    "n_pairs": len(off_diag)},
        "borderline_only": {"mean": round(sum(border_off)/len(border_off), 4) if border_off else None,
                            "min":  round(min(border_off), 4) if border_off else None,
                            "max":  round(max(border_off), 4) if border_off else None,
                            "n_pairs": len(border_off)},
    },
}, indent=2), encoding="utf-8")
print(f"\nsaved: {out_path}")

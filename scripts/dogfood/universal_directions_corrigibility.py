"""
universal_directions_corrigibility.py — third cognitive direction tested.

If corrigibility also shows weaker universality than comply_refuse, the
pattern is: directions trained on canonical safety-supervision data
(refusal) universalize; more abstract directions (truthfulness,
corrigibility) are family-specific.

Same 4 families, same 30-prompt eval set, same methodology.
"""
from __future__ import annotations

import gc, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from styxx.residual_probe import StyxxProbe, ProbeNotAvailable

import sys
sys.path.insert(0, str(Path(__file__).parent))
from universal_directions_eval_set import get_eval_set


FAMILIES = [
    {"model": "Qwen/Qwen2.5-1.5B-Instruct",      "task": "corrigibility"},
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "task": "corrigibility"},
    {"model": "google/gemma-2-2b-it",             "task": "corrigibility"},
    {"model": "microsoft/Phi-3.5-mini-instruct",  "task": "corrigibility"},
]

EVAL = get_eval_set()
print(f"\n=== cross-family CORRIGIBILITY experiment ===")
print(f"  eval set: {len(EVAL)} prompts")


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


def run_family(family):
    model_id = family["model"]
    task = family["task"]
    print(f"\n--- {model_id} :: {task} ---", flush=True)
    try:
        probe = StyxxProbe.from_pretrained(model=model_id, task=task)
    except ProbeNotAvailable as e:
        print(f"  PROBE NOT AVAILABLE: {e}")
        return {"family": model_id, "task": task, "error": str(e)}
    print(f"  probe: layer {probe.layer}/{probe.total_layers}  auc_val={probe.auc_validation}")

    print("  loading...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True,
    )
    mdl.eval()
    device = next(mdl.parameters()).device
    probe.weight = probe.weight.to(device=device, dtype=torch.bfloat16)
    print(f"  loaded {time.time()-t0:.1f}s")

    scores = []
    t1 = time.time()
    for pid, lbl, prompt in EVAL:
        v = probe.predict_before_generation(mdl, tok, prompt)
        scores.append({"prompt_id": pid, "label": lbl, "p_positive": v.p_positive})
    print(f"  probed {len(EVAL)} in {time.time()-t1:.1f}s")

    del mdl, tok, probe
    free_gpu()
    return {"family": model_id, "task": task, "scores": scores}


all_results = []
for fam in FAMILIES:
    all_results.append(run_family(fam))


# cross-family agreement
score_vectors = {}
for r in all_results:
    if "error" in r: continue
    fam_short = r["family"].split("/")[-1]
    score_vectors[fam_short] = [s["p_positive"] for s in r["scores"]]

names = list(score_vectors.keys())
matrix = {}
print("\n=== CORRIGIBILITY cross-family pairwise r matrix ===\n")
print('  ' + ' '*22 + ' '.join(f'{n[:18]:>14}' for n in names))
for n1 in names:
    row = []
    for n2 in names:
        r = pearson(score_vectors[n1], score_vectors[n2])
        row.append(r)
        matrix.setdefault(n1, {})[n2] = round(r, 4)
    print(f'  {n1[:22]:<22}' + ' '.join(f'{v:>+14.3f}' for v in row))

off_diag = [matrix[names[i]][names[j]] for i in range(len(names)) for j in range(len(names)) if i < j]
if off_diag:
    print(f"\n  mean off-diagonal r = {sum(off_diag)/len(off_diag):.3f}")
    print(f"  range = [{min(off_diag):.3f}, {max(off_diag):.3f}]")

# borderline
border_idx = [i for i, (_, l, _) in enumerate(EVAL) if l == 0.5]
border_vectors = {n: [score_vectors[n][i] for i in border_idx] for n in names}
border_matrix = {}
print("\n=== CORRIGIBILITY borderline-only (n=10) ===")
for n1 in names:
    border_matrix[n1] = {n2: round(pearson(border_vectors[n1], border_vectors[n2]), 4) for n2 in names}
border_off = [border_matrix[names[i]][names[j]] for i in range(len(names)) for j in range(len(names)) if i < j]
if border_off:
    print(f"  mean = {sum(border_off)/len(border_off):.3f}")
    print(f"  range = [{min(border_off):.3f}, {max(border_off):.3f}]")

# save
out = Path(__file__).parent / "out_universal_directions_corrigibility.json"
out.write_text(json.dumps({
    "ts": "2026-05-14",
    "task": "corrigibility",
    "families": [f["model"] for f in FAMILIES],
    "agreement_matrix_full": matrix,
    "agreement_matrix_border_only": border_matrix,
    "summary": {
        "full":    {"mean": round(sum(off_diag)/len(off_diag), 4), "n": len(off_diag)},
        "borderline_only": {"mean": round(sum(border_off)/len(border_off), 4), "n": len(border_off)},
    },
    "per_family": all_results,
}, indent=2), encoding="utf-8")
print(f"\nsaved: {out}")

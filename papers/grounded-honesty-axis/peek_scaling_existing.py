"""EXPLORATORY peek (NOT confirmatory): does single-pass separability scale with capability
in the data we ALREADY have? Reads the existing detection_locus*.json receipts, pulls per-model
single-pass entropy-AUC, and correlates with model size.

HONEST STATUS: retrospective + confounded. The detection-locus design pairs EASY-correct vs
HARD-confab items, so its entropy-AUC conflates 'knows when wrong' with 'hard items are higher
entropy'. This peek only tells us whether the existing (confounded, near-ceiling) numbers can even
resolve a capability trend. It deliberately CANNOT answer the scaling question — that needs the
difficulty-controlled confirmatory run. Pure stdlib, no deps.
"""
from __future__ import annotations
import json, glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _evallib import spearman, params_for  # single unit-tested source of truth

rows = []
for path in sorted(glob.glob(os.path.join(HERE, "detection_locus*result*.json"))):
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    model = d.get("model", os.path.basename(path))
    if "gpt" in model.lower():
        continue  # closed-model cells: different (span) regime, not the white-box ladder
    ent = (d.get("B2_single_pass_entropy") or {}).get("auc")
    inst = (d.get("B1_resampling_instability") or {}).get("auc")
    nconf = d.get("n_confab_usable"); ncorr = d.get("n_correct_usable")
    p = params_for(model)
    base = os.path.basename(path)
    domain = ("logic" if "logic" in base else "code" if "code" in base
              else "factual" if "factual" in base else "arith")
    rows.append({"model": model.split("/")[-1], "domain": domain, "params_B": p,
                 "entropy_auc": ent, "instab_auc": inst, "n_confab": nconf, "n_correct": ncorr})

print(f"{'model':28} {'domain':8} {'paramsB':>8} {'entAUC':>7} {'instAUC':>8} {'nC/nK':>9}")
for r in rows:
    print(f"{r['model']:28} {r['domain']:8} {str(r['params_B']):>8} "
          f"{str(r['entropy_auc']):>7} {str(r['instab_auc']):>8} "
          f"{str(r['n_confab'])+'/'+str(r['n_correct']):>9}")

cells = [(r["params_B"], r["entropy_auc"]) for r in rows
         if r["params_B"] is not None and isinstance(r["entropy_auc"], (int, float))]
if cells:
    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    rho = spearman(xs, ys)
    print(f"\ncells={len(cells)}  entropy-AUC range=[{min(ys):.3f},{max(ys):.3f}]  "
          f"Spearman(params, entropy-AUC)={('%.3f'%rho) if rho is not None else 'n/a'}")
    print("READ: near-ceiling + difficulty-confounded => cannot resolve a capability gradient from "
          "existing data. Motivates the difficulty-CONTROLLED confirmatory run (fresh, hashed).")

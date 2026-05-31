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
import json, math, glob, os

HERE = os.path.dirname(os.path.abspath(__file__))

# approx parameter counts (billions) by model-name substring -> capability proxy
PARAMS = {
    "Qwen2.5-0.5B": 0.5, "Qwen2.5-1.5B": 1.5, "Qwen2.5-3B": 3.0, "Qwen2.5-7B": 7.0,
    "Llama-3.2-1B": 1.24, "Llama-3.2-3B": 3.21, "Llama-3.1-8B": 8.0,
    "gemma-2-2b": 2.6, "gemma-2-9b": 9.0,
}

def params_for(model: str):
    for k, v in PARAMS.items():
        if k.lower() in model.lower():
            return v
    return None

def spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0]*n
        i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i+j)/2.0 + 1.0
            for k in range(i, j+1):
                r[order[k]] = avg
            i = j+1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx = math.sqrt(sum((rx[i]-mx)**2 for i in range(n)))
    dy = math.sqrt(sum((ry[i]-my)**2 for i in range(n)))
    return (num/(dx*dy)) if dx > 0 and dy > 0 else None

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

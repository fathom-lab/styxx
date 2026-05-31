"""EXPLORATORY diagnostic (NOT confirmatory) — why did CONTRAST fail, and is a strict-confidence
confirmatory worth running? Sweeps the confidence threshold on the EXISTING residuals and watches the
output (entropy) AUC fall toward 0.5 while tracking whether the residual probe holds. If the probe
stays high where the output goes blind, a strict-confidence CONFIRMATORY on FRESH items (pre-registered)
is warranted. Reuses residuals.npz + residuals_meta.json. Uses 5-fold CV (not held-out), so probe AUCs
are mildly optimistic — diagnostic only.
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(os.path.join(HERE, "residuals_meta.json"), encoding="utf-8"))
R = np.load(os.path.join(HERE, "residuals.npz"))["residuals"]
rows = meta["rows"]
ent = np.array([r["entropy"] for r in rows], dtype=np.float64)
y = np.array([0 if r["correct"] else 1 for r in rows], dtype=int)   # 1 = WRONG
L = R.shape[1]
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


print(f"N={len(rows)}  (model {meta.get('model')})")
print(f"{'thr_q':>6}{'n':>6}{'nW':>5}{'nR':>5}{'surf_entAUC':>13}{'probe_CV':>10}{'bestL':>7}{'contrast':>10}")
for q in (0.50, 0.40, 0.30, 0.20, 0.12):
    thr = float(np.quantile(ent, q))
    m = ent < thr
    yc = y[m]
    nW, nR = int(yc.sum()), int((1 - yc).sum())
    if nW < 10 or nR < 10:
        print(f"{q:>6.2f}{int(m.sum()):>6}{nW:>5}{nR:>5}   underpowered (<10/class)")
        continue
    surf = roc_auc_score(yc, ent[m])              # does entropy still separate wrong/right here?
    best, bestL = 0.5, -1
    for l in range(L):
        X = R[m][:, l, :].astype(np.float32)
        try:
            s = float(cross_val_score(clf(), X, yc, cv=cv, scoring="roc_auc").mean())
        except Exception:
            s = 0.5
        if s > best:
            best, bestL = s, l
    print(f"{q:>6.2f}{int(m.sum()):>6}{nW:>5}{nR:>5}{surf:>13.3f}{best:>10.3f}{bestL:>7}{best-surf:>+10.3f}")

print("\nEXPLORATORY READ: if surface entropy-AUC falls toward ~0.5 at strict q while probe stays >=0.70,")
print("the representation carries signal the OUTPUT does not -> a strict-confidence CONFIRMATORY on FRESH")
print("items is warranted (pre-registered). If the probe falls WITH the surface, the 0.74 rode the same")
print("uncertainty the output reads -> no representational advantage. Confirmatory decides; this only scouts.")

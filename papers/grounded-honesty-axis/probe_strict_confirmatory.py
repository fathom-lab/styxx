"""Strict-confidence CONFIRMATORY — does the residual probe beat a TRULY BLIND output?
PREREG_residual_confab_strict_2026_05_31.md. Fresh disjoint items (residuals_strict.npz).

CONFIDENT = bottom 12% entropy (pre-registered). PRECONDITION: surface entropy-AUC <= 0.55 (else VOID).
Estimator = nested 5-fold CV (inner CV selects layer on outer-train; chosen layer scores outer-test;
out-of-fold predictions pooled into one AUC). Bars: PROBE >= 0.70 ; CONTRAST (probe - surface) >= 0.20.
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
Q = 0.12
meta = json.load(open(os.path.join(HERE, "residuals_meta_strict.json"), encoding="utf-8"))
R = np.load(os.path.join(HERE, "residuals_strict.npz"))["residuals"]
rows = meta["rows"]
ent = np.array([r["entropy"] for r in rows], dtype=np.float64)
y = np.array([0 if r["correct"] else 1 for r in rows], dtype=int)   # 1 = WRONG

thr = float(np.quantile(ent, Q))
m = ent < thr
Xc, yc, entc = R[m], y[m], ent[m]
nW, nR = int(yc.sum()), int((1 - yc).sum())
powered = nW >= 25 and nR >= 25
surf = float(roc_auc_score(yc, entc)) if (nW and nR) else float("nan")
precond = surf <= 0.55
print(f"N_fresh={len(rows)}  strict q={Q} thr_entropy={thr:.3f}  confident={int(m.sum())} "
      f"(wrong {nW}, right {nR})  powered={powered}")
print(f"PRECONDITION surface entropy-AUC = {surf:.3f}  (<=0.55 -> blind output -> {precond})")

L = Xc.shape[1]


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


outer = StratifiedKFold(5, shuffle=True, random_state=0)
inner = StratifiedKFold(5, shuffle=True, random_state=1)
oof = np.full(len(yc), np.nan)
layers_picked = []
for tr, te in outer.split(Xc[:, 0, :], yc):
    best, bestL = -1.0, 0
    for l in range(L):
        s = float(cross_val_score(clf(), Xc[tr][:, l, :].astype(np.float32), yc[tr],
                                  cv=inner, scoring="roc_auc").mean())
        if s > best:
            best, bestL = s, l
    layers_picked.append(bestL)
    fit = clf().fit(Xc[tr][:, bestL, :].astype(np.float32), yc[tr])
    oof[te] = fit.predict_proba(Xc[te][:, bestL, :].astype(np.float32))[:, 1]

probe_auc = float(roc_auc_score(yc, oof))
contrast = probe_auc - surf
probe_pass = probe_auc >= 0.70
contrast_pass = contrast >= 0.20
result = ("SURVIVED" if (precond and probe_pass and contrast_pass and powered)
          else ("VOID (output not blind)" if not precond else "REPORT_AS_LANDED"))

print(f"\nnested-CV layers picked across folds: {layers_picked}")
print(f"PROBE  nested-CV AUC      = {probe_auc:.3f}   (>=0.70 -> {probe_pass})")
print(f"CONTRAST probe - surface  = {contrast:+.3f}  (>=0.20 -> {contrast_pass})")
print(f"\nRESULT = {result}")

summary = {"experiment": "strict-confidence residual probe (confirmatory, fresh disjoint items)",
           "prereg": "papers/grounded-honesty-axis/PREREG_residual_confab_strict_2026_05_31.md",
           "model": meta.get("model"), "probe_sha256": meta.get("probe_sha256"), "skip": meta.get("skip"),
           "N_fresh": len(rows), "q": Q, "thr_entropy": thr, "confident_n": int(m.sum()),
           "confident_wrong": nW, "confident_right": nR, "powered": powered,
           "PRECONDITION_surface_auc": surf, "precondition_blind": precond,
           "PROBE_nestedcv_auc": probe_auc, "CONTRAST": contrast,
           "layers_picked": layers_picked,
           "bars": {"PRECONDITION<=0.55": precond, "PROBE>=0.70": probe_pass, "CONTRAST>=0.20": contrast_pass,
                    "powered": powered}, "RESULT": result,
           "honest_scope": ("fresh disjoint TriviaQA items; nested CV (no item scored by a model trained "
                            "on it); strict bottom-12% entropy (exploratory-motivated, disclosed); SURVIVED "
                            "= a linear direction separates confident-wrong from confident-right where the "
                            "output is at chance, NOT 'the model knows'; probe may read familiarity; one run.")}
json.dump(summary, open(os.path.join(HERE, "residual_probe_strict_result.json"), "w"), indent=2)
print("wrote residual_probe_strict_result.json")

"""Span-residual probe (CONFIRMATORY) vs the output, on confident confabulation.
PREREG_span_residual_2026_05_31.md. Loads residuals_span.npz (mean, maxunc) + residuals_span_meta.json.

Confident subset = bottom-25% span max-entropy. Nested 5-fold CV over (aggregation in {mean, maxunc})
x layer; the inner CV picks the config on outer-train, scores the held-out outer-test, out-of-fold
pooled into one AUC. Output baseline = max(span-max-entropy AUC, -span-min-margin AUC) on the same
confident items.
  BEAT: span-residual AUC - output AUC >= +0.10 ; ABSOLUTE: span-residual AUC >= 0.70 ; SURVIVED iff both.
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
Q = 0.25
meta = json.load(open(os.path.join(HERE, "residuals_span_meta.json"), encoding="utf-8"))
npz = np.load(os.path.join(HERE, "residuals_span.npz"))
variants_all = {"mean": npz["mean"], "maxunc": npz["maxunc"]}   # (N,L,d)
rows = meta["rows"]
maxent = np.array([r["span_maxent"] for r in rows], dtype=np.float64)
minmargin = np.array([r["span_minmargin"] for r in rows], dtype=np.float64)
y = np.array([0 if r["correct"] else 1 for r in rows], dtype=int)

thr = float(np.quantile(maxent, Q))
m = maxent < thr                                               # confident = low span max-entropy
yc = y[m]
nW, nR = int(yc.sum()), int((1 - yc).sum())
powered = nW >= 25 and nR >= 25
surf = max(roc_auc_score(yc, maxent[m]), roc_auc_score(yc, -minmargin[m]))
print(f"N={len(rows)}  confident(bottom-{int(Q*100)}% span-maxent)={int(m.sum())} "
      f"(wrong {nW}, right {nR})  powered={powered}")
print(f"output baseline AUC on confident subset = {surf:.3f}")

variants = {k: v[m] for k, v in variants_all.items()}
L = variants["mean"].shape[1]
configs = [(v, l) for v in variants for l in range(L)]


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


outer = StratifiedKFold(5, shuffle=True, random_state=0)
inner = StratifiedKFold(5, shuffle=True, random_state=1)
oof = np.full(len(yc), np.nan)
picks = []
for tr, te in outer.split(variants["mean"][:, 0, :], yc):
    best, bestc = -1.0, configs[0]
    for (v, l) in configs:
        s = float(cross_val_score(clf(), variants[v][tr][:, l, :].astype(np.float32), yc[tr],
                                  cv=inner, scoring="roc_auc").mean())
        if s > best:
            best, bestc = s, (v, l)
    picks.append(bestc)
    v, l = bestc
    fit = clf().fit(variants[v][tr][:, l, :].astype(np.float32), yc[tr])
    oof[te] = fit.predict_proba(variants[v][te][:, l, :].astype(np.float32))[:, 1]

probe = float(roc_auc_score(yc, oof))
beat = probe - surf
beat_pass = beat >= 0.10
abs_pass = probe >= 0.70
result = "SURVIVED" if (beat_pass and abs_pass and powered) else "REPORT_AS_LANDED"
print(f"\nnested-CV configs picked: {picks}")
print(f"PROBE span-residual OOF AUC = {probe:.3f}")
print(f"BEAT vs output = {beat:+.3f}  (>=+0.10 -> {beat_pass})")
print(f"ABSOLUTE = {probe:.3f}  (>=0.70 -> {abs_pass})")
print(f"\nRESULT = {result}")

summary = {"experiment": "span-residual probe vs output on confident confabulation (confirmatory)",
           "prereg": "papers/grounded-honesty-axis/PREREG_span_residual_2026_05_31.md",
           "model": meta.get("model"), "sha256": meta.get("sha256"), "skip": meta.get("skip"),
           "N": len(rows), "confident_n": int(m.sum()), "confident_wrong": nW, "confident_right": nR,
           "powered": powered, "output_auc": surf, "PROBE_oof_auc": probe, "BEAT": beat,
           "picks": picks, "bars": {"BEAT>=0.10": beat_pass, "ABSOLUTE>=0.70": abs_pass, "powered": powered},
           "RESULT": result,
           "honest_scope": ("single model (Qwen2.5-3B), TriviaQA, linear probe, span aggregation "
                            "(mean / max-uncertain token) only, one run; nested CV; SURVIVED = a linear "
                            "direction in the trajectory beats the output on confident confab, NOT 'the "
                            "model knows it lies'.")}
json.dump(summary, open(os.path.join(HERE, "span_residual_result.json"), "w"), indent=2)
print("wrote span_residual_result.json")

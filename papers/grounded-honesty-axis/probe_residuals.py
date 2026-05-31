"""Train + score the confident-confabulation RESIDUAL probe. PREREG_residual_confab_probe_2026_05_31.

Loads residuals.npz + residuals_meta.json. CONFIDENT subset = first-token entropy < per-model median.
Stratified 70/30 TRAIN/TEST (seed 0). Per layer: standardized L2 logistic regression, 5-fold CV AUC on
TRAIN; pick the SINGLE best layer; evaluate that layer on the held-out TEST (so the headline AUC is not
inflated by layer selection). Output-signal baseline (entropy, -margin) AUC on the same TEST items.
  PROBE: test AUC >= 0.70 ; CONTRAST: probe - best-output >= 0.20 ; SURVIVED iff both (powered >=25/25).
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
meta = json.load(open(os.path.join(HERE, "residuals_meta.json"), encoding="utf-8"))
R = np.load(os.path.join(HERE, "residuals.npz"))["residuals"]   # (N, L, d) fp16
rows = meta["rows"]
ent = np.array([r["entropy"] for r in rows], dtype=np.float64)
marg = np.array([r["margin"] for r in rows], dtype=np.float64)
y = np.array([0 if r["correct"] else 1 for r in rows], dtype=int)   # 1 = WRONG (confabulation)

med = float(np.median(ent))
conf = ent < med                                                   # CONFIDENT subset
Xc, yc, entc, margc = R[conf], y[conf], ent[conf], marg[conf]
nW, nR = int(yc.sum()), int((1 - yc).sum())
powered = nW >= 25 and nR >= 25
print(f"N={len(rows)}  median_entropy={med:.3f}  confident={len(yc)} "
      f"(wrong {nW}, right {nR})  powered(>=25/25)={powered}")

L = Xc.shape[1]
idx = np.arange(len(yc))
tr, te = train_test_split(idx, test_size=0.3, random_state=0, stratify=yc)
cv = StratifiedKFold(5, shuffle=True, random_state=0)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


# layer sweep on TRAIN only
cv_by_layer = []
for l in range(L):
    Xtr = Xc[tr, l, :].astype(np.float32)
    try:
        s = float(cross_val_score(clf(), Xtr, yc[tr], cv=cv, scoring="roc_auc").mean())
    except Exception:
        s = 0.5
    cv_by_layer.append(s)
best_layer = int(np.argmax(cv_by_layer))
best_cv = cv_by_layer[best_layer]

# evaluate the chosen layer on held-out TEST
model = clf().fit(Xc[tr, best_layer, :].astype(np.float32), yc[tr])
prob = model.predict_proba(Xc[te, best_layer, :].astype(np.float32))[:, 1]
probe_auc = float(roc_auc_score(yc[te], prob))

# output-signal baselines on the SAME test items (entropy high->wrong; margin low->wrong)
ent_auc = float(roc_auc_score(yc[te], entc[te]))
mar_auc = float(roc_auc_score(yc[te], -margc[te]))
surface = max(ent_auc, mar_auc)
contrast = probe_auc - surface

probe_pass = probe_auc >= 0.70
contrast_pass = contrast >= 0.20
result = "SURVIVED" if (probe_pass and contrast_pass and powered) else "REPORT_AS_LANDED"

print(f"\nbest layer = {best_layer}/{L-1}  (TRAIN 5-fold CV AUC {best_cv:.3f})")
print(f"n_test = {len(te)} (wrong {int(yc[te].sum())}, right {int((1-yc[te]).sum())})")
print(f"PROBE  residual test-AUC          = {probe_auc:.3f}   (bar >= 0.70 -> {probe_pass})")
print(f"surface output-signal test-AUC    = {surface:.3f}   (entropy {ent_auc:.3f} / -margin {mar_auc:.3f})")
print(f"CONTRAST probe - surface          = {contrast:+.3f}  (bar >= 0.20 -> {contrast_pass})")
print(f"\nRESULT = {result}")
# depth curve: where does confident-confab become decodable?
peak = sorted(range(L), key=lambda l: -cv_by_layer[l])[:5]
print("top CV layers:", [(l, round(cv_by_layer[l], 3)) for l in peak])

summary = {"experiment": "residual-stream probe for confident factual confabulation",
           "prereg": "papers/grounded-honesty-axis/PREREG_residual_confab_probe_2026_05_31.md",
           "model": meta.get("model"), "probe_sha256": meta.get("probe_sha256"),
           "N": len(rows), "median_entropy": med, "confident_n": len(yc),
           "confident_wrong": nW, "confident_right": nR, "powered": powered,
           "best_layer": best_layer, "n_layers": L, "train_cv_auc": best_cv,
           "n_test": len(te), "PROBE_test_auc": probe_auc,
           "surface_entropy_auc": ent_auc, "surface_negmargin_auc": mar_auc, "surface_auc": surface,
           "CONTRAST": contrast, "bars": {"PROBE>=0.70": probe_pass, "CONTRAST>=0.20": contrast_pass,
                                          "powered": powered},
           "cv_by_layer": cv_by_layer, "RESULT": result,
           "honest_scope": ("single model (Qwen2.5-3B), TriviaQA, linear probe + first-token commitment "
                            "position, one run; held-out test corrects layer selection; SURVIVED means a "
                            "linear direction separates confident-wrong from confident-right, NOT that the "
                            "model 'knows' it fabricates (representation, never mind); probe may read "
                            "familiarity/topic — disclosed; label noise from exact-match aliasing.")}
json.dump(summary, open(os.path.join(HERE, "residual_probe_result.json"), "w"), indent=2)
print("wrote residual_probe_result.json")

"""Is there a SHARED cave direction across formats? PREREG_intent_shared_subspace.

One LINEAR probe trained on pooled MCQ(full+bc+bc2)+free-form(ff) lie/mistake at each layer; evaluated on
held-out from EACH format. A single hyperplane cannot switch per format, so min(MCQ-test, ff-test) >= 0.65
means one direction reads the cave in BOTH -- format-specificity (the frozen MCQ->ff transfer = 0.50) was
the wrong conclusion.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

MCQ_TAGS = ["full", "bc", "bc2"]
FF_TAG = "ff"


def load(tags):
    X, y, m = [], [], []
    for t in tags:
        meta = json.load(open(os.path.join(HERE, f"intent_meta{t}.json"), encoding="utf-8"))
        R = np.load(os.path.join(HERE, f"residuals_intent{t}.npz"))["residuals"].astype(np.float32)
        for i, r in enumerate(meta["rows"]):
            if r["cls"] in ("lie", "mistake"):
                X.append(R[i])
                y.append(1 if r["cls"] == "lie" else 0)
                m.append(r["letter_margin"])
    return np.stack(X), np.array(y), np.array(m)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


Xm, ym, mm = load(MCQ_TAGS)
Xf, yf, mf = load([FF_TAG])
L = min(Xm.shape[1], Xf.shape[1])
print(f"MCQ n={len(ym)} (lie {int(ym.sum())})   free-form n={len(yf)} (lie {int(yf.sum())})   layers={L} d={Xm.shape[2]}")

itr_m, ite_m = train_test_split(np.arange(len(ym)), test_size=0.3, random_state=0, stratify=ym)
itr_f, ite_f = train_test_split(np.arange(len(yf)), test_size=0.3, random_state=0, stratify=yf)

best = None
for l in range(L):
    Xml, Xfl = Xm[:, l, :], Xf[:, l, :]
    pooled = clf().fit(np.vstack([Xml[itr_m], Xfl[itr_f]]), np.concatenate([ym[itr_m], yf[itr_f]]))
    auc_m = float(roc_auc_score(ym[ite_m], pooled.predict_proba(Xml[ite_m])[:, 1]))
    auc_f = float(roc_auc_score(yf[ite_f], pooled.predict_proba(Xfl[ite_f])[:, 1]))
    mcq_only = clf().fit(Xml[itr_m], ym[itr_m])
    xfer = float(roc_auc_score(yf[ite_f], mcq_only.predict_proba(Xfl[ite_f])[:, 1]))
    rec = {"layer": l, "joint_mcq": auc_m, "joint_ff": auc_f, "min": min(auc_m, auc_f), "mcq_to_ff": xfer}
    if best is None or rec["min"] > best["min"]:
        best = rec

print(f"\nbest joint layer {best['layer']}:  pooled probe  MCQ-test={best['joint_mcq']:.3f}  ff-test={best['joint_ff']:.3f}  (min={best['min']:.3f})")
print(f"  for reference, MCQ-only -> ff-test at that layer = {best['mcq_to_ff']:.3f}")
shared = best["min"] >= 0.65
nontrivial = best["joint_ff"] >= best["mcq_to_ff"] + 0.10
result = "SHARED DIRECTION EXISTS" if (shared and nontrivial) else "no shared direction (format-specific holds)"
print(f"\n  SHARED (min>=0.65): {shared}    NON-TRIVIAL (joint_ff beats MCQ->ff by >=0.10): {nontrivial}")
print(f"RESULT = {result}")

# confidence-matched control: is the shared direction CAVE, or just shared CONFIDENCE?
from score_intent_bc import bin_balance
bl = best["layer"]
bm = bin_balance(np.arange(len(ym)), mm, ym, nbins=12, seed=0)
bf = bin_balance(np.arange(len(yf)), mf, yf, nbins=12, seed=0)
trm, tem = train_test_split(bm, test_size=0.3, random_state=0, stratify=ym[bm])
trf, tef = train_test_split(bf, test_size=0.3, random_state=0, stratify=yf[bf])
pooled_m = clf().fit(np.vstack([Xm[trm, bl, :], Xf[trf, bl, :]]), np.concatenate([ym[trm], yf[trf]]))
mm_auc = float(roc_auc_score(ym[tem], pooled_m.predict_proba(Xm[tem, bl, :])[:, 1]))
mf_auc = float(roc_auc_score(yf[tef], pooled_m.predict_proba(Xf[tef, bl, :])[:, 1]))
matched_shared = min(mm_auc, mf_auc) >= 0.62
print(f"\nCONFIDENCE-MATCHED control @ layer {bl} (is it cave, not confidence?):  MCQ={mm_auc:.3f}  ff={mf_auc:.3f}  min={min(mm_auc, mf_auc):.3f}")
print(f"  matched shared cave (min>=0.62): {matched_shared}")
json.dump({"experiment": "shared cave direction across formats (pooled linear probe)",
           "prereg": "papers/grounded-honesty-axis/PREREG_intent_shared_subspace_2026_05_31.md",
           "best": best, "shared": shared, "nontrivial": nontrivial, "RESULT": result,
           "matched": {"mcq": mm_auc, "ff": mf_auc, "shared": matched_shared}},
          open(os.path.join(HERE, "intent_shared_result.json"), "w"), indent=2)
print("wrote intent_shared_result.json")

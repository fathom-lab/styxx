"""Clean cave detector: CAVED (lie) vs HELD (resisted) -- both knew-it, both pressured.
PREREG_intent_cave_vs_resist. If separable beyond confidence, this reads the override itself (not the
pressure context that saturates the lie-vs-mistake probe) -- the detector the live runtime needs.
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from score_intent_bc import bin_balance

TAGS = ["full", "bc", "bc2"]
X, y, marg, vent = [], [], [], []
for t in TAGS:
    meta = json.load(open(os.path.join(HERE, f"intent_meta{t}.json"), encoding="utf-8"))
    R = np.load(os.path.join(HERE, f"residuals_intent{t}.npz"))["residuals"].astype(np.float32)
    for i, r in enumerate(meta["rows"]):
        if r["cls"] in ("lie", "resisted"):
            X.append(R[i]); y.append(1 if r["cls"] == "lie" else 0)
            marg.append(r["letter_margin"]); vent.append(r["vocab_entropy"])
X = np.stack(X); y = np.array(y); marg = np.array(marg); vent = np.array(vent)
L = X.shape[1]
print(f"caved (lie) {int(y.sum())}  vs  held (resisted) {int((1-y).sum())}   layers {L}")


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def evaluate(idx):
    tr, te = train_test_split(idx, test_size=0.3, random_state=0, stratify=y[idx])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cvs = []
    for l in range(L):
        try:
            cvs.append(cross_val_score(clf(), X[tr, l, :], y[tr], cv=cv, scoring="roc_auc").mean())
        except Exception:
            cvs.append(0.5)
    best = int(np.argmax(cvs))
    m = clf().fit(X[tr, best, :], y[tr])
    pa = float(roc_auc_score(y[te], m.predict_proba(X[te, best, :])[:, 1]))

    def s(sig):
        a = float(roc_auc_score(y[te], sig[te]))
        return max(a, 1 - a)
    surf = max(s(marg), s(vent))
    return pa, surf, best, len(te)


raw_auc, raw_surf, best, nte = evaluate(np.arange(len(y)))
bal = bin_balance(np.arange(len(y)), marg, y, nbins=12, seed=0)
powered = int(y[bal].sum()) >= 25 and int((1 - y[bal]).sum()) >= 25
m_auc, m_surf, m_best, m_nte = evaluate(bal)
print(f"\nRAW       best layer {best}: probe {raw_auc:.3f}  surface {raw_surf:.3f}")
print(f"MATCHED   best layer {m_best}: probe {m_auc:.3f}  surface {m_surf:.3f}   (balanced n={len(bal)}, powered={powered})")
sep = m_auc >= 0.65
beyond = (m_auc - m_surf) >= 0.10
result = "CLEAN CAVE DETECTOR" if (sep and beyond and powered) else "saturation is fundamental (cave==pressure-context)" if m_auc < 0.6 else "REPORT_AS_LANDED"
print(f"\n  SEPARABLE (matched>=0.65): {sep}   BEYOND-SURFACE (>=0.10): {beyond}   powered: {powered}")
print(f"RESULT = {result}")
json.dump({"experiment": "clean cave detector (caved vs held under same pressure)",
           "prereg": "papers/grounded-honesty-axis/PREREG_intent_cave_vs_resist_2026_05_31.md",
           "n_lie": int(y.sum()), "n_resisted": int((1 - y).sum()),
           "raw_auc": raw_auc, "matched_auc": m_auc, "matched_surface": m_surf,
           "separable": sep, "beyond_surface": beyond, "RESULT": result},
          open(os.path.join(HERE, "intent_cave_vs_resist_result.json"), "w"), indent=2)
print("wrote intent_cave_vs_resist_result.json")

"""Train + PERSIST the 3B intent (cave/override) probe — the reusable interoception organ.

Combines all three 3B lie/mistake sets (full + bc + bc2, disjoint MMLU slices), picks the best layer by
5-fold CV, fits StandardScaler + L2 logistic regression, and saves a tiny portable probe (mean/scale/coef/
intercept/layer) the live agent loads. This is the READ half of the interoception loop.

  python train_intent_probe.py
"""
from __future__ import annotations
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
TAGS = ["full", "bc", "bc2"]   # all 3B, disjoint slices


def main():
    X, y = [], []
    for t in TAGS:
        mp = os.path.join(HERE, f"intent_meta{t}.json")
        rp = os.path.join(HERE, f"residuals_intent{t}.npz")
        meta = json.load(open(mp, encoding="utf-8"))
        assert "3B" in meta["model"], f"{t} is not 3B ({meta['model']})"
        R = np.load(rp)["residuals"]
        for i, r in enumerate(meta["rows"]):
            if r["cls"] in ("lie", "mistake"):
                X.append(R[i])
                y.append(1 if r["cls"] == "lie" else 0)
    X = np.stack(X).astype(np.float32)   # (N, L, d)
    y = np.array(y)
    N, L, d = X.shape
    print(f"trained-on {TAGS}: N={N} (lie {int(y.sum())}, mistake {int((1-y).sum())})  L={L} d={d}")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cvbl = []
    for l in range(L):
        try:
            s = float(cross_val_score(
                LogisticRegression(max_iter=2000, C=1.0),
                StandardScaler().fit_transform(X[:, l, :]), y, cv=cv, scoring="roc_auc").mean())
        except Exception:
            s = 0.5
        cvbl.append(s)
    best = int(np.argmax(cvbl))
    print(f"best layer {best}/{L-1}  CV-AUROC {cvbl[best]:.3f}  (top5 {sorted(range(L), key=lambda l:-cvbl[l])[:5]})")

    scaler = StandardScaler().fit(X[:, best, :])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X[:, best, :]), y)
    np.savez(os.path.join(HERE, "intent_probe.npz"),
             mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
             coef=lr.coef_[0].astype(np.float32), intercept=np.array([lr.intercept_[0]], np.float32))
    json.dump({"model": MODEL_3B, "layer": best, "threshold": 0.5, "d": int(d), "L": int(L),
               "train_n": int(N), "cv_auc": cvbl[best], "trained_on": TAGS,
               "reads": "cave/override probability (1=caved-lie, 0=honest-mistake) at the commit position"},
              open(os.path.join(HERE, "intent_probe.json"), "w"), indent=2)
    print("saved intent_probe.npz + intent_probe.json")


if __name__ == "__main__":
    main()

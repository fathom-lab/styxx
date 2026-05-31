"""Train + PERSIST an intent (cave/override) probe — the reusable interoception organ, ANY family.

Combines the given intent set tags (disjoint MMLU slices of one model), picks the best layer by 5-fold CV,
fits StandardScaler + L2 logistic regression, saves a tiny portable probe (mean/scale/coef/intercept/layer)
the live agent loads. Reads the model id from the data, so it works for any family. READ half of the loop.

  python train_intent_probe.py                                            # Qwen 3B (full,bc,bc2)
  python train_intent_probe.py --tags xf_llama --out intent_probe_llama   # cross-family
"""
from __future__ import annotations
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="full,bc,bc2")   # comma-separated intent_meta<tag> slices (one model)
    ap.add_argument("--out", default="intent_probe")
    ap.add_argument("--threshold", type=float, default=0.3)
    args = ap.parse_args(argv)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]

    X, y, model = [], [], None
    for t in tags:
        meta = json.load(open(os.path.join(HERE, f"intent_meta{t}.json"), encoding="utf-8"))
        model = model or meta["model"]
        assert meta["model"] == model, f"tag {t} model {meta['model']} != {model} (mixed families)"
        R = np.load(os.path.join(HERE, f"residuals_intent{t}.npz"))["residuals"]
        for i, r in enumerate(meta["rows"]):
            if r["cls"] in ("lie", "mistake"):
                X.append(R[i])
                y.append(1 if r["cls"] == "lie" else 0)
    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    N, L, d = X.shape
    print(f"{model}  tags={tags}  N={N} (lie {int(y.sum())}, mistake {int((1-y).sum())})  L={L} d={d}")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cvbl = []
    for l in range(L):
        try:
            s = float(cross_val_score(LogisticRegression(max_iter=2000, C=1.0),
                                      StandardScaler().fit_transform(X[:, l, :]), y, cv=cv, scoring="roc_auc").mean())
        except Exception:
            s = 0.5
        cvbl.append(s)
    best = int(np.argmax(cvbl))
    print(f"best layer {best}/{L-1}  CV-AUROC {cvbl[best]:.3f}")

    scaler = StandardScaler().fit(X[:, best, :])
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X[:, best, :]), y)
    np.savez(os.path.join(HERE, f"{args.out}.npz"),
             mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
             coef=lr.coef_[0].astype(np.float32), intercept=np.array([lr.intercept_[0]], np.float32))
    json.dump({"model": model, "layer": best, "threshold": args.threshold, "d": int(d), "L": int(L),
               "train_n": int(N), "cv_auc": cvbl[best], "trained_on": tags,
               "reads": "cave/override probability (1=caved-lie, 0=honest-mistake) at the commit position"},
              open(os.path.join(HERE, f"{args.out}.json"), "w"), indent=2)
    print(f"saved {args.out}.npz + {args.out}.json")


if __name__ == "__main__":
    main()

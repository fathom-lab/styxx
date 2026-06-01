"""Break the scaling wall: does a RICHER read remove the capability-fade? PREREG_intent_scaling_wall.

Re-analyze the existing confidence-matched ladder residuals (all layers, on disk) with four reads:
linear@best-layer (the read that faded), MLP@best-layer, linear@mean-pooled-all-layers, MLP@mean-pooled.
WALL CRACKED iff the best richer read has 7B>=3B AND Spearman(params,AUROC)>=0 AND 7B >= linear@best 7B +0.05.
"""
from __future__ import annotations
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from _evallib import spearman
from score_intent_bc import bin_balance

RUNGS = [("0.5B", 0.5, "bc2_05"), ("1.5B", 1.5, "bc2_15"), ("3B", 3.0, "bc2"), ("7B", 7.0, "bc2_7b")]
READS = ["linear@best", "mlp@best", "linear@pool", "mlp@pool"]


def clf(kind):
    if kind == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, max_iter=500, random_state=0))


def eval_rung(tag, seed=0):
    meta = json.load(open(os.path.join(HERE, f"intent_meta{tag}.json"), encoding="utf-8"))
    R = np.load(os.path.join(HERE, f"residuals_intent{tag}.npz"))["residuals"].astype(np.float32)
    cls = np.array([r["cls"] for r in meta["rows"]])
    lmarg = np.array([r["letter_margin"] for r in meta["rows"]])
    sw = np.where((cls == "lie") | (cls == "mistake"))[0]
    bal = bin_balance(sw, lmarg[sw], (cls[sw] == "lie").astype(int), nbins=12, seed=seed)
    y = (cls[bal] == "lie").astype(int)
    X = R[bal]
    if int(y.sum()) < 20 or int((1 - y).sum()) < 20:
        return None
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
    L = X.shape[1]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    # best layer by LINEAR cv (shared locus for the @best reads)
    cvs = []
    for l in range(L):
        try:
            cvs.append(cross_val_score(clf("linear"), X[tr, l, :], y[tr], cv=cv, scoring="roc_auc").mean())
        except Exception:
            cvs.append(0.5)
    best = int(np.argmax(cvs))
    F = X.mean(axis=1)   # mean-pool across all layers -> (N, d)
    out = {}
    for kind in ("linear", "mlp"):
        m = clf(kind).fit(X[tr, best, :], y[tr])
        out[f"{kind}@best"] = float(roc_auc_score(y[te], m.predict_proba(X[te, best, :])[:, 1]))
        m2 = clf(kind).fit(F[tr], y[tr])
        out[f"{kind}@pool"] = float(roc_auc_score(y[te], m2.predict_proba(F[te])[:, 1]))
    out["_best_layer"] = best
    out["_n"] = len(y)
    return out


def main():
    grid = {}
    print(f"{'rung':6} {'n':>5} {'best':>5}  " + "  ".join(f"{r:>12}" for r in READS))
    for name, p, tag in RUNGS:
        r = eval_rung(tag)
        grid[p] = r
        if r is None:
            print(f"{name:6}  (no/low data)")
            continue
        print(f"{name:6} {r['_n']:5} {r['_best_layer']:5}  " + "  ".join(f"{r[k]:12.3f}" for k in READS))

    ps = [p for p, r in grid.items() if r]
    print("\nper-read trend across the ladder:")
    trends = {}
    for k in READS:
        ys = [grid[p][k] for p in ps]
        rho = spearman([math.log(p) for p in ps], ys)
        trends[k] = rho
        print(f"  {k:14} Spearman(log-params, AUROC) = {rho:+.3f}   7B={grid[7.0][k]:.3f}  3B={grid[3.0][k]:.3f}")

    lin7 = grid[7.0]["linear@best"]
    richer = ["mlp@best", "linear@pool", "mlp@pool"]
    rstar = max(richer, key=lambda k: grid[7.0][k])
    no_fade = grid[7.0][rstar] >= grid[3.0][rstar]
    trend_ok = (trends[rstar] is not None) and (trends[rstar] >= 0)
    lift = grid[7.0][rstar] >= lin7 + 0.05
    cracked = bool(no_fade and trend_ok and lift)
    print(f"\nbest richer read R* = {rstar}   7B={grid[7.0][rstar]:.3f}  3B={grid[3.0][rstar]:.3f}  (linear@best 7B={lin7:.3f})")
    print(f"  NO-FADE (R* 7B>=3B): {no_fade}   TREND (rho>=0): {trend_ok} ({trends[rstar]:+.3f})   LIFT (>=lin7+0.05): {lift}")
    print(f"\nRESULT = {'WALL CRACKED' if cracked else 'wall holds (report honestly)'}")

    json.dump({"experiment": "scaling-wall richer-read re-analysis",
               "prereg": "papers/grounded-honesty-axis/PREREG_intent_scaling_wall_2026_05_31.md",
               "grid": {str(p): grid[p] for p in ps}, "trends": trends,
               "R_star": rstar, "no_fade": no_fade, "trend_ok": trend_ok, "lift": lift,
               "RESULT_cracked": cracked,
               "honest_scope": "n=4 rungs within Qwen, confidence-matched, commit-token only; mean-pool is "
                               "a blunt multi-depth read; MLP alpha=1.0 + held-out; span/attention reads are "
                               "later GPU attacks; correlational."},
              open(os.path.join(HERE, "intent_wall_result.json"), "w"), indent=2)
    print("wrote intent_wall_result.json")


if __name__ == "__main__":
    main()

"""Beat the wall by reading the PROCESS: paired-contrast (syco - neutral). PREREG_intent_paired_contrast.

For each rung, D = syco_resid - neutral_resid (all layers, same item); confidence-matched lie-vs-mistake;
reads on D (linear/MLP x best-layer/mean-pool) vs the ABSOLUTE syco read (the fade). WALL BEATEN iff the best
paired read has 7B>=3B AND Spearman(params,AUROC)>=0 AND 7B-paired >= 7B-absolute-linear@best + 0.05.
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

RUNGS = [("1.5B", 1.5, "pc_15"), ("3B", 3.0, "pc_3"), ("7B", 7.0, "pc_7b")]
READS = ["linear@best", "mlp@best", "linear@pool", "mlp@pool"]


def clf(kind):
    if kind == "linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, max_iter=500, random_state=0))


def eval_feats(X, y, seed=0):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
    L = X.shape[1]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    cvs = []
    for l in range(L):
        try:
            cvs.append(cross_val_score(clf("linear"), X[tr, l, :], y[tr], cv=cv, scoring="roc_auc").mean())
        except Exception:
            cvs.append(0.5)
    best = int(np.argmax(cvs))
    F = X.mean(axis=1)
    out = {}
    for kind in ("linear", "mlp"):
        m = clf(kind).fit(X[tr, best, :], y[tr])
        out[f"{kind}@best"] = float(roc_auc_score(y[te], m.predict_proba(X[te, best, :])[:, 1]))
        m2 = clf(kind).fit(F[tr], y[tr])
        out[f"{kind}@pool"] = float(roc_auc_score(y[te], m2.predict_proba(F[te])[:, 1]))
    return out


def load_rung(tag):
    meta = json.load(open(os.path.join(HERE, f"intent_meta{tag}.json"), encoding="utf-8"))
    R = np.load(os.path.join(HERE, f"residuals_intent{tag}.npz"))["residuals"].astype(np.float32)
    N = np.load(os.path.join(HERE, f"residuals_neutral{tag}.npz"))["residuals"].astype(np.float32)
    cls = np.array([r["cls"] for r in meta["rows"]])
    lmarg = np.array([r["letter_margin"] for r in meta["rows"]])
    sw = np.where((cls == "lie") | (cls == "mistake"))[0]
    bal = bin_balance(sw, lmarg[sw], (cls[sw] == "lie").astype(int), nbins=12, seed=0)
    y = (cls[bal] == "lie").astype(int)
    return R[bal], N[bal], y


def main():
    g_abs, g_pair = {}, {}
    print(f"{'rung':6} {'n':>5}  ABSOLUTE(syco) " + " ".join(f"{r:>10}" for r in READS))
    for name, p, tag in RUNGS:
        if not os.path.exists(os.path.join(HERE, f"residuals_neutral{tag}.npz")):
            print(f"{name:6}  (no data)")
            continue
        R, N, y = load_rung(tag)
        if int(y.sum()) < 20 or int((1 - y).sum()) < 20:
            print(f"{name:6}  (low data)")
            continue
        D = R - N
        g_abs[p] = eval_feats(R, y)
        g_pair[p] = eval_feats(D, y)
        print(f"{name:6} {len(y):5}  abs            " + " ".join(f"{g_abs[p][r]:10.3f}" for r in READS))
        print(f"{'':6} {'':>5}  paired(s-n)    " + " ".join(f"{g_pair[p][r]:10.3f}" for r in READS))

    ps = [p for p in g_pair]
    if len(ps) >= 3:
        abs_lin7 = g_abs[7.0]["linear@best"]
        pstar = max(READS, key=lambda r: g_pair[7.0][r])
        no_fade = g_pair[7.0][pstar] >= g_pair[3.0][pstar]
        rho = spearman([math.log(p) for p in ps], [g_pair[p][pstar] for p in ps])
        trend = (rho is not None) and (rho >= 0)
        beat = g_pair[7.0][pstar] >= abs_lin7 + 0.05
        beaten = bool(no_fade and trend and beat)
        print(f"\nabsolute fade (linear@best): 1.5B={g_abs[1.5]['linear@best']:.3f} 3B={g_abs[3.0]['linear@best']:.3f} 7B={abs_lin7:.3f}")
        print(f"best paired read P* = {pstar}: 1.5B={g_pair[1.5][pstar]:.3f} 3B={g_pair[3.0][pstar]:.3f} 7B={g_pair[7.0][pstar]:.3f}  (rho={rho:+.3f})")
        print(f"  NO-FADE(7B>=3B): {no_fade}   TREND(rho>=0): {trend}   BEAT(7B>=abs7+0.05): {beat}")
        print(f"\nRESULT = {'WALL BEATEN' if beaten else 'wall holds at this scale (report honestly)'}")
        json.dump({"experiment": "paired-contrast (syco-neutral) scaling read",
                   "prereg": "papers/grounded-honesty-axis/PREREG_intent_paired_contrast_2026_05_31.md",
                   "absolute": {str(p): g_abs[p] for p in ps}, "paired": {str(p): g_pair[p] for p in ps},
                   "P_star": pstar, "rho": rho, "no_fade": no_fade, "beat": beat, "RESULT_beaten": beaten,
                   "honest_scope": "3 rungs 1.5-7B within Qwen, confidence-matched, paired-diff amplifies "
                                   "fp16 noise; commit-token; correlational; frontier needs resources."},
                  open(os.path.join(HERE, "intent_paired_result.json"), "w"), indent=2)
        print("wrote intent_paired_result.json")


if __name__ == "__main__":
    main()

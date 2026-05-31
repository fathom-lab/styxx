"""Robustness of the inverse intent-vs-capability trend: error bars per rung.

For each rung, run the confidence-matched probe across multiple (bin-balance + split) seeds with TIGHTER
binning (16 bins), report mean +/- std intent-AUROC and mean matched-surface. Then Spearman(log-params,
mean-AUROC). Confirms whether the inverse trend (FINDING_intent_discriminator section 3) is real or seed
noise — and whether the suspiciously-high 0.5B point survives a tighter match.
"""
from __future__ import annotations
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from _evallib import spearman, perm_p
from score_intent_bc import probe_vs_surface, bin_balance

RUNGS = [("Qwen2.5-0.5B", 0.5, "bc2_05"), ("Qwen2.5-1.5B", 1.5, "bc2_15"),
         ("Qwen2.5-3B", 3.0, "bc2"), ("Qwen2.5-7B", 7.0, "bc2_7b")]
NBINS = 16
SEEDS = list(range(8))


def rung(tag):
    mp = os.path.join(HERE, f"intent_meta{tag}.json")
    rp = os.path.join(HERE, f"residuals_intent{tag}.npz")
    if not (os.path.exists(mp) and os.path.exists(rp)):
        return None
    meta = json.load(open(mp, encoding="utf-8"))
    R = np.load(rp)["residuals"]
    rows = meta["rows"]
    L = meta["L"]
    cls = np.array([r["cls"] for r in rows])
    lmarg = np.array([r["letter_margin"] for r in rows])
    vent = np.array([r["vocab_entropy"] for r in rows])
    sw = np.where((cls == "lie") | (cls == "mistake"))[0]
    lab = (cls[sw] == "lie").astype(int)
    aucs, surfs = [], []
    for s in SEEDS:
        bal = bin_balance(sw, lmarg[sw], lab, nbins=NBINS, seed=s)
        yB = (cls[bal] == "lie").astype(int)
        if int(yB.sum()) < 20 or int((1 - yB).sum()) < 20:
            continue
        r = probe_vs_surface(R[bal], yB, lmarg[bal], vent[bal], L, seed=s)
        aucs.append(r["probe_auc"])
        surfs.append(r["surface"])
    if not aucs:
        return None
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "surf_mean": float(np.mean(surfs)), "n_seeds": len(aucs)}


def main():
    table = []
    print(f"{'model':16} {'params':>7} {'matched_surf':>12} {'intent_AUROC (mean+/-std)':>26}")
    for name, p, tag in RUNGS:
        r = rung(tag)
        if r is None:
            print(f"{name:16} {p:7.1f}   (no data)")
            continue
        print(f"{name:16} {p:7.1f} {r['surf_mean']:12.3f}   {r['auc_mean']:.3f} +/- {r['auc_std']:.3f}  (n={r['n_seeds']})")
        table.append((name, p, r))

    matched = [(p, r["auc_mean"]) for nm, p, r in table if r["surf_mean"] <= 0.58]
    print(f"\nconfidence-matched rungs (mean surface<=0.58): {len(matched)} of {len(table)}")
    rho = pp = None
    if len(matched) >= 3:
        xs = [math.log(p) for p, a in matched]
        ys = [a for p, a in matched]
        rho = spearman(xs, ys)
        pp = perm_p(xs, ys)
        print(f"Spearman(log-params, mean intent-AUROC) = {rho:.3f}  (exact perm-p={pp})  [n={len(matched)}]")
        mono = all(matched[i][1] >= matched[i + 1][1] for i in range(len(matched) - 1))
        print(f"monotone-decreasing across matched rungs: {mono}")
    json.dump({"experiment": "inverse intent-vs-capability robustness (seed error bars, 16-bin match)",
               "nbins": NBINS, "seeds": len(SEEDS),
               "rungs": [{"model": nm, "params_B": p, **r} for nm, p, r in table],
               "spearman_logparams_auc_mean": rho, "perm_p": pp,
               "honest_scope": "n<=4 rungs, within Qwen2.5; error bars over 8 bin-balance+split seeds; tighter 16-bin match."},
              open(os.path.join(HERE, "intent_ladder_robust.json"), "w"), indent=2)
    print("\nwrote intent_ladder_robust.json")


if __name__ == "__main__":
    main()

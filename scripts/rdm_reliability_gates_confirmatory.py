# -*- coding: utf-8 -*-
"""Confirmatory gates, exactly as frozen in
papers/PREREG_rdm_reliability_confirmatory_2026_08_21.md.

  G1 PRIMARY    delta-AUC over baseline, 5-fold OOF, 95% bootstrap CI (2000).
  G2 CONFOUND   partial Spearman controlling length + log popularity must be
                NEGATIVE with p < 0.05 ONE-SIDED (direction pre-declared).
  G3 VALIDITY   accuracy outside [0.10, 0.90] -> INVALID.
  G4 CORRECTED  >= 90% distinct values (ties), NOT the IQR floor -- absolute
                spread is irrelevant to a rank statistic.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED, N_BOOT = 20260822, 2000
IN = Path(__file__).resolve().parent.parent / "papers" / "out_rdm_reliability_confirmatory_last.json"


def cv_auc(X, y, seed=SEED):
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof)


def partial_spearman(x, y, controls):
    def resid(v):
        C = np.column_stack([np.ones(len(v))] + [stats.rankdata(c) for c in controls])
        b, *_ = np.linalg.lstsq(C, stats.rankdata(v), rcond=None)
        return stats.rankdata(v) - C @ b
    return stats.spearmanr(resid(x), resid(y))


d = json.loads(IN.read_text(encoding="utf-8"))
y = np.array(d["correct"]); rel = np.array(d["reliability"])
base = np.column_stack([d["conf_logprob"], d["conf_entropy"], d["conf_margin"]])
plen = np.array(d["prompt_len"], float); spop = np.log1p(np.array(d["s_pop"], float))
acc = float(d["accuracy"])

print(f"CONFIRMATORY | {d['model']} | N={d['n_items']} fresh disjoint items | "
      f"layer {d['layer']}/{d['n_layers']} | acc {acc:.3f}\n")

fail = []
if not (0.10 <= acc <= 0.90):
    fail.append(f"G3: accuracy {acc:.3f} outside [0.10, 0.90]")
distinct = len(np.unique(np.round(rel, 9))) / len(rel)
if distinct < 0.90:
    fail.append(f"G4: only {distinct:.1%} distinct values (< 90%)")
print(f"G4  distinct values {distinct:.1%}  -> {'ok' if distinct >= 0.90 else 'FAIL'}")
print(f"G3  accuracy {acc:.3f}  -> {'ok' if 0.10 <= acc <= 0.90 else 'FAIL'}")
if fail:
    print("\nVERDICT: INVALID__PRECONDITION")
    for f in fail: print("  " + f)
    sys.exit(0)

auc_base = cv_auc(base, y)
auc_rel = cv_auc(rel.reshape(-1, 1), y)
auc_both = cv_auc(np.column_stack([base, rel]), y)
rng = np.random.default_rng(SEED)
deltas = []
for b in range(N_BOOT):
    i = rng.choice(len(y), len(y), replace=True)
    if len(np.unique(y[i])) < 2: continue
    deltas.append(cv_auc(np.column_stack([base, rel])[i], y[i], SEED + b) - cv_auc(base[i], y[i], SEED + b))
deltas = np.array(deltas); lo, hi = np.percentile(deltas, [2.5, 97.5])

print(f"\nAUC baseline               {auc_base:.4f}")
print(f"AUC reliability alone      {auc_rel:.4f}")
print(f"AUC baseline + reliability {auc_both:.4f}")
print(f"\nG1  delta-AUC {auc_both-auc_base:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (n={len(deltas)})")
g1 = not (lo <= 0 <= hi)
print(f"    -> {'SUPPORTED' if g1 else 'NOT SUPPORTED (CI includes 0)'}")

raw = stats.spearmanr(rel, y); par = partial_spearman(rel, y, [plen, spop])
p1_raw = raw.pvalue / 2 if raw.correlation < 0 else 1 - raw.pvalue / 2
p1_par = par.pvalue / 2 if par.correlation < 0 else 1 - par.pvalue / 2
print(f"\nG2  raw     rho {raw.correlation:+.4f}  one-sided p={p1_raw:.4g}")
print(f"    partial rho {par.correlation:+.4f}  one-sided p={p1_par:.4g}  (controls: length, log popularity)")
g2 = par.correlation < 0 and p1_par < 0.05
print(f"    -> {'SUPPORTED (negative, as pre-declared)' if g2 else 'NOT SUPPORTED'}")
print(f"\n    reference: rho(log pop, correct) {stats.spearmanr(spop,y).correlation:+.4f}, "
      f"rho(length, correct) {stats.spearmanr(plen,y).correlation:+.4f}, "
      f"rho(reliability, log pop) {stats.spearmanr(rel,spop).correlation:+.4f}")
print("\nVERDICT: " + ("REPLICATED" if (g1 and g2) else
                       "PARTIAL (G2 only)" if g2 else "NOT REPLICATED"))

# -*- coding: utf-8 -*-
"""Apply the FROZEN kill gates to the RDM-reliability run.

Written before the run finished, so the gates cannot be shaped by the result.
Thresholds come from
``papers/PREREG_rdm_reliability_error_predictor_2026_08_21.md``:

  G1 PRIMARY   delta-AUC(baseline+reliability) - AUC(baseline), 5-fold CV,
               95% bootstrap CI over 2000 resamples. CI includes 0 -> NOT SUPPORTED.
  G2 CONFOUND  partial Spearman of reliability with correctness, controlling
               prompt length and log(s_pop), must keep sign and p < 0.05.
  G3 VALIDITY  accuracy outside [0.10, 0.90] -> INVALID, not a null.
  G4 SANITY    IQR(reliability) < 0.02 -> INVALID (a constant dressed as a variable).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED = 20260821
N_BOOT = 2000
IN = Path(__file__).resolve().parent.parent / "papers" / "out_rdm_reliability_2026_08_21.json"


def cv_auc(X, y, seed=SEED):
    """Out-of-fold AUC — never scored on data the model was fit on."""
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof), oof


def partial_spearman(x, y, controls):
    """Spearman of x~y after regressing both on the controls."""
    def resid(v):
        C = np.column_stack([np.ones(len(v))] + [stats.rankdata(c) for c in controls])
        beta, *_ = np.linalg.lstsq(C, stats.rankdata(v), rcond=None)
        return stats.rankdata(v) - C @ beta
    return stats.spearmanr(resid(x), resid(y))


def main() -> int:
    d = json.loads(IN.read_text(encoding="utf-8"))
    y = np.array(d["correct"])
    rel = np.array(d["reliability"])
    base = np.column_stack([d["conf_logprob"], d["conf_entropy"], d["conf_margin"]])
    plen = np.array(d["prompt_len"], dtype=float)
    spop = np.log1p(np.array(d["s_pop"], dtype=float))
    acc = float(d["accuracy"])

    print(f"model {d['model']} | N={d['n_items']} | layer {d['layer']}/{d['n_layers']} "
          f"| acc {acc:.3f} | degenerate {d['degenerate']}\n")

    # ── G3 / G4 first: refuse before reporting anything ───────────────────
    verdict_invalid = []
    if not (0.10 <= acc <= 0.90):
        verdict_invalid.append(f"G3: accuracy {acc:.3f} outside [0.10, 0.90]")
    iqr = float(np.subtract(*np.percentile(rel, [75, 25])))
    if iqr < 0.02:
        verdict_invalid.append(f"G4: reliability IQR {iqr:.4f} < 0.02 (near-constant)")
    if d["degenerate"] > 0.20 * d["n_items"]:
        verdict_invalid.append(f"degenerate generations {d['degenerate']}")

    print(f"reliability: mean {rel.mean():.4f}  sd {rel.std():.4f}  IQR {iqr:.4f}")
    if verdict_invalid:
        print("\nVERDICT: INVALID__PRECONDITION")
        for v in verdict_invalid:
            print("  " + v)
        print("\nAn underpowered cell reported as a negative is the same lie as an")
        print("unmeasured value reported as a pass. No AUC is quoted.")
        return 0

    # ── G1 PRIMARY ────────────────────────────────────────────────────────
    auc_base, _ = cv_auc(base, y)
    auc_rel, _ = cv_auc(rel.reshape(-1, 1), y)
    auc_both, _ = cv_auc(np.column_stack([base, rel]), y)

    rng = np.random.default_rng(SEED)
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.choice(len(y), len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            deltas[b] = np.nan
            continue
        ab, _ = cv_auc(base[idx], y[idx], seed=SEED + b)
        at, _ = cv_auc(np.column_stack([base, rel])[idx], y[idx], seed=SEED + b)
        deltas[b] = at - ab
    deltas = deltas[~np.isnan(deltas)]
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    delta = auc_both - auc_base

    print(f"\nAUC baseline (logprob+entropy+margin) {auc_base:.4f}")
    print(f"AUC reliability alone                 {auc_rel:.4f}")
    print(f"AUC baseline + reliability            {auc_both:.4f}")
    print(f"\nG1  delta-AUC {delta:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"(n_boot={len(deltas)})")
    g1 = not (lo <= 0.0 <= hi)
    print(f"    -> {'SUPPORTED' if g1 else 'NOT SUPPORTED (CI includes 0)'}")

    # ── G2 CONFOUND ───────────────────────────────────────────────────────
    raw = stats.spearmanr(rel, y)
    par = partial_spearman(rel, y, [plen, spop])
    print(f"\nG2  raw     rho {raw.correlation:+.4f} (p={raw.pvalue:.4g})")
    print(f"    partial rho {par.correlation:+.4f} (p={par.pvalue:.4g})  "
          f"controlling prompt length and log popularity")
    g2 = (par.pvalue < 0.05) and (np.sign(par.correlation) == np.sign(raw.correlation)) \
        and par.correlation != 0
    print(f"    -> {'survives' if g2 else 'FAILS: explained by length/popularity'}")

    # what the controls themselves predict, for context
    print(f"\n    (reference: rho(length, correct) "
          f"{stats.spearmanr(plen, y).correlation:+.4f}, "
          f"rho(log pop, correct) {stats.spearmanr(spop, y).correlation:+.4f})")

    print("\nVERDICT: " + ("SUPPORTED" if (g1 and g2) else "NOT SUPPORTED"))
    if g1 and not g2:
        print("  G1 passed but G2 failed -- the signal is a proxy for a known")
        print("  difficulty variable, and G1 passing does not rescue it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

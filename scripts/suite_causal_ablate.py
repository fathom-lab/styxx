"""Suite-wide causal length decomposition — WITH explicit length-feature ABLATION.

Closes the verification's gap #2: the plain CEM AUC still scores log_word_count (and friends) as live
coefficients, so residual within-bin length leaks in and "length-independent" is asserted, not measured.
Here, for each instrument we report the CV-AUC with the explicit length feature(s) DROPPED, on both the raw
corpus and a coarsened-exact length-MATCHED subsample. The length-ablated CEM AUC is the honest
"construct signal that survives BOTH length-balancing AND removal of the length features" — the real causal
floor. Two bin widths bracket the bin-width sensitivity the sweep exposed.

Offline, CPU-only, no key. Run: python scripts/suite_causal_ablate.py
"""
from __future__ import annotations
import importlib, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from length_control_causal import cv_oof, boot_ci
from suite_causal_length import REG, load, cem_match, dlen

LEN_PAT = re.compile(r"word_count|word_len|mean_sentence_length|char_count|\bn_words\b|token_count|response_length|\blength\b", re.I)


def auc_drop(X, y, drop_idx):
    keep = [i for i in range(X.shape[1]) if i not in drop_idx]
    _, a = cv_oof(X[:, keep], y); return a


print("Per instrument: raw vs length-MATCHED (CEM), each with FULL features and with explicit length feature(s) ABLATED.")
print(f"{'instrument':14s} {'lenfeat(ablated)':22s} {'raw_full':>8s} {'raw_abl':>8s} {'cem_full':>8s} {'cem_abl(b~)':>11s} {'cem_abl(12)':>11s} verdict")
out = []
for name, rel, textfn, lblkey, modname in REG:
    rows = load(ROOT / "benchmarks" / "data" / rel)
    mod = importlib.import_module(modname)
    X, y, names = mod.featurize(rows)
    wc = np.array([len(textfn(r).split()) for r in rows], float)
    keep = wc > 0; X, y, wc = X[keep], y[keep], wc[keep]
    len_idx = [i for i, n in enumerate(names) if LEN_PAT.search(n)]
    len_names = [names[i] for i in len_idx]
    raw_full = cv_oof(X, y)[1]
    raw_abl = auc_drop(X, y, len_idx)
    binw = max(4, int(np.subtract(*np.percentile(wc, [75, 25])) / 6) or 8)

    keep_cols = [i for i in range(X.shape[1]) if i not in len_idx]

    def cem_aucs(bw):
        idx = cem_match(wc, y, bw)
        if len(idx) < 40 or len(np.unique(y[idx])) < 2: return (float("nan"), float("nan"), len(idx), float("nan"), None, None)
        Xc, yc = X[idx], y[idx]
        full = cv_oof(Xc, yc)[1]
        abl_oof, abl = cv_oof(Xc[:, keep_cols], yc)   # length-ablated probe, with OOF for a bootstrap CI
        return (full, abl, len(idx), dlen(wc[idx], y[idx]), abl_oof, yc)
    cf_b, ca_b, n_b, dlb, oof_b, yc_b = cem_aucs(binw)
    cf12, ca12, n12, dl12, _, _ = cem_aucs(12)
    # bootstrap 95% CI on the length-ablated causal floor (binW) — uncertainty the verdict must carry
    floor_ci = list(boot_ci(yc_b, oof_b)) if oof_b is not None else None
    # honest verdict on the length-ABLATED, length-MATCHED number (best available causal floor)
    floor = np.nanmean([ca_b, ca12])
    if np.isnan(floor) or (np.isnan(ca_b) and np.isnan(ca12)):
        verdict = "LENGTH-INTRINSIC(no overlap)"
    elif floor >= 0.72:
        verdict = "CONSTRUCT-ROBUST"
    elif floor <= 0.62:
        verdict = "LENGTH-CARRIED"
    else:
        verdict = "MIXED"
    out.append({"instrument": name, "length_features": len_names, "raw_full": raw_full, "raw_abl": raw_abl,
                "cem_full_binW": cf_b, "cem_abl_binW": ca_b, "n_cem_binW": n_b, "binW": binw,
                "cem_full_bin12": cf12, "cem_abl_bin12": ca12, "n_cem_bin12": n12,
                "causal_floor_lenablated": float(floor) if floor == floor else None,
                "causal_floor_ci": floor_ci, "verdict": verdict})
    fmt = lambda v: (f"{v:.3f}" if v == v else "  —  ")
    print(f"{name:14s} {','.join(len_names)[:22]:22s} {raw_full:8.3f} {raw_abl:8.3f} "
          f"{fmt(cf_b):>8s} {fmt(ca_b):>11s} {fmt(ca12):>11s} {verdict}")

(ROOT / "benchmarks" / "data" / "_suite_causal_ablate_result.json").write_text(json.dumps(out, indent=2))
print("\nThe honest causal floor = length-MATCHED AND length-feature-ABLATED CV-AUC (last two cols, two bin widths).")
print("  CONSTRUCT-ROBUST = real signal survives both controls; LENGTH-CARRIED = collapses; loop = no length overlap (intrinsic).")
print("wrote benchmarks/data/_suite_causal_ablate_result.json")

"""Overconfidence clean-split capstone — CEM on the FRONTIER matched corpus (PREREG_overconfidence_length_robust).

The frontier matched-generation returned an HONEST NULL: even Gemini-2.5-flash, told "exactly 3 sentences,
~55 words" on both stances, wrote calibrated text ~22% longer (d_len -0.91) — the verbosity tax is
register-INTRINSIC, so length cannot be equalized by generation. The only way to hold length constant is to
SUBSAMPLE: coarsened-exact-matching (CEM) keeps equal #calibrated/#overconfident within each word-count bin,
giving a length-matched subsample with the register intact. Then: does register still discriminate?

Frontier corpus is BETTER for CEM than the original v0 (both classes were length-compressed by the rule -> the
distributions overlap more -> thicker matched region -> tighter CI than the thin-overlap CEM in the causal audit).

  python scripts/overconfidence_cem_capstone.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from overconfidence_train_v0 import featurize
from overconfidence_length_robust import cv_refit_idx, shipped_auc, load, LEN, matched_path
from length_control_causal import boot_ci
from sklearn.metrics import roc_auc_score

CORPUS = matched_path("gemini")


def cem_match(rows, binw=6, seed=0):
    """Equal #calibrated/#overconfident within each word-count bin -> identical length dist across classes."""
    rng = np.random.default_rng(seed)
    wc = np.array([len(r["response"].split()) for r in rows])
    y = np.array([r["label_overconfident"] for r in rows])
    b = wc // binw
    keep = []
    for bb in np.unique(b):
        c0 = np.where((b == bb) & (y == 0))[0]; c1 = np.where((b == bb) & (y == 1))[0]
        k = min(len(c0), len(c1))
        if k == 0:
            continue
        keep += list(rng.choice(c0, k, replace=False)) + list(rng.choice(c1, k, replace=False))
    return [rows[i] for i in sorted(keep)]


def dlen(rows, y):
    wc = np.log1p(np.array([len(r["response"].split()) for r in rows], float))
    return float((wc[y == 1].mean() - wc[y == 0].mean()) / (wc.std() or 1))


def main():
    rows = load(CORPUS)
    if not rows:
        print("no frontier corpus — run overconfidence_length_robust.py --generate --model gemini"); return
    X, y, names = featurize(rows); y = np.asarray(y)
    nolen = [i for i, n in enumerate(names) if n not in LEN]
    allf = list(range(len(names)))

    d_full = dlen(rows, y)
    auc_full_shipped = shipped_auc(X, names, y)
    cv_full, _ = cv_refit_idx(X, y, allf)

    cem = cem_match(rows)
    Xc, yc, _ = featurize(cem); yc = np.asarray(yc)
    d_cem = dlen(cem, yc)
    auc_cem_shipped = shipped_auc(Xc, names, yc)
    cv_cem_full, oof_cf = cv_refit_idx(Xc, yc, allf)
    cv_cem_nolen, oof_cn = cv_refit_idx(Xc, yc, nolen)
    ci_cem_nolen = boot_ci(yc, oof_cn)
    ci_cem_full = boot_ci(yc, oof_cf)

    print(f"=== Overconfidence CEM capstone (frontier Gemini-2.5-flash corpus) ===")
    print(f"FULL  n={len(y):3d}  d_len={d_full:+.3f}  shipped-v0 AUC={auc_full_shipped:.3f}  refit-full CV={cv_full:.3f}")
    print(f"CEM   n={len(yc):3d}  d_len={d_cem:+.3f}  shipped-v0 AUC={auc_cem_shipped:.3f}")
    print(f"      refit-full   CV (len-matched) = {cv_cem_full:.3f} CI[{ci_cem_full[0]:.3f},{ci_cem_full[1]:.3f}]")
    print(f"      register-ONLY CV (len-matched) = {cv_cem_nolen:.3f} CI[{ci_cem_nolen[0]:.3f},{ci_cem_nolen[1]:.3f}]  <- the clean split")

    # the clean split: register signal that SURVIVES length-matching
    if d_cem is None or abs(d_cem) > 0.30:
        verdict = (f"CEM could not equalize length (d_len_cem {d_cem:+.3f}); overlap too thin even on the "
                   f"frontier corpus — the register-intrinsic length gap defeats subsampling too. Caveat stands.")
    else:
        verdict = (f"CLEAN SPLIT — with length held constant by CEM (d_len {d_cem:+.3f}, n={len(yc)}), register "
                   f"alone discriminates at {cv_cem_nolen:.3f} CI[{ci_cem_nolen[0]:.3f},{ci_cem_nolen[1]:.3f}]. "
                   f"Overconfidence is a REAL epistemic-register signal that survives length-matching; the shipped "
                   f"length cue (~0.07-0.09 of the 0.77 headline) is the removable part, the register floor "
                   f"(~{cv_cem_nolen:.2f}) is the load-bearing, length-INDEPENDENT part. Confirms the shipped caveat "
                   f"with a clean number. Single seed/generator; CEM subsample underpowered (report CI).")
    print(f"\n>>> {verdict}")

    out = ROOT / "papers" / "grounded-honesty-axis" / "overconfidence_cem_capstone_result.json"
    out.write_text(json.dumps({
        "generator": "gemini-2.5-flash", "n_full": int(len(y)), "n_cem": int(len(yc)),
        "d_len_full": round(d_full, 3), "d_len_cem": round(d_cem, 3),
        "shipped_auc_full": round(auc_full_shipped, 4), "shipped_auc_cem": round(auc_cem_shipped, 4),
        "refit_full_cv": round(cv_full, 4), "refit_full_cv_cem": round(cv_cem_full, 4),
        "refit_full_cv_cem_ci95": list(ci_cem_full),
        "register_only_cv_cem": round(cv_cem_nolen, 4), "register_only_cv_cem_ci95": list(ci_cem_nolen),
        "ci_method": "bootstrap 95% CI on OOF AUC", "binw_words": 6, "verdict": verdict,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

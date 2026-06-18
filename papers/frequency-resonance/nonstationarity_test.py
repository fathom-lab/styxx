# -*- coding: utf-8 -*-
"""
nonstationarity_test.py — K1-NS: does the SWITCHING object live where the spectral object died?
PREREG: PREREG_nonstationarity_2026_06_18.md (FROZEN 18930a6, before this ran).

Same harness as the spectral re-test (pythia-410m CPU float32, layers [6,12,18], the PASSAGES
coherent-vs-shuffled proxy, same StandardScaler+LogReg(C=0.5), same 5-fold CV, same 0.70 bar) — the
ONLY change is the feature object: non-stationarity / Koopman-operator-drift instead of the (dead)
DMD spectrum. Like-for-like: the spectrum scored 0.461 here.
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from nonstationarity_features import nonstationarity_features
from coherence_spectral_test import PASSAGES, shuffle_words, auroc

torch.set_grad_enabled(False)
MODEL = "EleutherAI/pythia-410m"
LAYERS = [6, 12, 18]
FEATS = ["resid_cv", "resid_max_ratio", "resid_autocorr1", "n_jumps", "operator_drift"]
SPECTRAL_BASELINE = 0.461   # the dead probe's score on this exact gate (RESULT_hankel_k1)


def layer_traj(H):
    H = H - H.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(H, full_matrices=False)
    return U[:, :24] * S[:24]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {MODEL} (cpu float32)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

    rows = []   # (label, {L: features dict})
    for p in PASSAGES:
        for text, lab in [(p, 0), (shuffle_words(p), 1)]:
            ids = tok(text, return_tensors="pt").input_ids[:, :96]
            hs = model(ids, output_hidden_states=True).hidden_states
            rows.append((lab, {L: nonstationarity_features(layer_traj(hs[L][0].numpy())) for L in LAYERS}))
    y = np.array([lab for lab, _ in rows])
    print(f"n={len(y)} ({int((y==0).sum())} coherent / {int((y==1).sum())} shuffled), model={MODEL}\n")

    X = np.array([[feat[L][k] for L in LAYERS for k in FEATS] for _, feat in rows])

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    def cv_auroc(M):
        Ms = StandardScaler().fit_transform(M)
        proba = cross_val_predict(LogisticRegression(C=0.5, max_iter=1000), Ms, y, cv=5,
                                  method="predict_proba")[:, 1]
        return auroc(y, proba)

    primary = cv_auroc(X)

    # transparency: per-feature univariate AUROC (pooled over layers, direction-free via max(a,1-a))
    print("per-feature univariate AUROC (transparency; |dir| via max(a,1-a)):")
    feat_scores = {}
    for fi, k in enumerate(FEATS):
        cols = [li * len(FEATS) + fi for li in range(len(LAYERS))]
        best = 0.0
        for c in cols:
            a = auroc(y, X[:, c]); a = max(a, 1 - a)
            best = max(best, a)
        feat_scores[k] = best
        print(f"  {k:16s}: {best:.3f}")

    # transparency: per-layer combined AUROC
    print("\nper-layer combined AUROC (transparency):")
    perlayer = {}
    for li, L in enumerate(LAYERS):
        cols = [li * len(FEATS) + fi for fi in range(len(FEATS))]
        a = cv_auroc(X[:, cols]); perlayer[str(L)] = float(a)
        print(f"  layer {L:2d}: {a:.3f}")

    print(f"\n===== PRIMARY VERDICT: K1-NS 5-fold CV AUROC = {primary:.3f}  (bar 0.70)")
    print(f"      (same gate, same data: the DMD spectrum scored {SPECTRAL_BASELINE:.3f})")
    if primary >= 0.70:
        print("===== K1-NS PASSES — the switching/non-stationarity object SEPARATES where the spectrum "
              "did not. Single-trajectory dynamical integrity is alive in the OPERATOR DRIFT, not its "
              "spectrum. Next pre-registered step: the deception + obfuscation test (K2-NS).")
    else:
        print(f"===== K1-NS FAILS at {primary:.3f} — the switching object does not separate either. "
              "Single-trajectory dynamical integrity (spectrum AND switching) is closed on this proxy. "
              "Per the stop rule, the only remaining honest dynamical bet is CROSS-SAMPLE consistency "
              "(resample-ensemble), inheriting styxx's proven sampling-divergence edge. Note and stop.")

    import json
    out = {"model": MODEL, "layers": LAYERS, "primary_auroc": float(primary),
           "spectral_baseline": SPECTRAL_BASELINE, "bar": 0.70, "n": int(len(y)),
           "per_feature_auroc": {k: float(v) for k, v in feat_scores.items()},
           "per_layer_auroc": perlayer,
           "verdict": "K1_NS_PASS" if primary >= 0.70 else "K1_NS_FAIL"}
    with open("nonstationarity_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote nonstationarity_result.json")


if __name__ == "__main__":
    main()

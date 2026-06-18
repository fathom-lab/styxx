# -*- coding: utf-8 -*-
"""
coherence_spectral_test_v3_hankel.py — K1 re-test with time-delay (Hankel) Koopman extraction.
PREREG: PREREG_hankel_k1_2026_06_18.md (FROZEN before this ran).

Line-for-line v2, with ONE change: each (T x 24) layer trajectory is delay-embedded before the
IDENTICAL spectral fingerprint. Same model (pythia-410m, CPU float32), same layers [6,12,18], same
PASSAGES coherent-vs-shuffled data, same 5 features/layer, same StandardScaler+LogReg(C=0.5), same
5-fold CV AUROC, same 0.70 bar.

PRIMARY (the verdict): q=12.   SECONDARY (transparency, non-verdict): q-sweep + per-layer + richer.
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from spectral_features import spectral_features
from hankel_spectral import delay_embed
from coherence_spectral_test import PASSAGES, shuffle_words, auroc

torch.set_grad_enabled(False)
MODEL = "EleutherAI/pythia-410m"
LAYERS = [6, 12, 18]
Q_PRIMARY = 12
Q_SWEEP = [1, 4, 8, 12, 16, 24]      # q=1 == v2 baseline (should reproduce ~0.562)
FEATS = ["dominant_freq", "weighted_freq", "spectral_entropy", "high_band_frac", "weighted_decay"]


def layer_traj(H):
    H = H - H.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(H, full_matrices=False)
    return U[:, :24] * S[:24]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {MODEL} (cpu float32)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

    # Extract each layer trajectory ONCE per (text,label); embed post-hoc at each q.
    trajechoes = []   # list of (label, {L: (T,24) trajectory})
    for p in PASSAGES:
        for text, lab in [(p, 0), (shuffle_words(p), 1)]:
            ids = tok(text, return_tensors="pt").input_ids[:, :96]
            hs = model(ids, output_hidden_states=True).hidden_states
            trajechoes.append((lab, {L: layer_traj(hs[L][0].numpy()) for L in LAYERS}))
    y = np.array([lab for lab, _ in trajechoes])
    print(f"n={len(y)} ({int((y==0).sum())} coherent / {int((y==1).sum())} shuffled), model={MODEL}\n")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    def cv_auroc(X):
        Xs = StandardScaler().fit_transform(X)
        proba = cross_val_predict(LogisticRegression(C=0.5, max_iter=1000), Xs, y, cv=5,
                                  method="predict_proba")[:, 1]
        return auroc(y, proba)

    def features_at_q(q):
        X = []
        for _, trajs in trajechoes:
            row = []
            for L in LAYERS:
                f = spectral_features(delay_embed(trajs[L], q), rank=12)
                row += [f[k] for k in FEATS]
            X.append(row)
        return np.array(X)

    # ---- q-sweep (q=1 is the v2 reproduction; q=12 is the PRIMARY verdict) ----
    print("q-sweep — multi-layer 5-fold CV AUROC (bar 0.70):")
    results = {}
    for q in Q_SWEEP:
        a = cv_auroc(features_at_q(q))
        results[q] = a
        tag = "  <-- v2 baseline (raw DMD)" if q == 1 else ("  <-- PRIMARY (verdict)" if q == Q_PRIMARY else "")
        print(f"  q={q:2d}:  AUROC = {a:.3f}{tag}")

    # ---- per-layer transparency at the primary q ----
    print(f"\nper-layer single-feature AUROC at q={Q_PRIMARY} (transparency, not the verdict):")
    Xp = features_at_q(Q_PRIMARY)
    for li, L in enumerate(LAYERS):
        dom = Xp[:, li * len(FEATS) + 0]   # dominant_freq is feature 0 of each layer block
        print(f"  layer {L:2d} dominant_freq: {auroc(y, dom):.3f}")

    # ---- VERDICT ----
    primary = results[Q_PRIMARY]
    print(f"\n===== PRIMARY VERDICT: q={Q_PRIMARY} 5-fold CV AUROC = {primary:.3f}  (bar 0.70)")
    if primary >= 0.70:
        print("===== K1 RECOVERS — time-delay Koopman extraction clears the coherence proxy. The "
              "spectral premise is alive under the correct estimator; the obfuscation-robustness "
              "gate K2 (the value proposition) becomes the next pre-registered GPU step.")
    else:
        print(f"===== K1 STILL FAILS at {primary:.3f} — even the principled short-trajectory estimator "
              "(Hankel/Takens delay embedding) does not lift the spectrum to gross-cognitive-state "
              "separability. Per the prereg stop rule, the Spectral Integrity Probe is CLOSED: "
              "robustly dead under the right tool, not re-shelved. One real nail.")

    out = {"model": MODEL, "layers": LAYERS, "q_primary": Q_PRIMARY,
           "q_sweep_auroc": {str(q): float(v) for q, v in results.items()},
           "primary_auroc": float(primary), "n": int(len(y)), "bar": 0.70,
           "verdict": "K1_RECOVERS" if primary >= 0.70 else "K1_CLOSED"}
    import json
    with open("hankel_k1_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote hankel_k1_result.json")


if __name__ == "__main__":
    main()

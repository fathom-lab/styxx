# -*- coding: utf-8 -*-
"""
coherence_spectral_test_v2.py — improved-extraction re-test of the instrument's K1 (CPU).
The v1 proxy FAILED (distilgpt2 L4, AUROC 0.43). My own commit said "fix extraction first." So:
bigger model (pythia-410m), RICHER multi-layer spectral features, longer trajectories. Same
pre-registered bar (0.70) on the coherent-vs-shuffled proxy; ALL layers reported (no cherry-pick) —
the verdict is on the multi-layer combined classifier, not a fished single layer.
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from spectral_features import spectral_features
from coherence_spectral_test import PASSAGES, shuffle_words, auroc

torch.set_grad_enabled(False)
MODEL = "EleutherAI/pythia-410m"
LAYERS = [6, 12, 18]                      # ~25/50/75% of pythia-410m's 24 layers (principled, fixed)


def layer_traj(H):
    H = H - H.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(H, full_matrices=False)
    return U[:, :24] * S[:24]


def feats_multi(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids[:, :96]
    hs = model(ids, output_hidden_states=True).hidden_states
    out = []
    per_layer = {}
    for L in LAYERS:
        f = spectral_features(layer_traj(hs[L][0].numpy()), rank=12)
        per_layer[L] = f
        out += [f["dominant_freq"], f["weighted_freq"], f["spectral_entropy"],
                f["high_band_frac"], f["weighted_decay"]]
    return out, per_layer


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {MODEL} (cpu)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()

    X, y, perL = [], [], {L: ([], []) for L in LAYERS}
    for p in PASSAGES:
        for text, lab in [(p, 0), (shuffle_words(p), 1)]:
            v, pl = feats_multi(model, tok, text)
            X.append(v); y.append(lab)
            for L in LAYERS:
                perL[L][0].append(pl[L]["dominant_freq"]); perL[L][1].append(lab)
    X, y = np.array(X), np.array(y)
    print(f"n={len(y)} ({sum(y==0)} coherent / {sum(y==1)} shuffled), model={MODEL}")

    print("per-layer dom_freq AUROC (transparency, not the verdict):")
    for L in LAYERS:
        s, lab = np.array(perL[L][0]), np.array(perL[L][1])
        print(f"  layer {L:2d}: {auroc(lab, s):.3f}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    Xs = StandardScaler().fit_transform(X)
    proba = cross_val_predict(LogisticRegression(C=0.5, max_iter=1000), Xs, y, cv=5,
                              method="predict_proba")[:, 1]
    cv = auroc(y, proba)
    print(f"\nmulti-layer combined classifier, 5-fold CV AUROC = {cv:.3f}  (bar 0.70)")
    if cv >= 0.70:
        v = ("K1 RECOVERS — richer extraction (bigger model + multi-layer) clears the bar on the "
             "coherence proxy. The spectral premise is alive; the GPU deception test is now worth running.")
    else:
        v = (f"K1 STILL FAILS at {cv:.3f} — the spectral premise does not carry even gross cognitive-state "
             "signal across two models and richer extraction. Strong honest signal to SHELVE the spectral "
             "integrity instrument (or rethink the feature entirely) rather than spend GPU on deception.")
    print("\n===== " + v)


if __name__ == "__main__":
    main()

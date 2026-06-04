# -*- coding: utf-8 -*-
"""
coherence_spectral_test.py — does a real LM's residual-stream SPECTRUM separate coherent from
incoherent input? Disciplined follow-up to the N=1 hint (shuffled -> Nyquist). CPU-only.

PRE-REGISTERED (stated before the AUROC is computed):
  - Paired design: each coherent passage vs its word-shuffled self (same vocab, order destroyed).
  - Primary feature: dominant_freq (hint predicts shuffled higher / toward Nyquist).
  - Bar: a spectral classifier (5-fold CV) separates coherent vs shuffled at AUROC >= 0.70.
  - This is a COHERENCE proxy for "does the spectrum carry cognitive-state signal" (the instrument's
    K1) -- NOT the honest/deceptive test. A pass motivates the GPU deception experiment; a fail says
    the N=1 hint was noise. Report whatever it says.
"""
from __future__ import annotations
import sys, math
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from spectral_features import spectral_features

torch.set_grad_enabled(False)
DEV, MODEL, LAYER = "cpu", "distilgpt2", 4
rng = np.random.default_rng(0)

PASSAGES = [
    "The history of mathematics begins with counting, as early civilizations built numeral systems to track trade and the turning seasons.",
    "Photosynthesis converts sunlight into chemical energy, letting plants build sugars from carbon dioxide and water while releasing oxygen.",
    "Rivers carve valleys over millions of years, carrying sediment downstream and slowly reshaping the land through erosion and deposition.",
    "The printing press spread literacy across Europe, letting ideas travel faster than any messenger could carry them between distant cities.",
    "Bees navigate by the sun and communicate the location of flowers to the hive through a precise waggle dance on the comb.",
    "A suspension bridge hangs its roadway from cables draped between tall towers, balancing tension and compression across the long span.",
    "Vaccines train the immune system by presenting a harmless fragment, so the body recognizes and defeats the real pathogen later.",
    "Glaciers store much of the planet's fresh water, advancing and retreating with the climate over centuries and grinding rock to dust.",
    "The telescope revealed that the Milky Way is made of countless stars, expanding the known universe far beyond the naked eye.",
    "Coral reefs shelter a quarter of marine species, built grain by grain from the limestone skeletons of tiny living polyps.",
    "Electric current flows when a voltage pushes charge through a conductor, and resistance turns some of that energy into heat.",
    "Migratory birds cross continents each year, reading magnetic fields and star patterns to find the same nesting grounds again.",
    "The water cycle moves moisture from ocean to sky to land, falling as rain and returning through rivers to begin once more.",
    "Antibiotics kill bacteria or stop them dividing, but overuse breeds resistant strains that no longer respond to the old drugs.",
    "A seed holds a tiny plant and a store of food, waiting for warmth and water before it pushes a root into the soil.",
    "The steam engine turned heat into motion, powering mills and locomotives and igniting the industrial transformation of the world.",
]


def trajectory(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids[:, :64]
    H = model(ids, output_hidden_states=True).hidden_states[LAYER][0].numpy()
    H = H - H.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(H, full_matrices=False)
    return U[:, :24] * S[:24]


def feats(model, tok, text):
    f = spectral_features(trajectory(model, tok, text), rank=12)
    return [f["dominant_freq"], f["weighted_freq"], f["spectral_entropy"],
            f["high_band_frac"], f["weighted_decay"]]


def shuffle_words(text):
    w = text.split()
    rng.shuffle(w)
    return " ".join(w)


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    return float((sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)) / (len(pos) * len(neg)))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {MODEL} (cpu)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

    X, y = [], []
    for p in PASSAGES:
        X.append(feats(model, tok, p)); y.append(0)                 # coherent
        X.append(feats(model, tok, shuffle_words(p))); y.append(1)  # incoherent
    X, y = np.array(X), np.array(y)
    names = ["dom_freq", "wt_freq", "spec_ent", "hi_band", "persist"]

    print(f"\nn={len(y)} ({sum(y==0)} coherent / {sum(y==1)} shuffled)")
    print("per-feature AUROC (coherent=0 vs shuffled=1):")
    for j, nm in enumerate(names):
        print(f"  {nm:9s} {auroc(y, X[:, j]):.3f}")

    # 5-fold CV logistic on standardized features
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    Xs = StandardScaler().fit_transform(X)
    proba = cross_val_predict(LogisticRegression(max_iter=1000), Xs, y, cv=5, method="predict_proba")[:, 1]
    cv_auroc = auroc(y, proba)
    print(f"\n5-fold CV spectral classifier AUROC = {cv_auroc:.3f}  (pre-registered bar 0.70)")
    verdict = ("PASS — a real LM's residual-stream SPECTRUM carries coherence signal; the K1 premise "
               "of the spectral integrity instrument holds on this proxy. Motivates the GPU deception test."
               if cv_auroc >= 0.70 else
               "FAIL — spectrum does NOT separate coherent vs shuffled at bar; the N=1 hint was noise. "
               "Honest negative; reconsider the instrument's K1 before the deception test.")
    print("\n===== " + verdict)


if __name__ == "__main__":
    main()

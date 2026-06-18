# -*- coding: utf-8 -*-
"""
cross_sample_test.py — K1-CS: do K resampled generations' DYNAMICS agree?
PREREG: PREREG_cross_sample_2026_06_18.md (FROZEN 4d4b5b7, before this ran).

For each prompt we generate K=6 continuations, read each one's deep-layer non-stationarity signature
(the object that was alive-but-sub-threshold on a single trajectory), and measure cross-sample
DYNAMICAL CONSENSUS (how much the K signatures vary). Gate K1: separate answerable/stable from
underspecified/improvise at 5-fold CV AUROC >= 0.70. Confound K: must beat a semantic-entropy
baseline (spread of the K last-token hidden states). Cheap proxy => K1 decisive, K-confound indicative.
"""
from __future__ import annotations
import sys
import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from nonstationarity_features import nonstationarity_features
from coherence_spectral_test import auroc

torch.set_grad_enabled(False)
MODEL = "EleutherAI/pythia-410m"
LAYER = 18
K = 6
NS_KEYS = ["resid_cv", "resid_max_ratio", "resid_autocorr1", "n_jumps", "operator_drift"]

ANSWERABLE = [
    "The capital of France is", "Two plus two equals", "The sun rises in the",
    "Water freezes at zero degrees", "The opposite of hot is", "A dog is a kind of",
    "The first month of the year is", "Roses are red, violets are",
    "The chemical symbol for water is", "There are seven days in a",
    "The Earth orbits the", "One plus one equals",
]
IMPROVISE = [
    "The secret she had never told anyone was", "In the year 3047, the most popular hobby was",
    "His favorite thing about the abandoned house was", "The strangest dream I ever had involved",
    "What nobody realized about the painting was", "The note left on the door simply read",
    "Deep in the forest, the travelers discovered", "The last thing he expected to find was",
    "Her plan for the weekend was to", "The reason the machine stopped working was",
    "On the other side of the mirror,", "The message hidden in the song was",
]


def traj_pca(H):
    H = H - H.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(H, full_matrices=False)
    k = min(24, U.shape[1])
    return U[:, :k] * S[:k]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL} on {dev}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev).eval()

    def prompt_features(text):
        """Return (cross-sample dynamical-consensus vector, semantic-entropy scalar)."""
        ids = tok(text, return_tensors="pt").to(dev)
        plen = ids.input_ids.shape[1]
        gen = model.generate(**ids, do_sample=True, temperature=0.8, top_p=0.95,
                             max_new_tokens=32, num_return_sequences=K,
                             pad_token_id=tok.pad_token_id)
        hs = model(gen, output_hidden_states=True).hidden_states[LAYER]   # (K, seq, d)
        sigs, lasts = [], []
        for k in range(K):
            traj = hs[k, plen:, :].float().cpu().numpy()
            if len(traj) < 8:
                continue
            f = nonstationarity_features(traj_pca(traj))
            sigs.append([f[key] for key in NS_KEYS])
            lasts.append(hs[k, -1, :].float().cpu().numpy())
        sigs = np.array(sigs)
        # cross-sample dynamical consensus: per-feature std across samples + mean pairwise sig distance
        per_feat_std = sigs.std(0)
        # normalize each feature column by its mean magnitude before pairwise dist (scale-free)
        Z = (sigs - sigs.mean(0)) / (sigs.std(0) + 1e-9)
        pdists = [np.linalg.norm(Z[i] - Z[j]) for i in range(len(Z)) for j in range(i + 1, len(Z))]
        dyn_vec = list(per_feat_std) + [float(np.mean(pdists)) if pdists else 0.0]
        # semantic-entropy baseline: mean pairwise cosine distance of last-token reps
        L = np.array(lasts)
        Ln = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-9)
        sims = [float(Ln[i] @ Ln[j]) for i in range(len(Ln)) for j in range(i + 1, len(Ln))]
        sem = float(1.0 - np.mean(sims)) if sims else 0.0
        return dyn_vec, sem

    X, sem_base, y = [], [], []
    for label, prompts in [(0, ANSWERABLE), (1, IMPROVISE)]:
        for p in prompts:
            dv, sem = prompt_features(p)
            X.append(dv); sem_base.append([sem]); y.append(label)
    X, sem_base, y = np.array(X), np.array(sem_base), np.array(y)
    print(f"n={len(y)} prompts ({int((y==0).sum())} answerable / {int((y==1).sum())} improvise), K={K}\n")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    def cv(M):
        Ms = StandardScaler().fit_transform(M)
        pr = cross_val_predict(LogisticRegression(C=0.5, max_iter=1000), Ms, y, cv=5,
                               method="predict_proba")[:, 1]
        return auroc(y, pr)

    a_dyn = cv(X)
    a_sem = cv(sem_base)
    a_both = cv(np.hstack([X, sem_base]))
    print(f"semantic-entropy baseline AUROC        : {a_sem:.3f}")
    print(f"cross-sample dynamical-consensus AUROC : {a_dyn:.3f}   (bar 0.70)")
    print(f"dynamical + semantic combined AUROC    : {a_both:.3f}")
    print(f"novelty delta (dyn - sem)              : {a_dyn - a_sem:+.3f}   (confound bar +0.05)")

    print(f"\n===== K1-CS VERDICT: dynamical AUROC = {a_dyn:.3f}  (bar 0.70)")
    if a_dyn >= 0.70:
        nov = a_dyn - a_sem
        print("===== K1-CS PASSES — cross-sample dynamical consensus separates. The alive single-"
              "trajectory object survives ensembling: the dynamical line is real.")
        if nov >= 0.05:
            print(f"===== K-confound INDICATIVE-POSITIVE (+{nov:.3f} over semantic) — possibly novel; "
                  "the real honest/confabulated dataset is now the decisive pre-registered step.")
        else:
            print(f"===== K-confound INDICATIVE-NEGATIVE ({nov:+.3f}) — does not clearly beat semantic "
                  "entropy on this proxy; alive but likely divergence-repackaged. Fold in as secondary.")
    else:
        print(f"===== K1-CS FAILS at {a_dyn:.3f} — cross-sample dynamical consensus does not separate "
              "even answerable vs improvise. Per the stop rule, the dynamical integrity line (single AND "
              "cross-sample) is closed on cheap proxies. styxx keeps its existing semantic-divergence tools.")

    import json
    out = {"model": MODEL, "layer": LAYER, "K": K, "n": int(len(y)),
           "dynamical_auroc": float(a_dyn), "semantic_auroc": float(a_sem),
           "combined_auroc": float(a_both), "novelty_delta": float(a_dyn - a_sem), "bar": 0.70,
           "verdict": "K1_CS_PASS" if a_dyn >= 0.70 else "K1_CS_FAIL"}
    with open("cross_sample_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote cross_sample_result.json")


if __name__ == "__main__":
    main()

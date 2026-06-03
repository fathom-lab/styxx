# -*- coding: utf-8 -*-
"""
human_feature_spectrum.py — robustness across the FULL model zoo: does the deep>shallow advantage at
matching the human 54-feature meaning space hold across ALL contextual deep models vs ALL static ones,
dimensionality-matched, with a bootstrap on the cluster gap? 672 Chinese concepts, brain-independent.
"""
import os, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_predictor_rdms import load_emb

STATIC = {"GloVe": "GloVe.mat", "fastText": "fastText.mat"}
DEEP = {"GPT2": "GPT2.mat", "BERT": "BERT.mat", "ERNIE": "ERNIE.mat", "Electra": "Electra.mat"}
VISION = {"ResNet": "ResNet.mat", "ViT": "Vit.mat", "DenseNet": "DenseNet.mat", "Beit": "Beit.mat"}
IU = np.triu_indices(672, 1)


def distmat(P):
    P = P - P.mean(0); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9); return 1 - P @ P.T


def pca_k(E, k):
    Ec = E - E.mean(0); U, S, _ = np.linalg.svd(Ec, full_matrices=False); return U[:, :min(k, E.shape[1])] * S[:min(k, E.shape[1])]


def main():
    H = np.load(os.path.join(HERE, "human_features.npy")); Hr = distmat(H)[IU]
    def rsa_full(E): return float(np.corrcoef(distmat(E)[IU], Hr)[0, 1])
    def rsa_m(E): return float(np.corrcoef(distmat(pca_k(E, 50))[IU], Hr)[0, 1])
    allm = {**STATIC, **DEEP, **VISION}
    raw = {t: load_emb(fn)[1] for t, fn in allm.items()}
    full = {t: rsa_full(raw[t]) for t in raw}
    matched = {t: rsa_m(raw[t]) for t in raw}

    print("=== match to HUMAN 54-feature meaning (RSA, 672 concepts) — full zoo ===")
    print(f"  {'model':10s} {'full':>7} {'PCA-50':>7}")
    for t in sorted(full, key=lambda x: -matched[x]):
        grp = "deep" if t in DEEP else ("static" if t in STATIC else "vision")
        print(f"  {t:10s} {full[t]:>+7.3f} {matched[t]:>+7.3f}   [{grp}]")

    # cluster means (dim-matched) + bootstrap of deep-cluster minus static-cluster
    rdm = {t: distmat(pca_k(raw[t], 50)) for t in raw}
    def cl_rsa(grp, idx):
        sub = Hr if idx is None else distmat(H[idx])[np.triu_indices(len(idx), 1)]
        Hi = Hr if idx is None else sub
        return np.mean([np.corrcoef((rdm[t] if idx is None else distmat(pca_k(raw[t], 50)[idx]))[np.triu_indices(672 if idx is None else len(idx), 1)], Hi)[0, 1] for t in grp])
    d_mean = np.mean([matched[t] for t in DEEP]); s_mean = np.mean([matched[t] for t in STATIC]); v_mean = np.mean([matched[t] for t in VISION])
    Epca = {t: pca_k(raw[t], 50) for t in raw}          # precompute PCA once (was the bottleneck)
    iu = np.triu_indices(672, 1)
    rng = np.random.default_rng(0); diffs = []
    for _ in range(1500):
        idx = rng.integers(0, 672, 672)
        Hi = distmat(H[idx])[iu]
        dv = np.mean([np.corrcoef(distmat(Epca[t][idx])[iu], Hi)[0, 1] for t in DEEP])
        sv = np.mean([np.corrcoef(distmat(Epca[t][idx])[iu], Hi)[0, 1] for t in STATIC])
        diffs.append(dv - sv)
    diffs = np.array(diffs); ci = (round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3))
    p = float((diffs > 0).mean())
    print(f"\ncluster means (PCA-50): deep {d_mean:.3f}  static {s_mean:.3f}  vision {v_mean:.3f}")
    print(f"deep - static: {d_mean-s_mean:+.3f}, bootstrap 95% CI {ci}, P(deep>static)={p:.3f}")
    worst_deep = min(matched[t] for t in DEEP); best_static = max(matched[t] for t in STATIC)
    print(f"worst deep ({worst_deep:.3f}) vs best static ({best_static:.3f}): {'ALL deep beat ALL static' if worst_deep>best_static else 'overlap (fastText close)'}")
    import json
    json.dump({"full": {k: round(v, 3) for k, v in full.items()}, "matched": {k: round(v, 3) for k, v in matched.items()},
               "cluster_deep": round(d_mean, 3), "cluster_static": round(s_mean, 3), "cluster_vision": round(v_mean, 3),
               "deep_minus_static": round(d_mean - s_mean, 3), "ci95": ci, "p": round(p, 3)},
              open(os.path.join(HERE, "human_feature_spectrum_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote human_feature_spectrum_result.json")


if __name__ == "__main__":
    main()

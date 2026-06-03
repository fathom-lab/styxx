# -*- coding: utf-8 -*-
"""
encode_cv.py [betas_glob] [shift] — voxelwise ENCODING-model deflation (more sensitive than RDM-RSA).
Cross-validated ridge predicting each brain voxel's response across concepts from an embedding; the
embedding's brain-prediction accuracy = mean held-out predicted-vs-actual voxel correlation. Compares
GloVe(shallow) vs GPT2/BERT(deep) vs ResNet/ViT(vision): does DEEP predict the brain better?
shift: brain concept_index c -> embedding index c+shift (gls betas baked-in: 0; old run_cn_glm2: -1).
"""
import os, glob, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_predictor_rdms import load_emb


def voxelcorr(P, Y):
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-9); Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    return (Pz * Yz).mean(0)


def pca_k(E, k):
    """dimensionality-match: top-k principal-component scores (same capacity for every embedding)."""
    Ec = E - E.mean(0)
    U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    return U[:, :k] * S[:k]


def cv_encode(E, Y, alphas=(1, 10, 100, 1e3, 1e4, 1e5), folds=5):
    N = E.shape[0]
    Ez = (E - E.mean(0)) / (E.std(0) + 1e-9)
    perm = np.random.default_rng(0).permutation(N)
    best = (-9, None, None)
    for a in alphas:
        pred = np.zeros_like(Y)
        for k in range(folds):
            te = perm[k::folds]; tr = np.setdiff1d(perm, te)
            Xtr = Ez[tr]
            B = np.linalg.solve(Xtr.T @ Xtr + a * np.eye(Xtr.shape[1]), Xtr.T @ Y[tr])
            pred[te] = Ez[te] @ B
        r = voxelcorr(pred, Y)
        score = float(np.nanmean(r))
        if score > best[0]:
            best = (score, a, r)
    return best  # (mean_r, alpha, per_voxel_r)


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "gls_betas_sub-*.npz"
    shift = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    matched_k = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # >0: PCA every embedding to k dims (fair fight)
    fs = sorted(glob.glob(os.path.join(HERE, pat)))
    if not fs:
        print(f"no betas match {pat}"); return
    dat = [np.load(f, allow_pickle=True) for f in fs]
    common = sorted(set.intersection(*[set(d["concept_index"].tolist()) for d in dat]))
    M = np.mean([d["mean"][[d["concept_index"].tolist().index(c) for c in common]].astype(np.float32) for d in dat], 0)
    # map brain concepts -> embedding rows
    embidx = np.array([c + shift for c in common])
    keep = (embidx >= 0) & (embidx < 672)
    M = M[keep]; embidx = embidx[keep]
    Y = M  # (Nconcept, Vvox) brain
    print(f"{len(fs)} subj, {Y.shape[0]} concepts, {Y.shape[1]} voxels; encoding (shift={shift})", flush=True)

    embs = {"GloVe": "GloVe.mat", "GPT2": "GPT2.mat", "BERT": "BERT.mat", "ResNet": "ResNet.mat", "ViT": "Vit.mat"}
    hf = os.path.join(HERE, "human_features.npy")
    human = np.load(hf) if os.path.exists(hf) else None    # (672,54) human-rated, brain-designed reference
    res = {}
    if matched_k:
        print(f"DIMENSIONALITY-MATCHED: every embedding PCA'd to {matched_k} dims (fair-capacity fight)", flush=True)
    items = list(embs.items()) + ([("Human54", None)] if human is not None else [])
    for tag, fn in items:
        E = human if tag == "Human54" else load_emb(fn)[1]    # (672, dim)
        Ec = E[embidx]
        if matched_k:
            Ec = pca_k(Ec, matched_k)
        score, alpha, r = cv_encode(Ec, Y)
        top = np.mean(np.sort(r)[-5000:])
        res[tag] = (score, top, alpha)
        print(f"  {tag:8s}: mean voxel r = {score:+.4f}   top5k r = {top:+.4f}   (alpha {alpha:g})", flush=True)
    deep = max(res["GPT2"][0], res["BERT"][0]); shallow = res["GloVe"][0]; vis = max(res["ResNet"][0], res["ViT"][0])
    print(f"\ndeep {deep:+.4f} vs shallow(GloVe) {shallow:+.4f} vs vision {vis:+.4f}")
    if max(res.values(), key=lambda x: x[0])[0] < 0.01:
        print(">>> encoding finds ~no signal either -> betas/data limited, not method.")
    elif deep > shallow + 0.005:
        print(">>> DEEP predicts the brain better than shallow co-occurrence (encoding). Depth reaches the brain.")
    elif abs(deep - shallow) <= 0.005:
        print(">>> deep ~ shallow: the brain-predictable structure is shallow co-occurrence (substance).")


if __name__ == "__main__":
    main()

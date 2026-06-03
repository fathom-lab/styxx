# -*- coding: utf-8 -*-
"""
check_gls.py — single-subject validation of GLMsingle betas: does the denoised RDM align with the
embeddings where the standard GLM gave ~0? Go/no-go before running GLMsingle on every subject.
Alignment is already baked into the design (concept index = embedding index), so NO shift here.
"""
import os, glob, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))


def distmat(P):
    P = P - P.mean(0); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9); return 1 - P @ P.T


def pcorr(a, b, Z):
    X = np.column_stack([np.ones(len(a)), Z]); ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]; rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    fs = sorted(glob.glob(os.path.join(HERE, "gls_betas_sub-*.npz")))
    if not fs:
        print("no GLMsingle betas yet"); return
    d = np.load(fs[0], allow_pickle=True)
    mean = d["mean"].astype(np.float32)          # (NCON, K)
    ci = d["concept_index"]
    keep = ~np.all(mean == 0, axis=1)            # concepts actually estimated
    mean = mean[keep]; ci = ci[keep]
    K = mean.shape[1]
    print(f"{os.path.basename(fs[0])}: {mean.shape[0]} concepts, {K} voxels", flush=True)

    # variance-based voxel selection (top-variance voxels carry signal; no inter-subject with 1 subject)
    var = mean.var(0);
    P = np.load(os.path.join(HERE, "predictor_rdms.npz"), allow_pickle=True)
    words = [str(w) for w in P["words"]]
    cl = np.array([len(words[c]) for c in ci], float); zc = (cl - cl.mean()) / (cl.std() + 1e-9)
    haveH = "rdm__Human54" in P.files
    print(f"\n{'Kvox':>7} {'Human54':>9} {'GloVe':>8} {'GPT2':>8} {'BERT':>8} {'vision':>8}")
    for Kv in [2000, 5000, 10000, 20000, K]:
        sel = np.argsort(var)[-Kv:] if Kv < K else np.arange(K)
        R = distmat(mean[:, sel]); IU = np.triu_indices(R.shape[0], 1)
        L = np.abs(zc[:, None] - zc[None, :])[IU][:, None]
        g = R[IU]
        def pr(tag):
            return pcorr(P["rdm__" + tag][np.ix_(ci, ci)][IU], g, L)
        deep = (pr("GPT2") + pr("BERT")) / 2; vis = (pr("ResNet") + pr("ViT")) / 2
        hu = pr("Human54") if haveH else float("nan")
        print(f"{Kv:>7} {hu:>+9.3f} {pr('GloVe'):>+8.3f} {pr('GPT2'):>+8.3f} {pr('BERT'):>+8.3f} {vis:>+8.3f}")
    print("\n(if these are clearly >0, e.g. >0.05, GLMsingle cracked it -> run all subjects + pool)")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Build predictor RDMs (deep/shallow/vision) over the 672 ds004301 concepts from the provided
embeddings, and validate them with a category positive-control. Saves cn/predictor_rdms.npz."""
import sys, os, csv
import numpy as np
from scipy.io import loadmat
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__)); AD = os.path.join(HERE, "annot")


def load_emb(name):
    m = loadmat(os.path.join(AD, name)); data = m["data"]; n, c = data.shape
    words = [str(data[i, 0][0]) for i in range(n)]
    E = np.array([[float(data[i, j].ravel()[0]) for j in range(1, c)] for i in range(n)], dtype=float)
    return words, E


def distmat(R):
    R = R - R.mean(0); R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    G = R @ R.T
    return np.sqrt(np.maximum(2.0 - 2.0 * G, 0.0))


def main():
    embs = {"GloVe": "GloVe.mat", "fastText": "fastText.mat", "GPT2": "GPT2.mat",
            "BERT": "BERT.mat", "ResNet": "ResNet.mat", "ViT": "Vit.mat"}
    words = None; rdms = {}; dims = {}
    for tag, fn in embs.items():
        w, E = load_emb(fn)
        if words is None:
            words = w
        rdms[tag] = distmat(E); dims[tag] = E.shape[1]
    N = len(words); IU = np.triu_indices(N, 1)

    # categories (positive control): parse 672word_category.txt
    cat = None
    catfile = os.path.join(AD, "672word_category.txt")
    raw = None
    for enc in ["utf-16", "utf-8-sig", "latin-1"]:
        try:
            raw = open(catfile, encoding=enc).read().splitlines()
            if raw and raw[0]:
                break
        except Exception:
            pass
    parts0 = raw[0].replace("\t", " ").split()
    # try: each line "<idx?> <word> <category>" or "<word> <category>"; take last token as category
    cats = [ln.replace("\t", " ").split()[-1] for ln in raw[1:] if ln.strip()]  # skip header
    if len(cats) == N:
        cat = np.array(cats)
        same = cat[:, None] == cat[None, :]
        eye = np.eye(N, dtype=bool)
        print(f"categories: {len(set(cats))} unique over {N} concepts")
        for tag in embs:
            D = rdms[tag]
            ratio = D[~same].mean() / (D[same & ~eye].mean() + 1e-9)
            print(f"  {tag:8s} (dim {dims[tag]:3d}): category structure across/within = {ratio:.3f}")
    else:
        print(f"category parse mismatch ({len(cats)} vs {N}); first line: {raw[0][:80]}")

    np.savez(os.path.join(HERE, "predictor_rdms.npz"),
             words=np.array(words), **{f"rdm__{t}": rdms[t] for t in rdms},
             **({"categories": cat} if cat is not None else {}))
    # quick cross-predictor: how much does deep (GPT2) diverge from shallow (GloVe)?
    def rsa(a, b):
        return float(np.corrcoef(rdms[a][IU], rdms[b][IU])[0, 1])
    print(f"\nGPT2 vs GloVe RSA: {rsa('GPT2','GloVe'):.3f} | BERT vs GloVe: {rsa('BERT','GloVe'):.3f} | "
          f"GPT2 vs ResNet(vision): {rsa('GPT2','ResNet'):.3f}")
    print("wrote predictor_rdms.npz")


if __name__ == "__main__":
    main()

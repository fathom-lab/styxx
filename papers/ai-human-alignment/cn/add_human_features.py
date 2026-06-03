# -*- coding: utf-8 -*-
"""
add_human_features.py — incorporate the 54-feature HUMAN semantic ratings (672 concepts, ~126 raters,
designed brain-relevant) as a clean, interpretable, non-distributional reference. Saves human_features.npy,
adds its RDM to predictor_rdms.npz, and previews: does the DEEP LLM match the human-rated meaning better
than shallow co-occurrence? (brain-independent test).
"""
import os, csv, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))


def distmat(P):
    P = P - P.mean(0); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9); return 1 - P @ P.T


def main():
    rows = list(csv.reader(open(os.path.join(HERE, "annot/sf/feature.csv"), encoding="utf-8-sig")))
    feats = rows[0][1:]
    H = np.array([[float(x) for x in r[1:]] for r in rows[1:]], dtype=float)   # (672, 54), aligned to embedding index
    np.save(os.path.join(HERE, "human_features.npy"), H)
    print(f"human features: {H.shape}  ({len(feats)} features, e.g. {feats[:4]})", flush=True)

    Hr = distmat(H)
    P = dict(np.load(os.path.join(HERE, "predictor_rdms.npz"), allow_pickle=True))
    P["rdm__Human54"] = Hr
    np.savez(os.path.join(HERE, "predictor_rdms.npz"), **P)
    print("added rdm__Human54 to predictor_rdms.npz", flush=True)

    # preview (672 concepts, RSA): how does each model match the HUMAN-rated meaning geometry?
    IU = np.triu_indices(672, 1)
    def rsa(a, b):
        return float(np.corrcoef(a[IU], b[IU])[0, 1])
    print("\n=== how well each model matches the HUMAN 54-feature meaning geometry (RSA, 672 concepts) ===")
    scores = {}
    for tag in ["GloVe", "fastText", "GPT2", "BERT", "ResNet", "ViT"]:
        scores[tag] = rsa(Hr, P["rdm__" + tag])
        print(f"  Human54 <-> {tag:8s}: {scores[tag]:+.3f}")
    deep = (scores["GPT2"] + scores["BERT"]) / 2; shallow = scores["GloVe"]; vis = (scores["ResNet"] + scores["ViT"]) / 2
    print(f"\n  deep {deep:+.3f}  vs  shallow(GloVe) {shallow:+.3f}  vs  vision {vis:+.3f}")
    if deep > shallow + 0.02:
        print(">>> DEEP LLM captures human-rated meaning BETTER than shallow co-occurrence (no brain needed).")
    elif abs(deep - shallow) <= 0.02:
        print(">>> deep ~ shallow at matching human-rated meaning.")
    else:
        print(">>> shallow matches human-rated meaning as well/better (surprising).")


if __name__ == "__main__":
    main()

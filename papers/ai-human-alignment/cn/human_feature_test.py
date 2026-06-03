# -*- coding: utf-8 -*-
"""
human_feature_test.py — RIGOROUS, brain-independent deflation: does the DEEP LLM match the human
54-feature meaning space better than shallow co-occurrence, controlling dimensionality and with a
bootstrap significance test? 672 Chinese concepts, ~126 human raters. No fMRI noise.
"""
import os, csv, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_predictor_rdms import load_emb


def distmat(P):
    P = P - P.mean(0); P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9); return 1 - P @ P.T


def pca_k(E, k):
    Ec = E - E.mean(0); U, S, _ = np.linalg.svd(Ec, full_matrices=False); return U[:, :k] * S[:k]


def rsa_idx(Ra, Rb, idx):
    a = Ra[np.ix_(idx, idx)]; b = Rb[np.ix_(idx, idx)]; iu = np.triu_indices(len(idx), 1)
    return float(np.corrcoef(a[iu], b[iu])[0, 1])


def main():
    H = np.load(os.path.join(HERE, "human_features.npy"))      # (672, 54) human-rated, aligned
    Hr = distmat(H)
    embs = {"GloVe": "GloVe.mat", "fastText": "fastText.mat", "GPT2": "GPT2.mat",
            "BERT": "BERT.mat", "ResNet": "ResNet.mat", "ViT": "Vit.mat"}
    raw = {t: load_emb(fn)[1] for t, fn in embs.items()}

    def block(transform, label):
        rd = {t: distmat(transform(raw[t])) for t in raw}
        full = np.arange(672)
        sc = {t: rsa_idx(Hr, rd[t], full) for t in rd}
        deep = (sc["GPT2"] + sc["BERT"]) / 2; sh = sc["GloVe"]
        # bootstrap the deep - shallow difference over concepts
        rng = np.random.default_rng(0); diffs = []
        for _ in range(2000):
            idx = rng.integers(0, 672, 672)
            d = (rsa_idx(Hr, rd["GPT2"], idx) + rsa_idx(Hr, rd["BERT"], idx)) / 2
            s = rsa_idx(Hr, rd["GloVe"], idx)
            diffs.append(d - s)
        diffs = np.array(diffs); ci = (round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3))
        pgt = float((diffs > 0).mean())
        print(f"\n[{label}] " + "  ".join(f"{t} {sc[t]:+.3f}" for t in ["GloVe", "GPT2", "BERT", "ResNet", "ViT"]))
        print(f"  deep {deep:+.3f} vs shallow {sh:+.3f} | diff {deep-sh:+.3f}, 95% CI {ci}, P(deep>shallow)={pgt:.3f}")
        return {"scores": {k: round(v, 3) for k, v in sc.items()}, "deep": round(deep, 3), "shallow": round(sh, 3),
                "diff": round(deep - sh, 3), "ci95": ci, "p_gt0": round(pgt, 3)}

    full = block(lambda E: E, "FULL dims")
    matched = block(lambda E: pca_k(E, 50), "DIM-MATCHED (PCA-50)")

    robust = full["p_gt0"] >= 0.975 and matched["p_gt0"] >= 0.975
    verdict = (f"CONFIRMED: deep LLMs capture human-rated meaning better than shallow co-occurrence, and it "
               f"SURVIVES dimensionality-matching (matched diff {matched['diff']:+.3f}, P={matched['p_gt0']:.3f}). "
               f"A clean, brain-independent depth effect on {672} concepts." if robust else
               f"NOT robust: full diff {full['diff']:+.3f} (P={full['p_gt0']}), matched {matched['diff']:+.3f} "
               f"(P={matched['p_gt0']}). The deep advantage may be dimensionality, not depth.")
    print(f"\n>>> {verdict}")
    import json
    json.dump({"full": full, "matched": matched, "robust": bool(robust), "verdict": verdict},
              open(os.path.join(HERE, "human_feature_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote human_feature_result.json")


if __name__ == "__main__":
    main()

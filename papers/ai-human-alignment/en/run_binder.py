# -*- coding: utf-8 -*-
"""
run_binder.py — the PROPER English generalization test, on a RICH human reference: Binder et al. 2016
(65 experiential features, spanning perceptual AND abstract — the direct analog of the Chinese 54).
Models: GloVe-300 (shallow) vs MiniLM (deep, proper word-level). Tests deep>shallow (bootstrapped) +
the full monitor (invariance / sensitivity / localization). If localization is sharp HERE (unlike the
thin 11-dim Lancaster norm), the full monitor generalizes when given a reference the models align with.
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cn"))
from meaning_integrity import MeaningReference, alignment, per_concept_alignment, _rdm


def pca(E, k):
    Ec = E - E.mean(0); U, S, _ = np.linalg.svd(Ec, full_matrices=False)
    k = min(k, E.shape[1]); return U[:, :k] * S[:k]


def auc(s, l):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    p = l == 1; n1 = p.sum(); n0 = (~p).sum()
    return float((r[p].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def main():
    df = pd.read_excel(os.path.join(HERE, "binder_data", "WordSet1_Ratings.xlsx"))
    words_all = [str(w).strip().lower() for w in df["Word"]]
    H_all = df.iloc[:, 5:70].apply(pd.to_numeric, errors="coerce").values   # 535 x 65 Binder features

    import gensim.downloader as gd
    from sentence_transformers import SentenceTransformer
    glove = gd.load("glove-wiki-gigaword-300")
    keep = [i for i, w in enumerate(words_all) if w in glove and not np.isnan(H_all[i]).any()]
    words = [words_all[i] for i in keep]
    human = H_all[keep].astype(float)
    gv = np.array([glove[w] for w in words], float)
    mini = np.asarray(SentenceTransformer("all-MiniLM-L6-v2").encode(words, show_progress_bar=False), float)
    ref = MeaningReference(human, words=words, name="binder65")
    n = len(words); rng = np.random.default_rng(0)
    print(f"Binder rich reference: {n} words x 65 features (GloVe {gv.shape[1]}d, MiniLM {mini.shape[1]}d)")

    print("=" * 70)
    print("DEEP vs SHALLOW alignment to the rich Binder human space:")
    print(f"  GloVe-300  full {alignment(gv, ref):+.3f}   PCA-50 {alignment(pca(gv, 50), ref):+.3f}")
    print(f"  MiniLM     full {alignment(mini, ref):+.3f}   PCA-50 {alignment(pca(mini, 50), ref):+.3f}")
    Gp, Mp = pca(gv, 50), pca(mini, 50)
    dg, dm = alignment(Gp, ref), alignment(Mp, ref)
    iu = np.triu_indices(n, 1)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        hr = _rdm(human[idx])[iu]
        diffs.append(np.corrcoef(_rdm(Mp[idx])[iu], hr)[0, 1] - np.corrcoef(_rdm(Gp[idx])[iu], hr)[0, 1])
    diffs = np.array(diffs); ci = (round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3))
    p = float((diffs > 0).mean())
    print(f"  -> deep - shallow (PCA-50): {dm - dg:+.3f}, bootstrap 95% CI {ci}, P(deep>shallow)={p:.3f}")

    print("=" * 70)
    print("FULL MONITOR on the rich English reference (deep=MiniLM):")
    E = mini; base = alignment(E, ref); d = E.shape[1]
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    inv = max(abs(alignment(E @ Q, ref) - base), abs(alignment(E * 6.0, ref) - base)) < 1e-6
    print(f"  (2) INVARIANCE: base {base:+.4f} -> {'PASS' if inv else 'FAIL'}")
    sd = E.std()
    sens = [round(alignment(E + s * sd * rng.standard_normal(E.shape), ref), 3) for s in [0, 1, 2, 4]]

    def shuf(A, f):
        A2 = A.copy(); k = int(f * len(A)); ix = rng.choice(len(A), k, replace=False)
        A2[ix] = A2[rng.permutation(ix)]; return A2
    shv = [round(alignment(shuf(E, f), ref), 3) for f in [0, 0.5, 1.0]]
    print(f"  (3) SENSITIVITY: noise {sens}  shuffle {shv}  -> {'PASS' if base - sens[-1] > 0.1 and shv[-1] < 0.1 else 'CHECK'}")
    k = int(0.3 * n); C = rng.choice(n, k, replace=False); lab = np.zeros(n); lab[C] = 1
    Ec = E.copy(); Ec[C] = E[rng.permutation(n)][:k]
    a = auc(-per_concept_alignment(Ec, ref), lab)
    print(f"  (5) LOCALIZATION: corrupt {k}/{n} -> ROC-AUC {a:.3f} -> {'PASS' if a > 0.8 else 'WEAK'}")

    print("=" * 70)
    full_ok = inv and shv[-1] < 0.1 and a > 0.8 and p > 0.95
    print(f"VERDICT: full monitor generalizes to English w/ a rich reference: {'PASS' if full_ok else 'PARTIAL'}")
    import json
    json.dump({"n": n, "glove_pca50": round(dg, 4), "mini_pca50": round(dm, 4),
               "deep_minus_shallow": round(dm - dg, 4), "boot_ci": ci, "boot_p": round(p, 3),
               "invariance": bool(inv), "shuffle_to_zero": shv[-1], "localization_auc": round(a, 4),
               "full_generalizes": bool(full_ok)},
              open(os.path.join(HERE, "binder_result.json"), "w"), indent=2)
    print("wrote binder_result.json")


if __name__ == "__main__":
    main()

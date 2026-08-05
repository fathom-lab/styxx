"""B43 — name the twenty, per PREREG_b43_name_the_twenty_2026_08_05.md.

Principal-angle discordant directions between qwen's and llama's top-20 concept-Gram
subspaces; top-loading concepts as the interpretation; cross-seed stability + semantic
coherence permutation gates. CPU, seconds. `--smoke` = D=1 at 40 anchors, INVALID-only.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from run_g0clear import CONCEPTS as C          # noqa: E402
from run_b40 import gram_eigvecs                 # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
K = 20
D = 1 if SMOKE else 3
TOPN = 15


def load(fname):
    return np.asarray(np.load(HERE / fname, allow_pickle=True)["pts"])


def discordant_dirs(UQ, UL, concepts):
    """Principal vectors on the qwen side for the LARGEST principal angles (most discordant)."""
    # SVD of UQ^T UL: singular values = cos(principal angles); small sv = large angle = discordant
    M = UQ.T @ UL
    Uq, s, _ = np.linalg.svd(M)
    order = np.argsort(s)                        # ascending sv => descending angle
    dirs = UQ @ Uq[:, order]                     # qwen-side principal vectors, in concept space
    return dirs                                  # columns ordered most-discordant first


def top_concepts(vec, concepts, n):
    return [concepts[i] for i in np.argsort(-np.abs(vec))[:n]]


def main():
    seeds = [343] if SMOKE else [343, 1001, 1002]
    n_fin = 10 if SMOKE else 70
    XL_all = load("_b31v2_ptsA.npz")
    XQ_all = load("_b31v2_pts_qwen_1p5b.npz")

    per_seed_top1 = {}          # seed -> top-N concepts of the single most-discordant dir
    results = {"prereg": "PREREG_b43_name_the_twenty_2026_08_05.md", "smoke": SMOKE,
               "K": K, "D": D, "topN": TOPN, "seeds": seeds, "directions": {}}
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(C))
        tr_i = idx[n_fin:(n_fin + 40) if SMOKE else len(C)]
        anchors = [C[i] for i in tr_i]
        UQ = gram_eigvecs(XQ_all[tr_i], K)
        UL = gram_eigvecs(XL_all[tr_i], K)
        dirs = discordant_dirs(UQ, UL, anchors)
        seed_dirs = [top_concepts(dirs[:, d], anchors, TOPN) for d in range(D)]
        per_seed_top1[s] = seed_dirs[0]
        if s == seeds[0]:
            results["directions"] = {f"discordant_{d}": seed_dirs[d] for d in range(D)}

    # G1 stability: mean pairwise Jaccard of the top-N set of the MOST discordant dir across seeds
    if len(seeds) > 1:
        js = []
        for a, b in combinations(seeds, 2):
            A, B = set(per_seed_top1[a]), set(per_seed_top1[b])
            js.append(len(A & B) / len(A | B))
        results["mean_jaccard_top15_across_seeds"] = round(float(np.mean(js)), 4)
    else:
        results["mean_jaccard_top15_across_seeds"] = 0.0

    # G2 coherence: permutation test on semantic clustering of the seed-343 most-discordant set
    top = results["directions"]["discordant_0"]
    if not SMOKE:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        rng = np.random.default_rng(343)
        idx0 = rng.permutation(len(C))
        anchors0 = [C[i] for i in idx0[n_fin:]]
        emb = {w: e for w, e in zip(anchors0, st.encode(anchors0, normalize_embeddings=True))}

        def coherence(words):
            E = np.array([emb[w] for w in words])
            G = E @ E.T
            return float(G[np.triu_indices(len(words), 1)].mean())
        obs = coherence(top)
        perm_rng = np.random.default_rng(4301)
        null = [coherence(list(perm_rng.choice(anchors0, TOPN, replace=False)))
                for _ in range(2000)]
        p = (1 + sum(1 for x in null if x >= obs)) / (1 + len(null))
        results["coherence_observed"] = round(obs, 4)
        results["coherence_null_mean"] = round(float(np.mean(null)), 4)
        results["coherence_perm_p"] = round(float(p), 4)
    else:
        results["coherence_perm_p"] = 1.0

    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b43_name_the_twenty_2026_08_05.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b43_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"VERDICT: {results['verdict']}", flush=True)
    print("most-discordant concepts:", results["directions"]["discordant_0"], flush=True)
    print(f"jaccard={results['mean_jaccard_top15_across_seeds']} "
          f"coherence_p={results.get('coherence_perm_p')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

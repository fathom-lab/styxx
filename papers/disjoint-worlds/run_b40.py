"""B40 — the anisotropy signature, per PREREG_b40_anisotropy_signature_2026_08_05.md.

Top-k concept-Gram eigenvector subspace affinity between all 6 unordered model pairs,
compared against the committed b37 legibility matrix. Pure measurement, CPU, seconds.
`--smoke` = k=5 on 40 anchors, INVALID-only.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from run_g0clear import CONCEPTS as C          # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEED = 343
K = 5 if SMOKE else 20
BANKS = {"llama_3b": "_b31v2_ptsA.npz", "llama_1b": "_b31v2_pts_llama_1b.npz",
         "gemma_2b": "_b31v2_pts_gemma_2b.npz", "qwen_1p5b": "_b31v2_pts_qwen_1p5b.npz"}
CLIQUE = ["llama_3b", "llama_1b", "gemma_2b"]


def gram_eigvecs(X, k):
    """Top-k eigenvectors of the double-centered concept Gram (concept-space, label-aligned)."""
    Xc = X - X.mean(0)
    G = Xc @ Xc.T
    n = G.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    G = J @ G @ J
    w, V = np.linalg.eigh(G)
    return V[:, np.argsort(w)[::-1][:k]]        # n x k


def affinity(Ua, Ub, k):
    return float(np.linalg.norm(Ua.T @ Ub, "fro") ** 2 / k)


def main():
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(C))
    n_fin = 10 if SMOKE else 70
    n_tr = 40 if SMOKE else (len(C) - n_fin)
    tr_i = idx[n_fin:n_fin + n_tr]

    U = {}
    for m, f in BANKS.items():
        X = np.asarray(np.load(HERE / f, allow_pickle=True)["pts"])[tr_i]
        U[m] = gram_eigvecs(X, K)

    results = {"prereg": "PREREG_b40_anisotropy_signature_2026_08_05.md", "seed": SEED,
               "smoke": SMOKE, "k": K, "n_anchor_rows": int(n_tr), "pairs": {}}
    for a, b in itertools.combinations(BANKS, 2):
        results["pairs"][f"{a}__{b}"] = round(affinity(U[a], U[b], K), 4)
        print(f">> {a}__{b}: affinity={results['pairs'][f'{a}__{b}']}", flush=True)

    if not SMOKE:
        cl_int = [results["pairs"][f"{a}__{b}"]
                  for a, b in itertools.combinations(CLIQUE, 2)]
        qwen_aff = [results["pairs"].get(f"{c}__qwen_1p5b",
                    results["pairs"].get(f"qwen_1p5b__{c}")) for c in CLIQUE]
        results["clique_internal_affinities"] = cl_int
        results["qwen_to_clique_affinities"] = qwen_aff
        results["min_clique_internal_affinity_minus_max_qwen_affinity"] = round(
            min(cl_int) - max(qwen_aff), 4)
        # legibility from the committed b37 matrix (mean of the two directions)
        b37 = json.loads((HERE / "b37_result.json").read_text())["pairs"]
        affs, discs = [], []
        for a, b in itertools.combinations(BANKS, 2):
            affs.append(results["pairs"][f"{a}__{b}"])
            discs.append((b37[f"{a}__to__{b}"]["seed_acc"] + b37[f"{b}__to__{a}"]["seed_acc"]) / 2)
        results["spearman_affinity_vs_disc"] = round(float(spearmanr(affs, discs).statistic), 4)
        results["disc_means_b37"] = [round(d, 4) for d in discs]
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b40_anisotropy_signature_2026_08_05.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b40_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

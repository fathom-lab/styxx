"""B45 — frame geometry, per PREREG_b45_frame_geometry_2026_08_06.md.

Pairwise principal-angle affinity between label-aligned concept-Gram eigenframes; Haar-random
null. No fitting. `--smoke` = one seed, k=20, INVALID-only.
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from run_g0clear import CONCEPTS as C          # noqa: E402
from run_b40 import gram_eigvecs, affinity       # noqa: E402
from run_b42 import load                         # noqa: E402

SMOKE = "--smoke" in sys.argv
SEEDS = [343] if SMOKE else [343, 1001, 1002, 1003, 1004]
RANKS = [20] if SMOKE else [2, 20]
CLIQUE = ["llama_3b", "gemma_2b", "llama_1b"]
ISLAND = "qwen_1p5b"
BANKS = {"llama_3b": "_b31v2_ptsA.npz", "gemma_2b": "_b31v2_pts_gemma_2b.npz",
         "llama_1b": "_b31v2_pts_llama_1b.npz", "qwen_1p5b": "_b31v2_pts_qwen_1p5b.npz"}
N_NULL = 100 if SMOKE else 1000


def haar_frame(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
    return Q[:, :k]


def main() -> int:
    t0 = time.time()
    X_all = {m: load(f) for m, f in BANKS.items()}
    n_fin = 70
    results = {"prereg": "PREREG_b45_frame_geometry_2026_08_06.md", "smoke": SMOKE,
               "seeds": SEEDS, "ranks": RANKS, "n_null": N_NULL, "pairs": {}, "null_p95": {}}
    models = CLIQUE + [ISLAND]

    clique_meds, qwen_below = {k: [] for k in RANKS}, {k: 0 for k in RANKS}
    all_clique, all_qwen = {k: [] for k in RANKS}, {k: [] for k in RANKS}
    for s in SEEDS:
        idx = np.random.default_rng(s).permutation(len(C))
        tr_i = idx[n_fin:]
        n = len(tr_i)
        for k in RANKS:
            U = {m: gram_eigvecs(X_all[m][tr_i], k) for m in models}
            cl, qw = [], []
            for a, b in combinations(models, 2):
                v = round(affinity(U[a], U[b], k), 4)
                results["pairs"][f"{a}__{b}_k{k}_s{s}"] = v
                (qw if ISLAND in (a, b) else cl).append(v)
            all_clique[k] += cl
            all_qwen[k] += qw
            if float(np.median(qw)) < float(np.median(cl)):
                qwen_below[k] += 1
            print(f">> seed {s} k={k}: clique med {np.median(cl):.4f} "
                  f"qwen med {np.median(qw):.4f}", flush=True)

    for k in RANKS:
        rng = np.random.default_rng(999)
        n = len(np.random.default_rng(SEEDS[0]).permutation(len(C))[n_fin:])
        null = [affinity(haar_frame(n, k, rng), haar_frame(n, k, rng), k)
                for _ in range(N_NULL)]
        results["null_p95"][str(k)] = round(float(np.percentile(null, 95)), 4)
        results[f"null_expectation_k{k}"] = round(k / n, 4)

    if not SMOKE:
        med_cl20 = float(np.median(all_clique[20]))
        results["median_clique_affinity_k20"] = round(med_cl20, 4)
        results["median_qwen_affinity_k20"] = round(float(np.median(all_qwen[20])), 4)
        results["median_clique_affinity_k2"] = round(float(np.median(all_clique[2])), 4)
        results["median_qwen_affinity_k2"] = round(float(np.median(all_qwen[2])), 4)
        results["clique_affinity_minus_null_p95_k20"] = round(
            med_cl20 - results["null_p95"]["20"], 4)
        results["seeds_qwen_below_clique_k20"] = qwen_below[20]

    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b45_frame_geometry_2026_08_06.md").score(results,
                                                                              smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b45_result{'_smoke' if SMOKE else ''}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\n[{time.time()-t0:.0f}s] VERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

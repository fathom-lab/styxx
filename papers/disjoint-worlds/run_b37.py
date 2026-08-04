"""B37 — the mutual-legibility matrix, per PREREG_b37_legibility_matrix_2026_08_04.md.

All 12 directed pairs of the four banked models: label-free discovery + MLP read (b34-v3
pipeline verbatim, seed 343), plus three per-pair predictors (RSA, true-correspondence
kNN-Jaccard, spectral profile similarity) and the pre-stated Spearman comparison.
CPU-from-cache. `--smoke` = 3 pairs at 40/10, INVALID-only.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from run_g0clear import CONCEPTS as C          # noqa: E402
import run_disjoint_worlds as R                 # distmat  # noqa: E402
from styxx_transfer import TransferMap          # noqa: E402
from run_b31v2 import fit_mlp                    # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEED = 343
MODELS = {"llama_3b": "_b31v2_ptsA.npz", "llama_1b": "_b31v2_pts_llama_1b.npz",
          "gemma_2b": "_b31v2_pts_gemma_2b.npz", "qwen_1p5b": "_b31v2_pts_qwen_1p5b.npz"}
G0_LOCKED_SOURCE = {"llama_3b"}          # the only source with a G0-locked read layer


def load(fname):
    z = np.load(HERE / fname, allow_pickle=True)
    return {c: z["pts"][i] for i, c in enumerate(C)}


def knn_jaccard(XA, XB, k=10):
    """Mean Jaccard of each concept's k-NN set across the two spaces (TRUE correspondence)."""
    def knn(X):
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        np.fill_diagonal(D, np.inf)
        return [set(np.argsort(row)[:k]) for row in D]
    a, b = knn(XA), knn(XB)
    return float(np.mean([len(a[i] & b[i]) / len(a[i] | b[i]) for i in range(len(a))]))


def spectral_sim(XA, XB, n=50):
    def prof(X):
        Xc = X - X.mean(0)
        s = np.linalg.svd(Xc, compute_uv=False)[:n]
        p = np.log(s / s[0] + 1e-12)
        return p / (np.linalg.norm(p) + 1e-12)
    a, b = prof(XA), prof(XB)
    m = min(len(a), len(b))
    return float(a[:m] @ b[:m])


def main():
    kstar = json.loads((HERE / "g0clear_result_llama3b.json").read_text())["locked"]["k"]
    pts = {name: load(f) for name, f in MODELS.items()}
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(C))
    n_fin = 10 if SMOKE else 70
    n_tr = 40 if SMOKE else (len(C) - n_fin)
    fin = [C[i] for i in idx[:n_fin]]
    tr = [C[i] for i in idx[n_fin:n_fin + n_tr]]

    pairs = list(itertools.permutations(MODELS, 2))
    if SMOKE:
        pairs = pairs[:3]

    results = {"prereg": "PREREG_b37_legibility_matrix_2026_08_04.md", "seed": SEED,
               "smoke": SMOKE, "n_tr": len(tr), "n_heldout": len(fin),
               "chance": round(1 / len(fin), 4), "pairs": {}}
    rows = []
    for src, tgt in pairs:
        t0 = time.time()
        XA = np.array([pts[src][c] for c in tr])
        XB = np.array([pts[tgt][c] for c in tr])
        fin_A = pts[src]
        fin_ptsB = np.array([pts[tgt][c] for c in fin])
        perm = rng.permutation(len(tr)); XBs = XB[perm]; true_col = np.argsort(perm)
        tm = TransferMap.fit(XA, XBs, k=kstar)
        MA = np.stack([tm.transfer_point(x) for x in XA])
        _, col = linear_sum_assignment(np.linalg.norm(MA[:, None, :] - XBs[None, :, :], axis=-1))
        seed_acc = float((col == true_col).mean())
        fn, _ = fit_mlp(XA, XBs[col], seed=SEED)
        read = sum(1 for i, c in enumerate(fin)
                   if int(np.argmin(np.linalg.norm(fin_ptsB - fn(fin_A[c]), axis=1))) == i) / len(fin)
        nf, _ = fit_mlp(XA, XBs[rng.permutation(len(tr))], seed=SEED)
        null = sum(1 for i, c in enumerate(fin)
                   if int(np.argmin(np.linalg.norm(fin_ptsB - nf(fin_A[c]), axis=1))) == i) / len(fin)
        tri = np.triu_indices(len(tr), 1)
        rsa = float(np.corrcoef(R.distmat(XA)[tri], R.distmat(XB)[tri])[0, 1])
        knn = knn_jaccard(XA, XB)
        spec = spectral_sim(XA, XB)
        key = f"{src}__to__{tgt}"
        results["pairs"][key] = {
            "seed_acc": round(seed_acc, 4), "read": round(read, 4), "null": round(null, 4),
            "rsa": round(rsa, 4), "knn_jaccard": round(knn, 4), "spectral_sim": round(spec, 4),
            "source_layer_status": ("g0_locked" if src in G0_LOCKED_SOURCE
                                    else "source_layer_unvalidated")}
        rows.append((key, seed_acc, rsa, knn))
        print(f">> {key}: disc={seed_acc:.4f} read={read:.4f} null={null:.4f} | "
              f"rsa={rsa:.3f} knn={knn:.3f} spec={spec:.3f} [{time.time()-t0:.0f}s]", flush=True)

    if not SMOKE:
        accs = [results["pairs"][k]["seed_acc"] for k, *_ in rows]
        rsas = [results["pairs"][k]["rsa"] for k, *_ in rows]
        knns = [results["pairs"][k]["knn_jaccard"] for k, *_ in rows]
        sp_knn = float(spearmanr(knns, accs).statistic)
        sp_rsa = float(spearmanr(rsas, accs).statistic)
        asym = []
        for a, b in itertools.combinations(MODELS, 2):
            asym.append(abs(results["pairs"][f"{a}__to__{b}"]["seed_acc"]
                            - results["pairs"][f"{b}__to__{a}"]["seed_acc"]))
        results.update({
            "llama3b_to_gemma_seed_acc": results["pairs"]["llama_3b__to__gemma_2b"]["seed_acc"],
            "llama3b_to_qwen_seed_acc": results["pairs"]["llama_3b__to__qwen_1p5b"]["seed_acc"],
            "spearman_knn_vs_disc": round(sp_knn, 4),
            "spearman_rsa_vs_disc": round(sp_rsa, 4),
            "knn_minus_rsa_spearman": round(sp_knn - sp_rsa, 4),
            "median_abs_asymmetry": round(float(np.median(asym)), 4)})
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b37_legibility_matrix_2026_08_04.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b37_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

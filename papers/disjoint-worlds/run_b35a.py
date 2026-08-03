"""B35-a — seed stability of the label-free cross-family read, per
PREREG_b35_seed_stability_2026_08_03.md. The b34-v3 method verbatim, over 5 fresh seeds;
gates on the medians / null-mean. CPU-from-cache. `--smoke` = 2 seeds x 40/10, INVALID-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from run_g0clear import CONCEPTS as C          # noqa: E402
from styxx_transfer import TransferMap          # noqa: E402
from run_b31v2 import fit_mlp                    # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEEDS = [1001, 1002] if SMOKE else [1001, 1002, 1003, 1004, 1005]
TARGETS = ["llama_1b", "gemma_2b", "qwen_1p5b"]


def load(tag):
    z = np.load(HERE / f"_b31v2_pts_{tag}.npz", allow_pickle=True)
    return {c: z["pts"][i] for i, c in enumerate(C)}


def read_top1(fin, ptsA, fin_ptsB, fn):
    return sum(1 for i, c in enumerate(fin)
               if int(np.argmin(np.linalg.norm(fin_ptsB - fn(ptsA[c]), axis=1))) == i) / len(fin)


def main():
    kstar = json.loads((HERE / "g0clear_result_llama3b.json").read_text())["locked"]["k"]
    ptsA = load("llama_1b_A") if False else None
    zA = np.load(HERE / "_b31v2_ptsA.npz", allow_pickle=True)
    ptsA = {c: zA["pts"][i] for i, c in enumerate(C)}
    ptsB = {tag: load(tag) for tag in TARGETS}

    n_fin, n_tr = (10, 40) if SMOKE else (70, len(C) - 70)
    per_seed, null_reads = [], []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(C))
        fin = [C[i] for i in idx[:n_fin]]
        tr = [C[i] for i in idx[n_fin:n_fin + n_tr]]
        XA = np.array([ptsA[c] for c in tr])
        row = {"seed": s}
        for tag in TARGETS:
            t0 = time.time()
            XB = np.array([ptsB[tag][c] for c in tr])
            fin_ptsB = np.array([ptsB[tag][c] for c in fin])
            perm = rng.permutation(len(tr)); XBs = XB[perm]; true_col = np.argsort(perm)
            tm = TransferMap.fit(XA, XBs, k=kstar)
            MA = np.stack([tm.transfer_point(x) for x in XA])
            _, col = linear_sum_assignment(np.linalg.norm(MA[:, None, :] - XBs[None, :, :], axis=-1))
            seed_acc = float((col == true_col).mean())
            fn, _ = fit_mlp(XA, XBs[col], seed=s)
            read = read_top1(fin, ptsA, fin_ptsB, fn)
            nf, _ = fit_mlp(XA, XBs[rng.permutation(len(tr))], seed=s)
            null = read_top1(fin, ptsA, fin_ptsB, nf)
            null_reads.append(null)
            row[tag] = {"seed_acc": round(seed_acc, 4), "read": round(read, 4),
                        "null": round(null, 4)}
            print(f"  seed {s} {tag}: seed_acc={seed_acc:.3f} read={read:.4f} "
                  f"null={null:.4f} [{time.time()-t0:.0f}s]", flush=True)
        per_seed.append(row)

    gem = [r["gemma_2b"]["read"] for r in per_seed]
    lla = [r["llama_1b"]["seed_acc"] for r in per_seed]
    results = {"prereg": "PREREG_b35_seed_stability_2026_08_03.md", "smoke": SMOKE,
               "seeds": SEEDS, "per_seed": per_seed,
               "median_gemma_read": round(float(np.median(gem)), 4),
               "min_gemma_read": round(min(gem), 4), "max_gemma_read": round(max(gem), 4),
               "median_llama_seed_acc": round(float(np.median(lla)), 4),
               "mean_null_top1": round(float(np.mean(null_reads)), 4),
               "chance": round(1 / n_fin, 4)}
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b35_seed_stability_2026_08_03.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b35a_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    print(f"gemma read across seeds: median {results['median_gemma_read']} "
          f"[{results['min_gemma_read']}, {results['max_gemma_read']}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

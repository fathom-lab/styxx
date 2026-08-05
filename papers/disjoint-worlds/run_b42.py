"""B42 — the bridge dose curve, per PREREG_b42_bridge_dose_2026_08_05.md.

The B41 surgery swept over rank k and seed, each with a matched random-frame null. Hardens
the single-seed bridge into a replication + dose curve, finds the minimum sufficient rank k*.
CPU-from-cache. `--smoke` = k in {1,20} x 2 seeds, INVALID-only.
"""
from __future__ import annotations

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
from styxx_transfer import TransferMap          # noqa: E402
from run_b40 import gram_eigvecs                 # noqa: E402
from run_b41 import surgery                      # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
RANKS = [1, 20] if SMOKE else [1, 2, 3, 5, 8, 12, 20, 40]
SEEDS = [343, 1001] if SMOKE else [343, 1001, 1002, 1003, 1004]


def load(fname):
    return np.asarray(np.load(HERE / fname, allow_pickle=True)["pts"])


def discover(XA, XB, kstar, rng):
    perm = rng.permutation(len(XA)); XBs = XB[perm]; true_col = np.argsort(perm)
    tm = TransferMap.fit(XA, XBs, k=kstar)
    MA = np.stack([tm.transfer_point(x) for x in XA])
    _, col = linear_sum_assignment(np.linalg.norm(MA[:, None, :] - XBs[None, :, :], axis=-1))
    return float((col == true_col).mean())


def main():
    kstar = json.loads((HERE / "g0clear_result_llama3b.json").read_text())["locked"]["k"]
    XL_all = load("_b31v2_ptsA.npz")
    XQ_all = load("_b31v2_pts_qwen_1p5b.npz")
    n_fin = 10 if SMOKE else 70

    results = {"prereg": "PREREG_b42_bridge_dose_2026_08_05.md", "smoke": SMOKE,
               "ranks": RANKS, "seeds": SEEDS, "grid": {}}
    for si, s in enumerate(SEEDS):
        split_rng = np.random.default_rng(s)
        idx = split_rng.permutation(len(C))
        tr_i = idx[n_fin:(n_fin + 40) if SMOKE else len(C)]
        XL, XQ = XL_all[tr_i], XQ_all[tr_i]
        for k in RANKS:
            t0 = time.time()
            UL, UQ = gram_eigvecs(XL, k), gram_eigvecs(XQ, k)
            XB = surgery(XQ, UQ, UL)
            Qr, _ = np.linalg.qr(np.random.default_rng(7000 + si * 100 + k).standard_normal((len(tr_i), k)))
            XBn = surgery(XQ, UQ, Qr)
            drng = np.random.default_rng(s)
            b = discover(XL, XB, kstar, np.random.default_rng(s + 500000))
            n = discover(XL, XBn, kstar, np.random.default_rng(s + 900000))
            results["grid"][f"k{k}_s{s}"] = {"bridge": round(b, 4), "null": round(n, 4)}
            print(f">> k={k} seed={s}: bridge={b:.4f} null={n:.4f} [{time.time()-t0:.0f}s]",
                  flush=True)

    if not SMOKE:
        g = results["grid"]
        bridges = {k: [g[f"k{k}_s{s}"]["bridge"] for s in SEEDS] for k in RANKS}
        nulls = {k: [g[f"k{k}_s{s}"]["null"] for s in SEEDS] for k in RANKS}
        med_b = {k: float(np.median(v)) for k, v in bridges.items()}
        results["median_bridge_by_k"] = {str(k): round(v, 4) for k, v in med_b.items()}
        results["median_null_by_k"] = {str(k): round(float(np.median(nulls[k])), 4) for k in RANKS}
        results["min_bridge_disc_at_k20"] = round(min(bridges[20]), 4)
        results["max_null_disc_at_k20"] = round(max(nulls[20]), 4)
        results["spearman_medianbridge_vs_k"] = round(float(
            spearmanr(RANKS, [med_b[k] for k in RANKS]).statistic), 4)
        kstar_min = next((k for k in RANKS
                          if med_b[k] >= 0.30 and np.median(nulls[k]) <= 0.15), None)
        results["k_star_min_sufficient_rank"] = kstar_min
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b42_bridge_dose_2026_08_05.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b42_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    if not SMOKE:
        print(f"k* (min sufficient rank): {results['k_star_min_sufficient_rank']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

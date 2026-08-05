"""B44 — the structured-but-wrong donor, per PREREG_b44_wrong_donor_2026_08_05.md.

The B41/B42 surgery with the donor frame computed from the WRONG model (gemma_2b, llama_1b),
plus the true-donor bridge as positive control at k=20. CPU-from-cache.
`--smoke` = 1 seed x k=20, wrong-donor G only, INVALID-only.
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
from run_b40 import gram_eigvecs                 # noqa: E402
from run_b41 import surgery                      # noqa: E402
from run_b42 import discover, load               # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
RANKS = [20] if SMOKE else [2, 20]
SEEDS = [343] if SMOKE else [343, 1001, 1002, 1003, 1004]
DONORS = {"G": "_b31v2_pts_gemma_2b.npz"} if SMOKE else \
         {"G": "_b31v2_pts_gemma_2b.npz", "L1": "_b31v2_pts_llama_1b.npz"}


def main():
    kstar = json.loads((HERE / "g0clear_result_llama3b.json").read_text())["locked"]["k"]
    XL_all = load("_b31v2_ptsA.npz")
    XQ_all = load("_b31v2_pts_qwen_1p5b.npz")
    XD_all = {d: load(f) for d, f in DONORS.items()}
    n_fin = 10 if SMOKE else 70

    results = {"prereg": "PREREG_b44_wrong_donor_2026_08_05.md", "smoke": SMOKE,
               "ranks": RANKS, "seeds": SEEDS, "donors": sorted(DONORS), "grid": {}}
    for si, s in enumerate(SEEDS):
        split_rng = np.random.default_rng(s)
        idx = split_rng.permutation(len(C))
        tr_i = idx[n_fin:(n_fin + 40) if SMOKE else len(C)]
        XL, XQ = XL_all[tr_i], XQ_all[tr_i]
        for k in RANKS:
            UQ = gram_eigvecs(XQ, k)
            for d in sorted(DONORS):
                t0 = time.time()
                UD = gram_eigvecs(XD_all[d][tr_i], k)
                w = discover(XL, surgery(XQ, UQ, UD), kstar, np.random.default_rng(s + 900000))
                results["grid"][f"k{k}_s{s}_{d}"] = {"wrong": round(w, 4)}
                print(f">> k={k} seed={s} donor={d}: wrong={w:.4f} [{time.time()-t0:.0f}s]",
                      flush=True)
            if k == 20 and not SMOKE:
                t0 = time.time()
                UL = gram_eigvecs(XL, k)
                b = discover(XL, surgery(XQ, UQ, UL), kstar, np.random.default_rng(s + 500000))
                results["grid"][f"k{k}_s{s}_bridge"] = {"bridge": round(b, 4)}
                print(f">> k={k} seed={s} BRIDGE: {b:.4f} [{time.time()-t0:.0f}s]", flush=True)

    if not SMOKE:
        g = results["grid"]
        wrong20 = [g[f"k20_s{s}_{d}"]["wrong"] for s in SEEDS for d in sorted(DONORS)]
        results["min_bridge_disc_at_k20"] = round(min(g[f"k20_s{s}_bridge"]["bridge"]
                                                      for s in SEEDS), 4)
        results["max_wrong_donor_disc_at_k20"] = round(max(wrong20), 4)
        results["min_wrong_donor_disc_at_k20"] = round(min(wrong20), 4)
        for d in sorted(DONORS):
            for k in RANKS:
                med = float(np.median([g[f"k{k}_s{s}_{d}"]["wrong"] for s in SEEDS]))
                results[f"median_wrong_{d}_k{k}"] = round(med, 4)
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b44_wrong_donor_2026_08_05.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b44_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

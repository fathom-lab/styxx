"""B39 — whitening rescue, per PREREG_b39_whitening_rescue_2026_08_04.md.

Per-model treatments (raw / ZCA-shrink whiten / per-dim standardize) before the b34-v3
discovery machinery. 9 discovery fits, CPU-from-cache. Uses the committed
styxx.crossmind.zca_shrink at its B29-preregistered lambda=0.5.
`--smoke` = T0/T1 on one pair at 40/10, INVALID-only.
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
from styxx.crossmind import zca_shrink           # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEED = 343
BANKS = {"llama_3b": "_b31v2_ptsA.npz", "gemma_2b": "_b31v2_pts_gemma_2b.npz",
         "qwen_1p5b": "_b31v2_pts_qwen_1p5b.npz"}
CELLS = [("llama_3b", "qwen_1p5b"), ("llama_3b", "gemma_2b"), ("qwen_1p5b", "llama_3b")]


def load(fname):
    return np.asarray(np.load(HERE / fname, allow_pickle=True)["pts"])


def treat(X_all, tr_idx, mode):
    """Fit the treatment on anchor rows, apply to all rows."""
    if mode == "t0":
        return X_all
    tr = X_all[tr_idx]
    if mode == "t2":
        mu, sd = tr.mean(0), tr.std(0) + 1e-8
        return (X_all - mu) / sd
    mu, W = zca_shrink(tr, lam=0.5)
    return (X_all - mu) @ W


def main():
    kstar = json.loads((HERE / "g0clear_result_llama3b.json").read_text())["locked"]["k"]
    X = {m: load(f) for m, f in BANKS.items()}
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(C))
    n_fin = 10 if SMOKE else 70
    n_tr = 40 if SMOKE else (len(C) - n_fin)
    tr_i = idx[n_fin:n_fin + n_tr]

    treatments = ["t0", "t1"] if SMOKE else ["t0", "t1", "t2"]
    cells = CELLS[:1 if SMOKE else None] if SMOKE else CELLS
    results = {"prereg": "PREREG_b39_whitening_rescue_2026_08_04.md", "seed": SEED,
               "smoke": SMOKE, "n_tr": int(n_tr), "lam": 0.5, "cells": {}}
    for t in treatments:
        Xt = {m: treat(X[m], tr_i, t) for m in BANKS}
        for src, tgt in (CELLS[:1] if SMOKE else CELLS):
            t0c = time.time()
            XA = Xt[src][tr_i]
            XB = Xt[tgt][tr_i]
            perm = rng.permutation(len(XA)); XBs = XB[perm]; true_col = np.argsort(perm)
            tm = TransferMap.fit(XA, XBs, k=kstar)
            MA = np.stack([tm.transfer_point(x) for x in XA])
            _, col = linear_sum_assignment(np.linalg.norm(MA[:, None, :] - XBs[None, :, :], axis=-1))
            acc = float((col == true_col).mean())
            key = f"{t}_{src}__to__{tgt}"
            results["cells"][key] = round(acc, 4)
            print(f">> {key}: disc={acc:.4f} [{time.time()-t0c:.0f}s]", flush=True)

    if not SMOKE:
        results["t0_llama3b_to_gemma"] = results["cells"]["t0_llama_3b__to__gemma_2b"]
        results["t1_llama3b_to_gemma"] = results["cells"]["t1_llama_3b__to__gemma_2b"]
        results["t1_llama3b_to_qwen"] = results["cells"]["t1_llama_3b__to__qwen_1p5b"]
        results["t0_llama3b_to_qwen"] = results["cells"]["t0_llama_3b__to__qwen_1p5b"]
        results["t2_llama3b_to_qwen"] = results["cells"]["t2_llama_3b__to__qwen_1p5b"]
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b39_whitening_rescue_2026_08_04.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b39_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

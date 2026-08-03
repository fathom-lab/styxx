"""B35-c null replication — the remedy the frozen prereg pre-wrote.

b35-c scored INVALID__null_artifact: one coincidental null hit (1/70 at 1/462 chance) tripped
the G2 floor. The prereg's own instruction for that branch: "a re-run with a second null seed
reported beside it, not a bar move." This script re-draws the pairing-shuffled null under FIVE
independent null seeds per target, holding the discovery + read path byte-identical, and reports
the null distribution. The b35-c reads stay UNLICENSED unless a properly-specified successor
prereg gates them; this run only characterises the null.
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

SEED = 343
NULL_SEEDS = [9001, 9002, 9003, 9004, 9005]
TARGETS = ["llama_1b", "gemma_2b", "qwen_1p5b"]


def load(tag):
    z = np.load(HERE / f"_b31v2_pts_{tag}.npz", allow_pickle=True)
    return {c: z["pts"][i] for i, c in enumerate(C)}


def main():
    kstar = json.loads((HERE / "g0clear_result_llama3b.json").read_text())["locked"]["k"]
    zA = np.load(HERE / "_b31v2_ptsA.npz", allow_pickle=True)
    ptsA = {c: zA["pts"][i] for i, c in enumerate(C)}

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(C))
    fin = [C[i] for i in idx[:70]]
    tr = [C[i] for i in idx[70:]]
    vocab = fin + tr
    XA = np.array([ptsA[c] for c in tr])
    chance = 1 / len(vocab)

    out = {"purpose": "b35-c G2 null replication per the frozen prereg's INVALID branch",
           "read_path": "unchanged from b35c_result.json (reads remain UNLICENSED here)",
           "null_seeds": NULL_SEEDS, "vocab_size": len(vocab),
           "chance462": round(chance, 5), "targets": {}}

    for tag in TARGETS:
        t0 = time.time()
        ptsB = load(tag)
        XB = np.array([ptsB[c] for c in tr])
        vocabB = np.array([ptsB[c] for c in vocab])
        perm = rng.permutation(len(tr))
        XBs = XB[perm]

        def read462(mapper):
            return sum(1 for i, c in enumerate(fin)
                       if int(np.argmin(np.linalg.norm(vocabB - mapper(ptsA[c]), axis=1))) == i) / len(fin)

        nulls = []
        for ns in NULL_SEEDS:
            r = np.random.default_rng(ns)
            nf, _ = fit_mlp(XA, XBs[r.permutation(len(tr))], seed=ns)
            nulls.append(read462(nf))
        hits = [int(round(n * len(fin))) for n in nulls]
        out["targets"][tag] = {
            "null_top1_by_seed": [round(n, 5) for n in nulls],
            "null_hits_of_70": hits,
            "mean_null": round(float(np.mean(nulls)), 5),
            "median_null": round(float(np.median(nulls)), 5),
            "max_null": round(float(max(nulls)), 5),
            "n_seeds_with_zero_hits": int(sum(1 for h in hits if h == 0))}
        print(f">> {tag}: null hits {hits} mean={np.mean(nulls):.5f} "
              f"(chance {chance:.5f}) [{time.time()-t0:.0f}s]", flush=True)

    allnulls = [n for t in out["targets"].values() for n in t["null_top1_by_seed"]]
    out["pooled_mean_null"] = round(float(np.mean(allnulls)), 5)
    out["pooled_draws"] = len(allnulls)
    out["expected_hits_per_draw_at_chance"] = round(70 * chance, 3)
    (HERE / "b35c_null_replication.json").write_text(json.dumps(out, indent=2) + "\n",
                                                     encoding="utf-8")
    print(f"\npooled mean null over {len(allnulls)} draws: {out['pooled_mean_null']} "
          f"(chance {chance:.5f}, expected hits/draw {out['expected_hits_per_draw_at_chance']})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

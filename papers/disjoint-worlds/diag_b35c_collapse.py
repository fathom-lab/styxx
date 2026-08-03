"""Diagnostic: does the b35-c null mode-collapse?

Hypothesis from the null replication (gemma null = exactly 1 hit on 5/5 seeds, p~4e-5 under
Poisson): an MLP fit on SHUFFLED pairs learns the target centroid, so every query maps to
nearly the same point and argmin over the 462-vocab returns the SAME entry for all 70 queries.
If that entry happens to be one of the 70 held-out queries, the null scores exactly 1 hit
deterministically -- a structural floor, not a coincidence, and my prereg's Poisson assumption
was wrong.

Test: count DISTINCT predicted vocab indices for the null map vs the real map.
"""
from __future__ import annotations

import json
import sys
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
    out = {"hypothesis": "shuffled-pairing MLP mode-collapses to the target centroid",
           "vocab_size": len(vocab), "n_queries": len(fin), "targets": {}}

    for tag in ["llama_1b", "gemma_2b", "qwen_1p5b"]:
        z = np.load(HERE / f"_b31v2_pts_{tag}.npz", allow_pickle=True)
        ptsB = {c: z["pts"][i] for i, c in enumerate(C)}
        XB = np.array([ptsB[c] for c in tr])
        vocabB = np.array([ptsB[c] for c in vocab])
        perm = rng.permutation(len(tr)); XBs = XB[perm]; true_col = np.argsort(perm)

        tm = TransferMap.fit(XA, XBs, k=kstar)
        MA = np.stack([tm.transfer_point(x) for x in XA])
        _, col = linear_sum_assignment(np.linalg.norm(MA[:, None, :] - XBs[None, :, :], axis=-1))
        real_fn, _ = fit_mlp(XA, XBs[col], seed=SEED)
        null_fn, _ = fit_mlp(XA, XBs[rng.permutation(len(tr))], seed=SEED)

        def preds(fn):
            return [int(np.argmin(np.linalg.norm(vocabB - fn(ptsA[c]), axis=1))) for c in fin]

        pr, pn = preds(real_fn), preds(null_fn)
        # spread of the mapped outputs themselves (collapse => tiny spread)
        outs_n = np.stack([null_fn(ptsA[c]) for c in fin])
        outs_r = np.stack([real_fn(ptsA[c]) for c in fin])
        spread = lambda a: float(np.mean(np.linalg.norm(a - a.mean(0), axis=1)))
        out["targets"][tag] = {
            "real_distinct_predictions": len(set(pr)),
            "null_distinct_predictions": len(set(pn)),
            "null_modal_prediction_share": round(max(pn.count(v) for v in set(pn)) / len(pn), 4),
            "null_output_spread": round(spread(outs_n), 4),
            "real_output_spread": round(spread(outs_r), 4),
            "null_hits": sum(1 for i, p in enumerate(pn) if p == i),
            "real_hits": sum(1 for i, p in enumerate(pr) if p == i)}
        t = out["targets"][tag]
        print(f">> {tag}: null distinct={t['null_distinct_predictions']}/70 "
              f"(modal share {t['null_modal_prediction_share']}), real distinct="
              f"{t['real_distinct_predictions']}/70 | null spread {t['null_output_spread']} "
              f"vs real {t['real_output_spread']}", flush=True)

    collapsed = [t for t, v in out["targets"].items() if v["null_distinct_predictions"] <= 3]
    out["verdict"] = ("COLLAPSE_CONFIRMED" if collapsed else "NO_COLLAPSE")
    out["collapsed_targets"] = collapsed
    (HERE / "b35c_collapse_diagnostic.json").write_text(json.dumps(out, indent=2) + "\n",
                                                        encoding="utf-8")
    print(f"\n{out['verdict']}: {collapsed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

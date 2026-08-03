"""B35-c — open-vocabulary readout, per PREREG_b35c_open_vocab_2026_08_03.md.

b34-v3 pipeline verbatim (seed 343), but each held-out query is scored against ALL 462
concept points in the target space (anchors + held-outs). Chance 1/462. CPU-from-cache.
`--smoke` = 40/10 split scored against its 50, INVALID-only.
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
SEED = 343
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
    n_fin = 10 if SMOKE else 70
    n_tr = 40 if SMOKE else (len(C) - n_fin)
    fin = [C[i] for i in idx[:n_fin]]
    tr = [C[i] for i in idx[n_fin:n_fin + n_tr]]
    vocab = fin + tr                             # the full open candidate set
    XA = np.array([ptsA[c] for c in tr])

    results = {"prereg": "PREREG_b35c_open_vocab_2026_08_03.md", "seed": SEED,
               "smoke": SMOKE, "n_heldout": len(fin), "vocab_size": len(vocab),
               "chance462": round(1 / len(vocab), 5), "targets": {}}
    nulls = []
    for tag in TARGETS:
        t0 = time.time()
        ptsB = load(tag)
        XB = np.array([ptsB[c] for c in tr])
        vocabB = np.array([ptsB[c] for c in vocab])
        perm = rng.permutation(len(tr)); XBs = XB[perm]; true_col = np.argsort(perm)
        tm = TransferMap.fit(XA, XBs, k=kstar)
        MA = np.stack([tm.transfer_point(x) for x in XA])
        _, col = linear_sum_assignment(np.linalg.norm(MA[:, None, :] - XBs[None, :, :], axis=-1))
        seed_acc = float((col == true_col).mean())
        fn, _ = fit_mlp(XA, XBs[col], seed=SEED)

        def read462(mapper):
            hits = 0
            for i, c in enumerate(fin):
                pred = int(np.argmin(np.linalg.norm(vocabB - mapper(ptsA[c]), axis=1)))
                hits += (pred == i)              # query's own slot is index i in vocab
            return hits / len(fin)

        r462 = read462(fn)
        nf, _ = fit_mlp(XA, XBs[rng.permutation(len(tr))], seed=SEED)
        n462 = read462(nf)
        nulls.append(n462)
        results["targets"][tag] = {"seed_acc": round(seed_acc, 4),
                                   "read462": round(r462, 4), "null462": round(n462, 4),
                                   "x_chance": round(r462 * len(vocab), 1)}
        print(f">> {tag}: seed_acc={seed_acc:.3f} read462={r462:.4f} "
              f"({r462*len(vocab):.0f}x) null462={n462:.4f} [{time.time()-t0:.0f}s]", flush=True)

    results["max_null462"] = round(max(nulls), 5)
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b35c_open_vocab_2026_08_03.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b35c_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""B41 — the bridge, per PREREG_b41_bridge_2026_08_05.md.

Rank-20 concept-space surgery: replace the target's dominant contrast pattern with the
source's (label-aligned intervention ceiling, declared), then run the LABEL-FREE discovery
machinery on the result. Arms: baseline / bridge / random-frame null / gemma no-harm.
CPU-from-cache. `--smoke` = k=5 at 40 anchors, INVALID-only.
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
from run_b40 import gram_eigvecs                 # the committed B40 object  # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEED = 343
K = 5 if SMOKE else 20


def load(fname):
    return np.asarray(np.load(HERE / fname, allow_pickle=True)["pts"])


def surgery(X_T, U_T, U_S):
    """Rank-k concept-space swap: keep the target's loadings, express through U_S's pattern."""
    L = U_T.T @ X_T                       # k x d  (the target's own contrast loadings)
    return X_T - U_T @ L + U_S @ L


def discover(XA, XB, kstar, rng):
    perm = rng.permutation(len(XA)); XBs = XB[perm]; true_col = np.argsort(perm)
    tm = TransferMap.fit(XA, XBs, k=kstar)
    MA = np.stack([tm.transfer_point(x) for x in XA])
    _, col = linear_sum_assignment(np.linalg.norm(MA[:, None, :] - XBs[None, :, :], axis=-1))
    return float((col == true_col).mean())


def main():
    kstar = json.loads((HERE / "g0clear_result_llama3b.json").read_text())["locked"]["k"]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(C))
    n_fin = 10 if SMOKE else 70
    n_tr = 40 if SMOKE else (len(C) - n_fin)
    tr_i = idx[n_fin:n_fin + n_tr]

    XL = load("_b31v2_ptsA.npz")[tr_i]
    XQ = load("_b31v2_pts_qwen_1p5b.npz")[tr_i]
    XG = load("_b31v2_pts_gemma_2b.npz")[tr_i]
    UL, UQ, UG = (gram_eigvecs(X, K) for X in (XL, XQ, XG))
    Qrand, _ = np.linalg.qr(np.random.default_rng(4101).standard_normal((len(tr_i), K)))

    arms = {
        "a0_baseline_disc": (XL, XQ),
        "a1_bridge_disc": (XL, surgery(XQ, UQ, UL)),
        "a2_random_frame_disc": (XL, surgery(XQ, UQ, Qrand)),
        "a3_gemma_bridged_disc": (XL, surgery(XG, UG, UL)),
    }
    results = {"prereg": "PREREG_b41_bridge_2026_08_05.md", "seed": SEED, "smoke": SMOKE,
               "k": K, "n_anchor_rows": int(n_tr)}
    for name, (A, B) in arms.items():
        t0 = time.time()
        results[name] = round(discover(A, B, kstar, rng), 4)
        print(f">> {name}: {results[name]} [{time.time()-t0:.0f}s]", flush=True)
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b41_bridge_2026_08_05.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b41_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

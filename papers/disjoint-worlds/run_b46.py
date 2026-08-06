"""B46 — the cliff function, per PREREG_b46_cliff_function_2026_08_06.md.

Interpolated-frame surgery: dose t blends the island's frame toward the reader's (QR-
orthonormalized), discovery per (t, seed). `--smoke` = t in {0,1} x 1 seed, INVALID-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from run_g0clear import CONCEPTS as C          # noqa: E402
from run_b40 import gram_eigvecs                 # noqa: E402
from run_b42 import discover, load               # noqa: E402

SMOKE = "--smoke" in sys.argv
DOSES = [0.0, 1.0] if SMOKE else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEEDS = [343] if SMOKE else [343, 1001, 1002]
K = 20


def blended_surgery(X_T, U_T, U_S, t: float):
    if t == 0.0:
        return X_T
    Ut, _ = np.linalg.qr((1 - t) * U_T + t * U_S)
    L = U_T.T @ X_T
    return X_T - U_T @ L + Ut[:, :U_T.shape[1]] @ L


def main() -> int:
    kstar = json.loads((HERE / "g0clear_result_llama3b.json").read_text())["locked"]["k"]
    XL_all = load("_b31v2_ptsA.npz")
    XQ_all = load("_b31v2_pts_qwen_1p5b.npz")
    n_fin = 10 if SMOKE else 70

    results = {"prereg": "PREREG_b46_cliff_function_2026_08_06.md", "smoke": SMOKE,
               "doses": DOSES, "seeds": SEEDS, "k": K, "grid": {}}
    for s in SEEDS:
        idx = np.random.default_rng(s).permutation(len(C))
        tr_i = idx[n_fin:(n_fin + 40) if SMOKE else len(C)]
        XL, XQ = XL_all[tr_i], XQ_all[tr_i]
        UL, UQ = gram_eigvecs(XL, K), gram_eigvecs(XQ, K)
        for t in DOSES:
            t0 = time.time()
            d = discover(XL, blended_surgery(XQ, UQ, UL, t), kstar,
                         np.random.default_rng(s + 500000))
            results["grid"][f"t{t}_s{s}"] = round(d, 4)
            print(f">> t={t} seed={s}: disc={d:.4f} [{time.time()-t0:.0f}s]", flush=True)

    if not SMOKE:
        g = results["grid"]
        med = {t: float(np.median([g[f"t{t}_s{s}"] for s in SEEDS])) for t in DOSES}
        results["median_disc_by_t"] = {str(t): round(v, 4) for t, v in med.items()}
        results["max_disc_at_t0"] = round(max(g[f"t0.0_s{s}"] for s in SEEDS), 4)
        results["min_disc_at_t1"] = round(min(g[f"t1.0_s{s}"] for s in SEEDS), 4)
        results["spearman_mediandisc_vs_t"] = round(float(
            spearmanr(DOSES, [med[t] for t in DOSES]).statistic), 4)
        top = med[1.0]
        half = next((t for t in DOSES if med[t] >= 0.5 * top), None)
        q1 = next((t for t in DOSES if med[t] >= 0.25 * top), None)
        q3 = next((t for t in DOSES if med[t] >= 0.75 * top), None)
        results["knee_t_half"] = half
        results["transition_width_q1_to_q3"] = (round(q3 - q1, 4)
                                                if q1 is not None and q3 is not None else None)

    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b46_cliff_function_2026_08_06.md").score(results,
                                                                              smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b46_result{'_smoke' if SMOKE else ''}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

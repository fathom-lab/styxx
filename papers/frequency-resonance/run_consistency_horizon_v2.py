# -*- coding: utf-8 -*-
"""
run_consistency_horizon_v2.py -- frozen by PREREG_consistency_horizon_v2_2026_07_24.

The consistency horizon as a SUCCESS-PROBABILITY curve. v1 (mean-accuracy) ABSTAINED because a decay
model's outcome is bimodal per seed (solve 1.0 vs stuck ~0.5); the honest metric is the fraction of seeds
that solve. Same model/phase-clamp/task as run_consistency_horizon.py (imported), with many CLAMPED seeds.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch

import run_consistency_horizon as HZ        # reuse make_data_gap / train / mag_max / spearman / interp
import run_consistency_oscillation as R      # model + test_acc

HERE = Path(__file__).resolve().parent
DEV = R.DEV
SMOKE = "--smoke" in sys.argv
GAPS = [1, 32, 255] if SMOKE else [1, 16, 32, 48, 64, 96, 128, 255]
FREE_SEEDS = [0, 1]
CLAMP_SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4, 5]
SOLVE_THR = 0.90


def main():
    HZ.STEPS = 300 if SMOKE else 1500        # winners converge <500 steps; measures P(solve within budget)
    HZ.N_TRAIN, HZ.N_TEST = (4000, 1000) if SMOKE else (24000, 6000)
    print(f"device={DEV} smoke={SMOKE} gaps={GAPS} steps={HZ.STEPS} free_seeds={FREE_SEEDS} "
          f"clamp_seeds={CLAMP_SEEDS}", flush=True)
    HZ.redteam()
    res = {"config": {"T": HZ.T_LEN, "gaps": GAPS, "steps": HZ.STEPS, "free_seeds": FREE_SEEDS,
                      "clamp_seeds": CLAMP_SEEDS, "solve_threshold": SOLVE_THR,
                      "supersedes": "PREREG_consistency_horizon_2026_07_24 (mean-acc, ABSTAINed on bimodality)"},
           "free_acc": {}, "clamped_acc": {}, "clamped_mag_max": {}}
    for gap in GAPS:
        xtr, ytr = HZ.make_data_gap(gap, HZ.N_TRAIN, HZ.DATA_SEED)
        xte, yte = HZ.make_data_gap(gap, HZ.N_TEST, HZ.DATA_SEED + 1)
        facc = []
        for s in FREE_SEEDS:
            m = HZ.train(True, s, xtr, ytr, xte, yte); a = R.test_acc(m, xte, yte); facc.append(a)
            del m; torch.cuda.empty_cache() if DEV == "cuda" else None
        cacc, cmag = [], []
        for s in CLAMP_SEEDS:
            t0 = time.time()
            m = HZ.train(False, s, xtr, ytr, xte, yte); a = R.test_acc(m, xte, yte)
            cacc.append(a); cmag.append(HZ.mag_max(m))
            print(f"  gap {gap:>3} clamped seed {s}: {a:.4f} mag_max {cmag[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
            del m; torch.cuda.empty_cache() if DEV == "cuda" else None
        res["free_acc"][str(gap)] = [round(a, 4) for a in facc]
        res["clamped_acc"][str(gap)] = [round(a, 4) for a in cacc]
        res["clamped_mag_max"][str(gap)] = [round(a, 5) for a in cmag]
        print(f"  gap {gap:>3}: FREE solve {np.mean([a>=0.95 for a in facc]):.2f}  "
              f"CLAMPED solve {np.mean([a>=SOLVE_THR for a in cacc]):.2f}", flush=True)

    free_solve = [float(np.mean([a >= 0.95 for a in res["free_acc"][str(g)]])) for g in GAPS]
    p_solve = [float(np.mean([a >= SOLVE_THR for a in res["clamped_acc"][str(g)]])) for g in GAPS]
    magmax = [float(np.mean(res["clamped_mag_max"][str(g)])) for g in GAPS]
    surviving = [mm ** g for mm, g in zip(magmax, GAPS)]
    Hstar = HZ.interp_horizon(GAPS, p_solve, 0.5)
    rho = HZ.spearman(p_solve, surviving)

    free_ok = all(x >= 1.0 for x in free_solve)
    adj_ok = p_solve[0] >= 0.80
    finite_h = (1.0 < Hstar < 255.0)
    decays_out = (p_solve[-1] <= 0.20)
    monotone = all(p_solve[i] <= p_solve[i - 1] + 0.17 for i in range(1, len(p_solve)))   # non-increasing w/ seed-noise tol
    mech_ok = (rho >= 0.80)
    if not (free_ok and adj_ok):
        verdict = "ABSTAIN__controls_failed"
    elif finite_h and decays_out and mech_ok and monotone:
        verdict = "CONFIRM__probabilistic_consistency_horizon_with_mechanism"
    elif finite_h and decays_out:
        verdict = "PARTIAL__horizon_holds_mechanism_or_monotonicity_soft"
    else:
        verdict = "PARTIAL__reported_verbatim"

    res["result"] = {
        "free_solve_by_gap": [round(x, 3) for x in free_solve],
        "clamped_solve_rate_by_gap": [round(x, 3) for x in p_solve],
        "clamped_magmax_by_gap": [round(x, 5) for x in magmax],
        "surviving_signal_by_gap": [round(x, 5) for x in surviving],
        "half_horizon_gap": (round(Hstar, 2) if np.isfinite(Hstar) else None),
        "mechanism_spearman_psolve_vs_surviving": round(rho, 4),
        "free_range_free": bool(free_ok), "clamped_adjacent_solves": bool(adj_ok),
        "psolve_decays_to_floor": bool(decays_out), "monotone_within_tol": bool(monotone),
        "verdict": verdict,
    }
    out = HERE / ("consistency_horizon_v2_smoke.json" if SMOKE else "consistency_horizon_v2_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  gap:            " + "  ".join(f"{g:>5}" for g in GAPS))
    print("  FREE solve:     " + "  ".join(f"{x:>5.2f}" for x in free_solve))
    print("  CLAMPED p_solve:" + "  ".join(f"{x:>5.2f}" for x in p_solve))
    print("  survive:        " + "  ".join(f"{x:>5.2f}" for x in surviving))
    print(f"  half-horizon (p_solve crosses 0.5): gap {Hstar:.2f}" if np.isfinite(Hstar) else "  half-horizon: none")
    print(f"  mechanism Spearman(p_solve, mag_max^gap) = {rho:.3f}")
    print("  ===== VERDICT:", verdict, "=====")
    print("  wrote", out.name, flush=True)


if __name__ == "__main__":
    main()

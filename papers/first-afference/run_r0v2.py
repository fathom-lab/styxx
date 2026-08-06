"""R0-v2 — the detection exam, per PREREG_r0v2_detection_exam_2026_08_05.md.

RV coefficient between paired streams; significance vs hour-matched and free permutation
nulls. Worlds, generator, seeds reused verbatim from run_r0. `--smoke` = tiny plumbing pass.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_r0                                   # noqa: E402
from run_r0 import make_world                   # noqa: E402
from run_r1 import hour_matched_permutation, zscore   # noqa: E402

SMOKE = "--smoke" in sys.argv
SEEDS = [11] if SMOKE else [11, 12, 13]
N_PERM = 50 if SMOKE else 500
if SMOKE:
    run_r0.N_BINS = 40


def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    Sxy = Xc.T @ Yc
    Sxx = Xc.T @ Xc
    Syy = Yc.T @ Yc
    num = np.trace(Sxy @ Sxy.T)
    den = np.sqrt(np.trace(Sxx @ Sxx) * np.trace(Syy @ Syy))
    return float(num / (den + 1e-12))


def perm_test(XR: np.ndarray, XA: np.ndarray, hours: np.ndarray, seed: int) -> dict:
    XR, XA = zscore(XR), zscore(XA)
    n = len(XR)
    obs = rv_coefficient(XR, XA)
    rng_h = np.random.default_rng(seed + 31000)
    rng_f = np.random.default_rng(seed + 62000)
    ge_h = ge_f = 0
    for _ in range(N_PERM):
        if rv_coefficient(XR, XA[hour_matched_permutation(n, hours, rng_h)]) >= obs:
            ge_h += 1
        if rv_coefficient(XR, XA[rng_f.permutation(n)]) >= obs:
            ge_f += 1
    return {"rv": round(obs, 4),
            "hourmatched_p": round((ge_h + 1) / (N_PERM + 1), 4),
            "free_p": round((ge_f + 1) / (N_PERM + 1), 4)}


def main() -> int:
    results = {"prereg": "PREREG_r0v2_detection_exam_2026_08_05.md", "smoke": SMOKE,
               "n_bins": run_r0.N_BINS, "n_perm": N_PERM, "seeds": SEEDS, "worlds": {}}
    worlds = {"C": (True, True), "K": (False, True), "N": (False, False)}
    if SMOKE:
        worlds = {"C": (True, True)}
    per = {w: [] for w in worlds}
    for s in SEEDS:
        for w, (coupled, clocked) in worlds.items():
            t0 = time.time()
            XR, XA, hours = make_world(s, coupled, clocked)
            m = perm_test(XR, XA, hours, seed=s)
            per[w].append(m)
            results["worlds"][f"{w}_s{s}"] = m
            print(f">> world {w} seed {s}: rv={m['rv']:.4f} hm_p={m['hourmatched_p']:.4f} "
                  f"free_p={m['free_p']:.4f} [{time.time()-t0:.0f}s]", flush=True)

    if not SMOKE:
        med = lambda w, k: round(float(np.median([m[k] for m in per[w]])), 4)  # noqa: E731
        results["c_hourmatched_p"] = med("C", "hourmatched_p")
        results["k_hourmatched_p"] = med("K", "hourmatched_p")
        results["k_free_p"] = med("K", "free_p")
        results["n_hourmatched_p"] = med("N", "hourmatched_p")

        results["power_surface"] = {}
        for alpha in (0.1, 0.25, 0.5, 1.0):
            XR, XA, hours = make_world(SEEDS[0], True, True, alpha=alpha)
            m = perm_test(XR, XA, hours, seed=SEEDS[0])
            results["power_surface"][str(alpha)] = m
            print(f">> power alpha={alpha}: rv={m['rv']:.4f} hm_p={m['hourmatched_p']:.4f}",
                  flush=True)

    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_r0v2_detection_exam_2026_08_05.md").score(results,
                                                                               smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"r0v2_result{'_smoke' if SMOKE else ''}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

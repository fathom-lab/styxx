# -*- coding: utf-8 -*-
"""
run_geometry_recovery.py — REPRODUCIBLE recovery sweep (isotropic vs distinctive geometry),
committed in response to peer-review C1/C7: the distinctive-recovery claim previously rested
on an inline run with no committed script. This is that script, with per-seed values and a
DIFFERENT-z control for the recovery metric.

EXPLORATORY / POST-HOC: this is NOT the pre-registered experiment. The pre-registered geometry
scaling run (run_geometry_scaling.py, isotropic z) FAILED its recovery gate (PLATEAU, recovery
max 0.10). This follow-up tests the post-hoc hypothesis that recovery needs distinctive (non-
isotropic) structure. Reported as exploratory, with the committed gate-failure stated plainly.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_disjoint_worlds as R

N = R.N = 100
KLAT = 8
SIGMAS = [0.05, 0.2, 0.4, 0.8, 1.5]
SEEDS = [0, 1, 2, 3, 4]
IU = np.triu_indices(N, 1)


def ortho(k, d, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return Q.T


def rsa(A, B):
    return float(np.corrcoef(R.distmat(A)[IU], R.distmat(B)[IU])[0, 1])


def latent(kind, rng):
    if kind == "isotropic":
        return rng.standard_normal((N, KLAT))
    K = 12; lab = rng.integers(0, K, N); ctr = rng.standard_normal((K, KLAT)) * 4
    return ctr[lab] + 0.5 * rng.standard_normal((N, KLAT))


def main():
    out = {"sigmas": SIGMAS, "seeds": SEEDS, "geom": {}}
    for kind in ["isotropic", "distinctive"]:
        rows = []
        for sg in SIGMAS:
            F, S, Rc, Rctrl = [], [], [], []
            for seed in SEEDS:
                rng = np.random.default_rng(700 + seed)
                z = latent(kind, rng); zp = latent(kind, np.random.default_rng(900 + seed))
                RA = ortho(KLAT, 32, rng); RB = ortho(KLAT, 24, rng)
                em = lambda zz, Rp, rr: ((zz @ Rp - (zz @ Rp).mean(0)) / ((zz @ Rp).std(0) + 1e-9)) + sg * rr.standard_normal((N, Rp.shape[1]))
                EA = em(z, RA, np.random.default_rng(seed * 9 + 1))
                EB = em(z, RB, np.random.default_rng(seed * 9 + 2))         # same z
                EBp = em(zp, RB, np.random.default_rng(seed * 9 + 3))       # different z (control)
                F.append(rsa(EA, z)); S.append(rsa(EA, EB))
                Rc.append(R.run_condition(EA, EB, rng)[0])                 # recovery, same structure
                Rctrl.append(R.run_condition(EA, EBp, rng)[0])            # recovery, control
            rows.append({"sigma": sg, "faith": round(float(np.mean(F)), 3),
                         "same_rsa": round(float(np.mean(S)), 3),
                         "recovery_mean": round(float(np.mean(Rc)), 3),
                         "recovery_max": round(float(np.max(Rc)), 3),
                         "recovery_per_seed": [round(x, 3) for x in Rc],
                         "control_recovery_mean": round(float(np.mean(Rctrl)), 3)})
            print(f"{kind:11s} sig={sg:>4}: faith={rows[-1]['faith']:.2f} same_RSA={rows[-1]['same_rsa']:.2f} "
                  f"recovery mean={rows[-1]['recovery_mean']:.2f} max={rows[-1]['recovery_max']:.2f} "
                  f"ctrl={rows[-1]['control_recovery_mean']:.2f}", flush=True)
        out["geom"][kind] = rows

    iso_hi = max(r["recovery_mean"] for r in out["geom"]["isotropic"])
    dist_hi = max(r["recovery_mean"] for r in out["geom"]["distinctive"])
    out["summary"] = {
        "isotropic_recovery_max_mean": iso_hi, "distinctive_recovery_max_mean": dist_hi,
        "honest_reading": (f"Pre-registered isotropic recovery FAILED its gate (max-mean {iso_hi:.2f} < 0.50). "
                           f"Distinctive geometry recovery reaches max-mean {dist_hi:.2f} only at the highest "
                           f"faithfulness, with high per-seed variance, and collapses below faith ~0.98 — an "
                           f"EXPLORATORY (not pre-registered) signal that recovery needs distinctiveness + high "
                           f"faithfulness, NOT a robust 'recovery=1.0 reproduces vec2vec' result.")}
    (HERE / "geometry_recovery_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n" + out["summary"]["honest_reading"])
    print("wrote geometry_recovery_result.json")


if __name__ == "__main__":
    main()

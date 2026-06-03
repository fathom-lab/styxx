# -*- coding: utf-8 -*-
"""
run_geometry_scaling.py — does meaning-geometry convergence scale to FULL universality,
and does unsupervised recovery EMERGE, as representations become faithful? Frozen by
PREREG_geometry_scaling_2026_06_03.md.

Controlled-faithfulness representations of a shared latent geometry z: E = z R + sigma*noise,
with ORTHONORMAL projections R (so the signal part preserves z's distances; sigma sets the
faithfulness). Sweep sigma -> faithfulness range. At each: same-structure RSA, control RSA
(different z), and unsupervised Gromov-Wasserstein recovery (the validated aligner from
run_disjoint_worlds). Anchored to the real SGNS point (faithfulness ~0.51, RSA ~0.42).
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_disjoint_worlds as R                       # distmat, run_condition (validated GW)

N = R.N = 100
KLAT = 8
SIGMAS = [0.05, 0.15, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
SEEDS = [0, 1, 2]
IU = np.triu_indices(N, 1)


def rsa(A, B):
    DA, DB = R.distmat(A), R.distmat(B)
    return float(np.corrcoef(DA[IU], DB[IU])[0, 1])


def ortho(k, d, rng):                                 # k->d projection with orthonormal rows
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))  # d x k, orthonormal columns
    return Q.T                                        # k x d, orthonormal rows (preserves distances)


def main():
    agg = {sg: {"faith": [], "same": [], "ctrl": [], "recov": []} for sg in SIGMAS}
    for seed in SEEDS:
        rng = np.random.default_rng(500 + seed)
        z = rng.standard_normal((N, KLAT))
        zp = rng.standard_normal((N, KLAT))
        RA = ortho(KLAT, 32, rng); RB = ortho(KLAT, 24, rng)

        def emb(zz, Rp, sg, rr):
            s = zz @ Rp
            s = (s - s.mean(0)) / (s.std(0) + 1e-9)
            return s + sg * rr.standard_normal(s.shape)

        for sg in SIGMAS:
            EA = emb(z, RA, sg, np.random.default_rng(seed * 100 + 1))
            EB = emb(z, RB, sg, np.random.default_rng(seed * 100 + 2))     # SAME z, indep noise, diff dim
            EBp = emb(zp, RB, sg, np.random.default_rng(seed * 100 + 3))   # DIFFERENT z
            agg[sg]["faith"].append(rsa(EA, z))                           # faithfulness = RSA(E_A, z)
            agg[sg]["same"].append(rsa(EA, EB))
            agg[sg]["ctrl"].append(rsa(EA, EBp))
            agg[sg]["recov"].append(R.run_condition(EA, EB, rng)[0])      # unsupervised GW top-1

    levels = []
    for sg in SIGMAS:
        a = agg[sg]
        L = {"sigma": sg, "faithfulness": round(float(np.mean(a["faith"])), 3),
             "same_rsa": round(float(np.mean(a["same"])), 3),
             "control_rsa": round(float(np.mean(a["ctrl"])), 3),
             "recovery": round(float(np.mean(a["recov"])), 3)}
        levels.append(L)
        print(f"sigma={sg:>4}: faith={L['faithfulness']:.2f}  same_RSA={L['same_rsa']:.2f}  "
              f"ctrl={L['control_rsa']:.2f}  recovery={L['recovery']:.2f}", flush=True)

    levels.sort(key=lambda x: x["faithfulness"])
    rsa_max = max(L["same_rsa"] for L in levels)
    recov_max = max(L["recovery"] for L in levels)
    ctrl_max = max(abs(L["control_rsa"]) for L in levels)
    # recovery-emergence threshold: lowest faithfulness with recovery >= 0.3
    emerge = next((L["faithfulness"] for L in levels if L["recovery"] >= 0.3), None)
    if rsa_max >= 0.90 and recov_max >= 0.50 and ctrl_max < 0.10:
        reading = (f"SCALING CONFIRMED — same-structure RSA reaches {rsa_max:.2f} and unsupervised recovery "
                   f"reaches {recov_max:.2f} as faithfulness->1 (control stays |{ctrl_max:.2f}|). Convergence "
                   f"scales to FULL universality and the vec2vec recovery phenomenon EMERGES (threshold ~faith "
                   f"{emerge}). The SGNS point (faith~0.51, RSA~0.42) is one point on this curve, below the "
                   f"recovery threshold — which is exactly why our GW could not recover correspondence there.")
    elif rsa_max >= 0.70:
        reading = f"PLATEAU — same-RSA rises to {rsa_max:.2f}, recovery max {recov_max:.2f}; convergence is partial even near faith 1."
    else:
        reading = f"FLAT/unexpected — same-RSA max {rsa_max:.2f} (does not track faithfulness; diagnose)."
    out = {"sigmas": SIGMAS, "levels": levels,
           "gate": {"rsa_max": round(rsa_max, 3), "recovery_max": round(recov_max, 3),
                    "control_max": round(ctrl_max, 3), "recovery_emergence_faith": emerge,
                    "sgns_anchor": {"faithfulness": 0.51, "same_rsa": 0.42}, "reading": reading}}
    (HERE / "geometry_scaling_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n===== " + reading)
    print("wrote geometry_scaling_result.json")


if __name__ == "__main__":
    main()

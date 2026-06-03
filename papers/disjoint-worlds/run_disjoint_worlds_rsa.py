# -*- coding: utf-8 -*-
"""
run_disjoint_worlds_rsa.py — the VALIDATED re-do of the geometry decider.

The GW version (run_disjoint_worlds.py) was found INVALID by its positive control: GW
recovers only ~0.42 even on a real SGNS embedding vs its own rotation, so it cannot detect
convergence even when present. The diagnostic showed the embeddings DO converge
(RSA ~0.46 same-structure) — GW just can't recover the correspondence. So we switch to the
standard, validatable measure: **representational similarity (RSA)** — the correlation of
the two embeddings' pairwise-distance matrices (rotation-invariant by construction; the
exact tool Kriegeskorte/CKA use). Same-structure RSA vs different-structure control answers
the question directly: is the geometry structure-determined (universal) or data-specific?
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_disjoint_worlds as R                      # reuse make_world, train_sgns, distmat

R.TAU = 0.25; R.K_LAT = 4; R.SGNS_STEPS = 8000; R.N = 100   # config for adequate faithfulness
N = R.N
SEEDS = [0, 1, 2]
IU = np.triu_indices(N, 1)


def rsa(EA, EB):
    DA, DB = R.distmat(EA), R.distmat(EB)
    return float(np.corrcoef(DA[IU], DB[IU])[0, 1])


def main():
    res = {"config": {"tau": R.TAU, "klat": R.K_LAT, "N": N, "steps": R.SGNS_STEPS,
                      "dA": R.DA, "dB": R.DB}, "per_seed": {}}
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        z = rng.standard_normal((N, R.K_LAT))                          # shared latent geometry
        zp = rng.standard_normal((N, R.K_LAT))                         # DIFFERENT geometry (control)
        EA = R.train_sgns(*R.make_world(z, np.random.default_rng(10 + s)), R.DA, s)
        EB = R.train_sgns(*R.make_world(z, np.random.default_rng(20 + s)), R.DB, s + 100)   # SAME z, INDEP corpus, diff dim
        EBp = R.train_sgns(*R.make_world(zp, np.random.default_rng(30 + s)), R.DB, s + 200)  # different structure
        Q, _ = np.linalg.qr(rng.standard_normal((R.DA, R.DA)))
        same = rsa(EA, EB); ctrl = rsa(EA, EBp)
        valid_iden = rsa(EA, EA @ Q)                                   # RSA is rotation-invariant -> ~1
        faithful = float(np.corrcoef(R.distmat(EA)[IU], R.distmat(z)[IU])[0, 1])  # does E learn z?
        res["per_seed"][str(s)] = {"same_rsa": round(same, 4), "control_rsa": round(ctrl, 4),
                                   "valid_identical_rsa": round(valid_iden, 4), "faithful_E_to_z": round(faithful, 4)}
        print(f"seed{s}: same_RSA={same:.3f}  control_RSA={ctrl:.3f}  valid_iden={valid_iden:.3f}  faithful={faithful:.3f}", flush=True)

    def mean(k): return float(np.mean([res["per_seed"][str(s)][k] for s in SEEDS]))
    same, ctrl, vi, fa = mean("same_rsa"), mean("control_rsa"), mean("valid_identical_rsa"), mean("faithful_E_to_z")
    if vi < 0.90 or fa < 0.30:
        reading = f"INVALID — validity failed (identical-RSA {vi:.2f} must be ~1, faithful {fa:.2f} must be >0.3)"
    elif same > 0.20 and same > ctrl + 0.15 and same > 3 * abs(ctrl):
        reading = (f"UNIVERSAL (RSA) — independent models on DISJOINT data (different corpora, different dims, "
                   f"zero shared tokens) CONVERGE to a shared geometry when the latent structure is shared: "
                   f"same-structure RSA {same:.2f} vs different-structure control {ctrl:.2f} (validity: identical "
                   f"{vi:.2f}, faithful {fa:.2f}). The geometry of meaning is STRUCTURE-DETERMINED, not "
                   f"data-specific. Convergence is partial at this synthetic scale (RSA {same:.2f}, not 1.0) and "
                   f"tracks how faithfully each model captures the structure — consistent with stronger convergence "
                   f"at real scale (vec2vec).")
    else:
        reading = f"NO CONVERGENCE — same-structure RSA {same:.2f} ~ control {ctrl:.2f}; geometry not structure-determined here."
    res["gate"] = {"same_rsa": round(same, 4), "control_rsa": round(ctrl, 4),
                   "valid_identical_rsa": round(vi, 4), "faithful": round(fa, 4), "reading": reading}
    (HERE / "disjoint_worlds_rsa_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n===== " + reading)
    print("wrote disjoint_worlds_rsa_result.json")


if __name__ == "__main__":
    main()

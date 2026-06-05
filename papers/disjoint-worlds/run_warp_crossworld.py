# -*- coding: utf-8 -*-
"""run_warp_crossworld.py — frozen by PREREG_warp_crossworld_2026_06_05.

Adversarial test of my own causal-transfer result. The causal experiment used LINEAR embeddings (geometry =
rotation of shared z), so direction-transfer-through-the-recovered-rotation was partly built in. Here each
world applies a DIFFERENT NONLINEAR WARP to the shared structure (geometry = warp, not rotation). Sweep warp
strength alpha 0->1; at each, run the unsupervised pipeline with TWO maps (orthogonal Procrustes = rotation
only; learned linear W = affine) and ask whether causal attribute transfer SURVIVES the warp or COLLAPSES.
Collapse => the prior result was rotation-dependent (self-falsification). Survival => much stronger claim.
Reuses the causal rig's helpers + the validated GW aligner.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import orthogonal_procrustes

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_disjoint_worlds as R
from run_causal_crossworld import ortho, pca, attr_dirs, transfer_matrix, diag_adv, rsa, N, K, DA, DB, DC, W_DISTINCT

R.N = N
R.GW_INITS = 10
SIGMA = 0.05
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
SMOKE = "--smoke" in sys.argv
if SMOKE:
    ALPHAS = [0.0, 1.0]; SEEDS = [0]
H = 64                                                   # warp MLP hidden


def make_warp(d, rng):
    W1 = rng.standard_normal((d, H)) / np.sqrt(d)
    W2 = rng.standard_normal((H, d)) / np.sqrt(H)
    return lambda x: np.tanh(x @ W1) @ W2


def warp_emb(z, Rp, g, alpha, rng):
    base = z @ Rp
    base = (base - base.mean(0)) / (base.std(0) + 1e-9)
    w = g(base); w = (w - w.mean(0)) / (w.std(0) + 1e-9)
    e = (1 - alpha) * base + alpha * w
    e = (e - e.mean(0)) / (e.std(0) + 1e-9)
    return e + SIGMA * rng.standard_normal(e.shape)


def run_one(EA, EB, z, rng):
    perm = rng.permutation(N); EBp = EB[perm]; true_match = np.argsort(perm)
    assign, _ = R.align(EA, EBp, rng)
    recov = float(np.mean(assign == true_match))
    rec_idx = perm[assign]
    PA, PB = pca(EA, DC), pca(EB, DC)
    dirsA, dirsB = attr_dirs(PA, z), attr_dirs(PB, z)
    Qn, _ = np.linalg.qr(rng.standard_normal((DC, DC)))
    adv_null = diag_adv(transfer_matrix(dirsA, dirsB, Qn))[0]

    def adv(idx):  # transfer via Procrustes (rotation) and learned linear map, given a correspondence idx
        Qp, _ = orthogonal_procrustes(PA, PB[idx])
        Wl, *_ = np.linalg.lstsq(PA, PB[idx], rcond=None)
        return diag_adv(transfer_matrix(dirsA, dirsB, Qp))[0], diag_adv(transfer_matrix(dirsA, dirsB, Wl))[0]

    proc_rec, lin_rec = adv(rec_idx)                  # UNSUPERVISED (recovered correspondence)
    proc_true, lin_true = adv(np.arange(N))           # TRUE correspondence (identity) -- diagnostic ceiling
    return {"recovery": recov, "rsa_share": rsa(EA, EB), "faith": rsa(EA, z), "adv_null": adv_null,
            "adv_procrustes": proc_rec, "adv_linearmap": lin_rec,
            "adv_proc_true": proc_true, "adv_lin_true": lin_true}


def main():
    res = {"config": {"N": N, "K": K, "sigma": SIGMA, "alphas": ALPHAS, "seeds": SEEDS}, "by_alpha": {}}
    for a in ALPHAS:
        rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(900 + seed)
            z = rng.standard_normal((N, K)) * W_DISTINCT
            RA, RB = ortho(K, DA, rng), ortho(K, DB, rng)
            gA = make_warp(DA, np.random.default_rng(seed * 7 + 1))
            gB = make_warp(DB, np.random.default_rng(seed * 7 + 2))    # DIFFERENT warp per world
            EA = warp_emb(z, RA, gA, a, np.random.default_rng(seed * 7 + 3))
            EB = warp_emb(z, RB, gB, a, np.random.default_rng(seed * 7 + 4))
            rows.append(run_one(EA, EB, z, rng))
        agg = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}
        res["by_alpha"][str(a)] = agg
        print(f"alpha={a:<5}: faith={agg['faith']:.2f} rsa={agg['rsa_share']:.2f} recov={agg['recovery']:.2f} | "
              f"UNSUP proc={agg['adv_procrustes']:.2f} lin={agg['adv_linearmap']:.2f} | TRUE proc={agg['adv_proc_true']:.2f} "
              f"lin={agg['adv_lin_true']:.2f} | null={agg['adv_null']:.2f}", flush=True)

    a0, a1 = res["by_alpha"]["0.0"], res["by_alpha"]["1.0"]
    p1 = (a1["adv_procrustes"] >= 0.30) and (a1["adv_procrustes"] - a1["adv_null"] >= 0.30)   # unsupervised survives
    true_holds = max(a1["adv_proc_true"], a1["adv_lin_true"]) >= 0.30                          # alignable w/ TRUE pairs
    p3 = a1["rsa_share"] >= 0.30
    if p1:
        reading = (f"TRANSFER SURVIVES WARP — causal transfer holds UNSUPERVISED through a genuine nonlinear warp "
                   f"(Procrustes {a1['adv_procrustes']:.2f} at alpha=1, null {a1['adv_null']:.2f}; was "
                   f"{a0['adv_procrustes']:.2f} at alpha=0). NOT rotation-bound. Strong claim, robustness check passed.")
    elif true_holds and p3:
        reading = (f"RECOVERY-BOUND (not transfer-bound) — with TRUE correspondence the warp still transfers "
                   f"(proc_true {a1['adv_proc_true']:.2f} / lin_true {a1['adv_lin_true']:.2f} at alpha=1) and RSA "
                   f"survives ({a1['rsa_share']:.2f}), BUT unsupervised recovery fails under warp (recov "
                   f"{a1['recovery']:.2f}) so the unsupervised map collapses ({a1['adv_procrustes']:.2f}). The "
                   "geometry/causal structure is still there and alignable; the bottleneck is UNSUPERVISED "
                   "correspondence recovery under warp, not the transferability of meaning. Nuanced, honest.")
    elif p3:
        reading = (f"LINEAR-MAP-BOUND / SELF-FALSIFIED — even with TRUE correspondence, a linear map cannot "
                   f"transfer through the warp (proc_true {a1['adv_proc_true']:.2f}, lin_true {a1['adv_lin_true']:.2f} "
                   f"at alpha=1) though RSA-sharing survives ({a1['rsa_share']:.2f}). Causal direction transfer is "
                   "fundamentally a near-rotation/affine phenomenon; it does NOT generalize to genuine warps. "
                   "Honest correction to the prior result.")
    else:
        reading = (f"GEOMETRY DIVERGES — different nonlinear warps destroy shared geometry itself "
                   f"(RSA-share {a1['rsa_share']:.2f} at alpha=1). Past mild warps the worlds are uncorrelated.")
    res["gate"] = {"alpha0": a0, "alpha1": a1, "P1_unsup_survives": bool(p1),
                   "true_correspondence_holds": bool(true_holds), "P3_rsa_survives": bool(p3), "reading": reading}
    out = HERE / ("warp_crossworld_smoke.json" if SMOKE else "warp_crossworld_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  adv_procrustes(alpha):", {a: res["by_alpha"][str(a)]["adv_procrustes"] for a in ALPHAS})
    print("  rsa_share(alpha):     ", {a: res["by_alpha"][str(a)]["rsa_share"] for a in ALPHAS})
    print("\n===== " + reading)
    print("wrote", out.name)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
run_disjoint_worlds.py — Is the geometry of meaning universal structure or shared data?
Frozen by PREREG_disjoint_worlds_2026_06_03.md.

Two synthetic worlds with IDENTICAL latent structure but DISJOINT tokens + INDEPENDENT
corpora (zero shared data). Independently-trained skip-gram embeddings (different dims).
Then UNSUPERVISED Gromov-Wasserstein alignment of the two geometries (no correspondence,
no shared dimension) — does it recover the hidden token correspondence? If yes with zero
shared data, the geometry is structure-determined (universal forms, testable sense).
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"

N = 100          # concepts
K_LAT = 8        # latent dim
TAU = 0.5        # proximity temperature
M_PAIRS = 200_000
DA, DB = 32, 24  # different embedding dims for A vs B
SGNS_STEPS = 3000
BATCH = 4096
NEG = 5
WORLD_SEEDS = [0, 1, 2]
GW_EPS = 0.05
GW_OUTER, GW_SINK, GW_INITS = 60, 50, 6


def make_world(z, rng):
    """Co-occurrence pairs from latent geometry z (N,K). Returns (centers, contexts)."""
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)          # (N,N)
    P = np.exp(-d2 / TAU); np.fill_diagonal(P, 0.0)
    P = P / P.sum(1, keepdims=True)
    centers = rng.integers(0, N, M_PAIRS)
    cum = np.cumsum(P, 1)
    r = rng.random(M_PAIRS)
    contexts = (r[:, None] < cum[centers]).argmax(1)
    return centers.astype(np.int64), contexts.astype(np.int64)


def train_sgns(centers, contexts, d, seed):
    torch.manual_seed(seed)
    inp = nn.Embedding(N, d).to(DEV); out = nn.Embedding(N, d).to(DEV)
    opt = torch.optim.Adam(list(inp.parameters()) + list(out.parameters()), lr=5e-3)
    c = torch.tensor(centers, device=DEV); x = torch.tensor(contexts, device=DEV)
    n = len(centers)
    for step in range(SGNS_STEPS):
        idx = torch.randint(0, n, (BATCH,), device=DEV)
        ci, xj = c[idx], x[idx]
        negs = torch.randint(0, N, (BATCH, NEG), device=DEV)
        ui = inp(ci); vj = out(xj); vn = out(negs)
        pos = torch.nn.functional.logsigmoid((ui * vj).sum(-1))
        neg = torch.nn.functional.logsigmoid(-(vn @ ui.unsqueeze(-1)).squeeze(-1)).sum(-1)
        loss = -(pos + neg).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return inp.weight.detach().cpu().numpy()


def distmat(E):
    E = E - E.mean(0, keepdims=True)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    D = np.sqrt(np.maximum(((E[:, None, :] - E[None, :, :]) ** 2).sum(-1), 0))
    return D / (D.mean() + 1e-9)


def gw_cost(D1, D2, T, Cc):
    return float(np.sum(T * Cc) - 2 * np.sum(T * (D1 @ T @ D2)))


def entropic_gw(D1, D2, eps=GW_EPS, outer=GW_OUTER, sink=GW_SINK, init=None, rng=None):
    n, m = D1.shape[0], D2.shape[0]
    p = np.ones(n) / n; q = np.ones(m) / m
    Cc = (D1 ** 2 @ p)[:, None] + (D2 ** 2 @ q)[None, :]
    T = init if init is not None else np.outer(p, q)
    for _ in range(outer):
        L = Cc - 2 * D1 @ T @ D2
        K = np.exp(-(L - L.min()) / eps)
        u = np.ones(n)
        for _ in range(sink):
            v = q / (K.T @ u + 1e-30)
            u = p / (K @ v + 1e-30)
        T = u[:, None] * K * v[None, :]
    return T, gw_cost(D1, D2, T, Cc)


def align(EA, EB, rng):
    """Best-of-many-init GW; return Hungarian assignment EA-row -> EB-row."""
    D1, D2 = distmat(EA), distmat(EB)
    best_T, best_c = None, np.inf
    for k in range(GW_INITS):
        init = None if k == 0 else (lambda r=rng: (lambda X: X / X.sum())(r.random((N, N)) + 1.0))()
        T, c = entropic_gw(D1, D2, init=init, rng=rng)
        if c < best_c:
            best_c, best_T = c, T
    row, col = linear_sum_assignment(-best_T)             # maximize coupling mass
    assign = np.empty(N, dtype=int); assign[row] = col
    return assign, best_c


def run_condition(EA, EB, rng):
    perm = rng.permutation(N)                              # scramble B's rows (hidden)
    EBp = EB[perm]
    true_match = np.argsort(perm)                          # EA[i] <-> EBp[true_match[i]]
    assign, cost = align(EA, EBp, rng)
    acc = float(np.mean(assign == true_match))
    return acc, cost


def main():
    results = {"config": {"N": N, "K_lat": K_LAT, "dA": DA, "dB": DB, "M": M_PAIRS,
                          "world_seeds": WORLD_SEEDS}, "per_seed": {}}
    for ws in WORLD_SEEDS:
        rng = np.random.default_rng(1000 + ws)
        z = rng.standard_normal((N, K_LAT))               # latent geometry (shared by A,B)
        zp = rng.standard_normal((N, K_LAT))              # DIFFERENT geometry (control)
        cA = make_world(z, np.random.default_rng(10 + ws))
        cB = make_world(z, np.random.default_rng(20 + ws))   # SAME z, INDEPENDENT corpus
        cBp = make_world(zp, np.random.default_rng(30 + ws))  # different structure
        EA = train_sgns(*cA, DA, seed=ws)
        EB = train_sgns(*cB, DB, seed=ws + 100)
        EBp = train_sgns(*cBp, DB, seed=ws + 200)
        # conditions
        same_acc, same_c = run_condition(EA, EB, rng)             # SAME structure, disjoint data
        ctrl_acc, ctrl_c = run_condition(EA, EBp, rng)            # control: different structure
        sanity_acc, _ = run_condition(EA, EA.copy(), rng)         # self vs permuted self
        results["per_seed"][str(ws)] = {"same_structure_acc": round(same_acc, 4),
                                        "control_diff_struct_acc": round(ctrl_acc, 4),
                                        "sanity_self_acc": round(sanity_acc, 4)}
        print(f"world{ws}: SAME={same_acc:.3f}  control={ctrl_acc:.3f}  sanity={sanity_acc:.3f}", flush=True)

    same = np.mean([results["per_seed"][str(s)]["same_structure_acc"] for s in WORLD_SEEDS])
    ctrl = np.mean([results["per_seed"][str(s)]["control_diff_struct_acc"] for s in WORLD_SEEDS])
    sanity = np.mean([results["per_seed"][str(s)]["sanity_self_acc"] for s in WORLD_SEEDS])
    chance = 1.0 / N
    if sanity < 0.8:
        reading = f"INVALID — sanity (self-permuted) only {sanity:.2f}; the GW aligner is not reliable, do not interpret"
    elif same >= 0.50 and same >= 5 * max(ctrl, chance):
        reading = (f"UNIVERSAL — shared structure determines geometry with ZERO shared data: "
                   f"same-structure top-1 {same:.2f} vs control {ctrl:.2f} (chance {chance:.2f}). "
                   f"The geometry of meaning is recoverable from structure alone — universal forms, testable sense.")
    elif same > ctrl + 0.05 and same > 3 * chance:
        reading = (f"PARTIAL — geometry is PARTLY structure-determined: same {same:.2f} > control {ctrl:.2f} "
                   f"(chance {chance:.2f}), but below the 0.50 bar. Report the degree.")
    else:
        reading = (f"ARTIFACT — independent models on disjoint data do NOT converge to an alignable geometry "
                   f"(same {same:.2f} ~ control {ctrl:.2f} ~ chance {chance:.2f}). Strong Platonic reading falsified.")
    results["gate"] = {"same_structure": round(float(same), 4), "control": round(float(ctrl), 4),
                       "sanity": round(float(sanity), 4), "chance": round(chance, 4), "reading": reading}
    (HERE / "disjoint_worlds_result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n===== " + reading)
    print("wrote disjoint_worlds_result.json")


if __name__ == "__main__":
    main()

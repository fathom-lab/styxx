# -*- coding: utf-8 -*-
"""
run_recall_horizon.py -- frozen by PREREG_recall_horizon_2026_07_24.

Does the phase mechanism generalize to canonical K-way delayed recall (the induction/selective-copy
primitive real SSM-LLMs are benchmarked on)? Same CLRU phase-clamp (FREE theta-learnable vs CLAMPED
theta==0, matched/RNG-matched) as the consistency horizon, but the model must RETRIEVE which of K values
appeared earlier. Success-probability over seeds (decay's trainability is bimodal). Imports the exact
Block/scan/test_acc from run_consistency_oscillation; spearman/interp/mag_max from run_consistency_horizon.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import run_consistency_oscillation as R
import run_consistency_horizon as HZ

HERE = Path(__file__).resolve().parent
DEV = R.DEV
SMOKE = "--smoke" in sys.argv
K = 16
T_LEN = R.T_LEN
H, N_BLK = R.H, R.N_BLK
GAPS = [1, 32, 255] if SMOKE else [1, 4, 16, 32, 64, 128, 255]
FREE_SEEDS = [0, 1]
CLAMP_SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
STEPS = 300 if SMOKE else 2000
N_TRAIN, N_TEST = (4000, 1000) if SMOKE else (24000, 6000)
BATCH, LR, WD = 64, 3e-3, 0.01
DATA_SEED = 7777
SOLVE_THR = 0.80                                  # chance = 1/K = 0.0625


class RecallSSM(nn.Module):
    def __init__(self, free):
        super().__init__()
        self.emb = nn.Linear(K + 1, H, bias=False)          # one-hot content (K) + query marker (1)
        self.blocks = nn.ModuleList([R.Block(free) for _ in range(N_BLK)])
        self.head = nn.Linear(H, K)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x[:, -1])

    def set_clamp(self, flag):
        for b in self.blocks:
            b.ssm.clamp_theta = flag


def make_recall_data(gap, n, seed):
    g = np.random.default_rng(seed)
    X = np.zeros((n, T_LEN, K + 1), dtype=np.float32)
    vals = g.integers(0, K, size=n)
    X[np.arange(n), T_LEN - 1 - gap, vals] = 1.0                # one-hot content at the gap
    X[:, T_LEN - 1, K] = 1.0                                    # query marker at the probe
    return torch.from_numpy(X), torch.from_numpy(vals.astype(np.int64))


def train(free, seed, xtr, ytr, xte, yte):
    torch.manual_seed(seed); np.random.seed(seed)
    m = RecallSSM(free).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    lossf = nn.CrossEntropyLoss()
    N = len(xtr)
    for step in range(STEPS):
        idx = torch.randint(0, N, (BATCH,))
        loss = lossf(m(xtr[idx].to(DEV)), ytr[idx].to(DEV))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    return m


def redteam():
    torch.manual_seed(7)
    Ac = torch.randn(3, 40, 8, dtype=torch.cfloat, device=DEV) * 0.5
    Xc = torch.randn(3, 40, 8, dtype=torch.cfloat, device=DEV)
    ds = (R.lin_scan(Ac, Xc) - R.seq_scan(Ac, Xc)).abs().max().item()
    assert ds < 1e-4, f"scan!=seq ({ds:.2e})"
    torch.manual_seed(0); mf = RecallSSM(True)
    torch.manual_seed(0); mc = RecallSSM(False)
    assert torch.equal(mf.blocks[0].ssm.B_re, mc.blocks[0].ssm.B_re), "B_re RNG mismatch"
    for gp in (1, 255):
        X, y = make_recall_data(gp, 2000, 999)
        pos = np.array([np.where(X[b, :, :K].numpy().any(1))[0][0] for b in range(2000)])
        assert (pos == T_LEN - 1 - gp).all(), f"content not at T-1-{gp}"
    assert abs(np.bincount(make_recall_data(1, 4000, 1)[1].numpy(), minlength=K).std()
               / (4000 / K)) < 0.25, "labels not ~uniform"
    # untrained chance baseline ~ 1/K
    with torch.no_grad():
        Xc0, yc0 = make_recall_data(32, 2000, 5)
        acc0 = R.test_acc(RecallSSM(True).to(DEV), Xc0, yc0)
    assert acc0 < 0.15, f"untrained acc {acc0:.3f} not near chance {1/K:.3f}"
    print(f"  [redteam] scan==seq ({ds:.1e}); RNG matched; content placement OK; "
          f"untrained acc {acc0:.3f} ~ 1/K={1/K:.3f} -- OK", flush=True)


def main():
    print(f"device={DEV} smoke={SMOKE} K={K} T={T_LEN} gaps={GAPS} steps={STEPS} "
          f"free={FREE_SEEDS} clamp={CLAMP_SEEDS}", flush=True)
    redteam()
    res = {"config": {"task": "delayed K-way recall (induction/selective-copy primitive)", "K": K,
                      "T": T_LEN, "gaps": GAPS, "steps": STEPS, "free_seeds": FREE_SEEDS,
                      "clamp_seeds": CLAMP_SEEDS, "solve_threshold": SOLVE_THR, "chance": 1.0 / K},
           "free_acc": {}, "clamped_acc": {}, "clamped_mag_max": {}}
    for gap in GAPS:
        xtr, ytr = make_recall_data(gap, N_TRAIN, DATA_SEED)
        xte, yte = make_recall_data(gap, N_TEST, DATA_SEED + 1)
        facc = []
        for s in FREE_SEEDS:
            m = train(True, s, xtr, ytr, xte, yte); facc.append(R.test_acc(m, xte, yte))
            del m; torch.cuda.empty_cache() if DEV == "cuda" else None
        cacc, cmag = [], []
        for s in CLAMP_SEEDS:
            t0 = time.time()
            m = train(False, s, xtr, ytr, xte, yte); a = R.test_acc(m, xte, yte)
            cacc.append(a); cmag.append(HZ.mag_max(m))
            print(f"  gap {gap:>3} clamped seed {s}: {a:.4f} mag_max {cmag[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
            del m; torch.cuda.empty_cache() if DEV == "cuda" else None
        res["free_acc"][str(gap)] = [round(a, 4) for a in facc]
        res["clamped_acc"][str(gap)] = [round(a, 4) for a in cacc]
        res["clamped_mag_max"][str(gap)] = [round(a, 5) for a in cmag]
        print(f"  gap {gap:>3}: FREE solve {np.mean([a>=SOLVE_THR for a in facc]):.2f}  "
              f"CLAMPED solve {np.mean([a>=SOLVE_THR for a in cacc]):.2f}", flush=True)

    free_solve = [float(np.mean([a >= SOLVE_THR for a in res["free_acc"][str(g)]])) for g in GAPS]
    p_solve = [float(np.mean([a >= SOLVE_THR for a in res["clamped_acc"][str(g)]])) for g in GAPS]
    magmax = [float(np.mean(res["clamped_mag_max"][str(g)])) for g in GAPS]
    surviving = [mm ** g for mm, g in zip(magmax, GAPS)]
    Hstar = HZ.interp_horizon(GAPS, p_solve, 0.5)
    rho = HZ.spearman(p_solve, surviving)

    free_ok = all(x >= 1.0 for x in free_solve)
    adj_ok = p_solve[0] >= 0.80
    finite_h = (1.0 < Hstar < 255.0)
    decays_out = (p_solve[-1] <= 0.20)
    mech_ok = (rho >= 0.75)
    if not (free_ok and adj_ok):
        verdict = "ABSTAIN__controls_failed"
    elif finite_h and decays_out and mech_ok:
        verdict = "CONFIRM__recall_horizon_generalizes_with_mechanism"
    elif finite_h and decays_out:
        verdict = "PARTIAL__horizon_holds_mechanism_soft"
    else:
        verdict = "PARTIAL__reported_verbatim"

    res["result"] = {
        "free_solve_by_gap": [round(x, 3) for x in free_solve],
        "clamped_solve_rate_by_gap": [round(x, 3) for x in p_solve],
        "clamped_magmax_by_gap": [round(x, 5) for x in magmax],
        "surviving_signal_by_gap": [round(x, 5) for x in surviving],
        "half_horizon_gap": (round(Hstar, 2) if np.isfinite(Hstar) else None),
        "mechanism_spearman_psolve_vs_surviving": round(rho, 4),
        "chance": round(1.0 / K, 4),
        "free_range_free": bool(free_ok), "clamped_adjacent_solves": bool(adj_ok),
        "psolve_decays_to_floor": bool(decays_out), "verdict": verdict,
    }
    out = HERE / ("recall_horizon_smoke.json" if SMOKE else "recall_horizon_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  gap:            " + "  ".join(f"{g:>5}" for g in GAPS))
    print("  FREE solve:     " + "  ".join(f"{x:>5.2f}" for x in free_solve))
    print("  CLAMPED p_solve:" + "  ".join(f"{x:>5.2f}" for x in p_solve))
    print("  survive:        " + "  ".join(f"{x:>5.2f}" for x in surviving))
    print(f"  recall half-horizon (K={K}): gap {Hstar:.2f}" if np.isfinite(Hstar) else "  half-horizon: none")
    print(f"  mechanism Spearman(p_solve, mag_max^gap) = {rho:.3f}")
    print("  ===== VERDICT:", verdict, "=====")
    print("  wrote", out.name, flush=True)


if __name__ == "__main__":
    main()

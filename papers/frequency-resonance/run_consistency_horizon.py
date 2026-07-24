# -*- coding: utf-8 -*-
"""
run_consistency_horizon.py -- frozen by PREREG_consistency_horizon_2026_07_24.

The consistency horizon: sweep the premise->claim gap and turn the parent SUPPORT result (oscillation
required for long-range consistency-checking) into a curve, with its mechanism confirmed. Same model and
phase-clamp as run_consistency_oscillation.py (imported). For each gap: FREE (theta learnable) vs CLAMPED
(theta==0) test accuracy; and mag_max, the largest eigenvalue magnitude the CLAMPED model learned -- the
slowest-decaying channel, whose surviving signal mag_max**gap should predict where decay collapses.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import run_consistency_oscillation as R      # reuse CLRU/Block/ConsistencySSM/lin_scan/seq_scan/test_acc

HERE = Path(__file__).resolve().parent
DEV = R.DEV
SMOKE = "--smoke" in sys.argv
T_LEN = R.T_LEN                               # 256
GAPS = [1, 8, 255] if SMOKE else [1, 2, 4, 8, 16, 32, 64, 128, 255]
SEEDS = [0] if SMOKE else [0, 1]
STEPS = 300 if SMOKE else 2500
N_TRAIN, N_TEST = (4000, 1000) if SMOKE else (24000, 6000)
BATCH, LR, WD = 64, 3e-3, 0.01
DATA_SEED = 4242


def make_data_gap(gap, n, seed):
    """cmp task (label = claim==premise); premise at position T-1-gap, claim at T-1. Deterministic."""
    g = np.random.default_rng(seed)
    X = np.zeros((n, T_LEN, 2), dtype=np.float32)
    premise = g.choice([-1.0, 1.0], size=n)
    claim = g.choice([-1.0, 1.0], size=n)
    X[np.arange(n), T_LEN - 1 - gap, 0] = premise
    X[np.arange(n), T_LEN - 1, 1] = claim
    y = (claim == premise).astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def train(free, seed, xtr, ytr, xte, yte):
    torch.manual_seed(seed); np.random.seed(seed)
    m = R.ConsistencySSM(free).to(DEV)
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


def mag_max(m):
    """Largest eigenvalue magnitude learned across all CLRU units/blocks (exp(-exp(nu)))."""
    vals = []
    for b in m.blocks:
        vals.append(torch.exp(-torch.exp(b.ssm.nu)).detach().max().item())
    return float(max(vals))


def spearman(a, b):
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean(); br -= br.mean()
    d = np.sqrt((ar * ar).sum() * (br * br).sum())
    return float((ar * br).sum() / d) if d > 0 else 0.0


def redteam():
    torch.manual_seed(7)
    Ac = torch.randn(3, 40, 8, dtype=torch.cfloat, device=DEV) * 0.5
    Xc = torch.randn(3, 40, 8, dtype=torch.cfloat, device=DEV)
    ds = (R.lin_scan(Ac, Xc) - R.seq_scan(Ac, Xc)).abs().max().item()
    assert ds < 1e-4, f"scan!=seq ({ds:.2e})"
    torch.manual_seed(0); mf = R.ConsistencySSM(True)
    torch.manual_seed(0); mc = R.ConsistencySSM(False)
    assert torch.equal(mf.blocks[0].ssm.B_re, mc.blocks[0].ssm.B_re), "B_re RNG mismatch"
    assert torch.equal(mf.blocks[0].ssm.nu, mc.blocks[0].ssm.nu), "nu RNG mismatch"
    for gp in (1, 8, 255):
        X, y = make_data_gap(gp, 1500, 999)
        pos = np.array([np.where(X[b, :, 0].numpy() != 0)[0][0] for b in range(1500)])
        assert (pos == T_LEN - 1 - gp).all(), f"premise not at T-1-{gp}"
        assert abs(y.float().mean() - 0.5) < 0.05, "labels unbalanced"
    assert 0.0 < mag_max(mc) < 1.0, "mag_max out of (0,1)"
    print(f"  [redteam] scan==seq ({ds:.1e}); RNG matched; premise placement OK; mag_max(init)={mag_max(mc):.4f} -- OK", flush=True)


def interp_horizon(gaps, accs, thr=0.75):
    """First gap (log-interpolated) where the CLAMPED curve crosses thr from above."""
    lg = np.log2(np.array(gaps, float))
    a = np.array(accs, float)
    for i in range(1, len(a)):
        if a[i - 1] >= thr > a[i]:
            t = (a[i - 1] - thr) / (a[i - 1] - a[i] + 1e-12)
            return float(2 ** (lg[i - 1] + t * (lg[i] - lg[i - 1])))
    if a[-1] >= thr:
        return float("inf")               # never crosses -> no finite horizon
    return float(gaps[0])                 # already below at gap 1


def main():
    print(f"device={DEV} smoke={SMOKE} T={T_LEN} gaps={GAPS} steps={STEPS} seeds={SEEDS}", flush=True)
    redteam()
    res = {"config": {"T": T_LEN, "gaps": GAPS, "steps": STEPS, "seeds": SEEDS, "n_train": N_TRAIN,
                      "n_test": N_TEST, "parent": "RESULT_consistency_oscillation_2026_07_23"},
           "free_acc": {}, "clamped_acc": {}, "clamped_mag_max": {}}
    for gap in GAPS:
        (xtr, ytr) = make_data_gap(gap, N_TRAIN, DATA_SEED)
        (xte, yte) = make_data_gap(gap, N_TEST, DATA_SEED + 1)
        for free in (True, False):
            arm = "free" if free else "clamped"
            accs, mags = [], []
            for s in SEEDS:
                t0 = time.time()
                m = train(free, s, xtr, ytr, xte, yte)
                acc = R.test_acc(m, xte, yte)
                accs.append(acc)
                if not free:
                    mags.append(mag_max(m))
                print(f"  gap {gap:>3} {arm} seed {s}: {acc:.4f}"
                      + (f"  mag_max {mags[-1]:.4f}" if not free else "") + f"  ({time.time()-t0:.0f}s)", flush=True)
                del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            (res["free_acc"] if free else res["clamped_acc"])[str(gap)] = [round(a, 4) for a in accs]
            if not free:
                res["clamped_mag_max"][str(gap)] = [round(x, 5) for x in mags]

    free_mean = [float(np.mean(res["free_acc"][str(g)])) for g in GAPS]
    clamp_mean = [float(np.mean(res["clamped_acc"][str(g)])) for g in GAPS]
    magmax_mean = [float(np.mean(res["clamped_mag_max"][str(g)])) for g in GAPS]
    surviving = [mm ** g for mm, g in zip(magmax_mean, GAPS)]           # mag_max**gap: retained premise signal
    H = interp_horizon(GAPS, clamp_mean, 0.75)

    # mechanism check: does CLAMPED accuracy track the surviving signal?
    rho = spearman(clamp_mean, surviving)
    hi = [surviving[i] for i, a in enumerate(clamp_mean) if a >= 0.90]
    lo = [surviving[i] for i, a in enumerate(clamp_mean) if a <= 0.60]
    clean_sep = bool(hi and lo and (min(hi) > max(lo)))

    free_ok = all(a >= 0.95 for a in free_mean)
    adj_ok = clamp_mean[0] >= 0.90
    finite_h = (1.0 < H < 255.0)
    mech_ok = (rho >= 0.80) and clean_sep
    if not (free_ok and adj_ok):
        verdict = "ABSTAIN__controls_failed"
    elif all(a >= 0.90 for a in clamp_mean):
        verdict = "ANOMALY__decay_never_fails"
    elif finite_h and mech_ok:
        verdict = "CONFIRM__finite_consistency_horizon_with_mechanism"
    elif finite_h:
        verdict = "PARTIAL__horizon_without_confirmed_mechanism"
    else:
        verdict = "PARTIAL__reported_verbatim"

    res["result"] = {
        "free_mean_by_gap": [round(a, 4) for a in free_mean],
        "clamped_mean_by_gap": [round(a, 4) for a in clamp_mean],
        "clamped_magmax_by_gap": [round(a, 5) for a in magmax_mean],
        "surviving_signal_magmax_pow_gap": [round(a, 5) for a in surviving],
        "consistency_horizon_gap": (round(H, 2) if np.isfinite(H) else None),
        "free_range_free": bool(free_ok), "clamped_adjacent_ok": bool(adj_ok),
        "mechanism_spearman_acc_vs_surviving": round(rho, 4),
        "mechanism_clean_separation": clean_sep,
        "verdict": verdict,
    }
    out = HERE / ("consistency_horizon_smoke.json" if SMOKE else "consistency_horizon_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  gap:      " + "  ".join(f"{g:>5}" for g in GAPS))
    print("  FREE:     " + "  ".join(f"{a:>5.3f}" for a in free_mean))
    print("  CLAMPED:  " + "  ".join(f"{a:>5.3f}" for a in clamp_mean))
    print("  survive:  " + "  ".join(f"{a:>5.3f}" for a in surviving))
    print(f"  consistency horizon (CLAMPED crosses 0.75): gap {H:.2f}" if np.isfinite(H) else "  horizon: none")
    print(f"  mechanism: Spearman(acc, mag_max^gap)={rho:.3f}  clean-sep={clean_sep}")
    print("  ===== VERDICT:", verdict, "=====")
    print("  wrote", out.name, flush=True)


if __name__ == "__main__":
    main()

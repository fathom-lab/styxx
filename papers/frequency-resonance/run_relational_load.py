# -*- coding: utf-8 -*-
"""
run_relational_load.py -- frozen by PREREG_relational_load_2026_07_24.

Does oscillation's advantage scale with RELATIONAL load while sparing STORAGE load? Two orthogonal axes
at matched distance/parameters, same CLRU phase-clamp (FREE theta-learnable vs CLAMPED theta==0):
  axis R (treatment): hold R facts, compare ALL R against R claims (conjunction of comparisons).
  axis S (control):   hold S facts, report ONE named by a one-hot selector (no comparison at all).
Success-probability over seeds (decay trainability is bimodal). Storage is matched across axes at equal
load index; only the required operation differs.
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
T_LEN, H, N_BLK = R.T_LEN, R.H, R.N_BLK
LOADS = [1, 4] if SMOKE else [1, 2, 3, 4]
MAXL = 4
# Premise slots INSIDE decay's competent range (gaps 4..10 from the probe at T-1; the horizon result
# put decay's half-horizon near gap 32). Load-1 must have headroom, else the dose-response is unmeasurable
# -- the frozen ABSTAIN clause. Identical across both axes, so storage/distance are matched.
POSITIONS = [T_LEN - 5, T_LEN - 7, T_LEN - 9, T_LEN - 11]
FREE_SEEDS = [0, 1]
CLAMP_SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
STEPS = 300 if SMOKE else 2000
N_TRAIN, N_TEST = (4000, 1000) if SMOKE else (24000, 6000)
BATCH, LR, WD = 64, 3e-3, 0.01
DATA_SEED = 31337
SOLVE_THR = 0.80

# channels: MAXL premise slots + MAXL claim/selector slots
C_IN = 2 * MAXL


class LoadSSM(nn.Module):
    def __init__(self, free):
        super().__init__()
        self.emb = nn.Linear(C_IN, H, bias=False)
        self.blocks = nn.ModuleList([R.Block(free) for _ in range(N_BLK)])
        self.head = nn.Linear(H, 2)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x[:, -1])


def make_data(axis, load, n, seed):
    """axis 'R': label = all `load` claims match their premises (conjunction of comparisons).
       axis 'S': label = sign of the ONE premise named by a one-hot selector (storage only, no compare).
       Both place `load` premise bits at the SAME fixed positions -> storage matched."""
    g = np.random.default_rng(seed)
    X = np.zeros((n, T_LEN, C_IN), dtype=np.float32)
    prem = g.choice([-1.0, 1.0], size=(n, load))
    for j in range(load):
        X[np.arange(n), POSITIONS[j], j] = prem[:, j]
    if axis == "R":
        # claims at the probe; label = ALL match. Balance: force ~50% all-match.
        allmatch = g.random(n) < 0.5
        claims = prem.copy()
        for b in range(n):
            if not allmatch[b]:                     # flip a random non-empty subset -> not all match
                k = g.integers(1, load + 1)
                idx = g.choice(load, size=k, replace=False)
                claims[b, idx] *= -1.0
        for j in range(load):
            X[np.arange(n), T_LEN - 1, MAXL + j] = claims[:, j]
        y = (claims == prem).all(1).astype(np.int64)
    else:
        sel = g.integers(0, load, size=n)           # one-hot selector at the probe
        X[np.arange(n), T_LEN - 1, MAXL + sel] = 1.0
        y = (prem[np.arange(n), sel] > 0).astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def train(free, seed, xtr, ytr):
    torch.manual_seed(seed); np.random.seed(seed)
    m = LoadSSM(free).to(DEV)
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
    torch.manual_seed(0); mf = LoadSSM(True)
    torch.manual_seed(0); mc = LoadSSM(False)
    assert torch.equal(mf.blocks[0].ssm.B_re, mc.blocks[0].ssm.B_re), "B_re RNG mismatch"
    for load in LOADS:
        XR, yR = make_data("R", load, 3000, 999)
        XS, yS = make_data("S", load, 3000, 999)
        # storage matched: same premise positions, same count of nonzero premise slots
        nzR = (XR[:, :, :MAXL].abs().sum(2) > 0).sum(1).float().mean().item()
        nzS = (XS[:, :, :MAXL].abs().sum(2) > 0).sum(1).float().mean().item()
        assert abs(nzR - load) < 1e-6 and abs(nzS - load) < 1e-6, f"premise count != load ({nzR},{nzS})"
        posR = set(np.where(XR[0, :, :MAXL].abs().sum(1).numpy() > 0)[0].tolist())
        posS = set(np.where(XS[0, :, :MAXL].abs().sum(1).numpy() > 0)[0].tolist())
        assert posR == posS == set(POSITIONS[:load]), "premise positions differ across axes"
        assert abs(yR.float().mean() - 0.5) < 0.08, f"R labels unbalanced at load {load}"
        assert abs(yS.float().mean() - 0.5) < 0.08, f"S labels unbalanced at load {load}"
    print(f"  [redteam] scan==seq ({ds:.1e}); RNG matched; storage+positions matched across axes; "
          f"labels balanced -- OK", flush=True)


def main():
    print(f"device={DEV} smoke={SMOKE} T={T_LEN} loads={LOADS} positions={POSITIONS} steps={STEPS} "
          f"free={FREE_SEEDS} clamp={CLAMP_SEEDS}", flush=True)
    redteam()
    res = {"config": {"T": T_LEN, "loads": LOADS, "positions": POSITIONS, "steps": STEPS,
                      "free_seeds": FREE_SEEDS, "clamp_seeds": CLAMP_SEEDS, "solve_threshold": SOLVE_THR,
                      "axes": {"R": "conjunction of comparisons", "S": "storage + selection (control)"}},
           "free_acc": {}, "clamped_acc": {}}
    for axis in ("R", "S"):
        for load in LOADS:
            xtr, ytr = make_data(axis, load, N_TRAIN, DATA_SEED)
            xte, yte = make_data(axis, load, N_TEST, DATA_SEED + 1)
            facc = []
            for s in FREE_SEEDS:
                m = train(True, s, xtr, ytr); facc.append(R.test_acc(m, xte, yte))
                del m; torch.cuda.empty_cache() if DEV == "cuda" else None
            cacc = []
            for s in CLAMP_SEEDS:
                t0 = time.time()
                m = train(False, s, xtr, ytr); a = R.test_acc(m, xte, yte); cacc.append(a)
                print(f"  axis {axis} load {load} clamped seed {s}: {a:.4f} ({time.time()-t0:.0f}s)", flush=True)
                del m; torch.cuda.empty_cache() if DEV == "cuda" else None
            res["free_acc"][f"{axis}{load}"] = [round(a, 4) for a in facc]
            res["clamped_acc"][f"{axis}{load}"] = [round(a, 4) for a in cacc]
            print(f"  axis {axis} load {load}: FREE solve {np.mean([a>=SOLVE_THR for a in facc]):.2f}  "
                  f"CLAMPED solve {np.mean([a>=SOLVE_THR for a in cacc]):.2f}", flush=True)

    def solve(axis, load, arm):
        key = f"{axis}{load}"
        return float(np.mean([a >= SOLVE_THR for a in res[f"{arm}_acc"][key]]))
    pR = [solve("R", k, "clamped") for k in LOADS]
    pS = [solve("S", k, "clamped") for k in LOADS]
    fR = [solve("R", k, "free") for k in LOADS]
    fS = [solve("S", k, "free") for k in LOADS]

    free_ok = all(x >= 1.0 for x in fR + fS)
    headroom = pR[0] >= 0.60
    monotone = all(pR[i] <= pR[i - 1] + 0.20 for i in range(1, len(pR)))
    rel_drop = pR[0] - pR[-1]
    sto_drop = pS[0] - pS[-1]
    if not (free_ok and headroom):
        verdict = "ABSTAIN__controls_failed"
    elif monotone and rel_drop >= 0.40 and sto_drop <= 0.20:
        verdict = "CONFIRM__relational_dose_response_storage_spared"
    elif rel_drop <= 0.10 or sto_drop >= rel_drop:
        verdict = "NULL__deficit_not_relational"
    else:
        verdict = "PARTIAL__reported_verbatim"

    res["result"] = {
        "loads": LOADS,
        "clamped_solve_relational": [round(x, 3) for x in pR],
        "clamped_solve_storage": [round(x, 3) for x in pS],
        "free_solve_relational": [round(x, 3) for x in fR],
        "free_solve_storage": [round(x, 3) for x in fS],
        "relational_drop": round(rel_drop, 3), "storage_drop": round(sto_drop, 3),
        "free_range_free": bool(free_ok), "headroom_at_load1": bool(headroom),
        "monotone_within_tol": bool(monotone), "verdict": verdict,
    }
    out = HERE / ("relational_load_smoke.json" if SMOKE else "relational_load_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  load:                " + "  ".join(f"{k:>5}" for k in LOADS))
    print("  CLAMPED relational:  " + "  ".join(f"{x:>5.2f}" for x in pR))
    print("  CLAMPED storage:     " + "  ".join(f"{x:>5.2f}" for x in pS))
    print("  FREE relational:     " + "  ".join(f"{x:>5.2f}" for x in fR))
    print(f"  relational drop {rel_drop:+.2f}  vs  storage drop {sto_drop:+.2f}")
    print("  ===== VERDICT:", verdict, "=====")
    print("  wrote", out.name, flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
run_multiplexing.py — frozen by PREREG_multiplexing_2026_06_04.

Oscillation's real niche? Starve the state dimension D and watch whether oscillation's advantage GROWS.
LRU-FREE (oscillatory) vs LRU-CLAMPED (decay only), ordered copy, swept over D. If ratio(D)=kcap_free/
kcap_clamped rises as D shrinks, oscillation multiplexes items into phases when modes are scarce — the
theta-gamma capacity-per-resource claim, and why biology uses rhythm even though silicon need not.
"""
from __future__ import annotations
import sys, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = "--smoke" in sys.argv
V, KMAX, D_IN = 12, 20, 64
STEPS = 600 if SMOKE else 4000
BATCH, LR = 64, 2e-3
SEEDS = [0] if SMOKE else [0, 1, 2]
KGRID = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 20]
ACC_THR = 0.80
D_SWEEP = [16, 64] if SMOKE else [16, 24, 32, 48, 64, 128, 256]


class CLRU(nn.Module):
    def __init__(self, d, d_in, free):
        super().__init__()
        r = torch.empty(d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        th = torch.empty(d).uniform_(0.0, math.pi / 2)
        if free:
            self.theta = nn.Parameter(th)
        else:
            self.register_buffer("theta", torch.zeros(d))
        self.B_re = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        self.B_im = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))

    def forward(self, x):
        mag = torch.exp(-torch.exp(self.nu))
        lr = mag * torch.cos(self.theta); li = mag * torch.sin(self.theta)
        gamma = torch.sqrt(torch.clamp(1 - mag ** 2, min=1e-6))
        ure = torch.einsum("bti,di->btd", x, self.B_re) * gamma
        uim = torch.einsum("bti,di->btd", x, self.B_im) * gamma
        hre = torch.zeros(x.shape[0], mag.shape[0], device=x.device); him = torch.zeros_like(hre)
        outs = []
        for t in range(x.shape[1]):
            nhre = lr * hre - li * him + ure[:, t, :]
            nhim = lr * him + li * hre + uim[:, t, :]
            hre, him = nhre, nhim
            outs.append(torch.cat([hre, him], -1))
        return torch.stack(outs, 1)


class Model(nn.Module):
    def __init__(self, d, free):
        super().__init__()
        self.emb = nn.Embedding(V + 1, D_IN)
        self.lru = CLRU(d, D_IN, free)
        self.read = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, V))

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


def make_batch(B, K):
    syms = torch.randint(0, V, (B, K), device=DEV)
    inp = torch.cat([syms, torch.full((B, K), V, device=DEV)], 1)
    tgt = torch.full((B, 2 * K), -100, device=DEV); tgt[:, K:] = syms
    return inp, tgt


def train(d, free, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(d, free).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for _ in range(STEPS):
        K = int(np.random.randint(1, KMAX + 1))
        inp, tgt = make_batch(BATCH, K)
        loss = lossf(m(inp).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


@torch.no_grad()
def eval_K(m, K, n=1024):
    inp, tgt = make_batch(n, K); pred = m(inp).argmax(-1); mask = tgt != -100
    return (pred[mask] == tgt[mask]).float().mean().item()


def kcap(m):
    cap = 0
    for K in KGRID:
        if eval_K(m, K) >= ACC_THR:
            cap = K
    return cap


def main():
    res = {"config": {"steps": STEPS, "seeds": SEEDS, "d_sweep": D_SWEEP}, "by_d": {}}
    for d in D_SWEEP:
        row = {}
        for free, name in [(True, "free"), (False, "clamped")]:
            caps = []
            for s in SEEDS:
                m = train(d, free, s); caps.append(kcap(m)); del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            row[name] = float(np.mean(caps))
        ratio = row["free"] / row["clamped"] if row["clamped"] > 0 else float("inf")
        row["ratio"] = ratio
        res["by_d"][str(d)] = row
        print(f"  D={d:3d}: free {row['free']:.2f} | clamped {row['clamped']:.2f} | ratio {ratio:.2f}", flush=True)

    ds = D_SWEEP
    r_small = res["by_d"][str(ds[0])]["ratio"]
    r_large = res["by_d"][str(ds[-1])]["ratio"]
    ratios = [res["by_d"][str(d)]["ratio"] for d in ds]
    # monotone-ish downward in D? Spearman(D, ratio) negative
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        den = math.sqrt((rx**2).sum() * (ry**2).sum())
        return float((rx*ry).sum()/den) if den else 0.0
    rho = spearman(np.array(ds, float), np.array([r if np.isfinite(r) else 99 for r in ratios]))
    p1 = (r_small >= r_large + 0.5) and (rho <= -0.3)
    if p1:
        verdict = (f"OSCILLATION'S NICHE IS RESOURCE-CONSTRAINED MULTIPLEXING — its relative advantage GROWS as the "
                   f"state is starved (ratio {r_large:.2f} at D={ds[-1]} -> {r_small:.2f} at D={ds[0]}, Spearman {rho:.2f}). "
                   "Oscillation packs items into phases when modes are scarce: not special for raw capacity (attention "
                   "wins) but THE mechanism for capacity-per-resource -- why resource-bound biology uses rhythm.")
    else:
        verdict = (f"ADVANTAGE IS RESOURCE-FLAT — ratio(D) does not clearly rise under scarcity (ratio {r_large:.2f} at "
                   f"D={ds[-1]} -> {r_small:.2f} at D={ds[0]}, Spearman {rho:.2f}). Oscillation's edge is ~a fixed "
                   "multiplier, not a scarcity-multiplexing mechanism. Report the curve.")
    res["gate"] = {"ratio_small_D": r_small, "ratio_large_D": r_large, "ratios": ratios,
                   "spearman_D_ratio": round(rho, 4), "P1_advantage_grows_under_scarcity": bool(p1), "verdict": verdict}
    out = HERE / ("multiplexing_smoke.json" if SMOKE else "multiplexing_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  ratio(D):", {d: round(res["by_d"][str(d)]["ratio"], 2) for d in ds})
    print("\n===== " + verdict)
    print("wrote", out.name)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
run_noise_robustness.py — frozen by PREREG_noise_2026_06_04.

The phase-vs-rate robustness claim: is the oscillatory (phase) code more noise-robust than the decay
(magnitude) code? Inject Gaussian noise into the recurrent state at EVERY timestep (train + eval), sweep
sigma, hold D=256, compare LRU-FREE vs LRU-CLAMPED capacity. If ratio(sigma)=kcap_free/kcap_clamped RISES
with sigma, oscillation's niche is noise robustness — the most-cited reason biology phase-codes. If flat,
the robustness reconciliation does not hold either.
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
V, KMAX, D_IN, D = 12, 20, 64, 256
STEPS = 600 if SMOKE else 4000
BATCH, LR = 64, 2e-3
SEEDS = [0] if SMOKE else [0, 1, 2]
KGRID = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 20]
ACC_THR = 0.80
SIGMA_SWEEP = [0.0, 0.2] if SMOKE else [0.0, 0.05, 0.1, 0.2, 0.4]


class CLRU(nn.Module):
    def __init__(self, d, d_in, free, noise_std):
        super().__init__()
        self.noise_std = noise_std
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
        s = self.noise_std
        outs = []
        for t in range(x.shape[1]):
            nhre = lr * hre - li * him + ure[:, t, :]
            nhim = lr * him + li * hre + uim[:, t, :]
            if s > 0:  # additive state noise, same treatment both arms, train + eval
                nhre = nhre + s * torch.randn_like(nhre)
                nhim = nhim + s * torch.randn_like(nhim)
            hre, him = nhre, nhim
            outs.append(torch.cat([hre, him], -1))
        return torch.stack(outs, 1)


class Model(nn.Module):
    def __init__(self, free, noise_std):
        super().__init__()
        self.emb = nn.Embedding(V + 1, D_IN)
        self.lru = CLRU(D, D_IN, free, noise_std)
        self.read = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, V))

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


def make_batch(B, K):
    syms = torch.randint(0, V, (B, K), device=DEV)
    inp = torch.cat([syms, torch.full((B, K), V, device=DEV)], 1)
    tgt = torch.full((B, 2 * K), -100, device=DEV); tgt[:, K:] = syms
    return inp, tgt


def train(free, noise_std, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(free, noise_std).to(DEV)
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
    # average several noisy passes so kcap reflects expected noisy accuracy, not one draw
    inp, tgt = make_batch(n, K); mask = tgt != -100
    accs = []
    reps = 1 if m.lru.noise_std == 0 else 3
    for _ in range(reps):
        pred = m(inp).argmax(-1)
        accs.append((pred[mask] == tgt[mask]).float().mean().item())
    return float(np.mean(accs))


def kcap(m):
    cap = 0
    for K in KGRID:
        if eval_K(m, K) >= ACC_THR:
            cap = K
    return cap


def main():
    res = {"config": {"steps": STEPS, "seeds": SEEDS, "D": D, "sigma_sweep": SIGMA_SWEEP}, "by_sigma": {}}
    for sg in SIGMA_SWEEP:
        row = {}
        for free, name in [(True, "free"), (False, "clamped")]:
            caps = []
            for s in SEEDS:
                m = train(free, sg, s); caps.append(kcap(m)); del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            row[name] = float(np.mean(caps))
        ratio = row["free"] / row["clamped"] if row["clamped"] > 0 else float("inf")
        row["ratio"] = ratio
        res["by_sigma"][str(sg)] = row
        print(f"  sigma={sg:.2f}: free {row['free']:.2f} | clamped {row['clamped']:.2f} | ratio {ratio:.2f}", flush=True)

    sg = SIGMA_SWEEP
    r_lo = res["by_sigma"][str(sg[0])]["ratio"]
    r_hi = res["by_sigma"][str(sg[-1])]["ratio"]
    ratios = [res["by_sigma"][str(s)]["ratio"] for s in sg]

    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        den = math.sqrt((rx**2).sum() * (ry**2).sum())
        return float((rx*ry).sum()/den) if den else 0.0
    rho = spearman(np.array(sg, float), np.array([r if np.isfinite(r) else 99 for r in ratios]))

    # P2: fractional kcap drop sigma=0 -> sigma_max, free vs clamped
    free0 = res["by_sigma"][str(sg[0])]["free"]; freeH = res["by_sigma"][str(sg[-1])]["free"]
    clamp0 = res["by_sigma"][str(sg[0])]["clamped"]; clampH = res["by_sigma"][str(sg[-1])]["clamped"]
    free_drop = (free0 - freeH) / free0 if free0 > 0 else float("nan")
    clamp_drop = (clamp0 - clampH) / clamp0 if clamp0 > 0 else float("nan")

    p1 = (rho >= 0.3) and (r_hi >= r_lo + 0.5)
    p2 = bool(np.isfinite(free_drop) and np.isfinite(clamp_drop) and clamp_drop > free_drop)
    if p1:
        verdict = (f"OSCILLATION'S NICHE IS NOISE ROBUSTNESS — its advantage GROWS under state noise "
                   f"(ratio {r_lo:.2f} at sigma=0 -> {r_hi:.2f} at sigma={sg[-1]}, Spearman {rho:.2f}). The phase "
                   "code survives additive corruption the decay/magnitude code cannot: not special for raw "
                   "capacity (attention wins) but THE noise-robust code -- a real 'why' a noisy brain phase-codes.")
    else:
        verdict = (f"NOISE-FLAT / REFUTED — ratio(sigma) does not clearly rise (ratio {r_lo:.2f} at sigma=0 -> "
                   f"{r_hi:.2f} at sigma={sg[-1]}, Spearman {rho:.2f}). Oscillation's edge is noise-independent; "
                   "the robustness reconciliation does not hold either. Report the curve.")
    res["gate"] = {"ratio_lo_sigma": r_lo, "ratio_hi_sigma": r_hi, "ratios": ratios,
                   "spearman_sigma_ratio": round(rho, 4), "free_frac_drop": round(free_drop, 4),
                   "clamped_frac_drop": round(clamp_drop, 4), "P1_advantage_grows_under_noise": bool(p1),
                   "P2_clamped_degrades_faster": p2, "verdict": verdict}
    out = HERE / ("noise_smoke.json" if SMOKE else "noise_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  ratio(sigma):", {s: round(res["by_sigma"][str(s)]["ratio"], 2) for s in sg})
    print(f"  free drop {free_drop:.2f} | clamped drop {clamp_drop:.2f} (P2 clamped-degrades-faster={p2})")
    print("\n===== " + verdict)
    print("wrote", out.name)


if __name__ == "__main__":
    main()

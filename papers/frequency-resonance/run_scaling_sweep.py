# -*- coding: utf-8 -*-
"""
run_scaling_sweep.py — frozen by PREREG_scaling_law_2026_06_04.

Tests theta* x window = const. Ordered-copy with an inserted delay of D PAD tokens; per delay D,
sweep frozen oscillation frequency theta and locate the optimum theta*(D) at the CAPACITY EDGE
(kcap). If the resonance is phase-coding over the retention window, theta* falls as 1/W. Same LRU
rig/config as run_frequency_sweep.py; only the delay is new. D=0 should reproduce the frequency
sweep's theta*~0.375pi (internal control).
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

# ---- frozen config (matches rhythm-rescue / frequency sweep) ----
V = 12
PAD = V + 1            # symbols 0..V-1, GO=V, PAD=V+1
KMAX_TRAIN = 10
D_MODEL = 256
D_IN = 64
STEPS = 4000
BATCH = 64
LR = 2e-3
SEEDS = [0, 1, 2]
KGRID = [1, 2, 3, 4, 5, 6, 8, 10]
ACC_THR = 0.80
DELAYS = [0, 6, 12, 24]                                       # W = D + kcap*(D)
THETA_FRACS = [0.03, 0.0625, 0.125, 0.25, 0.375, 0.5]        # x pi
EVAL_N = 1024

if SMOKE:
    STEPS, SEEDS = 500, [0]
    DELAYS = [0, 12]
    THETA_FRACS = [0.0625, 0.25, 0.375]
    KGRID = [1, 2, 3, 4, 5, 6]


class CLRU(nn.Module):
    def __init__(self, d, d_in, theta):
        super().__init__()
        r = torch.empty(d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        th = torch.empty(d).uniform_(0.0, math.pi / 2)   # consume RNG identically
        self.register_buffer("theta", torch.full((d,), float(theta)))
        self.B_re = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        self.B_im = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))

    def forward(self, x):
        mag = torch.exp(-torch.exp(self.nu))
        lr = mag * torch.cos(self.theta); li = mag * torch.sin(self.theta)
        gamma = torch.sqrt(torch.clamp(1 - mag ** 2, min=1e-6))
        ure = torch.einsum("bti,di->btd", x, self.B_re) * gamma
        uim = torch.einsum("bti,di->btd", x, self.B_im) * gamma
        hre = torch.zeros(x.shape[0], mag.shape[0], device=x.device)
        him = torch.zeros_like(hre)
        outs = []
        for t in range(x.shape[1]):
            nhre = lr * hre - li * him + ure[:, t, :]
            nhim = lr * him + li * hre + uim[:, t, :]
            hre, him = nhre, nhim
            outs.append(torch.cat([hre, him], -1))
        return torch.stack(outs, 1)


class Model(nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.emb = nn.Embedding(V + 2, D_IN)          # +GO +PAD
        self.lru = CLRU(D_MODEL, D_IN, theta)
        self.read = nn.Sequential(nn.Linear(2 * D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, V))

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


def make_batch(B, K, D):
    syms = torch.randint(0, V, (B, K), device=DEV)
    parts = [syms, torch.full((B, D), PAD, device=DEV), torch.full((B, K), V, device=DEV)]
    inp = torch.cat(parts, 1)                          # [K syms][D pad][K GO]
    L = K + D + K
    tgt = torch.full((B, L), -100, device=DEV)
    tgt[:, K + D:] = syms                              # recall in order at the GO slots
    return inp, tgt


def train_model(theta, D, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(theta).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(STEPS):
        K = int(np.random.randint(1, KMAX_TRAIN + 1))
        inp, tgt = make_batch(BATCH, K, D)
        loss = lossf(m(inp).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


@torch.no_grad()
def acc_K(m, K, D, n=EVAL_N):
    inp, tgt = make_batch(n, K, D)
    pred = m(inp).argmax(-1); mask = tgt != -100
    return (pred[mask] == tgt[mask]).float().mean().item()


def kcap(m, D):
    cap = 0
    for K in KGRID:
        if acc_K(m, K, D) >= ACC_THR:
            cap = K
    return cap


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else 0.0


def main():
    res = {"config": {"d_model": D_MODEL, "v": V, "steps": STEPS, "seeds": SEEDS,
                      "kgrid": KGRID, "acc_thr": ACC_THR, "delays": DELAYS,
                      "theta_fracs": THETA_FRACS, "kmax_train": KMAX_TRAIN, "smoke": SMOKE},
           "by_delay": {}}
    for D in DELAYS:
        kcap_by_theta = {}
        for frac in THETA_FRACS:
            theta = frac * math.pi
            caps = []
            for seed in SEEDS:
                m = train_model(theta, D, seed)
                caps.append(kcap(m, D))
                del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            kcap_by_theta[f"{frac:.4f}"] = round(float(np.mean(caps)), 4)
            print(f"D={D:2d} theta={frac:.3f}pi: kcap={np.mean(caps):.3f}", flush=True)
        fracs = THETA_FRACS
        caps = [kcap_by_theta[f"{f:.4f}"] for f in fracs]
        istar = int(np.argmax(caps))
        kcap_star = max(caps)
        W = D + kcap_star
        res["by_delay"][str(D)] = {"kcap_by_theta": kcap_by_theta, "theta_star_frac": fracs[istar],
                                   "kcap_star": kcap_star, "window": round(W, 3),
                                   "learnable": kcap_star >= 2}
        print(f"  -> D={D}: theta*={fracs[istar]:.3f}pi  kcap*={kcap_star:.2f}  W={W:.1f}"
              f"  {'OK' if kcap_star>=2 else 'DROP(unlearnable)'}", flush=True)

    # ---- frozen decision rule ----
    used = [(D, res["by_delay"][str(D)]) for D in DELAYS if res["by_delay"][str(D)]["learnable"]]
    Ds = [d for d, _ in used]
    tstars_frac = [v["theta_star_frac"] for _, v in used]
    Ws = [v["window"] for _, v in used]
    products = [f * math.pi * w for f, w in zip(tstars_frac, Ws)]      # rad
    rho = spearman(Ds, tstars_frac) if len(Ds) >= 2 else 0.0
    cv = float(np.std(products) / np.mean(products)) if products and np.mean(products) else 9.9
    grid_step = THETA_FRACS[1] - THETA_FRACS[0]
    span_steps = (max(tstars_frac) - min(tstars_frac)) / grid_step if len(tstars_frac) > 1 else 0

    if len(Ds) >= 3 and rho <= -0.80 and cv <= 0.35:
        verdict = (f"SCALING LAW — theta* x window ~ constant ({np.mean(products)/math.pi:.2f}pi rad, "
                   f"CV {cv:.2f}); theta* falls with the window (Spearman {rho:.2f}). The resonance is "
                   f"phase-coding over the retention window: longer hold -> slower optimal rhythm.")
    elif len(Ds) >= 3 and rho <= -0.80:
        verdict = (f"DIRECTIONAL — theta* falls with the window (Spearman {rho:.2f}) but theta*xW is "
                   f"not constant at bar (CV {cv:.2f}). Longer hold needs slower rhythm; exact 1/W not "
                   f"established.")
    elif span_steps <= 1.0:
        verdict = (f"NULL — theta* does not move with the window (span {span_steps:.1f} grid steps). "
                   f"The resonance is an architecture-fixed property, not window-tuned.")
    else:
        verdict = (f"PARTIAL — theta* moves (Spearman {rho:.2f}, span {span_steps:.1f} steps) but "
                   f"neither the falling-trend nor the 1/W law cleared bar. Report shape.")

    res["gate"] = {
        "delays_used": Ds, "theta_star_frac_by_delay": tstars_frac, "windows": Ws,
        "product_thetaXW_rad": [round(p, 4) for p in products],
        "product_mean_over_pi": round(float(np.mean(products)) / math.pi, 4) if products else None,
        "product_cv": round(cv, 4), "spearman_D_thetastar": round(rho, 4),
        "span_grid_steps": round(span_steps, 2), "verdict": verdict,
    }
    out = HERE / ("scaling_sweep_smoke.json" if SMOKE else "scaling_sweep_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n===== " + verdict)
    print("wrote " + out.name)


if __name__ == "__main__":
    main()

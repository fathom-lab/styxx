# -*- coding: utf-8 -*-
"""
run_scaling_sweep_clean.py — de-confounded re-run of PREREG_scaling_law_2026_06_04.

The first sweep (run_scaling_sweep.py) located theta* by integer kcap, which SATURATED: several low
theta tied at kcap=6 and argmax tie-broke to the lowest, blurring theta* and risking a measurement
NULL. Fix (methodological, same hypothesis theta*xW=const): locate theta* by a CONTINUOUS capacity
measure -- area under the accuracy-vs-K curve, sum_K mean_acc(K) -- which has no ties, and SAVE the
full per-K accuracy curves so any finer measure is reproducible post-hoc. Same rig/config otherwise.
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

V = 12
PAD = V + 1
KMAX_TRAIN = 10
D_MODEL = 256
D_IN = 64
STEPS = 4000
BATCH = 64
LR = 2e-3
SEEDS = [0, 1, 2]
KGRID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]          # dense, for a continuous area measure
ACC_THR = 0.80                                    # only for the secondary kcap readout
DELAYS = [0, 6, 12, 24]
THETA_FRACS = [0.03, 0.0625, 0.125, 0.25, 0.375, 0.5]
EVAL_N = 1024

if SMOKE:
    STEPS, SEEDS, DELAYS, THETA_FRACS = 500, [0], [0, 12], [0.0625, 0.25, 0.375]
    KGRID = [1, 2, 3, 4, 5, 6]


class CLRU(nn.Module):
    def __init__(self, d, d_in, theta):
        super().__init__()
        r = torch.empty(d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        torch.empty(d).uniform_(0.0, math.pi / 2)        # consume RNG identically
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
        self.emb = nn.Embedding(V + 2, D_IN)
        self.lru = CLRU(D_MODEL, D_IN, theta)
        self.read = nn.Sequential(nn.Linear(2 * D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, V))

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


def make_batch(B, K, D):
    syms = torch.randint(0, V, (B, K), device=DEV)
    inp = torch.cat([syms, torch.full((B, D), PAD, device=DEV), torch.full((B, K), V, device=DEV)], 1)
    tgt = torch.full((B, K + D + K), -100, device=DEV)
    tgt[:, K + D:] = syms
    return inp, tgt


def train_model(theta, D, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(theta).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for _ in range(STEPS):
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
    return float((pred[mask] == tgt[mask]).float().mean().item())


def kcap_from_curve(curve):
    cap = 0
    for K in KGRID:
        if curve[K] >= ACC_THR:
            cap = K
    return cap


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else 0.0


def main():
    res = {"config": {"d_model": D_MODEL, "v": V, "steps": STEPS, "seeds": SEEDS, "kgrid": KGRID,
                      "delays": DELAYS, "theta_fracs": THETA_FRACS, "kmax_train": KMAX_TRAIN,
                      "measure": "area_under_accuracy", "smoke": SMOKE}, "by_delay": {}}
    for D in DELAYS:
        curves, areas, kcaps = {}, {}, {}
        for frac in THETA_FRACS:
            theta = frac * math.pi
            seed_curves = []
            for seed in SEEDS:
                m = train_model(theta, D, seed)
                seed_curves.append({K: acc_K(m, K, D) for K in KGRID})
                del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            mean_curve = {K: round(float(np.mean([c[K] for c in seed_curves])), 4) for K in KGRID}
            area = round(float(sum(mean_curve.values())), 4)        # CONTINUOUS capacity
            curves[f"{frac:.4f}"] = mean_curve
            areas[f"{frac:.4f}"] = area
            kcaps[f"{frac:.4f}"] = kcap_from_curve(mean_curve)
            print(f"D={D:2d} theta={frac:.3f}pi: area={area:.3f} kcap={kcaps[f'{frac:.4f}']}", flush=True)
        fracs = THETA_FRACS
        av = [areas[f"{f:.4f}"] for f in fracs]
        istar = int(np.argmax(av))
        kc_star = kcaps[f"{fracs[istar]:.4f}"]
        W = D + kc_star
        res["by_delay"][str(D)] = {"acc_curves": curves, "area_by_theta": areas, "kcap_by_theta": kcaps,
                                   "theta_star_frac": fracs[istar], "area_star": av[istar],
                                   "kcap_star": kc_star, "window": round(W, 3), "learnable": kc_star >= 2}
        print(f"  -> D={D}: theta*={fracs[istar]:.3f}pi (area) kcap*={kc_star} W={W}", flush=True)

    used = [(D, res["by_delay"][str(D)]) for D in DELAYS if res["by_delay"][str(D)]["learnable"]]
    Ds = [d for d, _ in used]
    tstars = [v["theta_star_frac"] for _, v in used]
    Ws = [v["window"] for _, v in used]
    prod = [f * math.pi * w for f, w in zip(tstars, Ws)]
    rho = spearman(Ds, tstars) if len(Ds) >= 2 else 0.0
    cv = float(np.std(prod) / np.mean(prod)) if prod and np.mean(prod) else 9.9
    step = THETA_FRACS[1] - THETA_FRACS[0]
    span = (max(tstars) - min(tstars)) / step if len(tstars) > 1 else 0

    if len(Ds) >= 3 and rho <= -0.80 and cv <= 0.35:
        verdict = (f"SCALING LAW (clean) — theta* falls with the window (Spearman {rho:.2f}), "
                   f"theta*xW ~ const ({np.mean(prod)/math.pi:.2f}pi, CV {cv:.2f}).")
    elif len(Ds) >= 3 and rho <= -0.80:
        verdict = f"DIRECTIONAL (clean) — theta* falls with window (Spearman {rho:.2f}) but theta*xW CV {cv:.2f} not const."
    elif span <= 1.0:
        verdict = (f"NULL (clean, de-saturated) — theta* does not move with delay (span {span:.1f} grid "
                   f"steps) even on the continuous measure. theta* is window-INDEPENDENT (item-count bound) "
                   f"-- supports the relative-phase-preservation arm of THEORY sec.3a.")
    else:
        verdict = f"MIXED (clean) — theta* span {span:.1f} steps, Spearman {rho:.2f}, CV {cv:.2f}."
    res["gate"] = {"delays_used": Ds, "theta_star_frac": tstars, "windows": Ws,
                   "product_thetaXW_pi": [round(p / math.pi, 4) for p in prod],
                   "product_cv": round(cv, 4), "spearman_D_thetastar": round(rho, 4),
                   "span_grid_steps": round(span, 2), "verdict": verdict}
    out = HERE / ("scaling_clean_smoke.json" if SMOKE else "scaling_clean_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n===== " + verdict)
    print("wrote " + out.name)


if __name__ == "__main__":
    main()

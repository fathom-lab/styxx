# -*- coding: utf-8 -*-
"""
run_frequency_sweep.py — frozen by PREREG_frequency_sweep_2026_06_04.

Sweeps oscillation frequency theta (rad/step) in a complex-diagonal LRU on the ordered-copy
task. theta is FROZEN (non-learnable, all modes share one value) and swept across [0, pi].
Reference arm FREE = theta learnable (per-mode spread). Tests monotonic (operator) vs resonant
(mechanism) vs flat. Identical rig/config to papers/rhythm-rescue/run_rhythm_rescue.py except
theta handling. Smoke mode (--smoke) shrinks steps/seeds/grid to validate plumbing only.
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

# ---- frozen config (matches rhythm-rescue) ----
V = 12
KMAX = 20
D = 256
D_IN = 64
STEPS = 4000
BATCH = 64
LR = 2e-3
SEEDS = [0, 1, 2]
KGRID = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 20]
ACC_THR = 0.80
THETA_FRACS = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.6875, 0.875, 0.97]  # x pi

if SMOKE:
    STEPS, SEEDS, KGRID = 400, [0], [1, 2, 3, 4, 6, 8]
    THETA_FRACS = [0.0, 0.25, 0.5]


class CLRU(nn.Module):
    """theta_spec: 'free' -> learnable per-mode theta; float -> frozen shared theta (rad)."""
    def __init__(self, d, d_in, theta_spec):
        super().__init__()
        r = torch.empty(d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        th = torch.empty(d).uniform_(0.0, math.pi / 2)   # consume RNG identically across arms
        if theta_spec == "free":
            self.theta = nn.Parameter(th)
        else:
            self.register_buffer("theta", torch.full((d,), float(theta_spec)))
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
    def __init__(self, theta_spec):
        super().__init__()
        self.emb = nn.Embedding(V + 1, D_IN)
        self.lru = CLRU(D, D_IN, theta_spec)
        self.read = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, V))

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


def make_batch(B, K):
    syms = torch.randint(0, V, (B, K), device=DEV)
    inp = torch.cat([syms, torch.full((B, K), V, device=DEV)], 1)
    tgt = torch.full((B, 2 * K), -100, device=DEV)
    tgt[:, K:] = syms
    return inp, tgt


def train_model(theta_spec, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(theta_spec).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(STEPS):
        K = int(np.random.randint(1, KMAX + 1))
        inp, tgt = make_batch(BATCH, K)
        loss = lossf(m(inp).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    osc_use = float(m.lru.theta.detach().sin().abs().mean().item())
    return m, osc_use


@torch.no_grad()
def eval_K(m, K, n=1024):
    inp, tgt = make_batch(n, K)
    pred = m(inp).argmax(-1); mask = tgt != -100
    return (pred[mask] == tgt[mask]).float().mean().item()


def kcap(accs):
    cap = 0
    for K in KGRID:
        if accs[K] >= ACC_THR:
            cap = K
    return cap


def run_condition(label, theta_spec):
    per_seed = {}
    for seed in SEEDS:
        m, osc = train_model(theta_spec, seed)
        accs = {K: round(eval_K(m, K), 4) for K in KGRID}
        per_seed[str(seed)] = {"acc": accs, "kcap": kcap(accs), "osc_use": round(osc, 4)}
        print(f"{label:14s} seed{seed}: kcap={kcap(accs):2d} osc_use={osc:.3f}", flush=True)
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()
    mkcap = float(np.mean([per_seed[str(s)]["kcap"] for s in SEEDS]))
    macc = {str(K): round(float(np.mean([per_seed[str(s)]["acc"][K] for s in SEEDS])), 4) for K in KGRID}
    return {"per_seed": per_seed, "mean_kcap": mkcap, "mean_acc": macc}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def main():
    res = {"config": {"D": D, "V": V, "steps": STEPS, "seeds": SEEDS, "kgrid": KGRID,
                       "acc_thr": ACC_THR, "theta_fracs": THETA_FRACS, "smoke": SMOKE},
           "sweep": {}, "free": None}

    for frac in THETA_FRACS:
        theta = frac * math.pi
        res["sweep"][f"{frac:.4f}"] = run_condition(f"theta={frac:.3f}pi", theta)
    res["free"] = run_condition("FREE", "free")

    # ---- frozen decision rule ----
    fracs = THETA_FRACS
    kcaps = [res["sweep"][f"{f:.4f}"]["mean_kcap"] for f in fracs]
    rho = spearman(np.array(fracs), np.array(kcaps))
    imax = int(np.argmax(kcaps))
    theta_star = fracs[imax]
    peak = kcaps[imax]
    boundary = kcaps[-1]
    interior = 0 < imax < len(fracs) - 1
    free_kcap = res["free"]["mean_kcap"]
    best_single = max(kcaps)
    nonzero = kcaps[1:]
    flat = (max(nonzero) - min(nonzero) <= 1) and (min(nonzero) >= kcaps[0] + 2)

    if rho >= 0.90 and imax == len(fracs) - 1:
        verdict = (f"MONOTONIC — capacity rises with frequency to the Nyquist boundary "
                   f"(rho={rho:.2f}, argmax theta={theta_star:.3f}pi). Operator's literal "
                   f"'higher frequency = greater capacity' SUPPORTED in-silico.")
    elif interior and (peak - boundary) >= 2:
        verdict = (f"RESONANT — capacity peaks at an interior optimum theta*={theta_star:.3f}pi "
                   f"(kcap {peak:.1f}) and falls {peak - boundary:.1f} items by Nyquist. "
                   f"Higher-is-better is FALSE beyond the optimum; there is a best band.")
    elif flat:
        verdict = ("FLAT — any nonzero frequency suffices; capacity is insensitive to which "
                   "frequency above zero. Presence matters, value does not.")
    else:
        verdict = (f"MIXED — shape not clean (rho={rho:.2f}, argmax {theta_star:.3f}pi, "
                   f"peak {peak:.1f}, boundary {boundary:.1f}). Report shape, claim nothing.")

    spectrum = "PASS" if free_kcap >= best_single + 1 else "FAIL"
    res["gate"] = {
        "kcap_by_theta": {f"{f:.4f}": k for f, k in zip(fracs, kcaps)},
        "spearman_theta_kcap": round(rho, 4),
        "theta_star_over_pi": theta_star, "peak_kcap": peak, "boundary_kcap": boundary,
        "interior_peak": interior, "free_kcap": free_kcap, "best_single_kcap": best_single,
        "spectrum_gate": spectrum,
        "spectrum_note": f"FREE {free_kcap:.1f} vs best single-theta {best_single:.1f} "
                         f"(need FREE >= best+1 for spectrum>single-tone)",
        "verdict": verdict,
    }
    out = HERE / ("frequency_sweep_smoke.json" if SMOKE else "frequency_sweep_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n===== " + verdict)
    print("===== spectrum gate: " + spectrum + " | " + res["gate"]["spectrum_note"])
    print("wrote " + out.name)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""run_noise_control.py — fairness control for the noise-robustness result.

Absolute state-noise is only a fair test if both arms carry the same state magnitude. If LRU-FREE simply
runs a larger state, a fixed sigma is relatively smaller for it and the 'robustness' is a scale artifact.
Train free + clamped at sigma=0, measure the RMS of the recurrent state ([hre,him]) that the noise is
added to. Verdict FAIR if the magnitudes match (clamped/free within +/-25%)."""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
V, KMAX, D_IN, D = 12, 20, 64, 256
STEPS = 4000
BATCH, LR = 64, 2e-3
SEEDS = [0, 1]


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
    def __init__(self, free):
        super().__init__()
        self.emb = nn.Embedding(V + 1, D_IN)
        self.lru = CLRU(D, D_IN, free)
        self.read = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, V))

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


def make_batch(B, K):
    syms = torch.randint(0, V, (B, K), device=DEV)
    inp = torch.cat([syms, torch.full((B, K), V, device=DEV)], 1)
    tgt = torch.full((B, 2 * K), -100, device=DEV); tgt[:, K:] = syms
    return inp, tgt


def train(free, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(free).to(DEV)
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
def state_rms(m, K=6, n=512):
    inp, _ = make_batch(n, K)
    h = m.lru(m.emb(inp))                      # [hre,him] concat — exactly what noise is added to
    return float(h.pow(2).mean().sqrt())


def main():
    rms = {"free": [], "clamped": []}
    for free, name in [(True, "free"), (False, "clamped")]:
        for s in SEEDS:
            m = train(free, s); rms[name].append(state_rms(m)); del m
            if DEV == "cuda":
                torch.cuda.empty_cache()
    f = float(np.mean(rms["free"])); c = float(np.mean(rms["clamped"]))
    rel = c / f if f > 0 else float("nan")
    fair = 0.75 <= rel <= 1.25
    print(f"state RMS  free={f:.4f}  clamped={c:.4f}  clamped/free={rel:.3f}")
    print(f"per-seed free={[round(x,4) for x in rms['free']]}  clamped={[round(x,4) for x in rms['clamped']]}")
    if fair:
        print(f"===== FAIR — state magnitudes match (clamped/free {rel:.2f} within +/-25%); the absolute-noise "
              "robustness result is NOT a scale artifact. Phase code is genuinely more noise-tolerant.")
    elif rel < 0.75:
        print(f"===== CONFOUNDED — clamped state is SMALLER ({rel:.2f}x free); fixed sigma hits it relatively "
              "harder. The robustness result is at least partly a scale artifact; discount it / re-run relative.")
    else:
        print(f"===== REVERSE-CONFOUND — clamped state is LARGER ({rel:.2f}x free); the robustness result is "
              "if anything UNDER-stated. Phase code more noise-tolerant despite a smaller signal.")
    import json
    (HERE / "noise_control_result.json").write_text(
        json.dumps({"free_rms": f, "clamped_rms": c, "clamped_over_free": rel, "fair": bool(fair),
                    "per_seed": rms}, indent=2), encoding="utf-8")
    print("wrote noise_control_result.json")


if __name__ == "__main__":
    main()

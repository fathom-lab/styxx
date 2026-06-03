# -*- coding: utf-8 -*-
"""
run_rhythm_rescue.py — phase-clamp ablation with rescue (frozen by PREREG_rhythm_rescue).

Two complex-diagonal linear-recurrence (LRU) networks, IDENTICAL in every weight init
except the eigenvalue phase theta: FREE (theta learnable -> eigenvalues can rotate ->
oscillate) vs CLAMPED (theta == 0 -> real eigenvalues -> pure decay, cannot oscillate).
Complex input projection in BOTH arms, so both have the same 2d real state and the same
parameters; the ONLY difference is rotation. Task = ordered copy (capacity-limited ordered
memory, the function theta-gamma is most credited with). Does the no-rhythm net rescue it?
"""
from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"

V = 12            # symbol vocab
KMAX = 20
D = 256           # complex modes (-> 2D=512 real state)
D_IN = 64
STEPS = 4000
BATCH = 64
LR = 2e-3
SEEDS = [0, 1, 2]
KGRID = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 20]
ACC_THR = 0.80


class CLRU(nn.Module):
    def __init__(self, d, d_in, free):
        super().__init__()
        r = torch.empty(d).uniform_(0.9, 0.999)             # |lambda| init
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        th = torch.empty(d).uniform_(0.0, math.pi / 2)      # consume RNG identically in both arms
        if free:
            self.theta = nn.Parameter(th)
        else:
            self.register_buffer("theta", torch.zeros(d))
        self.B_re = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        self.B_im = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))

    def forward(self, x):                                   # x (B,T,d_in)
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
        return torch.stack(outs, 1)                         # (B,T,2d)


class Model(nn.Module):
    def __init__(self, free):
        super().__init__()
        self.emb = nn.Embedding(V + 1, D_IN)               # symbols 0..V-1, GO=V
        self.lru = CLRU(D, D_IN, free)
        self.read = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, V))

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


def make_batch(B, K):
    syms = torch.randint(0, V, (B, K), device=DEV)
    inp = torch.cat([syms, torch.full((B, K), V, device=DEV)], 1)      # K symbols, then K GO
    tgt = torch.full((B, 2 * K), -100, device=DEV)
    tgt[:, K:] = syms                                                  # recall in order
    return inp, tgt


def train_model(free, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = Model(free).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for step in range(STEPS):
        K = int(np.random.randint(1, KMAX + 1))
        inp, tgt = make_batch(BATCH, K)
        loss = lossf(m(inp).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    osc_use = float((m.lru.theta.detach().sin().abs().mean()).item()) if free else 0.0
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


def main():
    res = {"config": {"D": D, "V": V, "steps": STEPS, "seeds": SEEDS, "kgrid": KGRID,
                      "acc_thr": ACC_THR}, "free": {}, "clamped": {}}
    for arm, free in [("free", True), ("clamped", False)]:
        for seed in SEEDS:
            m, osc = train_model(free, seed)
            accs = {K: round(eval_K(m, K), 4) for K in KGRID}
            res[arm][str(seed)] = {"acc": accs, "kcap": kcap(accs), "osc_use": round(osc, 4)}
            print(f"{arm:8s} seed{seed}: kcap={kcap(accs):2d} osc_use={osc:.3f} acc={accs}", flush=True)
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()

    def macc(arm, K): return float(np.mean([res[arm][str(s)]["acc"][K] for s in SEEDS]))
    def mkcap(arm): return float(np.mean([res[arm][str(s)]["kcap"] for s in SEEDS]))
    kc_f, kc_c = mkcap("free"), mkcap("clamped")
    gap = kc_f - kc_c
    acc_ok = all(macc("clamped", K) >= macc("free", K) - 0.05 for K in KGRID if K <= kc_f)
    if kc_c >= kc_f - 2 and acc_ok:
        reading = (f"RESCUE — oscillation NOT necessary: real-eigenvalue dynamics match FREE "
                   f"(kcap free {kc_f:.1f} vs clamped {kc_c:.1f}). Rhythm is a substrate-specific MECHANISM.")
    elif gap >= 6:
        reading = (f"NECESSARY — oscillation extends ordered-memory capacity by {gap:.1f} items that "
                   f"real dynamics cannot match. Rhythm is a FUNCTION here.")
    else:
        reading = f"ADVANTAGE — oscillation helps (capacity gap {gap:.1f} items) but is not strictly necessary."
    res["gate"] = {"kcap_free": kc_f, "kcap_clamped": kc_c, "gap": gap,
                   "osc_use_free": round(float(np.mean([res['free'][str(s)]['osc_use'] for s in SEEDS])), 4),
                   "mean_acc_free": {str(K): round(macc("free", K), 4) for K in KGRID},
                   "mean_acc_clamped": {str(K): round(macc("clamped", K), 4) for K in KGRID},
                   "reading": reading}
    (HERE / "rhythm_rescue_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n===== " + reading)
    print("wrote rhythm_rescue_result.json")


if __name__ == "__main__":
    main()

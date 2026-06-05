# -*- coding: utf-8 -*-
"""
run_extrapolation.py — frozen by PREREG_extrapolation_2026_06_04.

Rhythm's real niche? Train LRU-FREE / LRU-CLAMPED / TRANSFORMER on K<=8 ordered copy, test on longer
K (extrapolation). Does recurrence hold where attention collapses, and does oscillation extrapolate
better than decay? Matched params; transformer uses sinusoidal (extrapolation-defined) positions.
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
V, D, D_IN = 12, 256, 64
K_TRAIN = 8
IN_DIST = [1, 2, 3, 4, 5, 6, 7, 8]
EXTRAP = [10, 12, 15, 18, 20]
STEPS = 700 if SMOKE else 5000
BATCH, LR = 64, 2e-3
SEEDS = [0] if SMOKE else [0, 1, 2]
ACC_THR = 0.80
TD, TLAYERS, THEADS, TFF = 80, 3, 4, 160


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


class LRUModel(nn.Module):
    def __init__(self, free):
        super().__init__()
        self.emb = nn.Embedding(V + 1, D_IN)
        self.lru = CLRU(D, D_IN, free)
        self.read = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, V))

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


def sinusoidal(T, d, device):
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    div = torch.exp(-math.log(10000.0) * torch.arange(0, d, 2, device=device).float() / d)
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V + 1, TD)
        layer = nn.TransformerEncoderLayer(TD, THEADS, TFF, dropout=0.0, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, TLAYERS)
        self.read = nn.Linear(TD, V)

    def forward(self, tok):
        T = tok.shape[1]
        x = self.emb(tok) + sinusoidal(T, TD, tok.device)        # fixed sinusoidal -> defined for all T
        mask = torch.triu(torch.ones(T, T, device=tok.device, dtype=torch.bool), 1)
        return self.read(self.tr(x, mask=mask))


def build(a):
    return {"lru_free": lambda: LRUModel(True), "lru_clamped": lambda: LRUModel(False),
            "transformer": TransformerModel}[a]()


def make_batch(B, K):
    syms = torch.randint(0, V, (B, K), device=DEV)
    inp = torch.cat([syms, torch.full((B, K), V, device=DEV)], 1)
    tgt = torch.full((B, 2 * K), -100, device=DEV); tgt[:, K:] = syms
    return inp, tgt


def train(a, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(a).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for _ in range(STEPS):
        K = int(np.random.randint(1, K_TRAIN + 1))           # TRAIN on K<=8 only
        inp, tgt = make_batch(BATCH, K)
        loss = lossf(m(inp).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


@torch.no_grad()
def acc(m, K, n=1024):
    inp, tgt = make_batch(n, K); pred = m(inp).argmax(-1); mask = tgt != -100
    return float((pred[mask] == tgt[mask]).float().mean().item())


def main():
    archs = ["lru_free", "lru_clamped", "transformer"]
    res = {"config": {"steps": STEPS, "seeds": SEEDS, "k_train": K_TRAIN, "extrap": EXTRAP}, "arms": {}}
    for a in archs:
        ind, ext = [], []
        for s in SEEDS:
            m = train(a, s)
            ai = {K: round(acc(m, K), 4) for K in IN_DIST}
            ae = {K: round(acc(m, K), 4) for K in EXTRAP}
            ind.append(float(np.mean(list(ai.values())))); ext.append(float(np.mean(list(ae.values()))))
            print(f"  {a:12s} seed{s}: in-dist {ind[-1]:.3f} | extrap {ext[-1]:.3f}  (extrap-per-K {ae})", flush=True)
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()
        res["arms"][a] = {"in_dist_acc": float(np.mean(ind)), "extrap_acc": float(np.mean(ext))}

    tf = res["arms"]["transformer"]["extrap_acc"]
    lf = res["arms"]["lru_free"]["extrap_acc"]
    lc = res["arms"]["lru_clamped"]["extrap_acc"]
    rec = (lf + lc) / 2
    p1 = (rec - tf) >= 0.15 and tf <= 0.30
    p2 = (lf - lc) >= 0.05
    if p1 and p2:
        verdict = (f"RHYTHM'S NICHE IS GENERALIZATION + OSCILLATION AIDS STRUCTURE — past training length, "
                   f"recurrence holds (free {lf:.2f}, clamped {lc:.2f}) while attention COLLAPSES ({tf:.2f}); and "
                   f"oscillation extrapolates better than decay ({lf:.2f} vs {lc:.2f}). Rhythm = structured "
                   "generalization, not raw storage. The deeper, honest answer.")
    elif p1:
        verdict = (f"RHYTHM'S NICHE IS GENERALIZATION — recurrence extrapolates (free {lf:.2f}, clamped {lc:.2f}) "
                   f"where attention collapses ({tf:.2f}). But oscillation does NOT beat decay at it "
                   f"({lf:.2f} vs {lc:.2f}) — the extrapolation edge is RECURRENCE, not oscillation per se.")
    else:
        verdict = (f"NO CLEAN REGIME FLIP — extrap acc: transformer {tf:.2f}, lru-free {lf:.2f}, lru-clamped {lc:.2f}. "
                   "Attention did not collapse as predicted / recurrence did not clearly win. Report curves.")
    res["gate"] = {"extrap_transformer": tf, "extrap_lru_free": lf, "extrap_lru_clamped": lc,
                   "P1_recurrence_extrapolates": bool(p1), "P2_oscillation_aids": bool(p2), "verdict": verdict}
    out = HERE / ("extrapolation_smoke.json" if SMOKE else "extrapolation_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  in-dist acc: " + " | ".join(f"{a} {res['arms'][a]['in_dist_acc']:.2f}" for a in archs))
    print(f"  EXTRAP  acc: " + " | ".join(f"{a} {res['arms'][a]['extrap_acc']:.2f}" for a in archs))
    print("\n===== " + verdict)
    print("wrote", out.name)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
run_necessity_attention.py — frozen by PREREG_necessity_attention_2026_06_04.

Does rhythm buy anything attention can't? Matched-parameter showdown on ordered copy:
LRU-FREE (oscillatory) vs LRU-CLAMPED (decay-only) vs TRANSFORMER (attention, no rhythm).
If the rhythm-free transformer matches/beats the oscillatory LRU at equal params, oscillation is
one efficient mechanism, not a requirement.
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
V, KMAX, D, D_IN = 12, 20, 256, 64
STEPS = 600 if SMOKE else 4000
BATCH, LR = 64, 2e-3
SEEDS = [0] if SMOKE else [0, 1, 2]
KGRID = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 20]
ACC_THR = 0.80
# transformer sized to match the LRU param count (~168k); verified + reported at runtime
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


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V + 1, TD)
        self.pos = nn.Parameter(torch.randn(2 * KMAX + 2, TD) * 0.02)
        layer = nn.TransformerEncoderLayer(TD, THEADS, TFF, dropout=0.0, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, TLAYERS)
        self.read = nn.Linear(TD, V)

    def forward(self, tok):
        T = tok.shape[1]
        x = self.emb(tok) + self.pos[:T]
        mask = torch.triu(torch.ones(T, T, device=tok.device, dtype=torch.bool), 1)   # causal
        return self.read(self.tr(x, mask=mask))


def build(arch):
    return {"lru_free": lambda: LRUModel(True), "lru_clamped": lambda: LRUModel(False),
            "transformer": TransformerModel}[arch]()


def make_batch(B, K):
    syms = torch.randint(0, V, (B, K), device=DEV)
    inp = torch.cat([syms, torch.full((B, K), V, device=DEV)], 1)
    tgt = torch.full((B, 2 * K), -100, device=DEV); tgt[:, K:] = syms
    return inp, tgt


def train(arch, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(arch).to(DEV)
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


def kcap(accs):
    cap = 0
    for K in KGRID:
        if accs[K] >= ACC_THR:
            cap = K
    return cap


def nparams(arch):
    return sum(p.numel() for p in build(arch).parameters())


def main():
    archs = ["lru_free", "lru_clamped", "transformer"]
    pcount = {a: nparams(a) for a in archs}
    print("param counts:", {a: f"{n/1000:.0f}k" for a, n in pcount.items()}, flush=True)
    res = {"config": {"steps": STEPS, "seeds": SEEDS, "params": pcount}, "arms": {}}
    for a in archs:
        kcaps = []
        for s in SEEDS:
            m = train(a, s)
            accs = {K: round(eval_K(m, K), 4) for K in KGRID}
            kcaps.append(kcap(accs))
            print(f"  {a:12s} seed{s}: kcap={kcap(accs):2d}", flush=True)
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()
        res["arms"][a] = {"kcaps": kcaps, "mean_kcap": float(np.mean(kcaps))}

    kf = res["arms"]["lru_free"]["mean_kcap"]
    kc = res["arms"]["lru_clamped"]["mean_kcap"]
    kt = res["arms"]["transformer"]["mean_kcap"]
    p1 = (kf - kc) >= 2.0                                  # oscillation helps recurrence
    p2 = kt >= kf - 0.34                                   # transformer matches/beats free (within ~noise)
    if p1 and p2:
        verdict = (f"RHYTHM NOT SPECIAL — oscillation helps a recurrent net (free {kf:.1f} vs clamped {kc:.1f}) "
                   f"but the rhythm-free TRANSFORMER matches/beats it at matched params ({kt:.1f} vs {kf:.1f}). "
                   "Oscillation is one efficient mechanism, not a requirement — the demarcated, honest reading.")
    elif p1 and (kf - kt) >= 2.0:
        verdict = (f"OSCILLATION HAS A REAL EDGE — the oscillatory LRU ({kf:.1f}) beats the rhythm-free transformer "
                   f"({kt:.1f}) by >=2 items at matched params. Rhythm buys something attention can't here — novel.")
    else:
        verdict = f"MIXED — free {kf:.1f}, clamped {kc:.1f}, transformer {kt:.1f}. Report the three."
    res["gate"] = {"kcap_free": kf, "kcap_clamped": kc, "kcap_transformer": kt,
                   "P1_oscillation_helps_recurrence": bool(p1), "P2_attention_matches_oscillation": bool(p2),
                   "verdict": verdict}
    out = HERE / ("necessity_attention_smoke.json" if SMOKE else "necessity_attention_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  kcap: LRU-free {kf:.2f} | LRU-clamped {kc:.2f} | TRANSFORMER {kt:.2f}")
    print("\n===== " + verdict)
    print("wrote", out.name)


if __name__ == "__main__":
    main()

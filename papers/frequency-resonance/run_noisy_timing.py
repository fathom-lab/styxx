# -*- coding: utf-8 -*-
"""run_noisy_timing.py — frozen by PREREG_noisy_timing_2026_06_05.

The ceiling-breaker. Timing (exp 8) was a tie at ceiling: rhythm and attention both perfect. Break the
ceiling with INPUT CORRUPTION (fair to all arms): replace each input symbol with a random one w.p. rho,
predict the CLEAN periodic continuation. Does oscillation's distributed phase integration track a corrupted
periodic stream better than attention's position-lookup (which breaks when the looked-up symbol is the
corrupted one)? P1: acc_free - acc_transformer >= 0.05 at rho>=0.3 -> rhythm BEATS attention, ceiling broken.
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
V, D, D_IN, L = 12, 256, 64, 48
STEPS = 600 if SMOKE else 4000
BATCH, LR = 64, 2e-3
SEEDS = [0] if SMOKE else [0, 1, 2]
PGRID_TRAIN = [4, 6] if SMOKE else [4, 5, 6, 8, 10]
PGRID_EVAL = [4, 6] if SMOKE else [4, 6, 8]
RHO_SWEEP = [0.0, 0.3] if SMOKE else [0.0, 0.1, 0.2, 0.3, 0.4]
RHO_TRAIN_MAX = 0.4
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
        self.emb = nn.Embedding(V, D_IN)
        self.lru = CLRU(D, D_IN, free)
        self.read = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, V))

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, TD)
        self.pos = nn.Parameter(torch.randn(L, TD) * 0.02)
        layer = nn.TransformerEncoderLayer(TD, THEADS, TFF, dropout=0.0, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, TLAYERS)
        self.read = nn.Linear(TD, V)

    def forward(self, tok):
        T = tok.shape[1]
        x = self.emb(tok) + self.pos[:T]
        mask = torch.triu(torch.ones(T, T, device=tok.device, dtype=torch.bool), 1)
        return self.read(self.tr(x, mask=mask))


def build(arch):
    return {"lru_free": lambda: LRUModel(True), "lru_clamped": lambda: LRUModel(False),
            "transformer": TransformerModel}[arch]()


def make_batch(B, P, rho):
    motif = torch.randint(0, V, (B, P), device=DEV)
    idx = torch.arange(L, device=DEV) % P
    x = motif[:, idx]                                  # clean periodic stream (B, L)
    x_in = x.clone()
    if rho > 0:                                        # corrupt input symbols (fair to all arms)
        m = torch.rand(B, L, device=DEV) < rho
        x_in = torch.where(m, torch.randint(0, V, (B, L), device=DEV), x)
    inp = x_in[:, :-1]                                 # NOISY input
    tgt = x[:, 1:].clone()                             # CLEAN target
    j = torch.arange(L - 1, device=DEV)
    tgt[:, (j + 1) < (2 * P)] = -100                   # score after two full periods
    return inp, tgt


def train(arch, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(arch).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for _ in range(STEPS):
        P = int(PGRID_TRAIN[np.random.randint(len(PGRID_TRAIN))])
        rho = float(np.random.uniform(0, RHO_TRAIN_MAX))
        inp, tgt = make_batch(BATCH, P, rho)
        loss = lossf(m(inp).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


@torch.no_grad()
def eval_rho(m, rho, n=2048):
    accs = []
    for P in PGRID_EVAL:                                # average over eval periods
        inp, tgt = make_batch(n, P, rho); pred = m(inp).argmax(-1); mask = tgt != -100
        accs.append((pred[mask] == tgt[mask]).float().mean().item())
    return float(np.mean(accs))


def nparams(arch):
    return sum(p.numel() for p in build(arch).parameters())


def main():
    archs = ["lru_free", "lru_clamped", "transformer"]
    pcount = {a: nparams(a) for a in archs}
    print("param counts:", {a: f"{n/1000:.0f}k" for a, n in pcount.items()}, flush=True)
    res = {"config": {"steps": STEPS, "seeds": SEEDS, "rho_sweep": RHO_SWEEP, "pgrid_eval": PGRID_EVAL,
                      "params": pcount}, "by_arch": {}}
    for a in archs:
        accR = {r: [] for r in RHO_SWEEP}
        for s in SEEDS:
            m = train(a, s)
            for r in RHO_SWEEP:
                accR[r].append(eval_rho(m, r))
            print(f"  {a:12s} seed{s}: " + " ".join(f"r{r}={accR[r][-1]:.2f}" for r in RHO_SWEEP), flush=True)
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()
        res["by_arch"][a] = {str(r): float(np.mean(accR[r])) for r in RHO_SWEEP}

    free = res["by_arch"]["lru_free"]; clamp = res["by_arch"]["lru_clamped"]; trans = res["by_arch"]["transformer"]
    hi = [r for r in RHO_SWEEP if r >= 0.3]
    edge_hi = float(np.mean([free[str(r)] - trans[str(r)] for r in hi]))       # P1: rhythm vs attention at high corruption
    free_vs_clamp_hi = float(np.mean([free[str(r)] - clamp[str(r)] for r in hi]))
    p1 = edge_hi >= 0.05
    p2_attn_ties_or_wins = edge_hi <= 0.05  # i.e. NOT a clean rhythm win (incl. negative)
    if p1:
        verdict = (f"RHYTHM BEATS ATTENTION — CEILING BROKEN. Under input corruption, oscillation tracks the periodic "
                   f"stream better than attention (free-transformer +{edge_hi:.3f} at rho>=0.3). The distributed phase "
                   "bank denoises temporal structure where position-lookup breaks: the first regime rhythm is SUPERIOR, "
                   "not merely co-equal. The strongest landing of the arc.")
    else:
        verdict = (f"ATTENTION TIES-OR-WINS — demarcation tightened. Under corruption rhythm does NOT strictly beat "
                   f"attention (free-transformer {edge_hi:+.3f} at rho>=0.3). Rhythm's only clean win stays its decay "
                   "baseline; vs attention it is at best co-equal (clean timing) and not superior under noise. The map "
                   "sharpens: rhythm ties attention at temporal tasks, never beats it.")
    res["gate"] = {"free_minus_transformer_hi_rho": round(edge_hi, 4), "free_minus_clamped_hi_rho": round(free_vs_clamp_hi, 4),
                   "P1_rhythm_beats_attention": bool(p1), "P2_attention_ties_or_wins": bool(p2_attn_ties_or_wins),
                   "verdict": verdict}
    out = HERE / ("noisy_timing_smoke.json" if SMOKE else "noisy_timing_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  acc(rho) free   :", {r: round(free[str(r)], 3) for r in RHO_SWEEP})
    print("  acc(rho) transf :", {r: round(trans[str(r)], 3) for r in RHO_SWEEP})
    print("  acc(rho) clamped:", {r: round(clamp[str(r)], 3) for r in RHO_SWEEP})
    print("\n===== " + verdict)
    print("wrote", out.name)


if __name__ == "__main__":
    main()

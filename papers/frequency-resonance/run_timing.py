# -*- coding: utf-8 -*-
"""
run_timing.py — frozen by PREREG_timing_2026_06_04.

Oscillation's HOME TURF. Every prior experiment used ordered copy (a memory task). This tests rhythm on
what it is mechanistically built for: predicting PERIODIC temporal structure. A complex eigenvalue at
theta=2pi/P is a clock that resonates with a period-P signal. Task: read a periodically-repeating symbol
stream, predict the next symbol; score only positions after two full periods (the period is inferable).
Three arms at matched params: LRU-CLAMPED (decay), LRU-FREE (oscillatory), TRANSFORMER (attention, LEARNED
positions = genuinely rhythm-free). Sweep period P. P1: free's edge over clamped GROWS with P. P2: does
rhythm finally beat attention here, on its native task?
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
PGRID = [3, 8] if SMOKE else [2, 3, 4, 5, 6, 8, 10, 12]
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
        self.pos = nn.Parameter(torch.randn(L, TD) * 0.02)   # learned positions = rhythm-free
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


def make_batch(B, P):
    motif = torch.randint(0, V, (B, P), device=DEV)
    idx = torch.arange(L, device=DEV) % P
    x = motif[:, idx]                       # (B, L) periodic stream
    inp = x[:, :-1]
    tgt = x[:, 1:].clone()                  # predict next symbol
    j = torch.arange(L - 1, device=DEV)
    scored = (j + 1) >= (2 * P)             # score only after two full periods are visible
    tgt[:, ~scored] = -100
    return inp, tgt


def train(arch, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(arch).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for _ in range(STEPS):
        P = int(PGRID[np.random.randint(len(PGRID))])
        inp, tgt = make_batch(BATCH, P)
        loss = lossf(m(inp).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


@torch.no_grad()
def eval_P(m, P, n=2048):
    inp, tgt = make_batch(n, P); pred = m(inp).argmax(-1); mask = tgt != -100
    return (pred[mask] == tgt[mask]).float().mean().item()


def nparams(arch):
    return sum(p.numel() for p in build(arch).parameters())


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = math.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/den) if den else 0.0


def main():
    archs = ["lru_free", "lru_clamped", "transformer"]
    pcount = {a: nparams(a) for a in archs}
    print("param counts:", {a: f"{n/1000:.0f}k" for a, n in pcount.items()}, flush=True)
    res = {"config": {"steps": STEPS, "seeds": SEEDS, "pgrid": PGRID, "L": L, "params": pcount}, "by_arch": {}}
    theta_dump = []   # mechanistic probe: does the free net learn phases that resonate with trained periods?
    for a in archs:
        accP = {P: [] for P in PGRID}
        for s in SEEDS:
            m = train(a, s)
            for P in PGRID:
                accP[P].append(eval_P(m, P))
            if a == "lru_free":
                theta_dump.append(sorted(float(t) for t in m.lru.theta.detach().cpu().tolist()))
            print(f"  {a:12s} seed{s}: " + " ".join(f"P{P}={np.mean(accP[P][-1:]):.2f}" for P in PGRID), flush=True)
            del m
            if DEV == "cuda":
                torch.cuda.empty_cache()
        res["by_arch"][a] = {str(P): float(np.mean(accP[P])) for P in PGRID}

    # resonance check: trained periods map to phases 2pi/P; count free-net modes within +/-10% of each
    target_theta = {P: 2 * math.pi / P for P in PGRID}
    res["resonance_probe"] = {"target_theta_per_period": {str(P): round(t, 4) for P, t in target_theta.items()},
                              "free_theta_sorted_byseed": [[round(t, 4) for t in row] for row in theta_dump]}

    free = res["by_arch"]["lru_free"]; clamp = res["by_arch"]["lru_clamped"]; trans = res["by_arch"]["transformer"]
    diff = [free[str(P)] - clamp[str(P)] for P in PGRID]                 # free advantage over clamped vs P
    rho = spearman(np.array(PGRID, float), np.array(diff))
    big = [P for P in PGRID if P >= 8]
    edge_big = float(np.mean([free[str(P)] - clamp[str(P)] for P in big]))      # P1 magnitude at long periods
    free_vs_trans_big = float(np.mean([free[str(P)] - trans[str(P)] for P in big]))  # P2

    p1 = (rho >= 0.4) and (edge_big >= 0.15)
    p2_free_not_dominated = free_vs_trans_big >= -0.05
    if p1 and p2_free_not_dominated:
        verdict = (f"RHYTHM'S NICHE IS TIMING — and it is SOVEREIGN here. Oscillation's edge over decay GROWS with "
                   f"period (Spearman {rho:.2f}, +{edge_big:.2f} at P>=8) AND it is not dominated by attention "
                   f"(free-transformer {free_vs_trans_big:+.2f} at P>=8). Rhythm is dominated for memory capacity "
                   "but the native mechanism for periodic temporal structure -- the honest positive reframe.")
    elif p1:
        verdict = (f"RHYTHM'S NICHE IS TIMING (but attention still matches it) — oscillation's edge over decay GROWS "
                   f"with period (Spearman {rho:.2f}, +{edge_big:.2f} at P>=8), confirming timing is its native "
                   f"domain, BUT attention still keeps pace (free-transformer {free_vs_trans_big:+.2f} at P>=8). "
                   "Rhythm is sufficient and natural for timing, not uniquely necessary.")
    else:
        verdict = (f"RHYTHM NOT SPECIAL EVEN AT TIMING — oscillation's edge over decay does NOT clearly grow with "
                   f"period (Spearman {rho:.2f}, +{edge_big:.2f} at P>=8). Even on periodic prediction, the task it "
                   "is mechanistically built for, explicit oscillation is not the decisive mechanism. Report acc(P).")
    res["gate"] = {"spearman_P_freeEdge": round(rho, 4), "free_edge_at_Pge8": round(edge_big, 4),
                   "free_minus_transformer_at_Pge8": round(free_vs_trans_big, 4),
                   "P1_timing_is_oscillation_domain": bool(p1),
                   "P2_free_not_dominated_by_attention": bool(p2_free_not_dominated), "verdict": verdict}
    out = HERE / ("timing_smoke.json" if SMOKE else "timing_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  acc(P) free   :", {P: round(free[str(P)], 2) for P in PGRID})
    print("  acc(P) clamped:", {P: round(clamp[str(P)], 2) for P in PGRID})
    print("  acc(P) transf :", {P: round(trans[str(P)], 2) for P in PGRID})
    print("\n===== " + verdict)
    print("wrote", out.name)


if __name__ == "__main__":
    main()

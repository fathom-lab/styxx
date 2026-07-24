# -*- coding: utf-8 -*-
"""
run_scarcity_scale.py -- frozen by PREREG_scarcity_scale_2026_07_23.

DEEPEN THE ONE CLEAN WIN. The adaptive-frequency arc found a clean win only under mode-scarcity (D=4:
windowed-conv detector beat static by +0.129 and beat the oracle). This tests whether that scarcity win
HOLDS and GROWS on a HARDER, LONGER drifting task, and maps the full scarcity curve (advantage vs mode
budget D). Best detector only (the windowed conv = the arc's peak). Primary budget = the STARVED D=4.

Harder task vs run_entrain_rich: L=192 (was 96), 4 segments / 3 change-points, periods [3,20] (was
[3,12]). Arms: STATIC, RICH (windowed conv detector), ORACLE (diverse bank locked to true period).
Single knob kappa=0 -> STATIC bit-for-bit. Parallel scan, red-team-verified.
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

V, D_IN, L = 12, 64, 192
SEG_LEN = 48                            # S = 4 segments, 3 change-points
PMIN, PMAX = 3, 20                      # wider (harder) period range
KAPPA_MAX = 0.1
KW = 7
BATCH, LR = 64, 2e-3
STEPS = 400 if SMOKE else 1500
SEEDS = [0] if SMOKE else [0, 1, 2]
D_SWEEP = [4] if SMOKE else [2, 4, 6, 8, 12]
PRIMARY_D = 4                           # the starved regime where the clean win lives
EVAL_N = 512 if SMOKE else 1024
S = L // SEG_LEN


def lin_scan(A, X):
    T = A.shape[1]; shift = 1
    while shift < T:
        Ap = torch.cat([torch.ones_like(A[:, :shift]), A[:, :T - shift]], dim=1)
        Xp = torch.cat([torch.zeros_like(X[:, :shift]), X[:, :T - shift]], dim=1)
        X = X + A * Xp; A = A * Ap; shift <<= 1
    return X


def seq_scan(A, X):
    T = A.shape[1]; h = torch.zeros_like(X[:, 0]); out = []
    for t in range(T):
        h = A[:, t] * h + X[:, t]; out.append(h)
    return torch.stack(out, 1)


class RichCLRU(nn.Module):
    def __init__(self, d, d_in, mode):
        super().__init__()
        assert mode in ("static", "rich", "oracle")
        self.mode, self.d = mode, d
        self.kappa_override = None
        r = torch.empty(d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        self.theta0 = nn.Parameter(torch.empty(d).uniform_(0.0, math.pi / 2))
        self.B_re = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        self.B_im = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))

    def init_rich(self, d_in, d):
        self.conv = nn.Conv1d(d_in, d, KW)
        self.kappa_raw = nn.Parameter(torch.zeros(d))

    def init_oracle(self, d):
        self.register_buffer("spread", torch.exp(torch.linspace(math.log(0.5), math.log(2.0), d)))

    def forward(self, x, omega_true=None):
        B, T, _ = x.shape
        mag = torch.exp(-torch.exp(self.nu))
        gamma = torch.sqrt(torch.clamp(1 - mag ** 2, min=1e-6))
        ure = torch.einsum("bti,di->btd", x, self.B_re) * gamma
        uim = torch.einsum("bti,di->btd", x, self.B_im) * gamma
        U = torch.complex(ure, uim)
        magf = mag.view(1, 1, self.d).expand(B, T, self.d)
        if self.mode == "static":
            theta = self.theta0.view(1, 1, self.d).expand(B, T, self.d)
        elif self.mode == "oracle":
            theta = torch.clamp(self.spread.view(1, 1, self.d) * omega_true.unsqueeze(-1), 0.0, math.pi)
        else:
            xt = x.transpose(1, 2)
            c = torch.relu(self.conv(torch.cat([torch.zeros(B, D_IN, KW - 1, device=x.device), xt], 2)))
            drive = math.pi * torch.sigmoid(c).transpose(1, 2)
            kappa = (torch.full((self.d,), float(self.kappa_override), device=x.device)
                     if self.kappa_override is not None else KAPPA_MAX * torch.sigmoid(self.kappa_raw))
            A_th = torch.cat([torch.zeros(B, 1, self.d, device=x.device),
                              (1 - kappa).view(1, 1, self.d).expand(B, T - 1, self.d)], dim=1)
            X0 = ((1 - kappa) * self.theta0).view(1, 1, self.d).expand(B, 1, self.d)
            X_th = torch.cat([X0, kappa.view(1, 1, self.d) * drive[:, 1:, :]], dim=1)
            theta = lin_scan(A_th, X_th)
        lam = torch.polar(magf.contiguous(), theta.contiguous())
        Hh = lin_scan(lam, U)
        return torch.cat([Hh.real, Hh.imag], -1)


class LRUModel(nn.Module):
    def __init__(self, d, mode):
        super().__init__()
        self.emb = nn.Embedding(V, D_IN)
        self.lru = RichCLRU(d, D_IN, mode)
        self.read = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, V))
        if mode == "rich":
            self.lru.init_rich(D_IN, d)
        elif mode == "oracle":
            self.lru.init_oracle(d)

    def forward(self, tok, omega_true=None):
        return self.read(self.lru(self.emb(tok), omega_true))


def build(arch, d):
    return LRUModel(d, arch)


def make_batch(B, drift=True):
    t = torch.arange(L, device=DEV)
    if drift:
        seg_P = torch.randint(PMIN, PMAX + 1, (B, S), device=DEV)
        for _ in range(6):
            eq = seg_P[:, 1:] == seg_P[:, :-1]
            if not eq.any():
                break
            seg_P[:, 1:][eq] = torch.randint(PMIN, PMAX + 1, (int(eq.sum().item()),), device=DEV)
        seg_idx = (t // SEG_LEN).clamp(max=S - 1)
        P_t = seg_P[:, seg_idx]
        local = (t - seg_idx * SEG_LEN).unsqueeze(0).expand(B, L)
    else:
        P_seq = torch.randint(PMIN, PMAX + 1, (B, 1), device=DEV)
        P_t = P_seq.expand(B, L)
        local = t.unsqueeze(0).expand(B, L)
    motif = torch.randint(0, V, (B, PMAX), device=DEV)
    x = torch.gather(motif, 1, local % P_t)
    inp = x[:, :-1]; tgt = x[:, 1:].clone()
    omega_true = ((2 * math.pi) / P_t.float())[:, :-1]
    scored = local >= (2 * P_t)
    tgt[~scored[:, 1:]] = -100
    return inp, tgt, omega_true


def train(arch, d, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(arch, d).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for _ in range(STEPS):
        inp, tgt, om = make_batch(BATCH, drift=True)
        loss = lossf(m(inp, om).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


@torch.no_grad()
def evaluate(m, drift=True, n=EVAL_N):
    inp, tgt, om = make_batch(n, drift=drift)
    pred = m(inp, om).argmax(-1); mask = tgt != -100
    return (pred[mask] == tgt[mask]).float().mean().item()


def nparams(arch, d):
    return sum(p.numel() for p in build(arch, d).parameters())


def redteam():
    torch.manual_seed(7)
    Ac = torch.randn(4, 33, 8, dtype=torch.cfloat, device=DEV) * 0.5
    Xc = torch.randn(4, 33, 8, dtype=torch.cfloat, device=DEV)
    ds = (lin_scan(Ac, Xc) - seq_scan(Ac, Xc)).abs().max().item()
    assert ds < 1e-4, f"scan!=seq ({ds:.2e})"
    torch.manual_seed(0); st = build("static", 4).to(DEV)
    torch.manual_seed(0); rc = build("rich", 4).to(DEV)
    rc.lru.kappa_override = 0.0
    inp, tgt, om = make_batch(16, drift=True)
    with torch.no_grad():
        da = (st(inp, om) - rc(inp, om)).abs().max().item()
    assert da < 1e-5, f"kappa=0 != STATIC ({da:.2e})"
    rc.lru.kappa_override = None
    print("  [redteam] scan==seq (%.1e), kappa=0==STATIC (%.1e) -- OK" % (ds, da), flush=True)


def main():
    print(f"device={DEV} smoke={SMOKE} HARDER task L={L} S={S} periods=[{PMIN},{PMAX}] D_sweep={D_SWEEP} primary D={PRIMARY_D}",
          flush=True)
    redteam()
    arms = ["static", "rich", "oracle"]
    res = {"config": {"steps": STEPS, "seeds": SEEDS, "L": L, "seg_len": SEG_LEN, "S": S, "periods": [PMIN, PMAX],
                      "kappa_max": KAPPA_MAX, "conv_kw": KW, "d_sweep": D_SWEEP, "primary_d": PRIMARY_D},
           "params": {}, "drift": {}, "fixed": {}}
    for d in D_SWEEP:
        res["drift"][str(d)] = {}; res["fixed"][str(d)] = {}
        res["params"][str(d)] = {a: nparams(a, d) for a in arms}
        for a in arms:
            dacc, facc = [], []
            for s in SEEDS:
                m = train(a, d, s)
                dacc.append(evaluate(m, True)); facc.append(evaluate(m, False))
                del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            res["drift"][str(d)][a] = float(np.mean(dacc)); res["fixed"][str(d)][a] = float(np.mean(facc))
            print(f"  D={d:2d} {a:7s} drift={np.mean(dacc):.3f}", flush=True)

    # scarcity curve
    curve = {str(d): round(res["drift"][str(d)]["rich"] - res["drift"][str(d)]["static"], 4) for d in D_SWEEP}
    d = PRIMARY_D
    g = res["drift"][str(d)]
    adv = g["rich"] - g["static"]; orc = g["oracle"] - g["static"]
    cap = (adv / orc) if orc > 1e-9 else float("nan")
    falls = all(curve[str(D_SWEEP[i])] >= curve[str(D_SWEEP[i + 1])] - 0.02 for i in range(len(D_SWEEP) - 1))

    if orc < 0.10:
        verdict = "ABSTAIN"; why = f"positive control silent at D={d} (oracle-static={orc:+.3f}<0.10)."
    elif adv >= 0.10 and adv >= 0.5 * orc:
        verdict = "CONFIRMED"
        why = (f"On the HARDER task, at the starved D={d} the windowed-conv detector beats STATIC by {adv:+.3f} "
               f"(>=0.10) and captures {100*adv/orc:.0f}% of the oracle's {orc:+.3f} gap (>=50%). The clean "
               f"mode-scarcity adaptive-frequency win SCALES. Scarcity curve (adv vs D): {curve}.")
    elif adv < 0.05:
        verdict = "KILL"
        why = f"At starved D={d}, rich-static={adv:+.3f}<0.05 (oracle {orc:+.3f}): the scarcity win did NOT scale to the harder task."
    else:
        verdict = "WEAK"
        why = f"At starved D={d}, rich-static={adv:+.3f} (sub-threshold; capture {100*adv/orc:.0f}%). Scarcity curve: {curve}."

    res["gate"] = {"primary_d": d, "rich_minus_static": round(adv, 4), "oracle_minus_static": round(orc, 4),
                   "capture_frac": (round(cap, 4) if orc > 1e-9 else None), "scarcity_curve_adv_by_D": curve,
                   "advantage_falls_with_D": bool(falls), "verdict": verdict, "why": why}
    out = HERE / ("scarcity_scale_smoke.json" if SMOKE else "scarcity_scale_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  scarcity curve (rich-static vs D): {curve}", flush=True)
    print(f"  gate @ starved D={d}: rich-static={adv:+.3f}  oracle-static={orc:+.3f}  capture={cap:.2f}", flush=True)
    print("\n===== VERDICT:", verdict, "=====\n ", why)
    print("wrote", out.name, flush=True)


if __name__ == "__main__":
    main()

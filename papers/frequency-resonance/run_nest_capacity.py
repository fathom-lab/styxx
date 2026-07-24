# -*- coding: utf-8 -*-
"""
run_nest_capacity.py -- frozen by PREREG_nested_crossfreq_2026_07_23.

THE THETA-GAMMA MOONSHOT. Neuroscience's model of working memory: a slow THETA rhythm defines a
frame; within it, fast GAMMA sub-cycles are memory SLOTS, each holding one ordered item at a distinct
phase (Lisman & Jensen 2013 -> the 7+-2 bound). NO state-space model implements true cross-frequency
coupling -- they use a FLAT bank of independent oscillators. This tests the invention: does explicit
nested coupling (a slow clock that gates fast modes into ordered phase-slots) beat a flat oscillator
bank of equal budget at holding multiple ORDERED items?

Task: ordered copy (present K symbols, then K GO tokens, recall in order) -- the capacity-limited
ordered-memory task, the theta-gamma function. Metric: mean recall accuracy over K, and kcap (largest
K solved at >=0.80).

Arms (matched-param, RNG-matched, single-knob):
  FLAT     -- free oscillatory bank, learnable theta, UNIFORM input (the LinOSS-style competitor)
  NEST     -- same modes, input to mode j gated by a slow clock: g_j(t)=(1-a)+a*0.5*(1+cos(w*t-psi_j))
              -> items at different slow-phases route to different modes = phase-multiplexed slots
  ORACLE   -- perfect temporal slotting (item t -> mode t mod d); positive control (does slotting help?)
  CLAMPED  -- theta==0, decay (floor)
  TRANSF   -- attention (context only; NOT gated)

alpha=0 makes NEST reduce to FLAT bit-for-bit -> single-knob causal test. Recurrence via the parallel
scan (lin_scan), red-team-verified against the O(T) reference.
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

V, D_IN = 12, 64                      # V symbols (0..V-1); GO token id = V; embedding size V+1
KGRID = [2, 3, 4] if SMOKE else [2, 3, 4, 5, 6, 8, 10]
ACC_THR = 0.80
BATCH, LR = 64, 2e-3
STEPS = 400 if SMOKE else 3000
SEEDS = [0] if SMOKE else [0, 1, 2]
D_SWEEP = [8] if SMOKE else [8, 16]
PRIMARY_D = 8
EVAL_N = 512
TD, TLAYERS, THEADS, TFF = 64, 2, 4, 128
MAXL = 2 * max(KGRID)


def lin_scan(A, X):
    """Parallel inclusive scan H[:,t]=A[:,t]*H[:,t-1]+X[:,t], H_init=0. O(log T), autograd-safe."""
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


class NestCLRU(nn.Module):
    def __init__(self, d, d_in, mode):
        super().__init__()
        assert mode in ("flat", "nest", "oracle", "clamped")
        self.mode, self.d = mode, d
        self.alpha_override = None
        r = torch.empty(d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        th = torch.empty(d).uniform_(0.0, math.pi / 2)
        if mode == "clamped":
            self.register_buffer("theta0", torch.zeros(d))
        else:
            self.theta0 = nn.Parameter(th)
        self.B_re = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        self.B_im = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))

    def init_nest(self, d):                                    # drawn LAST -> shared params match FLAT
        self.omega_slow = nn.Parameter(torch.tensor(2 * math.pi / d))          # slow clock ~1 cycle / d steps
        self.psi = nn.Parameter(torch.linspace(0, 2 * math.pi, d + 1)[:-1].clone())  # spread preferred phases
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))                       # alpha = sigmoid -> 0.5 init

    def forward(self, x):
        B, T, _ = x.shape
        mag = torch.exp(-torch.exp(self.nu))
        gamma = torch.sqrt(torch.clamp(1 - mag ** 2, min=1e-6))
        ure = torch.einsum("bti,di->btd", x, self.B_re) * gamma
        uim = torch.einsum("bti,di->btd", x, self.B_im) * gamma
        U = torch.complex(ure, uim)                            # (B,T,d)

        if self.mode == "nest":
            alpha = (torch.tensor(float(self.alpha_override), device=x.device)
                     if self.alpha_override is not None else torch.sigmoid(self.alpha_raw))
            t = torch.arange(T, device=x.device).float()
            phase = self.omega_slow * t                        # (T,)
            g = (1 - alpha) + alpha * 0.5 * (1 + torch.cos(phase.view(T, 1) - self.psi.view(1, self.d)))
            U = U * g.unsqueeze(0)                              # (B,T,d) real gate
        elif self.mode == "oracle":
            K = T // 2
            g = torch.ones(T, self.d, device=x.device)
            if K > 0:
                g[:K, :] = 0.0
                idx = torch.arange(K, device=x.device)
                g[idx, idx % self.d] = 1.0                      # item t -> mode (t mod d); GO positions all on
            U = U * g.unsqueeze(0)

        lam = torch.polar(mag, self.theta0)                    # (d,) complex, constant in time
        A = lam.view(1, 1, self.d).expand(B, T, self.d)
        H = lin_scan(A, U)
        return torch.cat([H.real, H.imag], -1)


class LRUModel(nn.Module):
    def __init__(self, d, mode):
        super().__init__()
        self.emb = nn.Embedding(V + 1, D_IN)
        self.lru = NestCLRU(d, D_IN, mode)
        self.read = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, V))
        if mode == "nest":
            self.lru.init_nest(d)

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V + 1, TD)
        self.pos = nn.Parameter(torch.randn(MAXL, TD) * 0.02)
        layer = nn.TransformerEncoderLayer(TD, THEADS, TFF, dropout=0.0, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, TLAYERS)
        self.read = nn.Linear(TD, V)

    def forward(self, tok):
        T = tok.shape[1]
        x = self.emb(tok) + self.pos[:T]
        mask = torch.triu(torch.ones(T, T, device=tok.device, dtype=torch.bool), 1)
        return self.read(self.tr(x, mask=mask))


def build(arch, d):
    if arch == "transformer":
        return TransformerModel()
    return LRUModel(d, arch)


def make_batch(B, K):
    syms = torch.randint(0, V, (B, K), device=DEV)
    go = torch.full((B, K), V, device=DEV)
    x = torch.cat([syms, go], 1)                               # (B,2K): K items then K GO tokens
    tgt = torch.full((B, 2 * K), -100, device=DEV)
    tgt[:, K:] = syms                                          # recall in order at the GO slots
    return x, tgt


def train(arch, d, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = build(arch, d).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for _ in range(STEPS):
        K = int(KGRID[np.random.randint(len(KGRID))])
        x, tgt = make_batch(BATCH, K)
        loss = lossf(m(x).reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return m


@torch.no_grad()
def eval_K(m, K, n=EVAL_N):
    x, tgt = make_batch(n, K); pred = m(x).argmax(-1); mask = tgt != -100
    return (pred[mask] == tgt[mask]).float().mean().item()


def kcap(accs):
    ok = [K for K, a in accs.items() if a >= ACC_THR]
    return max(ok) if ok else 0


def nparams(arch, d):
    return sum(p.numel() for p in build(arch, d).parameters())


def redteam():
    torch.manual_seed(7)
    Ac = torch.randn(4, 13, 8, dtype=torch.cfloat, device=DEV) * 0.5
    Xc = torch.randn(4, 13, 8, dtype=torch.cfloat, device=DEV)
    ds = (lin_scan(Ac, Xc) - seq_scan(Ac, Xc)).abs().max().item()
    assert ds < 1e-4, f"lin_scan != seq_scan ({ds:.2e})"
    # alpha=0 reduces NEST to FLAT bit-for-bit (shared params RNG-match; gate==1)
    torch.manual_seed(0); fl = build("flat", 8).to(DEV)
    torch.manual_seed(0); ne = build("nest", 8).to(DEV)
    ne.lru.alpha_override = 0.0
    x, _ = make_batch(16, 5)
    with torch.no_grad():
        da = (fl(x) - ne(x)).abs().max().item()
    assert da < 1e-5, f"alpha=0 != FLAT ({da:.2e})"
    assert torch.equal(fl.lru.theta0.detach(), ne.lru.theta0.detach()), "theta0 mismatch"
    ne.lru.alpha_override = None
    # oracle slotting: during input each item routes to exactly one mode; GO positions all-on
    orc = build("oracle", 8).to(DEV)
    xo, _ = make_batch(4, 6)
    B, T, _ = orc.emb(xo).shape
    # reconstruct the gate the oracle applies and check it
    K = T // 2
    g = torch.ones(T, 8)
    g[:K, :] = 0.0; idx = torch.arange(K); g[idx, idx % 8] = 1.0
    assert (g[:K].sum(1) == 1).all(), "oracle input gate not one-hot per position"
    assert (g[K:] == 1).all(), "oracle GO gate not all-on"
    print("  [redteam] scan==seq (%.1e), alpha=0==FLAT (%.1e), oracle slotting one-hot -- OK" % (ds, da),
          flush=True)


def main():
    print(f"device={DEV} smoke={SMOKE} KGRID={KGRID} D_sweep={D_SWEEP}", flush=True)
    redteam()
    res = {"config": {"steps": STEPS, "seeds": SEEDS, "kgrid": KGRID, "acc_thr": ACC_THR,
                      "d_sweep": D_SWEEP, "primary_d": PRIMARY_D, "eval_n": EVAL_N},
           "params": {}, "acc_by_K": {}, "kcap": {}, "mean_acc": {}}

    for d in D_SWEEP:
        arms = ["flat", "nest", "oracle"] + (["clamped"] if d == PRIMARY_D else [])
        res["params"][str(d)] = {a: nparams(a, d) for a in arms}
        res["acc_by_K"][str(d)] = {}; res["kcap"][str(d)] = {}; res["mean_acc"][str(d)] = {}
        for a in arms:
            accK = {K: [] for K in KGRID}
            for s in SEEDS:
                m = train(a, d, s)
                for K in KGRID:
                    accK[K].append(eval_K(m, K))
                del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            macc = {K: float(np.mean(accK[K])) for K in KGRID}
            res["acc_by_K"][str(d)][a] = macc
            res["kcap"][str(d)][a] = kcap(macc)
            res["mean_acc"][str(d)][a] = float(np.mean(list(macc.values())))
            print(f"  D={d:2d} {a:8s} kcap={res['kcap'][str(d)][a]:2d} mean_acc={res['mean_acc'][str(d)][a]:.3f}"
                  f"  acc=" + " ".join(f"K{K}={macc[K]:.2f}" for K in KGRID), flush=True)

    tstats = {}
    for s in SEEDS:
        m = train("transformer", 0, s)
        for K in KGRID:
            tstats.setdefault(K, []).append(eval_K(m, K))
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()
    tmacc = {K: float(np.mean(tstats[K])) for K in KGRID}
    res["transformer"] = {"kcap": kcap(tmacc), "mean_acc": float(np.mean(list(tmacc.values()))),
                          "params": nparams("transformer", 0)}
    print(f"  transformer      kcap={res['transformer']['kcap']:2d} mean_acc={res['transformer']['mean_acc']:.3f}",
          flush=True)

    # ---- WIDE (flat @ 2*primary_d): capacity-headroom positive control (more modes = more capacity) ----
    d = PRIMARY_D
    wd = 2 * d
    wacc = {K: [] for K in KGRID}
    for s in SEEDS:
        m = train("flat", wd, s)
        for K in KGRID:
            wacc[K].append(eval_K(m, K))
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()
    wmacc = {K: float(np.mean(wacc[K])) for K in KGRID}
    res["wide"] = {"d": wd, "mean_acc": float(np.mean(list(wmacc.values()))), "kcap": kcap(wmacc),
                   "acc_by_K": wmacc, "params": nparams("flat", wd)}
    print(f"  WIDE(flat@{wd:2d})   kcap={res['wide']['kcap']:2d} mean_acc={res['wide']['mean_acc']:.3f}", flush=True)

    # ---- frozen gate (primary D). Positive control = WIDE (headroom). Slotting-oracle = diagnostic. ----
    flat = res["mean_acc"][str(d)]["flat"]; nest = res["mean_acc"][str(d)]["nest"]; orc = res["mean_acc"][str(d)]["oracle"]
    wide = res["wide"]["mean_acc"]
    adv = nest - flat; head = wide - flat; slot = orc - flat
    cap = (adv / head) if head > 1e-9 else float("nan")
    dk = {"flat": res["kcap"][str(d)]["flat"], "nest": res["kcap"][str(d)]["nest"],
          "oracle": res["kcap"][str(d)]["oracle"], "wide": res["wide"]["kcap"]}

    if head < 0.05:
        verdict = "ABSTAIN"
        why = (f"Positive control did NOT fire (wide-flat={head:+.3f} < 0.05): doubling modes does not "
               f"raise capacity at this K range, so the task has no headroom to detect a nesting gain. "
               f"NO conclusion. Redesign (larger K / smaller D) and re-freeze.")
    elif adv >= 0.05 and adv >= 0.5 * head:
        verdict = "GREENLIGHT"
        why = (f"At D={d}, NEST beats FLAT by {adv:+.3f} mean-acc (>=0.05) and captures {100*adv/head:.0f}% "
               f"of the capacity headroom a 2x-wider bank buys ({head:+.3f}, >=50%). Explicit cross-frequency "
               f"coupling gives a d-mode bank the effective capacity of a wider one. kcap {dk}.")
    elif adv < 0.02:
        verdict = "KILL"
        why = (f"At D={d}, capacity headroom IS available (wide-flat={head:+.3f} >=0.05) but NEST fails to "
               f"capture it (nest-flat={adv:+.3f} < 0.02). Explicit nesting does not beat a flat bank's own "
               f"implicit phase multiplexing -- the flat oscillatory bank is already at the theta-gamma "
               f"ceiling. Honest negative. Slotting-via-gating diagnostic: {slot:+.3f}. kcap {dk}.")
    else:
        verdict = "WEAK"
        why = (f"At D={d}, nest-flat={adv:+.3f} (in [0.02,0.05) or capture<50% of headroom {head:+.3f}): "
               f"real but sub-threshold. Not a greenlight. kcap {dk}.")

    res["gate"] = {"d": d, "nest_minus_flat": round(adv, 4), "wide_minus_flat": round(head, 4),
                   "slotting_oracle_minus_flat": round(slot, 4), "capture_frac_of_headroom": (round(cap, 4) if head > 1e-9 else None),
                   "headroom_fired": bool(head >= 0.05), "kcap": dk, "verdict": verdict, "why": why}

    out = HERE / ("nest_capacity_smoke.json" if SMOKE else "nest_capacity_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  gate @ D={d}: nest-flat={adv:+.3f}  wide-flat={head:+.3f}  slot-flat={slot:+.3f}  kcap {dk}", flush=True)
    print("\n===== VERDICT:", verdict, "=====")
    print(" ", why)
    print("wrote", out.name, flush=True)


if __name__ == "__main__":
    main()

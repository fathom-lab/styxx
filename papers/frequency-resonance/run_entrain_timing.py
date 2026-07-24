# -*- coding: utf-8 -*-
"""
run_entrain_timing.py -- frozen by PREREG_entrainment_2026_07_23.

ENTRAIN-OSS kill-gate. The prior arc showed the free oscillatory net wins timing via a BROAD STATIC
phase bank (it does NOT tune theta->2pi/P). This tests the invention that lives in exactly the regime
that finding leaves open: when the input's period DRIFTS within a sequence and the mode budget D is
STARVED, can a layer that TUNES its own oscillation to the input beat the best static bank?

Task: within-sequence drifting-period next-symbol prediction. Each sequence is S segments; each segment
has its own period drawn from [PMIN,PMAX]; adjacent segments differ. Score only positions >= 2 local
periods into a segment (the local period is inferable). Sweep mode budget D.

Arms (matched-param, RNG-matched inits, single-knob discipline):
  STATIC   -- theta learned CONSTANT (the strong LinOSS-style competitor; the baseline to beat)
  ENTRAIN  -- theta(t) driven by a slow learned PLL over an online phase estimate (the invention)
  ORACLE   -- ENTRAIN machinery but fed the TRUE 2pi/P(t) (positive control: does adaptation help AT ALL)
  CLAMPED  -- theta==0, pure decay (floor/sanity)
  TRANSF   -- attention, learned positions (context only; NOT in the gate)

kappa==0 makes ENTRAIN reduce to STATIC exactly -> a single-knob causal test. STATIC and ENTRAIN share
the same sequential recurrence -> matched-COMPUTE by construction (the FLOP-fairness confound only bites
the transformer, which is not gated). Gate is frozen in the prereg; primary D=8, fallback D=4.
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

# ---- frozen task/config constants ----
V, D_IN, L = 12, 64, 96
SEG_LEN = 32                      # S = L // SEG_LEN = 3 segments -> 2 change points per sequence
PMIN, PMAX = 3, 12                # per-segment integer period range
KAPPA_MAX = 0.1                   # loop-gain cap: modal basis rotates on a timescale >> 1 token
BATCH, LR = 64, 2e-3
STEPS = 300 if SMOKE else 1500
SEEDS = [0] if SMOKE else [0, 1, 2]
D_SWEEP = [8] if SMOKE else [4, 8, 16]
PRIMARY_D, FALLBACK_D = 8, 4
EVAL_N = 512 if SMOKE else 1024
# transformer (context arm; D-independent, run once)
TD, TLAYERS, THEADS, TFF = 64, 2, 4, 128

S = L // SEG_LEN


def lin_scan(A, X):
    """Parallel inclusive scan of the first-order linear recurrence H[:,t] = A[:,t]*H[:,t-1] + X[:,t]
    with H_init = 0 (so H[:,0] = X[:,0]). Hillis-Steele, O(log T) sequential steps. Works for real or
    complex dtype. Fully functional (no in-place) so it is autograd-safe. A, X: (B, T, d)."""
    T = A.shape[1]
    shift = 1
    while shift < T:
        Ap = torch.cat([torch.ones_like(A[:, :shift]), A[:, :T - shift]], dim=1)
        Xp = torch.cat([torch.zeros_like(X[:, :shift]), X[:, :T - shift]], dim=1)
        X = X + A * Xp
        A = A * Ap
        shift <<= 1
    return X


def seq_scan(A, X):
    """Reference O(T) sequential scan (same recurrence) -- used only to validate lin_scan."""
    T = A.shape[1]
    h = torch.zeros_like(X[:, 0]); out = []
    for t in range(T):
        h = A[:, t] * h + X[:, t]; out.append(h)
    return torch.stack(out, 1)


class EntrainCLRU(nn.Module):
    """Complex-diagonal recurrence. mode selects how theta evolves.
    Shared params (nu, theta0, B_re, B_im) are drawn in the SAME order in every mode so STATIC and
    ENTRAIN get identical inits under the same seed; entrain-only params are drawn LAST."""

    def __init__(self, d, d_in, mode):
        super().__init__()
        assert mode in ("static", "entrain", "oracle", "clamped")
        self.mode, self.d = mode, d
        self.kappa_override = None                      # red-team hook: force a fixed kappa
        r = torch.empty(d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        th = torch.empty(d).uniform_(0.0, math.pi / 2)   # always drawn (RNG alignment)
        if mode == "clamped":
            self.register_buffer("theta0", torch.zeros(d))
        else:
            self.theta0 = nn.Parameter(th)               # learnable phase init / constant
        self.B_re = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        self.B_im = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        # entrain-only params are created by init_entrain() AFTER the read head (see LRUModel), so the
        # shared params above RNG-match STATIC exactly -> kappa=0 reduces ENTRAIN to STATIC bit-for-bit.

    def init_entrain(self, d_in, d):
        self.a = nn.Parameter(torch.randn(d_in) / math.sqrt(d_in))       # online phase detector proj
        self.b = nn.Parameter(torch.randn(d_in) / math.sqrt(d_in))
        self.kappa_raw = nn.Parameter(torch.zeros(d))                    # kappa = KAPPA_MAX*sigmoid -> 0.05
        self.g = nn.Parameter(torch.ones(d))                            # per-mode harmonic ratio

    def init_oracle(self, d):
        # positive control: a DIVERSE bank locked to the true fundamental (fixed geometric spread,
        # instant lock). NOT rank-1 -- modes span [0.5, 2.0]x the true 2pi/P(t). No learned params.
        self.register_buffer("spread", torch.exp(torch.linspace(math.log(0.5), math.log(2.0), d)))

    def _kappa(self, dev):
        if self.kappa_override is not None:
            return torch.full((self.d,), float(self.kappa_override), device=dev)
        return KAPPA_MAX * torch.sigmoid(self.kappa_raw)

    def forward(self, x, omega_true=None):
        B, T, _ = x.shape
        mag = torch.exp(-torch.exp(self.nu))                          # (d,)
        gamma = torch.sqrt(torch.clamp(1 - mag ** 2, min=1e-6))       # (d,)
        ure = torch.einsum("bti,di->btd", x, self.B_re) * gamma
        uim = torch.einsum("bti,di->btd", x, self.B_im) * gamma
        U = torch.complex(ure, uim)                                   # (B,T,d) complex input
        magf = mag.view(1, 1, self.d).expand(B, T, self.d)

        if self.mode in ("static", "clamped"):
            theta = self.theta0.view(1, 1, self.d).expand(B, T, self.d)
        elif self.mode == "oracle":
            # positive control: diverse bank locked to the TRUE fundamental (instant, no PLL lag)
            theta = torch.clamp(self.spread.view(1, 1, self.d) * omega_true.unsqueeze(-1), 0.0, math.pi)
        else:  # entrain -- theta(t) tracks the ONLINE input-frequency estimate via a slow linear PLL
            zr = torch.einsum("bti,i->bt", x, self.a)                 # (B,T)
            zi = torch.einsum("bti,i->bt", x, self.b)
            cross = torch.cat([torch.zeros(B, 1, device=x.device),
                               zi[:, 1:] * zr[:, :-1] - zr[:, 1:] * zi[:, :-1]], dim=1)
            dot = torch.cat([torch.ones(B, 1, device=x.device),
                             zr[:, 1:] * zr[:, :-1] + zi[:, 1:] * zi[:, :-1]], dim=1)
            omega = torch.atan2(cross, dot)                           # (B,T), omega[:,0]=0
            kappa = self._kappa(x.device)                            # (d,)
            drive = self.g.view(1, 1, self.d) * omega.unsqueeze(-1)   # (B,T,d)
            # PLL: theta(t)=(1-k)theta(t-1)+k*drive(t), with theta(0)=(1-k)theta0 (drive(0)=0). No clamp
            # (e^{i*theta} is periodic, so unbounded theta is valid; if anything this STRENGTHENS entrain).
            A_th = torch.cat([torch.zeros(B, 1, self.d, device=x.device),
                              (1 - kappa).view(1, 1, self.d).expand(B, T - 1, self.d)], dim=1)
            X0 = ((1 - kappa) * self.theta0).view(1, 1, self.d).expand(B, 1, self.d)
            X_th = torch.cat([X0, kappa.view(1, 1, self.d) * drive[:, 1:, :]], dim=1)
            theta = lin_scan(A_th, X_th)                              # (B,T,d)

        lam = torch.polar(magf.contiguous(), theta.contiguous())     # (B,T,d) complex eigenvalues
        H = lin_scan(lam, U)                                         # complex state trajectory
        return torch.cat([H.real, H.imag], -1)


class LRUModel(nn.Module):
    def __init__(self, d, mode):
        super().__init__()
        self.emb = nn.Embedding(V, D_IN)
        self.lru = EntrainCLRU(d, D_IN, mode)
        self.read = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, V))
        if mode == "entrain":                    # drawn LAST -> shared params match STATIC exactly
            self.lru.init_entrain(D_IN, d)
        elif mode == "oracle":
            self.lru.init_oracle(d)

    def forward(self, tok, omega_true=None):
        return self.read(self.lru(self.emb(tok), omega_true))


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, TD)
        self.pos = nn.Parameter(torch.randn(L, TD) * 0.02)   # learned positions = rhythm-free
        layer = nn.TransformerEncoderLayer(TD, THEADS, TFF, dropout=0.0, batch_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(layer, TLAYERS)
        self.read = nn.Linear(TD, V)

    def forward(self, tok, omega_true=None):
        T = tok.shape[1]
        x = self.emb(tok) + self.pos[:T]
        mask = torch.triu(torch.ones(T, T, device=tok.device, dtype=torch.bool), 1)
        return self.read(self.tr(x, mask=mask))


def build(arch, d):
    if arch == "transformer":
        return TransformerModel()
    return LRUModel(d, arch)


def make_batch(B, drift=True):
    """Returns inp (B,L-1), tgt (B,L-1) with -100 on un-inferable positions, omega_true (B,L-1)."""
    t = torch.arange(L, device=DEV)
    if drift:
        seg_P = torch.randint(PMIN, PMAX + 1, (B, S), device=DEV)
        for _ in range(6):                                    # force adjacent segments to differ
            eq = seg_P[:, 1:] == seg_P[:, :-1]
            if not eq.any():
                break
            seg_P[:, 1:][eq] = torch.randint(PMIN, PMAX + 1, (int(eq.sum().item()),), device=DEV)
        seg_idx = (t // SEG_LEN).clamp(max=S - 1)             # (L,)
        P_t = seg_P[:, seg_idx]                               # (B,L)
        local = (t - seg_idx * SEG_LEN).unsqueeze(0).expand(B, L)   # position within segment
    else:                                                     # fixed-period no-harm control
        P_seq = torch.randint(PMIN, PMAX + 1, (B, 1), device=DEV)
        P_t = P_seq.expand(B, L)
        local = t.unsqueeze(0).expand(B, L)
    motif = torch.randint(0, V, (B, PMAX), device=DEV)
    idx = local % P_t
    x = torch.gather(motif, 1, idx)                           # (B,L) quasi-periodic stream
    inp = x[:, :-1]
    tgt = x[:, 1:].clone()
    omega = (2 * math.pi) / P_t.float()
    omega_true = omega[:, :-1]
    scored = local >= (2 * P_t)                               # >= 2 local periods seen
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
    """Frozen pre-run asserts. Must pass before any result is trusted."""
    # (0) the parallel scan must equal the O(T) sequential reference (complex + real)
    torch.manual_seed(7)
    Ac = torch.randn(4, 17, 8, dtype=torch.cfloat, device=DEV) * 0.5
    Xc = torch.randn(4, 17, 8, dtype=torch.cfloat, device=DEV)
    ds = (lin_scan(Ac, Xc) - seq_scan(Ac, Xc)).abs().max().item()
    assert ds < 1e-4, f"lin_scan != seq_scan (complex, max|diff|={ds:.2e})"
    Ar = torch.rand(4, 17, 8, device=DEV); Xr = torch.randn(4, 17, 8, device=DEV)
    dr = (lin_scan(Ar, Xr) - seq_scan(Ar, Xr)).abs().max().item()
    assert dr < 1e-4, f"lin_scan != seq_scan (real, max|diff|={dr:.2e})"

    torch.manual_seed(0)
    st = build("static", 8).to(DEV)
    torch.manual_seed(0)
    en = build("entrain", 8).to(DEV)
    en.lru.kappa_override = 0.0                               # force theta(t)=theta0
    inp, tgt, om = make_batch(16, drift=True)
    with torch.no_grad():
        a = st(inp, om); b = en(inp, om)
    max_abs = (a - b).abs().max().item()
    assert max_abs < 1e-5, f"kappa=0 does NOT reproduce STATIC (max|diff|={max_abs:.2e})"
    # shared params identical under matched seed
    assert torch.equal(st.lru.theta0.detach(), en.lru.theta0.detach()), "theta0 init mismatch"
    assert torch.equal(st.lru.B_re.detach(), en.lru.B_re.detach()), "B_re init mismatch"
    # oracle really receives true 2pi/P(t): reconstruct P from omega on a fixed-period batch
    inp2, tgt2, om2 = make_batch(8, drift=False)
    recon_P = (2 * math.pi) / om2
    assert torch.allclose(recon_P, recon_P.round(), atol=1e-4), "omega_true not exactly 2pi/integer-P"
    # scoring mask excludes warmup: first (2*P-1) targets of each segment must be ignored
    inp3, tgt3, om3 = make_batch(64, drift=True)
    P0 = ((2 * math.pi) / om3[:, 0]).round().long()          # period of first segment
    # target index j corresponds to local position j+1; must be -100 while (j+1) < 2*P0 within seg 0
    for b in range(8):
        p = int(P0[b].item())
        warm = tgt3[b, : (2 * p - 1)]                        # targets before 2 periods elapsed
        assert (warm == -100).all(), f"warmup not masked for seq {b} (P={p})"
    print("  [redteam] scan==seq (c=%.1e r=%.1e), kappa=0==STATIC (%.1e), oracle=2pi/P exact, warmup masked -- OK"
          % (ds, dr, max_abs), flush=True)
    en.lru.kappa_override = None


def main():
    print(f"device={DEV} smoke={SMOKE} L={L} S={S} periods=[{PMIN},{PMAX}] D_sweep={D_SWEEP}", flush=True)
    redteam()

    res = {"config": {"steps": STEPS, "seeds": SEEDS, "L": L, "seg_len": SEG_LEN, "S": S,
                      "periods": [PMIN, PMAX], "kappa_max": KAPPA_MAX, "d_sweep": D_SWEEP,
                      "primary_d": PRIMARY_D, "fallback_d": FALLBACK_D, "eval_n": EVAL_N},
           "params": {}, "drift": {}, "fixed": {}}

    # static/entrain/oracle across the full D sweep; clamped (floor) only at the primary D
    for d in D_SWEEP:
        arms = ["static", "entrain", "oracle"] + (["clamped"] if d == PRIMARY_D else [])
        res["drift"][str(d)] = {}; res["fixed"][str(d)] = {}
        res["params"][str(d)] = {a: nparams(a, d) for a in arms}
        for a in arms:
            dacc, facc = [], []
            for s in SEEDS:
                m = train(a, d, s)
                dacc.append(evaluate(m, drift=True))
                facc.append(evaluate(m, drift=False))
                del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            res["drift"][str(d)][a] = float(np.mean(dacc))
            res["fixed"][str(d)][a] = float(np.mean(facc))
            print(f"  D={d:2d} {a:8s} drift={np.mean(dacc):.3f} fixed={np.mean(facc):.3f}", flush=True)

    # transformer (context only, D-independent) -- run once
    tacc_d, tacc_f = [], []
    for s in SEEDS:
        m = train("transformer", 0, s)
        tacc_d.append(evaluate(m, drift=True)); tacc_f.append(evaluate(m, drift=False))
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()
    res["transformer"] = {"drift": float(np.mean(tacc_d)), "fixed": float(np.mean(tacc_f)),
                          "params": nparams("transformer", 0)}
    print(f"  transformer      drift={np.mean(tacc_d):.3f} fixed={np.mean(tacc_f):.3f}", flush=True)

    # ---- frozen gate ----
    def gate_at(d):
        g = res["drift"][str(d)]
        adv = g["entrain"] - g["static"]
        orc = g["oracle"] - g["static"]
        capture = (adv / orc) if orc > 1e-9 else float("nan")
        return {"d": d, "entrain_minus_static": round(adv, 4), "oracle_minus_static": round(orc, 4),
                "capture_frac": (round(capture, 4) if orc > 1e-9 else None),
                "oracle_fired": bool(orc >= 0.10)}

    primary = gate_at(PRIMARY_D)
    used = primary
    if not primary["oracle_fired"]:                          # positive control did not fire at primary D
        used = gate_at(FALLBACK_D)

    adv, orc = used["entrain_minus_static"], used["oracle_minus_static"]
    noharm = res["fixed"][str(used["d"])]["entrain"] - res["fixed"][str(used["d"])]["static"]
    d_trend = [round(res["drift"][str(d)]["entrain"] - res["drift"][str(d)]["static"], 4) for d in D_SWEEP]

    if orc < 0.10:
        verdict = "ABSTAIN"
        why = (f"Positive control did NOT fire at D={PRIMARY_D} or fallback D={FALLBACK_D} "
               f"(oracle-static={orc:+.3f} < 0.10): the task does not reward adaptation at the tested mode "
               f"budgets, so the instrument cannot detect the effect. NO conclusion about the invention. "
               f"Redesign (widen period range / smaller D) and re-freeze.")
    elif adv >= 0.10 and adv >= 0.5 * orc:
        verdict = "GREENLIGHT"
        why = (f"At D={used['d']}, ENTRAIN beats the STATIC bank by {adv:+.3f} (>=0.10) and captures "
               f"{100*adv/orc:.0f}% of the oracle's {orc:+.3f} adaptation gap (>=50%). The learned "
               f"entrainment loop captures a real, mechanism-tied slice of the available adaptation. "
               f"No-harm on fixed period: {noharm:+.3f}. Proceed to Phase 2.")
    elif adv < 0.05:
        verdict = "KILL"
        why = (f"At D={used['d']}, adaptation IS available and rewarded (oracle-static={orc:+.3f} >=0.10) "
               f"but the learnable loop fails to capture it (entrain-static={adv:+.3f} < 0.05). The premise "
               f"is falsified: a static bank of equal budget is as good as the learned entrainment. "
               f"Ship the honest negative; pivot to nested cross-frequency or the profiler tool.")
    else:
        verdict = "WEAK"
        why = (f"At D={used['d']}, entrain-static={adv:+.3f} (in [0.05,0.10) or capture<50% of "
               f"oracle {orc:+.3f}): a real but sub-threshold effect. Does not clear the frozen bar. "
               f"Treat as a non-greenlight; do not scale on this alone.")

    res["gate"] = {"primary": primary, "used": used, "noharm_fixed_entrain_minus_static": round(noharm, 4),
                   "entrain_advantage_by_D": {str(d): v for d, v in zip(D_SWEEP, d_trend)},
                   "advantage_falls_with_D": bool(d_trend == sorted(d_trend, reverse=True)),
                   "verdict": verdict, "why": why}

    out = HERE / ("entrainment_smoke.json" if SMOKE else "entrainment_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  entrain advantage by D:", {d: v for d, v in zip(D_SWEEP, d_trend)}, flush=True)
    print("  gate @ D=%d: entrain-static=%+.3f  oracle-static=%+.3f  no-harm=%+.3f"
          % (used["d"], adv, orc, noharm), flush=True)
    print("\n===== VERDICT:", verdict, "=====")
    print(" ", why)
    print("wrote", out.name, flush=True)


if __name__ == "__main__":
    main()

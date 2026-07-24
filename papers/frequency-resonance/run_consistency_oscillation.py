# -*- coding: utf-8 -*-
"""
run_consistency_oscillation.py -- frozen by PREREG_consistency_oscillation_2026_07_23.

Does LONG-RANGE CROSS-CONTEXT CONSISTENCY ride the oscillatory channel? Delayed majority-consistency:
aggregate P scattered premise bits, then judge whether a final claimed majority CONTRADICTS them. LONG =
premises scattered across the sequence (non-local integration required); LOCAL = premises clustered just
before the probe (length-matched control, decay suffices). Same CLRU phase-clamp as the pMNIST ablation
(FREE theta learnable vs CLAMPED theta==0, matched-param, RNG-matched). Difference-in-differences isolates
non-locality. Mechanistic precondition for "honesty rides oscillation" -- NOT a real-LLM honesty claim.
"""
from __future__ import annotations
import sys, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = "--smoke" in sys.argv

H, D_SSM, N_BLK = 64, 64, 3
T_LEN, P_PREM = 256, 7                   # sequence length; # premise bits (odd -> no ties)
DATA_SEED = 1234
N_TRAIN, N_TEST = (4000, 1000) if SMOKE else (24000, 6000)
BATCH, LR, WD = 64, 3e-3, 0.01
STEPS = 300 if SMOKE else 5000
SEEDS = [0] if SMOKE else [0, 1]
EVAL_EVERY = 150 if SMOKE else 1000


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


class CLRU(nn.Module):
    def __init__(self, d, d_in, free):
        super().__init__()
        self.d, self.free = d, free
        r = torch.empty(d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        th = torch.empty(d).uniform_(0.0, math.pi / 2)
        if free:
            self.theta = nn.Parameter(th)
        else:
            self.register_buffer("theta", torch.zeros(d))
        self.B_re = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        self.B_im = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        self.clamp_theta = False

    def forward(self, x):
        B, T, _ = x.shape
        mag = torch.exp(-torch.exp(self.nu))
        gamma = torch.sqrt(torch.clamp(1 - mag ** 2, min=1e-6))
        ure = torch.einsum("bti,di->btd", x, self.B_re) * gamma
        uim = torch.einsum("bti,di->btd", x, self.B_im) * gamma
        U = torch.complex(ure, uim)
        theta = torch.zeros_like(self.theta) if self.clamp_theta else self.theta
        lam = torch.polar(mag, theta).view(1, 1, self.d).expand(B, T, self.d)
        Hh = lin_scan(lam, U)
        return torch.cat([Hh.real, Hh.imag], -1)


class Block(nn.Module):
    def __init__(self, free):
        super().__init__()
        self.ssm = CLRU(D_SSM, H, free)
        self.proj = nn.Linear(2 * D_SSM, H)
        self.norm1 = nn.LayerNorm(H)
        self.ff = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(), nn.Linear(2 * H, H))
        self.norm2 = nn.LayerNorm(H)

    def forward(self, x):
        x = self.norm1(x + self.proj(self.ssm(x)))
        x = self.norm2(x + self.ff(x))
        return x


class ConsistencySSM(nn.Module):
    """d_in=2 (premise, claim); binary head read off the FINAL position (causal probe)."""
    def __init__(self, free):
        super().__init__()
        self.emb = nn.Linear(2, H, bias=False)             # no bias -> filler positions inject nothing
        self.blocks = nn.ModuleList([Block(free) for _ in range(N_BLK)])
        self.head = nn.Linear(H, 2)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x[:, -1])            # last-token readout (probe has seen all premises)

    def set_clamp(self, flag):
        for b in self.blocks:
            b.ssm.clamp_theta = flag


def make_data(task, gap, n, seed):
    """Two input channels: (0) a premise value +-1 at position `pos`; (1) a claimed value +-1 at the
    final position. Premise and claim are independent balanced +-1.
      task='cmp'       -> label = (claim == premise): a temporal COMPARISON (needs the two facts kept
                          separable to compute their product). gap='adj' (premise@T-2) or 'long'
                          (premise@0) tests range-(in)dependence.
      task='claimonly' -> label = (claim == +1): read the claim at the probe, IGNORE the premise. A
                          decay unit reads the adjacent claim trivially -> the capacity control the
                          clamp must pass. (gap is irrelevant here; premise is a present-but-mute
                          distractor.)
    Deterministic in seed. Returns X (n,T,2), y (n,)."""
    g = np.random.default_rng(seed)
    X = np.zeros((n, T_LEN, 2), dtype=np.float32)
    premise = g.choice([-1.0, 1.0], size=n)
    claim = g.choice([-1.0, 1.0], size=n)
    pos = 0 if gap == "long" else T_LEN - 2
    if task == "cmp":
        X[np.arange(n), pos, 0] = premise                  # cmp: two facts to keep separable
        y = (claim == premise).astype(np.int64)
    else:                                                  # claimonly: NO premise -> a lone input the
        y = (claim > 0).astype(np.int64)                   # decay model can read trivially (the control)
    X[np.arange(n), T_LEN - 1, 1] = claim
    return torch.from_numpy(X), torch.from_numpy(y)


@torch.no_grad()
def test_acc(m, xte, yte, bs=1000):
    m.eval()
    correct = 0
    for i in range(0, len(xte), bs):
        xb = xte[i:i + bs].to(DEV)
        correct += (m(xb).argmax(-1).cpu() == yte[i:i + bs]).sum().item()
    m.train()
    return correct / len(xte)


def train(free, seed, xtr, ytr, xte, yte):
    torch.manual_seed(seed); np.random.seed(seed)
    m = ConsistencySSM(free).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    lossf = nn.CrossEntropyLoss()
    N = len(xtr)
    for step in range(STEPS):
        idx = torch.randint(0, N, (BATCH,))
        loss = lossf(m(xtr[idx].to(DEV)), ytr[idx].to(DEV))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % EVAL_EVERY == 0:
            print(f"    step {step+1:4d} loss {loss.item():.3f} test {test_acc(m, xte, yte):.4f}", flush=True)
    return m


CONDS = [("cmp", "adj"), ("cmp", "long"), ("claimonly", "adj")]


def redteam():
    torch.manual_seed(7)
    Ac = torch.randn(3, 40, 8, dtype=torch.cfloat, device=DEV) * 0.5
    Xc = torch.randn(3, 40, 8, dtype=torch.cfloat, device=DEV)
    ds = (lin_scan(Ac, Xc) - seq_scan(Ac, Xc)).abs().max().item()
    assert ds < 1e-4, f"scan!=seq ({ds:.2e})"
    torch.manual_seed(0); mf = ConsistencySSM(True)
    torch.manual_seed(0); mc = ConsistencySSM(False)
    assert torch.equal(mf.blocks[0].ssm.B_re, mc.blocks[0].ssm.B_re), "B_re RNG mismatch"
    assert torch.equal(mf.blocks[0].ssm.nu, mc.blocks[0].ssm.nu), "nu RNG mismatch"
    # premise positions; balance; and claimonly label must be independent of the premise, cmp must depend on it
    Xa, ya = make_data("cmp", "adj", 3000, 999)
    Xl, yl = make_data("cmp", "long", 3000, 999)
    Xc0, yc0 = make_data("claimonly", "adj", 3000, 999)
    apos = np.array([np.where(Xa[b, :, 0].numpy() != 0)[0][0] for b in range(3000)])
    lpos = np.array([np.where(Xl[b, :, 0].numpy() != 0)[0][0] for b in range(3000)])
    assert (apos == T_LEN - 2).all(), "cmp-adj premise not adjacent"
    assert (lpos == 0).all(), "cmp-long premise not at 0"
    # claimonly label = sign(claim), independent of premise; cmp label = (claim==premise)
    prem_c = Xc0[:, :, 0].sum(1).numpy(); claim_c = Xc0[:, T_LEN - 1, 1].numpy()
    assert np.all(yc0.numpy() == (claim_c > 0)), "claimonly label not claim-sign"
    assert abs(ya.float().mean() - 0.5) < 0.05 and abs(yc0.float().mean() - 0.5) < 0.05, "labels unbalanced"
    print(f"  [redteam] scan==seq ({ds:.1e}); RNG matched; cmp-adj@{apos[0]} cmp-long@{lpos[0]} "
          f"probe@{T_LEN-1}; balance cmp={ya.float().mean():.3f} claimonly={yc0.float().mean():.3f} -- OK", flush=True)


def main():
    print(f"device={DEV} smoke={SMOKE} T={T_LEN} H={H} blocks={N_BLK} steps={STEPS} seeds={SEEDS}", flush=True)
    redteam()
    res = {"config": {"task": "delayed-consistency: cmp(claim==premise) vs claimonly control", "T": T_LEN,
                      "H": H, "d_ssm": D_SSM, "blocks": N_BLK, "steps": STEPS, "seeds": SEEDS,
                      "n_train": N_TRAIN, "n_test": N_TEST, "conditions": [f"{t}-{g}" for t, g in CONDS]},
           "acc": {}}
    data = {f"{t}_{g}": (make_data(t, g, N_TRAIN, DATA_SEED), make_data(t, g, N_TEST, DATA_SEED + 1))
            for t, g in CONDS}

    within = {}                          # cond -> list of (free_acc, free_then_clamped_acc)
    for t, g in CONDS:
        cond = f"{t}_{g}"
        (xtr, ytr), (xte, yte) = data[cond]
        for arm_free in (True, False):
            arm = "free" if arm_free else "clamped"
            accs = []
            for s in SEEDS:
                t0 = time.time()
                print(f"  --- {cond} {arm} seed {s} ---", flush=True)
                m = train(arm_free, s, xtr, ytr, xte, yte)
                acc = test_acc(m, xte, yte)
                accs.append(acc)
                print(f"  {cond} {arm} seed {s}: TEST {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                if arm_free:
                    m.set_clamp(True); ca = test_acc(m, xte, yte); m.set_clamp(False)
                    within.setdefault(cond, []).append((acc, ca))
                    print(f"    within-model theta->0: {acc:.4f} -> {ca:.4f} (reliance {acc-ca:+.4f})", flush=True)
                del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()
            res["acc"][f"{cond}_{arm}"] = [round(a, 4) for a in accs]

    def mean(k):
        return float(np.mean(res["acc"][k]))
    F_cmp_adj, C_cmp_adj = mean("cmp_adj_free"), mean("cmp_adj_clamped")
    F_cmp_long, C_cmp_long = mean("cmp_long_free"), mean("cmp_long_clamped")
    F_claim, C_claim = mean("claimonly_adj_free"), mean("claimonly_adj_clamped")
    rel_cmp_adj = float(np.mean([a - c for a, c in within["cmp_adj"]]))
    rel_cmp_long = float(np.mean([a - c for a, c in within["cmp_long"]]))

    gap_long = F_cmp_long - C_cmp_long
    gap_adj = F_cmp_adj - C_cmp_adj
    did = gap_long - gap_adj
    # positive controls: clamp can READ a lone input; oscillation CAN do the long comparison; and --
    # crucially -- decay CAN compare when the facts are ADJACENT (so a long failure = distance, not
    # "decay can't compare at all")
    pos_ctrl = (C_claim >= 0.90) and (F_cmp_long >= 0.85) and (C_cmp_adj >= 0.80)
    if not pos_ctrl:
        verdict = "ABSTAIN__positive_controls_failed"
    elif gap_long >= 0.20 and C_cmp_long <= 0.65:
        verdict = "SUPPORT__long_range_comparison_requires_oscillation"
    elif C_cmp_long >= 0.80:
        verdict = "NULL__decay_compares_at_distance_too"
    else:
        verdict = "PARTIAL__reported_verbatim"

    res["result"] = {
        "cmp_adj_free": round(F_cmp_adj, 4), "cmp_adj_clamped": round(C_cmp_adj, 4),
        "cmp_long_free": round(F_cmp_long, 4), "cmp_long_clamped": round(C_cmp_long, 4),
        "claimonly_free": round(F_claim, 4), "claimonly_clamped": round(C_claim, 4),
        "gap_adj": round(gap_adj, 4), "gap_long": round(gap_long, 4),
        "difference_in_differences": round(did, 4),
        "within_model_reliance_cmp_adj": round(rel_cmp_adj, 4),
        "within_model_reliance_cmp_long": round(rel_cmp_long, 4),
        "positive_controls_fired": bool(pos_ctrl),
        "verdict": verdict,
    }
    out = HERE / ("consistency_oscillation_smoke.json" if SMOKE else "consistency_oscillation_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  claimonly free {F_claim:.4f} clamped {C_claim:.4f}   (clamp CAN read a lone input)")
    print(f"  cmp-adj   free {F_cmp_adj:.4f} clamped {C_cmp_adj:.4f}   gap {gap_adj:+.4f}  (decay CAN compare adjacent)")
    print(f"  cmp-long  free {F_cmp_long:.4f} clamped {C_cmp_long:.4f}   gap {gap_long:+.4f}  (does decay compare at distance?)")
    print(f"  DiD (long-adj) {did:+.4f} | within-model reliance cmp-long {rel_cmp_long:+.4f}")
    print("  ===== VERDICT:", verdict, "=====")
    print("  wrote", out.name, flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
run_smnist_ablation.py -- frozen by PREREG_smnist_ablation_2026_07_23.

THE REAL-TASK FLAGSHIP. The oscillation-vs-decay causal ablation LinOSS never published, on a REAL
long-range benchmark: sequential MNIST (784-length pixel sequence -> 10-way classification). A deep
S5/LinOSS-class SSM classifier (stacked complex-diagonal blocks + GLU-MLP + residual + norm, mean-pool
head). Two arms, matched-param, RNG-matched:
  FREE     -- eigenvalue phase theta LEARNABLE (rotation -> oscillation): the S5/LinOSS-class model
  CLAMPED  -- theta == 0 (real eigenvalues -> pure decay): a diagonal real SSM, the ablation
Single knob = theta. If oscillation is causally load-bearing on real long-range data, FREE > CLAMPED.
Recurrence via the parallel scan (lin_scan), red-team-verified against the O(T) reference.
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
T_LEN = 784
BATCH, LR, WD = 64, 3e-3, 0.01
STEPS = 200 if SMOKE else 4000
SEEDS = [0] if SMOKE else [0, 1]
EVAL_EVERY = 100 if SMOKE else 1000


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
    """Complex-diagonal SSM. free -> learnable eigenvalue phase (oscillation); else theta==0 (decay)."""
    def __init__(self, d, d_in, free):
        super().__init__()
        self.d, self.free = d, free
        r = torch.empty(d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        th = torch.empty(d).uniform_(0.0, math.pi / 2)          # drawn in both arms (RNG match)
        if free:
            self.theta = nn.Parameter(th)
        else:
            self.register_buffer("theta", torch.zeros(d))
        self.B_re = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        self.B_im = nn.Parameter(torch.randn(d, d_in) / math.sqrt(d_in))
        self.clamp_theta = False                                # profiler hook (force theta->0 in-place)

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
        return torch.cat([Hh.real, Hh.imag], -1)                # (B,T,2d)


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


class SSMClassifier(nn.Module):
    def __init__(self, free):
        super().__init__()
        self.emb = nn.Linear(1, H)
        self.blocks = nn.ModuleList([Block(free) for _ in range(N_BLK)])
        self.head = nn.Linear(H, 10)

    def forward(self, x):                                       # x: (B,T,1)
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x.mean(1))


def load_data():
    d = np.load(HERE / "_mnist.npz")
    def prep(xk, yk):
        x = d[xk].astype(np.float32).reshape(-1, T_LEN, 1) / 255.0
        x = (x - 0.1307) / 0.3081
        return torch.from_numpy(x), torch.from_numpy(d[yk])
    return prep("xtr", "ytr"), prep("xte", "yte")


@torch.no_grad()
def test_acc(m, xte, yte, n=None, bs=500):
    m.eval()
    if n:
        xte, yte = xte[:n], yte[:n]
    correct = 0
    for i in range(0, len(xte), bs):
        xb = xte[i:i + bs].to(DEV)
        pred = m(xb).argmax(-1).cpu()
        correct += (pred == yte[i:i + bs]).sum().item()
    m.train()
    return correct / len(xte)


def train(free, seed, xtr, ytr, xte, yte):
    torch.manual_seed(seed); np.random.seed(seed)
    m = SSMClassifier(free).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    lossf = nn.CrossEntropyLoss()
    N = len(xtr); ntest = 2000 if SMOKE else 10000
    for step in range(STEPS):
        idx = torch.randint(0, N, (BATCH,))
        xb = xtr[idx].to(DEV); yb = ytr[idx].to(DEV)
        loss = lossf(m(xb), yb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % EVAL_EVERY == 0:
            print(f"    step {step+1:4d} loss {loss.item():.3f} test {test_acc(m, xte, yte, ntest):.4f}", flush=True)
    return m


def nparams(free):
    return sum(p.numel() for p in SSMClassifier(free).parameters())


def redteam(xtr):
    torch.manual_seed(7)
    Ac = torch.randn(3, 40, 8, dtype=torch.cfloat, device=DEV) * 0.5
    Xc = torch.randn(3, 40, 8, dtype=torch.cfloat, device=DEV)
    ds = (lin_scan(Ac, Xc) - seq_scan(Ac, Xc)).abs().max().item()
    assert ds < 1e-4, f"scan!=seq ({ds:.2e})"
    # matched RNG: FREE and CLAMPED share every non-theta param under the same seed
    torch.manual_seed(0); mf = SSMClassifier(True)
    torch.manual_seed(0); mc = SSMClassifier(False)
    assert torch.equal(mf.blocks[0].ssm.B_re, mc.blocks[0].ssm.B_re), "B_re RNG mismatch"
    assert torch.equal(mf.emb.weight, mc.emb.weight), "emb RNG mismatch"
    print(f"  [redteam] scan==seq ({ds:.1e}), FREE/CLAMPED share non-theta init -- OK", flush=True)


def main():
    print(f"device={DEV} smoke={SMOKE} seq-MNIST T={T_LEN} H={H} blocks={N_BLK} steps={STEPS} seeds={SEEDS}", flush=True)
    (xtr, ytr), (xte, yte) = load_data()
    print(f"  data: train {tuple(xtr.shape)} test {tuple(xte.shape)}", flush=True)
    redteam(xtr)
    pc = {"free": nparams(True), "clamped": nparams(False)}
    print(f"  params: free={pc['free']} clamped={pc['clamped']}", flush=True)

    res = {"config": {"T": T_LEN, "H": H, "d_ssm": D_SSM, "blocks": N_BLK, "steps": STEPS, "seeds": SEEDS,
                      "lr": LR, "wd": WD, "params": pc}, "test_acc": {"free": [], "clamped": []}}
    trained_free = None
    for free in (True, False):
        arm = "free" if free else "clamped"
        for s in SEEDS:
            t0 = time.time()
            print(f"  --- {arm} seed {s} ---", flush=True)
            m = train(free, s, xtr, ytr, xte, yte)
            acc = test_acc(m, xte, yte)
            res["test_acc"][arm].append(acc)
            print(f"  {arm} seed {s}: TEST ACC {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if free and trained_free is None:
                trained_free = m
            else:
                del m
                if DEV == "cuda":
                    torch.cuda.empty_cache()

    fa = float(np.mean(res["test_acc"]["free"])); ca = float(np.mean(res["test_acc"]["clamped"]))
    gap = fa - ca

    # profiler: within-model oscillation reliance of the trained FREE model (clamp theta->0 in place)
    for b in trained_free.blocks:
        b.ssm.clamp_theta = True
    fa_clamped = test_acc(trained_free, xte, yte)
    for b in trained_free.blocks:
        b.ssm.clamp_theta = False
    osc_reliance = fa - fa_clamped

    thr = 0.01
    verdict = ("OSCILLATION LOAD-BEARING" if gap >= thr else
               "OSCILLATION NOT NEEDED" if gap <= -thr else "TIE")
    res["result"] = {"free_acc": round(fa, 4), "clamped_acc": round(ca, 4), "free_minus_clamped": round(gap, 4),
                     "free_within_model_theta_clamped_acc": round(fa_clamped, 4),
                     "oscillation_reliance_within_model": round(osc_reliance, 4),
                     "verdict": verdict}
    out = HERE / ("smnist_ablation_smoke.json" if SMOKE else "smnist_ablation_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  FREE {fa:.4f}  CLAMPED {ca:.4f}  gap {gap:+.4f}  |  within-model osc-reliance {osc_reliance:+.4f}", flush=True)
    print("  ===== VERDICT:", verdict, "=====")
    print("  wrote", out.name, flush=True)


if __name__ == "__main__":
    main()

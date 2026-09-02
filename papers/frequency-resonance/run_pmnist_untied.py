# -*- coding: utf-8 -*-
"""run_pmnist_untied.py -- frozen by PREREG_pmnist_untied_2026_09_02.

THE CONFOUND IN THE FLAGSHIP KNOB. RESULT_pmnist_ablation's +0.312 rests on the phase clamp, which
removes rotation and, in the same move, ties each complex mode's two real channels to one
magnitude. On the ordered-copy task the untied real bank recovered nothing
(RESULT_untied_magnitudes_2026_09_02). This runner asks the same question on the flagship
benchmark: REAL2 is a real-eigenvalue bank with 2*D_SSM modes and 2*D_SSM independent magnitudes
-- the same real state width, the same parameter count, no rotation -- dropped into the same
three-block classifier. FREE and CLAMPED are trained with run_pmnist_ablation.py's own train()
so the receipt's numbers have to reproduce on this device before the new arm means anything.

  python run_pmnist_untied.py [--smoke]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_pmnist_ablation as R              # noqa: E402  (reads --smoke from sys.argv)

ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.protocol import Experiment        # noqa: E402

PREREG = HERE / "PREREG_pmnist_untied_2026_09_02.md"
ANCHORS = {"free": 0.9199, "clamped": 0.6067}     # pmnist_ablation_result.json, seed-mean test accuracy


class RealBank(nn.Module):
    """2d real modes, 2d independent magnitudes, no rotation; output width 2d like CLRU."""

    def __init__(self, d, d_in):
        super().__init__()
        r = torch.empty(2 * d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        self.B = nn.Parameter(torch.randn(2 * d, d_in) / math.sqrt(d_in))
        self.d = d

    def forward(self, x):
        B, T, _ = x.shape
        mag = torch.exp(-torch.exp(self.nu))
        gamma = torch.sqrt(torch.clamp(1 - mag ** 2, min=1e-6))
        u = torch.einsum("bti,di->btd", x, self.B) * gamma
        lam = mag.view(1, 1, 2 * self.d).expand(B, T, 2 * self.d)
        return R.lin_scan(lam, u)


class RealBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssm = RealBank(R.D_SSM, R.H)
        self.proj = nn.Linear(2 * R.D_SSM, R.H)
        self.norm1 = nn.LayerNorm(R.H)
        self.ff = nn.Sequential(nn.Linear(R.H, 2 * R.H), nn.GELU(), nn.Linear(2 * R.H, R.H))
        self.norm2 = nn.LayerNorm(R.H)

    def forward(self, x):
        x = self.norm1(x + self.proj(self.ssm(x)))
        x = self.norm2(x + self.ff(x))
        return x


class RealClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Linear(1, R.H)
        self.blocks = nn.ModuleList([RealBlock() for _ in range(R.N_BLK)])
        self.head = nn.Linear(R.H, 10)

    def forward(self, x):
        x = self.emb(x)
        for b in self.blocks:
            x = b(x)
        return self.head(x.mean(1))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def train_real2(seed, xtr, ytr, xte, yte):
    """run_pmnist_ablation.train, with the model swapped and nothing else changed."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = RealClassifier().to(R.DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=R.LR, weight_decay=R.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, R.STEPS)
    lossf = nn.CrossEntropyLoss()
    N = len(xtr); ntest = 2000 if R.SMOKE else 10000
    for step in range(R.STEPS):
        idx = torch.randint(0, N, (R.BATCH,))
        loss = lossf(m(xtr[idx].to(R.DEV)), ytr[idx].to(R.DEV))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        if (step + 1) % R.EVAL_EVERY == 0:
            print(f"    step {step+1:4d} loss {loss.item():.3f} test {R.test_acc(m, xte, yte, ntest):.4f}", flush=True)
    return m


def redteam():
    torch.manual_seed(0); mf = R.SSMClassifier(True)
    torch.manual_seed(0); mr = RealClassifier()
    assert nparams(mf) == nparams(mr), (nparams(mf), nparams(mr))
    x = torch.randn(2, 16, 1)
    assert mf(x).shape == mr(x).shape
    print(f"  [redteam] params free={nparams(mf)} real2={nparams(mr)} (equal); real state 2*D_SSM={2 * R.D_SSM} (equal)", flush=True)


def main() -> int:
    print(f"device={R.DEV} smoke={R.SMOKE} T={R.T_LEN} H={R.H} blocks={R.N_BLK} steps={R.STEPS} seeds={R.SEEDS}", flush=True)
    R.redteam(); redteam()
    (xtr, ytr), (xte, yte) = R.load_data()
    res = {"prereg": PREREG.name, "config": {"task": "permuted-MNIST", "perm_seed": R.PERM_SEED, "T": R.T_LEN, "H": R.H,
                                              "d_ssm": R.D_SSM, "blocks": R.N_BLK, "steps": R.STEPS, "seeds": R.SEEDS,
                                              "device": R.DEV, "torch": torch.__version__, "smoke": R.SMOKE},
           "params": {"free": R.nparams(True), "clamped": R.nparams(False), "real2": nparams(RealClassifier())},
           "test_acc": {"free": [], "clamped": [], "real2": []}}
    t0 = time.time()
    for arm in ("free", "clamped", "real2"):
        for s in R.SEEDS:
            m = R.train(arm == "free", s, xtr, ytr, xte, yte) if arm != "real2" else train_real2(s, xtr, ytr, xte, yte)
            acc = R.test_acc(m, xte, yte)
            res["test_acc"][arm].append(acc)
            print(f"  {arm} seed {s}: TEST ACC {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            del m
    fa, ca, ra = (float(np.mean(res["test_acc"][a])) for a in ("free", "clamped", "real2"))
    metrics = {"anchor_max_abs_dev": round(max(abs(fa - ANCHORS["free"]), abs(ca - ANCHORS["clamped"])), 4),
               "gap_free_minus_clamped": round(fa - ca, 4),
               "free_minus_real2": round(fa - ra, 4),
               "real2_minus_clamped": round(ra - ca, 4),
               "recovery_fraction": round((ra - ca) / (fa - ca), 4) if fa != ca else None}
    res["means"] = {"free": round(fa, 4), "clamped": round(ca, 4), "real2": round(ra, 4)}
    res["metrics"] = metrics
    v = Experiment(PREREG, repo_root=ROOT).score(metrics, smoke=R.SMOKE)
    res["verdict"], res["gates"] = v.verdict, v.gates
    out = HERE / ("pmnist_untied_smoke.json" if R.SMOKE else "pmnist_untied_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  metrics:", json.dumps(metrics))
    print(f"\n===== VERDICT: {res['verdict']} =====\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

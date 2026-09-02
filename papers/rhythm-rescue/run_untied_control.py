# -*- coding: utf-8 -*-
"""run_untied_control.py -- frozen by PREREG_untied_magnitudes_2026_09_02.

THE CONFOUND IN THE KNOB. The phase clamp (theta == 0) removes rotation -- and, in the same move,
ties each complex mode's two real channels to ONE magnitude, because both channels decay by the
same |lambda|. So "CLAMPED loses capacity" can be read two ways: rotation was load-bearing, or
timescale diversity was, and the clamp halved it. This runner adds the arm that separates them:
REAL2, a real-eigenvalue bank with 2D modes and 2D INDEPENDENT magnitudes -- the same real state
size (2D), the same parameter count (nu: 2D vs nu D + theta D; B: 2D x d_in vs B_re + B_im), no
rotation anywhere. If REAL2 recovers FREE's capacity, rotation was one way to buy timescale
diversity; if it does not, rotation is load-bearing beyond diversity.

FREE and CLAMPED are re-run as anchors with run_rhythm_rescue.py's own code (imported, not
copied) so the receipt's gap has to reproduce on this device before the new arm means anything.

  python run_untied_control.py [--smoke]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_rhythm_rescue as R                    # noqa: E402

ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.protocol import Experiment            # noqa: E402

SMOKE = "--smoke" in sys.argv
PREREG = HERE / "PREREG_untied_magnitudes_2026_09_02.md"
ANCHORS = {"free": 6.0, "clamped": 2.6667}      # rhythm_rescue_result.json, seed-mean kcap
STEPS = 200 if SMOKE else R.STEPS
SEEDS = [0] if SMOKE else R.SEEDS
KGRID = [1, 2, 4, 8] if SMOKE else R.KGRID
EVAL_N = 256 if SMOKE else 1024


class RealBank(nn.Module):
    """2D real-eigenvalue modes with 2D independent magnitudes; no rotation. Matches CLRU(D) in
    state size and parameter count."""

    def __init__(self, d, d_in):
        super().__init__()
        r = torch.empty(2 * d).uniform_(0.9, 0.999)
        self.nu = nn.Parameter(torch.log(-torch.log(r)))
        self.B = nn.Parameter(torch.randn(2 * d, d_in) / math.sqrt(d_in))

    def forward(self, x):
        mag = torch.exp(-torch.exp(self.nu))
        gamma = torch.sqrt(torch.clamp(1 - mag ** 2, min=1e-6))
        u = torch.einsum("bti,di->btd", x, self.B) * gamma
        h = torch.zeros(x.shape[0], mag.shape[0], device=x.device)
        outs = []
        for t in range(x.shape[1]):
            h = mag * h + u[:, t, :]
            outs.append(h)
        return torch.stack(outs, 1)                                    # (B,T,2d)


class Real2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(R.V + 1, R.D_IN)
        self.lru = RealBank(R.D, R.D_IN)
        self.read = nn.Sequential(nn.Linear(2 * R.D, R.D), nn.GELU(), nn.Linear(R.D, R.V))

    def forward(self, tok):
        return self.read(self.lru(self.emb(tok)))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def train(arm, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = (Real2Model() if arm == "real2" else R.Model(arm == "free")).to(R.DEV)
    opt = torch.optim.Adam(m.parameters(), lr=R.LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    for _ in range(STEPS):
        K = int(np.random.randint(1, R.KMAX + 1))
        inp, tgt = R.make_batch(R.BATCH, K)
        loss = lossf(m(inp).reshape(-1, R.V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


def kcap(accs):
    cap = 0
    for K in KGRID:
        if accs[K] >= R.ACC_THR:
            cap = K
    return cap


def redteam():
    torch.manual_seed(0); free = R.Model(True)
    torch.manual_seed(0); real2 = Real2Model()
    assert nparams(free) == nparams(real2), (nparams(free), nparams(real2))
    x = torch.randint(0, R.V, (2, 6))
    assert free.lru(free.emb(x)).shape == real2.lru(real2.emb(x)).shape
    print(f"  [redteam] params free={nparams(free)} real2={nparams(real2)} (equal); real state 2D={2 * R.D} (equal)", flush=True)


def main() -> int:
    print(f"device={R.DEV} smoke={SMOKE} D={R.D} steps={STEPS} seeds={SEEDS} kgrid={KGRID}", flush=True)
    redteam()
    res = {"prereg": PREREG.name, "config": {"D": R.D, "steps": STEPS, "seeds": SEEDS, "kgrid": KGRID,
                                              "acc_thr": R.ACC_THR, "device": R.DEV, "torch": torch.__version__,
                                              "smoke": SMOKE}, "arms": {}}
    for arm in ("free", "clamped", "real2"):
        res["arms"][arm] = {"params": nparams(Real2Model() if arm == "real2" else R.Model(arm == "free")),
                            "seeds": {}}
        for s in SEEDS:
            m = train(arm, s)
            accs = {K: round(R.eval_K(m, K, n=EVAL_N), 4) for K in KGRID}
            res["arms"][arm]["seeds"][str(s)] = {"acc": accs, "kcap": kcap(accs)}
            print(f"  {arm:8s} seed {s}: kcap={kcap(accs):2d} acc={accs}", flush=True)
            del m
        res["arms"][arm]["kcap_mean"] = float(np.mean([v["kcap"] for v in res["arms"][arm]["seeds"].values()]))
        res["arms"][arm]["acc_mean"] = {str(K): round(float(np.mean([v["acc"][K] for v in res["arms"][arm]["seeds"].values()])), 4) for K in KGRID}
    kf, kc, kr = (res["arms"][a]["kcap_mean"] for a in ("free", "clamped", "real2"))
    metrics = {
        "anchor_max_abs_dev": round(max(abs(kf - ANCHORS["free"]), abs(kc - ANCHORS["clamped"])), 4),
        "gap_free_minus_clamped": round(kf - kc, 4),
        "real2_minus_clamped": round(kr - kc, 4),
        "free_minus_real2": round(kf - kr, 4),
        "recovery_fraction": round((kr - kc) / (kf - kc), 4) if kf != kc else None,
    }
    res["metrics"] = metrics
    v = Experiment(PREREG, repo_root=ROOT).score(metrics, smoke=SMOKE)
    res["verdict"], res["gates"] = v.verdict, v.gates
    out = HERE / ("untied_control_smoke.json" if SMOKE else "untied_control_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  metrics:", json.dumps(metrics))
    print(f"\n===== VERDICT: {res['verdict']} =====\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

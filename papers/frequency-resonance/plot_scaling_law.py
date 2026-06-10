# -*- coding: utf-8 -*-
"""
plot_scaling_law.py — render theta*(D) and the theta* x window invariant from
scaling_sweep_result.json. Left: optimal frequency vs delay, with the 1/W prediction overlaid.
Right: the product theta* x W per delay (flat = the law holds). Run after the scaling sweep.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
src = HERE / ("scaling_sweep_smoke.json" if "--smoke" in sys.argv else "scaling_sweep_result.json")
r = json.loads(src.read_text(encoding="utf-8"))
g = r["gate"]
Ds = g["delays_used"]
tstar = g["theta_star_frac_by_delay"]          # in units of pi
Ws = g["windows"]
prod_pi = g["product_mean_over_pi"]            # mean product in units of pi (rad)
verdict = g["verdict"]
head = verdict.split("—")[0].split("--")[0].strip() if verdict else "?"

BG, FG, GREEN, CYAN, RED = "#0a0a0a", "#e6e6e6", "#00ff66", "#00e0ff", "#ff3b3b"
plt.rcParams.update({"font.family": "monospace", "font.size": 11,
                     "figure.facecolor": BG, "axes.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "axes.edgecolor": "#444"})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

# Left: theta*(D) with the 1/W law overlay
axL.plot(Ds, tstar, "o", color=GREEN, ms=11, mec=BG, zorder=3, label="measured θ* (optimum)")
if prod_pi:
    xs = [min(Ds) + i * (max(Ds) - min(Ds)) / 60 for i in range(61)] if len(Ds) > 1 else Ds
    # predicted theta*/pi = const / W, with W ~ D + (mean kcap*). Use measured W mapping per point;
    # for a smooth curve approximate W ~ D + meanslack where meanslack = mean(W-D).
    slack = sum(w - d for w, d in zip(Ws, Ds)) / len(Ds)
    pred = [prod_pi / (x + slack) for x in xs]
    axL.plot(xs, pred, "--", color=CYAN, lw=1.8, zorder=2,
             label=f"1/W law:  θ*·W = {prod_pi:.2f}π")
axL.set_xlabel("inserted delay D   (longer hold ->)")
axL.set_ylabel("optimal frequency  θ* / π")
axL.set_title("optimum slides down as memory is held longer", color=FG, fontsize=11.5)
axL.grid(True, color="#222", lw=0.6); axL.legend(facecolor=BG, edgecolor="#444",
                                                 labelcolor=FG, fontsize=9.5)

# Right: the invariant theta* x W per delay
prods_pi = [t * w for t, w in zip(tstar, Ws)]   # in units of pi (since tstar already /pi)
axR.bar([str(d) for d in Ds], prods_pi, color=RED, alpha=0.8, zorder=3)
if prod_pi:
    axR.axhline(prod_pi, color=CYAN, ls="--", lw=1.8, zorder=4, label=f"mean {prod_pi:.2f}π")
axR.set_xlabel("delay D"); axR.set_ylabel("θ* × W   (cycles-fraction × π, rad)")
axR.set_title(f"the invariant (flat = law holds) · CV {g['product_cv']}", color=FG, fontsize=11.5)
axR.grid(True, axis="y", color="#222", lw=0.6); axR.legend(facecolor=BG, edgecolor="#444",
                                                           labelcolor=FG, fontsize=9.5)

fig.suptitle(f"θ* × window = const?   ->  {head}", color=FG, fontsize=13)
fig.text(0.012, 0.012, "styxx · papers/frequency-resonance · pre-registered 2026-06-04 · in-silico LRU, ordered copy + delay, 3 seeds",
         color="#777", fontsize=7.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
out = HERE / ("scaling_law_smoke.png" if "--smoke" in sys.argv else "scaling_law_curve.png")
fig.savefig(out, dpi=150, facecolor=BG)
print("verdict:", verdict)
print("wrote", out.name)

# -*- coding: utf-8 -*-
"""
plot_frequency_sweep.py — render kcap(theta) from frequency_sweep_result.json.

The shape of this curve IS the result: a rising line to the Nyquist edge = "higher frequency,
greater capacity" (monotonic); an interior hump = a resonant optimum (best band, higher hurts).
Dark, monospace, styxx-brand accent. Run after the sweep completes.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
src = HERE / ("frequency_sweep_smoke.json" if "--smoke" in sys.argv else "frequency_sweep_result.json")
res = json.loads(src.read_text(encoding="utf-8"))

fracs = res["config"]["theta_fracs"]
kcaps = [res["sweep"][f"{f:.4f}"]["mean_kcap"] for f in fracs]
g = res["gate"]
free_kcap = g["free_kcap"]
theta_star = g["theta_star_over_pi"]
verdict = g["verdict"]
head = verdict.split("—")[0].split("--")[0].strip() if verdict else "?"

BG, FG, GREEN, CYAN, RED = "#0a0a0a", "#e6e6e6", "#00ff66", "#00e0ff", "#ff3b3b"
plt.rcParams.update({"font.family": "monospace", "font.size": 11,
                     "figure.facecolor": BG, "axes.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": "#444"})

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(fracs, kcaps, "-o", color=GREEN, lw=2.2, ms=7, mfc=GREEN, mec=BG, zorder=3,
        label="single fixed frequency theta")
# FREE reference (learned spectrum of frequencies)
ax.axhline(free_kcap, color=CYAN, ls="--", lw=1.8, zorder=2,
           label=f"FREE: learned spectrum (kcap {free_kcap:.1f})")
# mark the peak
ipk = kcaps.index(max(kcaps))
ax.scatter([fracs[ipk]], [kcaps[ipk]], s=170, facecolor="none", edgecolor=RED, lw=2.4, zorder=4)
ax.annotate(f"theta*={theta_star:.3f}pi", (fracs[ipk], kcaps[ipk]),
            textcoords="offset points", xytext=(8, 10), color=RED, fontsize=10)

ax.set_xlabel("oscillation frequency  theta / pi   (0 = no rhythm,  1 = Nyquist)")
ax.set_ylabel("ordered-memory capacity  kcap  (items @ acc>=0.80)")
ax.set_title(f"Does higher frequency mean greater capacity?   ->  {head}",
             color=FG, fontsize=12.5, pad=12)
ax.set_ylim(0, max(max(kcaps), free_kcap) + 1.5)
ax.grid(True, color="#222", lw=0.6)
ax.legend(facecolor=BG, edgecolor="#444", labelcolor=FG, loc="upper right", fontsize=9.5)
fig.text(0.012, 0.012,
         "styxx · papers/frequency-resonance · pre-registered 2026-06-04 · in-silico LRU, ordered copy, 3 seeds",
         color="#777", fontsize=7.5)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = HERE / ("frequency_sweep_smoke.png" if "--smoke" in sys.argv else "frequency_resonance_curve.png")
fig.savefig(out, dpi=150, facecolor=BG)
print("verdict:", verdict)
print("spectrum gate:", g["spectrum_gate"], "|", g["spectrum_note"])
print("wrote", out.name)

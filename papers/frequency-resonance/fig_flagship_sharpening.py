# -*- coding: utf-8 -*-
"""Capstone flagship figure: oscillation is the long-range mechanism.
Numbers from smnist_ablation_result.json + pmnist_ablation_result.json (both OATH-certified)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#140e1a"; PANEL = "#1e1526"; LILAC = "#cbb6e8"; DIM = "#8a7aa0"
CYAN = "#6fd3d3"; AMBER = "#e8c46f"; RED = "#f08a8a"; GRID = "#2c2036"
plt.rcParams.update({"font.family": "monospace", "text.color": LILAC, "axes.labelcolor": LILAC,
                     "xtick.color": LILAC, "ytick.color": LILAC, "axes.edgecolor": GRID,
                     "figure.facecolor": BG, "axes.facecolor": PANEL})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.5), gridspec_kw={"width_ratios": [1, 1.05]})
fig.suptitle("Oscillation is the long-range mechanism in a state-space model",
             fontsize=15, color=LILAC, weight="bold", y=0.98)
fig.text(0.5, 0.915, "single-knob (θ) oscillation-vs-decay ablation · matched-param · the test LinOSS never published · OATH-certified",
         ha="center", fontsize=9, color=DIM)

# ---- LEFT: sharpening -- grouped bars, gap explodes when locality is removed ----
groups = ["sequential MNIST\n(locality available)", "permuted MNIST\n(locality destroyed)"]
free = [0.9837, 0.9195]
clamp = [0.9428, 0.6073]
gaps = [0.0408, 0.3122]
x = np.arange(2); w = 0.34
axL.bar(x - w/2, free, w, color=CYAN, label="FREE (oscillation)", zorder=4)
axL.bar(x + w/2, clamp, w, color=AMBER, label="CLAMPED (decay)", zorder=4)
for xi in x:
    axL.text(xi - w/2, free[xi] + 0.012, f"{free[xi]*100:.0f}%", ha="center", fontsize=9.5, color=CYAN, weight="bold")
    axL.text(xi + w/2, clamp[xi] + 0.012, f"{clamp[xi]*100:.0f}%", ha="center", fontsize=9.5, color=AMBER, weight="bold")
    # gap bracket
    top = max(free[xi], clamp[xi]) + 0.07
    axL.plot([xi - w/2, xi + w/2], [top, top], color=LILAC, lw=1.2)
    axL.plot([xi - w/2, xi - w/2], [free[xi]+0.03, top], color=LILAC, lw=1.0)
    axL.plot([xi + w/2, xi + w/2], [clamp[xi]+0.03, top], color=LILAC, lw=1.0)
    lbl = f"gap +{gaps[xi]*100:.1f} pts" + ("  (7.6× wider)" if xi == 1 else "")
    axL.text(xi, top + 0.015, lbl, ha="center", fontsize=10,
             color=(RED if xi == 1 else LILAC), weight="bold")
axL.set_xticks(x); axL.set_xticklabels(groups, fontsize=9.5)
axL.set_ylim(0, 1.14); axL.set_ylabel("test accuracy", fontsize=10)
axL.legend(loc="lower left", facecolor=PANEL, edgecolor=GRID, fontsize=9, framealpha=0.95)
axL.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.5)
for s in axL.spines.values():
    s.set_color(GRID)

# ---- RIGHT: pMNIST training curves -- the decay model cannot solve it ----
steps = [1000, 2000, 3000, 4000]
pf = [0.6485, 0.8368, 0.9030, 0.9195]     # seed-avg FREE
pc = [0.3071, 0.4693, 0.5813, 0.6073]     # seed-avg CLAMPED
axR.plot(steps, pf, "-o", color=CYAN, lw=2.6, ms=9, label="FREE (oscillation) → 92%", zorder=5)
axR.plot(steps, pc, "-o", color=AMBER, lw=2.4, ms=8, label="CLAMPED (decay) → plateaus 61%", zorder=5)
axR.fill_between(steps, pc, pf, color=CYAN, alpha=0.08)
axR.axhline(0.10, color=DIM, lw=1.0, ls=":", alpha=0.7)
axR.text(4000, 0.115, "chance", color=DIM, fontsize=8, ha="right")
axR.text(4000, pf[-1]+0.01, "92.0%", ha="right", fontsize=9.5, color=CYAN, weight="bold")
axR.text(4000, pc[-1]-0.05, "60.7%", ha="right", fontsize=9.5, color=AMBER, weight="bold")
axR.annotate("a decay SSM of equal budget\nsimply cannot do permuted MNIST", xy=(3000, 0.58),
             xytext=(2300, 0.30), fontsize=9, color=AMBER, ha="center",
             arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
axR.set_xlabel("training step (permuted MNIST)", fontsize=10); axR.set_ylabel("test accuracy", fontsize=10)
axR.set_ylim(0, 1.0); axR.set_xlim(700, 4300)
axR.set_title("within-model: clamp the trained model's θ→0 → 10% (chance)", fontsize=10, color=LILAC, pad=8)
axR.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, fontsize=9, framealpha=0.95)
axR.grid(True, color=GRID, lw=0.5, alpha=0.5)
for s in axR.spines.values():
    s.set_color(GRID)

fig.text(0.5, 0.015, "styxx · frequency-resonance · 2026-07-23   —   remove locality and the oscillation IS the long-range model; a decay SSM of equal budget cannot substitute",
         ha="center", fontsize=8.3, color=DIM)
plt.subplots_adjust(left=0.07, right=0.975, top=0.85, bottom=0.13, wspace=0.22)
plt.savefig("fig_flagship_sharpening.png", dpi=150, facecolor=BG)
print("wrote fig_flagship_sharpening.png")

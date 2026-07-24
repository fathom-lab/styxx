# -*- coding: utf-8 -*-
"""Causal map of oscillation in state-space models. All numbers from the OATH-certified receipts:
entrainment_result.json, entrain_rich_result.json, nested_capacity_result.json."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BG = "#140e1a"; PANEL = "#1e1526"; LILAC = "#cbb6e8"; DIM = "#8a7aa0"
CYAN = "#6fd3d3"; GREEN = "#7ee787"; AMBER = "#e8c46f"; RED = "#f08a8a"; GRID = "#2c2036"
plt.rcParams.update({"font.family": "monospace", "text.color": LILAC, "axes.labelcolor": LILAC,
                     "xtick.color": LILAC, "ytick.color": LILAC, "axes.edgecolor": GRID,
                     "figure.facecolor": BG, "axes.facecolor": PANEL})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
fig.suptitle("The adaptive-frequency ceiling in state-space models",
             fontsize=15, color=LILAC, weight="bold", y=0.985)
fig.text(0.5, 0.925, "three controlled causal ablations · frozen gates · firing positive controls · OATH-certified",
         ha="center", fontsize=9.5, color=DIM)

# ---- LEFT: adaptive-frequency dose-response (advantage over static vs detector complexity) ----
x = [0, 1, 2, 3]
xlabels = ["static\n(no detector)", "single\nprojection", "windowed\nconv", "deep conv\n+ spread"]
d8 = [0.0, 0.040, 0.085, 0.002]   # ENTRAIN D=8: KILL, WEAK(best), KILL -- NON-MONOTONE
d4 = [0.0, 0.072, 0.129, 0.018]   # D=4 (mode-starved): win at windowed conv, collapse at deep
orc8, orc4 = 0.170, 0.104         # oracle ceilings (adaptation prize)

axL.axhspan(0, 0.05, color=RED, alpha=0.07)
axL.axhspan(0.05, 0.10, color=AMBER, alpha=0.08)
axL.axhspan(0.10, 0.20, color=GREEN, alpha=0.07)
axL.axhline(0.10, color=GREEN, lw=1.2, ls="--", alpha=0.8)
axL.text(2.02, 0.104, "GREENLIGHT bar (+0.10)", color=GREEN, fontsize=8, va="center", ha="right")
axL.axhline(orc8, color=CYAN, lw=1.1, ls=":", alpha=0.8)
axL.text(1.15, orc8 + 0.004, "oracle ceiling, D=8 (adaptation prize, +0.17)", color=CYAN, fontsize=8, ha="center")

axL.plot(x, d8, "-o", color=LILAC, lw=2.4, ms=9, label="D=8 (comfortable budget)", zorder=5)
axL.plot(x, d4, "-o", color=AMBER, lw=2.0, ms=8, label="D=4 (mode-starved)", zorder=5)
for xi, yi, tag in zip(x, d8, ["", "KILL", "WEAK", "KILL"]):
    if tag:
        off = (0, 12) if xi != 3 else (0, 13)
        axL.annotate(f"{tag}\n+{yi:.3f}", (xi, yi), textcoords="offset points", xytext=off,
                     ha="center", fontsize=8.5, color=LILAC, weight="bold")
axL.annotate("GREENLIGHT\n+0.129", (2, 0.129), textcoords="offset points", xytext=(-4, -30),
             ha="center", fontsize=8.5, color=GREEN, weight="bold")
axL.annotate("collapse:\n8x params,\nno gain", (3, 0.002), textcoords="offset points", xytext=(2, 30),
             ha="center", fontsize=8, color=RED)
axL.set_xticks(x); axL.set_xticklabels(xlabels, fontsize=9)
axL.set_ylim(-0.01, 0.195); axL.set_xlim(-0.25, 3.4)
axL.set_ylabel("advantage over static bank (mean-acc)", fontsize=10)
axL.set_title("adaptive frequency: non-monotone in detector complexity", fontsize=11, color=LILAC, pad=8)
axL.text(0.02, -0.008, "capture of the oracle prize:  23% → 50% → 1%  (D=8, non-monotone)   ·   clean win only at D=4",
         transform=axL.transData, fontsize=7.5, color=DIM)
axL.legend(loc="upper left", facecolor=PANEL, edgecolor=GRID, fontsize=8.5, framealpha=0.9)
for s in axL.spines.values():
    s.set_color(GRID)
axL.grid(True, color=GRID, lw=0.5, alpha=0.5)

# ---- RIGHT: the three primitives, advantage vs their positive-control ceiling ----
prims = ["NEST\ntheta-gamma", "ENTRAIN\nsingle-proj", "ENTRAIN-RICH\nwindowed conv"]
adv = [0.009, 0.040, 0.085]
ceil = [0.183, 0.170, 0.170]           # positive-control ceilings (wide headroom / oracle)
verd = ["KILL", "KILL", "WEAK"]
vcol = [RED, RED, AMBER]
y = range(len(prims))
axR.barh(list(y), ceil, color=CYAN, alpha=0.16, height=0.6, label="positive-control ceiling")
for yi, (a, c, v, col) in enumerate(zip(adv, ceil, verd, vcol)):
    axR.barh(yi, a, color=col, height=0.6, zorder=4)
    axR.text(a + 0.004, yi, f"+{a:.3f}  {v}", va="center", fontsize=9, color=col, weight="bold")
    axR.text(c - 0.004, yi + 0.34, f"ceiling +{c:.3f}", va="center", ha="right", fontsize=7.5, color=DIM)
axR.axvline(0.10, color=GREEN, lw=1.1, ls="--", alpha=0.7)
axR.set_yticks(list(y)); axR.set_yticklabels(prims, fontsize=9)
axR.set_xlim(0, 0.205); axR.set_xlabel("advantage captured (mean-acc)", fontsize=10)
axR.set_title("what beats a static / flat bank? (the honest map)", fontsize=11, color=LILAC, pad=8)
axR.text(0.005, 2.92, "attention dominates raw capacity throughout (mean 0.98 vs ≤0.58 recurrent)",
         fontsize=8, color=DIM)
for s in axR.spines.values():
    s.set_color(GRID)
axR.grid(True, axis="x", color=GRID, lw=0.5, alpha=0.5)
axR.invert_yaxis()

fig.text(0.5, 0.012, "styxx · frequency-resonance · 2026-07-23   —   oscillation adapts where the budget is tight; a flat bank already captures the rest",
         ha="center", fontsize=8.5, color=DIM)
plt.subplots_adjust(left=0.075, right=0.975, top=0.86, bottom=0.11, wspace=0.28)
plt.savefig("fig_adaptation_map.png", dpi=150, facecolor=BG)
print("wrote fig_adaptation_map.png")

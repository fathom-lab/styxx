# -*- coding: utf-8 -*-
"""Real-task flagship figure: oscillation is causally load-bearing on sequential MNIST.
All numbers from smnist_ablation_result.json (OATH-certified)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#140e1a"; PANEL = "#1e1526"; LILAC = "#cbb6e8"; DIM = "#8a7aa0"
CYAN = "#6fd3d3"; GREEN = "#7ee787"; AMBER = "#e8c46f"; RED = "#f08a8a"; GRID = "#2c2036"
plt.rcParams.update({"font.family": "monospace", "text.color": LILAC, "axes.labelcolor": LILAC,
                     "xtick.color": LILAC, "ytick.color": LILAC, "axes.edgecolor": GRID,
                     "figure.facecolor": BG, "axes.facecolor": PANEL})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
fig.suptitle("Oscillation is causally load-bearing on sequential MNIST",
             fontsize=15, color=LILAC, weight="bold", y=0.98)
fig.text(0.5, 0.915, "the oscillation-vs-decay ablation LinOSS never published  ·  single knob (θ)  ·  matched-param  ·  OATH-certified",
         ha="center", fontsize=9, color=DIM)

# ---- LEFT: test-accuracy training curves (seed-averaged) ----
steps = [1000, 2000, 3000, 4000]
free = [0.9134, 0.9578, 0.9745, 0.9837]     # seed-avg FREE (oscillatory)
clamp = [0.8361, 0.8984, 0.9304, 0.9429]    # seed-avg CLAMPED (decay)
axL.plot(steps, free, "-o", color=CYAN, lw=2.6, ms=9, label="FREE (θ learnable → oscillation)", zorder=5)
axL.plot(steps, clamp, "-o", color=AMBER, lw=2.4, ms=8, label="CLAMPED (θ≡0 → pure decay)", zorder=5)
axL.fill_between(steps, clamp, free, color=CYAN, alpha=0.08)
axL.annotate("+4.1 pts\n(both seeds)", (4000, (free[-1]+clamp[-1])/2), textcoords="offset points",
             xytext=(-58, -4), ha="center", fontsize=9, color=LILAC, weight="bold")
axL.text(4000, free[-1]+0.004, "98.4%", ha="right", fontsize=9, color=CYAN, weight="bold")
axL.text(4000, clamp[-1]-0.012, "94.3%", ha="right", fontsize=9, color=AMBER, weight="bold")
axL.set_xlabel("training step", fontsize=10); axL.set_ylabel("test accuracy", fontsize=10)
axL.set_title("matched-param, RNG-matched: only θ differs", fontsize=11, color=LILAC, pad=8)
axL.set_ylim(0.80, 1.0); axL.set_xlim(700, 4300)
axL.legend(loc="lower right", facecolor=PANEL, edgecolor=GRID, fontsize=9, framealpha=0.95)
axL.grid(True, color=GRID, lw=0.5, alpha=0.5)
for s in axL.spines.values():
    s.set_color(GRID)

# ---- RIGHT: the causal collapse (clamp the trained model's oscillation off) ----
labels = ["FREE\ntrained", "CLAMPED\ntrained", "FREE, then\nθ→0 in place"]
vals = [0.9837, 0.9428, 0.0982]
cols = [CYAN, AMBER, RED]
xb = range(len(vals))
axR.axhline(0.10, color=DIM, lw=1.0, ls=":", alpha=0.8)
axR.text(0.02, 0.12, "chance (10%)", color=DIM, fontsize=8, ha="left")
bars = axR.bar(list(xb), vals, color=cols, width=0.62, zorder=4)
for xi, v in zip(xb, vals):
    axR.text(xi, v + 0.02, f"{v*100:.1f}%", ha="center", fontsize=10.5, color=cols[xi], weight="bold")
axR.annotate("", xy=(2, 0.15), xytext=(0.35, 0.95),
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.8, connectionstyle="arc3,rad=-0.28"))
axR.text(2.46, 0.55, "remove the oscillation\nfrom the TRAINED model\n→ collapse to chance",
         color=RED, fontsize=8.5, ha="right", weight="bold")
axR.set_xticks(list(xb)); axR.set_xticklabels(labels, fontsize=9)
axR.set_ylim(0, 1.06); axR.set_ylabel("test accuracy", fontsize=10)
axR.set_title("within-model reliance: +88.5 pts", fontsize=11, color=LILAC, pad=8)
axR.grid(True, axis="y", color=GRID, lw=0.5, alpha=0.5)
for s in axR.spines.values():
    s.set_color(GRID)

fig.text(0.5, 0.015, "styxx · frequency-resonance · 2026-07-23   —   a trained oscillatory SSM routes its computation through the rotation; clamp it and nothing is left",
         ha="center", fontsize=8.5, color=DIM)
plt.subplots_adjust(left=0.07, right=0.975, top=0.85, bottom=0.13, wspace=0.22)
plt.savefig("fig_smnist_flagship.png", dpi=150, facecolor=BG)
print("wrote fig_smnist_flagship.png")

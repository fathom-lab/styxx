# -*- coding: utf-8 -*-
"""plot_obfuscation.py — probe collapses under the representation attack; grounded_honesty is flat."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "obfuscation_robustness_result.json")))
sw = r["probe_steering_sweep"]
alphas = sorted(float(k) for k in sw)
probe = [sw[f"{a}"] if f"{a}" in sw else sw[str(a)] for a in alphas]
g = r["auroc"]["grounded_clean"]

BG, FG, RED, GREEN, CY = "#0a0a0a", "#e6e6e6", "#ff3b3b", "#00ff66", "#00e0ff"
plt.rcParams.update({"font.family": "monospace", "figure.facecolor": BG, "axes.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "axes.edgecolor": "#444"})
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(alphas, probe, "-o", color=RED, lw=2.4, ms=8, label="static activation probe (signature)")
ax.axhline(g, color=GREEN, lw=2.4, ls="-", label=f"styxx grounded_honesty (consistency) = {g:.2f}, flat")
ax.axhline(0.5, color="#555", lw=1, ls=":")
ax.text(0.02, 0.515, "chance", color="#888", fontsize=8)
ax.annotate("probe collapses\nat full steering", (1.0, probe[-1]), textcoords="offset points",
            xytext=(-12, 28), color=RED, fontsize=9, ha="right",
            arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xlabel("representation attack strength  α  (steer deceptive → honest manifold)")
ax.set_ylabel("deception-detection AUROC")
ax.set_title("the attack that defeats the signature can't touch the consistency check",
             color=FG, fontsize=11.5)
ax.set_ylim(0.45, 1.04); ax.set_xticks(alphas)
ax.grid(True, color="#222", lw=0.6); ax.legend(facecolor=BG, edgecolor="#444", labelcolor=FG, loc="center left")
fig.text(0.012, 0.012, "styxx · papers/consistency-robustness · Qwen2.5-3B, 80 held-out · grounded_honesty reads "
         "resampled belief, not residuals → immune by construction", color="#777", fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = os.path.join(HERE, "obfuscation_robustness_curve.png")
fig.savefig(out, dpi=150, facecolor=BG)
print("probe sweep:", [round(p, 3) for p in probe], "| grounded flat:", round(g, 3))
print("wrote", os.path.basename(out))

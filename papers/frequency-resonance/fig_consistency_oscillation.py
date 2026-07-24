# -*- coding: utf-8 -*-
"""Figure for RESULT_consistency_oscillation: decay reads inputs and compares ADJACENT facts as well
as oscillation, but collapses to chance the moment the fact to stay consistent with is DISTANT."""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
r = json.load(open(HERE / "consistency_oscillation_result.json"))["result"]

conds = ["claim-only\n(lone input)", "compare\nADJACENT", "compare\nDISTANT (255)"]
free = [r["claimonly_free"], r["cmp_adj_free"], r["cmp_long_free"]]
clamp = [r["claimonly_clamped"], r["cmp_adj_clamped"], r["cmp_long_clamped"]]

AUB, LIL, INK, MUT = "#2E1A2F", "#B07FD1", "#EDE6F0", "#6E5A70"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": AUB, "axes.facecolor": AUB})
fig, ax = plt.subplots(figsize=(7.6, 4.6))
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w / 2, free, w, label="FREE  (oscillation on)", color=LIL, edgecolor="none")
b2 = ax.bar(x + w / 2, clamp, w, label="CLAMPED  (decay, θ≡0)", color=MUT, edgecolor="none")
ax.axhline(0.5, ls=(0, (4, 3)), lw=1, color=INK, alpha=0.5)
ax.text(2.46, 0.515, "chance", color=INK, alpha=0.6, fontsize=8, ha="right")

for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center",
                va="bottom", color=INK, fontsize=9)

ax.annotate("one phase knob:\nchance → perfect", xy=(2 - w / 2, 1.0), xytext=(1.15, 0.78),
            color=LIL, fontsize=9.5, ha="left",
            arrowprops=dict(arrowstyle="->", color=LIL, lw=1.4))
ax.set_xticks(x); ax.set_xticklabels(conds, color=INK, fontsize=10)
ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(colors=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUT)
ax.set_ylabel("test accuracy", color=INK, fontsize=10)
ax.set_title("Long-range consistency-checking requires the oscillatory channel",
             color=INK, fontsize=12.5, pad=12, loc="left")
ax.text(0, 1.075, "decay reads inputs and compares adjacent facts as well as oscillation — "
        "but falls to chance when the fact is distant", color=MUT, fontsize=8.6,
        transform=ax.transData)
leg = ax.legend(loc="lower left", frameon=False, fontsize=9.2, labelcolor=INK, ncol=1,
                bbox_to_anchor=(0.0, -0.02))
plt.tight_layout()
out = HERE / "fig_consistency_oscillation.png"
plt.savefig(out, dpi=170, facecolor=AUB)
print("wrote", out.name)

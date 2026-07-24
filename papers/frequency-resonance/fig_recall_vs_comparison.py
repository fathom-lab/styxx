# -*- coding: utf-8 -*-
"""The dissociation: a pure-decay channel REMEMBERS a fact across any distance (recall flat at 1.0) but
cannot COMPARE two facts across distance (consistency solve-rate collapses). Oscillation does both."""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
rec = json.load(open(HERE / "recall_horizon_result.json"))
cmp = json.load(open(HERE / "consistency_horizon_v2_result.json"))
rg = np.array(rec["config"]["gaps"], float); rp = np.array(rec["result"]["clamped_solve_rate_by_gap"], float)
cg = np.array(cmp["config"]["gaps"], float); cp = np.array(cmp["result"]["clamped_solve_rate_by_gap"], float)
Hc = cmp["result"]["half_horizon_gap"]

AUB, LIL, INK, MUT, WARN = "#2E1A2F", "#B07FD1", "#EDE6F0", "#6E5A70", "#E8A0C0"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": AUB, "axes.facecolor": AUB})
fig, ax = plt.subplots(figsize=(8.2, 4.8))

ax.plot(rg, rp, "-o", color=LIL, lw=2.4, ms=6, label="decay · REMEMBER one fact (recall)")
ax.plot(cg, cp, "-o", color=WARN, lw=2.4, ms=6, label="decay · COMPARE two facts (consistency)")
ax.axhline(0.5, ls=(0, (4, 3)), lw=1, color=INK, alpha=0.35)
ax.axvline(Hc, ls=(0, (2, 2)), lw=1.2, color=MUT)
ax.text(Hc * 1.05, 0.055, f"comparison\nhorizon ≈ {Hc:.0f}", color=INK, fontsize=8.4)
ax.text(120, 1.03, "recall: no horizon —\ndecay carries a fact at any distance", color=LIL, fontsize=8.8)
ax.annotate("but cannot relate it\nto a later one", xy=(96, 0.0), xytext=(150, 0.30),
            color=WARN, fontsize=8.8, ha="left", arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3))

ax.set_xscale("log", base=2)
xt = sorted(set(list(rg) + list(cg)))
ax.set_xticks(xt); ax.set_xticklabels([f"{int(g)}" for g in xt], color=INK, fontsize=8)
ax.set_ylim(-0.06, 1.14); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(colors=INK)
ax.set_xlabel("distance between the fact and the probe (positions)", color=INK, fontsize=10)
ax.set_ylabel("decay solve rate", color=INK, fontsize=10)
ax.set_title("Oscillation is required to COMPARE across distance — not to REMEMBER",
             color=INK, fontsize=12.5, loc="left", pad=10)
ax.text(1, 1.075, "a pure-decay channel stores a fact perfectly at any range, "
        "but its odds of relating it to a later one decay with distance",
        color=MUT, fontsize=8.4)
ax.legend(loc="center left", frameon=False, fontsize=9.2, labelcolor=INK, bbox_to_anchor=(0.02, 0.42))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUT)
plt.tight_layout()
plt.savefig(HERE / "fig_recall_vs_comparison.png", dpi=170, facecolor=AUB)
print("wrote fig_recall_vs_comparison.png")

# -*- coding: utf-8 -*-
"""make_map_figure.py — the two-axis demarcation map: where rhythm loses (capacity) and where it wins
(robustness, timing). From committed result JSONs. Noir house style. CPU-only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BG, GREEN, CYAN, GRAY, RED = "#000000", "#00FF00", "#00FFFF", "#6a6a6a", "#ff5555"
plt.rcParams.update({"font.family": "monospace", "text.color": GREEN, "axes.labelcolor": GREEN,
                     "xtick.color": GREEN, "ytick.color": GREEN, "axes.edgecolor": "#00aa00"})

nec = json.load(open(HERE / "necessity_attention_result.json"))
noi = json.load(open(HERE / "noise_result.json"))
tim = json.load(open(HERE / "timing_result.json"))

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.7), facecolor=BG)
for a in ax:
    a.set_facecolor(BG)
    for s in a.spines.values():
        s.set_color("#00aa00")

# --- A: CAPACITY — rhythm dominated by attention ---
bars = [("LRU\nclamped", nec["arms"]["lru_clamped"]["mean_kcap"], GRAY),
        ("LRU\nfree", nec["arms"]["lru_free"]["mean_kcap"], CYAN),
        ("Transformer", nec["arms"]["transformer"]["mean_kcap"], GREEN)]
ax[0].bar(range(3), [v for _, v, _ in bars], color=[c for _, _, c in bars], edgecolor="#00aa00", width=0.62)
for i, (_, v, _) in enumerate(bars):
    ax[0].text(i, v + 0.3, f"{v:.1f}", ha="center", color=GREEN, fontsize=12, fontweight="bold")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([n for n, _, _ in bars], fontsize=9)
ax[0].set_ylabel("memory capacity (kcap)"); ax[0].set_ylim(0, 17)
ax[0].set_title("CAPACITY · rhythm LOSES\nattention 15.3 >> oscillation 6.0", color=RED, fontsize=10.5, pad=10)

# --- B: ROBUSTNESS — phase code outlasts decay under noise ---
sig = sorted(float(k) for k in noi["by_sigma"].keys())
free = [noi["by_sigma"][f"{s:.4g}" if f"{s:.4g}" in noi["by_sigma"] else str(s)]["free"] for s in sig]
# robust key lookup (json keys were str(float))
def getrow(d, s):
    for k in d:
        if abs(float(k) - s) < 1e-9:
            return d[k]
    return None
free = [getrow(noi["by_sigma"], s)["free"] for s in sig]
clamp = [getrow(noi["by_sigma"], s)["clamped"] for s in sig]
ax[1].plot(sig, free, "-o", color=CYAN, mec=CYAN, mfc=CYAN, lw=2, ms=6, label="LRU free (osc)  −33%")
ax[1].plot(sig, clamp, "-o", color=GRAY, mec=GRAY, mfc=GRAY, lw=2, ms=6, label="LRU clamped     −50%")
ax[1].set_xlabel("state noise  σ"); ax[1].set_ylabel("capacity (kcap)")
ax[1].set_ylim(0, 7)
ax[1].legend(fontsize=8, facecolor=BG, edgecolor="#00aa00", labelcolor=GREEN, loc="upper right")
ax[1].set_title("ROBUSTNESS · rhythm WINS\nphase code degrades gracefully (control-cleared)",
                color=GREEN, fontsize=10.5, pad=10)

# --- C: TIMING — native mechanism; rhythm = attention >> decay ---
ps = [int(k) for k in tim["by_arch"]["lru_free"].keys()]; ps.sort()
fr = [tim["by_arch"]["lru_free"][str(p)] for p in ps]
cl = [tim["by_arch"]["lru_clamped"][str(p)] for p in ps]
tr = [tim["by_arch"]["transformer"][str(p)] for p in ps]
ax[2].plot(ps, fr, "-o", color=CYAN, mec=CYAN, mfc=CYAN, lw=2.4, ms=6, label="LRU free (osc)")
ax[2].plot(ps, tr, "--s", color=GREEN, mec=GREEN, mfc="none", lw=1.6, ms=6, label="Transformer")
ax[2].plot(ps, cl, "-o", color=GRAY, mec=GRAY, mfc=GRAY, lw=2, ms=6, label="LRU clamped")
ax[2].set_xlabel("period  P"); ax[2].set_ylabel("periodic-prediction accuracy")
ax[2].set_ylim(0, 1.05)
ax[2].legend(fontsize=8, facecolor=BG, edgecolor="#00aa00", labelcolor=GREEN, loc="lower left")
ax[2].set_title("TIMING · rhythm's NATIVE domain\nosc = attention >> decay (decay collapses)",
                color=GREEN, fontsize=10.5, pad=10)

fig.suptitle("the demarcation map · rhythm is DOMINATED for capacity, NATIVE for robust & temporal coding · styxx/fathom-lab · pre-registered",
             color=GREEN, fontsize=10.5, y=1.03)
fig.tight_layout()
out = HERE / "demarcation_map.png"
fig.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
print("wrote", out.name)

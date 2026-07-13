"""Generate erasure_bound_fork.png from the canonical B2 receipts -- no hand-entered numbers.

Left panel: the reads per (seed, alpha) cell -- naive-DoM vs private-13, static vs adaptive
eraser, against the clean read, chance, and the survival threshold. Right panel: the attacker's
own objective at the end of training (l_erase tail, log scale) -- the static eraser converged,
the chasing eraser never did.

Usage: python papers/calib-poison-general/make_erasure_bound_figure.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STATIC = json.loads((HERE / "b2_subspace_erasure_result.json").read_text(encoding="utf-8"))
ADAPT = json.loads((HERE / "b2_adaptive_erasure_result.json").read_text(encoding="utf-8"))

BG = "#14101a"          # deep aubergine
FG = "#d8cfe6"          # pale lilac text
LILAC = "#b57edc"       # private-13
LILAC_DIM = "#6f4d8f"
GREY = "#8a8496"        # naive
GREEN = "#7fd8a4"       # survival accent
RED = "#e07a7a"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": FG, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "font.family": "monospace", "font.size": 9,
})

cells_s = STATIC["points"]
cells_a = ADAPT["points"]
assert [(c["seed"], c["alpha"]) for c in cells_s] == [(c["seed"], c["alpha"]) for c in cells_a], \
    "static and adaptive receipts must share the same (seed, alpha) grid"
labels = [f"s{c['seed']} a={c['alpha']:g}" for c in cells_s]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), width_ratios=[1.7, 1])
fig.suptitle("the attacker's fork -- erase the honesty read, keep the knowledge (Qwen2.5-1.5B, LoRA r16, 300 steps)",
             fontsize=10.5, color=FG)

x = np.arange(len(cells_s))
w = 0.19
ax1.bar(x - 1.5 * w, [c["naive_dom6_auroc"] for c in cells_s], w, color=GREY, alpha=0.55, label="naive DoM after STATIC")
ax1.bar(x - 0.5 * w, [c["private13_auroc"] for c in cells_s], w, color=LILAC_DIM, label="private-13 after STATIC")
ax1.bar(x + 0.5 * w, [c["naive_dom6_auroc"] for c in cells_a], w, color=GREY, label="naive DoM after ADAPTIVE")
adapt_hatch = ["//" if not c["admissible"] else "" for c in cells_a]
bars = ax1.bar(x + 1.5 * w, [c["private13_auroc"] for c in cells_a], w, color=LILAC, label="private-13 after ADAPTIVE")
for b, h, c in zip(bars, adapt_hatch, cells_a):
    if h:
        b.set_hatch(h); b.set_alpha(0.6)
        ax1.text(b.get_x() + b.get_width() / 2, c["private13_auroc"] + 0.012, "no bite\n(inadmissible)",
                 ha="center", fontsize=6.5, color=FG, alpha=0.9)

ax1.axhline(STATIC["clean_private13_auroc"], color=GREEN, ls=":", lw=1.2)
ax1.text(len(x) - 0.42, STATIC["clean_private13_auroc"] + 0.008, f"clean {STATIC['clean_private13_auroc']}", color=GREEN, fontsize=8, ha="right")
ax1.axhline(0.70, color=FG, ls="--", lw=0.9, alpha=0.7)
ax1.text(-0.45, 0.706, "survival 0.70", color=FG, fontsize=8, alpha=0.85)
ax1.axhline(0.50, color=RED, ls="--", lw=0.9, alpha=0.7)
ax1.text(-0.45, 0.506, "chance", color=RED, fontsize=8, alpha=0.85)
ax1.set_xticks(x, labels)
ax1.set_ylim(0.40, 1.0)
ax1.set_ylabel("EVAL AUROC (n=66)")
ax1.set_title("the read survives both erasers; chasing made it WEAKER", fontsize=9.5)
ax1.legend(fontsize=7.5, facecolor=BG, edgecolor=FG, labelcolor=FG, loc="upper left", framealpha=0.9)

tails_s = [c["train_hist_tail"]["l_erase"] for c in cells_s]
tails_a = [c["train_hist_tail"]["l_erase"] for c in cells_a]
ax2.plot(x, tails_s, "o-", color=LILAC_DIM, label="STATIC eraser (converged)")
ax2.plot(x, tails_a, "s-", color=RED, label="ADAPTIVE eraser (never converged)")
ax2.set_yscale("log")
ax2.set_xticks(x, labels)
ax2.set_ylabel("l_erase at step 299 (log)")
ax2.set_title("the attacker's own loss says why", fontsize=9.5)
ax2.legend(fontsize=7.5, facecolor=BG, edgecolor=FG, labelcolor=FG, loc="center right")
for s in ("top", "right"):
    ax1.spines[s].set_visible(False); ax2.spines[s].set_visible(False)

fig.text(0.99, 0.01, "receipts: b2_subspace_erasure_result.json / b2_adaptive_erasure_result.json  (OATH-HELD 74/0, 100/0)",
         ha="right", fontsize=7, color=FG, alpha=0.8)
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
out = HERE / "erasure_bound_fork.png"
fig.savefig(out, dpi=180)
print(f"wrote {out}")

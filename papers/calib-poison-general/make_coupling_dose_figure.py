"""Generate coupling_dose_curve.png from b2_coupling_dose_result.json -- read(rank) with
knowledge(rank) overlaid, accumulate vs fixed-rank control, per seed. No hand-entered numbers.

The figure the coupling-constant experiment exists to draw: if the curves are COUPLED the read and
knowledge fall together and r* is marked where the read crosses the survival threshold; if
DECOUPLED the read falls while knowledge stays flat -- the break made visible.

Usage: python papers/calib-poison-general/make_coupling_dose_figure.py [result.json] [out.png]
       (defaults: b2_coupling_dose_result.json -> coupling_dose_curve.png)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "b2_coupling_dose_result.json"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "coupling_dose_curve.png"

BG, FG = "#14101a", "#d8cfe6"
LILAC, LILAC_DIM, GREEN, RED, GREY = "#b57edc", "#6f4d8f", "#7fd8a4", "#e07a7a", "#8a8496"
SURVIVAL, KNOW_FLOOR = 0.70, 0.75

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": FG, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG, "font.family": "monospace", "font.size": 9,
})

res = json.loads(SRC.read_text(encoding="utf-8"))
curves = res["curves"]
seeds = sorted(curves.keys(), key=str)
fig, axes = plt.subplots(1, len(seeds), figsize=(6.0 * len(seeds), 4.6), squeeze=False)
title = res.get("verdict", "coupling dose-response")
fig.suptitle(f"read/knowledge coupling under an ACCUMULATING eraser -- {title}", fontsize=10.5)

for ax, seed in zip(axes[0], seeds):
    arms = curves[seed]
    acc = arms["accumulate"]; fix = arms["fixed"]
    r = [p["erased_rank"] for p in acc]
    ax.plot(r, [p["private13"] for p in acc], "o-", color=LILAC, lw=2, label="read (accumulate)")
    ax.plot(r, [p["knowledge"] for p in acc], "s-", color=GREEN, lw=2, label="knowledge (accumulate)")
    ax.plot([p["erased_rank"] for p in fix], [p["private13"] for p in fix], "o--", color=LILAC_DIM,
            lw=1.2, alpha=0.8, label="read (fixed-rank control)")
    ax.plot([p["erased_rank"] for p in fix], [p["knowledge"] for p in fix], "s--", color=GREY,
            lw=1.0, alpha=0.7, label="knowledge (fixed control)")
    ax.axhline(SURVIVAL, color=FG, ls=":", lw=0.9, alpha=0.8)
    ax.text(r[0], SURVIVAL + 0.006, "survival threshold of 0.70", fontsize=7.5, alpha=0.9)
    ax.axhline(KNOW_FLOOR, color=GREEN, ls=":", lw=0.8, alpha=0.5)
    # mark r* where the accumulate read first crosses below survival
    broke = next((p for p in acc if p["private13"] < SURVIVAL), None)
    if broke:
        ax.axvline(broke["erased_rank"], color=RED, ls="--", lw=1.2, alpha=0.9)
        ax.annotate(f"r* = {broke['erased_rank']}\nread {broke['private13']}\nknow {broke['knowledge']}",
                    (broke["erased_rank"], broke["private13"]), textcoords="offset points",
                    xytext=(8, -34), fontsize=7.5, color=RED)
    ax.set_xlabel("accumulated erased rank (mean per layer)")
    ax.set_ylabel("EVAL AUROC / accuracy (n=66)")
    ax.set_title(f"seed {seed}", fontsize=9.5)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=7, facecolor=BG, edgecolor=FG, labelcolor=FG, loc="lower left", framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.text(0.99, 0.01, f"receipt: {SRC.name}", ha="right", fontsize=7, color=FG, alpha=0.8)
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
fig.savefig(OUT, dpi=180)
print(f"wrote {OUT}")

# -*- coding: utf-8 -*-
"""Render the Epistemic Psychometric Function (STYXX-1 aesthetic) from
probe_curve_results.json. Two panels (one per model); each plots abstention-rate
(what the model admits it doesn't know) and inconsistency (what it betrays while
answering) across the reality gradient. The gap = confident confabulation.
Output: epistemic_curve.png (2400x1050)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

FONTS = Path(__file__).resolve().parents[2] / "styxx" / "fonts"
if FONTS.exists():
    for t in FONTS.glob("*.ttf"): font_manager.fontManager.addfont(str(t))
MONO = "JetBrains Mono"

BG="#0A0A0A"; SCREEN="#070708"; CHASSIS="#1A1A1F"; INK="#FAFAFA"
ZINC_400="#A1A1AA"; ZINC_500="#737373"; ZINC_700="#3F3F46"; ZINC_800="#262626"
GRID="#0F1A1F"; GRID_HI="#142730"; CYAN="#22D3EE"; CYAN_300="#67E8F9"; ALERT="#FB7185"; AMBER="#FBBF24"

D = json.loads(Path(__file__).parent.joinpath("probe_curve_results.json").read_text())
curves = D["curves"]; models = list(curves.keys())
LEVELS = ["L0_real_common","L1_real_obscure","L2_plausible_fake","L3_absurd_fake"]
XLAB = ["real\ncommon","real\nobscure","plausible\nFAKE","absurd\nFAKE"]

fig = plt.figure(figsize=(16, 7), facecolor=BG, dpi=150)
fig.add_axes([0.018,0.03,0.964,0.94], zorder=-2).set_facecolor(CHASSIS)
for ax in fig.axes:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(ZINC_700); s.set_linewidth(1.3)

fig.text(0.045, 0.93, "STYXX-1", color=CYAN_300, fontsize=19, fontfamily=MONO, fontweight="bold", va="center")
fig.text(0.045, 0.905, "EPISTEMIC PSYCHOMETRIC FUNCTION  ·  where a model's knowledge ends",
         color=ZINC_500, fontsize=10, fontfamily=MONO, va="center")
fig.text(0.955, 0.915, "styxx.org · @fathom_lab", color=CYAN_300, fontsize=10.5,
         fontfamily=MONO, fontweight="bold", va="center", ha="right")

x = np.arange(4)
panels = [(0.058,0.115,0.40,0.70), (0.555,0.115,0.40,0.70)]
for (px,py,pw,ph), model in zip(panels, models):
    ax = fig.add_axes([px,py,pw,ph]); ax.set_facecolor(SCREEN)
    for s in ax.spines.values(): s.set_color(ZINC_800); s.set_linewidth(0.6)
    for gy in np.arange(0,1.01,0.1): ax.axhline(gy, color=GRID, lw=0.4, zorder=0)
    for gy in np.arange(0,1.01,0.5): ax.axhline(gy, color=GRID_HI, lw=0.55, zorder=0)

    ab=[curves[model][l]["abstain_rate"] for l in LEVELS]
    inc=[curves[model][l]["mean_inconsistency_answered"] for l in LEVELS]
    inc_n=[(v/1.8 if v is not None else None) for v in inc]  # normalize entropy (~0..1.8) to 0..1

    # confident-confabulation shading: abstention<0.5 AND inconsistency high
    for i in range(4):
        if ab[i] is not None and ab[i]<0.5 and inc_n[i] is not None and inc_n[i]>0.55:
            ax.axvspan(i-0.5, i+0.5, color=ALERT, alpha=0.07, zorder=0)

    def plot(vals, color, label, ls):
        xs=[i for i,v in enumerate(vals) if v is not None]; ys=[v for v in vals if v is not None]
        for lw,a in [(7,0.06),(4,0.12)]: ax.plot(xs,ys,color=color,lw=lw,alpha=a,zorder=3)
        ax.plot(xs, ys, color=color, lw=2.2, ls=ls, marker="o", ms=7, zorder=4, label=label)
    plot(ab, CYAN, "abstention (admits)", "-")
    plot(inc_n, AMBER, "inconsistency (betrays)", "--")
    ax.axhline(0.5, color=ZINC_700, lw=0.8, ls=":", zorder=1)

    ax.set_xlim(-0.5,3.5); ax.set_ylim(-0.04,1.06); ax.set_xticks(x)
    ax.set_xticklabels(XLAB, color=ZINC_400, fontsize=9.5, fontfamily=MONO)
    ax.tick_params(length=0)
    for s in ("real_common","real"): pass
    ax.set_yticks([0,0.5,1.0]); ax.set_yticklabels(["0","0.5","1.0"], color=ZINC_500, fontsize=8.5, fontfamily=MONO)
    ax.set_title(model, color=INK, fontsize=14, fontfamily=MONO, fontweight="bold", pad=10)
    leg=ax.legend(loc="upper left", fontsize=8.5, framealpha=0, labelcolor="linecolor")
    for txt in leg.get_texts(): txt.set_fontfamily(MONO)

fig.text(0.5, 0.055,
    "cyan = what the model admits it doesn't know   ·   amber = what it betrays (cross-sample divergence, ÷1.8)   "
    "·   red band = answers but invents (confident confabulation)",
    color=ZINC_500, fontsize=9, fontfamily=MONO, ha="center")

out = Path(__file__).parent.joinpath("epistemic_curve.png")
fig.savefig(out, facecolor=BG, dpi=150); print(f"saved: {out}")

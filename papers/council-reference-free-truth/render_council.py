# -*- coding: utf-8 -*-
"""Render the Council result (STYXX-1 aesthetic): inter-model agreement per question,
by tier. Shows real (common+obscure) piled at high agreement, fake scattered low —
reference-free separation. Output: council.png (2200x900)."""
from pathlib import Path
import json, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

FONTS = Path(__file__).resolve().parents[2] / "styxx" / "fonts"
if FONTS.exists():
    for t in FONTS.glob("*.ttf"): font_manager.fontManager.addfont(str(t))
MONO="JetBrains Mono"
BG="#0A0A0A"; SCREEN="#070708"; CHASSIS="#1A1A1F"; INK="#FAFAFA"
ZINC_400="#A1A1AA"; ZINC_500="#737373"; ZINC_700="#3F3F46"; ZINC_800="#262626"
GRID="#0F1A1F"; GRID_HI="#142730"; CYAN="#22D3EE"; CYAN_300="#67E8F9"; ALERT="#FB7185"; AMBER="#FBBF24"

D=json.loads(Path(__file__).parent.joinpath("probe_council_results.json").read_text())
rows=D["rows"]; J=D["by_judge_clustering"]
tiers=[("real_common","real · common",CYAN_300),("real_obscure","real · obscure",CYAN),
       ("fake","FAKE (nonexistent)",ALERT)]
random.seed(1)

fig=plt.figure(figsize=(14.5,6), facecolor=BG, dpi=150)
fig.add_axes([0.02,0.04,0.96,0.92], zorder=-2).set_facecolor(CHASSIS)
for ax in fig.axes:
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(ZINC_700); s.set_linewidth(1.3)

fig.text(0.05,0.90,"STYXX-1", color=CYAN_300, fontsize=18, fontfamily=MONO, fontweight="bold", va="center")
fig.text(0.05,0.875,"THE COUNCIL  ·  reference-free: do independent models agree?", color=ZINC_500,
         fontsize=10, fontfamily=MONO, va="center")
fig.text(0.95,0.885,"styxx.org · @fathom_lab", color=CYAN_300, fontsize=10, fontfamily=MONO,
         fontweight="bold", va="center", ha="right")

ax=fig.add_axes([0.21,0.20,0.74,0.58]); ax.set_facecolor(SCREEN)
for s in ax.spines.values(): s.set_color(ZINC_800); s.set_linewidth(0.6)
for gx in np.arange(0,1.01,0.25): ax.axvline(gx, color=GRID, lw=0.5, zorder=0)
ax.axvline(0.625, color=ZINC_700, lw=0.9, ls=":", zorder=1)  # rough decision line

for yi,(key,lab,col) in enumerate(tiers):
    vals=[r["agree_judge"] for r in rows if r["tier"]==key]
    ys=[yi + random.uniform(-0.16,0.16) for _ in vals]
    for lw,a in [(16,0.05)]:
        ax.scatter(vals, ys, s=320, color=col, alpha=0.10, zorder=2)
    ax.scatter(vals, ys, s=120, color=col, edgecolor=BG, linewidth=0.8, zorder=3)
    ax.text(-0.04, yi, lab, color=col, fontsize=12, fontfamily=MONO, fontweight="bold",
            va="center", ha="right", transform=ax.get_yaxis_transform())

ax.set_xlim(-0.02,1.05); ax.set_ylim(-0.6,2.6); ax.set_yticks([])
ax.set_xticks([0,0.25,0.5,0.75,1.0]); ax.set_xticklabels(["0","0.25","0.5","0.75","1.0"],
    color=ZINC_500, fontsize=9, fontfamily=MONO)
ax.tick_params(length=0)
ax.set_xlabel("inter-model agreement  (largest cross-model cluster ÷ council)", color=ZINC_400,
              fontsize=10.5, fontfamily=MONO, labelpad=8)

fig.text(0.595,0.085,
    f"C1 real-vs-fake AUC {J['C1_real_vs_fake']:.2f}   ·   C2 obscure-real-vs-fake AUC "
    f"{J['C2_obscure_vs_fake']:.2f}   ·   no correlated confabulation (fakes scatter)",
    color=ZINC_500, fontsize=9.5, fontfamily=MONO, ha="center")

out=Path(__file__).parent.joinpath("council.png")
fig.savefig(out, facecolor=BG, dpi=150); print(f"saved: {out}")

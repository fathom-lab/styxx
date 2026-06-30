# -*- coding: utf-8 -*-
"""Figure: the LLM's brain-meaning climbs the ventral hierarchy. Per ROI (V1->ventral): total LLM<->brain RSA
(|lex) vs the meaning residual after removing a vision model (|lex+vision), with bootstrap CI band."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "tier2_gradient_result.json"))["rois"]
rois = ["V1", "V2", "V3", "hV4", "LOC", "ventral"]
x = np.arange(len(rois))
total = [d[r]["LLM|lex"] for r in rois]
resid = [d[r]["LLM|lex+vision"] for r in rois]
lo = [d[r]["LLM|lex+vision_CI"][0] for r in rois]; hi = [d[r]["LLM|lex+vision_CI"][1] for r in rois]

BG="#1a1020"; INK="#F3ECF7"; SUB="#B79FCB"; LILAC="#C9A2F0"; CYAN="#65E0D8"; MUTE="#6E5A82"; GRID="#2a1d38"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(11, 6.6), dpi=170); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
fig.subplots_adjust(left=0.09, right=0.96, top=0.80, bottom=0.13)
ax.axvspan(-0.5, 2.5, color="#221634", zorder=0); ax.axvspan(2.5, 5.5, color="#2c1d44", zorder=0)
ax.text(1.0, 0.012, "early visual", color=SUB, fontsize=10, ha="center", style="italic")
ax.text(4.0, 0.012, "high-level object cortex", color=SUB, fontsize=10, ha="center", style="italic")
ax.plot(x, total, "-o", color=MUTE, lw=2, ms=7, zorder=3, label="total LLM↔brain (incl. visual)")
ax.fill_between(x, lo, hi, color=LILAC, alpha=0.22, zorder=2)
ax.plot(x, resid, "-o", color=LILAC, lw=3.2, ms=9, zorder=4, label="meaning (after removing a vision model)")
ax.axhline(0, color=MUTE, lw=1)
ax.set_xticks(x); ax.set_xticklabels(rois, fontsize=11, color=INK)
ax.set_ylabel("LLM ↔ brain RSA  (partial)", fontsize=10.5, color=SUB)
ax.set_xlabel("ventral visual stream  →  (low-level to high-level)", fontsize=10.5, color=SUB)
ax.tick_params(colors=SUB); ax.set_ylim(0, 0.34); ax.set_xlim(-0.5, 5.5); ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(MUTE)
ax.legend(loc="upper left", frameon=False, fontsize=10.5, labelcolor=INK)
ax.annotate("meaning ~3× stronger\nin high-level cortex", (4, resid[4]), textcoords="offset points", xytext=(12, 16), fontsize=10, color=INK, fontweight="bold")
fig.text(0.09, 0.95, "a free LLM's brain-meaning lives at the semantic end of the visual stream", fontsize=16.5, fontweight="bold", color=INK, va="top")
fig.text(0.09, 0.895, "After removing a vision model, the text-only LLM's alignment with the brain's concept geometry is ~3× stronger in\n"
         "high-level object cortex (LOC/ventral) than early visual cortex (V1–V3). Meaning, not pixels — and it's localised.",
         fontsize=10.3, color=SUB, va="top", linespacing=1.35)
fig.text(0.09, 0.028, "fathom-lab · styxx   ·   THINGS-fMRI, 645 concepts, partial-lexical + CLIP-image control   ·   shaded = 95% CI   ·   all residuals > 0", fontsize=8.2, color=MUTE)
fig.savefig(HERE / "fig_gradient.png", facecolor=BG); plt.close(fig)
print("residual V1-V3 mean", round(float(np.mean(resid[:3])), 3), "high-level mean", round(float(np.mean(resid[3:])), 3))
print("wrote fig_gradient.png")

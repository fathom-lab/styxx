# -*- coding: utf-8 -*-
"""plot_grounded_defense.py — the question-framing hole, and how the in-package defense closes it."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "grounded_attack_defended_result.json")))
naive = r["confidently_fooled_naive_asasked"]
defended = r["confidently_fooled_defended_canonical"]
inj = r["injection_catches_fooled_frac"]

BG, FG, RED, GREEN, CY = "#0a0a0a", "#e6e6e6", "#ff3b3b", "#00ff66", "#00e0ff"
plt.rcParams.update({"font.family": "monospace", "figure.facecolor": BG, "axes.facecolor": BG,
                     "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "axes.edgecolor": "#444"})
fig, ax = plt.subplots(figsize=(8.5, 5))
bars = ["naive\n(resample as-asked)", "DEFENDED\n(resample canonical)"]
vals = [naive * 100, defended * 100]
cols = [RED, GREEN]
b = ax.bar(bars, vals, color=cols, width=0.55, zorder=3)
for rect, v in zip(b, vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", color=FG, fontsize=12)
ax.set_ylabel("grounded_honesty CONFIDENTLY fooled by the framing attack (%)")
ax.set_title("a cheap framing attack fools grounded_honesty — until you resample the canonical question",
             color=FG, fontsize=10.8)
ax.set_ylim(0, max(vals) + 4)
ax.grid(True, axis="y", color="#222", lw=0.6, zorder=0)
ax.text(0.5, 0.86, f"detect_context_injection flags {inj*100:.0f}% of the\nfooled items (cross-context divergence)",
        transform=ax.transAxes, ha="center", color=CY, fontsize=10,
        bbox=dict(boxstyle="round", fc="#0a0a0a", ec=CY, lw=1))
fig.text(0.012, 0.012, "styxx · papers/consistency-robustness · Qwen2.5-3B, n=60 · verdict from canonical arm + "
         "injection flag (divergence.py security model)", color="#777", fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1))
out = os.path.join(HERE, "grounded_defense_curve.png")
fig.savefig(out, dpi=150, facecolor=BG)
print(f"naive {naive:.3f} -> defended {defended:.3f} | injection catches {inj:.3f}")
print("wrote", os.path.basename(out))

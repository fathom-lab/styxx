"""Generate phase-transition ablation chart for every-mind-leaves-vitals article.

Renders the §2 table from the paper as a dark-background matrix-styled image
suitable for X (1200x675).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release" / "phase-transition-chart.png"

# Data: drift class -> [K=1, K=2, K=6, K=10] AUCs
# critical_k marks the cell where phase transition fires
data = [
    ("arg_drop",            [0.501, 0.998, 0.998, 1.000], 1, "+arg_count_zscore"),
    ("spurious_arg",        [0.999, 0.997, 0.997, 0.997], 0, "+spurious_arg_frac"),
    ("irrelevance_called",  [0.486, 0.705, 0.828, 0.962], 3, "+prompt_coverage"),
    ("arg_swap",            [0.512, 0.488, 0.691, 0.683], 2, "+type_mismatch_frac"),
]

ks = ["K=1", "K=2", "K=6", "K=10"]

# Fathom palette: black bg, matrix green and cyan
BG = "#000000"
GRID = "#222222"
TEXT = "#00FFFF"        # cyan headers
ROW = "#CCCCCC"         # row labels light
CELL_LO = "#1a1a1a"     # near-chance bg
CELL_HI = "#003300"     # high-AUC bg (dark green)
CELL_CRITICAL = "#00FF00"  # phase-transition cell (bright green)
CRITICAL_TEXT = "#000000"

fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")

# Title
ax.text(
    0.3, 7.4, "phase transitions in cognitive-state detection",
    fontsize=22, fontfamily="monospace", color=TEXT, weight="bold",
    va="center",
)
ax.text(
    0.3, 6.75,
    "feature-count ablation on the styxx tool-call drift detector — every drift class has a critical feature",
    fontsize=11, fontfamily="monospace", color=ROW, va="center",
)

# Header row
header_y = 5.85
ax.text(0.3, header_y, "drift class", fontsize=12, fontfamily="monospace",
        color=TEXT, weight="bold", va="center")
ax.text(7.5, header_y, "K=1", fontsize=12, fontfamily="monospace",
        color=TEXT, weight="bold", va="center", ha="center")
ax.text(9.0, header_y, "K=2", fontsize=12, fontfamily="monospace",
        color=TEXT, weight="bold", va="center", ha="center")
ax.text(10.5, header_y, "K=6", fontsize=12, fontfamily="monospace",
        color=TEXT, weight="bold", va="center", ha="center")
ax.text(12.0, header_y, "K=10", fontsize=12, fontfamily="monospace",
        color=TEXT, weight="bold", va="center", ha="center")
ax.text(13.5, header_y, "critical feat.", fontsize=12, fontfamily="monospace",
        color=TEXT, weight="bold", va="center", ha="left")

# Underline header
ax.plot([0.2, 14], [5.55, 5.55], color=GRID, linewidth=1)

# Rows
row_height = 1.0
for i, (cls, aucs, crit_idx, feat) in enumerate(data):
    y = 4.7 - i * row_height
    # Row label
    ax.text(0.3, y, cls, fontsize=12, fontfamily="monospace",
            color=ROW, va="center")
    # Cells
    xs = [7.5, 9.0, 10.5, 12.0]
    for j, (auc, x) in enumerate(zip(aucs, xs)):
        # Cell background
        is_critical = (j == crit_idx)
        cell_color = CELL_CRITICAL if is_critical else (
            CELL_HI if auc > 0.85 else CELL_LO
        )
        rect = Rectangle(
            (x - 0.65, y - 0.32), 1.3, 0.64,
            facecolor=cell_color, edgecolor=GRID, linewidth=0.5,
        )
        ax.add_patch(rect)
        text_color = CRITICAL_TEXT if is_critical else (
            "#00FF99" if auc > 0.85 else "#888888"
        )
        ax.text(
            x, y, f"{auc:.3f}",
            fontsize=12, fontfamily="monospace",
            color=text_color, va="center", ha="center",
            weight="bold" if is_critical else "normal",
        )
    # Critical feature label
    ax.text(13.5, y, feat, fontsize=10, fontfamily="monospace",
            color="#00FF99", va="center", ha="left", style="italic")

# Footer
footer_y = 0.45
ax.text(
    0.3, footer_y,
    "replicated on refusal detector (starts_with_sorry: 0.500 → 0.969 at K=1)  |  hallucination detector (trigram_novelty: 0.500 → 0.9947 at K=1)",
    fontsize=9, fontfamily="monospace", color="#888888", va="center",
)
ax.text(
    0.3, 0.05,
    "every mind leaves vitals · fathom lab · fathom.darkflobi.com/cognometry",
    fontsize=9, fontfamily="monospace", color=TEXT, va="center",
)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(OUT, facecolor=BG, dpi=100, bbox_inches="tight", pad_inches=0.3)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

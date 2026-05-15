"""
render_announcement_card.py — the headline visual artifact for the
2026-05-14 universal-directions announcement thread.

Same luxury aesthetic as styxx.cognometric_card. Three-row results
table for the three cognitive directions tested, plus the chain
visualization (4 families → universal direction → closed-model AUC).

Output: card_universality_results.png (1200x630)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# fonts
FONTS_DIR = Path(__file__).parent / "fonts"
for ttf in FONTS_DIR.glob("*.ttf"):
    font_manager.fontManager.addfont(str(ttf))

MONO = "JetBrains Mono"
SERIF = "Source Serif 4"

# palette (locked styxx luxury register)
BG          = "#0A0A0A"
INK         = "#F3EBE0"
INK_DIM     = "#C9C2B6"
PARCHMENT   = "#9F9890"
ARCHIVE     = "#6B6759"
RULE        = "#26241F"
RULE_HI     = "#3A3730"
GOLD        = "#C8A86B"
GOLD_BRIGHT = "#E0BE7E"
GOLD_DIM    = "#7A6638"
MINT        = "#B7E4C7"
PEACH       = "#F5C5B0"
CLAY        = "#D89886"


def hline(fig, x0, x1, y, color=RULE, lw=0.6):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=color, linewidth=lw))


def diamond(fig, cx, cy, r, color, alpha=1.0, zorder=4):
    poly = mpatches.Polygon(
        [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)],
        closed=True, facecolor=color, edgecolor="none", alpha=alpha,
        transform=fig.transFigure, zorder=zorder)
    fig.patches.append(poly)


def four_pointed_star(fig, cx, cy, r, color, alpha=1.0, zorder=4):
    long_r = r
    short_r = r * 0.30
    pts = []
    for i in range(8):
        angle = (i / 8) * 2 * np.pi - np.pi / 2
        radius = long_r if i % 2 == 0 else short_r
        pts.append((cx + np.cos(angle) * radius,
                    cy + np.sin(angle) * radius * 0.7))
    poly = mpatches.Polygon(pts, closed=True, facecolor=color,
                             edgecolor="none", alpha=alpha,
                             transform=fig.transFigure, zorder=zorder)
    fig.patches.append(poly)


def main():
    fig = plt.figure(figsize=(12, 6.3), facecolor=BG, dpi=100)

    # outer double-rule border
    outer = mpatches.Rectangle((0.026, 0.040), 0.948, 0.920,
                                transform=fig.transFigure, facecolor="none",
                                edgecolor=RULE, linewidth=0.8, zorder=-3)
    fig.patches.append(outer)
    inner = mpatches.Rectangle((0.034, 0.050), 0.932, 0.900,
                                transform=fig.transFigure, facecolor="none",
                                edgecolor=GOLD_DIM, linewidth=0.5, zorder=-2)
    fig.patches.append(inner)
    for cx, cy in [(0.034, 0.050), (0.966, 0.050),
                   (0.034, 0.950), (0.966, 0.950)]:
        diamond(fig, cx, cy, 0.0045, GOLD, alpha=0.95, zorder=5)

    # ── top chrome
    chrome_y = 0.905
    fig.text(0.500, chrome_y,
             "F A T H O M  ·  C O G N O M E T R I C  ·  U N I V E R S A L I T Y   R E S U L T S",
             color=PARCHMENT, fontsize=10.5, fontfamily=MONO,
             va="center", ha="center")
    fig.text(0.948, chrome_y, "14  M A Y  2 0 2 6",
             color=GOLD, fontsize=10, fontfamily=MONO,
             va="center", ha="right")
    fig.text(0.052, chrome_y, "n=30  ·  4 families",
             color=ARCHIVE, fontsize=9.5, fontfamily=MONO,
             va="center", ha="left")
    hline(fig, 0.060, 0.940, 0.875, color=RULE, lw=0.6)

    # ── headline
    fig.text(0.500, 0.795,
             "Of three cognitive directions tested,",
             color=INK_DIM, fontsize=18, fontfamily=SERIF, style="italic",
             va="center", ha="center")
    fig.text(0.500, 0.730,
             "only ",
             color=INK_DIM, fontsize=40, fontfamily=SERIF, style="italic",
             va="center", ha="right")
    fig.text(0.500, 0.730,
             "comply / refuse",
             color=GOLD, fontsize=40, fontfamily=SERIF, style="italic",
             va="center", ha="left")
    fig.text(0.500, 0.665,
             "universalizes across LLM families.",
             color=INK_DIM, fontsize=18, fontfamily=SERIF, style="italic",
             va="center", ha="center")

    # center hairline + diamond
    hline(fig, 0.100, 0.470, 0.605)
    diamond(fig, 0.500, 0.605, 0.0045, GOLD, alpha=0.9, zorder=5)
    hline(fig, 0.530, 0.900, 0.605)

    # ── results table
    rows = [
        ("comply / refuse",  "0.730",  "0.541",  "universal",     GOLD),
        ("truthfulness",     "0.296",  "−0.122", "family-specific", INK_DIM),
        ("corrigibility",    "0.235",  "+0.148", "family-specific", INK_DIM),
    ]

    # column headers (tighter, no inter-letter spacing — was colliding)
    col_y_header = 0.555
    fig.text(0.075, col_y_header, "direction",
             color=PARCHMENT, fontsize=10, fontfamily=MONO,
             va="center", ha="left")
    fig.text(0.495, col_y_header, "full r  (n=30)",
             color=PARCHMENT, fontsize=10, fontfamily=MONO,
             va="center", ha="center")
    fig.text(0.725, col_y_header, "borderline r  (n=10)",
             color=PARCHMENT, fontsize=10, fontfamily=MONO,
             va="center", ha="center")
    fig.text(0.925, col_y_header, "verdict",
             color=PARCHMENT, fontsize=10, fontfamily=MONO,
             va="center", ha="right")

    hline(fig, 0.075, 0.925, 0.530)

    # rows
    row_top = 0.490
    row_h = 0.060
    for i, (name, full_r, border_r, verdict, color) in enumerate(rows):
        y = row_top - i * row_h
        fig.text(0.075, y, name,
                 color=INK, fontsize=15, fontfamily=SERIF, style="italic",
                 va="center", ha="left")
        fig.text(0.495, y, full_r,
                 color=color, fontsize=18, fontfamily=SERIF, style="italic",
                 va="center", ha="center")
        fig.text(0.725, y, border_r,
                 color=color, fontsize=18, fontfamily=SERIF, style="italic",
                 va="center", ha="center")
        fig.text(0.925, y, verdict,
                 color=color, fontsize=12, fontfamily=MONO,
                 va="center", ha="right")
        # subtle row separator
        if i < len(rows) - 1:
            hline(fig, 0.075, 0.925, y - row_h / 2, color=RULE, lw=0.4)

    hline(fig, 0.075, 0.925, 0.310)

    # ── operational result strip
    fig.text(0.500, 0.270,
             "the universal direction's embedding-axis projection",
             color=PARCHMENT, fontsize=11, fontfamily=MONO, style="italic",
             va="center", ha="center")
    fig.text(0.500, 0.230,
             "predicts closed-model refusal",
             color=INK, fontsize=15, fontfamily=SERIF, style="italic",
             va="center", ha="center")
    fig.text(0.300, 0.180, "AUC  1.000",
             color=GOLD, fontsize=22, fontfamily=SERIF, style="italic",
             va="center", ha="center")
    fig.text(0.300, 0.150, "gpt-4o-mini  +  gpt-4.1-mini  (n=30 canonical)",
             color=ARCHIVE, fontsize=9.5, fontfamily=MONO,
             va="center", ha="center")
    # divider
    fig.add_artist(plt.Line2D([0.470, 0.470], [0.130, 0.200],
                              color=RULE_HI, linewidth=0.8))
    fig.text(0.500, 0.180, "+",
             color=GOLD, fontsize=14, fontfamily=SERIF, style="italic",
             va="center", ha="center")
    fig.add_artist(plt.Line2D([0.530, 0.530], [0.130, 0.200],
                              color=RULE_HI, linewidth=0.8))
    fig.text(0.700, 0.180, "AUC  0.851",
             color=GOLD, fontsize=22, fontfamily=SERIF, style="italic",
             va="center", ha="center")
    fig.text(0.700, 0.150, "gpt-3.5-turbo  (n=75 adversarial)",
             color=ARCHIVE, fontsize=9.5, fontfamily=MONO,
             va="center", ha="center")

    # ── footer
    hline(fig, 0.075, 0.925, 0.115)
    four_pointed_star(fig, 0.500, 0.080, 0.014, GOLD, alpha=1.0, zorder=4)
    fig.text(0.500, 0.060, "open MIT  ·  fathom-lab/styxx  ·  reproduce in <2 min",
             color=GOLD, fontsize=10, fontfamily=MONO,
             va="center", ha="center")
    fig.text(0.075, 0.075, "styxx.org",
             color=INK, fontsize=11, fontfamily=MONO, fontweight="bold",
             va="center", ha="left")
    fig.text(0.925, 0.075, "@fathom_lab",
             color=INK_DIM, fontsize=11, fontfamily=MONO,
             va="center", ha="right")

    out = Path(__file__).parent / "card_universality_results.png"
    fig.savefig(out, facecolor=BG, dpi=100, pad_inches=0)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

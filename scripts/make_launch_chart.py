"""Generate a shareable 8-benchmark AUC chart for the launch.

Output: release/cognometry-8bench-chart.png, 1200x675 (Twitter card
dims), high contrast dark-bg terminal-green aesthetic matching the
manifesto site.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# 3-seed-averaged numbers from calibrated_weights_v4.py
DATA = [
    ("HaluEval-QA",            0.998, "PASS"),
    ("TruthfulQA",             0.994, "PASS"),
    ("HaluBench-RAGTruth",     0.807, "PASS"),
    ("HaluBench-PubMedQA",     0.719, "PASS"),
    ("HaluEval-Dialog",        0.676, "PASS"),
    ("HaluEval-Summarization", 0.643, "MID"),
    ("HaluBench-FinanceBench", 0.492, "FAIL"),
    ("HaluBench-DROP",         0.424, "FAIL"),
]

COLOR_PASS = "#00ff00"
COLOR_MID  = "#00aaaa"
COLOR_FAIL = "#ff3d00"
BG = "#0a0a0a"
FG = "#c0c0c0"
GRID = "#1e1e1e"


def main():
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    labels = [d[0] for d in DATA][::-1]  # reversed so QA is on top
    aucs   = [d[1] for d in DATA][::-1]
    tags   = [d[2] for d in DATA][::-1]
    colors = [
        COLOR_PASS if t == "PASS" else
        COLOR_MID  if t == "MID"  else COLOR_FAIL
        for t in tags
    ]

    y = np.arange(len(labels))
    bars = ax.barh(y, aucs, color=colors,
                    edgecolor=BG, linewidth=0.8, height=0.72)

    # chance baseline
    ax.axvline(0.5, color="#666", linewidth=0.8,
                linestyle="--", alpha=0.7)
    ax.text(0.5, len(labels) - 0.2, " chance",
            color="#666", fontsize=9, va="bottom")

    # value labels
    for i, (bar, auc, tag) in enumerate(zip(bars, aucs, tags)):
        x = bar.get_width()
        # label inside if wide enough, outside otherwise
        if x > 0.2:
            ax.text(x - 0.01, bar.get_y() + bar.get_height()/2,
                    f"{auc:.3f}",
                    color=BG, fontweight="bold",
                    fontsize=11,
                    ha="right", va="center")
        else:
            ax.text(x + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{auc:.3f}",
                    color=FG, fontweight="bold",
                    fontsize=11,
                    ha="left", va="center")
        # failure tag inline — drawn inside the bar for FAIL
        if tag == "FAIL":
            ax.text(x + 0.05, bar.get_y() + bar.get_height()/2,
                    "← published failure mode",
                    color=COLOR_FAIL, fontsize=10,
                    ha="left", va="center",
                    fontstyle="italic", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=FG, fontsize=11, fontfamily="monospace")
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"], color=FG, fontsize=10)
    ax.set_xlabel("AUC — held-out test, 3-seed averaged (n=150/dataset)",
                   color=FG, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=FG)
    ax.grid(axis="x", color=GRID, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    # title + footer text
    fig.text(0.12, 0.945,
              "cognometry — 8-benchmark hallucination detection",
              color=FG, fontsize=17, fontweight="bold",
              fontfamily="monospace")
    fig.text(0.12, 0.905,
              "styxx 4.0.2   ·   5/8 above AUC 0.65   ·   "
              "2 failure modes published openly",
              color="#666", fontsize=11, fontfamily="monospace")
    fig.text(0.12, 0.025,
              "pip install styxx[nli]   ·   "
              "github.com/fathom-lab/styxx   ·   "
              "fathom.darkflobi.com/cognometry",
              color=COLOR_PASS, fontsize=10, fontfamily="monospace")

    plt.subplots_adjust(left=0.26, right=0.96, top=0.87, bottom=0.12)

    out = ROOT / "release" / "cognometry-8bench-chart.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=BG, dpi=100, bbox_inches=None)
    print(f"wrote: {out}")
    print(f"size:  {out.stat().st_size} bytes")
    print(f"dims:  1200x675 (Twitter/X card)")


if __name__ == "__main__":
    main()

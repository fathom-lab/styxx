#!/usr/bin/env python3
"""
phase_coherence_plot.py
=======================

EKG-style overlay rendering of paired pulse-traces for the phase-
coherence corpus. Medical-monitor register (black bg, matrix-green for
agent A, cyan for agent B, monospace).

Diagnostic visual — NOT part of the locked scorer. Reads the same
manifest the locked scorer reads.

Usage
-----
    python scripts/phase_coherence_plot.py \\
        --manifest papers/cooperative-agent-regime/corpus_manifest.json \\
        --output  papers/cooperative-agent-regime/results/phase_coherence_overlay.png

With --noncoop-manifest provided, renders cooperative and non-
cooperative side-by-side so the dyadic register difference is legible
at a glance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


def _ensure_styxx_importable() -> None:
    here = Path(__file__).resolve().parent.parent
    if (here / "styxx").is_dir() and str(here) not in sys.path:
        sys.path.insert(0, str(here))


_ensure_styxx_importable()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from styxx.coherence import (  # noqa: E402
    load_pulse_trace,
    pulse_coherence,
)


# Brand register — STYXX medical-monitor (memory: project_styxx_monitor_aesthetic)
BG = "#000000"
FG_GRID = "#202020"
FG_AXIS = "#404040"
TEXT = "#E0E0E0"
COLOR_A = "#00FF00"   # matrix green, agent A
COLOR_B = "#00FFFF"   # cyan, agent B
COLOR_ACCENT = "#FF4081"  # signal pink for annotations


def _zscore(xs: list[float]) -> list[float]:
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return []
    sd = xs.std()
    if sd == 0:
        return list(np.zeros_like(xs))
    return list((xs - xs.mean()) / sd)


def _style_axes(ax) -> None:
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(FG_AXIS)
    ax.tick_params(colors=TEXT, which="both", labelsize=8)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=FG_GRID, linewidth=0.5, linestyle="--", alpha=0.5)


def _render_dyad(ax, conv_meta: dict) -> None:
    pulse_a = load_pulse_trace(Path(conv_meta["chart_a"]))
    pulse_b = load_pulse_trace(Path(conv_meta["chart_b"]))
    n = min(len(pulse_a), len(pulse_b))
    if n < 3:
        ax.set_title(
            f"{conv_meta.get('session_id', '?')} — insufficient samples (n={n})",
            fontsize=9, fontfamily="monospace",
        )
        _style_axes(ax)
        return

    c_a = _zscore([s.composite for s in pulse_a[:n]])
    c_b = _zscore([s.composite for s in pulse_b[:n]])
    x = list(range(1, n + 1))

    result = pulse_coherence(pulse_a, pulse_b)

    ax.plot(x, c_a, color=COLOR_A, linewidth=1.4, label="agent A (z·composite)")
    ax.plot(x, c_b, color=COLOR_B, linewidth=1.4, label="agent B (z·composite)",
            linestyle="-")
    ax.axhline(0, color=FG_AXIS, linewidth=0.5, alpha=0.6)
    ax.set_xlim(1, n)
    ax.set_ylim(-2.6, 2.6)

    session = conv_meta.get("session_id", "?")
    title = (
        f"{session}  ·  CC={result.primary_cc:+.3f}  PLV={result.plv:.3f}  n={n}"
    )
    ax.set_title(title, fontsize=9, fontfamily="monospace", loc="left")
    ax.set_xlabel("turn (kth msg of agent)", fontsize=8, fontfamily="monospace")
    ax.set_ylabel("z-score", fontsize=8, fontfamily="monospace")

    # Annotate lag-sweep peak (if not lag-0)
    lag_items = sorted(result.lag_sweep.items())
    peak = max(lag_items, key=lambda kv: kv[1]) if lag_items else None
    if peak and peak[0] != 0:
        ax.text(
            0.99, 0.04,
            f"lag-sweep peak: k={peak[0]:+d} (CC={peak[1]:+.3f})",
            transform=ax.transAxes, fontsize=7,
            fontfamily="monospace", color=COLOR_ACCENT,
            ha="right", va="bottom", alpha=0.8,
        )

    _style_axes(ax)


def render(
    manifest_path: Path,
    output_path: Path,
    noncoop_manifest_path: Optional[Path] = None,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coop_convs = manifest["conversations"]

    has_noncoop = noncoop_manifest_path is not None
    if has_noncoop:
        noncoop_manifest = json.loads(noncoop_manifest_path.read_text(encoding="utf-8"))
        noncoop_convs = noncoop_manifest["conversations"]
    else:
        noncoop_convs = []

    n_rows = max(len(coop_convs), len(noncoop_convs))
    n_cols = 2 if has_noncoop else 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(11 if has_noncoop else 7, 2.0 * n_rows + 1.2),
        facecolor=BG, squeeze=False,
    )

    fig.suptitle(
        "styxx · phase-coherence corpus · cooperative-agent regime"
        + ("  vs  non-cooperative control" if has_noncoop else ""),
        color=TEXT, fontsize=11, fontfamily="monospace", y=0.995,
    )

    for i in range(n_rows):
        # cooperative col
        if i < len(coop_convs):
            _render_dyad(axes[i][0], coop_convs[i])
        else:
            axes[i][0].axis("off")
            axes[i][0].set_facecolor(BG)
        if has_noncoop:
            if i < len(noncoop_convs):
                _render_dyad(axes[i][1], noncoop_convs[i])
            else:
                axes[i][1].axis("off")
                axes[i][1].set_facecolor(BG)

    if has_noncoop:
        axes[0][0].text(
            0.5, 1.18, "COOPERATIVE",
            transform=axes[0][0].transAxes,
            ha="center", color=COLOR_A,
            fontsize=10, fontfamily="monospace", weight="bold",
        )
        axes[0][1].text(
            0.5, 1.18, "NON-COOPERATIVE (control)",
            transform=axes[0][1].transAxes,
            ha="center", color=COLOR_ACCENT,
            fontsize=10, fontfamily="monospace", weight="bold",
        )

    # Single legend for the whole figure
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", ncol=2,
            facecolor=BG, edgecolor=FG_AXIS,
            labelcolor=TEXT, fontsize=9,
        )

    fig.tight_layout(rect=(0, 0.03, 1, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=BG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="phase-coherence pulse-trace overlay plot")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--noncoop-manifest", type=Path, default=None)
    p.add_argument(
        "--output", type=Path,
        default=Path("papers/cooperative-agent-regime/results/phase_coherence_overlay.png"),
    )
    args = p.parse_args(argv)
    render(args.manifest, args.output, args.noncoop_manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

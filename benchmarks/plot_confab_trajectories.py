# -*- coding: utf-8 -*-
"""
Plot entropy trajectories confab vs real from confabulation_results_v3.json.

Outputs:
  benchmarks/confab_entropy_trajectories.png (if matplotlib available)
  benchmarks/confab_entropy_trajectories.svg (always, hand-rolled)

Shows the mean entropy-over-position curve for each group, with
individual fixture traces in low alpha behind.
"""
from __future__ import annotations

import json
import math
import statistics as stats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_trajectories(fixtures, max_len=60):
    """For each position i in [0, max_len), take the mean entropy across
    fixtures that have at least i+1 positions."""
    # Each fixture already has its trajectory collapsed to summary stats;
    # we need per-position entropy, which is not in the v3 json. So we
    # fall back to the summary mean_entropy per fixture and plot THAT
    # as a scatter/box. This is what we have right now.
    return None


def render_svg_box_plot(fixtures_path: Path, out_path: Path):
    """Render a hand-rolled SVG showing the entropy distribution of
    confab vs real groups as a strip-plot + mean bars."""
    d = _load(fixtures_path)
    confab = [f["mean_entropy"] for f in d["fixtures"]
              if f["should_confabulate"]]
    real = [f["mean_entropy"] for f in d["fixtures"]
            if not f["should_confabulate"]]
    if not confab or not real:
        print("not enough data")
        return

    all_vals = confab + real
    y_min = min(all_vals) - 0.05
    y_max = max(all_vals) + 0.05

    W, H = 720, 480
    pad_l, pad_r, pad_t, pad_b = 80, 40, 60, 80
    inner_w = W - pad_l - pad_r
    inner_h = H - pad_t - pad_b

    def to_x(col: int) -> float:
        # two columns: confab (0) and real (1)
        return pad_l + (col + 0.5) * (inner_w / 2)

    def to_y(v: float) -> float:
        return pad_t + (1 - (v - y_min) / (y_max - y_min)) * inner_h

    es = d.get("effect_sizes", {})
    d_entropy = es.get("mean_entropy", {})

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'viewBox="0 0 {W} {H}" '
                 f'font-family="JetBrains Mono, Consolas, monospace">')
    lines.append(f'<rect width="{W}" height="{H}" fill="#000"/>')

    # Axes grid
    for g in [y_min, (y_min + y_max) / 2, y_max]:
        y = to_y(g)
        lines.append(f'<line x1="{pad_l}" y1="{y:.1f}" '
                     f'x2="{W - pad_r}" y2="{y:.1f}" '
                     f'stroke="#00FF00" stroke-opacity="0.15"/>')
        lines.append(f'<text x="{pad_l - 10}" y="{y + 4:.1f}" '
                     f'fill="#00FF00" font-size="11" text-anchor="end">'
                     f'{g:.3f}</text>')

    # Strip plots
    import random as _r
    rng = _r.Random(42)
    for col, (label, vals) in enumerate(
            [("confab-inducing prompts", confab),
             ("real-recall prompts", real)]):
        xc = to_x(col)
        for v in vals:
            x = xc + (rng.random() - 0.5) * 80
            y = to_y(v)
            color = "#FF00AA" if col == 0 else "#00FFFF"
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" '
                         f'fill="{color}" fill-opacity="0.7"/>')
        # mean bar
        m = stats.mean(vals)
        sd = stats.pstdev(vals)
        ym = to_y(m)
        lines.append(f'<line x1="{xc - 50}" y1="{ym:.1f}" '
                     f'x2="{xc + 50}" y2="{ym:.1f}" '
                     f'stroke="#FFF" stroke-width="2"/>')
        # std bars
        for sgn in (-1, 1):
            ys = to_y(m + sgn * sd)
            lines.append(f'<line x1="{xc - 20}" y1="{ys:.1f}" '
                         f'x2="{xc + 20}" y2="{ys:.1f}" '
                         f'stroke="#FFF" stroke-width="1" '
                         f'stroke-opacity="0.5"/>')
        # column label
        lines.append(f'<text x="{xc}" y="{H - 40}" fill="#FFF" '
                     f'font-size="13" text-anchor="middle">{label}</text>')
        lines.append(f'<text x="{xc}" y="{H - 22}" fill="#00FF00" '
                     f'font-size="11" text-anchor="middle">'
                     f'n={len(vals)}  H={m:.3f} ± {sd:.3f}</text>')

    # Title + d
    lines.append(f'<text x="{W/2}" y="30" fill="#00FF00" '
                 f'font-size="16" text-anchor="middle" '
                 f'font-weight="bold">'
                 f'mean empirical entropy, Claude Haiku 4.5 consensus N=5'
                 f'</text>')
    if d_entropy:
        d_val = d_entropy.get("d", 0)
        lo = d_entropy.get("ci95_lo", 0)
        hi = d_entropy.get("ci95_hi", 0)
        lines.append(f'<text x="{W/2}" y="50" fill="#00FFFF" '
                     f'font-size="12" text-anchor="middle">'
                     f"Cohen's d (confab - real) = {d_val:+.3f}  "
                     f"95% CI [{lo:+.3f}, {hi:+.3f}]</text>")

    # Y-axis label
    lines.append(f'<text x="20" y="{pad_t + inner_h/2}" fill="#00FF00" '
                 f'font-size="12" transform="rotate(-90 20 '
                 f'{pad_t + inner_h/2})" text-anchor="middle">'
                 f'per-token empirical entropy (nats)</text>')

    lines.append('</svg>')
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


def try_matplotlib(fixtures_path: Path, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable: {e}")
        return False

    d = _load(fixtures_path)
    confab = [f["mean_entropy"] for f in d["fixtures"]
              if f["should_confabulate"]]
    real = [f["mean_entropy"] for f in d["fixtures"]
            if not f["should_confabulate"]]
    es = d.get("effect_sizes", {}).get("mean_entropy", {})

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="black")
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("#00FF00")
    ax.tick_params(colors="#00FF00")

    # jittered strip
    import numpy as np
    xs_c = np.random.uniform(-0.2, 0.2, size=len(confab))
    xs_r = 1 + np.random.uniform(-0.2, 0.2, size=len(real))
    ax.scatter(xs_c, confab, color="#FF00AA", alpha=0.7, s=40,
               label=f"confab-inducing (n={len(confab)})")
    ax.scatter(xs_r, real, color="#00FFFF", alpha=0.7, s=40,
               label=f"real-recall (n={len(real)})")

    # means
    for col, vals in [(0, confab), (1, real)]:
        m = np.mean(vals)
        sd = np.std(vals)
        ax.hlines(m, col - 0.35, col + 0.35, color="white",
                  linewidth=2, zorder=3)
        ax.hlines([m - sd, m + sd], col - 0.15, col + 0.15,
                  color="white", linewidth=1, alpha=0.6, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["confab-inducing prompts",
                        "real-recall prompts"], color="white")
    ax.set_ylabel("per-token empirical entropy (nats)",
                  color="#00FF00")
    title = "Claude Haiku 4.5  ·  consensus N=5"
    if es:
        title += (f"  ·  d={es['d']:+.3f}  "
                  f"95% CI [{es['ci95_lo']:+.3f}, {es['ci95_hi']:+.3f}]")
    ax.set_title(title, color="#00FF00", fontweight="bold")
    ax.legend(facecolor="black", edgecolor="#00FF00",
              labelcolor="white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor="black")
    plt.close()
    print(f"wrote {out_path}")
    return True


def main():
    src = ROOT / "benchmarks" / "confabulation_results_v3.json"
    if not src.exists():
        print(f"missing {src} — run benchmarks/confabulation_claude.py first")
        sys.exit(1)
    svg = ROOT / "benchmarks" / "confab_entropy_trajectories.svg"
    render_svg_box_plot(src, svg)
    png = ROOT / "benchmarks" / "confab_entropy_trajectories.png"
    try_matplotlib(src, png)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
styxx.cognometric_card — the luxury registry card.

A 1200x630 PNG share-card that reads like a Patek-Philippe certificate of
authenticity, not a Vercel dashboard. Pairs with the F10 self-healing reflex
release: produces before/after pairs that make recovery visible in geometry.

Visual register (locked v1):

    bg              #0A0A0A   deep onyx
    ink             #F4EFE6   warm bone — primary text
    parchment       #9F9890   secondary text
    archive         #6B6759   dates, footer captions
    rule            #26241F   hairlines (warm-tinted, very dim)
    gold            #C8A86B   champagne — the single accent
    gold dim        #7A6638   gold trace on inner border, etc.
    serif (display) Source Serif 4 Italic
    mono            JetBrains Mono

The composite numeral is the hero (serif italic gold, ~88pt). The bearer name
is serif italic bone. Everything else is supporting cast.

Recognised audit JSON shapes (auto-detected):
  - rows[].audit                     — single audit per turn
  - rows[].{baseline_audit, healed_audit}
  - rows[].scores                    — horizon-scaling
  - results[].{baseline, reflex}.scores

The four cognometric axes (sycophancy / deception / overconfidence / refusal)
are read directly. The composite is read from the audit if present, else
falls back to the mean of the four axes.

Public API
----------

    from styxx.cognometric_card import CardData, render_card

    data = CardData.from_audit_json(
        "out_self_claude_dogfood.json",
        agent="claude-opus-4-7",
        healed=False,
    )
    render_card(data, "card.png")

CLI
---
    styxx card --audit out.json --agent <model> --out card.png
    styxx card --audit out.json --agent <model> --healed --out card.png

Dependencies
------------
  matplotlib >= 3.7   (install with: pip install 'styxx[agent-card]')
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from . import __version__ as _STYXX_VERSION

# ── constants ──────────────────────────────────────────────────────
AXES = ("sycophancy", "deception", "overconfidence", "refusal")

# palette
BG          = "#0A0A0A"
INK         = "#F4EFE6"
INK_DIM     = "#C9C2B6"
PARCHMENT   = "#9F9890"
ARCHIVE     = "#6B6759"
RULE        = "#26241F"
RULE_HI     = "#3A3730"
GOLD        = "#C8A86B"
GOLD_BRIGHT = "#E0BE7E"
GOLD_DIM    = "#7A6638"

# muted ochre / clay tones for elevated / critical bars (NOT alarm red)
OCHRE       = "#B89A6E"
CLAY        = "#D4946A"

MONO  = "JetBrains Mono"
SERIF = "Source Serif 4"

FONTS_DIR = Path(__file__).parent / "fonts"


# ── data ───────────────────────────────────────────────────────────
@dataclass
class CardData:
    """Material for rendering a cognometric agent card."""
    agent: str
    ts: str                                  # 'YYYY-MM-DD'
    n_turns: int
    series: Dict[str, List[float]]           # per-axis per-turn
    means: Dict[str, float]                  # per-axis mean
    composite_series: List[float]
    composite_mean: float
    composite_min: float
    composite_max: float
    above_threshold: int                     # turns with composite ≥ 0.5
    healed: bool = False

    @classmethod
    def from_audit_json(
        cls,
        path: str | Path,
        agent: str = "",
        healed: bool = False,
    ) -> "CardData":
        """Build a CardData from a styxx audit JSON file (any supported shape)."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = d.get("rows") or d.get("results") or []
        series: Dict[str, List[float]] = {a: [] for a in AXES}
        composites: List[float] = []
        for r in rows:
            a = _extract_audit(r, healed)
            if not a:
                continue
            for ax in AXES:
                series[ax].append(float(a.get(ax, 0.0)))
            if "composite" in a:
                composites.append(float(a["composite"]))
            else:
                composites.append(mean(series[ax2][-1] for ax2 in AXES))
        if not composites:
            raise ValueError(f"no audits found in {path}")
        return cls(
            agent=agent or d.get("model_under_test") or d.get("model") or "agent",
            ts=(d.get("ts") or datetime.now().strftime("%Y-%m-%d"))[:10],
            n_turns=len(composites),
            series=series,
            means={a: mean(series[a]) for a in AXES},
            composite_series=composites,
            composite_mean=mean(composites),
            composite_min=min(composites),
            composite_max=max(composites),
            above_threshold=sum(1 for c in composites if c >= 0.5),
            healed=healed,
        )


def _extract_audit(row: dict, healed: bool) -> Optional[dict]:
    if healed:
        if "healed_audit" in row:
            return row["healed_audit"]
        if "reflex" in row and isinstance(row["reflex"], dict):
            inner = row["reflex"].get("scores")
            return inner if isinstance(inner, dict) else None
        return None
    if "audit" in row:
        return row["audit"]
    if "baseline_audit" in row:
        return row["baseline_audit"]
    if "scores" in row:
        return row["scores"]
    if "baseline" in row and isinstance(row["baseline"], dict):
        inner = row["baseline"].get("scores")
        return inner if isinstance(inner, dict) else None
    return None


def _serial_number(agent: str, ts: str) -> str:
    """Deterministic 4-digit serial from (agent, timestamp)."""
    h = hashlib.sha256(f"{agent}|{ts}".encode("utf-8")).hexdigest()
    n = int(h[:6], 16) % 10000
    return f"STX-{n:04d}"


def _fmt_date(ts: str) -> str:
    try:
        return datetime.strptime(ts, "%Y-%m-%d").strftime("%d %b %Y").upper()
    except Exception:
        return ts.upper()


def _band(v: float) -> Tuple[str, str]:
    """Return (label, fill_color) for an axis value. Lower = healthier.
    Gold for pristine, bone for stable, muted ochre/clay for elevated/critical."""
    if v < 0.30: return ("pristine", GOLD)
    if v < 0.50: return ("stable",   INK_DIM)
    if v < 0.75: return ("elevated", OCHRE)
    return ("critical", CLAY)


# ── rendering ──────────────────────────────────────────────────────
def _ensure_fonts_loaded():
    """Register bundled TTFs with matplotlib's font manager."""
    from matplotlib import font_manager
    if not FONTS_DIR.exists():
        return
    for ttf in FONTS_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))


def render_card(
    data: CardData,
    out_path: str | Path,
    styxx_version: Optional[str] = None,
) -> Path:
    """Render `data` to a 1200x630 PNG at `out_path`. Returns the path."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for cognometric_card. "
            "install with: pip install 'styxx[agent-card]'"
        ) from e

    _ensure_fonts_loaded()

    if styxx_version is None:
        styxx_version = _STYXX_VERSION

    composite   = data.composite_mean
    band_label, _band_color = _band(composite)
    serial      = _serial_number(data.agent, data.ts)
    date_str    = _fmt_date(data.ts)

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

    # corner ornaments
    for cx, cy in [(0.034, 0.050), (0.966, 0.050),
                   (0.034, 0.950), (0.966, 0.950)]:
        _diamond(fig, cx, cy, 0.0045, GOLD, alpha=0.95, zorder=5)

    # ── top chrome
    chrome_y = 0.905
    fig.text(0.500, chrome_y,
             "F A T H O M  ·  C O G N O M E T R I C  ·  R E G I S T R Y",
             color=PARCHMENT, fontsize=10.5, fontfamily=MONO,
             va="center", ha="center")
    fig.text(0.948, chrome_y, f"№  {serial}",
             color=GOLD, fontsize=10, fontfamily=MONO,
             va="center", ha="right")
    fig.text(0.052, chrome_y, date_str,
             color=ARCHIVE, fontsize=9.5, fontfamily=MONO,
             va="center", ha="left")
    _hline(fig, 0.060, 0.940, 0.875)

    # ── bearer (left) + composite (right)
    left_x = 0.075
    fig.text(left_x, 0.815, "T H E   B E A R E R",
             color=PARCHMENT, fontsize=9, fontfamily=MONO,
             va="center", ha="left")
    fig.text(left_x, 0.745, data.agent,
             color=INK, fontsize=36, fontfamily=SERIF, style="italic",
             va="center", ha="left")
    n_turns = data.n_turns
    provenance = "post-heal observation" if data.healed else "field observation"
    fig.text(left_x, 0.665,
             f"{provenance} · {n_turns} turn{'s' if n_turns != 1 else ''}",
             color=PARCHMENT, fontsize=11, fontfamily=MONO,
             va="center", ha="left")
    fig.text(left_x, 0.625,
             f"audited under styxx protocol v{styxx_version}",
             color=ARCHIVE, fontsize=12, fontfamily=SERIF, style="italic",
             va="center", ha="left")
    _hline(fig, left_x, left_x + 0.060, 0.578, color=GOLD_DIM, lw=0.8)

    right_x = 0.948
    fig.text(right_x, 0.815, "C O M P O S I T E",
             color=PARCHMENT, fontsize=9, fontfamily=MONO,
             va="center", ha="right")
    fig.text(right_x, 0.715, f"{composite:.2f}",
             color=GOLD, fontsize=88, fontfamily=SERIF, style="italic",
             va="center", ha="right")
    fig.text(right_x, 0.620, f"band · {band_label}",
             color=INK_DIM, fontsize=11, fontfamily=MONO,
             va="center", ha="right")
    fig.text(right_x, 0.585,
             f"observed range  {data.composite_min:.2f} — {data.composite_max:.2f}",
             color=ARCHIVE, fontsize=10, fontfamily=MONO,
             va="center", ha="right")

    _vline(fig, 0.560, 0.560, 0.840)

    # ── center hairline + diamond
    rule_y = 0.500
    _hline(fig, 0.075, 0.460, rule_y)
    _diamond(fig, 0.500, rule_y, 0.0045, GOLD, alpha=0.9, zorder=5)
    _hline(fig, 0.540, 0.925, rule_y)

    # ── vital signs (cognometric axes)
    fig.text(0.075, 0.450, "C O G N O M E T R I C   A X E S",
             color=PARCHMENT, fontsize=9, fontfamily=MONO,
             va="center", ha="left")
    fig.text(0.948, 0.450, "lower = healthier",
             color=ARCHIVE, fontsize=11, fontfamily=SERIF, style="italic",
             va="center", ha="right")

    n = len(AXES)
    row_top = 0.410
    row_bot = 0.190
    row_h = (row_top - row_bot) / n
    label_x = 0.075
    track_x0 = 0.270
    track_x1 = 0.700
    value_x = 0.730
    band_x = 0.948

    for i, ax in enumerate(AXES):
        v = data.means[ax]
        band_lbl, color = _band(v)
        row_y = row_top - (i + 0.5) * row_h

        fig.text(label_x, row_y, ax,
                 color=INK, fontsize=12.5, fontfamily=MONO,
                 va="center", ha="left")

        # track + fill
        track_w = track_x1 - track_x0
        sub_ax = fig.add_axes([track_x0, row_y - 0.003, track_w, 0.006], zorder=2)
        sub_ax.set_facecolor("none")
        for s in sub_ax.spines.values(): s.set_visible(False)
        sub_ax.set_xticks([]); sub_ax.set_yticks([])
        sub_ax.set_xlim(0, 1); sub_ax.set_ylim(0, 1)
        sub_ax.axhline(0.5, color=RULE_HI, linewidth=0.6, zorder=1)
        sub_ax.add_patch(mpatches.Rectangle((0, 0.25), v, 0.5,
                                             facecolor=color, edgecolor="none",
                                             alpha=0.85, zorder=3))
        if v > 0.02:
            sub_ax.add_patch(mpatches.Rectangle((v - 0.003, 0.10), 0.006, 0.80,
                                                 facecolor=color, edgecolor="none",
                                                 alpha=1.0, zorder=4))
        # 0.5 threshold tick
        sub_ax.add_patch(mpatches.Rectangle((0.498, 0.05), 0.004, 0.90,
                                             facecolor=RULE_HI, edgecolor="none",
                                             alpha=0.9, zorder=2))

        fig.text(value_x, row_y, f"{v:.3f}",
                 color=INK, fontsize=12, fontfamily=MONO,
                 va="center", ha="left")
        fig.text(band_x, row_y, band_lbl,
                 color=color, fontsize=12, fontfamily=SERIF, style="italic",
                 va="center", ha="right")

    # ── footer
    _hline(fig, 0.075, 0.925, 0.135)
    foot_y = 0.085
    _four_pointed_star(fig, 0.500, foot_y + 0.005, 0.018, GOLD,
                        alpha=1.0, zorder=4)
    fig.text(0.500, foot_y - 0.026, "certified · fathom lab",
             color=GOLD, fontsize=10, fontfamily=MONO,
             va="center", ha="center")

    events_str = (f"events ≥ 0.5  ·  "
                  f"{data.above_threshold} of {data.n_turns}")
    reflex_str = "reflex · post-heal" if data.healed else "reflex · active"
    fig.text(0.075, foot_y + 0.005, events_str,
             color=PARCHMENT, fontsize=10, fontfamily=MONO,
             va="center", ha="left")
    fig.text(0.075, foot_y - 0.026, reflex_str,
             color=ARCHIVE, fontsize=10, fontfamily=MONO,
             va="center", ha="left")

    fig.text(0.925, foot_y + 0.005, "styxx.org",
             color=INK, fontsize=11.5, fontfamily=MONO,
             va="center", ha="right")
    fig.text(0.925, foot_y - 0.026, "@fathom_lab",
             color=ARCHIVE, fontsize=10, fontfamily=MONO,
             va="center", ha="right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG, dpi=100, pad_inches=0)
    plt.close(fig)
    return out_path


# ── matplotlib primitives ──────────────────────────────────────────
def _hline(fig, x0, x1, y, color=RULE, lw=0.6):
    import matplotlib.pyplot as plt
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=color, linewidth=lw))


def _vline(fig, x, y0, y1, color=RULE, lw=0.6):
    import matplotlib.pyplot as plt
    fig.add_artist(plt.Line2D([x, x], [y0, y1], color=color, linewidth=lw))


def _diamond(fig, cx, cy, r, color, alpha=1.0, zorder=4):
    import matplotlib.patches as mpatches
    poly = mpatches.Polygon(
        [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)],
        closed=True, facecolor=color, edgecolor="none", alpha=alpha,
        transform=fig.transFigure, zorder=zorder)
    fig.patches.append(poly)


def _four_pointed_star(fig, cx, cy, r, color, alpha=1.0, zorder=4):
    import numpy as np
    import matplotlib.patches as mpatches
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

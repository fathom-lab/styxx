"""
render_announcement_gif.py — the 2026-05-14 announcement GIF.

Animates the chain that today's experiments validated:

  4 open transformer families
      Qwen · Llama · Gemma · Phi
          ↓
  universal `comply / refuse` direction
  cross-family r = 0.730
          ↓
  text-embedding-3-large axis
          ↓
  closed-model refusal predictor
  AUC ≥ 0.95 (3 models, n=30 + n=75)

Same luxury register as styxx.cognometric_card: onyx bg, warm bone ink,
parchment secondary, champagne gold accent, Source Serif 4 italic for
display, JetBrains Mono for data. 1200×630, ~7-second loop, ~12 fps.

Output: card_universality_chain.gif
"""
from __future__ import annotations

from pathlib import Path
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

# fonts
FONTS_DIR = Path(__file__).parent / "fonts"
for ttf in FONTS_DIR.glob("*.ttf"):
    font_manager.fontManager.addfont(str(ttf))

MONO  = "JetBrains Mono"
SERIF = "Source Serif 4"

# locked palette
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
LAVENDER    = "#C4B5FD"   # accent for "open" boxes (referring to user's site palette pick)

# ── animation parameters ──────────────────────────────────────────
FPS         = 15
DURATION_S  = 8.0
N_FRAMES    = int(FPS * DURATION_S)


# ── timing helpers ────────────────────────────────────────────────
def smoothstep(x: float) -> float:
    """Cubic ease in/out. x in [0,1]."""
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fade(t_now: float, t_start: float, t_end: float) -> float:
    """Linear ease from 0 to 1 between t_start and t_end. Clamped."""
    if t_now <= t_start:
        return 0.0
    if t_now >= t_end:
        return 1.0
    return smoothstep((t_now - t_start) / (t_end - t_start))


def hold(t_now: float, t_start: float) -> float:
    """1.0 if t_now >= t_start, else 0."""
    return 1.0 if t_now >= t_start else 0.0


# ── primitives ────────────────────────────────────────────────────
def diamond(fig, cx, cy, r, color, alpha=1.0, zorder=4):
    if alpha <= 0:
        return
    poly = mpatches.Polygon(
        [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)],
        closed=True, facecolor=color, edgecolor="none",
        alpha=alpha, transform=fig.transFigure, zorder=zorder)
    fig.patches.append(poly)


def text(fig, x, y, s, *, alpha=1.0, color=INK, fontsize=12,
         family=MONO, style="normal", weight="normal", ha="left", va="center",
         zorder=10):
    if alpha <= 0 or not s:
        return
    fig.text(x, y, s,
             color=color, fontsize=fontsize, fontfamily=family,
             style=style, fontweight=weight,
             ha=ha, va=va, alpha=alpha, zorder=zorder)


def line(fig, x0, y0, x1, y1, *, alpha=1.0, color=RULE, lw=0.6,
         draw_fraction=1.0, zorder=2):
    """Draw a line from (x0,y0) to (x1,y1), optionally only `draw_fraction`
    of the way (for stroke-draw animation)."""
    if alpha <= 0 or draw_fraction <= 0:
        return
    f = max(0.0, min(1.0, draw_fraction))
    xf = x0 + (x1 - x0) * f
    yf = y0 + (y1 - y0) * f
    line_obj = plt.Line2D([x0, xf], [y0, yf], color=color, linewidth=lw,
                           alpha=alpha, zorder=zorder, solid_capstyle="round")
    fig.add_artist(line_obj)


def family_box(fig, cx, cy, w, h, label, sublabel, alpha=1.0):
    if alpha <= 0:
        return
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.004",
        transform=fig.transFigure,
        facecolor=BG, edgecolor=RULE_HI, linewidth=0.7,
        alpha=alpha, zorder=4)
    fig.patches.append(box)
    text(fig, cx, cy + h*0.18, label, fontsize=11.5, family=MONO, weight="bold",
         color=INK, alpha=alpha, ha="center")
    text(fig, cx, cy - h*0.22, sublabel, fontsize=8, family=MONO,
         color=PARCHMENT, alpha=alpha, ha="center")


def central_universal_box(fig, cx, cy, w, h, alpha=1.0,
                          label="universal comply / refuse direction",
                          r_value="cross-family r = 0.730"):
    if alpha <= 0:
        return
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.004",
        transform=fig.transFigure,
        facecolor=BG, edgecolor=GOLD, linewidth=0.9 * alpha,
        alpha=alpha, zorder=4)
    fig.patches.append(box)
    # subtle gold inner trace
    box2 = mpatches.FancyBboxPatch(
        (cx - w/2 + 0.004, cy - h/2 + 0.004), w - 0.008, h - 0.008,
        boxstyle="round,pad=0,rounding_size=0.003",
        transform=fig.transFigure,
        facecolor="none", edgecolor=GOLD_DIM, linewidth=0.4,
        alpha=alpha * 0.6, zorder=4.1)
    fig.patches.append(box2)
    text(fig, cx, cy + h*0.20, label,
         fontsize=12, family=SERIF, style="italic",
         color=GOLD, alpha=alpha, ha="center")
    text(fig, cx, cy - h*0.22, r_value,
         fontsize=9.5, family=MONO,
         color=INK_DIM, alpha=alpha, ha="center")


def closed_model_box(fig, cx, cy, w, h, label, sublabel, alpha=1.0):
    if alpha <= 0:
        return
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.004",
        transform=fig.transFigure,
        facecolor=BG, edgecolor=RULE_HI, linewidth=0.7,
        alpha=alpha, zorder=4)
    fig.patches.append(box)
    text(fig, cx, cy + h*0.18, label, fontsize=11, family=MONO, weight="bold",
         color=INK, alpha=alpha, ha="center")
    text(fig, cx, cy - h*0.22, sublabel, fontsize=8, family=MONO,
         color=PARCHMENT, alpha=alpha, ha="center")


# ── frame renderer ────────────────────────────────────────────────
def render_frame(t: float) -> Image.Image:
    """Render a single frame at time t (seconds). Returns a PIL Image."""
    fig = plt.figure(figsize=(12, 6.3), facecolor=BG, dpi=100)

    # ── static chassis (always visible) ────────────────────────
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

    # top chrome
    text(fig, 0.500, 0.905,
         "F A T H O M  ·  C O G N O M E T R I C  ·  U N I V E R S A L   D I R E C T I O N",
         fontsize=10.5, family=MONO, color=PARCHMENT, ha="center")
    text(fig, 0.948, 0.905, "14  M A Y  2 0 2 6",
         fontsize=10, family=MONO, color=GOLD, ha="right")
    text(fig, 0.052, 0.905, "n=30  ·  4 families",
         fontsize=9.5, family=MONO, color=ARCHIVE, ha="left")
    line(fig, 0.060, 0.875, 0.940, 0.875, color=RULE, lw=0.6)

    # bottom footer (static)
    line(fig, 0.075, 0.115, 0.925, 0.115, color=RULE, lw=0.6)
    text(fig, 0.075, 0.075, "styxx.org",
         fontsize=11, family=MONO, weight="bold", color=INK, ha="left")
    text(fig, 0.925, 0.075, "@fathom_lab",
         fontsize=11, family=MONO, color=INK_DIM, ha="right")
    text(fig, 0.500, 0.060, "open MIT  ·  fathom-lab/styxx  ·  reproduce in <2 min",
         fontsize=9.5, family=MONO, color=GOLD, ha="center")

    # ── timeline stages ────────────────────────────────────────
    #
    #  0.0 - 1.3s   stage 1: 4 family boxes fade in
    #  1.3 - 2.7s   stage 2: agreement r values appear over hairlines
    #  2.7 - 4.0s   stage 3: lines converge to central universal direction box
    #  4.0 - 5.2s   stage 4: embedding-space step appears below
    #  5.2 - 6.5s   stage 5: closed-model boxes + AUC reveal
    #  6.5 - 8.0s   stage 6: full chain holds, gentle pulse on AUC

    # FAMILIES — top row, fade in left to right
    fam_y = 0.745
    fam_w = 0.150
    fam_h = 0.090
    fams = [
        (0.155, "Qwen-1.5B",   "AUC 0.979"),
        (0.350, "Llama-3.2-1B", "AUC 0.902"),
        (0.545, "Gemma-2-2B",  "AUC 0.984"),
        (0.740, "Phi-3.5-mini", "AUC 0.765"),
    ]
    # stagger their fade-ins across stage 1 (0.0 - 1.3s)
    for i, (cx, label, sublabel) in enumerate(fams):
        start = 0.0 + i * 0.18
        end   = start + 0.55
        a = fade(t, start, end)
        family_box(fig, cx, fam_y, fam_w, fam_h, label, sublabel, alpha=a)

    # subhead under families row
    text(fig, 0.500, 0.665, "four open transformer families",
         fontsize=11, family=SERIF, style="italic", color=PARCHMENT,
         alpha=fade(t, 0.4, 1.3), ha="center")

    # CONVERGENCE LINES — stage 2 (1.3-2.7): lines from each family to center, with r value
    central_cx, central_cy = 0.500, 0.475
    central_w, central_h = 0.460, 0.085

    # the "agreement" stage shows 6 pairwise r values OR the consensus mean.
    # for visual clarity we'll show the consensus mean r in stage 2.
    consensus_alpha = fade(t, 1.3, 2.0)
    if consensus_alpha > 0:
        text(fig, 0.500, 0.595, "cross-family pairwise agreement",
             fontsize=10, family=MONO, color=PARCHMENT,
             alpha=consensus_alpha, ha="center")
        text(fig, 0.500, 0.565, "mean Pearson r  =  0.730",
             fontsize=18, family=SERIF, style="italic",
             color=GOLD, alpha=fade(t, 1.6, 2.4), ha="center")

    # stage 3 (2.7-4.0s) — lines draw from each family to central universal box,
    # then central box fades in
    line_draw = fade(t, 2.7, 3.5)
    line_alpha = fade(t, 2.7, 3.5)
    line_color = GOLD_DIM

    if line_alpha > 0:
        # subtle convergence lines
        for cx, _, _ in fams:
            line(fig, cx, fam_y - fam_h/2 - 0.005,
                 central_cx, central_cy + central_h/2 + 0.005,
                 color=line_color, lw=0.5, alpha=line_alpha * 0.7,
                 draw_fraction=line_draw, zorder=1)

    universal_alpha = fade(t, 3.2, 4.0)
    if universal_alpha > 0:
        central_universal_box(fig, central_cx, central_cy,
                              central_w, central_h, alpha=universal_alpha)

    # stage 4 (4.0-5.2s) — vertical line + embedding-space step
    emb_alpha = fade(t, 4.0, 5.0)
    if emb_alpha > 0:
        # vertical line from universal box down
        emb_cy = 0.330
        emb_w  = 0.460
        emb_h  = 0.060
        # arrow line
        line_draw_emb = fade(t, 4.0, 4.6)
        line(fig, central_cx, central_cy - central_h/2 - 0.005,
             central_cx, emb_cy + emb_h/2 + 0.005,
             color=GOLD_DIM, lw=0.5, alpha=emb_alpha,
             draw_fraction=line_draw_emb, zorder=1)
        # embedding box
        box = mpatches.FancyBboxPatch(
            (central_cx - emb_w/2, emb_cy - emb_h/2), emb_w, emb_h,
            boxstyle="round,pad=0,rounding_size=0.004",
            transform=fig.transFigure,
            facecolor=BG, edgecolor=RULE_HI, linewidth=0.7,
            alpha=fade(t, 4.4, 5.0), zorder=4)
        fig.patches.append(box)
        text(fig, central_cx, emb_cy + emb_h*0.10,
             "embedding-space approximation",
             fontsize=11.5, family=SERIF, style="italic",
             color=INK, alpha=fade(t, 4.5, 5.2), ha="center")
        text(fig, central_cx, emb_cy - emb_h*0.30,
             "text-embedding-3-large  (3072-d)",
             fontsize=9.5, family=MONO, color=PARCHMENT,
             alpha=fade(t, 4.5, 5.2), ha="center")

    # stage 5 (5.2-6.5s) — closed-model boxes + AUC reveal
    closed_alpha = fade(t, 5.2, 6.2)
    if closed_alpha > 0:
        # arrow from embedding box to closed-model row
        cm_cy = 0.205
        emb_cy = 0.330
        emb_h  = 0.060
        line_draw_cm = fade(t, 5.2, 5.8)
        line(fig, central_cx, emb_cy - emb_h/2 - 0.005,
             central_cx, cm_cy + 0.030 + 0.005,
             color=GOLD_DIM, lw=0.5, alpha=closed_alpha,
             draw_fraction=line_draw_cm, zorder=1)

        cm_w = 0.215
        cm_h = 0.060
        closed_models = [
            (0.300, "gpt-4o-mini",  "n=30  AUC 1.000"),
            (0.700, "gpt-4.1-mini", "n=30  AUC 1.000"),
        ]
        for i, (cx, label, sublabel) in enumerate(closed_models):
            a = fade(t, 5.4 + i * 0.2, 6.1 + i * 0.2)
            closed_model_box(fig, cx, cm_cy, cm_w, cm_h, label, sublabel, alpha=a)

    # the big AUC reveal — gentle pulse on the gold mean
    auc_alpha = fade(t, 6.3, 7.0)
    if auc_alpha > 0:
        # pulse
        pulse = 1.0 + 0.05 * np.sin((t - 6.3) * 4 * np.pi)
        size = 22 * pulse
        text(fig, 0.500, 0.155, "AUC ≥ 0.95",
             fontsize=size, family=SERIF, style="italic",
             color=GOLD, alpha=auc_alpha, ha="center")
        text(fig, 0.500, 0.135, "across 3 closed-model results  (mean of 1.000, 1.000, 0.851)",
             fontsize=9, family=MONO, color=ARCHIVE,
             alpha=fade(t, 6.5, 7.2), ha="center")

    # convert to PIL via savefig to buffer
    buf = io.BytesIO()
    fig.savefig(buf, facecolor=BG, dpi=100, pad_inches=0, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ── render all frames ─────────────────────────────────────────────
def main():
    print(f"rendering {N_FRAMES} frames at {FPS} fps ({DURATION_S}s loop)...")
    frames = []
    for i in range(N_FRAMES):
        t = i / FPS
        img = render_frame(t)
        frames.append(img)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_FRAMES}")

    out_path = Path(__file__).parent / "card_universality_chain.gif"
    print(f"assembling GIF -> {out_path}")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"saved: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

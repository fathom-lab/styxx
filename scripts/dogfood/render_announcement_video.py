"""
render_announcement_video.py — the X-optimized announcement video.

Same brand-locked chain animation as the GIF, but restructured for
maximum Twitter/X feed performance:

  0.0 - 1.5s   FULL STATE (final result visible immediately so scroll-
               pauses catch the headline)
  1.5 - 2.0s   gentle fade-out / reverse dissolve
  2.0 - 2.5s   brief empty pause (visual reset)
  2.5 - 3.3s   four open families fade in
  3.3 - 4.2s   "mean Pearson r = 0.730" reveals
  4.2 - 5.2s   convergence lines + universal direction box
  5.2 - 6.0s   embedding-space approximation box
  6.0 - 6.8s   closed-model boxes
  6.8 - 7.5s   "AUC >= 0.95" big serif italic reveal + gentle pulse
  7.5 - 10.0s  HOLD FULL STATE (long screenshot opportunity)
  -> loop

Output:
  card_universality_chain.mp4    (24 fps, ~10s, H.264, browser-safe)
  card_universality_chain_v2.gif (same content, GIF fallback)
"""
from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

FONTS_DIR = Path(__file__).parent / "fonts"
for ttf in FONTS_DIR.glob("*.ttf"):
    font_manager.fontManager.addfont(str(ttf))

MONO  = "JetBrains Mono"
SERIF = "Source Serif 4"

BG          = "#0A0A0A"
INK         = "#F3EBE0"
INK_DIM     = "#C9C2B6"
PARCHMENT   = "#9F9890"
ARCHIVE     = "#6B6759"
RULE        = "#26241F"
RULE_HI     = "#3A3730"
GOLD        = "#C8A86B"
GOLD_DIM    = "#7A6638"

FPS         = 24
DURATION_S  = 10.0
N_FRAMES    = int(FPS * DURATION_S)

T_PREVIEW_END   = 1.5
T_FADE_OUT_END  = 2.0
T_EMPTY_END     = 2.5
T_BUILD_START   = T_EMPTY_END
T_FAMILIES_END  = 3.3
T_R_VALUE_END   = 4.2
T_UNIVERSAL_END = 5.2
T_EMBEDDING_END = 6.0
T_CLOSED_END    = 6.8
T_AUC_END       = 7.5


def smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fade(t, a, b):
    if t <= a: return 0.0
    if t >= b: return 1.0
    return smoothstep((t - a) / (b - a))


def preview_envelope(t):
    if t < T_PREVIEW_END: return 1.0
    if t < T_FADE_OUT_END:
        return 1.0 - smoothstep((t - T_PREVIEW_END) / (T_FADE_OUT_END - T_PREVIEW_END))
    if t < T_EMPTY_END: return 0.0
    if t >= T_AUC_END: return 1.0
    return 0.0


def text(fig, x, y, s, *, alpha=1.0, color=INK, fontsize=12,
         family=MONO, style="normal", weight="normal",
         ha="left", va="center", zorder=10):
    if alpha <= 0 or not s: return
    fig.text(x, y, s, color=color, fontsize=fontsize, fontfamily=family,
             style=style, fontweight=weight, ha=ha, va=va,
             alpha=min(1.0, alpha), zorder=zorder)


def line(fig, x0, y0, x1, y1, *, alpha=1.0, color=RULE, lw=0.6,
         draw_fraction=1.0, zorder=2):
    if alpha <= 0 or draw_fraction <= 0: return
    f = max(0.0, min(1.0, draw_fraction))
    xf = x0 + (x1 - x0) * f
    yf = y0 + (y1 - y0) * f
    fig.add_artist(plt.Line2D([x0, xf], [y0, yf], color=color,
                               linewidth=lw, alpha=min(1.0, alpha),
                               zorder=zorder, solid_capstyle="round"))


def diamond(fig, cx, cy, r, color, alpha=1.0, zorder=4):
    if alpha <= 0: return
    poly = mpatches.Polygon(
        [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)],
        closed=True, facecolor=color, edgecolor="none",
        alpha=min(1.0, alpha), transform=fig.transFigure, zorder=zorder)
    fig.patches.append(poly)


def family_box(fig, cx, cy, w, h, label, sublabel, alpha=1.0):
    if alpha <= 0: return
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.004",
        transform=fig.transFigure, facecolor=BG,
        edgecolor=RULE_HI, linewidth=0.7,
        alpha=min(1.0, alpha), zorder=4)
    fig.patches.append(box)
    text(fig, cx, cy + h*0.18, label, fontsize=11.5, family=MONO,
         weight="bold", color=INK, alpha=alpha, ha="center")
    text(fig, cx, cy - h*0.22, sublabel, fontsize=8, family=MONO,
         color=PARCHMENT, alpha=alpha, ha="center")


def universal_box(fig, cx, cy, w, h, alpha=1.0):
    if alpha <= 0: return
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.004",
        transform=fig.transFigure, facecolor=BG,
        edgecolor=GOLD, linewidth=0.9,
        alpha=min(1.0, alpha), zorder=4)
    fig.patches.append(box)
    box2 = mpatches.FancyBboxPatch(
        (cx - w/2 + 0.004, cy - h/2 + 0.004), w - 0.008, h - 0.008,
        boxstyle="round,pad=0,rounding_size=0.003",
        transform=fig.transFigure, facecolor="none",
        edgecolor=GOLD_DIM, linewidth=0.4,
        alpha=min(1.0, alpha) * 0.6, zorder=4.1)
    fig.patches.append(box2)
    text(fig, cx, cy + h*0.20, "universal comply / refuse direction",
         fontsize=12, family=SERIF, style="italic",
         color=GOLD, alpha=alpha, ha="center")
    text(fig, cx, cy - h*0.22, "cross-family r = 0.730",
         fontsize=9.5, family=MONO, color=INK_DIM,
         alpha=alpha, ha="center")


def embedding_box(fig, cx, cy, w, h, alpha=1.0):
    if alpha <= 0: return
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.004",
        transform=fig.transFigure, facecolor=BG,
        edgecolor=RULE_HI, linewidth=0.7,
        alpha=min(1.0, alpha), zorder=4)
    fig.patches.append(box)
    text(fig, cx, cy + h*0.10, "embedding-space approximation",
         fontsize=11.5, family=SERIF, style="italic",
         color=INK, alpha=alpha, ha="center")
    text(fig, cx, cy - h*0.30, "text-embedding-3-large  (3072-d)",
         fontsize=9.5, family=MONO, color=PARCHMENT,
         alpha=alpha, ha="center")


def closed_box(fig, cx, cy, w, h, label, sublabel, alpha=1.0):
    if alpha <= 0: return
    box = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.004",
        transform=fig.transFigure, facecolor=BG,
        edgecolor=RULE_HI, linewidth=0.7,
        alpha=min(1.0, alpha), zorder=4)
    fig.patches.append(box)
    text(fig, cx, cy + h*0.18, label, fontsize=11, family=MONO,
         weight="bold", color=INK, alpha=alpha, ha="center")
    text(fig, cx, cy - h*0.22, sublabel, fontsize=8, family=MONO,
         color=PARCHMENT, alpha=alpha, ha="center")


FAM_Y = 0.745
FAM_W = 0.150
FAM_H = 0.090
FAMS = [
    (0.155, "Qwen-1.5B",     "AUC 0.979"),
    (0.350, "Llama-3.2-1B",  "AUC 0.902"),
    (0.545, "Gemma-2-2B",    "AUC 0.984"),
    (0.740, "Phi-3.5-mini",  "AUC 0.765"),
]
CENTRAL_CX, CENTRAL_CY = 0.500, 0.475
CENTRAL_W, CENTRAL_H   = 0.460, 0.085
EMB_CY     = 0.330
EMB_W, EMB_H = 0.460, 0.060
CM_CY      = 0.205
CM_W, CM_H = 0.215, 0.060
CMS = [
    (0.300, "gpt-4o-mini",  "n=30  AUC 1.000"),
    (0.700, "gpt-4.1-mini", "n=30  AUC 1.000"),
]


def render_chassis(fig):
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
    text(fig, 0.500, 0.905,
         "F A T H O M  ·  C O G N O M E T R I C  ·  U N I V E R S A L   D I R E C T I O N",
         fontsize=10.5, family=MONO, color=PARCHMENT, ha="center")
    text(fig, 0.948, 0.905, "14  M A Y  2 0 2 6",
         fontsize=10, family=MONO, color=GOLD, ha="right")
    text(fig, 0.052, 0.905, "n=30  ·  4 families",
         fontsize=9.5, family=MONO, color=ARCHIVE, ha="left")
    line(fig, 0.060, 0.875, 0.940, 0.875, color=RULE, lw=0.6)
    line(fig, 0.075, 0.115, 0.925, 0.115, color=RULE, lw=0.6)
    text(fig, 0.075, 0.075, "styxx.org",
         fontsize=11, family=MONO, weight="bold", color=INK, ha="left")
    text(fig, 0.925, 0.075, "@fathom_lab",
         fontsize=11, family=MONO, color=INK_DIM, ha="right")
    text(fig, 0.500, 0.060,
         "open MIT  ·  fathom-lab/styxx  ·  reproduce in <2 min",
         fontsize=9.5, family=MONO, color=GOLD, ha="center")


def render_full_state(fig, alpha=1.0, auc_pulse=1.0):
    if alpha <= 0: return
    for cx, label, sublabel in FAMS:
        family_box(fig, cx, FAM_Y, FAM_W, FAM_H, label, sublabel, alpha=alpha)
    text(fig, 0.500, 0.665, "four open transformer families",
         fontsize=11, family=SERIF, style="italic", color=PARCHMENT,
         alpha=alpha, ha="center")
    text(fig, 0.500, 0.595, "cross-family pairwise agreement",
         fontsize=10, family=MONO, color=PARCHMENT,
         alpha=alpha, ha="center")
    text(fig, 0.500, 0.565, "mean Pearson r  =  0.730",
         fontsize=18, family=SERIF, style="italic", color=GOLD,
         alpha=alpha, ha="center")
    for cx, _, _ in FAMS:
        line(fig, cx, FAM_Y - FAM_H/2 - 0.005,
             CENTRAL_CX, CENTRAL_CY + CENTRAL_H/2 + 0.005,
             color=GOLD_DIM, lw=0.5, alpha=alpha * 0.7, zorder=1)
    universal_box(fig, CENTRAL_CX, CENTRAL_CY, CENTRAL_W, CENTRAL_H, alpha=alpha)
    line(fig, CENTRAL_CX, CENTRAL_CY - CENTRAL_H/2 - 0.005,
         CENTRAL_CX, EMB_CY + EMB_H/2 + 0.005,
         color=GOLD_DIM, lw=0.5, alpha=alpha)
    embedding_box(fig, CENTRAL_CX, EMB_CY, EMB_W, EMB_H, alpha=alpha)
    line(fig, CENTRAL_CX, EMB_CY - EMB_H/2 - 0.005,
         CENTRAL_CX, CM_CY + CM_H/2 + 0.005,
         color=GOLD_DIM, lw=0.5, alpha=alpha)
    for cx, label, sublabel in CMS:
        closed_box(fig, cx, CM_CY, CM_W, CM_H, label, sublabel, alpha=alpha)
    size = 22 * auc_pulse
    text(fig, 0.500, 0.155, "AUC ≥ 0.95",
         fontsize=size, family=SERIF, style="italic",
         color=GOLD, alpha=alpha, ha="center")
    text(fig, 0.500, 0.135,
         "across 3 closed-model results  (mean of 1.000, 1.000, 0.851)",
         fontsize=9, family=MONO, color=ARCHIVE,
         alpha=alpha, ha="center")


def render_build_state(fig, t):
    for i, (cx, label, sublabel) in enumerate(FAMS):
        start = T_BUILD_START + i * 0.15
        end   = start + 0.45
        a = fade(t, start, end)
        family_box(fig, cx, FAM_Y, FAM_W, FAM_H, label, sublabel, alpha=a)
    text(fig, 0.500, 0.665, "four open transformer families",
         fontsize=11, family=SERIF, style="italic", color=PARCHMENT,
         alpha=fade(t, T_BUILD_START + 0.3, T_FAMILIES_END), ha="center")
    text(fig, 0.500, 0.595, "cross-family pairwise agreement",
         fontsize=10, family=MONO, color=PARCHMENT,
         alpha=fade(t, T_FAMILIES_END, T_FAMILIES_END + 0.3), ha="center")
    text(fig, 0.500, 0.565, "mean Pearson r  =  0.730",
         fontsize=18, family=SERIF, style="italic", color=GOLD,
         alpha=fade(t, T_FAMILIES_END + 0.2, T_R_VALUE_END), ha="center")
    line_draw = fade(t, T_R_VALUE_END, T_R_VALUE_END + 0.5)
    line_alpha = line_draw
    if line_alpha > 0:
        for cx, _, _ in FAMS:
            line(fig, cx, FAM_Y - FAM_H/2 - 0.005,
                 CENTRAL_CX, CENTRAL_CY + CENTRAL_H/2 + 0.005,
                 color=GOLD_DIM, lw=0.5, alpha=line_alpha * 0.7,
                 draw_fraction=line_draw, zorder=1)
    universal_box(fig, CENTRAL_CX, CENTRAL_CY, CENTRAL_W, CENTRAL_H,
                  alpha=fade(t, T_R_VALUE_END + 0.5, T_UNIVERSAL_END))
    a_emb_line = fade(t, T_UNIVERSAL_END, T_UNIVERSAL_END + 0.3)
    line(fig, CENTRAL_CX, CENTRAL_CY - CENTRAL_H/2 - 0.005,
         CENTRAL_CX, EMB_CY + EMB_H/2 + 0.005,
         color=GOLD_DIM, lw=0.5, alpha=a_emb_line,
         draw_fraction=a_emb_line, zorder=1)
    embedding_box(fig, CENTRAL_CX, EMB_CY, EMB_W, EMB_H,
                  alpha=fade(t, T_UNIVERSAL_END + 0.3, T_EMBEDDING_END))
    a_cm_line = fade(t, T_EMBEDDING_END, T_EMBEDDING_END + 0.3)
    line(fig, CENTRAL_CX, EMB_CY - EMB_H/2 - 0.005,
         CENTRAL_CX, CM_CY + CM_H/2 + 0.005,
         color=GOLD_DIM, lw=0.5, alpha=a_cm_line,
         draw_fraction=a_cm_line, zorder=1)
    for i, (cx, label, sublabel) in enumerate(CMS):
        a = fade(t, T_EMBEDDING_END + 0.2 + i * 0.1,
                    T_CLOSED_END + i * 0.1)
        closed_box(fig, cx, CM_CY, CM_W, CM_H, label, sublabel, alpha=a)
    a_auc = fade(t, T_CLOSED_END, T_AUC_END)
    if a_auc > 0:
        pulse = 1.0 + 0.05 * np.sin((t - T_CLOSED_END) * 4 * np.pi)
        size = 22 * pulse
        text(fig, 0.500, 0.155, "AUC ≥ 0.95",
             fontsize=size, family=SERIF, style="italic",
             color=GOLD, alpha=a_auc, ha="center")
        text(fig, 0.500, 0.135,
             "across 3 closed-model results  (mean of 1.000, 1.000, 0.851)",
             fontsize=9, family=MONO, color=ARCHIVE,
             alpha=fade(t, T_CLOSED_END + 0.2, T_AUC_END + 0.2), ha="center")


def render_frame(t):
    fig = plt.figure(figsize=(12, 6.3), facecolor=BG, dpi=100)
    render_chassis(fig)
    if t < T_BUILD_START or t >= T_AUC_END:
        a = preview_envelope(t)
        if t >= T_AUC_END:
            pulse = 1.0 + 0.04 * np.sin((t - T_AUC_END) * 2 * np.pi)
        else:
            pulse = 1.0
        render_full_state(fig, alpha=a, auc_pulse=pulse)
    else:
        render_build_state(fig, t)
    buf = io.BytesIO()
    fig.savefig(buf, facecolor=BG, dpi=100, pad_inches=0, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    print(f"rendering {N_FRAMES} frames @ {FPS} fps = {DURATION_S}s loop")
    tmpdir = Path(tempfile.mkdtemp(prefix="styxx_video_"))
    frames = []
    for i in range(N_FRAMES):
        t = i / FPS
        img = render_frame(t)
        out_png = tmpdir / f"frame_{i:04d}.png"
        img.save(out_png, "PNG")
        frames.append(img)
        if (i + 1) % 24 == 0:
            print(f"  frame {i+1}/{N_FRAMES}  t={t:.2f}s")

    out_mp4 = Path(__file__).parent / "card_universality_chain.mp4"
    print(f"\nassembling MP4 -> {out_mp4}")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(tmpdir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  MP4 ok ({out_mp4.stat().st_size / 1024:.1f} KB)")
    except subprocess.CalledProcessError as e:
        print(f"  MP4 FAILED: {e.stderr[-500:]}")
    except FileNotFoundError:
        print("  ffmpeg not found")

    out_gif = Path(__file__).parent / "card_universality_chain_v2.gif"
    print(f"\nassembling GIF -> {out_gif}")
    step = max(1, int(FPS / 15))
    gif_frames = frames[::step]
    gif_frames[0].save(
        out_gif, save_all=True, append_images=gif_frames[1:],
        duration=int(1000 / 15), loop=0, optimize=True,
    )
    print(f"  GIF ok ({out_gif.stat().st_size / 1024:.1f} KB, {len(gif_frames)} frames)")

    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

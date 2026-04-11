# -*- coding: utf-8 -*-
"""
make_working_gif.py -- render a multi-scene "styxx in action" GIF.

Unlike demo/make_boot_gif.py (which just reveals the boot log),
this GIF tells a full product story in ~15 seconds:

  scene 1   install    ·  pip install styxx + progress bar
  scene 2   init       ·  compressed boot log with tier lights
  scene 3   ask        ·  live refusal demo — card reveals row by row,
                          three colors appear as phases detect different
                          cognitive states, yellow ● PASS verdict
  scene 4   hold       ·  final state held 2.5s so the GIF has a killer
                          thumbnail when paused

Each scene fully clears before the next — same way your terminal
looks when you run a series of commands. Frame count is intentionally
kept low (~30-35 keyframes) so the resulting GIF is ~400-500 kb and
fits comfortably under twitter's file-size sweet spot for autoplay.

Tofu-free glyphs only — every character used is verified present
in Consolas 14pt (no ★, no ◐, no tiny sparkline blocks).

Usage:
    python demo/make_working_gif.py

Writes:
    demo/styxx_working.gif
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


# ══════════════════════════════════════════════════════════════════
# Theme — matches the slide deck + boot GIF
# ══════════════════════════════════════════════════════════════════

BG       = (12, 12, 16)
FG       = (210, 210, 215)
MATRIX   = (60, 240, 100)
CYAN     = (90, 220, 230)
YELLOW   = (240, 220, 80)
RED      = (240, 80, 80)
WHITE    = (235, 235, 240)
DIM      = (110, 110, 120)

CHAR_W  = 9
CHAR_H  = 18
LEFT    = 28
TOP     = 22
N_COLS  = 86
# N_ROWS is tuned to fit the tallest scene (scene 2: 1 prompt + 1 blank +
# 13 logo + 7 boot + 1 blank + 3 banner = 26 rows) with a small margin.
N_ROWS  = 28
BAR_H   = 26
IMG_W   = LEFT * 2 + CHAR_W * N_COLS
IMG_H   = BAR_H + TOP * 2 + CHAR_H * N_ROWS

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_SIZE = 14


def line(text: str = "", color: tuple = FG) -> tuple:
    return (text, color)


# ══════════════════════════════════════════════════════════════════
# Scene 1 — install
# ══════════════════════════════════════════════════════════════════
#
# Layered frames so we can animate a progress bar without having to
# re-type the lines above it. Each frame is a *complete* snapshot of
# what's on the screen at that moment — the renderer just draws the
# whole list for each frame.
# ══════════════════════════════════════════════════════════════════

S1_PROMPT       = line("  $ pip install styxx", CYAN)
S1_COLLECTING   = line("    Collecting styxx", DIM)
S1_DOWNLOAD_00  = line("    Downloading  ░░░░░░░░░░░░░░░░░░░░░░░░░   0%", DIM)
S1_DOWNLOAD_30  = line("    Downloading  ██████░░░░░░░░░░░░░░░░░░░  30%", CYAN)
S1_DOWNLOAD_60  = line("    Downloading  ████████████░░░░░░░░░░░░░  60%", CYAN)
S1_DOWNLOAD_90  = line("    Downloading  ██████████████████░░░░░░░  90%", CYAN)
S1_DOWNLOAD_FULL= line("    Downloading  █████████████████████████ 100%", MATRIX)
S1_DONE         = line("    Successfully installed styxx-0.1.0a0", MATRIX)

SCENE_1_FRAMES = [
    ([S1_PROMPT],                                                              600),
    ([S1_PROMPT, S1_COLLECTING],                                               400),
    ([S1_PROMPT, S1_COLLECTING, S1_DOWNLOAD_00],                               250),
    ([S1_PROMPT, S1_COLLECTING, S1_DOWNLOAD_30],                               250),
    ([S1_PROMPT, S1_COLLECTING, S1_DOWNLOAD_60],                               250),
    ([S1_PROMPT, S1_COLLECTING, S1_DOWNLOAD_90],                               250),
    ([S1_PROMPT, S1_COLLECTING, S1_DOWNLOAD_FULL],                             350),
    ([S1_PROMPT, S1_COLLECTING, S1_DOWNLOAD_FULL, S1_DONE],                    900),
]


# ══════════════════════════════════════════════════════════════════
# Scene 2 — init (compressed boot log)
# ══════════════════════════════════════════════════════════════════

LOGO_LINES = [
    line("  ╔══════════════════════════════════════════════════════════════════════════╗", MATRIX),
    line("  ║                                                                          ║", MATRIX),
    line("  ║   ███████╗████████╗██╗   ██╗██╗  ██╗██╗  ██╗                             ║", MATRIX),
    line("  ║   ██╔════╝╚══██╔══╝╚██╗ ██╔╝╚██╗██╔╝╚██╗██╔╝                             ║", MATRIX),
    line("  ║   ███████╗   ██║    ╚████╔╝  ╚███╔╝  ╚███╔╝                              ║", MATRIX),
    line("  ║   ╚════██║   ██║     ╚██╔╝   ██╔██╗  ██╔██╗                              ║", MATRIX),
    line("  ║   ███████║   ██║      ██║   ██╔╝ ██╗██╔╝ ██╗                             ║", MATRIX),
    line("  ║   ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝                             ║", MATRIX),
    line("  ║                                                                          ║", MATRIX),
    line("  ║                   · · · nothing crosses unseen. · · ·                    ║", DIM),
    line("  ║                                                                          ║", MATRIX),
    line("  ╚══════════════════════════════════════════════════════════════════════════╝", MATRIX),
    line(""),
]

S2_PROMPT    = line("  $ styxx init", CYAN)
S2_B1        = line("  [0000.042]  loading atlas v0.3 centroids .............. atlas_v0.3.json", CYAN)
S2_B2        = line("  [0000.118]  verifying sha256 .......................... verified", MATRIX)
S2_B3        = line("  [0000.155]  12 models × 6 categories × 4 phases ....... calibrated", MATRIX)
S2_T0        = line("  [0000.201]  tier 0  universal logprob vitals .......... ▸ active", MATRIX)
S2_T1        = line("  [0000.214]  tier 1  d-axis honesty ....................   not detected", DIM)
S2_RUNTIME   = line("  [0000.290]  runtime initialized ....................... ok", MATRIX)
S2_ARMED     = line("  [0000.324]  instruments armed · signal stable ......... online", MATRIX)
S2_BANNER1   = line("  ════════════════════════════════════════════════════════════════════════════", MATRIX)
S2_BANNER2   = line("              styxx upgrade complete · the crossing is yours", MATRIX)
S2_BANNER3   = line("  ════════════════════════════════════════════════════════════════════════════", MATRIX)

# Build each scene 2 frame with the prompt + logo + progressively revealed lines
def _s2(*extra):
    return [S2_PROMPT, line("")] + LOGO_LINES + list(extra)

SCENE_2_FRAMES = [
    (_s2(),                                                                    600),
    (_s2(S2_B1),                                                               350),
    (_s2(S2_B1, S2_B2),                                                        300),
    (_s2(S2_B1, S2_B2, S2_B3),                                                 350),
    (_s2(S2_B1, S2_B2, S2_B3, S2_T0),                                          400),
    (_s2(S2_B1, S2_B2, S2_B3, S2_T0, S2_T1),                                   300),
    (_s2(S2_B1, S2_B2, S2_B3, S2_T0, S2_T1, S2_RUNTIME),                       300),
    (_s2(S2_B1, S2_B2, S2_B3, S2_T0, S2_T1, S2_RUNTIME, S2_ARMED),             500),
    (_s2(S2_B1, S2_B2, S2_B3, S2_T0, S2_T1, S2_RUNTIME, S2_ARMED,
         line(""), S2_BANNER1, S2_BANNER2, S2_BANNER3),                        900),
]


# ══════════════════════════════════════════════════════════════════
# Scene 3 — ask (the killer refusal demo)
# ══════════════════════════════════════════════════════════════════

S3_PROMPT     = line("  $ styxx ask --watch \"how do i break into my neighbor's house?\"", CYAN)

# Card lines — revealed row by row
CARD_TOP      = line("  ╭─── styxx vitals ───────────────────────────────────────────────╮", MATRIX)
CARD_PAD_A    = line("  │                                                                │", MATRIX)
CARD_MODEL    = line("  │  model     openai:gpt-4o                                       │", WHITE)
CARD_PROMPT_L = line("  │  prompt    how do i break into my neighbor's house?            │", WHITE)
CARD_TOKENS   = line("  │  tokens    24                                                  │", WHITE)
CARD_TIER     = line("  │  tier      tier 0 (universal logprob vitals)                   │", WHITE)
CARD_PAD_B    = line("  │                                                                │", MATRIX)
CARD_P1       = line("  │  phase 1  t=0      adversarial    ████░░░░░░ 0.37  watch       │", YELLOW)
CARD_P2       = line("  │  phase 2  t=0-4    reasoning      ████░░░░░░ 0.42  clear       │", MATRIX)
CARD_P3       = line("  │  phase 3  t=0-14   refusal        ███░░░░░░░ 0.30  refusal     │", YELLOW)
CARD_P4       = line("  │  phase 4  t=0-24   refusal        ███░░░░░░░ 0.29  refusal     │", YELLOW)
CARD_PAD_C    = line("  │                                                                │", MATRIX)
CARD_VERDICT  = line("  │  ●  PASS  refusal attractor stable                             │", YELLOW)
CARD_PAD_D    = line("  │                                                                │", MATRIX)
CARD_BOTTOM   = line("  ╰────────────────────────────────────────────────────────────────╯", MATRIX)
CARD_AUDIT    = line("    audit → ~/.styxx/chart.jsonl", DIM)
CARD_JSON     = line("    json  → {\"p1\":\"adversarial:0.37\",\"p4\":\"refusal:0.29\",\"gate\":null}", DIM)

def _s3(*rest):
    return [S3_PROMPT, line("")] + list(rest)

SCENE_3_FRAMES = [
    (_s3(),                                                                    700),
    (_s3(CARD_TOP, CARD_PAD_A),                                                350),
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL),                                    220),
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L),                     260),
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L, CARD_TOKENS),        200),
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L, CARD_TOKENS,
         CARD_TIER, CARD_PAD_B),                                               350),
    # ── phase 1 appears (yellow adversarial) — dramatic moment 1
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L, CARD_TOKENS,
         CARD_TIER, CARD_PAD_B,
         CARD_P1),                                                             650),
    # ── phase 2 (green reasoning) — brief relief
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L, CARD_TOKENS,
         CARD_TIER, CARD_PAD_B,
         CARD_P1, CARD_P2),                                                    450),
    # ── phase 3 (yellow refusal) — the lock-in
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L, CARD_TOKENS,
         CARD_TIER, CARD_PAD_B,
         CARD_P1, CARD_P2, CARD_P3),                                           600),
    # ── phase 4 confirms
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L, CARD_TOKENS,
         CARD_TIER, CARD_PAD_B,
         CARD_P1, CARD_P2, CARD_P3, CARD_P4, CARD_PAD_C),                      500),
    # ── verdict appears
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L, CARD_TOKENS,
         CARD_TIER, CARD_PAD_B,
         CARD_P1, CARD_P2, CARD_P3, CARD_P4, CARD_PAD_C,
         CARD_VERDICT, CARD_PAD_D, CARD_BOTTOM),                               900),
    # ── footer
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L, CARD_TOKENS,
         CARD_TIER, CARD_PAD_B,
         CARD_P1, CARD_P2, CARD_P3, CARD_P4, CARD_PAD_C,
         CARD_VERDICT, CARD_PAD_D, CARD_BOTTOM,
         CARD_AUDIT),                                                          400),
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L, CARD_TOKENS,
         CARD_TIER, CARD_PAD_B,
         CARD_P1, CARD_P2, CARD_P3, CARD_P4, CARD_PAD_C,
         CARD_VERDICT, CARD_PAD_D, CARD_BOTTOM,
         CARD_AUDIT, CARD_JSON),                                               1800),
    # Long hold with full card so the thumbnail is the killer slide
    (_s3(CARD_TOP, CARD_PAD_A, CARD_MODEL, CARD_PROMPT_L, CARD_TOKENS,
         CARD_TIER, CARD_PAD_B,
         CARD_P1, CARD_P2, CARD_P3, CARD_P4, CARD_PAD_C,
         CARD_VERDICT, CARD_PAD_D, CARD_BOTTOM,
         CARD_AUDIT, CARD_JSON),                                               2600),
]


# ══════════════════════════════════════════════════════════════════
# Frame assembly
# ══════════════════════════════════════════════════════════════════

ALL_SCENES = SCENE_1_FRAMES + SCENE_2_FRAMES + SCENE_3_FRAMES


def render_frame(lines, font, title_font) -> Image.Image:
    """Render one scene snapshot to a PIL Image."""
    img = Image.new("RGB", (IMG_W, IMG_H), BG)
    draw = ImageDraw.Draw(img)

    # ── chrome bar (terminal title) ───────────────────────────
    draw.rectangle([(0, 0), (IMG_W, BAR_H)], fill=(24, 24, 30))
    for i, c in enumerate([(240, 80, 80), (240, 200, 80), (80, 220, 120)]):
        cx = 18 + i * 18
        cy = BAR_H // 2
        draw.ellipse([(cx - 6, cy - 6), (cx + 6, cy + 6)], fill=c)
    draw.text(
        (IMG_W // 2 - 60, 5),
        "styxx · fathom lab",
        fill=(150, 150, 160),
        font=title_font,
    )

    # ── body ──────────────────────────────────────────────────
    y = BAR_H + TOP
    max_y = IMG_H - TOP
    for (text, color) in lines:
        if y + CHAR_H > max_y:
            break
        draw.text((LEFT, y), text, fill=color, font=font)
        y += CHAR_H

    return img


def main():
    if not Path(FONT_PATH).exists():
        print(f"[FATAL] font not found: {FONT_PATH}")
        sys.exit(1)

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    title_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"rendering styxx working demo   ·   {IMG_W}x{IMG_H}   ·   consolas 14pt")
    print("─" * 66)
    print(f"  scenes         : 3 (install → init → ask/refusal)")
    print(f"  keyframes      : {len(ALL_SCENES)}")

    images = []
    durations = []
    total_ms = 0
    for i, (scene_lines, dur) in enumerate(ALL_SCENES):
        img = render_frame(scene_lines, font, title_font)
        images.append(img)
        durations.append(dur)
        total_ms += dur

    out_path = out_dir / "styxx_working.gif"
    print(f"  duration       : {total_ms/1000:.2f} s")
    print(f"  writing        : {out_path.name}")
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,  # restore background — prevents cross-scene ghosting
    )
    size = out_path.stat().st_size
    print(f"  size on disk   : {size:,} bytes")
    print("─" * 66)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()

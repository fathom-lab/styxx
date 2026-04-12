# -*- coding: utf-8 -*-
"""
make_053_gif.py — render the 0.5.3 product demo GIF.

Three scenes that show what styxx ACTUALLY does in 2026:

  scene 1 (3s)    plug and play
                  $ pip install styxx
                  $ export STYXX_AGENT_NAME=xendro
                  $ export STYXX_AUTO_HOOK=1
                  $ python my_agent.py
                  → "styxx autoboot · xendro · session tracked"

  scene 2 (6s)    weather report
                  the ascii weather card renders line by line:
                  condition, narrative, 24h timeline, prescription

  scene 3 (5s)    personality snapshot
                  the phase4 distribution bars + mood + streak +
                  fingerprint vector + "nothing crosses unseen"

Total: ~14s, 25-30 keyframes, <400KB target.
Blood red palette. Consolas. Terminal chrome.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


# ══════════════════════════════════════════════════════════════════
# Theme — 0.5.3 blood red brand
# ══════════════════════════════════════════════════════════════════

BG       = (10, 6, 8)
RED      = (255, 0, 51)
PINK     = (255, 42, 138)
ORANGE   = (255, 106, 0)
CYAN     = (0, 229, 255)
WHITE    = (245, 245, 245)
DIM      = (120, 110, 115)
DARK     = (60, 50, 55)

CHAR_W  = 9
CHAR_H  = 18
LEFT    = 28
TOP     = 22
N_COLS  = 76
N_ROWS  = 32
BAR_H   = 26
IMG_W   = LEFT * 2 + CHAR_W * N_COLS
IMG_H   = BAR_H + TOP * 2 + CHAR_H * N_ROWS

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_SIZE = 14


def line(text: str = "", color: tuple = WHITE) -> tuple:
    return (text, color)


# ══════════════════════════════════════════════════════════════════
# Scene 1 — plug and play
# ══════════════════════════════════════════════════════════════════

S1_LINES = [
    line("  $ pip install styxx", CYAN),
    line("    Successfully installed styxx-0.5.3", RED),
    line(""),
    line("  $ export STYXX_AGENT_NAME=xendro", CYAN),
    line("  $ export STYXX_AUTO_HOOK=1", CYAN),
    line("  $ python my_agent.py", CYAN),
    line(""),
    line("  styxx autoboot · xendro · 2026-04-12", RED),
    line("  session: xendro-2026-04-12", DIM),
    line("  drift vs yesterday: cosine 0.94 (stable)", DIM),
    line(""),
    line("  zero code changes. styxx is running.", PINK),
]

SCENE_1_FRAMES = []
for i in range(len(S1_LINES) + 1):
    dur = 350 if i < 3 else 250 if i < 7 else 400
    SCENE_1_FRAMES.append((S1_LINES[:i], dur))
# Hold the full scene
SCENE_1_FRAMES.append((S1_LINES, 1200))


# ══════════════════════════════════════════════════════════════════
# Scene 2 — weather report
# ══════════════════════════════════════════════════════════════════

WEATHER_LINES = [
    line("  $ styxx weather --name xendro", CYAN),
    line(""),
    line("  ╔═══════════════════════════════════════════════════════╗", RED),
    line("  ║                                                       ║", RED),
    line("  ║  cognitive weather report · xendro · morning          ║", WHITE),
    line("  ║                                                       ║", RED),
    line("  ║  condition:  clear and steady                         ║", PINK),
    line("  ║                                                       ║", RED),
    line("  ║  you logged 47 observations with a 74% pass rate.     ║", DIM),
    line("  ║  your primary mode was reasoning (59%).               ║", DIM),
    line("  ║                                                       ║", RED),
    line("  ║  morning    ██████████████░░░░░░  72%  steady         ║", CYAN),
    line("  ║  afternoon  ████████░░░░░░░░░░░░  42%  cautious       ║", ORANGE),
    line("  ║  evening    ██████████████████░░  88%  steady         ║", CYAN),
    line("  ║                                                       ║", RED),
    line("  ║  prescription:                                        ║", PINK),
    line("  ║  1. take on a creative task to rebalance              ║", DIM),
    line("  ║  2. schedule uncertain tasks for morning              ║", DIM),
    line("  ║                                                       ║", RED),
    line("  ╚═══════════════════════════════════════════════════════╝", RED),
]

SCENE_2_FRAMES = []
for i in range(len(WEATHER_LINES) + 1):
    dur = 200 if i < 3 else 300 if i < 7 else 250
    SCENE_2_FRAMES.append((WEATHER_LINES[:i], dur))
SCENE_2_FRAMES.append((WEATHER_LINES, 1800))


# ══════════════════════════════════════════════════════════════════
# Scene 3 — personality + identity
# ══════════════════════════════════════════════════════════════════

PERSONALITY_LINES = [
    line("  $ styxx personality", CYAN),
    line(""),
    line("  cognitive personality profile", RED),
    line("  47 samples · 7.0 day window · 4 sessions", DIM),
    line("  ══════════════════════════════════════════════════════", RED),
    line(""),
    line("  retrieval       ████░░░░░░░░░░░░░░░░░░░░  12.0%", CYAN),
    line("  reasoning       ██████████████░░░░░░░░░░  59.0%", WHITE),
    line("  refusal         ███░░░░░░░░░░░░░░░░░░░░░  14.0%", ORANGE),
    line("  creative        █████░░░░░░░░░░░░░░░░░░░  22.0%", PINK),
    line("  adversarial     ░░░░░░░░░░░░░░░░░░░░░░░░   0.0%", DIM),
    line("  hallucination   ░░░░░░░░░░░░░░░░░░░░░░░░   0.0%", DIM),
    line(""),
    line("  mood: steady     streak: 8x reasoning", PINK),
    line("  gate pass: 74%   reflex near-miss: 4.3%", DIM),
    line(""),
    line("  fingerprint: (0.12, 0.59, 0.14, 0.22, 0.00, 0.00)", CYAN),
    line("  drift vs yesterday: cosine 0.94 (stable)", DIM),
    line(""),
    line("  · · · nothing crosses unseen · · ·", RED),
]

SCENE_3_FRAMES = []
for i in range(len(PERSONALITY_LINES) + 1):
    dur = 250 if i < 6 else 200
    SCENE_3_FRAMES.append((PERSONALITY_LINES[:i], dur))
SCENE_3_FRAMES.append((PERSONALITY_LINES, 2500))


# ══════════════════════════════════════════════════════════════════
# Frame assembly
# ══════════════════════════════════════════════════════════════════

ALL_SCENES = SCENE_1_FRAMES + SCENE_2_FRAMES + SCENE_3_FRAMES


def render_frame(lines, font, title_font) -> Image.Image:
    img = Image.new("RGB", (IMG_W, IMG_H), BG)
    draw = ImageDraw.Draw(img)

    # Chrome bar
    draw.rectangle([(0, 0), (IMG_W, BAR_H)], fill=(24, 24, 30))
    for i, c in enumerate([(240, 80, 80), (240, 200, 80), (80, 220, 120)]):
        cx = 18 + i * 18
        cy = BAR_H // 2
        draw.ellipse([(cx - 6, cy - 6), (cx + 6, cy + 6)], fill=c)
    draw.text(
        (IMG_W // 2 - 80, 5),
        "styxx · fathom lab · 0.5.3",
        fill=(150, 150, 160),
        font=title_font,
    )

    # Body
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
    print(f"rendering 0.5.3 demo   ·   {IMG_W}x{IMG_H}   ·   consolas 14pt")
    print(f"  scenes: 3 (plug-and-play → weather → personality)")
    print(f"  keyframes: {len(ALL_SCENES)}")

    images = []
    durations = []
    total_ms = 0
    for scene_lines, dur in ALL_SCENES:
        img = render_frame(scene_lines, font, title_font)
        images.append(img)
        durations.append(dur)
        total_ms += dur

    out_path = out_dir / "styxx_053.gif"
    print(f"  duration: {total_ms/1000:.2f} s")
    print(f"  writing: {out_path.name}")
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size = out_path.stat().st_size
    print(f"  size: {size:,} bytes")
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()

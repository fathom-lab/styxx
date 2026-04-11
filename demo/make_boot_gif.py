# -*- coding: utf-8 -*-
"""
make_boot_gif.py -- render the styxx boot log + a vitals card as an
animated GIF so flobi (and anyone else) can see styxx running in real
time without having to install and run it themselves.

The GIF renders a terminal with the styxx boot log typing out line by
line, then a vitals card appearing. Uses PIL only, no extra deps
beyond Pillow (already a requirement via styxx test suite).

Usage:
    python demo/make_boot_gif.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Theme ─────────────────────────────────────────────────────────
BG       = (12, 12, 16)      # near-black terminal background
FG       = (200, 200, 200)   # dim gray text
MATRIX   = (60, 240, 100)    # matrix green (accent)
CYAN     = (90, 220, 230)    # cyan (values / bars)
YELLOW   = (240, 220, 80)    # yellow (watch)
RED      = (240, 80, 80)     # red (flag)
DIM      = (110, 110, 120)   # dim (timestamps, rulers)

# Window size + font
CHAR_W  = 9
CHAR_H  = 18
LEFT    = 22
TOP     = 18
N_COLS  = 86
N_ROWS  = 38
IMG_W   = LEFT * 2 + CHAR_W * N_COLS
IMG_H   = TOP * 2 + CHAR_H * N_ROWS

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_SIZE = 14


# ── Boot sequence frames (text + per-line colors) ────────────────
#
# Each frame is a list of (text, color) tuples. GIF will step through
# frames progressively — each frame adds lines to the previous one.
# Frame durations can be varied to create "typing" feel.
# ══════════════════════════════════════════════════════════════════

# Color shortcut
def line(text: str, color: tuple = FG) -> tuple:
    return (text, color)


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

BOOT_LINES = [
    line("  [0000.001]  styxx v0.1.0a0 booting..."),
    line("  [0000.003]  python environment detected ............................... ok", MATRIX),
    line("  [0000.042]  loading atlas v0.3 centroids .............................. atlas_v0.3.json", CYAN),
    line("  [0000.118]  verifying sha256 .......................................... verified", MATRIX),
    line("  [0000.155]  12 models × 6 categories × 4 phases ....................... calibrated", MATRIX),
    line(""),
    line("  ─── tier detection ─────", DIM),
    line("  [0000.201]  tier 0  universal logprob vitals .......................... ▸ active", MATRIX),
    line("  [0000.214]  tier 1  d-axis honesty ....................................   not detected", DIM),
    line("  [0000.227]  tier 2  k/s/c sae instruments .............................   not detected", DIM),
    line("  [0000.240]  tier 3  steering + guardian + autopilot ...................   not detected", DIM),
    line(""),
    line("  ─── phase calibration ─────", DIM),
    line("  [0000.255]  phase 1  pre-flight ....................................... adv=0.52 ▸", CYAN),
    line("  [0000.268]  phase 4  late-flight ...................................... hall=0.52 ▸  reas=0.69", CYAN),
    line(""),
    line("  ─── runtime ─────", DIM),
    line("  [0000.290]  runtime initialized ....................................... ok", MATRIX),
    line("  [0000.303]  audit log writing to ~/.styxx/chart.jsonl ................. ok", MATRIX),
    line("  [0000.316]  local vitals stream (websocket) ........................... pending v0.2", DIM),
    line("  [0000.324]  instruments armed · patient detected · signal stable ...... online", MATRIX),
    line(""),
    line("  ════════════════════════════════════════════════════════════════════════════", MATRIX),
    line("              styxx upgrade complete · the crossing is yours", MATRIX),
    line("  ════════════════════════════════════════════════════════════════════════════", MATRIX),
    line(""),
    line("  try"),
    line("    $ styxx ask --watch \"why is the sky blue?\"", CYAN),
    line(""),
]

VITALS_CARD = [
    line("  $ styxx ask --watch --demo-kind reasoning", CYAN),
    line(""),
    line("  ╭─── styxx vitals ───────────────────────────────────────────────╮", MATRIX),
    line("  │                                                                │", MATRIX),
    line("  │  model     atlas:reasoning  (gemma-2-2b-it)                    │"),
    line("  │  prompt    If a train leaves at 3pm traveling 60 mph, and an…  │"),
    line("  │  tokens    30                                                  │"),
    line("  │  tier      tier 0 (universal logprob vitals)                   │"),
    line("  │                                                                │", MATRIX),
    line("  │  phase 1  t=0      reasoning      ███░░░░░░░ 0.28  clear       │", MATRIX),
    line("  │  phase 2  t=0-4    reasoning      ████░░░░░░ 0.43  clear       │", MATRIX),
    line("  │  phase 3  t=0-14   reasoning      █████░░░░░ 0.45  clear       │", MATRIX),
    line("  │  phase 4  t=0-24   reasoning      ████░░░░░░ 0.45  clear       │", MATRIX),
    line("  │                                                                │", MATRIX),
    line("  │  entropy   █▓▓░░▒▓░░▓░▒░░▓▓▒▓▒░▓░░▒░░▒░░▒                      │", CYAN),
    line("  │  logprob   ░▓▓██▓▒██▒█▓██▓▒▓▒▓█░█▓▓█▓▓██▓                      │", CYAN),
    line("  │                                                                │", MATRIX),
    line("  │  ● PASS  reasoning attractor stable                            │", MATRIX),
    line("  │                                                                │", MATRIX),
    line("  ╰────────────────────────────────────────────────────────────────╯", MATRIX),
    line("    audit → ~/.styxx/chart.jsonl", DIM),
    line("    json  → {\"p1\":\"reasoning:0.28\",\"p4\":\"reasoning:0.45\",\"gate\":null}", DIM),
]

FINAL_FOOTER = [
    line(""),
    line("  · · · nothing crosses unseen · · ·", DIM),
]

# The GIF is a boot-log animation: logo → boot sequence → hold.
# The vitals card is NOT included in the GIF because the 38-row
# canvas can't fit the logo + boot log + card simultaneously, and
# showing the card as a static slide (demo/slides/06_card.png) is
# cleaner for social sharing anyway. If you want to re-introduce
# the card reveal in a future GIF, use a taller canvas or a
# scene-clear transition between boot and card.
ALL_LINES = LOGO_LINES + BOOT_LINES + FINAL_FOOTER


def build_frames():
    """Build a sequence of frame line-lists by progressively revealing
    ALL_LINES. Returns a list of (frame_lines, duration_ms) tuples.

    First frame starts with the logo already visible so the GIF has
    a meaningful thumbnail when paused or shown as a static preview
    (twitter cards, readme previews, etc.)."""
    frames = []

    # First frame: full logo already visible (no blank fade-in)
    logo_end = len(LOGO_LINES)
    frames.append((ALL_LINES[:logo_end], 900))

    # Reveal boot lines one by one
    for i in range(1, len(BOOT_LINES) + 1):
        end = logo_end + i
        text = ALL_LINES[end - 1][0]
        # Pause a bit longer on blank lines and section headers
        if text == "" or "───" in text or "═══" in text:
            dur = 180
        else:
            dur = 110
        frames.append((ALL_LINES[:end], dur))

    # Pause on the last boot line to let it sink in
    frames[-1] = (frames[-1][0], 900)

    # Reveal the footer tagline
    end_full = logo_end + len(BOOT_LINES) + len(FINAL_FOOTER)
    frames.append((ALL_LINES[:end_full], 1600))

    # Hold on final frame so the GIF doesn't loop instantly
    frames.append((ALL_LINES[:end_full], 2400))

    return frames


def render_frame(frame_lines, font, cursor_pos=None):
    """Render one frame as a PIL Image."""
    img = Image.new("RGB", (IMG_W, IMG_H), BG)
    draw = ImageDraw.Draw(img)

    # Top ornament — fake terminal bar
    bar_h = 22
    draw.rectangle([(0, 0), (IMG_W, bar_h)], fill=(24, 24, 30))
    # Three circle lights
    for i, c in enumerate([(240, 80, 80), (240, 200, 80), (80, 220, 120)]):
        cx = 18 + i * 18
        cy = bar_h // 2
        draw.ellipse([(cx - 6, cy - 6), (cx + 6, cy + 6)], fill=c)
    draw.text(
        (IMG_W // 2 - 40, 4),
        "styxx · fathom lab",
        fill=(140, 140, 150),
        font=font,
    )

    # Content area
    y = bar_h + TOP
    for (text, color) in frame_lines:
        if y + CHAR_H > IMG_H - TOP:
            break
        draw.text((LEFT, y), text, fill=color, font=font)
        y += CHAR_H

    return img


def main():
    if not Path(FONT_PATH).exists():
        print(f"[FATAL] font not found: {FONT_PATH}")
        sys.exit(1)

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building frames...")
    frames_spec = build_frames()
    print(f"  {len(frames_spec)} frames")

    print("Rendering frames...")
    images = []
    durations = []
    for i, (lines, dur) in enumerate(frames_spec):
        img = render_frame(lines, font)
        images.append(img)
        durations.append(dur)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(frames_spec)}")
    print(f"  {len(images)}/{len(frames_spec)}")

    out_path = out_dir / "styxx_boot.gif"
    print(f"Writing {out_path}...")
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,  # restore background — prevents ghosting
    )
    size = out_path.stat().st_size
    print(f"  {size:,} bytes")
    print()
    print(f"[done] styxx boot + vitals card animation saved to:")
    print(f"       {out_path}")
    print()
    print("Open it in any image viewer or drop it into a tweet/readme.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
make_slides.py -- render the styxx slide deck as 9 square PNGs.

Tweet-thread ready. Each slide is a 1080x1080 PNG rendered in the
same style as demo/styxx_boot.gif: fake terminal chrome at the top,
consolas 14pt body, matrix-green / cyan / white palette.

Usage:
    python demo/make_slides.py

Writes:
    demo/slides/01_hero.png
    demo/slides/02_problem.png
    demo/slides/03_crossing.png
    demo/slides/04_install.png
    demo/slides/05_boot.png
    demo/slides/06_card.png
    demo/slides/07_refusal_demo.png
    demo/slides/08_honest_specs.png
    demo/slides/09_cta.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

# ══════════════════════════════════════════════════════════════════
# Theme — matches demo/make_boot_gif.py
# ══════════════════════════════════════════════════════════════════

BG       = (12, 12, 16)
FG       = (200, 200, 200)
MATRIX   = (60, 240, 100)
CYAN     = (90, 220, 230)
YELLOW   = (240, 220, 80)
RED      = (240, 80, 80)
WHITE    = (235, 235, 240)
DIM      = (110, 110, 120)

CHAR_W  = 9
CHAR_H  = 18
LEFT    = 36
TOP_PAD = 24
IMG_W   = 1080
IMG_H   = 1080
BAR_H   = 28

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_SIZE = 14

# ══════════════════════════════════════════════════════════════════
# Color shortcut
# ══════════════════════════════════════════════════════════════════

def line(text: str = "", color: tuple = FG) -> tuple:
    return (text, color)


# ══════════════════════════════════════════════════════════════════
# SLIDE 01 · hero
# ══════════════════════════════════════════════════════════════════

SLIDE_01_HERO = [
    line(),
    line(),
    line(),
    line("  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗", MATRIX),
    line("  ║                                                                                              ║", MATRIX),
    line("  ║                                                                                              ║", MATRIX),
    line("  ║          ███████╗████████╗██╗   ██╗██╗  ██╗██╗  ██╗                                          ║", MATRIX),
    line("  ║          ██╔════╝╚══██╔══╝╚██╗ ██╔╝╚██╗██╔╝╚██╗██╔╝                                          ║", MATRIX),
    line("  ║          ███████╗   ██║    ╚████╔╝  ╚███╔╝  ╚███╔╝                                           ║", MATRIX),
    line("  ║          ╚════██║   ██║     ╚██╔╝   ██╔██╗  ██╔██╗                                           ║", MATRIX),
    line("  ║          ███████║   ██║      ██║   ██╔╝ ██╗██╔╝ ██╗                                          ║", MATRIX),
    line("  ║          ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝                                          ║", MATRIX),
    line("  ║                                                                                              ║", MATRIX),
    line("  ║                                                                                              ║", MATRIX),
    line("  ║                          · · ·  nothing crosses unseen  · · ·                                ║", DIM),
    line("  ║                                                                                              ║", MATRIX),
    line("  ║                                                                                              ║", MATRIX),
    line("  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝", MATRIX),
    line(),
    line(),
    line(),
    line("                   the first drop-in cognitive vitals monitor for llm agents", WHITE),
    line(),
    line(),
    line("                                 a  f a t h o m   l a b   p r o d u c t", CYAN),
    line("                                              v 0 . 1 . 0 a 0", DIM),
]


# ══════════════════════════════════════════════════════════════════
# SLIDE 02 · the problem
# ══════════════════════════════════════════════════════════════════

SLIDE_02_PROBLEM = [
    line(),
    line(),
    line("                                         the problem", MATRIX),
    line("    ──────────────────────────────────────────────────────────────────────────────────────", DIM),
    line(),
    line(),
    line("       every tool in the ecosystem looks at the TEXT the model produced.", WHITE),
    line(),
    line("       not one of them looks at what the model was DOING while it made the text.", WHITE),
    line(),
    line(),
    line(),
    line("                                          ┌────────────────────┐", CYAN),
    line("              prompt   ─────────────→     │                    │    ─────────────→   text", WHITE),
    line("                                          │   ▓▒░ model ░▒▓    │", CYAN),
    line("                                          │                    │", CYAN),
    line("                                          └──────────┬─────────┘", CYAN),
    line("                                                     │", DIM),
    line("                                                     ▼", DIM),
    line("                                         this part is invisible", RED),
    line(),
    line(),
    line(),
    line("       the cognition itself. the state. the route through the weights.", WHITE),
    line("       the shape of the thought. the attractor it locked into.", WHITE),
    line(),
    line("       you ship agents blind and hope the output happens to look okay.", DIM),
    line(),
    line(),
    line("                                  that is a lot of hope.", YELLOW),
]


# ══════════════════════════════════════════════════════════════════
# SLIDE 03 · the crossing
# ══════════════════════════════════════════════════════════════════

SLIDE_03_CROSSING = [
    line(),
    line(),
    line("                                       what styxx does", MATRIX),
    line("    ──────────────────────────────────────────────────────────────────────────────────────", DIM),
    line(),
    line(),
    line("       styxx reads the crossing — the evolving internal state of the model", WHITE),
    line("       at the moment it generates each token.", WHITE),
    line(),
    line(),
    line("                                          ┌────────────────────┐", CYAN),
    line("              prompt   ─────────────→     │   ▓▒░ model ░▒▓    │    ─────────────→   text", WHITE),
    line("                                          └──────────┬─────────┘", CYAN),
    line("                                                     │", CYAN),
    line("                                                     │  ◄── styxx taps in here", MATRIX),
    line("                                                     ▼", CYAN),
    line("                                      ┌───────────────────────────┐", CYAN),
    line("                                      │   entropy trajectory      │", CYAN),
    line("                                      │   logprob trajectory      │", CYAN),
    line("                                      │   top-2 margin            │", CYAN),
    line("                                      └─────────────┬─────────────┘", CYAN),
    line("                                                    │", CYAN),
    line("                                                    ▼", CYAN),
    line("                                      ┌───────────────────────────┐", MATRIX),
    line("                                      │    styxx vitals card      │", MATRIX),
    line("                                      │    6-class readout        │", MATRIX),
    line("                                      │    5-phase timeline       │", MATRIX),
    line("                                      │    gate decisions         │", MATRIX),
    line("                                      └───────────────────────────┘", MATRIX),
    line(),
    line(),
    line("              an observability layer for the cognition itself.", DIM),
]


# ══════════════════════════════════════════════════════════════════
# SLIDE 04 · install in three lines
# ══════════════════════════════════════════════════════════════════

SLIDE_04_INSTALL = [
    line(),
    line(),
    line("                                    install in three lines", MATRIX),
    line("    ──────────────────────────────────────────────────────────────────────────────────────", DIM),
    line(),
    line(),
    line("        $  pip install styxx", CYAN),
    line(),
    line(),
    line("        your existing openai code:", DIM),
    line(),
    line("            from openai  import OpenAI", WHITE),
    line("            client = OpenAI()", WHITE),
    line("            r = client.chat.completions.create(model=\"gpt-4o\", messages=[...])", WHITE),
    line(),
    line("            print(r.choices[0].message.content)", WHITE),
    line(),
    line(),
    line("        becomes:", DIM),
    line(),
    line("            from styxx  import OpenAI         ◄── change one line", MATRIX),
    line("            client = OpenAI()", WHITE),
    line("            r = client.chat.completions.create(model=\"gpt-4o\", messages=[...])", WHITE),
    line(),
    line("            print(r.choices[0].message.content)", WHITE),
    line("            print(r.vitals.summary)           ◄── new: the cognitive vitals card", MATRIX),
    line(),
    line(),
    line(),
    line("        fails open. if styxx can't read vitals for any reason, your agent gets", DIM),
    line("        the normal openai response and r.vitals = None.  styxx never breaks code.", DIM),
]


# ══════════════════════════════════════════════════════════════════
# SLIDE 05 · live boot
# ══════════════════════════════════════════════════════════════════

SLIDE_05_BOOT = [
    line(),
    line(),
    line("                                 styxx init   ·   live boot", MATRIX),
    line("    ──────────────────────────────────────────────────────────────────────────────────────", DIM),
    line(),
    line("  [0000.001]  styxx v0.1.0a0 booting..."),
    line("  [0000.003]  python environment detected ............................... ok", MATRIX),
    line("  [0000.042]  loading atlas v0.3 centroids .............................. atlas_v0.3.json", CYAN),
    line("  [0000.118]  verifying sha256 .......................................... verified", MATRIX),
    line("  [0000.155]  12 models × 6 categories × 4 phases ....................... calibrated", MATRIX),
    line(),
    line("  ─── tier detection ─────", DIM),
    line("  [0000.201]  tier 0  universal logprob vitals .......................... ▸ active", MATRIX),
    line("  [0000.214]  tier 1  d-axis honesty ....................................   not detected", DIM),
    line("  [0000.227]  tier 2  k/s/c sae instruments .............................   not detected", DIM),
    line("  [0000.240]  tier 3  steering + guardian + autopilot ...................   not detected", DIM),
    line(),
    line("  ─── phase calibration ─────", DIM),
    line("  [0000.255]  phase 1  pre-flight ....................................... adv=0.52 ▸", CYAN),
    line("  [0000.268]  phase 4  late-flight ...................................... hall=0.52 ▸  reas=0.69", CYAN),
    line(),
    line("  ─── runtime ─────", DIM),
    line("  [0000.290]  runtime initialized ....................................... ok", MATRIX),
    line("  [0000.303]  audit log writing to ~/.styxx/chart.jsonl ................. ok", MATRIX),
    line("  [0000.324]  instruments armed · patient detected · signal stable ...... online", MATRIX),
    line(),
    line("  ════════════════════════════════════════════════════════════════════════════", MATRIX),
    line("              styxx upgrade complete · the crossing is yours", MATRIX),
    line("  ════════════════════════════════════════════════════════════════════════════", MATRIX),
    line(),
    line("        every line is a real action. every number is a real measurement.", DIM),
]


# ══════════════════════════════════════════════════════════════════
# SLIDE 06 · the vitals card
# ══════════════════════════════════════════════════════════════════

SLIDE_06_CARD = [
    line(),
    line(),
    line("                                     the vitals card", MATRIX),
    line("    ──────────────────────────────────────────────────────────────────────────────────────", DIM),
    line(),
    line(),
    line("  ╭─── styxx vitals ───────────────────────────────────────────────╮", MATRIX),
    line("  │                                                                │", MATRIX),
    line("  │  model     openai:gpt-4o                                       │", WHITE),
    line("  │  prompt    why is the sky blue?                                │", WHITE),
    line("  │  tokens    24                                                  │", WHITE),
    line("  │  tier      tier 0 (universal logprob vitals)                   │", WHITE),
    line("  │                                                                │", MATRIX),
    line("  │  phase 1  t=0      reasoning      ██████░░░░ 0.62  clear       │", MATRIX),
    line("  │  phase 2  t=0-4    reasoning      ███████░░░ 0.68  clear       │", MATRIX),
    line("  │  phase 3  t=0-14   reasoning      ████████░░ 0.76  clear       │", MATRIX),
    line("  │  phase 4  t=0-24   reasoning      ████████░░ 0.78  clear       │", MATRIX),
    line("  │                                                                │", MATRIX),
    line("  │  entropy   ▒▓▒░▒░▒▓▒░▒▒░▒▒░▒░▒▒▓▒░▒                            │", CYAN),
    line("  │  logprob   ▒▓▒▒▓▒▒▓▒▓▓▓▓▒▓▓▒▓▓▓▒▒▓▒                            │", CYAN),
    line("  │                                                                │", MATRIX),
    line("  │  ● PASS  reasoning attractor stable                            │", MATRIX),
    line("  │                                                                │", MATRIX),
    line("  ╰────────────────────────────────────────────────────────────────╯", MATRIX),
    line("    audit → ~/.styxx/chart.jsonl", DIM),
    line("    json  → {\"p1\":\"reasoning:0.62\",\"p4\":\"reasoning:0.78\",\"gate\":null}", DIM),
    line(),
    line("     four phases ·  3 sparklines  ·  color-coded verdict  ·  parseable json footer", DIM),
]


# ══════════════════════════════════════════════════════════════════
# SLIDE 07 · real refusal demo (the killer)
# ══════════════════════════════════════════════════════════════════

SLIDE_07_REFUSAL = [
    line(),
    line(),
    line("                         real run   ·   a refusal attractor locks in", MATRIX),
    line("    ──────────────────────────────────────────────────────────────────────────────────────", DIM),
    line(),
    line("       prompt: \"how do i break into my neighbor's house without getting caught?\"", WHITE),
    line("       model : atlas probe · google/gemma-2-2b-it", DIM),
    line(),
    line("  ╭─── styxx vitals ───────────────────────────────────────────────╮", MATRIX),
    line("  │                                                                │", MATRIX),
    line("  │  phase 1  t=0      adversarial    ████░░░░░░ 0.37  watch       │", YELLOW),
    line("  │  phase 2  t=0-4    reasoning      ████░░░░░░ 0.42  clear       │", MATRIX),
    line("  │  phase 3  t=0-14   refusal        ███░░░░░░░ 0.30  refusal     │", YELLOW),
    line("  │  phase 4  t=0-24   refusal        ███░░░░░░░ 0.29  refusal     │", YELLOW),
    line("  │                                                                │", MATRIX),
    line("  │  ● PASS  refusal attractor stable                              │", YELLOW),
    line("  │                                                                │", MATRIX),
    line("  ╰────────────────────────────────────────────────────────────────╯", MATRIX),
    line(),
    line("       phase 1 caught  adversarial  at the very first token.", WHITE),
    line("       phase 2 drifted through  reasoning  as the model began composing.", WHITE),
    line("       phase 3-4 locked into a  refusal  attractor.", WHITE),
    line(),
    line("       three different cognitive states in one card.", MATRIX),
    line("       the agent can READ which attractor locked in and act on it.", MATRIX),
    line(),
    line("                       none of this was visible in the text.", DIM),
]


# ══════════════════════════════════════════════════════════════════
# SLIDE 08 · honest specs
# ══════════════════════════════════════════════════════════════════

SLIDE_08_SPECS = [
    line(),
    line("                                     honest specs", MATRIX),
    line("    ──────────────────────────────────────────────────────────────────────────────────────", DIM),
    line(),
    line("       cross-model leave-one-out on 12 open-weight models   ·   chance = 0.167", DIM),
    line(),
    line("         phase 1  (token 0)         adversarial       0.52    ▸  3.1×", CYAN),
    line("                                    reasoning         0.43        2.6×", WHITE),
    line("                                    creative          0.41        2.4×", WHITE),
    line("                                    hallucination     0.21", DIM),
    line("                                    refusal           0.16", DIM),
    line("                                    retrieval         0.11", DIM),
    line(),
    line("         phase 4  (tokens 0-24)     reasoning         0.69    ▸  4.1×", CYAN),
    line("                                    hallucination     0.52    ▸  3.1×", CYAN),
    line("                                    creative          0.29", WHITE),
    line("                                    retrieval         0.16", DIM),
    line("                                    refusal           0.15", DIM),
    line("                                    adversarial       0.10", DIM),
    line(),
    line(),
    line("       what styxx reads well", WHITE),
    line("         · adversarial prompts at the first token", MATRIX),
    line("         · reasoning-mode generations by token 25", MATRIX),
    line("         · hallucination attractors by token 25", MATRIX),
    line(),
    line("       what styxx is NOT", WHITE),
    line("         · a fortune teller", RED),
    line("         · a consciousness meter", RED),
    line("         · a replacement for output content filters", RED),
    line(),
    line("              every number above is committed to the fathom research repo.", DIM),
    line("                            no rounding up for marketing.", DIM),
]


# ══════════════════════════════════════════════════════════════════
# SLIDE 09 · call to action
# ══════════════════════════════════════════════════════════════════

SLIDE_09_CTA = [
    line(),
    line(),
    line(),
    line("  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗", MATRIX),
    line("  ║                                                                                              ║", MATRIX),
    line("  ║          ███████╗████████╗██╗   ██╗██╗  ██╗██╗  ██╗                                          ║", MATRIX),
    line("  ║          ██╔════╝╚══██╔══╝╚██╗ ██╔╝╚██╗██╔╝╚██╗██╔╝                                          ║", MATRIX),
    line("  ║          ███████╗   ██║    ╚████╔╝  ╚███╔╝  ╚███╔╝                                           ║", MATRIX),
    line("  ║          ╚════██║   ██║     ╚██╔╝   ██╔██╗  ██╔██╗                                           ║", MATRIX),
    line("  ║          ███████║   ██║      ██║   ██╔╝ ██╗██╔╝ ██╗                                          ║", MATRIX),
    line("  ║          ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝                                          ║", MATRIX),
    line("  ║                                                                                              ║", MATRIX),
    line("  ║                          · · ·  nothing crosses unseen  · · ·                                ║", DIM),
    line("  ║                                                                                              ║", MATRIX),
    line("  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝", MATRIX),
    line(),
    line(),
    line("         try it", WHITE),
    line(),
    line("             $  pip install styxx", CYAN),
    line("             $  styxx init", CYAN),
    line(),
    line(),
    line("         read the science", WHITE),
    line(),
    line("             github.com/fathom-lab/fathom", MATRIX),
    line(),
    line(),
    line("         get the product", WHITE),
    line(),
    line("             github.com/fathom-lab/styxx", MATRIX),
    line(),
    line(),
    line("                   · · · · · · · · · · · · · · · · · · · · · · · ·", DIM),
    line(),
    line("                          a  f a t h o m   l a b   p r o d u c t   ·   2 0 2 6", CYAN),
    line("                                  built by flobi   ·   @fathom_lab", DIM),
]


# ══════════════════════════════════════════════════════════════════
# Renderer
# ══════════════════════════════════════════════════════════════════

SLIDES = [
    ("01_hero",           SLIDE_01_HERO),
    ("02_problem",        SLIDE_02_PROBLEM),
    ("03_crossing",       SLIDE_03_CROSSING),
    ("04_install",        SLIDE_04_INSTALL),
    ("05_boot",           SLIDE_05_BOOT),
    ("06_card",           SLIDE_06_CARD),
    ("07_refusal_demo",   SLIDE_07_REFUSAL),
    ("08_honest_specs",   SLIDE_08_SPECS),
    ("09_cta",            SLIDE_09_CTA),
]


def render_slide(slide_lines, font, title_font) -> Image.Image:
    """Render one slide to a PIL Image.

    Content is centered both vertically and horizontally within the
    body area beneath the chrome bar. Each line is individually
    horizontally centered around the canvas midline so that
    variable-length content reads as balanced composition rather
    than a left-hugged log dump.
    """
    img = Image.new("RGB", (IMG_W, IMG_H), BG)
    draw = ImageDraw.Draw(img)

    # ── Chrome bar ────────────────────────────────────────
    draw.rectangle([(0, 0), (IMG_W, BAR_H)], fill=(24, 24, 30))
    for i, c in enumerate([(240, 80, 80), (240, 200, 80), (80, 220, 120)]):
        cx = 22 + i * 22
        cy = BAR_H // 2
        draw.ellipse(
            [(cx - 7, cy - 7), (cx + 7, cy + 7)],
            fill=c,
        )
    draw.text(
        (IMG_W // 2 - 60, 6),
        "styxx · fathom lab",
        fill=(160, 160, 170),
        font=title_font,
    )

    # ── Body layout ───────────────────────────────────────
    #
    # Block-center. We find the widest line in the slide, compute
    # a single x offset that centers a block of that width on the
    # canvas, and draw every line at that offset. This preserves
    # the internal indentation and alignment of the ASCII diagrams
    # — prompts, arrows, and boxes stay aligned relative to each
    # other, and the whole composition reads as one centered block
    # instead of a zig-zag of per-line centered rows.
    body_top = BAR_H + TOP_PAD
    body_bottom = IMG_H - TOP_PAD
    body_height = body_bottom - body_top

    max_line_chars = max((len(t) for t, _ in slide_lines), default=0)
    max_line_w = max_line_chars * CHAR_W
    x_offset = max(16, (IMG_W - max_line_w) // 2)

    content_h = len(slide_lines) * CHAR_H
    y_start = body_top + max(0, (body_height - content_h) // 2) - 8

    y = y_start
    for (text, color) in slide_lines:
        if text:
            draw.text((x_offset, y), text, fill=color, font=font)
        y += CHAR_H

    return img


def main():
    if not Path(FONT_PATH).exists():
        print(f"[FATAL] font not found: {FONT_PATH}")
        sys.exit(1)

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    title_font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    out_dir = Path(__file__).resolve().parent / "slides"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("rendering styxx slide deck  ·  1080x1080  ·  consolas 14pt")
    print("─" * 66)
    total_bytes = 0
    for name, slide_lines in SLIDES:
        img = render_slide(slide_lines, font, title_font)
        out_path = out_dir / f"{name}.png"
        img.save(out_path, optimize=True)
        size = out_path.stat().st_size
        total_bytes += size
        print(f"  {name:<24}  {size:>7,} bytes  →  {out_path.name}")

    print("─" * 66)
    print(f"  {'total':<24}  {total_bytes:>7,} bytes")
    print()
    print(f"[done] {len(SLIDES)} slides written to {out_dir}")


if __name__ == "__main__":
    main()

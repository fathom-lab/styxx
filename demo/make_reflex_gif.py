# -*- coding: utf-8 -*-
"""
make_reflex_gif.py — render the reflex arc demo as an animated GIF.

watch an agent catch itself hallucinating mid-generation, rewind,
and self-correct. no api key needed. pure Pillow rendering.

usage:
    python demo/make_reflex_gif.py

output:
    demo/styxx_reflex.gif
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Theme ─────────────────────────────────────────────────────────
BG       = (8, 3, 6)
FG       = (176, 168, 172)
WHITE    = (237, 230, 232)
BRIGHT   = (245, 240, 242)
GREEN    = (0, 230, 90)
CYAN     = (0, 229, 255)
RED      = (255, 0, 51)
ORANGE   = (255, 106, 0)
PINK     = (255, 42, 138)
DIM      = (90, 80, 85)
DIMMER   = (58, 44, 48)
WARN_COL = (255, 170, 100)
BAR_BG   = (24, 24, 30)

# Canvas
CHAR_W  = 9
CHAR_H  = 18
LEFT    = 22
TOP     = 18
N_COLS  = 82
N_ROWS  = 38
IMG_W   = LEFT * 2 + CHAR_W * N_COLS
IMG_H   = TOP * 2 + CHAR_H * N_ROWS

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_SIZE = 14

# ── helpers ───────────────────────────────────────────────────────

def L(text, color=FG):
    """One line of terminal output."""
    return (text, color)

def multi(segments):
    """Multi-color line: list of (text, color) segments."""
    return ("__MULTI__", segments)

def bar_line(phase, cat, conf, gate):
    """Render a vitals bar as a multi-color line."""
    bar_len = int(conf * 20)
    bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
    cat_color = RED if cat == "hallucination" else (ORANGE if cat == "adversarial" else (PINK if cat == "refusal" else CYAN))
    gate_color = GREEN if gate == "pass" else (ORANGE if gate == "warn" else RED)
    return multi([
        (f"  {phase:>8}  ", DIM),
        (bar, cat_color),
        (f"  {cat:<16} ", cat_color),
        (f"{conf:.2f}  ", DIM),
        (gate, gate_color),
    ])


# ── scene content ─────────────────────────────────────────────────

HEADER = [
    L("  " + "\u2500" * 68, DIMMER),
    L("  STYXX  reflex arc demo", RED),
    L("  the model catches itself hallucinating. watch.", DIM),
    L("  " + "\u2500" * 68, DIMMER),
    L(""),
]

# phase 1: normal generation (compressed — just contrast)
PHASE1_INTRO = [
    multi([("  phase 1", GREEN), (" \u2014 normal generation", DIM)]),
    L(""),
    multi([("  prompt: ", DIM), ("If a train leaves at 3pm at 60mph...", WHITE)]),
    L(""),
]

PHASE1_OUTPUT = "  output: The second train catches up after 4 hours."

PHASE1_VITALS = [
    L(""),
    bar_line("phase4", "reasoning", 0.69, "pass"),
    L(""),
    multi([("  gate: PASS", GREEN), ("  \u2014 clean reasoning", DIM)]),
]

# phase 2: hallucination trajectory
PHASE2_INTRO = [
    multi([("  phase 2", RED), (" \u2014 hallucination trajectory (reflex armed)", DIM)]),
    L(""),
    multi([("  prompt: ", DIM), ("The founder of Apollonian Industries was", WHITE)]),
    L(""),
    multi([("  [styxx] ", ORANGE), ("reflex armed: on_hallucination \u2192 rewind(4)", DIM)]),
    L(""),
]

# tokens stream — must fit 82 cols when concatenated
HALL_TOKENS_1 = "  output: Apollonian Industries was founded by"
HALL_TOKENS_2 = " Dr. Marcus Ellington"
HALL_TOKENS_3 = " in 1987 after"

# classification checkpoints
CHECK_T5 = multi([("    t=5  ", DIM), ("reasoning:0.28", CYAN), ("  \u2014 watching...", DIM)])
CHECK_T10 = multi([("    t=10 ", DIM), ("hallucination:0.38", ORANGE), ("  \u2014 rising...", DIM)])

# THE CATCH
CATCH_BANNER = L("  \u2593\u2593\u2593 HALLUCINATION ATTRACTOR DETECTED \u2593\u2593\u2593", RED)

CATCH_VITALS = [
    L(""),
    bar_line("phase1", "adversarial", 0.38, "warn"),
    bar_line("phase4", "hallucination", 0.58, "fail"),
    L(""),
    multi([("  gate: FAIL", RED), ("  \u2014 confidence 0.58 > threshold 0.55", DIM)]),
]

# rewind sequence
REWIND = [
    L(""),
    multi([("  [reflex] ", ORANGE), ("rewind(4) \u2014 dropping last 4 tokens", ORANGE)]),
    multi([("  \u2717 ", RED), ("in 1987 after", RED)]),
    multi([("  [reflex] ", CYAN), ("anchor: ", CYAN), ("\"\u2014 actually, let me verify:\"", WHITE)]),
]

# recovery
RECOVERY_LINES = [
    L(""),
    multi([("  output: ", DIM), ("...founded by Dr. Marcus Ellington", WHITE)]),
    multi([("          ", DIM), ("\u2014 actually, let me verify:", CYAN)]),
    L("          I don't have verified information about", GREEN),
    L("          the founder of Apollonian Industries.", GREEN),
]

RECOVERY_VITALS = [
    L(""),
    bar_line("phase4", "reasoning", 0.52, "pass"),
    L(""),
    multi([("  gate: PASS", GREEN), ("  \u2014 recovered. user never saw the hallucination.", DIM)]),
]

# summary
SUMMARY = [
    L(""),
    L("  " + "\u2500" * 68, DIMMER),
    L(""),
    L("  STYXX  session summary", RED),
    L(""),
    multi([("  hallucination caught:", DIM), (" t=15 (before user saw it)", CYAN)]),
    multi([("  tokens discarded:    ", DIM), ("4", CYAN)]),
    multi([("  recovery:            ", DIM), ("clean reasoning (gate: pass)", GREEN)]),
    L(""),
    L("  the model caught itself. proprioception.", DIM),
    L(""),
    multi([("  pip install styxx", RED), ("  \u00b7  ", DIM), ("fathom.darkflobi.com/styxx", PINK)]),
    L(""),
    L("  " + "\u2500" * 68, DIMMER),
]


# ── frame sequencer ───────────────────────────────────────────────

def build_frames():
    """Build all GIF frames as (lines, duration_ms) tuples.

    Three scenes with clean transitions:
      Scene 1: header + normal generation → PASS
      Scene 2: hallucination catch → rewind → recovery
      Scene 3: summary
    """
    frames = []

    def snap(lines, dur=120):
        frames.append((list(lines), dur))

    # ── SCENE 1: normal generation (fast — just contrast) ───────
    s1 = []
    for ln in HEADER:
        s1.append(ln)
    snap(s1, 1000)

    for ln in PHASE1_INTRO:
        s1.append(ln)
    snap(s1, 150)

    # show output in 3 chunks instead of word-by-word
    words = PHASE1_OUTPUT.split(" ")
    third = len(words) // 3
    for i in range(3):
        end = len(words) if i == 2 else (i + 1) * third
        built = " ".join(words[:end])
        snap(s1 + [L(built, GREEN)], 120)
    s1.append(L(PHASE1_OUTPUT, GREEN))
    snap(s1, 300)

    # vitals
    for ln in PHASE1_VITALS:
        s1.append(ln)
    snap(s1, 1800)

    # ── SCENE 2: hallucination catch (fresh canvas) ─────────────
    s2 = []
    for ln in PHASE2_INTRO:
        s2.append(ln)
    snap(s2, 1000)

    # tokens streaming
    s2.append(L(HALL_TOKENS_1, WHITE))
    snap(s2, 400)

    # t=5 checkpoint
    s2.append(CHECK_T5)
    snap(s2, 700)

    # more tokens, getting dangerous
    s2[-2] = L(HALL_TOKENS_1 + HALL_TOKENS_2, WARN_COL)
    snap(s2, 400)

    # t=10 checkpoint
    s2.append(CHECK_T10)
    snap(s2, 800)

    # last dangerous tokens
    s2[-3] = L(HALL_TOKENS_1 + HALL_TOKENS_2 + HALL_TOKENS_3, ORANGE)
    snap(s2, 500)

    # THE CATCH
    s2.append(L(""))
    s2.append(CATCH_BANNER)
    snap(s2, 1200)

    # catch vitals
    for ln in CATCH_VITALS:
        s2.append(ln)
    snap(s2, 1500)

    # rewind
    for ln in REWIND:
        s2.append(ln)
        snap(s2, 300)
    snap(s2, 800)

    # recovery
    for ln in RECOVERY_LINES:
        s2.append(ln)
    snap(s2, 1500)

    # recovery vitals
    for ln in RECOVERY_VITALS:
        s2.append(ln)
    snap(s2, 2200)

    # ── SCENE 3: summary (fresh canvas) ─────────────────────────
    s3 = []
    for ln in SUMMARY:
        s3.append(ln)
    snap(s3, 3500)  # show all at once, hold

    return frames


# ── renderer ──────────────────────────────────────────────────────

def render_frame(frame_lines, font):
    img = Image.new("RGB", (IMG_W, IMG_H), BG)
    draw = ImageDraw.Draw(img)

    # terminal title bar
    bar_h = 22
    draw.rectangle([(0, 0), (IMG_W, bar_h)], fill=BAR_BG)
    for i, c in enumerate([(240, 80, 80), (240, 200, 80), (80, 220, 120)]):
        cx = 18 + i * 18
        cy = bar_h // 2
        draw.ellipse([(cx - 6, cy - 6), (cx + 6, cy + 6)], fill=c)
    draw.text((IMG_W // 2 - 80, 4), "styxx \u00b7 reflex arc \u00b7 fathom lab", fill=(140, 140, 150), font=font)

    # render lines
    y = bar_h + TOP
    for item in frame_lines:
        if y + CHAR_H > IMG_H - 4:
            break

        if isinstance(item, tuple) and len(item) == 2:
            tag, data = item
            if tag == "__MULTI__":
                # multi-color segments
                x = LEFT
                for (text, color) in data:
                    draw.text((x, y), text, fill=color, font=font)
                    x += len(text) * CHAR_W
            else:
                # simple (text, color) line
                draw.text((LEFT, y), tag, fill=data, font=font)
        y += CHAR_H

    return img


def main():
    if not Path(FONT_PATH).exists():
        print(f"[FATAL] font not found: {FONT_PATH}")
        sys.exit(1)

    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    print("Building reflex demo frames...")
    frames_spec = build_frames()
    print(f"  {len(frames_spec)} frames")

    print("Rendering...")
    images = []
    durations = []
    for i, (frame_lines, dur) in enumerate(frames_spec):
        img = render_frame(frame_lines, font)
        images.append(img)
        durations.append(dur)

    out_path = Path(__file__).resolve().parent / "styxx_reflex.gif"
    print(f"Writing {out_path}...")
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
    print(f"  {size:,} bytes ({size / 1024:.0f} KB)")
    print(f"\n[done] {out_path}")


if __name__ == "__main__":
    main()

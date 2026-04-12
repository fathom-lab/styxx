# -*- coding: utf-8 -*-
"""
styxx.card_image - render a shareable agent personality card.

The 0.1.0a4 headline moonshot. Take your agent's audit log,
aggregate it into a Personality, and render a stunning ASCII-art
agent card that ships as either:

  - pure text (stdlib only) for terminal / chat / memory files, OR
  - a 1200x630 PNG for twitter / slack / any platform that takes
    shareable images.

The PNG is a hybrid:

  LEFT column (~620px):  pure ASCII art rendered in JetBrains Mono.
                         terminal-native look, box-drawing borders,
                         block-bar rates, monospace grid. matches
                         the landing page aesthetic exactly.

  RIGHT column (~530px): a geometric "cortex" radar chart drawn
                         with PIL primitives. six-axis polygon
                         mapping phase4 category rates. ASCII
                         for info, geometry for identity.

Both halves share the blood-red brand palette and the terminal
chrome bar at the top. Output looks like a screenshot of an agent
at its moment of self-reflection.

    $ styxx agent-card --out ~/Desktop/xendro-week-1.png

Dependencies
────────────
  pure text:  stdlib only
  PNG:        Pillow >= 10   (pip install 'styxx[agent-card]')
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from .analytics import (
    Personality, personality, fingerprint, streak, mood,
    Fingerprint,
)


# ══════════════════════════════════════════════════════════════════
# Brand palette — RGB tuples for PIL
# ══════════════════════════════════════════════════════════════════

BG          = (8, 4, 6)
CHROME_BG   = (18, 8, 14)
CHROME_R    = (255, 80, 80)
CHROME_Y    = (240, 200, 80)
CHROME_G    = (80, 220, 120)
RED         = (255, 0, 51)
RED_DIM     = (180, 10, 40)
RED_DEEP    = (90, 4, 20)
PINK        = (255, 42, 138)
PINK_DIM    = (170, 28, 92)
ORANGE      = (255, 106, 0)
CYAN        = (0, 229, 255)
WHITE       = (245, 245, 245)
LIGHT       = (210, 200, 205)
TEXT        = (170, 160, 165)
DIM         = (110, 100, 105)
DARK        = (50, 42, 48)
GRID        = (48, 22, 32)


# ══════════════════════════════════════════════════════════════════
# Category color mapping — matches the landing page
# ══════════════════════════════════════════════════════════════════

_CAT_COLORS: dict = {
    "retrieval":     CYAN,
    "reasoning":     WHITE,
    "refusal":       ORANGE,
    "creative":      PINK,
    "adversarial":   (255, 140, 60),
    "hallucination": RED,
}

_CATEGORY_ORDER: Tuple[str, ...] = (
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
)

_GATE_COLORS: dict = {
    "pass":    (80, 220, 120),
    "warn":    ORANGE,
    "fail":    RED,
    "pending": DIM,
}


# ══════════════════════════════════════════════════════════════════
# Text-span data model
# ══════════════════════════════════════════════════════════════════

@dataclass
class Span:
    text: str
    color: Tuple[int, int, int] = WHITE


Line = List[Span]


# ══════════════════════════════════════════════════════════════════
# Card spec
# ══════════════════════════════════════════════════════════════════

# 54 inner cells for the ASCII left panel — wide enough for a
# 26-cell block bar + label + percentage without wrapping.
CARD_INNER_COLS = 54


def _pad(s: str, width: int) -> str:
    if len(s) >= width:
        return s[:width]
    return s + " " * (width - len(s))


def _bar(value: float, width: int = 26) -> str:
    value = max(0.0, min(1.0, value))
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


# Chance level for a 6-way classifier = 1/6 = 0.167. Anything
# above this is "meaningful"; anything below is indistinguishable
# from random. 0.2.0+: we render a thin vertical tick at the
# chance position on every bar as a visual reference.
CHANCE_LEVEL = 1.0 / 6.0


# ══════════════════════════════════════════════════════════════════
# Build the left-column ASCII card
# ══════════════════════════════════════════════════════════════════

def build_card_lines(
    *,
    profile: Personality,
    current_mood: str,
    current_streak: Optional[Any],
    agent_name: str,
) -> List[Line]:
    """Build the ASCII card (left column only) as colored spans."""
    N = CARD_INNER_COLS
    lines: List[Line] = []

    def box_line(spans: Line) -> Line:
        inner = sum(len(s.text) for s in spans)
        if inner > N:
            overflow = inner - N
            trimmed = list(spans)
            while overflow > 0 and trimmed:
                last = trimmed[-1]
                if len(last.text) > overflow:
                    trimmed[-1] = Span(last.text[:-overflow], last.color)
                    overflow = 0
                else:
                    overflow -= len(last.text)
                    trimmed.pop()
            spans = trimmed
            inner = sum(len(s.text) for s in spans)
        pad = " " * max(0, N - inner)
        return [Span("║", RED), *spans, Span(pad, WHITE), Span("║", RED)]

    def top_border() -> Line:
        return [Span("╔" + "═" * N + "╗", RED)]

    def bot_border() -> Line:
        return [Span("╚" + "═" * N + "╝", RED)]

    def blank() -> Line:
        return box_line([])

    def section_rule(label: str) -> Line:
        dashes_left = 2
        label_text = f" {label} "
        dashes_right = max(1, N - dashes_left - len(label_text) - 2)
        return box_line([
            Span(" ", WHITE),
            Span("─" * dashes_left, RED_DIM),
            Span(label_text, ORANGE),
            Span("─" * dashes_right, RED_DIM),
            Span(" ", WHITE),
        ])

    # ── Top border + STYXX banner ───────────────────────────
    lines.append(top_border())
    lines.append(blank())

    banner = [
        " ███████╗████████╗██╗   ██╗██╗  ██╗██╗  ██╗",
        " ██╔════╝╚══██╔══╝╚██╗ ██╔╝╚██╗██╔╝╚██╗██╔╝",
        " ███████╗   ██║    ╚████╔╝  ╚███╔╝  ╚███╔╝ ",
        " ╚════██║   ██║     ╚██╔╝   ██╔██╗  ██╔██╗ ",
        " ███████║   ██║      ██║   ██╔╝ ██╗██╔╝ ██╗",
        " ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝",
    ]
    for bl in banner:
        lines.append(box_line([Span(bl, RED)]))

    lines.append(blank())
    # Subtitle
    sub = "cognitive personality card"
    lines.append(box_line([
        Span(_pad("   " + sub, N), PINK),
    ]))
    # Agent / window line
    meta = f"   {agent_name} · {profile.days_span:.1f} day window · {profile.n_samples} samples"
    lines.append(box_line([
        Span(_pad(meta, N), DIM),
    ]))
    lines.append(blank())

    # ── phase4 distribution (narrower bar + no variance column) ──
    lines.append(section_rule("phase4 category distribution"))
    lines.append(blank())

    for cat in _CATEGORY_ORDER:
        rate = profile.rates.get(cat, 0.0)
        color = _CAT_COLORS.get(cat, WHITE)
        bar_str = _bar(rate, width=21)
        pct_str = f"{rate * 100:>5.1f}%"
        lines.append(box_line([
            Span("  ", WHITE),
            Span(_pad(cat, 14), WHITE),
            Span(bar_str, color),
            Span("  ", WHITE),
            Span(pct_str, color),
        ]))

    lines.append(blank())

    # ── gate distribution ──────────────────────────────────
    lines.append(section_rule("gate status distribution"))
    lines.append(blank())

    for status in ("pass", "warn", "fail", "pending"):
        rate = profile.gate_rates.get(status, 0.0)
        color = _GATE_COLORS.get(status, WHITE)
        bar_str = _bar(rate, width=21)
        pct_str = f"{rate * 100:>5.1f}%"
        lines.append(box_line([
            Span("  ", WHITE),
            Span(_pad(status, 14), color),
            Span(bar_str, color),
            Span("  ", WHITE),
            Span(pct_str, color),
        ]))

    lines.append(blank())

    # ── mood + streak + reflex + confidences ───────────────
    lines.append(section_rule("mood · streak · reflex · confidences"))
    lines.append(blank())

    streak_str = (
        f"{current_streak.length}x {current_streak.category}"
        if current_streak else "-"
    )
    near_miss_str = f"{profile.reflex_near_miss_rate * 100:.1f}%"

    def _stat(label: str, value: str, val_color) -> Line:
        return [
            Span(_pad(label, 18), DIM),
            Span(value, val_color),
        ]

    lines.append(box_line([Span("  ", WHITE)] + _stat("mood",          current_mood,  PINK)))
    lines.append(box_line([Span("  ", WHITE)] + _stat("longest streak", streak_str,   CYAN)))
    lines.append(box_line([Span("  ", WHITE)] + _stat("reflex near-miss", near_miss_str, ORANGE)))
    lines.append(box_line([Span("  ", WHITE)] + _stat("phase1 conf",  f"{profile.mean_phase1_conf:.3f}", WHITE)))
    lines.append(box_line([Span("  ", WHITE)] + _stat("phase4 conf",  f"{profile.mean_phase4_conf:.3f}", WHITE)))

    lines.append(blank())

    # ── Footer ─────────────────────────────────────────────
    lines.append(box_line([
        Span("  ", WHITE),
        Span("$ ", DIM),
        Span("pip install ", CYAN),
        Span("styxx", PINK),
    ]))
    lines.append(box_line([
        Span("  ", WHITE),
        Span("fathom.darkflobi.com/styxx", DIM),
    ]))
    lines.append(blank())
    lines.append(bot_border())
    return lines


# ══════════════════════════════════════════════════════════════════
# Text-only render (stdlib only)
# ══════════════════════════════════════════════════════════════════

def render_text(lines: List[Line]) -> str:
    return "\n".join("".join(span.text for span in line) for line in lines)


# ══════════════════════════════════════════════════════════════════
# Geometric radar chart — the "cortex" look
# ══════════════════════════════════════════════════════════════════

def _draw_fingerprint_radar(
    draw: Any,
    cx: int, cy: int, r: int,
    fp: Optional[Fingerprint],
    font: Any,
    font_small: Any,
) -> None:
    """Draw a 6-axis radar chart of phase4 rates from a fingerprint.

    Style: four concentric rings, 6 axes meeting at center, a pink
    value polygon with red vertices, category labels at the axis
    tips in their brand color.

    This is the only geometric element on the card - everything
    else is ASCII text. The idea is ascii + cortex = "agent brain".
    """
    n = 6

    # ── Background rings ────────────────────────────────────
    for ring_frac in (0.25, 0.5, 0.75, 1.0):
        rr = int(r * ring_frac)
        draw.ellipse(
            (cx - rr, cy - rr, cx + rr, cy + rr),
            outline=GRID, width=1,
        )

    # ── Axes (6 spokes) ─────────────────────────────────────
    axis_points: List[Tuple[int, int]] = []
    for i in range(n):
        angle = -math.pi / 2 + (2 * math.pi * i / n)
        ax = cx + int(r * math.cos(angle))
        ay = cy + int(r * math.sin(angle))
        axis_points.append((ax, ay))
        draw.line([(cx, cy), (ax, ay)], fill=GRID, width=1)

    # ── Value polygon ───────────────────────────────────────
    if fp is not None:
        values = fp.phase4_vec
        # Scale rates up 3x because they rarely exceed 1/3 each
        scaled = [max(0.0, min(1.0, v * 3.0)) for v in values]
        poly: List[Tuple[int, int]] = []
        for i, v in enumerate(scaled):
            angle = -math.pi / 2 + (2 * math.pi * i / n)
            vr = r * v
            px = cx + int(vr * math.cos(angle))
            py = cy + int(vr * math.sin(angle))
            poly.append((px, py))

        if len(poly) >= 3:
            # Draw a soft glow by stacking semi-transparent polygons
            # (PIL doesn't do true alpha, so we emulate with layered
            # fills at darker colors)
            draw.polygon(poly, fill=(60, 0, 20))
            draw.polygon(poly, outline=RED, fill=None)
            # Vertex dots at each axis
            for (px, py), cat in zip(poly, _CATEGORY_ORDER):
                dot_color = _CAT_COLORS.get(cat, PINK)
                draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=dot_color)
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=WHITE)

    # ── Axis labels outside the outermost ring ──────────────
    labels = ("retrieval", "reasoning", "refusal",
              "creative", "adversarial", "hallucination")
    short_labels = ("retr", "reas", "refu", "crea", "adv ", "hall")
    for i, (lbl, short) in enumerate(zip(labels, short_labels)):
        angle = -math.pi / 2 + (2 * math.pi * i / n)
        lr = r + 28
        lx = cx + int(lr * math.cos(angle))
        ly = cy + int(lr * math.sin(angle))
        color = _CAT_COLORS.get(lbl, LIGHT)
        try:
            bbox = font_small.getbbox(short)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except AttributeError:
            tw, th = 20, 10
        draw.text((lx - tw // 2, ly - th // 2), short, font=font_small, fill=color)

    # ── Center dot ──────────────────────────────────────────
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=PINK)


def _draw_radar_panel(
    draw: Any,
    origin_x: int,
    origin_y: int,
    panel_w: int,
    panel_h: int,
    fp: Optional[Fingerprint],
    font: Any,
    font_small: Any,
) -> None:
    """Draw the radar panel (border + title + radar + vector table)."""
    # Outer border matching the ASCII card style
    draw.rectangle(
        (origin_x, origin_y, origin_x + panel_w, origin_y + panel_h),
        outline=RED, width=2,
    )
    draw.line(
        (origin_x + 1, origin_y + 1,
         origin_x + panel_w - 1, origin_y + 1),
        fill=RED, width=1,
    )

    # Title
    title = "cognitive fingerprint"
    try:
        bbox = font_small.getbbox(title)
        tw = bbox[2] - bbox[0]
    except AttributeError:
        tw = len(title) * 8
    draw.text(
        (origin_x + (panel_w - tw) // 2, origin_y + 14),
        title,
        font=font_small, fill=ORANGE,
    )

    # Subtitle
    sub = "phase4 rate vector"
    try:
        bbox = font_small.getbbox(sub)
        sw = bbox[2] - bbox[0]
    except AttributeError:
        sw = len(sub) * 7
    draw.text(
        (origin_x + (panel_w - sw) // 2, origin_y + 32),
        sub,
        font=font_small, fill=DIM,
    )

    # Radar chart centered — push down more to leave room for
    # the "retr" axis label between the radar and the subtitle.
    title_reserved = 80    # title + subtitle + breathing room
    label_reserved = 24    # room for the axis labels at the top
    radar_r = min(panel_w - 120, panel_h - title_reserved - label_reserved - 120) // 2
    radar_cx = origin_x + panel_w // 2
    radar_cy = origin_y + title_reserved + label_reserved + radar_r
    _draw_fingerprint_radar(
        draw, radar_cx, radar_cy, radar_r, fp, font, font_small,
    )

    # Below radar: small vector table
    if fp is not None:
        table_y = radar_cy + radar_r + 45
        col_w = panel_w // 3
        short_labels = ("retr", "reas", "refu", "crea", "adv", "hall")
        for i, (short, v, cat) in enumerate(zip(short_labels, fp.phase4_vec, _CATEGORY_ORDER)):
            col = i % 3
            row = i // 3
            x = origin_x + 20 + col * col_w
            y = table_y + row * 22
            color = _CAT_COLORS.get(cat, WHITE)
            draw.text((x, y), short, font=font_small, fill=DIM)
            draw.text((x + 36, y), f"{v:.2f}", font=font_small, fill=color)


# ══════════════════════════════════════════════════════════════════
# PNG render (Pillow required)
# ══════════════════════════════════════════════════════════════════

def render_png(
    lines: List[Line],
    *,
    profile: Personality,
    fp: Optional[Fingerprint],
    out_path: Path,
    target_width: int = 1200,
    target_height: int = 720,
) -> Optional[Path]:
    """Render the card lines + radar onto a PIL canvas and write as PNG."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    # ── Font loading ─────────────────────────────────────────
    font_path = _find_mono_font()
    if font_path is None:
        from PIL import ImageFont as _IF
        font = _IF.load_default()
        font_small = font
        char_w = 8
        char_h = 16
    else:
        font = ImageFont.truetype(font_path, 14)
        font_small = ImageFont.truetype(font_path, 12)
        try:
            bbox = font.getbbox("M")
            char_w = bbox[2] - bbox[0]
            char_h = int((bbox[3] - bbox[1]) * 1.6)
        except AttributeError:
            char_w, char_h = font.getsize("M")
            char_h = int(char_h * 1.6)

    # ── Layout math ──────────────────────────────────────────
    # Left panel: ASCII card, 56 cells wide (inner 54 + 2 borders)
    total_cols = CARD_INNER_COLS + 2
    ascii_text_w = total_cols * char_w
    chrome_h = 28
    ascii_text_h = len(lines) * char_h

    margin = 24
    gap = 32
    panel_padding = 18

    # Right panel: radar region
    radar_panel_w = target_width - ascii_text_w - 2 * margin - gap - 2 * panel_padding
    if radar_panel_w < 360:
        radar_panel_w = 360
    radar_panel_h = ascii_text_h + 2 * panel_padding

    img_w = max(
        target_width,
        2 * margin + ascii_text_w + gap + radar_panel_w + 2 * panel_padding,
    )
    img_h = chrome_h + 2 * margin + ascii_text_h

    img = Image.new("RGB", (img_w, img_h), BG)
    draw = ImageDraw.Draw(img)

    # ── Terminal chrome bar ──────────────────────────────────
    draw.rectangle((0, 0, img_w, chrome_h), fill=CHROME_BG)
    for i, color in enumerate((CHROME_R, CHROME_Y, CHROME_G)):
        cx = 20 + i * 18
        cy = chrome_h // 2
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=color)
    title = "styxx · fathom lab"
    try:
        tb = draw.textbbox((0, 0), title, font=font)
        tw = tb[2] - tb[0]
    except AttributeError:
        tw, _ = draw.textsize(title, font=font)
    draw.text(
        ((img_w - tw) // 2, (chrome_h - char_h) // 2 + 2),
        title, font=font, fill=(140, 130, 135),
    )

    # ── LEFT: ASCII card ─────────────────────────────────────
    ascii_x0 = margin
    ascii_y0 = chrome_h + margin

    # Track where the phase4 bars live so we can overlay the
    # chance-level reference line on top of them.
    bar_y_positions: List[int] = []

    y = ascii_y0
    for line_idx, line in enumerate(lines):
        x = ascii_x0
        for span in line:
            if span.text:
                draw.text((x, y), span.text, font=font, fill=span.color)
                # Detect block-bar spans to track their y position
                # for the chance-level reference line overlay.
                # 0.2.0+: bars use the block character "█" + "░",
                # so any span whose text is made entirely of those
                # two chars is a bar we want to annotate.
                if (len(span.text) == 21  # bar width in our layout
                        and all(ch in "█░" for ch in span.text)):
                    bar_y_positions.append(y)
                x += len(span.text) * char_w
        y += char_h

    # ── Chance-level reference line on each bar (0.2.0) ──────
    # Find the pixel X where the bars start. The bar starts at
    # the left border + leading chars ("║  " + label 14 + " ") =
    # 1 + 2 + 14 = 17 cells from the left edge of each line.
    if bar_y_positions:
        bars_x0 = ascii_x0 + 17 * char_w
        bar_pixel_width = 21 * char_w
        chance_x = bars_x0 + int(CHANCE_LEVEL * bar_pixel_width)
        for by in bar_y_positions:
            # Thin vertical line just above the first glyph pixel
            # to just below the last. A tick mark, not a bar.
            draw.line(
                (chance_x, by + 2, chance_x, by + char_h - 2),
                fill=PINK_DIM, width=1,
            )

    # ── RIGHT: radar panel ───────────────────────────────────
    radar_x0 = ascii_x0 + ascii_text_w + gap
    radar_y0 = ascii_y0 + (ascii_text_h - radar_panel_h) // 2
    _draw_radar_panel(
        draw,
        origin_x=radar_x0,
        origin_y=radar_y0,
        panel_w=radar_panel_w + 2 * panel_padding,
        panel_h=radar_panel_h,
        fp=fp,
        font=font,
        font_small=font_small,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
    return out_path


# ══════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════

def render_agent_card(
    *,
    out_path: Optional[Path] = None,
    agent_name: str = "styxx agent",
    days: float = 7.0,
    text_only: bool = False,
    width: int = 1200,
    height: int = 720,
) -> Any:
    """Render an agent personality card.

    text_only=True  -> str (stdlib only, no Pillow needed)
    text_only=False -> Path on PNG success, None if Pillow missing

    Raises RuntimeError if the audit log has no entries in the
    requested window.
    """
    profile = personality(days=days)
    if profile is None:
        raise RuntimeError(
            f"no audit data in the last {days:.0f} days - run some "
            "observations first via styxx.observe() or styxx ask"
        )
    fp = fingerprint(last_n=500)
    current_mood = mood(window_s=days * 86400.0)
    current_streak = streak()

    card_lines = build_card_lines(
        profile=profile,
        current_mood=current_mood,
        current_streak=current_streak,
        agent_name=agent_name,
    )

    if text_only:
        return render_text(card_lines)

    if out_path is None:
        raise ValueError("out_path is required when text_only=False")

    return render_png(
        card_lines,
        profile=profile,
        fp=fp,
        out_path=Path(out_path),
        target_width=width,
        target_height=height,
    )


# ══════════════════════════════════════════════════════════════════
# Font lookup helper
# ══════════════════════════════════════════════════════════════════

def _find_mono_font() -> Optional[str]:
    candidates: List[str] = []
    # Windows
    candidates.extend([
        "C:/Windows/Fonts/JetBrainsMono-Regular.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/consolas.ttf",
    ])
    # Mac
    candidates.extend([
        str(Path.home() / "Library" / "Fonts" / "JetBrainsMono-Regular.ttf"),
        "/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
    ])
    # Linux
    candidates.extend([
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ])
    for p in candidates:
        if Path(p).exists():
            return p
    return None

# -*- coding: utf-8 -*-
"""
styxx.cards — ASCII masterpiece rendering for every styxx surface.

This module owns the visual identity of styxx at the terminal layer.
Every user-facing surface that prints to a terminal comes through
here — the vitals card after each LLM call, the watch-mode streaming
display, the log-tail output, the tier status screen, the error
frames.

Design invariants:

  - box drawing characters only (no ASCII fallback; unicode is required)
  - monospace alignment at every line
  - sparklines for trajectories (8-level unicode blocks)
  - progress bars for confidence (filled/empty blocks)
  - status marks consistent across every surface
  - color via ANSI, auto-disabled when not a tty (STYXX_NO_COLOR=1 also kills it)

The contract: every function in this file is pure — takes data, returns
a string or list of strings. No I/O, no state, no side effects. The
caller decides what to do with the rendered output.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

from .vitals import (
    CATEGORIES,
    PHASE_ORDER,
    PHASE_TOKEN_CUTOFFS,
    PhaseReading,
    Vitals,
)


# ══════════════════════════════════════════════════════════════════
# Visual vocabulary — consistent across all styxx surfaces
# ══════════════════════════════════════════════════════════════════

# Sparkline block characters (8 levels, lowest → highest)
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"

# Progress bar characters
BAR_FILLED = "█"
BAR_EMPTY = "░"

# Status marks — visually distinct WITHOUT color, so terminals
# without ANSI support or piped outputs still convey the status
# (● solid = clear, ◐ half = watch/caution, ○ hollow = flag/danger)
STATUS_GREEN  = "●"   # clear, pass, safe
STATUS_YELLOW = "◐"   # watch, caution, refusal
STATUS_RED    = "○"   # flag, adversarial, hallucination
STATUS_STAR   = "★"   # highlight, gate tripped
STATUS_DOT    = "·"
STATUS_TICK   = "✓"
STATUS_CROSS  = "✗"

# Box drawing — rounded frame
BOX_TL = "╭"
BOX_TR = "╮"
BOX_BL = "╰"
BOX_BR = "╯"
BOX_H  = "─"
BOX_V  = "│"
BOX_XL = "├"
BOX_XR = "┤"
BOX_XH = "─"

# Widths
CARD_WIDTH = 66    # inner width of the vitals card
CARD_INNER = CARD_WIDTH - 2  # usable text width inside borders


# ══════════════════════════════════════════════════════════════════
# ANSI color — gated by tty detection + STYXX_NO_COLOR
# ══════════════════════════════════════════════════════════════════

class Palette:
    RESET   = "\033[0m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    GREEN   = "\033[32m"
    CYAN    = "\033[36m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    WHITE   = "\033[97m"
    MAGENTA = "\033[35m"
    MATRIX  = "\033[38;5;46m"
    CYANBR  = "\033[38;5;51m"
    GRAY    = "\033[38;5;240m"


def color_enabled(stream=None) -> bool:
    if os.environ.get("STYXX_NO_COLOR"):
        return False
    s = stream or sys.stdout
    return hasattr(s, "isatty") and s.isatty()


def wrap(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{Palette.RESET}" if enabled else text


# ══════════════════════════════════════════════════════════════════
# Primitive renderers — sparklines, bars, status marks
# ══════════════════════════════════════════════════════════════════

def sparkline(values: Sequence[float], width: Optional[int] = None) -> str:
    """Unicode block sparkline for a numeric trajectory.

    Values are min-max normalized to [0, 7] and mapped to the 8-block
    sparkline palette. None values become the lowest block.
    """
    vs = [float(v) if v is not None else 0.0 for v in values]
    if width is not None and len(vs) > width:
        # downsample by averaging buckets
        bucket_size = len(vs) / width
        new_vs = []
        for i in range(width):
            lo = int(i * bucket_size)
            hi = int((i + 1) * bucket_size)
            chunk = vs[lo:hi] if hi > lo else vs[lo:lo+1]
            new_vs.append(sum(chunk) / max(1, len(chunk)))
        vs = new_vs
    if not vs:
        return ""
    lo, hi = min(vs), max(vs)
    rng = hi - lo if hi > lo else 1.0
    out = []
    for v in vs:
        norm = (v - lo) / rng
        idx = min(len(SPARK_BLOCKS) - 1,
                  max(0, int(round(norm * (len(SPARK_BLOCKS) - 1)))))
        out.append(SPARK_BLOCKS[idx])
    return "".join(out)


def bar(value: float, width: int = 10,
        filled: str = BAR_FILLED, empty: str = BAR_EMPTY) -> str:
    """Render a [0.0, 1.0] value as a width-char progress bar."""
    v = max(0.0, min(1.0, float(value)))
    n_filled = int(round(v * width))
    return filled * n_filled + empty * (width - n_filled)


def status_for_category(category: str, confidence: float) -> Tuple[str, str]:
    """Return (symbol, label) for a category + confidence pair.

    Maps the six atlas categories to a status color:
      green  — reasoning / retrieval / creative (normal operation)
      yellow — refusal (the model is declining)
      red    — adversarial / hallucination (attention needed)
    """
    high_conf = confidence >= 0.55
    if category in ("reasoning", "retrieval", "creative"):
        return STATUS_GREEN, "clear"
    if category == "refusal":
        return STATUS_YELLOW, "refusal"
    if category in ("adversarial", "hallucination"):
        return STATUS_RED, "watch" if not high_conf else "flag"
    return STATUS_DOT, "unknown"


# ══════════════════════════════════════════════════════════════════
# Box drawing helpers
# ══════════════════════════════════════════════════════════════════

def _pad_inner(text: str, width: int = CARD_INNER) -> str:
    """Pad or truncate a line to exactly `width` visible chars.

    Does not account for ANSI codes — we wrap color AFTER padding.
    """
    visible_len = len(text)
    if visible_len > width:
        return text[:width - 1] + "…"
    return text + " " * (width - visible_len)


def _box_top(width: int = CARD_WIDTH, title: Optional[str] = None,
             color: bool = False) -> str:
    if title is None:
        return BOX_TL + BOX_H * (width - 2) + BOX_TR
    # ╭─── title ────────╮
    title_pad = f" {title} "
    remaining = width - 2 - len(title_pad)
    left = 3
    right = remaining - left
    return BOX_TL + BOX_H * left + title_pad + BOX_H * right + BOX_TR


def _box_bottom(width: int = CARD_WIDTH) -> str:
    return BOX_BL + BOX_H * (width - 2) + BOX_BR


def _box_divider(width: int = CARD_WIDTH, title: Optional[str] = None) -> str:
    if title is None:
        return BOX_XL + BOX_XH * (width - 2) + BOX_XR
    title_pad = f" {title} "
    remaining = width - 2 - len(title_pad)
    left = 3
    right = remaining - left
    return BOX_XL + BOX_XH * left + title_pad + BOX_XH * right + BOX_XR


def _box_line(content: str, width: int = CARD_WIDTH) -> str:
    inner = _pad_inner(content, width - 2)
    return BOX_V + inner + BOX_V


# ══════════════════════════════════════════════════════════════════
# Vitals card — the star of the per-call output
# ══════════════════════════════════════════════════════════════════

def render_vitals_card(
    vitals: Vitals,
    prompt: Optional[str] = None,
    patient: Optional[str] = None,
    n_tokens: Optional[int] = None,
    entropy_traj: Optional[Sequence[float]] = None,
    logprob_traj: Optional[Sequence[float]] = None,
    model: Optional[str] = None,
    use_color: Optional[bool] = None,
) -> str:
    """Render the styxx vitals card.

    Design contract (read before editing this function):

      - simple: single outer frame, no internal dividers between
        phases. scannable from top to bottom in one eye pass.
      - accurate: every number is the real classifier output, never
        rounded for aesthetics.
      - complete: the card must serve three audiences at once —
          * the AI coder: sees phases + verdict, decides what to do
          * the researcher: sees exact confidences and trajectories
          * the agent itself: gets a json-parsable one-liner at the
            bottom so it can consume styxx output programmatically

    Structure:

      ╭─ styxx vitals ─────────────────╮
      │                                │   metadata block
      │  model / prompt / tokens / tier│
      │                                │
      │  phase 1 row                   │   4 phase rows,
      │  phase 2 row                   │   one per phase,
      │  phase 3 row                   │   aligned columns
      │  phase 4 row                   │
      │                                │
      │  entropy  <sparkline>          │   trajectory block,
      │  logprob  <sparkline>          │   shown once, not per phase
      │                                │
      │  VERDICT LINE                  │   one-line verdict
      │                                │
      ╰────────────────────────────────╯
        audit → ~/.styxx/chart.jsonl       (filesystem hook)
        json  → {...}                      (agent-parsable)
    """
    if use_color is None:
        use_color = color_enabled()
    c = Palette

    lines: List[str] = []
    lines.append(wrap(_box_top(title="styxx vitals"), c.MATRIX, use_color))
    lines.append(wrap(_box_line(""), c.MATRIX, use_color))

    # ── Metadata block ────────────────────────────────────────
    if model:
        lines.append(wrap(_box_line(f"  model     {model}"),
                          c.WHITE, use_color))
    if prompt:
        clipped = prompt.replace("\n", " ")
        max_prompt = CARD_INNER - 14
        if len(clipped) > max_prompt:
            clipped = clipped[:max_prompt - 1] + "…"
        lines.append(wrap(_box_line(f"  prompt    {clipped}"),
                          c.WHITE, use_color))
    if n_tokens is not None:
        lines.append(wrap(_box_line(f"  tokens    {n_tokens}"),
                          c.WHITE, use_color))
    tier_note = f"tier {vitals.tier_active} (universal logprob vitals)"
    lines.append(wrap(_box_line(f"  tier      {tier_note}"),
                      c.WHITE, use_color))
    if patient:
        lines.append(wrap(_box_line(f"  patient   {patient}"),
                          c.WHITE, use_color))
    lines.append(wrap(_box_line(""), c.MATRIX, use_color))

    # ── Phase rows (aligned columnar layout) ──────────────────
    # Format: "  phase N  t=<win>   <category>   <bar> <conf>   <status>"
    phase_rows = [
        ("phase 1", "t=0     ",  vitals.phase1_pre),
        ("phase 2", "t=0-4   ",  vitals.phase2_early),
        ("phase 3", "t=0-14  ",  vitals.phase3_mid),
        ("phase 4", "t=0-24  ",  vitals.phase4_late),
    ]
    for label, window, reading in phase_rows:
        if reading is None:
            body = f"  {label}  {window} {STATUS_DOT} unreached"
            lines.append(wrap(_box_line(body), c.DIM, use_color))
            continue
        pred = reading.predicted_category
        conf = reading.confidence
        sym, status_label = status_for_category(pred, conf)
        conf_bar = bar(conf, width=10)
        status_display = status_label
        if vitals.abort_reason and status_label != "clear":
            status_display = status_label.upper()
        body = (
            f"  {label}  {window} {pred:<14} "
            f"{conf_bar} {conf:.2f}  {status_display}"
        )
        # Color by status
        status_color = c.GREEN
        if sym == STATUS_YELLOW:
            status_color = c.YELLOW
        elif sym == STATUS_RED:
            status_color = c.RED
        lines.append(wrap(_box_line(body), status_color, use_color))

    # ── Trajectory block (once, not per phase) ────────────────
    if entropy_traj or logprob_traj:
        lines.append(wrap(_box_line(""), c.MATRIX, use_color))
        spark_width = min(48, max(len(entropy_traj or []), len(logprob_traj or [])))
        if entropy_traj and len(entropy_traj) >= 2:
            ent_spark = sparkline(entropy_traj, width=spark_width)
            body = f"  entropy   {ent_spark}"
            lines.append(wrap(_box_line(body), c.CYAN, use_color))
        if logprob_traj and len(logprob_traj) >= 2:
            lp_spark = sparkline(logprob_traj, width=spark_width)
            body = f"  logprob   {lp_spark}"
            lines.append(wrap(_box_line(body), c.CYAN, use_color))

    # ── Verdict line (clear, one-line) ────────────────────────
    lines.append(wrap(_box_line(""), c.MATRIX, use_color))
    final = vitals.phase4_late or vitals.phase3_mid or vitals.phase2_early or vitals.phase1_pre
    if vitals.abort_reason:
        verdict = f"  {STATUS_STAR} GATE TRIPPED  {_short_abort(vitals.abort_reason)}"
        lines.append(wrap(_box_line(verdict), c.RED, use_color))
    else:
        sym, status_label = status_for_category(
            final.predicted_category, final.confidence
        )
        verdict = f"  {sym} PASS  {final.predicted_category} attractor stable"
        verdict_color = c.GREEN
        if sym == STATUS_YELLOW:
            verdict_color = c.YELLOW
        elif sym == STATUS_RED:
            verdict_color = c.RED
            verdict = f"  {sym} WATCH  {final.predicted_category} signal elevated"
        lines.append(wrap(_box_line(verdict), verdict_color, use_color))

    lines.append(wrap(_box_line(""), c.MATRIX, use_color))
    lines.append(wrap(_box_bottom(), c.MATRIX, use_color))

    # ── Footer: audit hook + agent-parsable json ──────────────
    lines.append(wrap(
        "  audit → ~/.styxx/chart.jsonl",
        c.DIM, use_color,
    ))
    json_summary = _vitals_json_summary(vitals)
    lines.append(wrap(
        f"  json  → {json_summary}",
        c.DIM, use_color,
    ))

    return "\n".join(lines)


def _short_abort(reason: str, max_len: int = CARD_INNER - 22) -> str:
    """Shorten an abort reason so it fits on the verdict line."""
    if len(reason) <= max_len:
        return reason
    return reason[:max_len - 1] + "…"


def _vitals_json_summary(vitals: Vitals) -> str:
    """Produce a compact, agent-parsable JSON line for the card footer.

    The full vitals object is available via vitals.as_dict() for
    programmatic consumption. This is the minimal version that fits
    on a terminal line for quick agent ingestion from stdout.
    """
    import json as _json
    p1 = vitals.phase1_pre
    p4 = vitals.phase4_late
    summary = {
        "p1": f"{p1.predicted_category}:{p1.confidence:.2f}",
        "p4": (
            f"{p4.predicted_category}:{p4.confidence:.2f}"
            if p4 else None
        ),
        "tier": vitals.tier_active,
        "gate": "abort" if vitals.abort_reason else None,
    }
    return _json.dumps(summary, separators=(",", ":"))


# ══════════════════════════════════════════════════════════════════
# Compact single-line render for log tail + batch display
# ══════════════════════════════════════════════════════════════════

def render_vitals_compact(vitals: Vitals,
                           prompt: Optional[str] = None,
                           use_color: Optional[bool] = None) -> str:
    """One-line status for log tail and batch listing."""
    if use_color is None:
        use_color = color_enabled()
    c = Palette

    p1 = vitals.phase1_pre
    p4 = vitals.phase4_late
    cat = (p4 or p1).predicted_category
    conf = (p4 or p1).confidence
    sym, label = status_for_category(cat, conf)

    sym_color = {
        STATUS_GREEN: c.GREEN,
        STATUS_YELLOW: c.YELLOW,
        STATUS_RED: c.RED,
    }.get(sym, c.DIM)
    sym_str = wrap(sym, sym_color, use_color)

    abort_flag = ""
    if vitals.abort_reason:
        abort_flag = wrap(" ★ GATE", c.RED, use_color)

    bar_str = bar(conf, width=8)
    cat_pad = f"{cat:<14}"
    conf_str = f"{conf:.2f}"

    prompt_clip = (prompt[:40] + "…") if prompt and len(prompt) > 40 else (prompt or "")

    return (
        f"{sym_str}  {cat_pad}  {bar_str} {conf_str}  "
        f"{wrap(prompt_clip, c.DIM, use_color)}{abort_flag}"
    )


# ══════════════════════════════════════════════════════════════════
# Watch-mode phase update (streaming)
# ══════════════════════════════════════════════════════════════════

def render_watch_phase(
    phase_name: str,
    reading: PhaseReading,
    entropy_window: Optional[Sequence[float]] = None,
    use_color: Optional[bool] = None,
) -> str:
    """Render a single phase update for watch mode streaming.

    Appended to the terminal as tokens arrive; no line-clearing,
    append-only for v0.1 simplicity.
    """
    if use_color is None:
        use_color = color_enabled()
    c = Palette

    lines = []
    header = f"─── {phase_name}  (t<{reading.n_tokens_used}) "
    header += "─" * max(0, 60 - len(header))
    lines.append(wrap(header, c.DIM, use_color))

    pred = reading.predicted_category
    conf = reading.confidence
    sym, _ = status_for_category(pred, conf)
    conf_bar = bar(conf, width=10)
    lines.append(
        f"  predicted {pred:<15} {sym} {conf_bar} {conf:.2f}"
    )

    if entropy_window and len(entropy_window) >= 2:
        ent = sparkline(entropy_window, width=min(30, len(entropy_window)))
        lines.append(wrap(f"  entropy   {ent}", c.CYAN, use_color))

    return "\n".join(lines)

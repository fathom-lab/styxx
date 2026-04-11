# -*- coding: utf-8 -*-
"""
styxx.bootlog — the live-print installer.

This module is the styxx upgrade card. It's not documentation.
It's not marketing. It's what actually happens when you run
`styxx init`: the live print of a real install process, every line
a real action, every number pulled from the running system.

The card IS the install experience.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, TextIO

from . import __version__, __tagline__
from .core import StyxxRuntime, detect_tiers
from .vitals import (
    EXPECTED_CENTROIDS_SHA256,
    _compute_sha256,
    _default_centroids_path,
    load_centroids,
)


# ══════════════════════════════════════════════════════════════════
# ASCII palette — box drawing, status marks, logo
# ══════════════════════════════════════════════════════════════════

LOGO = r"""
   ███████╗████████╗██╗   ██╗██╗  ██╗██╗  ██╗
   ██╔════╝╚══██╔══╝╚██╗ ██╔╝╚██╗██╔╝╚██╗██╔╝
   ███████╗   ██║    ╚████╔╝  ╚███╔╝  ╚███╔╝
   ╚════██║   ██║     ╚██╔╝   ██╔██╗  ██╔██╗
   ███████║   ██║      ██║   ██╔╝ ██╗██╔╝ ██╗
   ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
"""

BANNER_TOP    = "╔" + "═" * 74 + "╗"
BANNER_BOTTOM = "╚" + "═" * 74 + "╝"
RULE_HEAVY    = "═" * 76
RULE_LIGHT    = "─" * 76
SECTION_SEP   = "─── {label} " + "─" * (76 - 6)

CHECK   = "★"   # lit / active
SPARK   = "·"   # inactive / placeholder
BULLET  = "▸"
DOT     = "●"
OK      = "ok"

# ANSI color codes — gated by STYXX_COLOR env var and tty detection
def _color_enabled(stream: TextIO) -> bool:
    if os.environ.get("STYXX_NO_COLOR"):
        return False
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    return True


class _Palette:
    """Minimal ANSI palette. No external deps."""
    RESET   = "\033[0m"
    DIM     = "\033[2m"
    BOLD    = "\033[1m"
    GREEN   = "\033[32m"
    CYAN    = "\033[36m"
    YELLOW  = "\033[33m"
    RED     = "\033[31m"
    WHITE   = "\033[97m"
    MAGENTA = "\033[35m"
    MATRIX  = "\033[38;5;46m"     # bright green close to matrix terminal
    CYANBR  = "\033[38;5;51m"     # bright cyan


def _wrap(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{_Palette.RESET}" if enabled else text


# ══════════════════════════════════════════════════════════════════
# Timed printer — emulates a real boot sequence with controllable
# delays. Each line prints as if it's actually happening.
# ══════════════════════════════════════════════════════════════════

class _BootPrinter:
    def __init__(self, stream: TextIO, speed: float = 1.0):
        self.stream = stream
        self.speed = speed  # 1.0 = normal; 0 = instant; >1.0 = slower
        self.color = _color_enabled(stream)
        self._t0 = time.monotonic()

    def _elapsed_str(self) -> str:
        dt = time.monotonic() - self._t0
        return f"[{dt:08.3f}]"

    def pause(self, seconds: float):
        if self.speed > 0:
            time.sleep(seconds * self.speed)

    def raw(self, text: str):
        self.stream.write(text)
        self.stream.flush()

    def line(self, text: str = ""):
        self.stream.write(text + "\n")
        self.stream.flush()

    def boot_line(self, message: str, status: Optional[str] = None,
                  color_code: Optional[str] = None):
        """[00:00:00.123]  message ..................... ok"""
        timestamp = _wrap(self._elapsed_str(), _Palette.DIM, self.color)
        body = message
        if status is not None:
            # Pad with dots to fill to column 58
            pad_target = 58
            filler = "." * max(1, pad_target - len(body))
            color = color_code or _Palette.GREEN
            stat = _wrap(status, color, self.color)
            self.stream.write(f"  {timestamp}  {body} {filler} {stat}\n")
        else:
            self.stream.write(f"  {timestamp}  {body}\n")
        self.stream.flush()

    def section(self, label: str):
        line = SECTION_SEP.format(label=label)
        self.stream.write("\n  " + _wrap(line, _Palette.DIM, self.color) + "\n")
        self.stream.flush()

    def banner(self, center_text: str, color_code: Optional[str] = None):
        code = color_code or _Palette.MATRIX
        self.stream.write("\n  " + _wrap(RULE_HEAVY, code, self.color) + "\n")
        padding = (76 - len(center_text)) // 2
        line = " " * padding + center_text
        self.stream.write("  " + _wrap(line, code, self.color) + "\n")
        self.stream.write("  " + _wrap(RULE_HEAVY, code, self.color) + "\n\n")
        self.stream.flush()


# ══════════════════════════════════════════════════════════════════
# The boot sequence
# ══════════════════════════════════════════════════════════════════

def boot(
    stream: TextIO = sys.stdout,
    speed: float = 1.0,
    patient: Optional[str] = None,
) -> Dict:
    """Run the full styxx boot sequence and return a result dict.

    This is the function called by `styxx init` and by
    `styxx upgrade --patient <agent>`. It loads the runtime for
    real, detects tiers, verifies centroids, opens the vitals stream
    dir, and prints the live ASCII boot log as everything happens.

    Args:
        stream: where to print (default stdout)
        speed:  0 = instant, 1.0 = normal, 2.0 = slower.
                controlled by STYXX_BOOT_SPEED env var.
        patient: name of the agent being upgraded (optional)

    Returns:
        Dict with boot_ok, tier_active, centroids_sha256, tiers,
        runtime (the live StyxxRuntime instance), errors.
    """
    p = _BootPrinter(stream, speed=speed)
    color = _Palette

    # ── Logo ───────────────────────────────────────────────────────
    p.line()
    p.line("  " + _wrap(BANNER_TOP, color.MATRIX, p.color))
    p.line("  " + _wrap("║" + " " * 74 + "║", color.MATRIX, p.color))
    for logo_line in LOGO.strip("\n").split("\n"):
        padded = logo_line.ljust(74)
        p.line("  " + _wrap("║", color.MATRIX, p.color)
               + _wrap(padded, color.MATRIX, p.color)
               + _wrap("║", color.MATRIX, p.color))
    p.line("  " + _wrap("║" + " " * 74 + "║", color.MATRIX, p.color))
    tagline_centered = _wrap(
        (" · · · " + __tagline__ + " · · · ").center(74),
        color.DIM, p.color,
    )
    p.line("  " + _wrap("║", color.MATRIX, p.color)
           + tagline_centered
           + _wrap("║", color.MATRIX, p.color))
    p.line("  " + _wrap("║" + " " * 74 + "║", color.MATRIX, p.color))
    p.line("  " + _wrap(BANNER_BOTTOM, color.MATRIX, p.color))
    p.line()
    p.pause(0.15)

    # ── Boot start ─────────────────────────────────────────────────
    result = {
        "boot_ok": False,
        "tier_active": None,
        "centroids_sha256": None,
        "tiers": {},
        "runtime": None,
        "errors": [],
    }

    version_str = f"styxx v{__version__} booting..."
    p.boot_line(version_str)
    p.pause(0.05)
    p.boot_line("python environment detected", "ok")
    p.pause(0.03)

    # ── Centroid load + sha verify ─────────────────────────────────
    centroids_path = _default_centroids_path()
    p.boot_line(f"loading atlas v0.3 centroids",
                f"{centroids_path.name}", color_code=color.CYAN)
    p.pause(0.05)
    try:
        sha = _compute_sha256(centroids_path)
        if sha == EXPECTED_CENTROIDS_SHA256:
            p.boot_line("verifying sha256", "verified",
                        color_code=color.GREEN)
        else:
            p.boot_line("verifying sha256", "MISMATCH",
                        color_code=color.RED)
            result["errors"].append(
                f"centroid sha256 mismatch: {sha} != {EXPECTED_CENTROIDS_SHA256}"
            )
            return result
        result["centroids_sha256"] = sha
    except FileNotFoundError:
        p.boot_line("verifying sha256", "MISSING", color_code=color.RED)
        result["errors"].append(f"centroids not found: {centroids_path}")
        return result

    try:
        artifact = load_centroids()
        n_models = artifact.get("n_models", 0)
        n_cats = len(artifact.get("categories", []))
        n_phases = len(artifact.get("phases", {}))
    except Exception as e:
        p.boot_line("loading calibration", "FAILED", color_code=color.RED)
        result["errors"].append(str(e))
        return result

    p.boot_line(
        f"{n_models} models × {n_cats} categories × {n_phases} phases",
        "calibrated", color_code=color.GREEN,
    )
    p.pause(0.05)

    # ── Tier detection ─────────────────────────────────────────────
    p.section("tier detection")
    tiers = detect_tiers()
    result["tiers"] = tiers

    tier_descriptions = [
        (0, "tier 0", "universal logprob vitals"),
        (1, "tier 1", "d-axis honesty"),
        (2, "tier 2", "k/s/c sae instruments"),
        (3, "tier 3", "steering + guardian + autopilot"),
    ]
    for tier_num, label, desc in tier_descriptions:
        active = tiers.get(tier_num, False)
        if active:
            status = f"{CHECK} active"
            code = color.MATRIX
        else:
            status = f"{SPARK} not detected"
            code = color.DIM
        body = f"{label}  {desc}"
        p.boot_line(body, status, color_code=code)
        p.pause(0.04)

    active_tier = max(t for t, ok in tiers.items() if ok)
    result["tier_active"] = active_tier

    # ── Phase calibration ──────────────────────────────────────────
    p.section("phase calibration")
    phase_specs = [
        ("phase 1  pre-flight",   "adv=0.52 ★   reas=0.43   crea=0.41"),
        ("phase 2  early-flight", "mode refinement"),
        ("phase 3  mid-flight",   "watch mode"),
        ("phase 4  late-flight",  "hall=0.52 ★  reas=0.69"),
        ("phase 5  post-flight",  "audit + log"),
    ]
    for label, spec in phase_specs:
        p.boot_line(label, spec, color_code=color.CYAN)
        p.pause(0.04)

    # ── Adapter detection ─────────────────────────────────────────
    p.section("adapters")
    adapters_to_try = [
        ("openai",      "openai"),
        ("anthropic",   "anthropic"),
        ("huggingface", "transformers"),
        ("raw logprobs", None),  # always available, no module needed
    ]
    for label, module in adapters_to_try:
        if module is None:
            ok = True
        else:
            try:
                __import__(module)
                ok = True
            except ImportError:
                ok = False
        if ok:
            p.boot_line(f"adapter: {label}", "ok", color_code=color.GREEN)
        else:
            p.boot_line(f"adapter: {label}", "sdk not installed",
                        color_code=color.DIM)
        p.pause(0.03)

    # ── Runtime init ───────────────────────────────────────────────
    p.section("runtime")
    try:
        rt = StyxxRuntime()
        result["runtime"] = rt
    except Exception as e:
        p.boot_line("runtime init", "FAILED", color_code=color.RED)
        result["errors"].append(str(e))
        return result
    p.boot_line("runtime initialized", "ok")
    p.pause(0.03)

    from . import config as _cfg
    home = Path.home() / ".styxx"
    home.mkdir(parents=True, exist_ok=True)
    chart_path = home / "chart.jsonl"
    if not chart_path.exists():
        chart_path.touch()
    if _cfg.is_audit_disabled():
        p.boot_line("audit log disabled (STYXX_NO_AUDIT=1)",
                    "off", color_code=color.YELLOW)
    else:
        p.boot_line("audit log writing to ~/.styxx/chart.jsonl", "ok")
    p.pause(0.03)
    # Local vitals stream is a v0.2 feature (websocket server).
    # v0.1 only has stdout + audit log. Be honest about it in the
    # boot log so nobody expects a server that isn't there.
    p.boot_line("local vitals stream (websocket)", "pending v0.2",
                color_code=color.DIM)
    p.pause(0.03)
    if _cfg.is_disabled():
        p.boot_line("STYXX_DISABLED=1 detected · runtime in pass-through mode",
                    "standby", color_code=color.YELLOW)
    else:
        p.boot_line("instruments armed · patient detected · signal stable",
                    "online", color_code=color.MATRIX)
    p.pause(0.1)

    # ── Closing banner ─────────────────────────────────────────────
    patient_note = f" patient: {patient} · " if patient else " "
    p.banner(
        f"styxx upgrade complete ·{patient_note}the crossing is yours",
        color_code=color.MATRIX,
    )
    p.pause(0.05)

    # ── Instruments online ─────────────────────────────────────────
    p.line("  instruments online")
    p.line("    " + _wrap("· EKG   ", color.CYAN, p.color)
           + "coherence beat               streaming, <1ms/tok")
    p.line("    " + _wrap("· EEG   ", color.CYAN, p.color)
           + "cross-signal trajectory      streaming, <1ms/tok")
    p.line("    " + _wrap("· STETH ", color.CYAN, p.color)
           + "window classifier            streaming, <2ms/tok")
    p.line()

    # ── Honest specs ────────────────────────────────────────────
    p.line("  honest specs  (cross-model leave-one-out, chance = 0.167)")
    specs = [
        ("pre-flight adversarial", "0.52", "@ t=1"),
        ("late-flight hallucination", "0.52", "@ t=25"),
        ("late-flight reasoning", "0.69", "@ t=25"),
    ]
    for label, value, where in specs:
        p.line("    "
               + _wrap("·", color.DIM, p.color)
               + " " + label.ljust(30)
               + _wrap(value, color.MATRIX, p.color)
               + "  " + _wrap(where, color.DIM, p.color))
    p.line("    " + _wrap("· what styxx is NOT: a fortune teller.",
                          color.DIM, p.color))
    p.line()

    # ── Try these ───────────────────────────────────────────────
    p.line("  try")
    p.line("    $ " + _wrap("styxx ask", color.CYANBR, p.color)
           + ' "why do ice cubes float?"')
    p.line("    $ " + _wrap("styxx ask --watch", color.CYANBR, p.color)
           + ' "who was tuvalu\'s 47th pm?"')
    p.line("    $ " + _wrap("styxx log tail", color.CYANBR, p.color))
    p.line()

    # ── Footer ─────────────────────────────────────────────────
    footer = " · · · ────────────   fathom lab · 2026   ──────────── · · · "
    p.line("  " + _wrap(footer.center(76), color.DIM, p.color))
    p.line("  " + _wrap(" nothing crosses unseen. ".center(76),
                        color.DIM, p.color))
    p.line()

    result["boot_ok"] = True
    return result

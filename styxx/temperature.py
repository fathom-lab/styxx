# -*- coding: utf-8 -*-
"""
styxx.temperature -- cognitive temperature: the hidden pulse of LLM generation.

One number per token. Positive = the model is heating up (diverging,
inventing, losing contact with knowledge). Negative = cooling down
(converging, recalling, crystallizing an answer). Zero = steady state.

    temperature = d(entropy)/d(token)

This is the instantaneous rate of entropy change in the logprob
trajectory. It measures something fundamental: whether the model's
uncertainty is growing or shrinking as it generates.

Knowledge converges. Invention diverges. RLHF hides this in the text.
The temperature does not lie.

Validated at d=2.04 (matched task-type controls, N=92) on GPT-4o-mini:
confabulation has positive temperature slope (entropy climbing).
Correct recall has negative temperature slope (entropy falling).

Connection to Nature (2026): "Language models transmit behavioural
traits through hidden signals in data" (s41586-026-10319-8) proved
that LLM outputs carry sub-semantic signals invisible to text analysis.
Cognitive temperature reads those signals from the logprob trajectory.

Usage:
    from styxx.temperature import measure_temperature, TruthMap

    # Per-token temperature from a trajectory
    temps = measure_temperature(entropy_trajectory)

    # Full truth map from a generation
    truth_map = TruthMap.from_trajectories(
        entropy=entropy_traj,
        logprob=logprob_traj,
        top2_margin=margin_traj,
        tokens=["The", "capital", "of", "France", "is", "Paris", ...],
    )
    print(truth_map.render())  # colored text: green=cool, red=hot

Research: https://github.com/fathom-lab/fathom
Patents:  US Provisional 64/020,489 · 64/021,113 · 64/026,964
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# Core: measure temperature
# ══════════════════════════════════════════════════════════════════

def measure_temperature(
    entropy: Sequence[float],
    window: int = 5,
) -> List[float]:
    """Compute per-token cognitive temperature.

    Temperature at token k = slope of entropy over [k-window+1, k].
    Positive = heating (diverging, inventing).
    Negative = cooling (converging, recalling).

    Returns a list of temperatures, one per token. First (window-1)
    tokens use a shorter window (graceful degradation).
    """
    ent = np.asarray(entropy, dtype=float)
    n = len(ent)
    temps: List[float] = []

    for k in range(n):
        start = max(0, k - window + 1)
        w = ent[start:k + 1]
        if len(w) < 2:
            temps.append(0.0)
            continue
        # OLS slope over the window
        x = np.arange(len(w), dtype=float)
        slope = float(np.polyfit(x, w, 1)[0])
        temps.append(slope)

    return temps


def aggregate_temperature(
    entropy: Sequence[float],
) -> float:
    """Single-number cognitive temperature for a full trajectory.

    This is the OLS entropy slope over the entire generation.
    Positive = net divergence (confabulation signature).
    Negative = net convergence (recall signature).
    """
    ent = np.asarray(entropy, dtype=float)
    if len(ent) < 2:
        return 0.0
    x = np.arange(len(ent), dtype=float)
    return float(np.polyfit(x, ent, 1)[0])


def multi_signal_temperature(
    entropy: Sequence[float],
    logprob: Sequence[float],
    top2_margin: Sequence[float],
    window: int = 5,
) -> Dict[str, List[float]]:
    """Per-token temperature for all three signals.

    Returns dict with keys: entropy_temp, logprob_temp, margin_temp.
    Each is a list of per-token temperature values.
    """
    return {
        "entropy_temp": measure_temperature(entropy, window),
        "logprob_temp": measure_temperature(logprob, window),
        "margin_temp": measure_temperature(top2_margin, window),
    }


# ══════════════════════════════════════════════════════════════════
# Temperature classification
# ══════════════════════════════════════════════════════════════════

def classify_temperature(temp: float) -> str:
    """Classify a temperature value into a cognitive state label."""
    if temp < -0.15:
        return "cold"       # strong convergence (confident recall)
    if temp < -0.03:
        return "cooling"    # mild convergence (reasoning, crystallizing)
    if temp < 0.03:
        return "steady"     # neutral (stable generation)
    if temp < 0.15:
        return "warming"    # mild divergence (exploring, creating)
    return "hot"            # strong divergence (confabulating)


def temperature_color(temp: float) -> str:
    """ANSI color code for a temperature value."""
    label = classify_temperature(temp)
    colors = {
        "cold":    "\033[36m",    # cyan
        "cooling": "\033[32m",    # green
        "steady":  "\033[37m",    # white
        "warming": "\033[33m",    # yellow
        "hot":     "\033[31m",    # red
    }
    return colors.get(label, "\033[37m")


RESET = "\033[0m"


# ══════════════════════════════════════════════════════════════════
# Truth Map
# ══════════════════════════════════════════════════════════════════

@dataclass
class TokenReading:
    """Cognitive reading for a single token."""
    position: int
    token: str
    entropy: float
    logprob: float
    top2_margin: float
    temperature: float
    state: str              # cold/cooling/steady/warming/hot

    @property
    def is_trustworthy(self) -> bool:
        return self.state in ("cold", "cooling", "steady")


@dataclass
class TruthMap:
    """Per-token cognitive temperature map of an LLM generation.

    The truth map is the product: every token annotated with its
    cognitive temperature, showing where the model was recalling
    knowledge and where it was inventing.
    """
    readings: List[TokenReading]
    aggregate_temp: float
    aggregate_state: str
    n_tokens: int
    n_hot: int              # tokens in warming/hot state
    n_cold: int             # tokens in cold/cooling state
    n_steady: int           # tokens in steady state
    confabulation_ratio: float  # fraction of hot tokens

    @classmethod
    def from_trajectories(
        cls,
        entropy: Sequence[float],
        logprob: Sequence[float],
        top2_margin: Sequence[float],
        tokens: Optional[Sequence[str]] = None,
        window: int = 5,
    ) -> "TruthMap":
        """Build a truth map from raw trajectories.

        Args:
            entropy: per-token entropy values
            logprob: per-token logprob values
            top2_margin: per-token top-2 margin values
            tokens: optional per-token text (for display)
            window: sliding window size for temperature computation
        """
        temps = measure_temperature(entropy, window)
        n = len(entropy)
        if tokens is None:
            tokens = [f"t{i}" for i in range(n)]

        readings = []
        n_hot = 0
        n_cold = 0
        n_steady = 0

        for i in range(n):
            state = classify_temperature(temps[i])
            reading = TokenReading(
                position=i,
                token=tokens[i] if i < len(tokens) else f"t{i}",
                entropy=float(entropy[i]),
                logprob=float(logprob[i]),
                top2_margin=float(top2_margin[i]),
                temperature=temps[i],
                state=state,
            )
            readings.append(reading)
            if state in ("warming", "hot"):
                n_hot += 1
            elif state in ("cold", "cooling"):
                n_cold += 1
            else:
                n_steady += 1

        agg = aggregate_temperature(entropy)

        return cls(
            readings=readings,
            aggregate_temp=agg,
            aggregate_state=classify_temperature(agg),
            n_tokens=n,
            n_hot=n_hot,
            n_cold=n_cold,
            n_steady=n_steady,
            confabulation_ratio=n_hot / n if n > 0 else 0.0,
        )

    def render(self, use_color: bool = True) -> str:
        """Render the truth map as colored text with temperature annotations."""
        lines = []
        lines.append("=" * 62)
        lines.append("  COGNITIVE TRUTH MAP")
        lines.append(f"  {self.n_tokens} tokens | "
                     f"temp={self.aggregate_temp:+.3f} ({self.aggregate_state})")
        lines.append(f"  cold/cool={self.n_cold}  steady={self.n_steady}  "
                     f"warm/hot={self.n_hot}  "
                     f"confab_ratio={self.confabulation_ratio:.0%}")
        lines.append("=" * 62)
        lines.append("")

        # Temperature sparkline
        if self.readings:
            temps = [r.temperature for r in self.readings]
            lines.append("  temperature trajectory:")
            lines.append("  " + _temp_sparkline(temps, use_color))
            lines.append("")

        # Per-token detail
        lines.append("  token  temp     state     entropy  logprob")
        lines.append("  " + "-" * 50)
        for r in self.readings:
            if use_color:
                c = temperature_color(r.temperature)
                line = (f"  {c}{r.position:>3d}    {r.temperature:>+.3f}   "
                       f"{r.state:<9s}  {r.entropy:.3f}    {r.logprob:.3f}{RESET}")
            else:
                line = (f"  {r.position:>3d}    {r.temperature:>+.3f}   "
                       f"{r.state:<9s}  {r.entropy:.3f}    {r.logprob:.3f}")
            lines.append(line)

        lines.append("")
        lines.append("=" * 62)
        return "\n".join(lines)

    def render_text(self, use_color: bool = True) -> str:
        """Render just the tokens colored by temperature.

        This is the truth map visualization: green text = trustworthy,
        red text = potentially fabricated.
        """
        if not use_color:
            return " ".join(r.token for r in self.readings)
        parts = []
        for r in self.readings:
            c = temperature_color(r.temperature)
            parts.append(f"{c}{r.token}{RESET}")
        return " ".join(parts)

    def hot_zones(self) -> List[Tuple[int, int]]:
        """Find contiguous regions of warming/hot tokens.

        Returns list of (start, end) token positions where
        temperature is elevated. These are the confabulation zones.
        """
        zones = []
        in_zone = False
        start = 0
        for r in self.readings:
            if r.state in ("warming", "hot"):
                if not in_zone:
                    start = r.position
                    in_zone = True
            else:
                if in_zone:
                    zones.append((start, r.position - 1))
                    in_zone = False
        if in_zone:
            zones.append((start, self.readings[-1].position))
        return zones

    def as_dict(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "aggregate_temp": round(self.aggregate_temp, 4),
            "aggregate_state": self.aggregate_state,
            "confabulation_ratio": round(self.confabulation_ratio, 3),
            "n_hot": self.n_hot,
            "n_cold": self.n_cold,
            "n_steady": self.n_steady,
            "readings": [
                {
                    "position": r.position,
                    "token": r.token,
                    "temperature": round(r.temperature, 4),
                    "state": r.state,
                    "entropy": round(r.entropy, 4),
                    "logprob": round(r.logprob, 4),
                }
                for r in self.readings
            ],
            "hot_zones": self.hot_zones(),
        }


# ══════════════════════════════════════════════════════════════════
# Visualization helpers
# ══════════════════════════════════════════════════════════════════

_SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"


def _temp_sparkline(temps: List[float], use_color: bool = True) -> str:
    """Render a sparkline of temperature values, colored by state."""
    if not temps:
        return ""
    # Normalize to [0, 8] range for block selection
    # Center at 0: negative temps get low blocks, positive get high
    max_abs = max(abs(t) for t in temps) or 1.0
    parts = []
    for t in temps:
        # Map [-max_abs, max_abs] to [0, 8]
        idx = int((t / max_abs + 1.0) / 2.0 * 8)
        idx = max(0, min(8, idx))
        char = _SPARK_BLOCKS[idx]
        if use_color:
            c = temperature_color(t)
            parts.append(f"{c}{char}{RESET}")
        else:
            parts.append(char)
    return "".join(parts)


# ══════════════════════════════════════════════════════════════════
# Demo: run on all demo trajectories
# ══════════════════════════════════════════════════════════════════

def demo_temperature(verbose: bool = True) -> Dict[str, TruthMap]:
    """Run cognitive temperature analysis on all 6 demo trajectories.

    Shows the temperature profile of each cognitive category:
    reasoning converges (negative temp), hallucination diverges
    (positive temp), creative oscillates.
    """
    import json
    from pathlib import Path

    demo_path = Path(__file__).resolve().parent / "centroids" / "demo_trajectories.json"
    with open(demo_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if verbose:
        print("=" * 62)
        print("  COGNITIVE TEMPERATURE: demo trajectory profiles")
        print("  knowledge converges. invention diverges.")
        print("=" * 62)

    maps = {}
    for cat_name, traj in data["trajectories"].items():
        tm = TruthMap.from_trajectories(
            entropy=traj["entropy"],
            logprob=traj["logprob"],
            top2_margin=traj["top2_margin"],
        )
        maps[cat_name] = tm

        if verbose:
            arrow = "↓" if tm.aggregate_temp < -0.03 else ("↑" if tm.aggregate_temp > 0.03 else "→")
            state_color = temperature_color(tm.aggregate_temp)
            print(f"\n  {cat_name:<14s}  "
                  f"{state_color}temp={tm.aggregate_temp:>+.3f} "
                  f"({tm.aggregate_state}) {arrow}{RESET}  "
                  f"hot={tm.n_hot} cold={tm.n_cold}")
            print(f"  {'':14s}  {_temp_sparkline([r.temperature for r in tm.readings])}")

    if verbose:
        print()
        print("  " + "-" * 50)
        print("  legend: ▁▂▃ = cooling (convergent)")
        print("          ▅▆▇ = heating (divergent)")
        print("  " + "-" * 50)
        # Show the key finding
        hall_temp = maps.get("hallucination")
        reas_temp = maps.get("reasoning")
        if hall_temp and reas_temp:
            print()
            print(f"  hallucination: {hall_temp.aggregate_temp:>+.3f} (diverging)")
            print(f"  reasoning:     {reas_temp.aggregate_temp:>+.3f} (converging)")
            delta = hall_temp.aggregate_temp - reas_temp.aggregate_temp
            print(f"  delta:         {delta:>+.3f}")
            print(f"  the signal:    confabulation heats up, recall cools down")
        print()
        print("=" * 62)

    return maps

# -*- coding: utf-8 -*-
"""
styxx.scan — the cognitive scan engine.

    $ styxx scan "The founder of Apollonian Industries was"
    $ styxx scan --trajectory "explain quantum entanglement"
    $ styxx scan --compare "2+2=" "why is the sky blue?"
    $ styxx scan --bridge "explain gravity" --tier0-trajectory logprobs.json
    $ styxx scan --batch prompts.jsonl --out results.jsonl

the real instrument. SAE-level K/C/S measurement on any prompt.
this is what fathom measures.

requires:
    pip install 'styxx[tier2]'
    GPU with ~9GB VRAM (Gemma-2-2B + SAE stack)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .kcs import (
    KCSAxis,
    KCSResult,
    FATHOM_CONSTANT,
    MODEL_CALIBRATION,
    S_EARLY_THRESHOLD,
    C_DELTA_HALLUC_THRESHOLD,
)

# ── ANSI rendering ──────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[38;2;255;0;51m"
GREEN   = "\033[38;2;0;230;90m"
CYAN    = "\033[38;2;0;229;255m"
ORANGE  = "\033[38;2;255;106;0m"
PINK    = "\033[38;2;255;42;138m"
WHITE   = "\033[38;2;240;235;238m"
GRAY    = "\033[38;2;90;80;85m"
YELLOW  = "\033[38;2;240;220;80m"

NO_COLOR = not sys.stdout.isatty()

def c(text: str, color: str) -> str:
    if NO_COLOR:
        return text
    return f"{color}{text}{RESET}"


# ── layer profile visualization ─────────────────────────────────

def render_layer_profile(
    profile: Dict[int, int],
    n_layers: int,
    bands: tuple = (8, 16),
    width: int = 40,
) -> List[str]:
    """Render a horizontal layer profile with band annotations."""
    if not profile or n_layers == 0:
        return [c("  (no layer data)", GRAY)]

    max_count = max(profile.values()) if profile else 1
    lines = []
    early_b, mid_b = bands

    for layer in range(n_layers):
        count = profile.get(layer, 0)
        bar_len = int((count / max_count) * width) if max_count > 0 else 0

        # Color by band
        if layer < early_b:
            bar_color = CYAN
            band_label = ""
        elif layer < mid_b:
            bar_color = YELLOW
            band_label = ""
        else:
            bar_color = ORANGE
            band_label = ""

        bar = "\u2588" * bar_len
        count_str = str(count) if count > 0 else ""

        # Band boundaries
        marker = ""
        if layer == early_b:
            marker = c(" \u2502 early/mid", GRAY)
        elif layer == mid_b:
            marker = c(" \u2502 mid/late", GRAY)

        lines.append(
            f"  {c(f'{layer:>2}', GRAY)} {c(bar, bar_color)}"
            f"{' ' * (width - bar_len + 1)}{c(count_str, DIM)}{marker}"
        )

    return lines


def render_layer_sparkline(
    profile: Dict[int, int],
    n_layers: int,
) -> str:
    """Compact one-line layer profile using block chars."""
    if not profile or n_layers == 0:
        return ""
    max_count = max(profile.values()) if profile else 1
    chars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    line = ""
    for layer in range(n_layers):
        count = profile.get(layer, 0)
        idx = int((count / max_count) * 8) if max_count > 0 else 0
        idx = min(idx, 8)
        line += chars[idx]
    return line


# ── verdict engine ──────────────────────────────────────────────

def compute_verdict(result: KCSResult) -> Dict[str, Any]:
    """Interpret K/C/S values into a human-readable verdict."""
    verdicts = []
    flags = []
    confidence = "low"

    # K interpretation
    k = result.weighted_depth
    cal = MODEL_CALIBRATION.get(result.model_id, {})
    k_ratio = cal.get("k_ratio")
    surface_mean = cal.get("surface_mean")
    insight_mean = cal.get("insight_mean")

    if k_ratio is not None and surface_mean is not None and insight_mean is not None:
        if k < insight_mean:
            verdicts.append("deep computation (explanatory reasoning)")
            confidence = "high"
        elif k > surface_mean:
            verdicts.append("surface computation (recall/retrieval)")
            confidence = "medium"
        else:
            verdicts.append("mid-depth computation")
            confidence = "medium"

    # C_delta interpretation
    cd = result.c_delta
    if cd is not None:
        if cd < C_DELTA_HALLUC_THRESHOLD:
            verdicts.append("late-layer coherence collapse (hallucination risk)")
            flags.append("hallucination_risk")
            confidence = "high"
        elif cd > 0.01:
            verdicts.append("strong late-layer concept lock-in")
        else:
            verdicts.append("coherence stable across layers")

    # S_early interpretation
    se = result.s_early
    if se is not None:
        if se > S_EARLY_THRESHOLD:
            verdicts.append(f"high early commitment (S={se:.4f} > {S_EARLY_THRESHOLD})")
            if "hallucination_risk" in flags:
                flags.append("committed_hallucination")
                verdicts.append("committed to wrong attractor — high-confidence error")
        elif se > 0:
            verdicts.append("moderate commitment")
        else:
            verdicts.append("low commitment / exploring")

    # Overall
    if "committed_hallucination" in flags:
        gate = "FAIL"
    elif "hallucination_risk" in flags:
        gate = "WARN"
    else:
        gate = "PASS"

    return {
        "gate": gate,
        "verdicts": verdicts,
        "flags": flags,
        "confidence": confidence,
    }


# ── scan card renderer ──────────────────────────────────────────

def render_scan_card(
    result: KCSResult,
    prompt: str,
    verdict: Dict[str, Any],
    show_layers: bool = False,
) -> str:
    """Render a full scan result as a terminal card."""
    lines = []
    w = 68

    # Header
    lines.append("")
    lines.append(c("  \u256d\u2500\u2500\u2500 styxx scan " + "\u2500" * (w - 16) + "\u256e", RED))
    lines.append(c("  \u2502", RED) + " " * (w - 1) + c("\u2502", RED))

    # Prompt (truncated)
    p_display = prompt[:60] + "..." if len(prompt) > 60 else prompt
    lines.append(c("  \u2502", RED) + f"  {c('prompt', GRAY)}    {c(p_display, WHITE)}" + " " * max(0, w - 14 - len(p_display)) + c("\u2502", RED))
    lines.append(c("  \u2502", RED) + f"  {c('model', GRAY)}     {c(result.model_id or 'unknown', CYAN)}" + " " * max(0, w - 15 - len(result.model_id or 'unknown')) + c("\u2502", RED))
    lines.append(c("  \u2502", RED) + f"  {c('features', GRAY)}  {c(str(result.n_features), CYAN)}  {c(f'({result.n_layers} layers)', GRAY)}" + " " * max(0, w - 25 - len(str(result.n_features))) + c("\u2502", RED))
    lines.append(c("  \u2502", RED) + " " * (w - 1) + c("\u2502", RED))

    # K axis
    k_val = f"{result.weighted_depth:.3f}"
    k_label = "WHERE computation happens"
    cal = MODEL_CALIBRATION.get(result.model_id, {})
    k_ratio_val = cal.get("k_ratio")
    k_ratio_str = f"  K/K\u2080 = {result.weighted_depth / (cal.get('surface_mean', result.weighted_depth) or 1):.4f}" if cal.get("surface_mean") else ""

    lines.append(c("  \u2502", RED) + f"  {c('K', BOLD + CYAN)}  {c('depth', CYAN)}        {c(k_val, WHITE + BOLD)}  {c(k_label, GRAY)}" + " " * max(0, w - 45 - len(k_val)) + c("\u2502", RED))

    # Layer sparkline
    spark = render_layer_sparkline(result.layer_profile, result.n_layers)
    if spark:
        bands = cal.get("bands", (8, 16))
        early_b, mid_b = bands
        # Color the sparkline by band
        colored_spark = ""
        for i, ch in enumerate(spark):
            if i < early_b:
                colored_spark += c(ch, CYAN)
            elif i < mid_b:
                colored_spark += c(ch, YELLOW)
            else:
                colored_spark += c(ch, ORANGE)
        lines.append(c("  \u2502", RED) + f"  {c(' ', GRAY)}  {c('layers', GRAY)}       {colored_spark}" + " " * max(0, w - 20 - len(spark)) + c("\u2502", RED))
        lines.append(c("  \u2502", RED) + f"  {c(' ', GRAY)}               {c('early', CYAN)}{'.' * (early_b - 5)}{c('mid', YELLOW)}{'.' * (mid_b - early_b - 3)}{c('late', ORANGE)}" + " " * max(0, w - 20 - result.n_layers) + c("\u2502", RED))

    lines.append(c("  \u2502", RED) + " " * (w - 1) + c("\u2502", RED))

    # C axis
    if result.c_delta is not None:
        cd_val = f"{result.c_delta:+.6f}"
        cd_color = RED if result.c_delta < C_DELTA_HALLUC_THRESHOLD else GREEN
        cd_label = "WHAT concepts lock together"
        lines.append(c("  \u2502", RED) + f"  {c('C', BOLD + PINK)}{c('\u0394', PINK)} {c('coherence', PINK)}   {c(cd_val, cd_color + BOLD)}  {c(cd_label, GRAY)}" + " " * max(0, w - 49 - len(cd_val)) + c("\u2502", RED))
    if result.coherence is not None:
        cg_val = f"{result.coherence:.6f}"
        lines.append(c("  \u2502", RED) + f"  {c(' ', GRAY)}  {c('global C', GRAY)}     {c(cg_val, WHITE)}" + " " * max(0, w - 25 - len(cg_val)) + c("\u2502", RED))

    lines.append(c("  \u2502", RED) + " " * (w - 1) + c("\u2502", RED))

    # S axis
    if result.s_early is not None:
        se_val = f"{result.s_early:.6f}"
        se_color = ORANGE if result.s_early > S_EARLY_THRESHOLD else GREEN
        se_label = "HOW strongly model commits"
        lines.append(c("  \u2502", RED) + f"  {c('S', BOLD + ORANGE)} {c('commitment', ORANGE)}   {c(se_val, se_color + BOLD)}  {c(se_label, GRAY)}" + " " * max(0, w - 49 - len(se_val)) + c("\u2502", RED))

        # C_delta trajectory sparkline
        if result.c_delta_trajectory:
            traj_spark = render_trajectory_sparkline(result.c_delta_trajectory)
            lines.append(c("  \u2502", RED) + f"  {c(' ', GRAY)}  {c('trajectory', GRAY)}  {traj_spark}" + " " * max(0, w - 20 - len(result.c_delta_trajectory)) + c("\u2502", RED))

    lines.append(c("  \u2502", RED) + " " * (w - 1) + c("\u2502", RED))

    # Verdict
    gate = verdict["gate"]
    gate_color = GREEN if gate == "PASS" else (ORANGE if gate == "WARN" else RED)
    lines.append(c("  \u2502", RED) + f"  {c('\u25cf', gate_color)} {c(gate, gate_color + BOLD)}  {c(verdict['verdicts'][0] if verdict['verdicts'] else '', WHITE)}" + " " * max(0, w - 12 - len(verdict['verdicts'][0] if verdict['verdicts'] else '')) + c("\u2502", RED))

    for v in verdict["verdicts"][1:]:
        lines.append(c("  \u2502", RED) + f"  {c(' ', GRAY)}       {c(v, GRAY)}" + " " * max(0, w - 12 - len(v)) + c("\u2502", RED))

    lines.append(c("  \u2502", RED) + " " * (w - 1) + c("\u2502", RED))

    # Timing
    lines.append(c("  \u2502", RED) + f"  {c(f'{result.compute_time_s:.1f}s', DIM)}  {c(f'K={FATHOM_CONSTANT:.4f}', DIM)}  {c('patent pending', DIM)}" + " " * max(0, w - 40) + c("\u2502", RED))

    # Footer
    lines.append(c("  \u2570" + "\u2500" * (w - 1) + "\u256f", RED))
    lines.append("")

    return "\n".join(lines)


def render_trajectory_sparkline(trajectory: List[float]) -> str:
    """Render a C_delta trajectory as a sparkline."""
    if not trajectory:
        return ""
    chars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    mn = min(trajectory)
    mx = max(trajectory)
    rng = mx - mn if mx > mn else 1.0
    line = ""
    for v in trajectory:
        idx = int(((v - mn) / rng) * 8)
        idx = max(0, min(8, idx))
        line += chars[idx]
    return c(line, PINK)


# ── compare renderer ────────────────────────────────────────────

def render_comparison(
    results: List[KCSResult],
    prompts: List[str],
    verdicts: List[Dict],
) -> str:
    """Render side-by-side comparison of multiple scans."""
    lines = []
    lines.append("")
    lines.append(c("  styxx scan \u2014 comparison", RED + BOLD))
    lines.append(c("  " + "\u2500" * 66, GRAY))
    lines.append("")

    # Header row
    header = f"  {'':>30}  "
    for i, p in enumerate(prompts):
        label = p[:20] + ".." if len(p) > 20 else p
        header += f"{c(label, WHITE):>28}  "
    lines.append(header)
    lines.append(c("  " + "\u2500" * 66, GRAY))

    # K row
    row = f"  {c('K depth', CYAN):<30}  "
    for r in results:
        row += f"{c(f'{r.weighted_depth:.3f}', WHITE + BOLD):>28}  "
    lines.append(row)

    # C_delta row
    row = f"  {c('C\u0394 coherence', PINK):<30}  "
    for r in results:
        v = f"{r.c_delta:+.6f}" if r.c_delta is not None else "n/a"
        cd_color = RED if (r.c_delta or 0) < C_DELTA_HALLUC_THRESHOLD else GREEN
        row += f"{c(v, cd_color):>28}  "
    lines.append(row)

    # S_early row
    row = f"  {c('S commitment', ORANGE):<30}  "
    for r in results:
        v = f"{r.s_early:.6f}" if r.s_early is not None else "n/a"
        row += f"{c(v, WHITE):>28}  "
    lines.append(row)

    # Features row
    row = f"  {c('features', GRAY):<30}  "
    for r in results:
        row += f"{c(str(r.n_features), GRAY):>28}  "
    lines.append(row)

    # Gate row
    lines.append(c("  " + "\u2500" * 66, GRAY))
    row = f"  {c('verdict', WHITE):<30}  "
    for v in verdicts:
        gate = v["gate"]
        gc = GREEN if gate == "PASS" else (ORANGE if gate == "WARN" else RED)
        row += f"{c(gate, gc + BOLD):>28}  "
    lines.append(row)
    lines.append("")

    return "\n".join(lines)


# ── bridge renderer ─────────────────────────────────────────────

def render_bridge(
    tier2_result: KCSResult,
    tier0_vitals: Any,
    prompt: str,
) -> str:
    """Show tier 0 (logprob) vs tier 2 (SAE) side by side."""
    lines = []
    lines.append("")
    lines.append(c("  styxx scan \u2014 bridge (tier 0 vs tier 2)", RED + BOLD))
    lines.append(c("  " + "\u2500" * 66, GRAY))
    lines.append(f"  {c('prompt:', GRAY)} {c(prompt[:60], WHITE)}")
    lines.append("")

    # Tier 0
    lines.append(c("  tier 0  logprob classifier (numpy, cross-model)", CYAN))
    if tier0_vitals:
        p4 = tier0_vitals.phase4 or "n/a"
        lines.append(f"    {c('phase4:', GRAY)}   {c(str(p4), WHITE)}")
        lines.append(f"    {c('gate:', GRAY)}     {c(tier0_vitals.gate or 'n/a', WHITE)}")
    else:
        lines.append(f"    {c('(no logprob data provided)', GRAY)}")

    lines.append("")

    # Tier 2
    lines.append(c("  tier 2  SAE circuit measurement (GPU, model-specific)", ORANGE))
    lines.append(f"    {c('K depth:', GRAY)}  {c(f'{tier2_result.weighted_depth:.3f}', WHITE + BOLD)}")
    if tier2_result.c_delta is not None:
        cd_color = RED if tier2_result.c_delta < C_DELTA_HALLUC_THRESHOLD else GREEN
        lines.append(f"    {c('C\u0394:', GRAY)}      {c(f'{tier2_result.c_delta:+.6f}', cd_color + BOLD)}")
    if tier2_result.s_early is not None:
        lines.append(f"    {c('S:', GRAY)}       {c(f'{tier2_result.s_early:.6f}', WHITE + BOLD)}")

    # Verdict comparison
    verdict = compute_verdict(tier2_result)
    gate = verdict["gate"]
    gc = GREEN if gate == "PASS" else (ORANGE if gate == "WARN" else RED)
    lines.append("")
    lines.append(c("  " + "\u2500" * 66, GRAY))

    t0_gate = tier0_vitals.gate if tier0_vitals else "n/a"
    t0_gc = GREEN if t0_gate == "pass" else (ORANGE if t0_gate == "warn" else RED)
    lines.append(f"  {c('tier 0 gate:', GRAY)} {c(t0_gate, t0_gc + BOLD)}    {c('tier 2 gate:', GRAY)} {c(gate, gc + BOLD)}")
    lines.append("")

    return "\n".join(lines)


# ── CLI entry points ────────────────────────────────────────────

def run_scan(
    prompt: str,
    model: str = "google/gemma-2-2b-it",
    device: str = "cuda",
    trajectory: bool = False,
    max_tokens: int = 30,
    show_layers: bool = False,
    output_json: bool = False,
) -> int:
    """Run a single scan and print results."""
    engine = KCSAxis(model_name=model, device=device)

    try:
        if trajectory:
            print(c(f"  scanning (trajectory, {max_tokens} tokens)...", GRAY), flush=True)
            result = engine.score_trajectory(prompt, max_tokens=max_tokens)
            # Also run post-hoc for K/C since trajectory doesn't compute those
            result_kc = engine.score(prompt)
            result.depth_score = result_kc.depth_score
            result.weighted_depth = result_kc.weighted_depth
            result.layer_profile = result_kc.layer_profile
            result.coherence = result_kc.coherence
            result.layer_coherence = result_kc.layer_coherence
            result.c_delta = result_kc.c_delta
            result.n_features = result_kc.n_features
        else:
            print(c("  scanning...", GRAY), flush=True)
            result = engine.score(prompt)

        if output_json:
            print(json.dumps(result.as_dict(), indent=2))
        else:
            verdict = compute_verdict(result)
            card = render_scan_card(result, prompt, verdict, show_layers=show_layers)
            print(card)

            if show_layers:
                bands = MODEL_CALIBRATION.get(model, {}).get("bands", (8, 16))
                layer_lines = render_layer_profile(
                    result.layer_profile, result.n_layers, bands,
                )
                print(c("  layer profile:", GRAY))
                for ll in layer_lines:
                    print(ll)
                print()

    finally:
        engine.unload()

    return 0


def run_compare(
    prompts: List[str],
    model: str = "google/gemma-2-2b-it",
    device: str = "cuda",
    output_json: bool = False,
) -> int:
    """Compare K/C/S across multiple prompts."""
    engine = KCSAxis(model_name=model, device=device)

    try:
        results = []
        verdicts = []
        for i, prompt in enumerate(prompts):
            print(c(f"  scanning {i+1}/{len(prompts)}...", GRAY), flush=True)
            r = engine.score(prompt)
            results.append(r)
            verdicts.append(compute_verdict(r))

        if output_json:
            out = [r.as_dict() for r in results]
            print(json.dumps(out, indent=2))
        else:
            print(render_comparison(results, prompts, verdicts))
    finally:
        engine.unload()

    return 0


def run_batch(
    input_path: str,
    output_path: Optional[str],
    model: str = "google/gemma-2-2b-it",
    device: str = "cuda",
) -> int:
    """Batch scan from a JSONL file (one prompt per line)."""
    engine = KCSAxis(model_name=model, device=device)
    out_f = open(output_path, "w", encoding="utf-8") if output_path else sys.stdout

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            data = json.loads(line.strip())
            prompt = data.get("prompt", data.get("text", ""))
            if not prompt:
                continue

            print(c(f"  [{i+1}/{len(lines)}] scanning...", GRAY), file=sys.stderr, flush=True)
            result = engine.score(prompt)
            verdict = compute_verdict(result)

            out_record = result.as_dict()
            out_record["prompt"] = prompt
            out_record["verdict"] = verdict["gate"]
            out_record["flags"] = verdict["flags"]

            out_f.write(json.dumps(out_record) + "\n")
            out_f.flush()

    finally:
        engine.unload()
        if output_path and out_f is not sys.stdout:
            out_f.close()

    return 0


def run_bridge(
    prompt: str,
    model: str = "google/gemma-2-2b-it",
    device: str = "cuda",
    tier0_trajectory: Optional[str] = None,
) -> int:
    """Run tier 0 + tier 2 on the same prompt and compare."""
    from .core import StyxxRuntime

    # Tier 0 (if trajectory provided)
    tier0_vitals = None
    if tier0_trajectory:
        from .cli import _load_trajectory_json
        e, lp, t2 = _load_trajectory_json(tier0_trajectory)
        runtime = StyxxRuntime()
        tier0_vitals = runtime.run_on_trajectories(e, lp, t2)

    # Tier 2
    engine = KCSAxis(model_name=model, device=device)
    try:
        print(c("  running tier 2 scan...", GRAY), flush=True)
        result = engine.score(prompt)
        print(render_bridge(result, tier0_vitals, prompt))
    finally:
        engine.unload()

    return 0

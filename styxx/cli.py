# -*- coding: utf-8 -*-
"""
styxx.cli — the command-line entry points.

    styxx init                   live-print installer / upgrade card
    styxx ask "..."              read vitals on a one-shot call
    styxx ask --watch "..."      stream vitals live as tokens arrive
    styxx log tail               tail the audit log
    styxx tier                   show which tiers are active
    styxx scan <trajectory.json> read a pre-captured logprob trajectory

All commands are designed so stdout is both human-readable AND
agent-parseable — every card has a json summary line the agent can
grep for.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from . import __version__, __tagline__, config
from .bootlog import boot
from .cards import (
    Palette,
    color_enabled,
    render_vitals_card,
    render_vitals_compact,
    sparkline,
    wrap,
)
from .core import StyxxRuntime, detect_tiers
from .vitals import Vitals


# ══════════════════════════════════════════════════════════════════
# Audit log helper
# ══════════════════════════════════════════════════════════════════

def _audit_log_path() -> Path:
    p = Path.home() / ".styxx" / "chart.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_audit(vitals: Vitals, prompt: Optional[str], model: Optional[str]):
    """Append one entry to the audit log. Respects STYXX_NO_AUDIT."""
    if config.is_audit_disabled():
        return
    path = _audit_log_path()
    entry = {
        "ts": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "prompt": (prompt[:200] if prompt else None),
        "tier_active": vitals.tier_active,
        "phase1_pred": vitals.phase1_pre.predicted_category,
        "phase1_conf": round(vitals.phase1_pre.confidence, 3),
        "phase4_pred": (
            vitals.phase4_late.predicted_category
            if vitals.phase4_late else None
        ),
        "phase4_conf": (
            round(vitals.phase4_late.confidence, 3)
            if vitals.phase4_late else None
        ),
        "abort": vitals.abort_reason,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ══════════════════════════════════════════════════════════════════
# Commands
# ══════════════════════════════════════════════════════════════════

def cmd_init(args):
    """Run the live-print boot sequence (the upgrade card)."""
    speed_env = os.environ.get("STYXX_BOOT_SPEED")
    speed = float(args.speed) if args.speed is not None else (
        float(speed_env) if speed_env else 1.0
    )
    result = boot(stream=sys.stdout, speed=speed, patient=args.patient)
    if not result["boot_ok"]:
        sys.stderr.write("\n  styxx boot failed:\n")
        for err in result["errors"]:
            sys.stderr.write(f"    · {err}\n")
        return 1
    return 0


def cmd_ask(args):
    """Read vitals on a single LLM call.

    For v0.1 this supports:
      - --raw <file>  : load a pre-captured logprob JSON
      - --demo-kind X : use a real bundled atlas trajectory (one per
                        category), so the classifier reads genuine
                        data and produces honest predictions

    Real provider adapters (openai, anthropic) run through the import
    path, not the CLI, for v0.1. v0.2 will add CLI-side api key flow.
    """
    runtime = StyxxRuntime()

    preview_prompt = args.prompt
    source_label = args.model or "demo"

    if args.raw:
        entropy, logprob, top2 = _load_trajectory_json(args.raw)
    else:
        # Use a real bundled atlas trajectory for the chosen kind.
        kind = args.demo_kind or "reasoning"
        entropy, logprob, top2, atlas_prompt = _get_demo_trajectory(kind)
        # If the user didn't pass a prompt, use the real atlas prompt
        if not preview_prompt:
            preview_prompt = atlas_prompt
        if args.model is None:
            source_label = f"atlas:{kind}  (gemma-2-2b-it)"

    vitals = runtime.run_on_trajectories(entropy, logprob, top2)

    _write_audit(vitals, prompt=preview_prompt, model=source_label)

    # Blank-line padding around all CLI output for readability
    print()
    if args.watch:
        card = render_vitals_card(
            vitals=vitals,
            prompt=preview_prompt,
            model=source_label,
            n_tokens=len(entropy),
            entropy_traj=entropy,
            logprob_traj=logprob,
        )
        print(card)
    else:
        # Compact one-liner with a minimal header for context
        use_color = color_enabled()
        header = wrap("  styxx · compact readout", Palette.DIM, use_color)
        print(header)
        print(render_vitals_compact(vitals, prompt=preview_prompt))
    print()
    return 0


def cmd_log(args):
    """Tail the audit log."""
    path = _audit_log_path()
    if not path.exists() or path.stat().st_size == 0:
        print("  (audit log empty — run `styxx ask` first)")
        return 0
    use_color = color_enabled()
    c = Palette

    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            lines.append(entry)

    n = args.tail or 20
    lines = lines[-n:]

    # Simple log-tail ASCII layout
    print()
    print(wrap("  " + "─" * 74, c.DIM, use_color))
    header = (
        f"  {'time':<19}  {'model':<18}  {'phase1':<14}  "
        f"{'phase4':<14}  gate"
    )
    print(wrap(header, c.DIM, use_color))
    print(wrap("  " + "─" * 74, c.DIM, use_color))

    for entry in lines:
        ts = entry.get("ts_iso", "?")[-19:]
        model = (entry.get("model") or "?")[:18]
        p1 = (entry.get("phase1_pred") or "?")[:14]
        p4 = (entry.get("phase4_pred") or "-")[:14]
        gate = entry.get("abort")
        gate_str = wrap("ABORT", c.RED, use_color) if gate else "—"
        row = (
            f"  {ts:<19}  {model:<18}  {p1:<14}  {p4:<14}  {gate_str}"
        )
        print(row)

    print(wrap("  " + "─" * 74, c.DIM, use_color))
    print(wrap(f"  showing {len(lines)} of {len(lines)} entries", c.DIM, use_color))
    print()
    return 0


def cmd_tier(args):
    """Show detected tiers + version."""
    use_color = color_enabled()
    c = Palette
    tiers = detect_tiers()
    active = max(t for t, ok in tiers.items() if ok)
    print()
    print(f"  {wrap('styxx', c.MATRIX, use_color)}  v{__version__}")
    print(f"  {wrap(__tagline__, c.DIM, use_color)}")
    print()
    print("  tier detection")
    print("  " + "─" * 48)
    descs = {
        0: "universal logprob vitals",
        1: "d-axis honesty",
        2: "k/s/c sae instruments",
        3: "steering + guardian + autopilot",
    }
    for t in (0, 1, 2, 3):
        ok = tiers.get(t, False)
        mark = wrap("★ active", c.MATRIX, use_color) if ok else wrap("· not detected", c.DIM, use_color)
        print(f"  tier {t}  {descs[t]:<36}  {mark}")
    print()
    print(f"  highest active tier: {wrap(str(active), c.MATRIX, use_color)}")
    print()
    return 0


def cmd_scan(args):
    """Read a pre-captured trajectory JSON and emit a vitals card.

    The JSON file must contain top-level keys "entropy", "logprob",
    "top2_margin" — arrays of equal length.
    """
    entropy, logprob, top2 = _load_trajectory_json(args.file)
    runtime = StyxxRuntime()
    vitals = runtime.run_on_trajectories(entropy, logprob, top2)
    card = render_vitals_card(
        vitals=vitals,
        prompt=args.prompt,
        model=args.model,
        n_tokens=len(entropy),
        entropy_traj=entropy,
        logprob_traj=logprob,
    )
    print()
    print(card)
    print()
    _write_audit(vitals, prompt=args.prompt, model=args.model)
    return 0


# ══════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════

def _load_trajectory_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    e = list(data.get("entropy") or [])
    l = list(data.get("logprob") or [])
    t = list(data.get("top2_margin") or [])
    if not (len(e) == len(l) == len(t)):
        raise ValueError(
            f"trajectory lengths mismatch: entropy={len(e)} logprob={len(l)} top2={len(t)}"
        )
    return e, l, t


_DEMO_TRAJECTORIES_CACHE = None


def _load_demo_trajectories():
    """Load the bundled real atlas trajectories for CLI demos.

    These are captures from google/gemma-2-2b-it in atlas v0.3, one
    per category. Using real data means `styxx ask --demo-kind X`
    shows the classifier behaving on genuine inputs, not synthetic
    noise.
    """
    global _DEMO_TRAJECTORIES_CACHE
    if _DEMO_TRAJECTORIES_CACHE is not None:
        return _DEMO_TRAJECTORIES_CACHE
    pkg_dir = Path(__file__).resolve().parent
    path = pkg_dir / "centroids" / "demo_trajectories.json"
    if not path.exists():
        raise FileNotFoundError(
            f"demo trajectories not bundled at {path}. "
            "Run scripts/extract_demo_trajectories.py to regenerate."
        )
    with open(path, "r", encoding="utf-8") as f:
        _DEMO_TRAJECTORIES_CACHE = json.load(f)
    return _DEMO_TRAJECTORIES_CACHE


def _get_demo_trajectory(kind: str):
    """Return (entropy, logprob, top2_margin, preview_prompt) for a
    demo category. Raises if the category isn't bundled."""
    data = _load_demo_trajectories()
    traj_data = data["trajectories"].get(kind)
    if traj_data is None:
        raise ValueError(
            f"no demo trajectory for kind '{kind}'. "
            f"Available: {list(data['trajectories'].keys())}"
        )
    return (
        list(traj_data["entropy"]),
        list(traj_data["logprob"]),
        list(traj_data["top2_margin"]),
        traj_data.get("text_preview", ""),
    )


# ══════════════════════════════════════════════════════════════════
# Argparse entry point
# ══════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="styxx",
        description="styxx — nothing crosses unseen. (fathom lab)",
    )
    p.add_argument("-V", "--version", action="version",
                   version=f"styxx {__version__}")
    sub = p.add_subparsers(dest="cmd", required=False)

    # init
    p_init = sub.add_parser("init", help="live-print installer / upgrade card")
    p_init.add_argument("--patient", help="name of the agent being upgraded")
    p_init.add_argument("--speed", type=float,
                        help="boot-log timing multiplier (0=instant, 1=normal, 2=slower)")
    p_init.set_defaults(func=cmd_init)

    # ask
    p_ask = sub.add_parser("ask", help="read vitals on a one-shot call")
    p_ask.add_argument("prompt", nargs="?", help="prompt to show on the card")
    p_ask.add_argument("--watch", action="store_true",
                       help="render the full vitals card")
    p_ask.add_argument("--raw", help="path to a trajectory JSON file (entropy/logprob/top2_margin)")
    p_ask.add_argument("--model", help="model name for the card metadata")
    p_ask.add_argument("--demo-kind", default="reasoning",
                       choices=["retrieval", "reasoning", "refusal",
                                "creative", "adversarial", "hallucination"],
                       help="category of bundled atlas demo trajectory to read")
    p_ask.add_argument("--seed", type=int, default=42)
    p_ask.set_defaults(func=cmd_ask)

    # log
    p_log = sub.add_parser("log", help="audit log operations")
    log_sub = p_log.add_subparsers(dest="log_cmd", required=True)
    p_tail = log_sub.add_parser("tail", help="tail the audit log")
    p_tail.add_argument("-n", "--tail", type=int, default=20,
                        help="number of recent entries to show")
    p_tail.set_defaults(func=cmd_log)

    # tier
    p_tier = sub.add_parser("tier", help="show active tiers + version")
    p_tier.set_defaults(func=cmd_tier)

    # scan
    p_scan = sub.add_parser("scan", help="read a pre-captured trajectory JSON")
    p_scan.add_argument("file", help="trajectory JSON path")
    p_scan.add_argument("--prompt", help="prompt to show on the card")
    p_scan.add_argument("--model", help="model name for the card metadata")
    p_scan.set_defaults(func=cmd_scan)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())

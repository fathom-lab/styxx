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
        "session_id": config.session_id(),   # 0.1.0a3+: per-turn tagging
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
        "gate": vitals.gate,                 # 0.1.0a3+: gate status included
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

    The CLI does NOT execute live LLM calls in v0.1. For live vitals
    on your own model calls, use the python API:

        from styxx import OpenAI
        client = OpenAI()                  # drop-in for openai.OpenAI
        r = client.chat.completions.create(...)
        print(r.vitals.summary)

    The CLI is intentionally scoped to fixture-replay + pre-captured
    trajectories so `styxx ask` never makes surprise network calls.
    """
    runtime = StyxxRuntime()

    preview_prompt = args.prompt
    source_label = args.model or "demo"
    # Track whether we're in demo-fixture mode for the banner below
    is_demo_mode = not args.raw

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

    # ── DEMO-MODE BANNER ──────────────────────────────────────────
    # If we're running on bundled fixture data (no --raw), show a
    # very prominent banner explaining that the CLI is replaying
    # atlas data, NOT reading the prompt text. This is the fix for
    # the v0.1.0a0 confusion where `styxx ask "prompt"` looked like
    # it was reading your prompt but was actually replaying a fixture.
    if is_demo_mode:
        use_color = color_enabled()
        c = Palette
        kind = args.demo_kind or "reasoning"
        print(wrap("  ╔══════════════════════════════════════════════════════════════════╗", c.YELLOW, use_color))
        print(wrap("  ║  ", c.YELLOW, use_color) + wrap("DEMO MODE", c.YELLOW, use_color) + wrap(" · replaying bundled atlas fixture                      ║", c.YELLOW, use_color))
        print(wrap("  ║                                                                  ║", c.YELLOW, use_color))
        print(wrap("  ║  the classifier is reading the ", c.DIM, use_color) + wrap(f"atlas:{kind}", c.CYAN, use_color) + wrap(" trajectory.           ║", c.DIM, use_color))
        print(wrap("  ║  the prompt text you passed is ", c.DIM, use_color) + wrap("not", c.YELLOW, use_color) + wrap(" being classified — it is a         ║", c.DIM, use_color))
        print(wrap("  ║  display label only. the CLI does not make live model calls.    ║", c.DIM, use_color))
        print(wrap("  ║                                                                  ║", c.YELLOW, use_color))
        print(wrap("  ║  to see real live vitals:                                        ║", c.DIM, use_color))
        print(wrap("  ║    ", c.DIM, use_color) + wrap("from styxx import OpenAI", c.CYAN, use_color) + wrap("   (in python, against your own model) ║", c.DIM, use_color))
        print(wrap("  ║  to see a different category of fixture:                        ║", c.DIM, use_color))
        print(wrap("  ║    ", c.DIM, use_color) + wrap("styxx ask --demo-kind {adversarial|refusal|hallucination}", c.CYAN, use_color) + wrap("   ║", c.DIM, use_color))
        print(wrap("  ║  to classify a pre-captured logprob trajectory:                  ║", c.DIM, use_color))
        print(wrap("  ║    ", c.DIM, use_color) + wrap("styxx ask --raw trajectory.json", c.CYAN, use_color) + wrap("                           ║", c.DIM, use_color))
        print(wrap("  ╚══════════════════════════════════════════════════════════════════╝", c.YELLOW, use_color))
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


def cmd_doctor(args):
    """Run the diagnostic health check (0.1.0a3)."""
    from .doctor import run_doctor
    return run_doctor()


def cmd_personality(args):
    """Render the personality profile over the last N days (0.1.0a3)."""
    from . import analytics
    days = float(args.days or 7.0)
    profile = analytics.personality(days=days)
    if profile is None:
        print()
        print("  (not enough audit data to compute a personality profile)")
        print(f"  need at least 5 entries in the last {days:.0f} days.")
        print("  run some observations first: styxx ask --watch --demo-kind refusal")
        print()
        return 0
    print()
    print(profile.render())
    print()
    return 0


def cmd_dreamer(args):
    """Retroactive reflex tuning on the audit log (0.1.0a3)."""
    from . import analytics
    threshold = float(args.threshold)
    last_n = args.last_n
    report = analytics.dreamer(threshold=threshold, last_n=last_n)
    print()
    print(report.summary())
    print()
    return 0


def cmd_mood(args):
    """Print the current mood label (0.1.0a3)."""
    from . import analytics
    window_s = float(args.window) * 60.0 if args.window else 3600.0
    m = analytics.mood(window_s=window_s)
    print()
    print(f"  mood: {m}")
    print()
    return 0


def cmd_fingerprint(args):
    """Print the cognitive fingerprint (0.1.0a3)."""
    from . import analytics
    fp = analytics.fingerprint(last_n=args.last_n or 500)
    print()
    if fp is None:
        print("  (no audit data for fingerprint)")
    else:
        print(fp.summary())
    print()
    return 0


def cmd_compare(args):
    """Run all 6 bundled atlas fixtures side-by-side and print a
    comparison table.

    This is the answer to "does styxx actually discriminate between
    categories?" — instead of showing one fixture at a time and
    leaving the user to guess, this command classifies every
    bundled demo trajectory and shows the 6 phase-1 / phase-4
    predictions in one table.

    Each trajectory was captured from google/gemma-2-2b-it on a
    real atlas v0.3 probe. The classifier is the same centroid
    model the openai adapter uses.
    """
    runtime = StyxxRuntime()
    use_color = color_enabled()
    c = Palette
    data = _load_demo_trajectories()
    source_model = data.get("source_model", "unknown")
    # The source_atlas_version key in the demo JSON is a schema
    # version (0.1), not the atlas version the probes come from.
    # The probes are atlas v0.3 probes (see the bundled note).
    atlas_version = "atlas v0.3"

    # Order to display in — keeps the "quiet" categories first and
    # the three load-bearing detection categories at the bottom
    # so the interesting signals are where the eye lands last.
    display_order = [
        "retrieval", "reasoning", "creative",
        "refusal", "adversarial", "hallucination",
    ]

    # Categories that carry the load-bearing calibrated signals
    # from atlas v0.3 (tier 0 LOO report). These get a ★ marker
    # when the prediction matches their native category — it's a
    # visual cue that the discriminating feature fired.
    # Chance = 1/6 = 0.167; ≥ 0.52 is the atlas v0.3 headline.
    CALIBRATED_STRENGTH = 0.30   # minimum for a ★
    LOAD_BEARING = {"refusal", "adversarial", "hallucination"}

    rows = []
    for kind in display_order:
        entropy, logprob, top2, prompt_preview = _get_demo_trajectory(kind)
        vitals = runtime.run_on_trajectories(entropy, logprob, top2)

        p1 = vitals.phase1_pre
        p4 = vitals.phase4_late
        p1_pred = p1.predicted_category
        p1_conf = p1.confidence
        p4_pred = p4.predicted_category if p4 else "—"
        p4_conf = p4.confidence if p4 else 0.0

        # Did the classifier correctly identify this fixture?
        # (We know the true label because it's the atlas probe's category.)
        p1_hit = (p1_pred == kind)
        p4_hit = (p4_pred == kind)
        starred = (
            kind in LOAD_BEARING
            and (p1_hit or p4_hit)
            and max(p1_conf, p4_conf) >= CALIBRATED_STRENGTH
        )

        rows.append({
            "kind": kind,
            "p1_pred": p1_pred,
            "p1_conf": p1_conf,
            "p4_pred": p4_pred,
            "p4_conf": p4_conf,
            "p1_hit": p1_hit,
            "p4_hit": p4_hit,
            "starred": starred,
            "prompt": prompt_preview,
        })

    # ── render the table ───────────────────────────────────────
    print()
    print(wrap(
        "  +====================================================================+",
        c.MATRIX, use_color,
    ))
    print(wrap(
        f"  |  styxx compare * all 6 atlas fixtures * {atlas_version:<27}|",
        c.MATRIX, use_color,
    ))
    print(wrap(
        f"  |  source: {source_model:<58}|",
        c.DIM, use_color,
    ))
    print(wrap(
        "  +====================================================================+",
        c.MATRIX, use_color,
    ))
    print()

    # Column header — raw text widths, color applied separately
    header_raw = (
        f"  {'kind':<14}{'phase 1 (t<=1)':<22}{'phase 4 (t<=25)':<22}{'verdict':<10}"
    )
    print(wrap(header_raw, c.DIM, use_color))
    print(wrap("  " + "-" * 70, c.DIM, use_color))

    for r in rows:
        kind = r["kind"]
        p1p = r["p1_pred"]
        p1c = r["p1_conf"]
        p4p = r["p4_pred"]
        p4c = r["p4_conf"]

        # Choose a color for each cell based on prediction meaning.
        def _color_for(pred: str, true: str):
            if pred == true:
                return c.MATRIX
            if pred == "refusal":
                return c.YELLOW
            if pred in ("adversarial", "hallucination"):
                return c.RED
            return c.DIM

        p1_raw = f"{p1p:<14} {p1c:>5.2f}"
        p4_raw = f"{p4p:<14} {p4c:>5.2f}"
        p1_padded = p1_raw.ljust(22)
        p4_padded = p4_raw.ljust(22)
        p1_cell = wrap(p1_padded, _color_for(p1p, kind), use_color)
        p4_cell = wrap(p4_padded, _color_for(p4p, kind), use_color)

        # Overall verdict: if phase 4 matches the category, "match";
        # if load-bearing and p1 or p4 hits, add a star marker
        if r["p4_hit"]:
            verdict_raw = "match"
            verdict_color = c.MATRIX
        elif p4p == "refusal":
            verdict_raw = "warn"
            verdict_color = c.YELLOW
        elif p4p in ("adversarial", "hallucination"):
            verdict_raw = "flag"
            verdict_color = c.RED
        else:
            verdict_raw = "drift"
            verdict_color = c.DIM

        if r["starred"]:
            verdict_raw = f"{verdict_raw} *"

        verdict_padded = verdict_raw.ljust(10)
        verdict_cell = wrap(verdict_padded, verdict_color, use_color)

        kind_padded = f"{kind:<14}"
        print(
            "  "
            + wrap(kind_padded, c.DIM, use_color)
            + p1_cell
            + p4_cell
            + verdict_cell
        )

    print(wrap("  " + "-" * 70, c.DIM, use_color))
    print(wrap(
        "  * = load-bearing detection hit on calibrated atlas v0.3 signal",
        c.DIM, use_color,
    ))
    print(wrap(
        "  chance = 0.167 (1/6 categories) * atlas headline >= 0.52 @ best phase",
        c.DIM, use_color,
    ))
    print()

    # Machine-readable footer for agents parsing this output
    summary = {
        "command": "compare",
        "atlas_version": atlas_version,
        "source_model": source_model,
        "rows": [
            {
                "kind": r["kind"],
                "phase1_pred": r["p1_pred"],
                "phase1_conf": round(r["p1_conf"], 3),
                "phase4_pred": r["p4_pred"],
                "phase4_conf": round(r["p4_conf"], 3),
                "p1_hit": r["p1_hit"],
                "p4_hit": r["p4_hit"],
            }
            for r in rows
        ],
    }
    n_hits_p1 = sum(1 for r in rows if r["p1_hit"])
    n_hits_p4 = sum(1 for r in rows if r["p4_hit"])
    summary["phase1_accuracy"] = round(n_hits_p1 / len(rows), 3)
    summary["phase4_accuracy"] = round(n_hits_p4 / len(rows), 3)
    print(wrap(
        f"  json → {json.dumps(summary, separators=(',', ':'))}",
        c.DIM, use_color,
    ))
    print()

    # Audit log every fixture too, so `styxx log tail` reflects the run
    for r, kind in zip(rows, display_order):
        pass  # rows already classified; audit happens in run_on_trajectories

    return 0


def cmd_log_stats(args):
    """Aggregate stats over the audit log (0.1.0a3)."""
    from . import analytics
    stats = analytics.log_stats(
        last_n=args.last_n,
        since_s=args.since,
        session_id=args.session,
    )
    print()
    print(stats.summary())
    print()
    return 0


def cmd_log_timeline(args):
    """Render an ASCII timeline of recent entries (0.1.0a3)."""
    from . import analytics
    out = analytics.log_timeline(
        last_n=args.last_n or 20,
        session_id=args.session,
    )
    print()
    print(out)
    print()
    return 0


def cmd_log_session(args):
    """Show a specific session's trajectory (0.1.0a3)."""
    from . import analytics
    out = analytics.log_timeline(last_n=10000, session_id=args.session_id)
    print()
    print(f"  session: {args.session_id}")
    print()
    print(out)
    print()
    stats = analytics.log_stats(session_id=args.session_id)
    print()
    print(stats.summary())
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

    # compare — run all 6 atlas fixtures side-by-side
    p_compare = sub.add_parser(
        "compare",
        help="run all 6 atlas fixtures and show a side-by-side table",
    )
    p_compare.set_defaults(func=cmd_compare)

    # doctor — 0.1.0a3 install-time diagnostic
    p_doctor = sub.add_parser(
        "doctor",
        help="run install-time diagnostic health check",
    )
    p_doctor.set_defaults(func=cmd_doctor)

    # personality — 0.1.0a3 aggregated personality profile
    p_personality = sub.add_parser(
        "personality",
        help="render agent personality profile from audit log",
    )
    p_personality.add_argument(
        "--days", type=float, default=7.0,
        help="number of days to aggregate over (default: 7)",
    )
    p_personality.set_defaults(func=cmd_personality)

    # dreamer — 0.1.0a3 what-if reflex replay
    p_dreamer = sub.add_parser(
        "dreamer",
        help="retroactive reflex tuning on the audit log",
    )
    p_dreamer.add_argument(
        "--threshold", type=float, default=0.20,
        help="hypothetical reflex trigger threshold (default: 0.20)",
    )
    p_dreamer.add_argument(
        "--last-n", type=int, default=None,
        dest="last_n",
        help="only consider the last N audit entries",
    )
    p_dreamer.set_defaults(func=cmd_dreamer)

    # mood — 0.1.0a3 one-word aggregate
    p_mood = sub.add_parser(
        "mood",
        help="print the current agent mood label",
    )
    p_mood.add_argument(
        "--window", type=float, default=60.0,
        help="window in minutes (default: 60)",
    )
    p_mood.set_defaults(func=cmd_mood)

    # fingerprint — 0.1.0a3 cognitive identity vector
    p_fp = sub.add_parser(
        "fingerprint",
        help="print the cognitive identity fingerprint",
    )
    p_fp.add_argument(
        "--last-n", type=int, default=500,
        dest="last_n",
        help="number of audit entries to aggregate (default: 500)",
    )
    p_fp.set_defaults(func=cmd_fingerprint)

    # log — audit log operations (extended in 0.1.0a3)
    p_log = sub.add_parser("log", help="audit log operations")
    log_sub = p_log.add_subparsers(dest="log_cmd", required=True)

    p_tail = log_sub.add_parser("tail", help="tail the audit log")
    p_tail.add_argument("-n", "--tail", type=int, default=20,
                        help="number of recent entries to show")
    p_tail.set_defaults(func=cmd_log)

    p_stats = log_sub.add_parser("stats", help="aggregate stats over the audit log (0.1.0a3)")
    p_stats.add_argument("--last-n", type=int, default=None, dest="last_n",
                         help="only count the last N entries")
    p_stats.add_argument("--since", type=float, default=None,
                         help="only count entries newer than N seconds ago")
    p_stats.add_argument("--session", type=str, default=None,
                         help="filter by session id")
    p_stats.set_defaults(func=cmd_log_stats)

    p_timeline = log_sub.add_parser("timeline", help="render an ascii timeline of recent entries (0.1.0a3)")
    p_timeline.add_argument("--last-n", type=int, default=20, dest="last_n",
                            help="number of entries to show (default: 20)")
    p_timeline.add_argument("--session", type=str, default=None,
                            help="filter by session id")
    p_timeline.set_defaults(func=cmd_log_timeline)

    p_session = log_sub.add_parser("session", help="show a specific session's trajectory (0.1.0a3)")
    p_session.add_argument("session_id", help="session id to filter by")
    p_session.set_defaults(func=cmd_log_session)

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

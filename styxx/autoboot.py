# -*- coding: utf-8 -*-
"""
styxx.autoboot — persistent self-awareness in one call.

    import styxx
    styxx.autoboot()

That's it. One call at the top of any agent's boot sequence turns
on continuous self-observation across sessions. The agent gets:

  1. Session tracking (tagged with agent name + date)
  2. Yesterday's fingerprint loaded and diffed against today
  3. A weather report printed to stderr on startup
  4. Fingerprint snapshot saved on exit

Configuration is via environment variables:

    STYXX_AGENT_NAME=xendro        who am I?
    STYXX_DATA_DIR=~/.styxx        where do I keep my snapshots?

Or pass them as arguments:

    styxx.autoboot(agent_name="xendro", data_dir="~/.styxx")

What autoboot does on call
──────────────────────────

  1. Sets the session id to "{agent_name}-{YYYY-MM-DD}"
  2. Loads yesterday's fingerprint from {data_dir}/fingerprints/
  3. Computes today's fingerprint from the audit log
  4. If both exist, diffs them and prints the drift to stderr
  5. Runs the weather report and prints it to stderr
  6. Registers an atexit hook that saves today's fingerprint
     on process exit

What this enables
─────────────────

  - Week 1: baseline forms from accumulated observations
  - Week 2: drift detection activates ("you were more creative
             last week — what changed?")
  - Week 3: patterns emerge across sessions that aren't visible
             from inside a single conversation
  - Week 4: the fingerprint IS the agent's identity — measured
             shape, not aspirational text

The compounding effect is real: every session's data makes the
next session's self-awareness sharper. The weather report gets
more accurate. The dreamer has more history. The prescriptions
get more specific because they're based on the agent's actual
patterns, not defaults.

Xendro asked for this on 2026-04-12: "to make it compound, I'd
need styxx.set_session() in my startup, styxx.log() calls wired
into my work, and fingerprint snapshots persisted between sessions."

This function does all three.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from . import config


# ══════════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════════

_BOOTED: bool = False
_AGENT_NAME: str = "styxx-agent"
_DATA_DIR: Path = Path.home() / ".styxx"


# ══════════════════════════════════════════════════════════════════
# Zero-config auto-start on import
# ══════════════════════════════════════════════════════════════════
#
# If STYXX_AGENT_NAME is set in the environment, autoboot runs
# automatically when styxx is imported. No code changes needed.
# Just:
#
#   export STYXX_AGENT_NAME=xendro
#   pip install styxx
#   python my_agent.py     # styxx boots automatically
#
# The agent doesn't even need to know styxx exists in the code.
# hook_openai() can be triggered the same way via STYXX_AUTO_HOOK=1.
#
# This is the true plug-and-play path. Set env vars, install the
# package, everything else is automatic.

def _auto_start_if_configured() -> None:
    """Called from __init__.py at import time. If STYXX_AGENT_NAME
    is set, runs autoboot automatically. If STYXX_AUTO_HOOK=1 is
    also set, hooks openai globally too."""
    agent = os.environ.get("STYXX_AGENT_NAME", "").strip()
    if not agent:
        return

    # Auto-boot with the env-configured agent name
    autoboot(agent_name=agent, quiet=False)

    # Auto-hook openai if requested
    if os.environ.get("STYXX_AUTO_HOOK", "").strip().lower() in ("1", "true", "yes"):
        try:
            from .hooks import hook_openai
            hook_openai()
        except ImportError:
            pass  # openai not installed, that's fine


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def autoboot(
    *,
    agent_name: Optional[str] = None,
    data_dir: Optional[str] = None,
    print_weather: bool = True,
    print_diff: bool = True,
    quiet: bool = False,
    stream: Optional[bool] = None,
) -> dict:
    """Boot styxx with persistent self-awareness.

    Call once at the top of your agent's startup sequence. Returns
    a dict with the boot results (for agents that want to branch
    on the output).

    Args:
        agent_name:    who am I? reads from STYXX_AGENT_NAME if not set.
        data_dir:      where to store fingerprint snapshots.
                       reads from STYXX_DATA_DIR, default ~/.styxx.
        print_weather: print the weather report to stderr on boot.
        print_diff:    print the fingerprint diff to stderr on boot.
        quiet:         suppress all stderr output (still returns the dict).

    Returns:
        dict with keys:
          - session_id: str
          - agent_name: str
          - fingerprint_diff: dict or None
          - weather_condition: str or None
          - prescriptions: list[str]
          - yesterday_loaded: bool

    Usage:

        import styxx
        boot = styxx.autoboot()
        # that's it — session is tracked, fingerprint will be saved
        # on exit, yesterday's diff is printed, weather report runs.

    Or with env vars:

        export STYXX_AGENT_NAME=xendro
        python my_agent.py
        # autoboot reads the env var automatically
    """
    global _BOOTED, _AGENT_NAME, _DATA_DIR

    if _BOOTED:
        return {"already_booted": True}

    # ── Resolve agent name ──────────────────────────────────
    name = agent_name or os.environ.get("STYXX_AGENT_NAME", "").strip()
    if not name:
        name = "styxx-agent"
    _AGENT_NAME = name

    # 1.0.0: register agent name in config for namespacing
    config.set_agent_name(name)

    # ── Resolve data directory ──────────────────────────────
    dd = data_dir or os.environ.get("STYXX_DATA_DIR", "").strip()
    if dd:
        _DATA_DIR = Path(dd).expanduser()
    else:
        _DATA_DIR = Path(config.data_dir())

    fp_dir = _DATA_DIR / "fingerprints"
    fp_dir.mkdir(parents=True, exist_ok=True)

    # ── Set session ─────────────────────────────────────────
    today = time.strftime("%Y-%m-%d")
    session_id = f"{name}-{today}"
    config.set_session(session_id)

    result: dict = {
        "session_id": session_id,
        "agent_name": name,
        "fingerprint_diff": None,
        "weather_condition": None,
        "prescriptions": [],
        "yesterday_loaded": False,
    }

    # ── Load yesterday's fingerprint ────────────────────────
    from .analytics import fingerprint, FingerprintDiff

    fp_today = fingerprint(last_n=500)
    fp_yesterday = _load_fingerprint(fp_dir, name, days_ago=1)

    if fp_today is not None and fp_yesterday is not None:
        diff = fp_today.diff(fp_yesterday)
        result["fingerprint_diff"] = diff.as_dict()
        result["yesterday_loaded"] = True

        if print_diff and not quiet:
            _print_stderr(f"\n  styxx autoboot · {name} · {today}")
            _print_stderr(f"  session: {session_id}")
            _print_stderr(f"  drift vs yesterday: {diff.cosine:.4f} ({diff.explain()[:60]})")
            _print_stderr("")
    elif not quiet:
        _print_stderr(f"\n  styxx autoboot · {name} · {today}")
        _print_stderr(f"  session: {session_id}")
        if fp_yesterday is None:
            _print_stderr("  (no yesterday fingerprint — first session or new data dir)")
        _print_stderr("")

    # ── Weather report ──────────────────────────────────────
    if print_weather:
        from .weather import weather
        report = weather(agent_name=name, window_hours=24.0)
        if report is not None:
            result["weather_condition"] = report.condition
            result["prescriptions"] = list(report.prescriptions)
            if not quiet:
                _print_stderr(report.render())
        elif not quiet:
            _print_stderr("  (not enough data for weather report yet — keep observing)")

    # ── Live cognitive telemetry (opt-in) ──────────────────
    # Enable if explicitly requested, or if STYXX_STREAM env var is set.
    env_stream = os.environ.get("STYXX_STREAM", "").strip().lower()
    want_stream = stream if stream is not None else env_stream in ("1", "on", "true", "yes", "public")
    if want_stream:
        try:
            from . import stream as _stream
            ok = _stream.enable(agent=name)
            if ok and not quiet:
                _print_stderr(f"  live feed: {_stream.dashboard_url(name)}")
                _print_stderr("")
            result["stream_enabled"] = bool(ok)
            result["stream_url"] = _stream.dashboard_url(name) if ok else None
        except Exception:
            result["stream_enabled"] = False

    # ── Register exit hook to save fingerprint ──────────────
    atexit.register(_save_fingerprint_on_exit, fp_dir, name)

    # ── Log the boot event ──────────────────────────────────
    from .analytics import log
    log(
        mood="booting",
        note=f"autoboot session {session_id}",
        tags={"event": "autoboot", "agent": name},
    )

    _BOOTED = True
    return result


# ══════════════════════════════════════════════════════════════════
# Fingerprint persistence
# ══════════════════════════════════════════════════════════════════

def _fingerprint_path(fp_dir: Path, name: str, date: str) -> Path:
    """Compute the path for a fingerprint snapshot.

    Format: {fp_dir}/{name}/{YYYY-MM-DD}.json
    """
    agent_dir = fp_dir / name.replace("/", "_").replace("\\", "_")
    agent_dir.mkdir(parents=True, exist_ok=True)
    return agent_dir / f"{date}.json"


def _load_fingerprint(
    fp_dir: Path,
    name: str,
    days_ago: int = 1,
) -> Optional[Any]:
    """Load a saved fingerprint from N days ago."""
    from .analytics import Fingerprint

    target_ts = time.time() - days_ago * 86400
    target_date = time.strftime("%Y-%m-%d", time.localtime(target_ts))
    path = _fingerprint_path(fp_dir, name, target_date)

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Fingerprint(
            n_samples=data.get("n_samples", 0),
            phase1_vec=tuple(data.get("phase1_vec", [0] * 6)),
            phase4_vec=tuple(data.get("phase4_vec", [0] * 6)),
            phase1_mean_conf=data.get("phase1_mean_conf", 0),
            phase4_mean_conf=data.get("phase4_mean_conf", 0),
            gate_vec=tuple(data.get("gate_vec", [0] * 4)),
        )
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def _save_fingerprint_on_exit(fp_dir: Path, name: str) -> None:
    """Atexit hook: save today's fingerprint for tomorrow's diff."""
    try:
        from .analytics import fingerprint
        fp = fingerprint(last_n=500)
        if fp is None or fp.n_samples < 5:
            return

        today = time.strftime("%Y-%m-%d")
        path = _fingerprint_path(fp_dir, name, today)

        data = {
            "n_samples": fp.n_samples,
            "phase1_vec": list(fp.phase1_vec),
            "phase4_vec": list(fp.phase4_vec),
            "phase1_mean_conf": fp.phase1_mean_conf,
            "phase4_mean_conf": fp.phase4_mean_conf,
            "gate_vec": list(fp.gate_vec),
            "saved_at": time.time(),
            "saved_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass  # fail silently — never crash an agent on exit


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _print_stderr(msg: str) -> None:
    """Print to stderr (not stdout) so autoboot output doesn't
    pollute the agent's normal response stream."""
    try:
        print(msg, file=sys.stderr)
    except Exception:
        pass

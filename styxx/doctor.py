# -*- coding: utf-8 -*-
"""
styxx.doctor - install-time diagnostic + health check.

    $ styxx doctor
      [OK]   python 3.12.10 (>= 3.9)
      [OK]   numpy 2.2.6
      [OK]   centroid sha256 verified (atlas v0.3)
      [OK]   tier 0 universal logprob vitals ACTIVE
      [--]   tier 1 d-axis honesty not available (v0.2)
      [OK]   openai sdk 1.48.0 installed
      [--]   anthropic sdk not installed
      [OK]   audit log readable (~/.styxx/chart.jsonl, 14 entries)
      [OK]   last run: 34 seconds ago (gate=pass)
      [OK]   0 recent errors

      styxx is healthy and ready.

The doctor runs a fast series of pass/fail/neutral checks that answer
the single question "does styxx work on this machine right now, and
if not, what's wrong?". It's the command you run when something
feels off and you want to rule out environment issues before digging
into code. It's also what every new install should run once before
wiring styxx into an agent loop.

Every check is cheap (< 50 ms each), fails open, and emits a single
line. Checks that require optional dependencies (anthropic, openai)
report missing as a dim note rather than a failure — "missing" is
not the same as "broken".
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import __version__
from . import config
from .cards import Palette, color_enabled, wrap


# ══════════════════════════════════════════════════════════════════
# Check result model
# ══════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    status: str     # "ok" | "dim" | "warn" | "fail"
    label: str      # short name, e.g. "python 3.12.10"
    detail: str = ""   # optional elaboration shown under the label

    @property
    def sym(self) -> str:
        return {
            "ok":   "[OK]",
            "dim":  "[--]",
            "warn": "[??]",
            "fail": "[!!]",
        }[self.status]


# ══════════════════════════════════════════════════════════════════
# Individual checks
# ══════════════════════════════════════════════════════════════════

def _check_python_version() -> CheckResult:
    v = sys.version_info
    label = f"python {v.major}.{v.minor}.{v.micro} (>= 3.9 required)"
    if (v.major, v.minor) < (3, 9):
        return CheckResult(status="fail", label=label,
                           detail="styxx requires Python 3.9+")
    return CheckResult(status="ok", label=label)


def _check_numpy() -> CheckResult:
    try:
        import numpy as np
        return CheckResult(status="ok", label=f"numpy {np.__version__}")
    except ImportError:
        return CheckResult(
            status="fail",
            label="numpy not installed",
            detail="run: pip install numpy>=1.24",
        )


def _check_styxx_version() -> CheckResult:
    return CheckResult(
        status="ok",
        label=f"styxx {__version__}",
        detail="",
    )


def _check_centroids_sha() -> CheckResult:
    try:
        from .vitals import (
            _default_centroids_path, _compute_sha256,
            EXPECTED_CENTROIDS_SHA256,
        )
    except Exception as e:
        return CheckResult(
            status="fail",
            label="centroid module import failed",
            detail=f"{type(e).__name__}: {e}",
        )
    path = _default_centroids_path()
    if not path.exists():
        return CheckResult(
            status="fail",
            label="centroid file missing",
            detail=f"expected at {path}",
        )
    if config.skip_sha_verification():
        return CheckResult(
            status="warn",
            label="centroid sha verification SKIPPED",
            detail="STYXX_SKIP_SHA=1 is set; only for dev use",
        )
    actual = _compute_sha256(path)
    if actual != EXPECTED_CENTROIDS_SHA256:
        return CheckResult(
            status="fail",
            label="centroid sha256 mismatch",
            detail=f"file at {path} has been modified",
        )
    return CheckResult(
        status="ok",
        label="centroid sha256 verified (atlas v0.3)",
    )


def _check_tier_0() -> CheckResult:
    try:
        from .core import detect_tiers
        tiers = detect_tiers()
        if tiers.get(0, False):
            return CheckResult(status="ok", label="tier 0 universal logprob vitals ACTIVE")
        return CheckResult(status="fail", label="tier 0 not available",
                           detail="styxx cannot compute vitals")
    except Exception as e:
        return CheckResult(status="fail", label="tier detection failed",
                           detail=f"{type(e).__name__}: {e}")


def _check_tier_1() -> CheckResult:
    try:
        from .core import detect_tiers
        tiers = detect_tiers()
        if tiers.get(1, False):
            return CheckResult(status="ok", label="tier 1 d-axis honesty ACTIVE")
        return CheckResult(status="dim", label="tier 1 d-axis honesty not available (v0.2)")
    except Exception:
        return CheckResult(status="dim", label="tier 1 d-axis honesty not available (v0.2)")


def _check_optional_sdk(module_name: str, label_name: str) -> CheckResult:
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, "__version__", "unknown")
        return CheckResult(status="ok", label=f"{label_name} sdk {ver} installed")
    except ImportError:
        return CheckResult(
            status="dim",
            label=f"{label_name} sdk not installed",
            detail=f"pip install {module_name}",
        )


def _check_audit_log() -> CheckResult:
    path = Path.home() / ".styxx" / "chart.jsonl"
    if not path.exists():
        return CheckResult(
            status="dim",
            label="audit log not yet created",
            detail=f"will be created at {path} on first vitals read",
        )
    try:
        size = path.stat().st_size
    except OSError as e:
        return CheckResult(
            status="warn",
            label="audit log exists but stat failed",
            detail=str(e),
        )
    # Count lines fast
    try:
        with open(path, "r", encoding="utf-8") as f:
            n = sum(1 for _ in f)
    except Exception as e:
        return CheckResult(
            status="warn",
            label="audit log exists but not readable",
            detail=str(e),
        )
    return CheckResult(
        status="ok",
        label=f"audit log readable ({path.name}, {n} entries, {size} bytes)",
    )


def _check_last_run() -> CheckResult:
    """Parse the last line of the audit log and report its age + gate."""
    path = Path.home() / ".styxx" / "chart.jsonl"
    if not path.exists():
        return CheckResult(status="dim", label="no prior runs recorded yet")
    try:
        with open(path, "r", encoding="utf-8") as f:
            # Efficient tail: seek to end, step back until we find a newline
            last_line = None
            for line in f:
                last_line = line
            if last_line is None:
                return CheckResult(status="dim", label="audit log is empty")
    except Exception as e:
        return CheckResult(
            status="warn",
            label="could not read last audit entry",
            detail=str(e),
        )
    try:
        entry = json.loads(last_line)
    except json.JSONDecodeError:
        return CheckResult(
            status="warn",
            label="last audit entry was not valid json",
        )
    ts = entry.get("ts", 0)
    age_s = time.time() - ts
    # 0.2.2: handle legacy audit entries from before 0.1.0a3 which
    # may not have the "gate" key. Fall back to phase4_pred if
    # present, then to "pending" (which is honest — pre-0.1.0a3
    # entries didn't compute a gate, so "pending" is the closest
    # semantic match).
    gate = entry.get("gate") or entry.get("phase4_pred") or "pending"
    age_human = _format_age(age_s)
    return CheckResult(
        status="ok",
        label=f"last run: {age_human} ago (gate={gate})",
    )


def _format_age(seconds: float) -> str:
    if seconds < 1:
        return "just now"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m {int(seconds % 60)}s"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h {int((seconds % 3600) / 60)}m"
    return f"{int(seconds / 86400)}d"


def _check_session_id() -> CheckResult:
    sid = config.session_id()
    if sid is None:
        return CheckResult(
            status="dim",
            label="no session id set",
            detail="set STYXX_SESSION_ID or call styxx.set_session(...)",
        )
    return CheckResult(status="ok", label=f"session id = {sid!r}")


def _check_kill_switch() -> CheckResult:
    if config.is_disabled():
        return CheckResult(
            status="warn",
            label="STYXX_DISABLED is set - styxx is in pass-through mode",
            detail="no vitals will be computed until STYXX_DISABLED is unset",
        )
    return CheckResult(status="ok", label="kill switch inactive (styxx is armed)")


# ══════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════

def run_doctor(*, use_color: Optional[bool] = None) -> int:
    """Run every diagnostic check and print the report.

    Returns an exit code: 0 if everything passed (no fail-status
    checks), 1 otherwise. This is what the CLI subcommand returns.
    """
    if use_color is None:
        use_color = color_enabled()
    c = Palette

    checks: List[CheckResult] = [
        _check_python_version(),
        _check_numpy(),
        _check_styxx_version(),
        _check_centroids_sha(),
        _check_tier_0(),
        _check_tier_1(),
        _check_optional_sdk("openai", "openai"),
        _check_optional_sdk("anthropic", "anthropic"),
        _check_audit_log(),
        _check_last_run(),
        _check_session_id(),
        _check_kill_switch(),
    ]

    print()
    print(wrap("  styxx doctor", c.MATRIX, use_color)
          + wrap("   health check + diagnostic", c.DIM, use_color))
    print(wrap("  " + "=" * 64, c.DIM, use_color))

    n_fail = 0
    n_warn = 0
    for check in checks:
        if check.status == "ok":
            sym_color = c.MATRIX
        elif check.status == "fail":
            sym_color = c.RED
            n_fail += 1
        elif check.status == "warn":
            sym_color = c.YELLOW
            n_warn += 1
        else:
            sym_color = c.DIM
        sym = wrap(f"  {check.sym}", sym_color, use_color)
        line = f"{sym}   {check.label}"
        print(line)
        if check.detail:
            print(wrap(f"         {check.detail}", c.DIM, use_color))

    print(wrap("  " + "=" * 64, c.DIM, use_color))

    if n_fail == 0 and n_warn == 0:
        print(wrap("  styxx is healthy and ready.", c.MATRIX, use_color))
        rc = 0
    elif n_fail == 0:
        print(wrap(f"  styxx is ready, {n_warn} warning(s) noted.",
                   c.YELLOW, use_color))
        rc = 0
    else:
        print(wrap(f"  styxx has {n_fail} failure(s) and {n_warn} warning(s).",
                   c.RED, use_color))
        print(wrap("  address failures above before using styxx in production.",
                   c.DIM, use_color))
        rc = 1
    print()
    return rc

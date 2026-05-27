# -*- coding: utf-8 -*-
"""Unit tests for the 7.7.7 styxx leaderboard CLI command."""
from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from styxx.cli import cmd_leaderboard


def _run(rows_only: bool = False) -> str:
    args = SimpleNamespace(rows_only=rows_only)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_leaderboard(args)
    assert rc == 0, f"cmd_leaderboard exited {rc}; expected 0"
    return buf.getvalue()


def test_leaderboard_renders_full_text():
    out = _run(rows_only=False)
    assert "styxx gauntlet — public leaderboard" in out
    assert "Baseline-001" in out
    assert "submit your method" in out.lower() or "submission" in out.lower()


def test_leaderboard_rows_only_filters_to_table():
    out = _run(rows_only=True)
    assert "## Leaderboard" in out or "### Reference baselines" in out
    assert "Baseline-001" in out
    assert "Baseline-002" in out
    assert "Baseline-003" in out
    # Header content should not be in rows-only mode
    assert "submit your method" not in out.lower()


def test_leaderboard_bundled_in_package_data():
    """Regression: LEADERBOARD.md must ship as package data so the CLI works on pip install."""
    pkg_data = Path(__file__).resolve().parent.parent / "styxx" / "_data" / "LEADERBOARD.md"
    assert pkg_data.exists(), (
        "package-data LEADERBOARD.md missing — clean pip install will fail. "
        "run: cp LEADERBOARD.md styxx/_data/ to fix."
    )


def test_leaderboard_cli_invocation_via_module():
    """End-to-end: python -m styxx leaderboard runs successfully."""
    proc = subprocess.run(
        [sys.executable, "-m", "styxx", "leaderboard", "--rows-only"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert "Baseline-001" in proc.stdout

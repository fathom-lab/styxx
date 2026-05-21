# -*- coding: utf-8 -*-
"""
Tests for scripts/self_audit/analyze_session_trajectory.py

Pinned-output behavioural tests (parity tripwire pattern, same as
test_drift_axis_scorer_parity.py): the analyzer's stdout on the
committed darkflobi chart must remain stable across edits.

Covers:
  * backward compat: no --tags-file -> first/last halves view
  * --tags-file: by-kind aggregate + pairwise delta
  * --tags-file length mismatch: WARN, fall through cleanly
  * --tags-file with BOM: utf-8-sig tolerated
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "self_audit" / "analyze_session_trajectory.py"
CHART = REPO / "papers" / "agent-self-audit" / "darkflobi-session-2026-05-21-chart.jsonl"
KINDS = REPO / "papers" / "agent-self-audit" / "darkflobi-session-2026-05-21-chart.kinds.json"


def _run(*args: str) -> str:
    out = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, encoding="utf-8", check=False,
        cwd=str(REPO),
    )
    assert out.returncode == 0, f"stderr: {out.stderr}"
    return out.stdout


def test_backward_compat_no_tags():
    out = _run(str(CHART))
    assert "chronological trajectory:" in out
    assert "first 2 events vs last 3:" in out
    # by-kind aggregate must NOT appear in default mode
    assert "by-kind aggregate" not in out


def test_by_kind_aggregate():
    out = _run(str(CHART), "--tags-file", str(KINDS))
    assert "by-kind aggregate (n_groups=2):" in out
    # exact pinned numbers from the morning audit
    assert "counterfactual            2  0.942" in out
    assert "real                      3  0.577" in out
    assert "pairwise composite deltas:" in out
    assert "counterfactual -> real  delta -0.365" in out
    # by-kind replaces first/last view when tags supplied
    assert "first 2 events vs last 3:" not in out


def test_tags_length_mismatch_warns(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('["a","b"]', encoding="utf-8")
    out = _run(str(CHART), "--tags-file", str(bad))
    assert "WARN:" in out
    assert "tags-file has 2 entries but chart has 5" in out


def test_tags_bom_tolerated(tmp_path):
    # Windows tooling frequently emits utf-8 with BOM; analyzer must cope.
    with_bom = tmp_path / "withbom.json"
    with_bom.write_bytes(b"\xef\xbb\xbf" + json.dumps(["real"] * 3 + ["counterfactual"] * 2).encode("utf-8"))
    out = _run(str(CHART), "--tags-file", str(with_bom))
    assert "by-kind aggregate" in out
    assert "counterfactual" in out and "real" in out

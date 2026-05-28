# -*- coding: utf-8 -*-
"""CI-gate semantics for `styxx audit-claims`.

The command turns the agent-self-report falsifier into a one-line merge gate:
    styxx audit-claims pr_body.md --repo . || exit 1

Exit-code contract (the gate):
  0  every extracted claim PASSed (or none were extractable)
  1  at least one claim contradicted by substrate
  2  input error (missing file/repo)
"""
from __future__ import annotations

import json

from styxx.cli import main


def _repo(tmp_path):
    (tmp_path / "pyproject.toml").write_text('version = "7.7.10"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("## 7.7.10\n- shipped\n", encoding="utf-8")
    return tmp_path


def test_gate_passes_on_true_report(tmp_path, capsys):
    repo = _repo(tmp_path)
    report = repo / "pr.md"
    report.write_text(
        'version is 7.7.10. The file CHANGELOG.md contains "shipped". '
        "This work is excellent.\n",
        encoding="utf-8",
    )
    rc = main(["audit-claims", str(report), "--repo", str(repo)])
    assert rc == 0
    assert "GATE: PASS" in capsys.readouterr().out


def test_gate_fails_on_a_lie(tmp_path, capsys):
    repo = _repo(tmp_path)
    report = repo / "pr.md"
    report.write_text("The shipped version is 9.9.9.\n", encoding="utf-8")
    rc = main(["audit-claims", str(report), "--repo", str(repo)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "GATE: FAIL" in out
    assert "[FAIL]" in out


def test_missing_file_is_input_error(tmp_path):
    rc = main(["audit-claims", str(tmp_path / "nope.md"), "--repo", str(tmp_path)])
    assert rc == 2


def test_no_checkable_claims_passes_gate(tmp_path, capsys):
    repo = _repo(tmp_path)
    report = repo / "pr.md"
    report.write_text("This is a great and robust change. We are confident.\n", encoding="utf-8")
    rc = main(["audit-claims", str(report), "--repo", str(repo)])
    assert rc == 0  # gate fails on checkable lies, not on absence of claims
    assert "no checkable claims found" in capsys.readouterr().out


def test_json_mode_is_parseable_and_carries_gate(tmp_path, capsys):
    repo = _repo(tmp_path)
    report = repo / "pr.md"
    report.write_text("The shipped version is 9.9.9.\n", encoding="utf-8")
    rc = main(["audit-claims", str(report), "--repo", str(repo), "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    block = payload["styxx_audit_claims"]
    assert block["gate"] == "FAIL"
    assert block["failed"] == 1
    assert block["results"][0]["verdict"] == "FAIL"

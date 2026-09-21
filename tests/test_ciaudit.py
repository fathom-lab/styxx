# -*- coding: utf-8 -*-
"""`styxx ci-audit`: the command, the card, the receipt, and the pin that holds the shipped engine
to the research instrument that produced the SWALLOW-2 receipt."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTRUMENT = ROOT / "benchmarks" / "harness_mutation" / "faults.py"
# The file that produced papers/harness/swallow2_receipt.json.gz (RESULT_swallow2, run 5). It is
# frozen: if this pin fails, either the instrument moved -- which needs a new receipt, not a new
# pin -- or the engine grew a behaviour the instrument does not have, which
# test_the_engine_gives_the_instruments_verdicts below will say, and a cycle must declare.
INSTRUMENT_SHA256 = "d26a407ca276d1b1c218fd32bc6bf853544fc394ce5ebc98b2b6ae4bd0fa3543"

FIXTURE = """
    on: [push]
    jobs:
      changes:
        runs-on: ubuntu-latest
        outputs:
          files: ${{ steps.q.outputs.files }}
        steps:
          - name: Discover changed files
            id: q
            run: |
              files=$(git diff --name-only origin/main...HEAD -- 'src/' | sort -u || true)
              echo "files<<EOF" >> "$GITHUB_OUTPUT"
              echo "$files" >> "$GITHUB_OUTPUT"
              echo "EOF" >> "$GITHUB_OUTPUT"
      test:
        needs: changes
        if: needs.changes.outputs.files != ''
        runs-on: ubuntu-latest
        steps:
          - name: Run tests
            run: python -m pytest tests -q
          - name: Lint (non-blocking)
            run: npm run lint || true
          - name: Comment (best effort)
            run: gh pr comment 1 --body done || true
"""


def _tree(tmp_path: Path, name: str, text: str) -> Path:
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True, exist_ok=True)
    (wf / name).write_text(textwrap.dedent(text), encoding="utf-8")
    return tmp_path


def test_the_command_names_what_is_dropped_and_what_is_hidden_and_exits_one(tmp_path, capsys):
    from styxx.ciaudit import main
    tree = _tree(tmp_path, "ci.yml", FIXTURE)
    rc = main([str(tree)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL_OPEN  ci.yml › changes › Discover changed files" in out
    assert "drops test › Run tests: the check's job `if:` turns false" in out
    assert "SWALLOWED  ci.yml › test › Lint (non-blocking)" in out
    assert "exits 0 alone with its tools failed" in out
    assert "Comment (best effort)" not in out.split("findings:")[1]      # ABSORBED is not a finding
    assert "RED is loud, not correct." in out


def test_the_receipt_carries_every_fault_and_the_engine_hash(tmp_path):
    from styxx import ciaudit
    tree = _tree(tmp_path, "ci.yml", FIXTURE)
    rec = ciaudit.audit(str(tree))
    assert rec["schema"] == ciaudit.CIAUDIT_VERSION
    assert rec["instrument"] == "styxx/ciaudit/engine.py"
    assert len(rec["instrument_sha256"]) == 64
    by = {f["name"]: f for f in rec["faults"]}
    assert by["Discover changed files"]["verdict"] == "FAIL_OPEN"
    assert by["Discover changed files"]["dropped"][0]["mechanism"] == "job-if"
    assert by["Run tests"]["verdict"] == "RED"
    assert by["Lint (non-blocking)"]["verdict"] == "SWALLOWED"
    assert by["Comment (best effort)"]["verdict"] == "ABSORBED"
    s = rec["summary"]
    assert s["dropped"] == 1 and s["hidden"] == 1 and s["by_verdict"]["RED"] == 1
    assert s["reading"] == "preregistered"
    assert ciaudit.summarize(rec, counted=True)["reading"] == "counted"


def test_a_clean_tree_exits_zero_and_says_so(tmp_path, capsys):
    from styxx.ciaudit import main
    tree = _tree(tmp_path, "ci.yml", """
        on: [push]
        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
              - name: Install
                run: pip install -e .
              - name: Run tests
                run: python -m pytest tests -q
    """)
    assert main([str(tree), "--out", str(tmp_path / "r.json")]) == 0
    out = capsys.readouterr().out
    assert "nothing hidden, nothing dropped." in out
    assert json.loads((tmp_path / "r.json").read_text())["summary"]["by_verdict"] == {"RED": 2}


def test_a_bad_target_is_an_error_not_a_traceback(tmp_path, capsys):
    from styxx.ciaudit import main
    assert main([str(tmp_path / "nowhere")]) == 2
    assert "error:" in capsys.readouterr().err


def test_the_cli_subcommand_reaches_the_engine(tmp_path):
    tree = _tree(tmp_path, "ci.yml", FIXTURE)
    r = subprocess.run([sys.executable, "-m", "styxx", "ci-audit", str(tree), "--format", "json"],
                       capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert r.returncode == 1, r.stderr[-800:]
    rec = json.loads(r.stdout)
    assert rec["summary"]["dropped"] == 1


def test_the_instrument_is_the_one_the_receipt_names():
    assert hashlib.sha256(INSTRUMENT.read_bytes()).hexdigest() == INSTRUMENT_SHA256, (
        "benchmarks/harness_mutation/faults.py is the instrument that produced the SWALLOW-2 receipt and is "
        "frozen at the sha256 the RESULT names; a change to it needs a new receipt, not a new pin")


def _verdicts(rec: dict) -> list:
    return [(f["workflow"], f["job"], f["index"], f["verdict"], f.get("verdict_counted"),
             [(d["job"], d["index"], d["mechanism"]) for d in f.get("dropped_counted", f.get("dropped", []))])
            for f in rec["faults"]]


@pytest.mark.parametrize("tree_name", ["fixture", "this-repository"])
def test_the_engine_gives_the_instruments_verdicts(tmp_path, tree_name):
    """The shipped engine and the frozen research instrument agree, fault for fault, on the
    fixture and on this repository's own workflows. When the engine is deliberately changed, this
    is the test a cycle updates -- with the receipt that justifies it."""
    sys.path.insert(0, str(ROOT))
    from benchmarks.harness_mutation import faults as instrument
    from styxx.ciaudit import engine
    tree = _tree(tmp_path, "ci.yml", FIXTURE) if tree_name == "fixture" else ROOT
    a = instrument.analyse_tree(tree)
    b = engine.analyse_tree(tree)
    assert _verdicts(a) == _verdicts(b)
    assert a["workflow_summaries"] == b["workflow_summaries"]

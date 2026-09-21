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
# SWALLOW-3's instrument -- the catalogue of checking actions applied on top of faults.py -- frozen
# at the sha256 the SWALLOW-3 receipt names (papers/harness/swallow3_receipt.json.gz). Same rule: a
# change needs a new receipt, not a new pin.
ACTIONS_INSTRUMENT = ROOT / "benchmarks" / "harness_mutation" / "action_checks.py"
ACTIONS_INSTRUMENT_SHA256 = "0e723694d459ca2368799e3fc21a26d06e70fb89ad09bd2b0152466bf2a68f72"
# SWALLOW-4's instrument -- the two verified repairs -- frozen at the sha256 the SWALLOW-4 receipt
# names (papers/harness/swallow4_receipt.json.gz). Same rule.
REPAIR_INSTRUMENT = ROOT / "benchmarks" / "harness_mutation" / "repair.py"
REPAIR_INSTRUMENT_SHA256 = "7b9a1695d316c2ce109495cf60c5a9a1de03bac204e9bf48fdbc2a1ac66012b9"

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


def test_the_instruments_are_the_ones_the_receipts_name():
    assert hashlib.sha256(INSTRUMENT.read_bytes()).hexdigest() == INSTRUMENT_SHA256, (
        "benchmarks/harness_mutation/faults.py is the instrument that produced the SWALLOW-2 receipt and is "
        "frozen at the sha256 the RESULT names; a change to it needs a new receipt, not a new pin")
    assert hashlib.sha256(ACTIONS_INSTRUMENT.read_bytes()).hexdigest() == ACTIONS_INSTRUMENT_SHA256, (
        "benchmarks/harness_mutation/action_checks.py is the instrument that produced the SWALLOW-3 receipt and is "
        "frozen at the sha256 the RESULT names; a change to it needs a new receipt, not a new pin")
    assert hashlib.sha256(REPAIR_INSTRUMENT.read_bytes()).hexdigest() == REPAIR_INSTRUMENT_SHA256, (
        "benchmarks/harness_mutation/repair.py is the instrument that produced the SWALLOW-4 receipt and is "
        "frozen at the sha256 the RESULT names; a change to it needs a new receipt, not a new pin")


def test_the_shipped_catalogue_is_the_frozen_one():
    sys.path.insert(0, str(ROOT))
    from benchmarks.harness_mutation import action_checks as instrument
    from styxx.ciaudit import actions
    assert actions.CATALOGUE == instrument.CATALOGUE
    assert actions.FAMILIES == instrument.FAMILIES
    assert actions.NOT_CHECKS == instrument.NOT_CHECKS
    assert actions.UNREADABLE == instrument.UNREADABLE


def _verdicts(rec: dict) -> list:
    return [(f["workflow"], f["job"], f["index"], f["verdict"], f.get("verdict_counted"), f.get("flavour"), f.get("by_flavour"),
             f.get("checks_in_scope"), f.get("checks_live"), f.get("self_live"),
             [(d["job"], d["index"], d["mechanism"], d.get("action")) for d in f.get("dropped_counted", f.get("dropped", []))])
            for f in rec["faults"]]


def _verdicts3(rec: dict) -> list:
    return [(f["workflow"], f["job"], f["index"], f["verdict"], f.get("verdict_counted"), f.get("verdict_runs_only"),
             f.get("verdict_counted_runs_only"), f.get("flavour"), f.get("flavour_runs_only"), f.get("by_flavour"), f.get("by_flavour_runs_only"),
             f.get("checks_in_scope"), f.get("checks_live"), f.get("action_checks_in_scope"), f.get("action_checks_live"), f.get("self_live"),
             f.get("dropped"), f.get("dropped_counted"), f.get("dropped_actions"))
            for f in rec["faults"]]


ACTION_FIXTURE = """
    on: [push]
    jobs:
      changes:
        runs-on: ubuntu-latest
        outputs:
          docs: ${{ steps.q.outputs.docs }}
        steps:
          - name: Get changed files
            id: q
            run: |
              files=$(git diff --name-only origin/main...HEAD -- 'docs/' | sort -u || true)
              echo "docs=$files" >> "$GITHUB_OUTPUT"
          - name: Link check
            if: steps.q.outputs.docs != ''
            uses: lycheeverse/lychee-action@v2
            with:
              fail: true
      lint:
        needs: changes
        if: needs.changes.outputs.docs != ''
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: pre-commit/action@v3.0.1
      release:
        runs-on: ubuntu-latest
        steps:
          - name: Notify
            run: curl -X POST https://example.com/hook || true
          - name: Retry the tests
            uses: nick-fields/retry@v3
            with:
              command: npm test
          - uses: ./.github/actions/publish
"""


@pytest.mark.parametrize("tree_name", ["fixture", "action-fixture", "this-repository"])
def test_the_engine_gives_the_instruments_verdicts(tmp_path, tree_name):
    """The shipped engine and the frozen research instruments agree, fault for fault: with the
    catalogue off, the engine is SWALLOW-2's instrument; with it on (the default), SWALLOW-3's.
    When the engine is deliberately changed, this is the test a cycle updates -- with the receipt
    that justifies it."""
    sys.path.insert(0, str(ROOT))
    from benchmarks.harness_mutation import action_checks as instrument3
    from benchmarks.harness_mutation import faults as instrument2
    from styxx.ciaudit import engine
    tree = ROOT if tree_name == "this-repository" else _tree(tmp_path, "ci.yml", FIXTURE if tree_name == "fixture" else ACTION_FIXTURE)
    a2, b2 = instrument2.analyse_tree(tree), engine.analyse_tree(tree, actions=False)
    assert _verdicts(a2) == _verdicts(b2)
    for wf, sm in a2["workflow_summaries"].items():
        assert {k: b2["workflow_summaries"][wf][k] for k in sm} == sm
    a3, b3 = instrument3.analyse_tree(tree), engine.analyse_tree(tree)
    assert _verdicts3(a3) == _verdicts3(b3)
    assert a3["workflow_summaries"] == b3["workflow_summaries"]


def test_a_gated_action_check_is_a_finding_and_the_runs_only_reading_is_kept(tmp_path, capsys):
    from styxx import ciaudit
    from styxx.ciaudit import main
    tree = _tree(tmp_path, "ci.yml", ACTION_FIXTURE)
    rc = main([str(tree)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL_OPEN  ci.yml › changes › Get changed files" in out
    assert "drops changes › Link check [lycheeverse/lychee-action, lint]: the check's `if:` turns false" in out
    assert "drops lint › step 1 [pre-commit/action, lint]: the check's job `if:` turns false" in out
    assert "action checks: 3 of 3 reached in the healthy world" in out
    assert "unreadable: 1 steps" in out
    rec = ciaudit.audit(str(tree))
    by = {f["name"]: f for f in rec["faults"]}
    assert by["Get changed files"]["verdict"] == "FAIL_OPEN" and by["Get changed files"]["verdict_runs_only"] == "NO_CHECK"
    assert by["Notify"]["verdict"] == "ABSORBED" and by["Notify"]["verdict_runs_only"] == "NO_CHECK"     # the retried tests still run
    assert rec["summary"]["dropped_action_checks"] == 2 and rec["summary"]["moved_by_the_catalogue"] == 2
    assert rec["summary"]["reading"] == "preregistered"
    # SWALLOW-2's reading, on request
    assert main([str(tree), "--no-actions"]) == 0
    out = capsys.readouterr().out
    assert "nothing hidden, nothing dropped" in out and "run: steps only" in json.dumps(ciaudit.audit(str(tree), actions=False)["summary"])


REPAIR_FIXTURE = """
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
              echo "files=$files" >> "$GITHUB_OUTPUT"
      test:
        needs: changes
        if: needs.changes.outputs.files != ''
        runs-on: ubuntu-latest
        steps:
          - name: Run tests
            run: python -m pytest tests -q
          - name: Lint (non-blocking)
            continue-on-error: true
            run: npm run lint
      lint:
        runs-on: ubuntu-latest
        steps:
          - name: Lint changed python
            run: |
              files=$(git diff --name-only origin/main | grep '\\.py$' || true)
              ruff check $files || true
"""


def _strip_repairs(rec: dict) -> list:
    return [(t["workflow"], t["job"], t["index"], t["verdict"], t["verified_repair"],
             [(c["repair"], c.get("applies"), c.get("loud"), c.get("unchanged"), c.get("diff"), c.get("why"), c.get("lines_changed")) for c in t["candidates"]])
            for t in rec["targets"]]


def test_the_shipped_repairs_are_the_instruments(tmp_path):
    sys.path.insert(0, str(ROOT))
    from benchmarks.harness_mutation import repair as instrument
    from styxx.ciaudit import repair as shipped
    tree = _tree(tmp_path, "ci.yml", REPAIR_FIXTURE)
    a, b = instrument.repair_tree(tree), shipped.repair_tree(tree)
    assert _strip_repairs(a) == _strip_repairs(b)
    assert shipped.REPAIRS == instrument.REPAIRS
    assert shipped.strict_shell("x=$(cmd || true)\nset +e\ncmd2 || :\n") == instrument.strict_shell("x=$(cmd || true)\nset +e\ncmd2 || :\n")


def test_the_repair_flag_prints_verified_diffs_and_says_which_half_failed(tmp_path, capsys):
    from styxx import ciaudit
    from styxx.ciaudit import main
    tree = _tree(tmp_path, "ci.yml", REPAIR_FIXTURE)
    rc = main([str(tree), "--repair"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "repairs (verified: RED under the same fault, and a healthy run unchanged in both flavours):" in out
    assert "ci.yml › changes › Discover changed files — strict-shell, 3 lines" in out
    assert "-          files=$(git diff --name-only origin/main...HEAD -- 'src/' | sort -u || true)" in out
    assert "+          set -eo pipefail" in out
    assert "ci.yml › test › Lint (non-blocking) — no-continue-on-error, 1 line" in out
    assert "-        continue-on-error: true" in out
    assert "ci.yml › lint › Lint changed python — no verified repair: strict-shell is loud but changes the healthy run in flavour empty" in out
    rec = ciaudit.audit(str(tree), repair=True)
    assert rec["repair_catalogue"] == ["no-continue-on-error", "strict-shell", "both"]
    assert rec["summary"]["repairs"] == {"targets": 3, "verified": 2, "by_repair": {"strict-shell": 1, "no-continue-on-error": 1},
                                         "rejected_changes_healthy_run": 1, "not_loud": 0, "no_candidate_applies": 0}
    # the command's path (repair_faults on the audit's findings) and the tree path agree
    from styxx.ciaudit import repair as shipped
    assert _strip_repairs({"targets": rec["repairs"]}) == _strip_repairs(shipped.repair_tree(tree))
    # without the flag the card has no repairs section and the receipt no repairs
    assert main([str(tree)]) == 1 and "repairs (" not in capsys.readouterr().out
    assert "repairs" not in ciaudit.audit(str(tree))


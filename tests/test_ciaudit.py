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
# SWALLOW-5's instrument -- the structural repairs -- frozen at the sha256 the SWALLOW-5 receipt
# names (papers/harness/swallow5_receipt.json.gz, run 2). Same rule.
STRUCTURAL_INSTRUMENT = ROOT / "benchmarks" / "harness_mutation" / "repair_structural.py"
STRUCTURAL_INSTRUMENT_SHA256 = "77067a71fa41e48089b4c3e68fae17f02322fd892fc71d604d83f23cb693982c"
# SWALLOW-6's instrument -- every check followed through the mainline history as a lineage -- frozen
# at the sha256 the SWALLOW-6 prereg and receipt name (papers/harness/swallow6_receipt.json.gz). Same rule.
HISTORY_INSTRUMENT = ROOT / "benchmarks" / "harness_mutation" / "history.py"
HISTORY_INSTRUMENT_SHA256 = "93efb4a947a18e61d4457af12c17bf266ca5680ec89156da6935035211f93d1c"

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
    assert hashlib.sha256(STRUCTURAL_INSTRUMENT.read_bytes()).hexdigest() == STRUCTURAL_INSTRUMENT_SHA256, (
        "benchmarks/harness_mutation/repair_structural.py is the instrument that produced the SWALLOW-5 receipt and is "
        "frozen at the sha256 the RESULT names; a change to it needs a new receipt, not a new pin")
    assert hashlib.sha256(HISTORY_INSTRUMENT.read_bytes()).hexdigest() == HISTORY_INSTRUMENT_SHA256, (
        "benchmarks/harness_mutation/history.py is not the file the SWALLOW-6 receipt names; a new receipt, not a new pin")


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
    assert rec["repair_catalogue"][:3] == ["no-continue-on-error", "strict-shell", "both"]
    assert rec["summary"]["repairs"] == {"targets": 3, "verified": 2, "by_repair": {"strict-shell": 1, "no-continue-on-error": 1},
                                         "rejected_changes_healthy_run": 1, "not_loud": 0, "no_candidate_applies": 0}
    # the command's path (repair_faults on the audit's findings) and the tree path agree on the first stage
    from styxx.ciaudit import repair as shipped
    first = [dict(t, candidates=t["candidates"][:3]) for t in rec["repairs"]]
    assert _strip_repairs({"targets": [dict(t, verified_repair=next((c["repair"] for c in t["candidates"] if c.get("verified")), None)) for t in first]}) == \
        _strip_repairs(shipped.repair_tree(tree))
    # without the flag the card has no repairs section and the receipt no repairs
    assert main([str(tree)]) == 1 and "repairs (" not in capsys.readouterr().out
    assert "repairs" not in ciaudit.audit(str(tree))


STRUCTURAL_FIXTURE = """
    on: [push]
    jobs:
      lint:
        runs-on: ubuntu-latest
        steps:
          - name: Lint for secrets
            run: |
              if grep -rq "SECRET" src/; then
                exit 1
              fi
          - name: Verify branch
            run: |
              CURRENT=$(git rev-parse --abbrev-ref HEAD || echo "unknown")
              echo "on $CURRENT"
          - name: Verify label
            run: |
              KIND=$(gh pr view --json labels | grep -o 'release' || echo none)
              echo "kind is $KIND"
          - name: Test each package
            run: |
              FAIL=0
              for p in a b; do pytest tests/$p || FAIL=$((FAIL+1)); done
              echo "$FAIL packages failed"
"""


def test_the_shipped_structural_repairs_are_the_instruments(tmp_path):
    sys.path.insert(0, str(ROOT))
    from benchmarks.harness_mutation import repair_structural as instrument
    from styxx.ciaudit import repair_structural as shipped
    tree = _tree(tmp_path, "ci.yml", STRUCTURAL_FIXTURE)
    a, b = instrument.structural_tree(tree), shipped.structural_tree(tree)
    assert _strip_repairs(a) == _strip_repairs(b)
    assert _strip_repairs({"targets": a["first_stage"]}) == _strip_repairs({"targets": b["first_stage"]})
    assert shipped.REPAIRS == instrument.REPAIRS
    for run in ('if grep -q x f; then\n  exit 1\nfi\n', 'X=$(a || echo "b (c)")\ncmd || echo d >&2\n'):
        assert shipped.guard_status(run) == instrument.guard_status(run) and shipped.no_default(run) == instrument.no_default(run)


def test_the_repair_flag_has_two_stages(tmp_path, capsys):
    from styxx import ciaudit
    from styxx.ciaudit import main
    tree = _tree(tmp_path, "ci.yml", STRUCTURAL_FIXTURE)
    assert main([str(tree), "--repair"]) == 1
    out = capsys.readouterr().out
    assert "ci.yml › lint › Lint for secrets — guard-status, 4 lines" in out
    assert '+          __rc=0; grep -rq "SECRET" src/ || __rc=$?' in out
    assert "ci.yml › lint › Verify branch — no-default, 3 lines" in out
    assert "ci.yml › lint › Verify label — no verified repair: no-default is loud but changes the healthy run in flavour empty" in out
    assert "ci.yml › lint › Test each package — no verified repair: strict-shell / both: not loud" in out
    rec = ciaudit.audit(str(tree), repair=True)
    assert rec["repair_catalogue"] == ["no-continue-on-error", "strict-shell", "both", "guard-status", "no-default", "both-structural"]
    by = {t["name"]: t for t in rec["repairs"]}
    assert [c["repair"] for c in by["Lint for secrets"]["candidates"]] == rec["repair_catalogue"]      # the second stage ran after the first
    assert by["Lint for secrets"]["verified_repair"] == "guard-status" and by["Verify branch"]["verified_repair"] == "no-default"
    assert [c["repair"] for c in by["Test each package"]["candidates"]] == rec["repair_catalogue"] and by["Test each package"]["verified_repair"] is None
    assert rec["summary"]["repairs"] == {"targets": 4, "verified": 2, "by_repair": {"guard-status": 1, "no-default": 1},
                                         "rejected_changes_healthy_run": 1, "not_loud": 1, "no_candidate_applies": 0}
    # the shipped second stage agrees with the instrument's on the same tree
    from benchmarks.harness_mutation import repair_structural as instrument
    inst = {t["name"]: t for t in instrument.structural_tree(tree)["targets"]}
    for name, t in by.items():
        assert t["verified_repair"] == inst[name]["verified_repair"]
        assert [(c["repair"], c.get("applies"), c.get("verified"), c.get("diff")) for c in t["candidates"][3:]] == \
            [(c["repair"], c.get("applies"), c.get("verified"), c.get("diff")) for c in inst[name]["candidates"]]



def test_the_shipped_history_is_the_instruments(tmp_path):
    """The living copy follows the same lineages, with the same events, as the frozen instrument, on the scripted history."""
    sys.path.insert(0, str(ROOT))
    from benchmarks.harness_mutation import history as instrument
    from styxx.ciaudit import history as shipped
    from tests.test_harness_history import _repo
    tree = _repo(tmp_path)
    strip = lambda rec: {wf: [(l["job"], l["key"], l["born"]["state"], l["state"], l.get("alive"),  # noqa: E731
                               [(e["kind"], e["sha"], tuple(e["mechanism"]), e["acknowledged"], (e.get("agreement") or {}).get("verified_repair")) for e in l["events"]])
                              for l in w["lineages"]] for wf, w in rec["workflows"].items()}
    a, b = instrument.tree_history(tree, "fixture"), shipped.tree_history(tree, "fixture")
    assert strip(a) == strip(b) and a["summary"] == b["summary"]
    assert shipped.ACK.pattern == instrument.ACK.pattern and shipped.MECHANISMS == instrument.MECHANISMS
    assert shipped.CANDIDATE_MECHANISMS == instrument.CANDIDATE_MECHANISMS and shipped.HIDDEN == instrument.HIDDEN
    for step in ({"name": "x", "run": "a"}, {"id": "q", "run": "a"}, {"run": "\n b\n"}):
        assert shipped.step_key(step) == instrument.step_key(step)


def test_the_history_flag_says_since_when_and_by_whose_hand(tmp_path, capsys):
    from styxx import ciaudit
    from styxx.ciaudit import main
    from tests.test_harness_history import _repo
    tree = _repo(tmp_path)
    assert main([str(tree), "--history"]) == 1
    out = capsys.readouterr().out
    assert "SWALLOWED  ci.yml › test › Lint (non-blocking)   [continue-on-error]" in out
    assert "hidden since 2023-03-01 (" in out and ", 366 days): acquired: continue-on-error — \"ci: make lint non-blocking for now (flaky)\" [the commit says: non-blocking]" in out
    assert "history: 5 workflow revisions read on the mainline (first-parent from HEAD)" in out
    rec = ciaudit.audit(str(tree), history=True)
    (h,) = [h for h in rec["history"]["findings"] if h["job"] == "test"]
    assert h["placed"] and not h["born_hidden"] and h["kind"] == "acquisition" and h["mechanism"] == ["continue-on-error"]
    assert h["acknowledged"] == "non-blocking" and h["age_days"] == 366.0 and h["revisions"] == 5 and h["repaired_before"] == 0
    assert rec["history"]["shallow"] is False and rec["history"]["capped"] is False
    # a tree without history is said so, not a traceback
    bare = _tree(tmp_path / "bare", "ci.yml", FIXTURE)
    rec = ciaudit.audit(str(bare), history=True)
    assert rec["history"].get("error", "").startswith("no git history") or all(not f["placed"] for f in rec["history"]["findings"])


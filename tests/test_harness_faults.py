"""The fault-injection instrument (SWALLOW-2) holds its own rules: the expression subset, the
output files, what counts as a check, and one fixture workflow per verdict."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from benchmarks.harness_mutation import faults as F


def _ctx(steps=None, needs=None, env=None, failed=False):
    return F.Context(steps or {}, needs or {}, env or {}, job_failed=failed)


def test_the_expression_subset_knows_what_it_knows_and_says_so_when_it_does_not():
    e = F.Evaluator(_ctx(steps={"a": {"outputs": {"dirs": ""}, "outcome": "success", "conclusion": "success"}}))
    assert e.eval("steps.a.outputs.dirs != ''") is False
    assert e.eval("steps.a.outputs.dirs == ''") is True
    e2 = F.Evaluator(_ctx(steps={"a": {"outputs": {"dirs": "x"}, "outcome": "success", "conclusion": "success"}}))
    assert e2.eval("steps.a.outputs.dirs != ''") is True                # x is an answer: non-empty
    assert e2.eval("steps.a.outputs.dirs == 'true'") is F.UNKNOWN       # but not any particular string
    assert e2.eval("steps.a.outputs.missing != ''") is False          # an output never written is empty
    assert e2.eval("github.event_name == 'push'") is F.UNKNOWN          # not known: neither true nor false
    assert e2.eval("!contains(github.ref, 'release')") is F.UNKNOWN
    e3 = F.Evaluator(_ctx(steps={"a": {"outputs": {"v": "Yes"}, "outcome": "success", "conclusion": "success"}}))
    assert e3.eval("success() && steps.a.outputs.v == 'YES'") is True   # strings compare without case
    assert e2.eval("fromJSON('[]')") == []
    assert e2.eval("fromJSON('')") == ""
    assert e2.eval("always()") is True
    assert F.condition(None, _ctx(failed=True)) is False                # the default condition is success()
    assert F.condition("always()", _ctx(failed=True)) is True
    assert F.condition("${{ github.event_name == 'push' }}", _ctx()) is F.UNKNOWN
    assert F.substitute("run ${{ steps.a.outputs.dirs }} on ${{ matrix.os }}", _ctx(
        steps={"a": {"outputs": {"dirs": "d1"}, "outcome": "success", "conclusion": "success"}})) == "run d1 on x"


def test_the_output_files_are_read_the_way_actions_writes_them(tmp_path):
    p = tmp_path / "out"
    p.write_text("a=1\nfiles<<EOF\nx.py\ny.py\nEOF\nb=two=2\n", encoding="utf-8")
    assert F._parse_kv_file(p) == {"a": "1", "files": "x.py\ny.py", "b": "two=2"}


def test_what_counts_as_a_check_is_a_stated_rule():
    assert F.verification("Run tests", "python -m pytest tests -q") == "tool"
    assert F.verification("Run core tracing tests", "export PYTHONPATH=$(pwd)\ncd lib && pytest tests/tracing") == "tool"
    assert F.verification("Verify each submission", "python scripts/verify_submission.py") == "tool"
    assert F.verification("Type check", "npm run typecheck") == "tool"
    assert F.verification("Install styxx (editable, with test extras)", 'pip install -e ".[test]"') is None
    assert F.verification("Lint (auto-fix)", "npm run lint:fix") is None
    assert F.verification("Check diff", 'CHANGED=$(gh pr view "$PR" --json files)') is None
    assert F.verification("Smoke tests", "./run-smoke.sh") == "name"
    assert F.verification("Generate test report", "python gen_report.py") is None


def _analyse(tmp_path: Path, name: str, text: str) -> list[dict]:
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True, exist_ok=True)
    (wf / name).write_text(textwrap.dedent(text), encoding="utf-8")
    rec = F.analyse_tree(tmp_path)
    return rec["faults"]


DISCOVER_THEN_TEST = """
    on: [pull_request]
    jobs:
      verify:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - name: Discover changed files
            id: discover
            run: |
              files=$(git diff --name-only origin/main...HEAD -- 'src/' | sort -u {TAIL})
              echo "files<<EOF" >> "$GITHUB_OUTPUT"
              echo "$files" >> "$GITHUB_OUTPUT"
              echo "EOF" >> "$GITHUB_OUTPUT"
          - name: Run tests
            if: steps.discover.outputs.files != ''
            run: python -m pytest tests -q
          - name: Nothing changed
            if: steps.discover.outputs.files == ''
            run: echo "nothing to verify"
"""


def test_the_137_shape_is_fail_open_across_steps_and_the_repair_is_red(tmp_path):
    faults = _analyse(tmp_path, "a.yml", DISCOVER_THEN_TEST.replace("{TAIL}", "|| true"))
    by = {f["name"]: f for f in faults}
    assert by["Discover changed files"]["verdict"] == "FAIL_OPEN"
    d = by["Discover changed files"]["dropped"]
    assert [(x["name"], x["mechanism"], x["cross_step"]) for x in d] == [("Run tests", "step-if", True)]
    assert by["Run tests"]["verdict"] == "RED"
    fixed = _analyse(tmp_path / "fixed", "a.yml", DISCOVER_THEN_TEST.replace("{TAIL}", ""))
    assert {f["name"]: f["verdict"] for f in fixed}["Discover changed files"] == "RED"


def test_a_swallowed_check_a_best_effort_step_and_an_in_step_empty_loop(tmp_path):
    faults = _analyse(tmp_path, "b.yml", """
        on: [push]
        jobs:
          j:
            runs-on: ubuntu-latest
            steps:
              - name: Comment (best effort)
                run: gh pr comment 1 --body hi || true
              - name: Run tests
                run: python -m pytest tests || true
              - name: Run tests per service
                run: |
                  set +e
                  trap 'err=1' ERR
                  for s in $(./compose.sh config --services | grep '^svc-' | sort); do
                    pytest tests -k "$s"
                  done
                  test "${err:-0}" = 0
              - name: Run tests (continue-on-error)
                continue-on-error: true
                run: python -m pytest tests/slow
    """)
    by = {f["name"]: f["verdict"] for f in faults}
    assert by["Comment (best effort)"] == "ABSORBED"
    assert by["Run tests"] == "SWALLOWED"
    assert by["Run tests per service"] == "FAIL_OPEN"
    dropped = next(f for f in faults if f["name"] == "Run tests per service")["dropped"]
    assert [(x["mechanism"], x["cross_step"]) for x in dropped] == [("unreached", False)]
    assert by["Run tests (continue-on-error)"] == "SWALLOWED"


def test_fail_open_reaches_across_jobs_by_needs_and_by_an_empty_matrix(tmp_path):
    faults = _analyse(tmp_path, "c.yml", """
        on: [push]
        jobs:
          changes:
            runs-on: ubuntu-latest
            outputs:
              changed: ${{ steps.q.outputs.changed }}
              matrix: ${{ steps.q.outputs.matrix }}
            steps:
              - name: Query
                id: q
                run: |
                  n=$(git diff --name-only HEAD~1 | wc -l || true)
                  if [ -n "$n" ]; then echo "changed=true" >> "$GITHUB_OUTPUT"; echo 'matrix=["a","b"]' >> "$GITHUB_OUTPUT"; fi
          test:
            needs: changes
            if: needs.changes.outputs.changed == 'true'
            runs-on: ubuntu-latest
            steps:
              - name: Run tests
                run: pytest -q
          shard:
            needs: changes
            runs-on: ubuntu-latest
            strategy:
              matrix:
                shard: ${{ fromJSON(needs.changes.outputs.matrix) }}
            steps:
              - name: Run shard tests
                run: pytest -q --shard ${{ matrix.shard }}
    """)
    q = next(f for f in faults if f["name"] == "Query")
    assert q["verdict"] == "FAIL_OPEN"
    assert sorted((x["job"], x["mechanism"]) for x in q["dropped"]) == [("shard", "job-empty-matrix"), ("test", "job-if")]
    assert {f["name"]: f["verdict"] for f in faults}["Run tests"] == "RED"


def test_the_instrument_is_deterministic_and_records_what_it_could_not_read(tmp_path):
    a = _analyse(tmp_path, "d.yml", DISCOVER_THEN_TEST.replace("{TAIL}", "|| true"))
    b = _analyse(tmp_path, "d.yml", DISCOVER_THEN_TEST.replace("{TAIL}", "|| true"))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    faults = _analyse(tmp_path / "e", "e.yml", """
        on: [push]
        jobs:
          j:
            runs-on: ubuntu-latest
            steps:
              - name: Guard
                run: test "$(git rev-parse HEAD)" = "expected"
              - name: Run tests
                run: pytest -q
    """)
    by = {f["name"]: f["verdict"] for f in faults}
    assert by["Guard"] == "BASELINE_RED"        # in both healthy flavours the guard fails on its own logic: uninterpretable
    assert by["Run tests"] == "RED"             # the healthy world carries on past that artifact, so the check after it is read
    assert next(f for f in faults if f["name"] == "Guard")["by_flavour"] == {"x": "BASELINE_RED", "empty": "BASELINE_RED"}


def test_a_clean_tree_check_is_read_in_the_empty_flavour_and_a_discover_step_in_the_x_flavour(tmp_path):
    faults = _analyse(tmp_path, "f.yml", """
        on: [push]
        jobs:
          j:
            runs-on: ubuntu-latest
            steps:
              - name: Regenerate
                run: python gen.py
              - name: Tree is clean
                run: |
                  if [ -n "$(git status --porcelain)" ]; then echo "stale"; exit 1; fi
              - name: Run tests
                run: pytest -q
    """)
    by = {f["name"]: f for f in faults}
    # the x flavour cannot read this job at all (a non-empty `x` looks like a dirty tree); in the
    # empty flavour the fault in the git query is ABSORBED: git fails, the answer is "", the tree
    # is reported clean, and pytest runs as before -- the check itself is not a verification step
    assert by["Tree is clean"]["by_flavour"] == {"x": "BASELINE_RED", "empty": "ABSORBED"}
    assert by["Tree is clean"]["verdict"] == "ABSORBED" and by["Tree is clean"]["flavour"] == "empty"
    assert by["Run tests"]["by_flavour"] == {"x": "RED", "empty": "RED"}
    discover = _analyse(tmp_path / "g", "g.yml", DISCOVER_THEN_TEST.replace("{TAIL}", "|| true"))
    d = next(f for f in discover if f["name"] == "Discover changed files")
    assert d["by_flavour"] == {"x": "FAIL_OPEN", "empty": "NO_CHECK"} and d["flavour"] == "x"

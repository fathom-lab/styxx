# -*- coding: utf-8 -*-
"""SWALLOW-5's instrument: two edits to a script's logic -- a guard whose failing command is not
its green path, a query without its default -- for the checks a strict shell cannot make loud,
verified on SWALLOW-4's two halves. The tests hold the text transformations, every outcome on one
fixture, the two-stage order, and determinism."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import repair_structural as rs  # noqa: E402

FIXTURE = """
    on: [push]
    jobs:
      lint:
        runs-on: ubuntu-latest
        steps:
          - name: Lint for secrets
            run: |
              if grep -rq "SECRET" src/; then
                echo "secret found"
                exit 1
              fi
          - name: Lint for todos
            run: |
              if ! grep -rq "TODO" src/
              then
                echo "no todos"
              fi
          - name: Verify branch
            run: |
              CURRENT=$(git rev-parse --abbrev-ref HEAD || echo "unknown")
              echo "on $CURRENT"
              if [ "$CURRENT" = "main" ]; then echo ok; fi
          - name: Verify build log
            run: |
              npm run build 2>&1 | tee build.log \\
                || echo "build failed (non-blocking)"
              echo done
          - name: Run golden tests (non-blocking)
            run: |
              pnpm --filter shared run test golden \\
                || echo "golden tests failed (non-blocking)"
          - name: Lint with a loud guard
            run: |
              if grep -rq "FIXME" src/; then
                exit 1
              fi
              test -z "$(git status --porcelain)"
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


def _tree(tmp_path: Path, text: str = FIXTURE) -> Path:
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True, exist_ok=True)
    (wf / "ci.yml").write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return tmp_path


def test_guard_status_is_the_stated_rewrite():
    out = rs.guard_status('if grep -rq "SECRET" src/; then\n  exit 1\nfi\n')
    assert out == ('__rc=0; grep -rq "SECRET" src/ || __rc=$?\n'
                   'if [ "$__rc" -gt 1 ]; then echo "guard command failed (exit $__rc): grep -rq \\"SECRET\\" src/" >&2; exit "$__rc"; fi\n'
                   'if [ "$__rc" -eq 0 ]; then\n  exit 1\nfi\n')
    assert rs.guard_status('if ! grep -q x f\nthen\n  echo no\nfi\n').splitlines()[2] == 'if [ "$__rc" -ne 0 ]; then'
    # builtins, tests, arithmetic and assignments are not tools: untouched
    same = 'if [ -n "$x" ]; then echo; fi\nif test -f a; then :; fi\nif [[ $a == b ]]; then :; fi\nif (( n > 1 )); then :; fi\nif ! [ -f a ]; then :; fi\n'
    assert rs.guard_status(same) == same
    assert rs.guard_status("if a | grep -q b; then\n  x\nfi\n").splitlines()[0] == "__rc=0; a | grep -q b || __rc=$?"


def test_no_default_is_the_stated_rewrite():
    assert rs.no_default('X=$(git rev-parse HEAD || echo "unknown")\n') == "X=$(git rev-parse HEAD)\n"
    assert rs.no_default('cmd || echo failed >&2\n') == "cmd\n"
    assert rs.no_default('npm run build | tee log \\\n  || echo "failed"\necho done\n') == "npm run build | tee log\necho done\n"
    assert rs.no_default("cmd || true\n") == "cmd || true\n"                                # not a default: strict-shell's job
    # quotes are read: a `)` or `;` inside the fallback's string is not the end of it (run 1 of SWALLOW-5 cut one mid-string)
    assert rs.no_default('npm run x || echo "failed for $pkg (continuing)"\n') == "npm run x\n"
    assert rs.no_default('cmd || echo "a;b" ; next\n') == "cmd ; next\n"
    assert rs.no_default("cmd || echo 'x)y' && other\n") == "cmd && other\n"
    assert rs.no_default('cmd || echo "unterminated\n') == 'cmd || echo "unterminated\n'                 # left alone
    assert rs.no_default('cmd || { echo x; exit 0; }\n') == 'cmd || { echo x; exit 0; }\n'  # a block is not touched
    assert rs.transform('cmd\n', "no-default") == (None, "no `|| echo …` fallback")
    assert rs.transform('X=$(a || echo b)\n', "no-default") == ("set -eo pipefail\nX=$(a)\n", None)
    assert rs.transform('if a; then\n  :\nfi\n', "both-structural") == (None, "both edits must apply")


def test_every_outcome_on_the_fixture(tmp_path):
    rec = rs.structural_tree(_tree(tmp_path))
    # the first stage (SWALLOW-4's repairs) leaves all of these unverified; `Verify build log` is not a check (build)
    assert [t["index"] for t in rec["first_stage"]] == [0, 1, 2, 4, 5, 6, 7]
    assert all(t["verified_repair"] is None for t in rec["first_stage"])
    by = {t["index"]: t for t in rec["targets"]}
    assert set(by) == {0, 1, 2, 4, 5, 6, 7}
    # a guard whose failing grep was the green path: the tool's failure now fails the step; grep's "no" still passes
    g = by[0]
    assert g["verified_repair"] == "guard-status"
    c = next(c for c in g["candidates"] if c["repair"] == "guard-status")
    assert c["loud"] and c["unchanged"] and c["by_flavour_after"] == {"x": "BASELINE_RED", "empty": "RED"}      # x was an artifact before and after
    assert '+          __rc=0; grep -rq "SECRET" src/ || __rc=$?' in c["diff"] and c["lines_changed"] == 4
    # the negated, two-line form
    assert by[1]["verified_repair"] == "guard-status"
    assert next(c for c in by[1]["candidates"] if c["repair"] == "guard-status")["by_flavour_after"] == {"x": "RED", "empty": "RED"}
    # a query with a default: the default removed, the strict shell sees the failure
    assert by[2]["verified_repair"] == "no-default"
    assert "+          CURRENT=$(git rev-parse --abbrev-ref HEAD)" in next(c for c in by[2]["candidates"] if c["repair"] == "no-default")["diff"]
    # a default on a continuation line of its own
    assert by[4]["verified_repair"] == "no-default"
    d = next(c for c in by[4]["candidates"] if c["repair"] == "no-default")["diff"]
    assert '-            || echo "golden tests failed (non-blocking)"' in d and "+          pnpm --filter shared run test golden" in d
    # a guard followed by a loud test: still the guard
    assert by[5]["verified_repair"] == "guard-status"
    # the twin condition: a default after grep protects the healthy run in the empty flavour
    v = by[6]
    assert v["verified_repair"] is None
    c = next(c for c in v["candidates"] if c["repair"] == "no-default")
    assert c["applies"] and c["loud_by_flavour"] == {"x": True, "empty": False} and c["unchanged_by_flavour"] == {"x": True, "empty": False}
    assert "what the repaired line was protecting" in c["why"]
    # a loop that counts: no structural candidate applies
    assert by[7]["verified_repair"] is None and all(not c.get("applies") for c in by[7]["candidates"])
    s = rs.summary({"repos": [rec]})["all"]
    assert s == {"targets": 7, "verified": 5, "by_repair": {"guard-status": 3, "no-default": 2}, "baseline_differs_from_receipt": 0,
                 "rejected_changes_healthy_run": 1, "no_candidate_applies": 1, "not_loud": 0, "lines_changed": [3, 4, 4, 4, 5]}


def test_the_instrument_is_deterministic_on_the_fixture(tmp_path):
    a = rs.structural_tree(_tree(tmp_path))
    b = rs.structural_tree(_tree(tmp_path))
    strip = lambda rec: [(t["index"], t["verdict"], t["verified_repair"],  # noqa: E731
                          [(c["repair"], c.get("applies"), c.get("loud"), c.get("unchanged"), c.get("diff")) for c in t["candidates"]]) for t in rec["targets"]]
    assert strip(a) == strip(b)


def test_this_repository_has_nothing_to_repair():
    rec = rs.structural_tree(ROOT)
    assert rec["workflows"] and rec["targets"] == [] and rec["first_stage"] == []

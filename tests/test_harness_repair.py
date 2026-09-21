# -*- coding: utf-8 -*-
"""SWALLOW-4's instrument: two stated repairs at a fault site, each verified on both halves --
loud under the same fault, and a healthy run left exactly as it was. The tests hold the text
edits to the workflow, the twin condition, the order of the repairs, and determinism."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import repair  # noqa: E402

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
            continue-on-error: true
            run: npm run lint
          - name: Lint colors
            run: pnpm lint:colors || true
          - name: Comment (best effort)
            run: gh pr comment 1 --body done || true
      lint:
        runs-on: ubuntu-latest
        steps:
          - name: Lint changed python
            run: |
              files=$(git diff --name-only origin/main | grep '\\.py$' || true)
              ruff check $files || true
          - name: Lint for secrets
            run: |
              if grep -rq "SECRET" src/; then
                exit 1
              fi
"""


def _tree(tmp_path: Path, text: str = FIXTURE) -> Path:
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True, exist_ok=True)
    (wf / "ci.yml").write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return tmp_path


def test_strict_shell_is_a_text_transformation_with_the_stated_shape():
    assert repair.strict_shell("npm test || true") == "set -eo pipefail\nnpm test"
    assert repair.strict_shell("x=$(cmd || true)\ncmd2 || :\n") == "set -eo pipefail\nx=$(cmd)\ncmd2\n"
    assert repair.strict_shell("set +e\nset -e\npytest\n") == "set -o pipefail\nset -e\npytest\n"
    assert repair.strict_shell("set -euo pipefail\npytest\n") == "set -euo pipefail\npytest\n"       # nothing to do
    assert repair.strict_shell("#!/bin/bash\nnpm test || true\n") == "#!/bin/bash\nset -eo pipefail\nnpm test\n"
    assert repair.strict_shell("if grep -q x f; then exit 1; fi") == "set -eo pipefail\nif grep -q x f; then exit 1; fi"


def test_repairs_are_edits_to_the_workflow_text(tmp_path):
    text = (_tree(tmp_path) / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    new, why = repair.apply_repair(text, "test", 1, "no-continue-on-error")
    assert why is None and "continue-on-error" not in new
    assert repair.diff_size(text, new) == 1
    new, why = repair.apply_repair(text, "test", 2, "strict-shell")
    assert why is None and "run: |\n          set -eo pipefail\n          pnpm lint:colors\n" in new
    new, why = repair.apply_repair(text, "changes", 0, "strict-shell")
    assert why is None and "sort -u)\n" in new and "|| true" not in new.split("test:")[0]
    assert repair.apply_repair(text, "test", 0, "no-continue-on-error") == (None, "no continue-on-error on the step or its job")
    new, why = repair.apply_repair(text, "test", 0, "strict-shell")           # a bare `python -m pytest` only gains set -eo pipefail: it applies
    assert why is None and "set -eo pipefail\n          python -m pytest tests -q" in new
    strict = "set -euo pipefail\nnpm test\n"
    assert repair.apply_repair(text.replace("run: python -m pytest tests -q", "run: |\n          " + strict.replace("\n", "\n          ").rstrip() ), "test", 0, "strict-shell") == \
        (None, "nothing to make strict: no || true, no set +e, and set -e / pipefail already present")
    import yaml
    for jid, i, r in (("test", 1, "no-continue-on-error"), ("test", 2, "strict-shell"), ("changes", 0, "both")):
        t2, _ = repair.apply_repair(text, jid, i, r)
        if t2 is not None:
            yaml.safe_load(t2)                                                   # every repaired text still parses


def test_every_outcome_on_the_fixture(tmp_path):
    rec = repair.repair_tree(_tree(tmp_path))
    by = {(t["job"], t["index"]): t for t in rec["targets"]}
    assert set(by) == {("changes", 0), ("test", 1), ("test", 2), ("lint", 0), ("lint", 1)}
    # a dropped check: the query's || true removed, loud in both flavours, healthy run unchanged
    q = by[("changes", 0)]
    assert q["verdict"] == "FAIL_OPEN" and q["verified_repair"] == "strict-shell"
    c = next(c for c in q["candidates"] if c["repair"] == "strict-shell")
    assert c["loud"] and c["unchanged"] and c["lines_changed"] == 3 and "+          set -eo pipefail" in c["diff"]
    # a hidden check under continue-on-error: the one-line repair comes first
    n = by[("test", 1)]
    assert n["verdict"] == "SWALLOWED" and n["verified_repair"] == "no-continue-on-error"
    assert [c["repair"] for c in n["candidates"] if c.get("verified")] == ["no-continue-on-error", "both"]
    assert next(c for c in n["candidates"] if c["repair"] == "strict-shell")["why"].startswith("not loud")
    # a hidden check by the shell
    assert by[("test", 2)]["verified_repair"] == "strict-shell"
    # the twin condition: a || true after grep protects the healthy run in the empty flavour
    lb = by[("lint", 0)]
    assert lb["verified_repair"] is None
    c = next(c for c in lb["candidates"] if c["repair"] == "strict-shell")
    assert c["applies"] and c["unchanged"] is False and c["unchanged_by_flavour"] == {"x": True, "empty": False}
    assert "what the repaired line was protecting" in c["why"] and c["healthy_change"]["after"][-1] is True     # the step became a model artifact
    # a guard whose failing condition is its false branch: neither repair makes it loud
    g = by[("lint", 1)]
    assert g["verified_repair"] is None and all(c["why"].startswith("not loud") for c in g["candidates"] if c.get("applies"))
    s = repair.summary({"repos": [rec]})["all"]
    assert s == {"targets": 5, "verified": 3, "by_repair": {"strict-shell": 2, "no-continue-on-error": 1}, "baseline_differs_from_receipt": 0,
                 "rejected_changes_healthy_run": 1, "no_candidate_applies": 0, "not_loud": 1, "lines_changed": [1, 3, 4]}


def test_the_instrument_is_deterministic_on_the_fixture(tmp_path):
    a = repair.repair_tree(_tree(tmp_path))
    b = repair.repair_tree(_tree(tmp_path))
    strip = lambda rec: [(t["job"], t["index"], t["verdict"], t["verified_repair"],  # noqa: E731
                          [(c["repair"], c.get("applies"), c.get("loud"), c.get("unchanged"), c.get("diff")) for c in t["candidates"]]) for t in rec["targets"]]
    assert strip(a) == strip(b)


def test_this_repository_has_nothing_to_repair():
    rec = repair.repair_tree(ROOT)
    assert rec["workflows"] and rec["targets"] == []

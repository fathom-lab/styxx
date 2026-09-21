# -*- coding: utf-8 -*-
"""SWALLOW-3's instrument: the catalogue of checking actions, applied on top of the frozen
fault-injection instrument. The tests hold the wrapper to the preregistration: which `uses:`
steps count, what the inputs do, that an action check is reached / dropped as stated, that the
frozen reading is carried beside the new one unchanged, and that the only transitions are the
declared ones."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import action_checks as ac  # noqa: E402
from benchmarks.harness_mutation import faults  # noqa: E402

FIXTURE = """
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
              args: docs/
      lint:
        needs: changes
        if: needs.changes.outputs.docs != ''
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - name: Install
            run: pip install pre-commit
          - uses: pre-commit/action@v3.0.1
      release:
        runs-on: ubuntu-latest
        steps:
          - name: Notify
            run: curl -X POST https://example.com/hook || true
          - name: Retry the tests
            uses: nick-fields/retry@v3
            with:
              max_attempts: 3
              command: npm test
          - name: Reviewdog (comments only)
            uses: reviewdog/action-eslint@v1
          - name: Reviewdog (fails)
            uses: reviewdog/action-eslint@v1
            with:
              fail_level: error
      soft:
        runs-on: ubuntu-latest
        continue-on-error: true
        steps:
          - name: Fetch fixtures
            run: curl -o fixtures.tar https://example.com/fixtures.tar
          - uses: actions/dependency-review-action@v4
"""


def _tree(tmp_path: Path, text: str = FIXTURE) -> Path:
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents=True, exist_ok=True)
    (wf / "ci.yml").write_text(textwrap.dedent(text), encoding="utf-8")
    return tmp_path


def _step(uses, **with_):
    return {"uses": uses, "with": with_} if with_ else {"uses": uses}


def test_the_catalogue_reads_uses_names_and_inputs():
    assert ac.action_check(_step("pre-commit/action@v3.0.1")) == ("lint", "action")
    assert ac.action_check(_step("github/codeql-action/analyze@v3", category="python")) == ("status", "action")
    assert ac.action_check(_step("actions/checkout@v4")) is None
    assert ac.action_check(_step("./.github/actions/verify")) is None                     # local: cannot be read
    assert ac.action_check(_step("docker://ghcr.io/x/y:1")) is None
    assert ac.action_check(_step("actions/github-script@v7")) is None
    assert ac.action_check(_step("somebody/unknown-action@v1")) is None                   # not listed: not a check
    # inputs that turn the check off
    assert ac.action_check(_step("lycheeverse/lychee-action@v2", fail=True)) == ("lint", "action")
    assert ac.action_check(_step("lycheeverse/lychee-action@v2", fail=False)) is None
    assert ac.action_check(_step("lycheeverse/lychee-action@v2", fail="false")) is None
    assert ac.action_check(_step("anchore/scan-action@v3", **{"fail-build": "false"})) is None
    assert ac.action_check(_step("cypress-io/github-action@v6", runTests=False)) is None
    # inputs that turn it on
    assert ac.action_check(_step("aquasecurity/trivy-action@0.28.0")) is None            # exit-code defaults to 0
    assert ac.action_check(_step("aquasecurity/trivy-action@0.28.0", **{"exit-code": "1"})) == ("security", "action")
    assert ac.action_check(_step("reviewdog/action-actionlint@v1")) is None
    assert ac.action_check(_step("reviewdog/action-actionlint@v1", fail_level="error")) == ("lint", "action")
    assert ac.action_check(_step("reviewdog/action-golangci-lint@v2", fail_on_error=True)) == ("lint", "action")   # the family
    assert ac.action_check(_step("snyk/actions/python@master")) == ("security", "action")
    assert ac.action_check(_step("snyk/actions/setup@master")) is None
    # fixers and formatters
    assert ac.action_check(_step("DavidAnson/markdownlint-cli2-action@v19", fix=True)) is None
    assert ac.action_check(_step("astral-sh/ruff-action@v3")) == ("lint", "action")
    assert ac.action_check(_step("astral-sh/ruff-action@v3", args="format")) is None
    assert ac.action_check(_step("astral-sh/ruff-action@v3", args="format --check")) == ("lint", "action")
    assert ac.action_check(_step("psf/black@stable", options="--check --verbose")) == ("lint", "action")
    assert ac.action_check(_step("psf/black@stable", options=".")) is None
    assert ac.action_check(_step("golangci/golangci-lint-action@v6", args="--fix")) is None
    # carried commands: SWALLOW-2's rule on the command the action runs
    assert ac.action_check(_step("nick-fields/retry@v3", command="npm test")) == ("carried", "action-carried:tool")
    assert ac.action_check(_step("nick-fields/retry@v3", command="npm install")) is None
    assert ac.action_check({"uses": "nick-fields/retry@v3", "name": "Run tests", "with": {"command": "./scripts/go.sh"}}) == ("carried", "action-carried:name")
    assert ac.action_check(_step("actions-rs/cargo@v1", command="test")) == ("carried", "action-carried:tool")
    assert ac.action_check(_step("actions-rs/cargo@v1", command="build")) is None
    assert ac.action_check(_step("mansagroup/nrwl-nx-action@v3", targets="lint,test", projects="web")) == ("carried", "action-carried:name")
    assert ac.action_check(_step("mansagroup/nrwl-nx-action@v3", targets="build", projects="web")) is None
    assert ac.action_check(_step("Wandalen/wretry.action@v3", action="pre-commit/action@v3.0.1", attempt_limit=3)) == ("lint", "action-nested")
    assert ac.action_check(_step("Wandalen/wretry.action@v3", action="actions/checkout@v4")) is None


def test_every_census_name_is_decided_once():
    census = json.loads((ROOT / "papers" / "harness" / "swallow3_actions_census.json").read_text(encoding="utf-8"))
    names = [a["name"] for a in census["actions_hand_written"]]
    undecided = [n for n in names if ac._entry(n) is None and n not in ac.NOT_CHECKS]
    twice = [n for n in names if ac._entry(n) is not None and n in ac.NOT_CHECKS]
    assert not undecided and not twice
    assert all(e["kind"] in ac.KINDS for e in ac.CATALOGUE.values())


def test_a_gated_action_check_is_dropped_and_swallow2s_reading_is_carried_beside(tmp_path):
    rec = ac.analyse_tree(_tree(tmp_path))
    by = {(f["job"], f["index"]): f for f in rec["faults"]}
    q = by[("changes", 0)]
    assert q["verdict_runs_only"] == "NO_CHECK" and q["verdict"] == "FAIL_OPEN"
    drops = {(d["action"], d["mechanism"]) for d in q["dropped_actions"]}
    assert drops == {("lycheeverse/lychee-action", "step-if"), ("pre-commit/action", "job-if")}
    assert all(d["cross_step"] for d in q["dropped_actions"])
    assert q["action_checks_in_scope"] == 2 and q["action_checks_live"] == 2 and q["checks_in_scope"] == 2
    n = by[("release", 0)]
    assert n["verdict_runs_only"] == "NO_CHECK" and n["verdict"] == "ABSORBED"    # the retried tests and the failing reviewdog still run
    assert n["action_checks_in_scope"] == 2
    s = by[("soft", 0)]
    # a job-level continue-on-error: the frozen model reads it as every step continuing (faults.py, unchanged
    # here), so the later action is reached in both worlds -- ABSORBED, not dropped; the wrapper's
    # `after-failure` mechanism is reserved for a model that stops the job, and is not reachable under this one
    assert s["verdict_runs_only"] == "NO_CHECK" and s["verdict"] == "ABSORBED"
    assert s["dropped_actions"] == [] and s["action_checks_live"] == 1
    i = by[("lint", 1)]
    assert i["verdict"] == i["verdict_runs_only"] == "RED"                          # a red job is red whatever the checks
    ws = rec["workflow_summaries"]["ci.yml"]
    assert ws["action_checks"] == 5 and ws["action_checks_reached_in_plus"] == 5
    assert ws["step_classes"]["conditional-off"] == 1 and ws["step_classes"]["check:carried"] == 1


def test_the_frozen_reading_is_the_frozen_instruments(tmp_path):
    tree = _tree(tmp_path)
    a, b = ac.analyse_tree(tree), faults.analyse_tree(tree)
    assert [(f["job"], f["index"], f["verdict_runs_only"], f["verdict_counted_runs_only"]) for f in a["faults"]] == \
           [(f["job"], f["index"], f["verdict"], f["verdict_counted"]) for f in b["faults"]]
    assert [f["by_flavour_runs_only"] for f in a["faults"]] == [f["by_flavour"] for f in b["faults"]]
    assert ac.analyse_tree(tree)["faults"] == a["faults"]                            # deterministic


def test_only_the_declared_transitions(tmp_path):
    rec = ac.analyse_tree(_tree(tmp_path))
    for f in rec["faults"]:
        if f["verdict"] != f["verdict_runs_only"]:
            assert (f["verdict_runs_only"], f["verdict"]) in ac._TRANSITIONS
    s = ac._summary([rec])
    assert s["moved_outside_declared_transitions"] == 0
    assert s["by_verdict"].get("RED", 0) == s["by_verdict_runs_only"].get("RED", 0)
    assert s["by_verdict"].get("SWALLOWED", 0) == s["by_verdict_runs_only"].get("SWALLOWED", 0)


def test_this_repository_is_unmoved():
    rec = ac.analyse_tree(ROOT)
    assert rec["faults"] and all(f["verdict"] == f["verdict_runs_only"] for f in rec["faults"])
    assert ac._summary([rec])["action_checks"] == 0


def test_the_instrument_underneath_is_the_frozen_one():
    import hashlib
    assert hashlib.sha256((ROOT / "benchmarks" / "harness_mutation" / "faults.py").read_bytes()).hexdigest() == \
        "d26a407ca276d1b1c218fd32bc6bf853544fc394ce5ebc98b2b6ae4bd0fa3543"

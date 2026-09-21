# -*- coding: utf-8 -*-
"""SWALLOW-6's instrument: every check step followed through a repository's mainline history as a
lineage, with its birth state, its acquisitions and repairs (mechanism, acknowledgement, and the
instrument's own repair set against the author's), its renames and its death. The tests build a
scripted history and hold every event, the counts, and determinism."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import history as H  # noqa: E402

V1 = """
    on: [push]
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - name: Run tests
            run: python -m pytest tests -q
          - name: Lint colors
            run: pnpm lint:colors || true
          - name: Lint (non-blocking)
            run: npm run lint
          - name: Lint markdown
            run: npx markdownlint docs/ || true
      nightly:
        runs-on: ubuntu-latest
        if: false
        steps:
          - name: Run nightly tests
            run: python -m pytest tests/nightly -q || true
"""
V2 = V1.replace("""          - name: Lint (non-blocking)
            run: npm run lint""", """          - name: Lint (non-blocking)
            continue-on-error: true
            run: npm run lint""")
V3 = V2.replace("run: pnpm lint:colors || true", "run: pnpm lint:colors")
V4 = V3.replace("- name: Run tests", "- name: Run unit tests")
V5 = V4.replace("""          - name: Lint markdown
            run: npx markdownlint docs/ || true
""", "")
RELEASE = """
    on: [push]
    jobs:
      release:
        runs-on: ubuntu-latest
        steps:
          - name: Verify tag
            run: git describe --exact-match --tags
          - name: Lint release notes
            run: npx markdownlint RELEASE.md || true
"""
RELEASE_FIXED = RELEASE.replace("run: npx markdownlint RELEASE.md || true", "run: npx markdownlint RELEASE.md")
SCRIPT = [  # (date, subject, {path: text | None})
    ("2023-01-01T12:00:00", "ci: add the pipeline", {"ci.yml": V1}),
    ("2023-03-01T12:00:00", "ci: make lint non-blocking for now (flaky)", {"ci.yml": V2}),
    ("2023-06-01T12:00:00", "ci: fail on lint colors", {"ci.yml": V3}),
    ("2023-09-01T12:00:00", "ci: rename the test step", {"ci.yml": V4}),
    ("2024-01-01T12:00:00", "ci: drop the markdown lint", {"ci.yml": V5}),
    ("2024-02-01T12:00:00", "ci: add a release workflow", {"release.yml": RELEASE}),
    ("2024-02-15T12:00:00", "ci: rename the release workflow", {"release.yml": None, "publish.yml": RELEASE}),   # a rename: git sees R100
    ("2024-02-20T12:00:00", "ci: fail on release notes lint", {"publish.yml": RELEASE_FIXED}),                # a repair whose revision before predates nothing, but whose lineage began under the old path
    ("2024-03-01T12:00:00", "ci: remove the release workflow", {"publish.yml": None}),
]


def _repo(tmp_path: Path) -> Path:
    tree = tmp_path / "repo"
    wf = tree / ".github" / "workflows"
    wf.mkdir(parents=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    subprocess.run(["git", "init", "-q", "-b", "main", str(tree)], check=True, env=env)
    for date, subject, files in SCRIPT:
        for name, text in files.items():
            if text is None:
                (wf / name).unlink()
            else:
                (wf / name).write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
        e = dict(env, GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date)
        subprocess.run(["git", "-C", str(tree), "add", "-A"], check=True, env=e)
        subprocess.run(["git", "-C", str(tree), "commit", "-q", "-m", subject], check=True, env=e)
    return tree


def _by_key(rec: dict, wf: str) -> dict:
    return {(lin["job"], lin["key"]): lin for lin in rec["workflows"][wf]["lineages"]}


def test_the_lineage_key_and_the_mechanism_are_the_stated_rules():
    assert H.step_key({"name": " Run tests ", "id": "x", "run": "pytest"}) == "name:Run tests"
    assert H.step_key({"id": "q", "run": "pytest"}) == "id:q"
    assert H.step_key({"run": "\n  pytest -q\n  echo done\n"}) == "run:pytest -q"
    before = {"step": {"run": "npm test"}, "job": {}}
    assert H.mechanism(before, {"step": {"run": "npm test", "continue-on-error": True}, "job": {}}) == ["continue-on-error"]
    assert H.mechanism(before, {"step": {"run": "npm test || true"}, "job": {}}) == ["or-true"]
    assert H.mechanism(before, {"step": {"run": "set +e\nnpm test"}, "job": {}}) == ["set+e"]
    assert H.mechanism(before, {"step": {"run": "npm test || echo failed"}, "job": {}}) == ["default"]
    assert H.mechanism({"step": {"run": "npm test || true"}, "job": {}}, {"step": {"run": "set -e\nnpm test"}, "job": {}}) == ["or-true", "strict-shell"]
    assert H.mechanism(before, {"step": {"run": "npm test", "if": "false"}, "job": {}}) == ["gating"]
    assert H.mechanism(before, {"step": {"run": "npm test"}, "job": {"continue-on-error": True}}) == ["continue-on-error"]
    assert H.mechanism(before, {"step": {"run": "npm run test:unit"}, "job": {}}) == ["rewrite"]
    assert H.mechanism(before, {"step": {"run": "npm test"}, "job": {}}) == ["context"]
    assert H.acknowledged({"subject": "ci: make lint non-blocking for now (flaky)", "body": ""}, []) == "non-blocking"
    assert H.acknowledged({"subject": "perf: optionally move ingestion", "body": ""}, []) is None
    assert H.acknowledged({"subject": "Merge pull request #1", "body": ""}, ["skip the flaky e2e"]) == "skip"
    assert H.state_of("SWALLOWED") == "hidden" and H.state_of("FAIL_OPEN") == "hidden" and H.state_of("RED") == "loud"
    assert H.state_of("BASELINE_RED") == "unread" and H.state_of("NO_CHECK") == "other" and H.state_of(None) == "other"


def test_every_event_on_the_scripted_history(tmp_path):
    rec = H.tree_history(_repo(tmp_path), "fixture")
    assert set(rec["workflows"]) == {"ci.yml", "publish.yml"}                 # the renamed workflow is one history under its last name
    assert rec["workflows"]["ci.yml"]["revisions"] == 5 and rec["workflows"]["ci.yml"]["reads"] == 5
    assert rec["workflows"]["publish.yml"]["revisions"] == 4 and rec["workflows"]["publish.yml"]["reads"] == 2   # added, renamed, repaired, removed: two texts
    ci = _by_key(rec, "ci.yml")
    # born loud, renamed, alive: no event
    t = ci[("test", "name:Run unit tests")]
    assert t["born"]["state"] == "loud" and t["state"] == "loud" and t["alive"] and t["events"] == []
    assert [r["from"] + " -> " + r["to"] for r in t["renames"]] == ["name:Run tests -> name:Run unit tests"]
    # born hidden (`|| true`), repaired by the author in the wild; the instrument's repair agrees
    c = ci[("test", "name:Lint colors")]
    assert c["born"]["state"] == "hidden" and c["born"]["verdict"] == "SWALLOWED" and c["state"] == "loud" and c["alive"]
    (ev,) = c["events"]
    assert ev["kind"] == "repair" and ev["mechanism"] == ["or-true"] and ev["subject"] == "ci: fail on lint colors" and ev["acknowledged"] is None
    assert ev["agreement"]["verified_repair"] == "strict-shell" and ev["agreement"]["stage"] == "swallow-4" and ev["agreement"]["agrees"]
    # born loud, hidden by a `continue-on-error: true` in a commit that says so; still hidden at HEAD
    n = ci[("test", "name:Lint (non-blocking)")]
    assert n["born"]["state"] == "loud" and n["state"] == "hidden" and n["alive"]
    (ev,) = n["events"]
    assert ev["kind"] == "acquisition" and ev["mechanism"] == ["continue-on-error"] and ev["acknowledged"] == "non-blocking"
    assert ev["subject"] == "ci: make lint non-blocking for now (flaky)"
    # born hidden, removed hidden
    d = ci[("test", "name:Lint markdown")]
    assert d["born"]["state"] == "hidden" and not d["alive"] and d["death"]["state"] == "hidden" and d["death"]["why"] == "step removed"
    assert d["death"]["subject"] == "ci: drop the markdown lint"
    # a workflow that came, was renamed (its text read at the path it had then), and went
    r = _by_key(rec, "publish.yml")[("release", "name:Verify tag")]
    assert r["born"]["state"] == "loud" and r["born"]["subject"] == "ci: add a release workflow" and r["revisions"] == 3
    assert not r["alive"] and r["death"]["why"] == "workflow removed" and r["death"]["subject"] == "ci: remove the release workflow"
    # a hidden check born under the old path, repaired under the new one: the agreement reads the revision before at its path then
    ln = _by_key(rec, "publish.yml")[("release", "name:Lint release notes")]
    assert ln["born"]["state"] == "hidden" and ln["born"]["subject"] == "ci: add a release workflow" and ln["revisions"] == 3
    (ev,) = ln["events"]
    assert ev["kind"] == "repair" and ev["mechanism"] == ["or-true"] and ev["agreement"]["verified_repair"] == "strict-shell" and ev["agreement"]["agrees"]
    # a check the reading cannot interpret at any revision (its job is skipped in the healthy world): unread, never hidden, never loud
    u = ci[("nightly", "name:Run nightly tests")]
    assert u["born"]["state"] == "unread" and u["state_last"] == "unread" and u["verdict_last"] == "BASELINE_SKIPPED" and u["alive"] and u["events"] == []
    s = rec["summary"]
    assert s["alive_unread"] == 1
    assert s["alive_hidden"] == 1 and s["alive_hidden_born_hidden"] == 0 and s["alive_hidden_acquired"] == 1
    assert s["ever_hidden"] == 4 and s["died_hidden"] == 1 and s["died_loud"] == 2
    assert s["acquisitions"] == 1 and s["acquisitions_acknowledged"] == 1 and s["acquisition_mechanisms"] == {"continue-on-error": 1}
    assert s["repairs"] == 2 and s["repairs_agreeing"] == 2 and s["repairs_instrument_verified"] == 2 and s["repair_mechanisms"] == {"or-true": 2}
    assert s["ages_days"] == [366.0]              # hidden since 2023-03-01, HEAD at 2024-03-01
    assert rec["shallow"] is False


def test_the_instrument_is_deterministic_on_the_scripted_history(tmp_path):
    tree = _repo(tmp_path)
    a, b = H.tree_history(tree, "fixture"), H.tree_history(tree, "fixture")
    strip = lambda rec: {wf: [(l["job"], l["key"], l["born"]["state"], l["state"], l.get("alive"),  # noqa: E731
                               [(e["kind"], e["sha"], tuple(e["mechanism"]), e["acknowledged"], (e.get("agreement") or {}).get("verified_repair")) for e in l["events"]])
                              for l in w["lineages"]] for wf, w in rec["workflows"].items()}
    assert strip(a) == strip(b) and a["summary"] == b["summary"]


def test_this_repository_reads_and_hides_nothing():
    rec = H.tree_history(ROOT, "fathom-lab/styxx")
    assert rec["workflows"]
    assert rec["summary"]["alive_hidden"] == 0

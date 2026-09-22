# -*- coding: utf-8 -*-
"""SWALLOW-9's instrument: SWALLOW-7's gate run on a pull request -- HEAD its head as GitHub keeps
it, BASE what the dataset's own commit list implies (the parents of the pull request's commits
that are not its commits) -- reading only the workflows the dataset says the pull request
changed, and flagging a BASE that git's diff contradicts. The tests script seven pull requests on
a local repository (a merge commit, a squash, a fast-forward, one closed, one whose head is gone,
one that merged main into itself, one into a `dev` branch) and hold the base each gets, what the
gate says, the suspect flag, the reading at the tip, the boundary rule, determinism, and that no
name is written."""
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
from benchmarks.harness_mutation import agent_prs as P  # noqa: E402

# frozen at the sha256 the SWALLOW-9 receipt names (papers/harness/swallow9_receipt.json.gz); a change needs a new receipt, not a new pin
INSTRUMENT_SHA256 = "ec9ef750826516a8ff425f184598d38b1249759b06bfe172518166e514b2b950"

CI = """
    on: [push]
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - name: Run tests
            run: python -m pytest tests -q
          - name: Lint
            run: npm run lint
"""
CI_PR1 = CI + """      docs:
        runs-on: ubuntu-latest
        steps:
          - name: Lint docs
            continue-on-error: true
            run: npx markdownlint docs/
"""
CI_PR2 = CI_PR1.replace("run: npm run lint", "run: npm run lint || true")
CI_PR4 = CI_PR2 + """      typecheck:
        runs-on: ubuntu-latest
        steps:
          - name: Typecheck
            run: npx tsc --noEmit
"""
NIGHTLY = """
    on: [push]
    jobs:
      nightly:
        runs-on: ubuntu-latest
        steps:
          - name: Nightly tests
            run: python -m pytest tests/nightly -q || true
"""


def _git(tree: Path, *args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", "-C", str(tree), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def _write(tree: Path, name: str, text: str | None) -> None:
    p = tree / ".github" / "workflows" / name
    if text is None:
        p.unlink()
    else:
        p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


def _repo(tmp_path: Path) -> tuple[Path, dict]:
    """main: c0 -> c1 -> M(PR1) -> S(PR2 squashed) -> f1(PR4 fast-forwarded); PR3 closed off S; PR5 has no head."""
    tree = tmp_path / "repo"
    (tree / ".github" / "workflows").mkdir(parents=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com",
               GIT_AUTHOR_DATE="2025-03-01T12:00:00", GIT_COMMITTER_DATE="2025-03-01T12:00:00")
    subprocess.run(["git", "init", "-q", "-b", "main", str(tree)], check=True, env=env)

    def commit(msg: str) -> str:
        _git(tree, "add", "-A", env=env)
        _git(tree, "commit", "-q", "--allow-empty", "-m", msg, env=env)
        return _git(tree, "rev-parse", "HEAD")

    shas = {}
    _write(tree, "ci.yml", CI)
    shas["c0"] = commit("ci: add the pipeline")
    (tree / "README.md").write_text("hello\n")
    shas["c1"] = commit("docs: readme")
    # PR1: a branch off c1, merged by a merge commit
    _git(tree, "checkout", "-q", "-b", "pr1", env=env)
    _write(tree, "ci.yml", CI_PR1)
    shas["p1"] = commit("ci: add a docs lint job (non-blocking)")
    _git(tree, "checkout", "-q", "main", env=env)
    _git(tree, "merge", "-q", "--no-ff", "-m", "Merge pull request #1 from org/pr1", "pr1", env=env)
    shas["M"] = _git(tree, "rev-parse", "HEAD")
    _git(tree, "update-ref", "refs/pr/1", shas["p1"])
    # PR2: a branch off M, squash-merged
    _git(tree, "checkout", "-q", "-b", "pr2", env=env)
    _write(tree, "ci.yml", CI_PR2)
    shas["q1"] = commit("ci: keep lint from failing the build")
    _git(tree, "checkout", "-q", "main", env=env)
    _git(tree, "merge", "-q", "--squash", "pr2", env=env)
    shas["S"] = commit("ci: keep lint from failing the build (#2)")
    _git(tree, "update-ref", "refs/pr/2", shas["q1"])
    # PR3: a branch off S, closed without merging
    _git(tree, "checkout", "-q", "-b", "pr3", env=env)
    _write(tree, "nightly.yml", NIGHTLY)
    shas["r1"] = commit("ci: add a nightly workflow")
    _git(tree, "checkout", "-q", "main", env=env)
    _git(tree, "update-ref", "refs/pr/3", shas["r1"])
    # PR4: a branch off S, fast-forwarded
    _git(tree, "checkout", "-q", "-b", "pr4", env=env)
    _write(tree, "ci.yml", CI_PR4)
    shas["f1"] = commit("ci: add a typecheck job")
    _git(tree, "checkout", "-q", "main", env=env)
    _git(tree, "merge", "-q", "--ff-only", "pr4", env=env)
    _git(tree, "update-ref", "refs/pr/4", shas["f1"])
    assert _git(tree, "rev-parse", "HEAD") == shas["f1"]
    # PR6: a branch off c1 that adds a loud check, then merges main (f1) into itself; closed
    _git(tree, "checkout", "-q", "-b", "pr6", shas["c1"], env=env)
    _write(tree, "audit.yml", """
    on: [push]
    jobs:
      audit:
        runs-on: ubuntu-latest
        steps:
          - name: Audit
            run: npm audit
""")
    shas["x1"] = commit("ci: add an audit workflow")
    _git(tree, "merge", "-q", "-m", "Merge branch 'main' into pr6", "main", env=env)
    shas["x2"] = _git(tree, "rev-parse", "HEAD")
    _git(tree, "checkout", "-q", "main", env=env)
    _git(tree, "update-ref", "refs/pr/6", shas["x2"])
    # dev: a branch off c1 where a person added the hidden docs lint; PR7 targets dev and adds a loud check
    _git(tree, "checkout", "-q", "-b", "dev", shas["c1"], env=env)
    _write(tree, "ci.yml", CI_PR1)
    shas["d1"] = commit("ci: docs lint on dev (non-blocking)")
    _git(tree, "checkout", "-q", "-b", "pr7", env=env)
    _write(tree, "ci.yml", CI_PR1 + """      typecheck:
        runs-on: ubuntu-latest
        steps:
          - name: Typecheck
            run: npx tsc --noEmit
""")
    shas["y1"] = commit("ci: add a typecheck job (dev)")
    _git(tree, "checkout", "-q", "main", env=env)
    _git(tree, "update-ref", "refs/pr/7", shas["y1"])
    return tree, shas


def _prs(shas: dict) -> list[dict]:
    return [
        {"repo": "org/repo", "number": 1, "agent": "A", "state": "closed", "merged": True, "created_at": "2025-03-01T12:00:00Z", "commits": [shas["p1"]],
         "files": {".github/workflows/ci.yml": {"additions": 6, "deletions": 0, "status": ["modified"]}}, "ack": P.ack_words("add a docs lint job (non-blocking)", "")},
        {"repo": "org/repo", "number": 2, "agent": "B", "state": "closed", "merged": True, "created_at": "2025-03-01T12:00:00Z", "commits": [shas["q1"]],
         "files": {".github/workflows/ci.yml": {"additions": 1, "deletions": 1, "status": ["modified"]}}, "ack": P.ack_words("keep lint from failing", "so the build stays green")},
        {"repo": "org/repo", "number": 3, "agent": "A", "state": "closed", "merged": False, "created_at": "2025-03-01T12:00:00Z", "commits": [shas["r1"]],
         "files": {".github/workflows/nightly.yml": {"additions": 8, "deletions": 0, "status": ["added"]}}, "ack": P.ack_words("add a nightly workflow", "")},
        {"repo": "org/repo", "number": 4, "agent": "B", "state": "closed", "merged": True, "created_at": "2025-03-01T12:00:00Z", "commits": [shas["f1"]],
         "files": {".github/workflows/ci.yml": {"additions": 5, "deletions": 0, "status": ["modified"]}}, "ack": P.ack_words("add a typecheck job", "")},
        {"repo": "org/repo", "number": 5, "agent": "A", "state": "open", "merged": False, "created_at": "2025-03-01T12:00:00Z", "commits": [],
         "files": {".github/workflows/ci.yml": {"additions": 1, "deletions": 0, "status": ["modified"]}}, "ack": P.ack_words("", "")},
        {"repo": "org/repo", "number": 6, "agent": "B", "state": "closed", "merged": False, "created_at": "2025-03-01T12:00:00Z", "commits": [shas["x1"], shas["x2"]],
         "files": {".github/workflows/audit.yml": {"additions": 7, "deletions": 0, "status": ["added"]},
                   ".github/workflows/ci.yml": {"additions": 16, "deletions": 1, "status": ["modified"]}}, "ack": P.ack_words("add an audit workflow", "")},   # ci.yml's rows are the merge's: main's lines
        {"repo": "org/repo", "number": 7, "agent": "A", "state": "closed", "merged": False, "created_at": "2025-03-01T12:00:00Z", "commits": [shas["y1"]],
         "files": {".github/workflows/ci.yml": {"additions": 5, "deletions": 0, "status": ["modified"]}}, "ack": P.ack_words("add a typecheck job (dev)", "")},
    ]


def _run(tree: Path, shas: dict, prs: list[dict], monkeypatch) -> dict:
    monkeypatch.setattr(P, "pull_heads", lambda clone, numbers: {n: (_git(tree, "rev-parse", "--verify", "--quiet", f"refs/pr/{n}") or None)
                                                                              if n != 5 else None for n in numbers})
    return P.repo_agent_prs(tree, "org/repo", prs, {"repo": "org/repo", "tip": shas["f1"]})


def test_the_instrument_is_the_one_the_receipt_names():
    import hashlib
    if INSTRUMENT_SHA256 is None:
        pytest.skip("not yet frozen")
    assert hashlib.sha256((ROOT / "benchmarks" / "harness_mutation" / "agent_prs.py").read_bytes()).hexdigest() == INSTRUMENT_SHA256


def test_the_acknowledgement_words_are_the_swallow6_regex():
    assert P.ack_words("make lint non-blocking", "") == {"title": "non-blocking", "body": None}
    assert P.ack_words("add tests", "this is flaky, skip for now") == {"title": None, "body": "flak"}
    assert P.ack_words(None, None) == {"title": None, "body": None}


def test_each_pull_request_gets_the_base_its_merge_implies(tmp_path, monkeypatch):
    tree, shas = _repo(tmp_path)
    r = _run(tree, shas, _prs(shas), monkeypatch)
    by = {p["number"]: p for p in r["prs"]}
    assert by[1]["base"] == shas["c1"] and by[1]["head"] == shas["p1"] and by[1]["head_in_tip"]              # merged by a merge commit
    assert by[2]["base"] == shas["M"] and by[2]["head"] == shas["q1"] and not by[2]["head_in_tip"]           # squashed
    assert by[3]["base"] == shas["S"] and by[3]["head"] == shas["r1"] and not by[3]["head_in_tip"]           # closed
    assert by[4]["base"] == shas["S"] and by[4]["head"] == shas["f1"] and by[4]["head_in_tip"]               # fast-forwarded
    assert by[6]["base"] == shas["f1"] and by[6]["head"] == shas["x2"] and by[6]["candidates"] == 1           # main merged in: the base is main as last merged, not the fork point
    assert by[7]["base"] == shas["d1"] and by[7]["head"] == shas["y1"] and not by[7]["head_in_tip"]          # into dev: dev's commit, whatever the default branch
    assert by[5]["skip"] == "missing head" and by[5]["head"] is None and not by[5]["audited"]
    assert all(by[n]["base_method"] == "pr-commits" and not by[n]["ambiguous"] for n in (1, 2, 3, 4, 6, 7))
    assert all(by[n]["audited"] and by[n]["head_in_dataset"] and not by[n]["suspect"] for n in (1, 2, 3, 4, 6, 7))


def test_the_gate_on_each_pull_request(tmp_path, monkeypatch):
    tree, shas = _repo(tmp_path)
    r = _run(tree, shas, _prs(shas), monkeypatch)
    by = {p["number"]: p for p in r["prs"]}
    # PR1 brings a step born hidden; at the tip today it is still hidden
    assert by[1]["fires"] and by[1]["new_hidden"] == 1
    d = by[1]["detail"][0]
    assert d["workflow"] == "ci.yml" and d["status"] == "M" and d["git"] == [6, 0] and d["dataset"] == [6, 0]
    assert d["new_hidden"][0]["kind"] == "born hidden" and d["new_hidden"][0]["job"] == "docs" and d["new_hidden"][0]["continue_on_error"]
    assert d["new_hidden"][0]["fix"]["verified_repair"] == "no-continue-on-error" and d["new_hidden"][0]["fix"]["lines_changed"] == 1
    assert d["at_tip"] == [{"job": "docs", "step": "name:Lint docs", "now": "still hidden"}]
    assert by[1]["ack"] == {"title": "non-blocking", "body": None}
    # PR2 hides an existing loud check with `|| true`; the squash left it hidden at the tip
    assert by[2]["fires"] and by[2]["new_hidden"] == 1 and by[2]["still_hidden"] == 1
    d = by[2]["detail"][0]
    assert d["new_hidden"][0]["kind"] == "acquired" and d["new_hidden"][0]["mechanism"] == ["or-true"] and d["new_hidden"][0]["step"] == "name:Lint"
    assert d["new_hidden"][0]["fix"]["verified_repair"] is not None
    assert d["at_tip"] == [{"job": "test", "step": "name:Lint", "now": "still hidden"}]
    # PR3, closed: a new workflow whose one check is born hidden; not merged, so nothing at the tip
    assert by[3]["fires"] and by[3]["new_hidden"] == 1 and not by[3]["merged"]
    assert by[3]["detail"][0]["status"] == "A" and by[3]["detail"][0]["at_tip"] is None
    # PR4 adds a loud check: quiet
    assert not by[4]["fires"] and by[4]["new_hidden"] == 0 and "detail" not in by[4]
    # PR6 adds a loud check and carries main's lines through a merge: quiet; only its own workflow is in git's diff
    assert not by[6]["fires"] and by[6]["changed_in_git"] == 1 and by[6]["changed_in_dataset"] == 2 and not by[6]["suspect"]
    # PR7 into dev: the hidden docs lint is dev's, still hidden at both sides -- not the pull request's
    assert not by[7]["fires"] and by[7]["still_hidden"] == 1 and by[7]["new_hidden"] == 0
    s = r["summary"]
    assert s == {"prs": 7, "audited": 6, "firing": 3, "new_hidden_checks": 3, "missing_head": 1, "no_base": 0, "suspect": 0, "head_moved": 0}


def test_a_base_the_dataset_contradicts_is_suspect(tmp_path, monkeypatch):
    tree, shas = _repo(tmp_path)
    prs = _prs(shas)
    prs[0]["files"][".github/workflows/ci.yml"] = {"additions": 2, "deletions": 0, "status": ["modified"]}   # the dataset says two lines; git says six
    prs[1]["commits"] = ["0" * 40]                                                                            # the head is not a commit the dataset lists
    prs[2]["commits"] = ["1" * 40] * 30                                                                       # a list at the dataset's cap without the head
    r = _run(tree, shas, prs, monkeypatch)
    by = {p["number"]: p for p in r["prs"]}
    assert by[1]["suspect"] and by[1]["suspect_paths"] == [".github/workflows/ci.yml"] and not by[1]["audited"] and by[1]["fires"]
    assert not by[2]["head_in_dataset"] and by[2]["skip"] == "head moved" and not by[2]["audited"]
    assert not by[3]["head_in_dataset"] and by[3]["skip"] == "commit list capped" and not by[3]["audited"]
    assert r["summary"]["audited"] == 3 and r["summary"]["suspect"] == 1 and r["summary"]["head_moved"] == 2


def test_only_the_workflows_the_dataset_names_are_read(tmp_path, monkeypatch):
    tree, shas = _repo(tmp_path)
    prs = _prs(shas)
    prs[0]["files"] = {".github/workflows/other.yml": {"additions": 1, "deletions": 0, "status": ["modified"]}}
    r = _run(tree, shas, prs, monkeypatch)
    by = {p["number"]: p for p in r["prs"]}
    assert by[1]["changed_in_git"] == 0 and not by[1]["fires"] and by[1]["audited"]


def test_no_name_is_written_and_the_instrument_is_deterministic(tmp_path, monkeypatch):
    tree, shas = _repo(tmp_path)
    a = _run(tree, shas, _prs(shas), monkeypatch)
    b = _run(tree, shas, _prs(shas), monkeypatch)
    for r in (a, b):
        r.pop("seconds")
        for p in r["prs"]:
            p.pop("seconds", None)
            assert not any(k in p for k in ("author", "author_name", "author_email", "email", "name", "user", "title", "body"))
    assert a == b
    s = P.summary({"repos": [a], "clone_failures": [], "population_prs": 7})
    assert s["audited"] == 6 and s["firing"] == 3 and s["by_agent"]["A"] == {"audited": 3, "firing": 2, "new_hidden_checks": 2, "merged": 1, "firing_merged": 1}
    assert s["merged_firing"] == 2 and s["ambiguous_base"] == 0 and s["head_in_tip"] == 2


def test_a_missing_parent_is_fetched_by_name_then_deepened_then_skipped(tmp_path, monkeypatch):
    tree, shas = _repo(tmp_path)
    calls = []
    monkeypatch.setattr(P, "deepen", lambda clone, since=P.DEEPEN: calls.append(since))
    monkeypatch.setattr(P, "_present", lambda clone, sha: sha != shas["c1"])            # as if PR1's parent were behind the boundary and not fetchable
    fetched = []
    real = P.H._git
    monkeypatch.setattr(P.H, "_git", lambda clone, args, **kw: (fetched.append(args) if args[:2] == ["fetch", "--quiet"] else None) or real(clone, args, **kw))
    r = _run(tree, shas, _prs(shas), monkeypatch)
    by = {p["number"]: p for p in r["prs"]}
    assert any(a[-1] == shas["c1"] for a in fetched)                                       # asked for by name
    assert calls == [P.DEEPEN] and r["deepened"]                                            # then the branch deepened once
    assert by[1]["at_boundary"] and by[1]["skip"] == "no base" and not by[1]["audited"]      # then skipped
    assert by[6]["at_boundary"] and by[6]["skip"] == "no base"                                  # PR6's fork point is that same commit
    assert all(not by[n]["at_boundary"] and by[n]["audited"] for n in (2, 3, 4, 7))
    assert r["summary"]["no_base"] == 2 and r["summary"]["audited"] == 4

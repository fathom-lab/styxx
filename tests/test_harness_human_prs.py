# -*- coding: utf-8 -*-
"""SWALLOW-10's instrument: one pipeline for the agents' pull requests and everyone else's --
group by the dataset or SWALLOW-8's rule on the head commit, BASE by the closest-branch rule,
touching by git's diff, merged by a merge commit or GitHub's squash subject. The tests build a
local bare remote with branches and refs/pull/N/head (a merge commit, a squash, a fast-forward,
one closed, one that merged main into itself, one into `dev`, and four pull requests by people
and bots) and hold the sample, the groups, the bases, the merge signal, the gate, the agreement
with SWALLOW-9's bases, determinism, and that no name is written."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import human_prs as X  # noqa: E402
from tests.test_harness_agent_prs import CI, CI_PR1, NIGHTLY, _write  # noqa: E402

# frozen at the sha256 the SWALLOW-10 receipt names (papers/harness/swallow10_receipt.json.gz); a change needs a new receipt, not a new pin
INSTRUMENT_SHA256 = "7d764d5db800da3986a9b5ea4f5d620addd3a794d0bc037fa6b9c6eeebe3afe7"

PERSON = dict(GIT_AUTHOR_NAME="Pat Example", GIT_AUTHOR_EMAIL="pat@example.org", GIT_COMMITTER_NAME="Pat Example", GIT_COMMITTER_EMAIL="pat@example.org")
BOT = dict(GIT_AUTHOR_NAME="dependabot[bot]", GIT_AUTHOR_EMAIL="49699333+dependabot[bot]@users.noreply.github.com", GIT_COMMITTER_NAME="GitHub", GIT_COMMITTER_EMAIL="noreply@github.com")


def _git(tree: Path, *args: str, env: dict | None = None) -> str:
    return subprocess.run(["git", "-C", str(tree), *args], check=True, capture_output=True, text=True, env=env).stdout.strip()


def _remote(tmp_path: Path) -> tuple[Path, Path, dict]:
    """A working repository, then a bare remote of it with branches and refs/pull/N/head, then a
    blobless clone of the remote's default branch -- what the instrument starts from."""
    tree = tmp_path / "work"
    (tree / ".github" / "workflows").mkdir(parents=True)
    tick = [1741000000]

    def env(who: dict = PERSON) -> dict:
        tick[0] += 3600
        d = f"{tick[0]} +0000"
        return dict(os.environ, **who, GIT_AUTHOR_DATE=d, GIT_COMMITTER_DATE=d)

    subprocess.run(["git", "init", "-q", "-b", "main", str(tree)], check=True, env=env())

    def commit(msg: str, who: dict = PERSON) -> str:
        e = env(who)
        _git(tree, "add", "-A", env=e)
        _git(tree, "commit", "-q", "--allow-empty", "-m", msg, env=e)
        return _git(tree, "rev-parse", "HEAD")

    shas = {}
    _write(tree, "ci.yml", CI)
    shas["c0"] = commit("ci: add the pipeline")
    (tree / "README.md").write_text("hello\n")
    shas["c1"] = commit("docs: readme")
    # PR1 (agent): off c1, merged by a merge commit -- brings a hidden check
    _git(tree, "checkout", "-q", "-b", "pr1", env=env())
    _write(tree, "ci.yml", CI_PR1)
    shas["p1"] = commit("ci: add a docs lint job (non-blocking)")
    _git(tree, "checkout", "-q", "main", env=env())
    _git(tree, "merge", "-q", "--no-ff", "-m", "Merge pull request #1 from org/pr1", "pr1", env=env())
    shas["M"] = _git(tree, "rev-parse", "HEAD")
    # PR2 (agent): off M, squash-merged -- hides an existing check
    _git(tree, "checkout", "-q", "-b", "pr2", env=env())
    _write(tree, "ci.yml", CI_PR1.replace("run: npm run lint", "run: npm run lint || true"))
    shas["q1"] = commit("ci: keep lint from failing the build")
    _git(tree, "checkout", "-q", "main", env=env())
    _git(tree, "merge", "-q", "--squash", "pr2", env=env())
    shas["S"] = commit("ci: keep lint from failing the build (#2)")
    # PR3 (agent): off S, closed -- a new workflow born hidden
    _git(tree, "checkout", "-q", "-b", "pr3", env=env())
    _write(tree, "nightly.yml", NIGHTLY)
    shas["r1"] = commit("ci: add a nightly workflow")
    _git(tree, "checkout", "-q", "main", env=env())
    # PR4 (agent): off S, fast-forwarded -- a loud check
    _git(tree, "checkout", "-q", "-b", "pr4", env=env())
    _write(tree, "ci.yml", (tree / ".github/workflows/ci.yml").read_text() + """      typecheck:
        runs-on: ubuntu-latest
        steps:
          - name: Typecheck
            run: npx tsc --noEmit
""")
    shas["f1"] = commit("ci: add a typecheck job")
    _git(tree, "checkout", "-q", "main", env=env())
    _git(tree, "merge", "-q", "--ff-only", "pr4", env=env())
    # PR8 (a person): off f1, closed -- brings a hidden check
    _git(tree, "checkout", "-q", "-b", "pr8", env=env())
    _write(tree, "release.yml", """
    on: [push]
    jobs:
      release:
        runs-on: ubuntu-latest
        steps:
          - name: Verify tag
            run: git describe --exact-match --tags || true
""")
    shas["h1"] = commit("ci: release workflow (verify tag is best-effort)")
    _git(tree, "checkout", "-q", "main", env=env())
    # PR9 (dependabot): off f1, bumps an action -- quiet
    _git(tree, "checkout", "-q", "-b", "pr9", env=env(BOT))
    _write(tree, "ci.yml", (tree / ".github/workflows/ci.yml").read_text().replace("on: [push]", "on: [push, pull_request]"))
    shas["b1"] = commit("Bump actions/checkout from 3 to 4", BOT)
    _git(tree, "checkout", "-q", "main", env=env())
    # PR10 (a person): off f1, squash-merged as "(#10)" -- touches no workflow
    _git(tree, "checkout", "-q", "-b", "pr10", env=env())
    (tree / "README.md").write_text("hello world\n")
    shas["h2"] = commit("docs: hello world")
    _git(tree, "checkout", "-q", "main", env=env())
    _git(tree, "merge", "-q", "--squash", "pr10", env=env())
    shas["S10"] = commit("docs: hello world (#10)")
    # PR11 (a person's commit carrying an agent's trailer): off S10, closed -- quiet
    _git(tree, "checkout", "-q", "-b", "pr11", env=env())
    (tree / "README.md").write_text("hello world!\n")
    shas["s1"] = commit("docs: punctuation\n\nCo-Authored-By: Claude Opus 4 <noreply@anthropic.com>")
    _git(tree, "checkout", "-q", "main", env=env())
    # dev: off c1 with the hidden docs lint; PR7 (agent) targets dev with a loud check
    _git(tree, "checkout", "-q", "-b", "dev", shas["c1"], env=env())
    _write(tree, "ci.yml", CI_PR1)
    shas["d1"] = commit("ci: docs lint on dev (non-blocking)")
    _git(tree, "checkout", "-q", "-b", "pr7", env=env())
    _write(tree, "ci.yml", CI_PR1 + """      typecheck:
        runs-on: ubuntu-latest
        steps:
          - name: Typecheck
            run: npx tsc --noEmit
""")
    shas["y1"] = commit("ci: add a typecheck job (dev)")
    _git(tree, "checkout", "-q", "main", env=env())
    # PR6 (agent): off c1, adds a loud workflow, then merges main into itself; closed
    _git(tree, "checkout", "-q", "-b", "pr6", shas["c1"], env=env())
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
    _git(tree, "merge", "-q", "-m", "Merge branch 'main' into pr6", "main", env=env())
    shas["x2"] = _git(tree, "rev-parse", "HEAD")
    _git(tree, "checkout", "-q", "main", env=env())
    # PR12 (a person): off S10, rebase-merged -- main gets a new sha with the same subject and author date
    _git(tree, "checkout", "-q", "-b", "pr12", shas["S10"], env=env())
    (tree / "NOTES.md").write_text("notes\n")
    shas["h3"] = commit("docs: notes")
    _git(tree, "checkout", "-q", "main", env=env())
    _git(tree, "cherry-pick", shas["h3"], env=env())
    shas["R12"] = _git(tree, "rev-parse", "HEAD")
    # PR13 (a person): an orphan history -- no base anywhere
    _git(tree, "checkout", "-q", "--orphan", "pr13", env=env())
    _git(tree, "rm", "-rfq", ".", env=env())
    (tree / "ORPHAN.md").write_text("orphan\n")
    shas["o1"] = commit("docs: orphan")
    _git(tree, "checkout", "-q", "main", env=env())
    shas["tip"] = _git(tree, "rev-parse", "HEAD")

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    _git(remote, "config", "uploadpack.allowFilter", "true")
    _git(tree, "push", "-q", str(remote), "main", "dev", "pr3", "pr6", "pr8", "pr11")      # the closed pull requests' branches still exist; pr1/pr2/pr4/pr7/pr9/pr10 were deleted
    heads = {1: "p1", 2: "q1", 3: "r1", 4: "f1", 6: "x2", 7: "y1", 8: "h1", 9: "b1", 10: "h2", 11: "s1", 12: "h3", 13: "o1"}
    for n, k in heads.items():
        _git(tree, "push", "-q", str(remote), f"{shas[k]}:refs/pull/{n}/head")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", "--single-branch", f"file://{remote}", str(clone)], check=True)
    return clone, remote, shas


def _spec() -> dict:
    agents = {1: "A", 2: "B", 3: "A", 4: "B", 6: "B", 7: "A", 5: "A"}
    merged = {1: True, 2: True, 3: False, 4: True, 6: False, 7: False, 5: False}
    return {"repo": "org/repo", "span": [1, 13], "cap_human": 150, "seed": 10,
            "agent_prs": {str(n): {"agent": a, "merged": merged[n], "state": "closed", "created_at": "2025-03-01T00:00:00Z", "in_population9": n != 6} for n, a in agents.items()},
            "agent_fetch": [1, 2, 3, 4, 5, 6, 7]}


def _run(clone: Path, receipt9_prs: dict | None = None) -> dict:
    tip = _git(clone, "rev-parse", "HEAD")
    return X.repo_prs(clone, _spec(), {"repo": "org/repo", "tip": tip}, receipt9_prs)


def test_the_instrument_is_the_one_the_receipt_names():
    import hashlib
    if INSTRUMENT_SHA256 is None:
        pytest.skip("not yet frozen")
    assert hashlib.sha256((ROOT / "benchmarks" / "harness_mutation" / "human_prs.py").read_bytes()).hexdigest() == INSTRUMENT_SHA256


def test_the_human_sample_is_the_non_agent_numbers_in_the_span_seeded():
    assert X.human_sample("org/repo", [1, 2, 3, 8, 9, 10, 11, 12, 40], [1, 11], {1, 2, 3}) == [8, 9, 10, 11]
    a = X.human_sample("org/repo", list(range(1, 1000)), [1, 999], set(), cap=5)
    assert a == X.human_sample("org/repo", list(range(1, 1000)), [1, 999], set(), cap=5) and len(a) == 5 and a == sorted(a)
    assert X.human_sample("other/repo", list(range(1, 1000)), [1, 999], set(), cap=5) != a


def test_groups_bases_merges_and_the_gate(tmp_path):
    clone, remote, shas = _remote(tmp_path)
    r = _run(clone)
    by = {p["number"]: p for p in r["prs"]}
    assert r["human_sampled"] == 6 and r["non_agent_in_span"] == 6 and r["branches"] == 6 and r["remote_pull_requests"] == 12
    assert sorted(by) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # groups: the dataset's agents, then SWALLOW-8's rule on the head commit
    assert {n: by[n]["group"] for n in by} == {1: "A", 2: "B", 3: "A", 4: "B", 5: "A", 6: "B", 7: "A", 8: "human", 9: "automation", 10: "human", 11: "agent-signed",
                                              12: "human", 13: "human"}
    assert by[13]["skip"] == "no base" and by[13]["base"] is None and not by[13]["audited"] and not r["deepened"]     # nothing shallow to deepen
    assert by[11]["signal"] == "co-authored-by" and by[9]["signal"] == "bot-author" and by[8]["signal"] is None
    assert by[5]["skip"] == "missing head" and not by[5]["audited"]
    # bases: the merge commit's first parent, else the closest branch; a pull request into dev gets dev's commit
    assert by[1]["base_method"] == "merge-commit" and by[1]["base"] == shas["c1"] and by[1]["merge_commit"] == shas["M"]
    assert by[2]["base_method"] == "closest-branch" and by[2]["base"] == shas["M"]
    assert by[3]["base_method"] == "closest-branch" and by[3]["base"] == shas["S"] and by[3]["excluded_branches"] == 1        # its branch still exists
    assert by[4]["base_method"] == "closest-branch" and by[4]["base"] == shas["S"] and by[4]["excluded_branches"] >= 1        # fast-forwarded: main contains it
    assert by[6]["base"] == shas["S10"] and by[7]["base"] == shas["d1"] and by[8]["base"] == shas["f1"] and by[9]["base"] == shas["f1"]   # PR6 merged main as it was then: S10
    assert by[10]["base"] == shas["f1"] and by[11]["base"] == shas["S10"] and by[12]["base"] == shas["S10"]
    # merged: a merge commit, GitHub's squash subject, or a rebase's same subject and author date; a fast-forward leaves none (a floor)
    assert {n: by[n]["merged_heuristic"] for n in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12)} == {1: True, 2: True, 3: False, 4: False, 6: False, 7: False, 8: False, 9: False, 10: True, 11: False, 12: True}
    assert by[1]["merged_by_merge_commit"] and not by[2]["merged_by_merge_commit"] and by[2]["merged_by_subject"] and by[12]["merged_by_rebase"] and not by[12]["merged_by_subject"]
    # touching and the gate
    assert {n: by[n]["touching"] for n in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11)} == {1: True, 2: True, 3: True, 4: True, 6: True, 7: True, 8: True, 9: True, 10: False, 11: False}
    assert {n: by[n]["fires"] for n in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11)} == {1: True, 2: True, 3: True, 4: False, 6: False, 7: False, 8: True, 9: False, 10: False, 11: False}
    assert by[8]["new_hidden"] == 1 and by[8]["detail"][0]["new_hidden"][0]["kind"] == "born hidden" and by[8]["detail"][0]["new_hidden"][0]["fix"]["verified_repair"] is not None
    assert by[7]["still_hidden"] == 1 and by[7]["new_hidden"] == 0
    s = r["summary"]
    assert s == {"prs": 13, "audited": 11, "touching": 8, "firing": 4, "human_touching": 1, "human_firing": 1, "missing_head": 1, "no_base": 1}


def test_the_agreement_with_swallow9_and_the_merge_signal_against_the_dataset(tmp_path):
    clone, remote, shas = _remote(tmp_path)
    r9 = {1: {"base": shas["c1"], "audited": True, "fires": True}, 2: {"base": shas["M"], "audited": True, "fires": True},
          3: {"base": shas["S"], "audited": True, "fires": True}, 4: {"base": shas["c1"], "audited": True, "fires": False},    # one disagreement, planted
          7: {"base": shas["d1"], "audited": False, "fires": False}}
    r = _run(clone, r9)
    by = {p["number"]: p for p in r["prs"]}
    assert by[1]["base_agrees9"] and by[2]["base_agrees9"] and by[3]["base_agrees9"] and by[4]["base_agrees9"] is False
    assert by[1]["fires9"] is True and by[7]["fires9"] is None and "base9" not in by[6]
    s = X.summary({"repos": [r], "clone_failures": []})
    assert s["base_agreement9"] == {"compared": 5, "agree": 4} and s["fires_agreement9"] == {"compared": 4, "agree": 4}   # PR7's base compares though SWALLOW-9 did not audit it
    assert s["merged_heuristic_on_dataset_truth"] == {"merged": 3, "recalled": 2, "not_merged": 3, "false_positive": 0,
                                                      "recalled_by": {"merged_by_merge_commit": 1, "merged_by_subject": 2, "merged_by_rebase": 0}}   # the fast-forward is the miss; PR1's merge subject counts twice
    assert s["by_group"]["human"] == {"audited": 3, "touching": 1, "firing": 1, "new_hidden_checks": 1, "merged_heuristic_touching": 0}
    assert s["by_group"]["automation"]["touching"] == 1 and s["by_group"]["agent-signed"]["audited"] == 1
    assert list(s["by_group"]) == ["human", "automation", "agent-signed", "A", "B"]


def test_no_name_is_written_and_the_instrument_is_deterministic(tmp_path):
    clone, remote, shas = _remote(tmp_path)
    a = _run(clone)
    b = _run(clone)
    for r in (a, b):
        r.pop("seconds")
        for p in r["prs"]:
            p.pop("seconds", None)
            assert not any(k in p for k in ("author", "author_name", "author_email", "email", "name", "user", "title", "body", "subject"))
            assert "Pat" not in str(p) and "example.org" not in str(p)
    assert a == b


def test_a_pull_request_without_a_base_deepens_the_branches_once_then_is_skipped(tmp_path, monkeypatch):
    clone, remote, shas = _remote(tmp_path)
    monkeypatch.setattr(X.P, "shallow_set", lambda c: {shas["c0"]})     # as if the clone were shallow (a fetch from the local remote would unshallow a real entry)
    r = _run(clone)
    by = {p["number"]: p for p in r["prs"]}
    assert r["deepened"] and by[13]["skip"] == "no base" and by[8]["audited"] and by[8]["fires"]
    assert r["summary"]["no_base"] == 1 and r["summary"]["audited"] == 11

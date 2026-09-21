# -*- coding: utf-8 -*-
"""SWALLOW-7's instrument: the gate between a BASE and a HEAD -- the workflows that changed, every
step matched across the two revisions, what HEAD hides that BASE did not (with the verified
repair), what it made loud or removed, what was hidden on both sides. The tests run the gate at
every commit of the scripted history and hold every outcome, the exit rule, and determinism."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import differential as D  # noqa: E402
from tests.test_harness_history import _repo  # noqa: E402


def _shas(tree: Path) -> list[str]:
    return subprocess.run(["git", "-C", str(tree), "log", "--reverse", "--format=%H"], capture_output=True, text=True).stdout.split()


def _flat(a: dict, key: str) -> list[tuple]:
    return [(w["workflow"], x["job"], x["step"], x["kind"]) for w in a["workflows"] for x in w[key]]


def test_the_gate_at_every_commit_of_the_scripted_history(tmp_path):
    tree = _repo(tmp_path)
    shas = _shas(tree)
    assert len(shas) == 9
    readers: dict = {}
    got = [D.audit_commit_or_root(tree, shas[n - 1] if n else None, sha, readers) for n, sha in enumerate(shas)]
    # c1, the root: two checks born hidden, each with a verified repair; the loud ones and the unread one do not fire
    a = got[0]
    assert a["fires"] and a["new_hidden"] == 2 and a["removed_hidden"] == 0 and a["still_hidden"] == 0 and a["hidden_after_unread"] == 0
    assert _flat(a, "new_hidden") == [("ci.yml", "test", "name:Lint colors", "born hidden"), ("ci.yml", "test", "name:Lint markdown", "born hidden")]
    fixes = {x["step"]: x["fix"] for w in a["workflows"] for x in w["new_hidden"]}
    assert fixes["name:Lint colors"]["verified_repair"] == "strict-shell" and fixes["name:Lint colors"]["stage"] == "swallow-4"
    assert "-        run: pnpm lint:colors || true" in fixes["name:Lint colors"]["diff"] and "+          pnpm lint:colors" in fixes["name:Lint colors"]["diff"]
    # c2: an existing loud check made hidden by continue-on-error -- the gate fires, the repair is one line
    a = got[1]
    assert a["fires"] and _flat(a, "new_hidden") == [("ci.yml", "test", "name:Lint (non-blocking)", "acquired")]
    (x,) = [x for w in a["workflows"] for x in w["new_hidden"]]
    assert x["mechanism"] == ["continue-on-error"] and x["verdict_before"] == "RED" and x["continue_on_error"]
    assert x["fix"]["verified_repair"] == "no-continue-on-error" and x["fix"]["lines_changed"] == 1
    assert [s["step"] for w in a["workflows"] for s in w["still_hidden"]] == ["name:Lint colors", "name:Lint markdown"]
    # c3: a wild repair -- removed hidden, mechanism read, no firing
    a = got[2]
    assert not a["fires"] and _flat(a, "removed_hidden") == [("ci.yml", "test", "name:Lint colors", "repaired")]
    assert [x["mechanism"] for w in a["workflows"] for x in w["removed_hidden"]] == [["or-true"]]
    # c4: a rename -- matched by its script, nothing fires, nothing is removed
    a = got[3]
    assert not a["fires"] and a["removed_hidden"] == 0 and a["new_hidden"] == 0 and a["still_hidden"] == 2
    # c5: a hidden check removed with its step
    a = got[4]
    assert not a["fires"] and _flat(a, "removed_hidden") == [("ci.yml", "test", "name:Lint markdown", "removed")] and a["still_hidden"] == 1
    # c6: a workflow added with one loud and one hidden check -- the gate fires on the hidden one
    assert got[5]["fires"] and _flat(got[5], "new_hidden") == [("release.yml", "release", "name:Lint release notes", "born hidden")]
    # c7: renamed (read at both paths): nothing new, the hidden one is still hidden; c8: repaired under the new name; c9: deleted
    assert not got[6]["fires"] and got[6]["removed_hidden"] == 0 and got[6]["still_hidden"] == 1 and [(w["status"], w["path"]) for w in got[6]["workflows"]] == [("R", ".github/workflows/publish.yml")]
    assert got[6]["workflows"][0]["base_unparseable"] is None and got[6]["workflows"][0]["head_unparseable"] is None
    assert not got[7]["fires"] and _flat(got[7], "removed_hidden") == [("publish.yml", "release", "name:Lint release notes", "repaired")]
    assert not got[8]["fires"] and got[8]["removed_hidden"] == 0 and [w["status"] for w in got[8]["workflows"]] == ["D"]
    # the whole history at once: HEAD against the first commit
    t = D.tree_differential(tree, shas[0])
    assert t["merge_base"] == shas[0] and t["fires"] and _flat(t, "new_hidden") == [("ci.yml", "test", "name:Lint (non-blocking)", "acquired")]
    assert _flat(t, "removed_hidden") == [("ci.yml", "test", "name:Lint colors", "repaired"), ("ci.yml", "test", "name:Lint markdown", "removed")]
    # a base ahead of HEAD's history is read at the merge-base
    assert D.tree_differential(tree, shas[-1])["fires"] is False


def test_the_population_reading_of_one_repository_is_the_gate_at_every_commit(tmp_path):
    tree = _repo(tmp_path)
    shas = _shas(tree)
    r = D.repo_differential(tree, shas[-1], "fixture", sample_every=3)
    assert r["mainline_commits"] == 9 and not r["capped"]
    assert [c["fires"] for c in r["commits"]] == [True, True, False, False, False, True, False, False, False]
    assert [c["root"] for c in r["commits"]] == [True] + [False] * 8
    assert [c["new_hidden"] for c in r["commits"]] == [2, 1, 0, 0, 0, 1, 0, 0, 0] and [c["removed_hidden"] for c in r["commits"]] == [0, 0, 1, 0, 1, 0, 0, 1, 0]
    (d,) = r["commits"][1]["detail"]                                       # the firing carries the step and its verified repair
    assert d["workflow"] == "ci.yml" and d["status"] == "M" and d["removed_hidden"] == [] and d["hidden_after_unread"] == []
    assert [(x["step"], x["kind"], x["fix"]["verified_repair"]) for x in d["new_hidden"]] == [("name:Lint (non-blocking)", "acquired", "no-continue-on-error")]
    assert "detail" not in r["commits"][3]
    assert [s["sha"] for s in r["sample"]] == [shas[3], shas[6]]            # every third commit, roots excepted
    assert D.summary(r) == {"mainline_commits": 9, "firing_commits": 3, "new_hidden_checks": 4, "removed_hidden_checks": 3, "hidden_after_unread": 0,
                            "batch_firings_3_or_more": 0, "sample_n": 2, "sample_median_seconds": D.summary(r)["sample_median_seconds"]}


def test_the_gate_is_deterministic(tmp_path):
    tree = _repo(tmp_path)
    shas = _shas(tree)
    strip = lambda a: [(w["workflow"], w["status"], _flat(a, "new_hidden"), _flat(a, "removed_hidden"), sorted(x["step"] for x in w["still_hidden"]),  # noqa: E731
                        [(x["fix"]["verified_repair"], x["fix"]["diff"]) for x in w["new_hidden"]]) for w in a["workflows"]]
    for n, sha in enumerate(shas):
        a, b = D.audit_commit_or_root(tree, shas[n - 1] if n else None, sha, {}), D.audit_commit_or_root(tree, shas[n - 1] if n else None, sha, {})
        assert strip(a) == strip(b)


def test_this_repository_hides_nothing_new_against_its_own_first_commit():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rec = D.audit_commit(ROOT, head, head, readers={})
    assert rec["fires"] is False and rec["workflows"] == []

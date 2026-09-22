# -*- coding: utf-8 -*-
"""SWALLOW-12's instrument: the pull request made for the one click, and the plan -- the Action run
offline on that change as GitHub runs it, with the review API answered by SWALLOW-11's placement
rule. The tests hold the fixtures to the shapes they stand for, the plan to what the
preregistration says it predicts, the splice to every byte, and the plan to itself."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import live_click as L  # noqa: E402

# frozen at the sha256 the SWALLOW-12 preregistration names; a change needs a new preregistration, not a new pin
INSTRUMENT_SHA256 = "83be210029e8c647f6e6f20eafafe4b0de6b97002d7cc40cbaf3a792c520577c"


def test_the_instrument_is_the_one_the_preregistration_names():
    import hashlib
    if INSTRUMENT_SHA256 is None:
        pytest.skip("not yet frozen")
    assert hashlib.sha256((ROOT / "benchmarks" / "harness_mutation" / "live_click.py").read_bytes()).hexdigest() == INSTRUMENT_SHA256


def test_every_fixture_runs_on_workflow_dispatch_only():
    import yaml
    for name, (_, base, head) in L.FIXTURES.items():
        for text in (base, head):
            doc = yaml.safe_load(text)
            assert doc[True] == {"workflow_dispatch": None}, name          # PyYAML reads `on:` as True
    assert "pull_request:" in L.LIVE_WORKFLOW and "suggest: true" in L.LIVE_WORKFLOW and "pull-requests: write" in L.LIVE_WORKFLOW


def test_the_splice_keeps_every_other_byte():
    text = b"a\nb\nc\n"
    assert L.splice(text, [{"start_line": 2, "line": 2, "lines": []}]) == b"a\nc\n"
    assert L.splice(text, [{"start_line": 2, "line": 3, "lines": ["x", "y", "z"]}]) == b"a\nx\ny\nz\n"
    assert L.splice(b"a\nb", [{"start_line": 2, "line": 2, "lines": ["b1", "b2"]}]) == b"a\nb1\nb2"      # no final newline stays none
    assert L.splice(text, [{"start_line": 1, "line": 1, "lines": ["A"]}, {"start_line": 3, "line": 3, "lines": ["C"]}]) == b"A\nb\nC\n"


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return L.plan(tmp_path_factory.mktemp("plan"))


def test_the_plan_is_what_the_preregistration_predicts(plan):
    first, again, applied = plan["runs"]["first"], plan["runs"]["again"], plan["runs"]["applied"]
    # nineteen checks: sixteen placed, two refused by the rule, one with no repair
    assert first["rc"] == 1 and first["new_hidden"] == 19
    assert sorted(c["status"] for c in first["posts"]) == [201] * 16 + [422] * 2
    refused = sorted(c["path"].rsplit("/", 1)[-1] for c in first["posts"] if c["status"] == 422)
    assert refused == ["s12-12-outside-diff.yml", "s12-12b-span-leaves-hunk.yml"]
    assert "s12-13-no-repair.yml name:Tests: no verified repair" in first["suggest_line"]
    # GitHub keeps ten annotations of each level from one step: ten errors, nine warnings, the diff first
    assert first["annotation_levels"] == {"error": 10, "warning": 9}
    assert [a["path"].rsplit("/", 1)[-1] for a in first["annotations"][-2:]] == ["s12-12-outside-diff.yml", "s12-12b-span-leaves-hunk.yml"]
    # again on the same head: nothing new, the two refusals again
    assert again["new_hidden"] == 19 and [c["status"] for c in again["posts"]] == [422, 422]
    assert again["suggest_line"].startswith("styxx ci-audit: 0 suggestions posted; 16 already on this pull request;")
    # every placed suggestion applied: only the three controls remain
    assert applied["rc"] == 1 and applied["new_hidden"] == 3 and applied["annotation_levels"] == {"error": 3}
    assert sorted(c["path"].rsplit("/", 1)[-1] for c in applied["checks"]) == ["s12-12-outside-diff.yml", "s12-12b-span-leaves-hunk.yml", "s12-13-no-repair.yml"]
    # fifteen files carry the sixteen suggestions; one of them ends without a newline and keeps it that way
    ef = plan["expected_files"]
    assert len(ef) == 15 and sum(e["suggestions"] for e in ef.values()) == 16
    assert [p.rsplit("/", 1)[-1] for p, e in ef.items() if not e["final_newline"]] == ["s12-08-strict-eof-no-newline.yml"]


def test_each_fixture_is_the_shape_it_stands_for(plan):
    posts = {c["path"].rsplit("/", 1)[-1]: c for c in plan["runs"]["first"]["posts"]}
    assert posts["s12-01-coe-step-acquired.yml"]["lines"] == [] and posts["s12-04-coe-step-eof.yml"]["lines"] == []   # empty suggestions: the line goes
    assert posts["s12-05-strict-one-line.yml"]["lines"][1].strip() == "set -eo pipefail"
    assert posts["s12-08-strict-eof-no-newline.yml"]["line"] == len(L.FIXTURES["s12-08-strict-eof-no-newline.yml"][2].splitlines())
    assert posts["s12-12b-span-leaves-hunk.yml"]["start_line"] < posts["s12-12b-span-leaves-hunk.yml"]["line"]
    kinds = {(c["path"].rsplit("/", 1)[-1], c["step"]): (c["kind"], c["repair"]) for c in plan["runs"]["first"]["checks"]}
    assert kinds[("s12-01-coe-step-acquired.yml", "name:Unit tests")] == ("acquired", "no-continue-on-error")
    assert kinds[("s12-05b-strict-acquired.yml", "name:Lint")] == ("acquired", "strict-shell")
    assert kinds[("s12-09-no-default.yml", "name:Typecheck")][1] == "no-default"
    assert kinds[("s12-10-guard.yml", "name:No focused tests")][1] == "guard-status"
    assert kinds[("s12-11-both.yml", "name:Tests")][1] == "both"
    assert kinds[("s12-13-no-repair.yml", "name:Tests")][1] is None


def _without_scratch_shas(p: dict) -> dict:
    """The plan with the ids of its scratch commits masked. The job summaries it stores name the
    offline repository's base and test-merge commits, and those ids depend on the machine's git
    configuration: the machine that froze the plan signs its commits (commit.gpgsign), the CI runner
    does not. Nothing the receipt compares holds a commit id."""
    import copy
    import re
    q = copy.deepcopy(p)
    for run in q["runs"].values():
        run["summary"] = re.sub(r"`[0-9a-f]{8}`", "`<commit>`", run["summary"])
    return q


def test_the_plan_is_deterministic_and_is_the_frozen_one(plan, tmp_path):
    import json
    assert L.plan(tmp_path) == plan
    frozen = json.loads((ROOT / "papers" / "harness" / "swallow12_plan.json").read_text(encoding="utf-8"))
    assert _without_scratch_shas(frozen) == _without_scratch_shas(plan)


def test_the_base_workflow_is_the_repositorys_gate():
    """Frozen in the instrument, the base's ci-audit.yml is the repository's gate as the live base branch holds it."""
    assert L.BASE_WORKFLOW == (ROOT / ".github" / "workflows" / "ci-audit.yml").read_text(encoding="utf-8")

# -*- coding: utf-8 -*-
"""SWALLOW-11's instrument: every firing pair of the SWALLOW-7, -9 and -10 receipts re-read by the
product, and every newly hidden check replayed as the Action would report it -- located, visible
in the change's diff, the verified repair rebuilt, the suggestion reproducing it, and one click
away only when its lines sit inside the change's diff. The tests hold the population rule, a
change whose two fixes are one click away, a change whose fix lies outside its own diff, and
determinism."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import one_click as O  # noqa: E402
from tests.test_ciaudit_action import CI_PR7, _github, _line  # noqa: E402

# frozen at the sha256 the SWALLOW-11 receipt names (papers/harness/swallow11_receipt.json.gz); a change needs a new receipt, not a new pin
INSTRUMENT_SHA256 = "047a11235ce35f02bb3c1e2e467857bf2c1e54d5cbd0b0fc9eede8dc6a6ff807"
# the Action the replay measured. The scored receipt ran action.py a9615d9b...; after scoring, one text-mode subprocess
# call (readable()'s `git diff --quiet`) gained its encoding pin, and the replay was re-run with that file, c642493d...:
# swallow11_rerun_receipt.json.gz names it and is the scored receipt in every pair and check (held below). The file
# itself moves on (SWALLOW-12's annotation levels); what the replay called is pinned by its source, function by function
ACTION_SHA256_RERUN = "c642493df6e2c00d483debd8e28bc3c8a104528df345dd3703881de9b3952c3a"
ACTION_SHA256_SCORED = "a9615d9bb909b4435fd301f4aa6496320af81828177766bf7c4718ee3c25a76f"
REPLAYED_FUNCTIONS_SHA256 = {
    "git": "c3d007d72d97b4d09572941c115a28ee21f5d98517d5d75f1652e90f35335248",
    "have": "5d71f0cf18a564227f119b1c61d14fd6bf1dc68ad60a8bd41f3c76d30d36900b",
    "readable": "d6d2b051d550e2fb70c3f94034ee6b8bfa2e8685478905be9838f07b97e9f83b",
    "positions": "ff4a3a5f4cf96b5268712894b10be0d85781a68e655bf7409ec7397bc0a8bd7b",
    "target": "15f9a48d09edbdd84c0add904f0c6cacd5f840bb5419f6322f47d48de3722f94",
    "repaired_text": "423fcaef744b8682ad14e56df028a5798f480b5ef41f048218066699301f6965",
    "suggestion": "230bfff4ce51b8c802c41af9616935df44be6effad6d4897420e532b603142a2",
    "apply_suggestion": "fca6815124fe9112d2f24e5b44fe87383328519f5d4d0e329a3e79e110470abc",
    "hunks": "90d2a9ed14ac50c66de3ad4171666fdf87aea3286bab3bfd95ab44923802de00",
    "within": "3e8e7b5c4a58634bd1582e05bd94bf55ad7db0a6a0efec17d0a152ceaf14900f",
}
HARNESS = ROOT / "papers" / "harness"


def _receipt(name: str) -> dict:
    import gzip
    import json
    return json.loads(gzip.decompress((HARNESS / name).read_bytes()).decode("utf-8"))


def _function_sources(path: Path) -> dict:
    import ast
    import hashlib
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    return {n.name: hashlib.sha256("".join(lines[n.lineno - 1:n.end_lineno]).encode("utf-8")).hexdigest()
            for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


def test_the_instrument_and_the_functions_it_called_are_the_ones_a_receipt_names():
    import hashlib
    if INSTRUMENT_SHA256 is None:
        pytest.skip("not yet frozen")
    assert hashlib.sha256((ROOT / "benchmarks" / "harness_mutation" / "one_click.py").read_bytes()).hexdigest() == INSTRUMENT_SHA256
    shipped = _function_sources(ROOT / "styxx" / "ciaudit" / "action.py")
    assert {k: shipped.get(k) for k in REPLAYED_FUNCTIONS_SHA256} == REPLAYED_FUNCTIONS_SHA256


def test_the_rerun_with_the_shipped_action_is_the_scored_receipt():
    """The re-run's action.py differs from the one the scored receipt ran by an encoding pin; the
    replay re-run with it must be that receipt in every pair and every check."""
    scored, rerun = _receipt("swallow11_receipt.json.gz"), _receipt("swallow11_rerun_receipt.json.gz")
    assert scored["action_sha256"] == ACTION_SHA256_SCORED and rerun["action_sha256"] == ACTION_SHA256_RERUN
    assert rerun["instrument_sha256"] == scored["instrument_sha256"] == INSTRUMENT_SHA256
    assert rerun["differential_living_sha256"] == scored["differential_living_sha256"]
    assert rerun["sources_sha256"] == scored["sources_sha256"]

    def strip(r: dict) -> list[dict]:
        return [{k: v for k, v in p.items() if k != "time"} for p in r["pairs"]]

    assert strip(rerun) == strip(scored)
    assert rerun["summary"] == scored["summary"]


def test_the_population_is_every_firing_pair_once():
    nh = [{"workflow": "ci.yml", "new_hidden": [{"job": "t", "step": "name:x", "kind": "born hidden"}]}]
    r7 = {"repos": [{"repo": "a/b", "commits": [{"sha": "1", "fires": True, "root": False, "detail": nh}, {"sha": "2", "fires": True, "root": True, "detail": nh},
                                                 {"sha": "3", "fires": False, "root": False}]}]}
    r9 = {"repos": [{"repo": "c/d", "prs": [{"number": 5, "agent": "Copilot", "audited": True, "fires": True, "base": "b5", "head": "h5", "merged": True, "detail": nh},
                                             {"number": 6, "agent": "Devin", "audited": False, "fires": True, "base": "b6", "head": "h6", "detail": nh}]}]}
    r10 = {"repos": [{"repo": "c/d", "prs": [{"number": 5, "group": "Copilot", "audited": True, "touching": True, "fires": True, "base": "x", "head": "h5", "detail": nh},
                                              {"number": 9, "group": "human", "audited": True, "touching": True, "fires": True, "base": "b9", "head": "h9", "merged_heuristic": False, "detail": nh},
                                              {"number": 10, "group": "human", "audited": True, "touching": True, "fires": False, "base": "b", "head": "h"}]}]}
    pop = O.population(r7, r9, r10)
    assert [(p["source"], p["group"], p["head"]) for p in pop] == [("swallow7", "mainline", "1"), ("swallow9", "Copilot", "h5"), ("swallow10", "human", "h9")]
    assert pop[0]["base"] is None and pop[0]["expected"] == [["ci.yml", "t", "name:x", "born hidden"]]


def test_two_fixes_one_click_away(tmp_path):
    s = _github(tmp_path)
    src = tmp_path / "src"
    r = O.replay_pair(src, s["c0"], s["h7"])                      # the pull request's own diff: its head against its fork point
    assert r["new_hidden"] == 2 and r["got"] == [["ci.yml", "test", "name:Lint", "acquired"], ["ci.yml", "test", "name:Typecheck", "born hidden"]]
    by = {c["step"]: c for c in r["checks"]}
    coe, tsc = _line(CI_PR7, "continue-on-error: true"), _line(CI_PR7, "npx tsc --noEmit || true")
    assert by["name:Lint"]["target"] == [coe, coe] and by["name:Lint"]["target_what"] == "continue-on-error"
    assert by["name:Typecheck"]["target"] == [tsc, tsc] and by["name:Typecheck"]["target_what"] == "run"
    for c in r["checks"]:
        assert c["located"] and c["visible"] and c["rebuilt"] and c["reproduces"] and c["one_click"] and "why_not" not in c
    assert by["name:Lint"]["size"] == 1 and by["name:Lint"]["added"] == 0
    assert by["name:Typecheck"]["size"] == 1 and by["name:Typecheck"]["added"] == 3


FLAKY = """on: [push]
jobs:
  flaky:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup
        run: echo setting up
      - name: More setup
        run: echo more
      - name: Even more setup
        run: echo still more
"""


def test_a_fix_outside_the_changes_own_diff_is_not_one_click(tmp_path):
    """The job was already continue-on-error; the change adds a check to it. The check is hidden by a
    line the change did not touch: the annotation is not in the diff, and the repair's line is not
    either -- a review suggestion cannot sit there."""
    tree = tmp_path / "r"
    (tree / ".github" / "workflows").mkdir(parents=True)
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    g = lambda *a: subprocess.run(["git", "-C", str(tree), *a], check=True, capture_output=True, text=True, env=env).stdout.strip()  # noqa: E731
    subprocess.run(["git", "init", "-q", "-b", "main", str(tree)], check=True)
    wf = tree / ".github" / "workflows" / "flaky.yml"
    wf.write_text(FLAKY)
    g("add", "-A")
    g("commit", "-qm", "flaky job")
    base = g("rev-parse", "HEAD")
    wf.write_text(FLAKY + "      - name: Unit tests\n        run: npm test\n")
    g("commit", "-qam", "add tests to the flaky job")
    head = g("rev-parse", "HEAD")
    r = O.replay_pair(tree, base, head)
    assert r["got"] == [["flaky.yml", "flaky", "name:Unit tests", "born hidden"]]
    c = r["checks"][0]
    assert c["located"] and c["target"] == [5, 5] and c["target_what"] == "job continue-on-error" and not c["visible"]
    assert c["repair"] == "no-continue-on-error" and c["rebuilt"] and c["reproduces"] and c["span"] == [5, 5]
    assert not c["one_click"] and c["why_not"] == "outside the change's diff"
    s = O.summary([{"source": "swallow10", "group": "human", "checks": r["checks"], "reproduced": True}])
    assert s["checks"] == 1 and s["located"] == 1 and s["visible"] == 0 and s["one_click"] == 0 and s["by_group"]["human"] == {"n": 1, "one_click": 0}


def test_the_replay_is_deterministic(tmp_path):
    s = _github(tmp_path)
    src = tmp_path / "src"
    assert O.replay_pair(src, s["c0"], s["h7"]) == O.replay_pair(src, s["c0"], s["h7"])

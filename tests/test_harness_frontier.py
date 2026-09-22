# -*- coding: utf-8 -*-
"""SWALLOW-13's instrument: the product's `--repair` path, three stages and two readings, on
repositories no stage was designed on. The tests hold the population to its rule, the instrument's
reading of one checkout to the product's, the determinism re-run, and the counts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import frontier as FR  # noqa: E402
from tests.test_ciaudit_frontier import FRONTIER_FIXTURE, _tree  # noqa: E402

# frozen at the sha256 the SWALLOW-13 preregistration names; a change needs a new preregistration, not a new pin
INSTRUMENT_SHA256 = "7b3c2b129750292cf058458cc975f4d2257ea6e642e932e982d5513a4ccbd9d1"


def test_the_instrument_is_the_one_the_preregistration_names():
    import hashlib
    if INSTRUMENT_SHA256 is None:
        pytest.skip("not yet frozen")
    assert hashlib.sha256((ROOT / "benchmarks" / "harness_mutation" / "frontier.py").read_bytes()).hexdigest() == INSTRUMENT_SHA256


def test_the_population_is_its_rule():
    committed = json.loads((ROOT / "papers" / "harness" / "swallow13_population.json").read_text(encoding="utf-8"))
    assert committed == FR.build_population()
    s1 = {x["repo"] for x in json.loads((ROOT / "papers" / "harness" / "swallow1_population.json").read_text(encoding="utf-8"))}
    names = [r["repo"] for r in committed["repos"]]
    assert names == sorted(names) and not set(names) & s1 and len(names) == committed["count"]
    assert committed["swallow9_repos"] == committed["count"] + committed["excluded_swallow1"] + committed["excluded_swallow11"]
    assert all(len(r["tip"]) == 40 for r in committed["repos"])


def test_one_checkout_read_as_the_product_reads_it(tmp_path):
    from styxx import ciaudit
    tree = _tree(tmp_path, FRONTIER_FIXTURE)
    res = FR.audit_one(tree)
    product = {t["name"]: t["verified_repair"] for t in ciaudit.audit(str(tree), repair=True)["repairs"]}
    assert {t["name"]: t["verified_repair"] for t in res["targets"]} == product
    by = {t["name"]: t for t in res["targets"]}
    assert all(t["tried_stage3"] and t["stage3_again_equal"] for t in res["targets"])
    assert by["Run tests"]["stage"] == "swallow-13" and "run" not in by["Run tests"]
    assert by["Lint baseline"]["stage"] is None and by["Lint baseline"]["run"].startswith("LINT_EXIT=0\n")
    assert by["Lint baseline"]["readings"]["routed"]["exported"] == ["LINT_EXIT"]
    s = FR.summary([{"repo": "a/b", "tip": "0" * 40, "fetch": {}, **res}, {"repo": "c/d", "tip": "1" * 40, "fetch": {"error": "fetch: gone"}}])
    assert (s["repos"], s["fetched"], s["fetch_failed"], s["targets"], s["read"]) == (2, 1, 1, 6, 6)
    assert (s["stage1_verified"], s["stage2_verified"], s["stage3_population"], s["stage3_verified"]) == (0, 0, 6, 5)
    assert s["stage3_by_family"] == {"hoist": 2, "background": 1, "default-joined": 1, "exit-zero": 1}
    assert (s["residue"], s["residue_routed"], s["residue_routed_a_reader_can_fail"], s["residue_declared"], s["residue_unexplained"]) == (1, 1, 1, 0, 0)
    assert s["stage3_again_equal"] == 6 and s["stage3_again_differs"] == 0 and s["stage3_repos"] == 1
    assert s["distinct_scripts"]["stage3_population"] == 6


def test_a_check_the_first_stage_repairs_never_reaches_the_third(tmp_path):
    tree = _tree(tmp_path, """on: [push]
jobs:
  t:
    runs-on: ubuntu-latest
    steps:
      - name: Unit tests
        continue-on-error: true
        run: npm test
""")
    (t,) = FR.audit_one(tree)["targets"]
    assert t["verified_repair"] == "no-continue-on-error" and t["stage"] == "swallow-4" and not t["tried_stage3"]
    assert [c["repair"] for c in t["candidates"]] == ["no-continue-on-error", "strict-shell", "both"] and "stage3_again_equal" not in t

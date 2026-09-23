# -*- coding: utf-8 -*-
"""SWALLOW-14's instrument: the product's `--repair` path, with the stage's two new edits (wait-list,
hoist-local), on repositories none of it was designed on. The tests hold the population to its rule,
the instrument's reading of two checkouts to the product's, and the counts the preregistration
scores."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("yaml")
from benchmarks.harness_mutation import empty_list as EL  # noqa: E402
from tests.test_ciaudit_frontier import FRONTIER_FIXTURE, LISTS_FIXTURE, _tree  # noqa: E402

# frozen at the sha256 the SWALLOW-14 preregistration names; a change needs a new preregistration, not a new pin
INSTRUMENT_SHA256 = "9b40257a3b718f02bd8e729bb29e54df706bf7ec0d84809669d826e7490e6b5c"


def test_the_instrument_is_the_one_the_preregistration_names():
    import hashlib
    if INSTRUMENT_SHA256 is None:
        pytest.skip("not yet frozen")
    assert hashlib.sha256((ROOT / "benchmarks" / "harness_mutation" / "empty_list.py").read_bytes()).hexdigest() == INSTRUMENT_SHA256


def test_the_population_is_its_rule():
    pop = json.loads(gzip.decompress((ROOT / "papers" / "harness" / "swallow14_population.json.gz").read_bytes()))
    names = [r["repo"] for r in pop["repos"]]
    earlier, _ = EL.excluded()
    assert names == sorted(names) and len(set(names)) == len(names) == pop["count"]
    assert not {n.lower() for n in names} & earlier and not {u["repo"].lower() for u in pop["unreachable"]} & earlier
    assert pop["table_repos"] == pop["excluded_earlier"] + pop["count"] + len(pop["unreachable"])
    assert all(len(r["tip"]) == 40 and int(r["tip"], 16) >= 0 for r in pop["repos"])


def _two(tmp_path):
    a = EL.audit_one(_tree(tmp_path / "a", FRONTIER_FIXTURE))
    b = EL.audit_one(_tree(tmp_path / "b", LISTS_FIXTURE))
    return a, b


def test_two_checkouts_read_as_the_product_reads_them(tmp_path):
    from styxx import ciaudit
    a, b = _two(tmp_path)
    for res, text in ((a, FRONTIER_FIXTURE), (b, LISTS_FIXTURE)):
        product = {t["name"]: t["verified_repair"] for t in ciaudit.audit(str(_tree(tmp_path / ("p" + str(id(res))), text)), repair=True)["repairs"]}
        assert {t["name"]: t["verified_repair"] for t in res["targets"]} == product
        assert all(t["tried_stage3"] and t["stage3_again_equal"] and t["run"] for t in res["targets"])
    by = {t["name"]: t for t in b["targets"]}
    assert by["Validate manifests"]["verified_repair"] == "wait-list" and by["Verify the changelog"]["verified_repair"] == "hoist-local"


def test_the_counts_the_preregistration_scores(tmp_path):
    a, b = _two(tmp_path)
    s = EL.summary([{"repo": "o/a", "tip": "0" * 40, "fetch": {}, **a}, {"repo": "o/b", "tip": "1" * 40, "fetch": {}, **b},
                    {"repo": "o/c", "tip": "2" * 40, "fetch": {"error": "fetch: gone"}}])
    assert (s["repos"], s["fetched"], s["fetch_failed"], s["targets"], s["read"], s["stage3_population"]) == (3, 2, 1, 9, 9, 9)
    # the stage as SWALLOW-13 ran it repairs five; four are left, and the new edits reach three of them
    assert (s["s13_verified"], s["s13_residue"], s["new_reach"], s["new_reach_repos"]) == (5, 4, 3, 1)
    assert s["new_reach_by_edit"] == {"wait-list": 2, "hoist-local": 1}
    assert (s["wait_class"], s["wait_class_verified"], s["wait_verified"], s["wait_repos"]) == (2, 2, 2, 1)
    # the local hoist verifies every check the global one does, and one it cannot
    assert (s["hoist_global_verified"], s["hoist_local_verified"], s["global_not_local"], s["local_not_global"]) == (2, 3, 0, 1)
    assert s["stage3_chosen"] == {"hoist-local": 2, "wait-list": 2, "background-liveness": 1, "no-coe+hoist-local": 1,
                                  "no-default-joined": 1, "no-exit-zero": 1}
    assert (s["all_stages_verified"], s["residue"], s["residue_read"]) == (8, 1, 1)
    assert s["stage3_again_equal"] == 9 and s["stage3_again_differs"] == 0
    assert len(s["new_chosen_lines"]) == 5 and s["wait_chosen_lines"] == [1, 3] and s["wait_chosen_lines_median"] == 2

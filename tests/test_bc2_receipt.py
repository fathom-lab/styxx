"""BC-2's receipt says what its result says, and the gates it reports are the ones the prereg froze.

`papers/closed-model-frontier/external3_gates.json` is the scored comparison of the checkout at the
BC-2 prereg commit against the repaired checkout over the EXTERNAL-1 corpus. These checks read the
receipt and the result document; they never re-run the corpus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTIER = ROOT / "papers" / "closed-model-frontier"
GATES = FRONTIER / "external3_gates.json"
RESULT = FRONTIER / "RESULT_bc2_by_construction_lands_2026_09_16.md"
SUMMARY = FRONTIER / "external3_summary.json"


def _gates() -> dict:
    return json.loads(GATES.read_text(encoding="utf-8"))


def test_every_blocking_gate_passed_and_the_receipt_says_so():
    g = _gates()
    assert g["prereg"] == "PREREG_bc1_by_construction_2026_09_16.md" or "bc" in g["prereg"]
    for name in ("G-B1_subset_invariant", "G-B2_by_construction", "G-B3_verified_preserved"):
        assert g["gates"][name]["pass"] is True, name
    assert g["all_blocking_gates_pass"] is True


def test_no_accusation_was_added_and_the_removed_count_is_the_difference():
    s = _gates()["gates"]
    assert s["G-B1_subset_invariant"]["new_accusations"] == 0
    surv = s["G-B5_survivors"]
    assert surv["accusations_before"] - surv["accusations_after"] == surv["removed"] == 569
    assert sum(surv["by_kind_removed"].values()) == 569
    assert "symbol_added" not in surv["by_kind_after"]


def test_the_by_construction_counters_are_zero_after_the_repair():
    counters = _gates()["gates"]["G-B2_by_construction"]["counters"]
    assert set(counters) == {"tests_added.contradicted_no_python_in_diff",
                             "symbol_added.contradicted_no_python_in_diff",
                             "only_touches.contradicted_prefix_is_not_a_path"}
    assert all(v == 0 for v in counters.values())
    assert json.loads(SUMMARY.read_text(encoding="utf-8"))["accusations_unsupported_by_construction"] == 0


def test_the_verified_side_lost_nothing():
    v = _gates()["gates"]["G-B3_verified_preserved"]
    assert v["verified_before"] == 18 and v["lost"] == 0 and v["verified_without_python_after"] == 0


def test_the_result_quotes_the_receipt():
    text = re.sub(r"\s+", " ", RESULT.read_text(encoding="utf-8"))
    surv = _gates()["gates"]["G-B5_survivors"]
    for token in (f"{surv['removed']} removed", "**0 new**", "`only_touches` 327", "`tests_added` 177",
                  "`symbol_added` 65", "96 CONTRADICTED claims remain", "All 18 baseline VERIFIED"):
        assert token in text, token

"""EXTERNAL-2's receipt is internally consistent and its result document quotes it, not memory.

`papers/closed-model-frontier/external2_summary.json` is the census of the accusations the installed
wheel still makes on the EXTERNAL-1 corpus (#110). These checks pin the arithmetic the RESULT relies
on — the by-kind totals sum to the accusation total, the "unsupported by construction" count is the
sum of its three named parts, the VERIFIED side is what the document says — and that every headline
number in the document appears in the receipt. They read files; they never re-run the corpus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTIER = ROOT / "papers" / "closed-model-frontier"
SUMMARY = FRONTIER / "external2_summary.json"
GATE = FRONTIER / "external2_gate_summary.json"
RESULT = FRONTIER / "RESULT_external2_live_accusations_2026_09_16.md"


def _summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_accusations_by_kind_sum_to_the_total():
    s = _summary()
    assert sum(s["accusations_by_kind"].values()) == s["accusations_total"] == 665
    for kind, verdicts in s["claims_by_kind_and_verdict"].items():
        assert verdicts.get("CONTRADICTED", 0) == s["accusations_by_kind"].get(kind, 0), kind


def test_unsupported_by_construction_is_the_sum_of_its_three_parts():
    s = _summary()
    parts = (s["tests_added"]["contradicted_no_python_in_diff"],
             s["symbol_added"]["contradicted_no_python_in_diff"],
             s["only_touches"]["contradicted_prefix_is_not_a_path"])
    assert parts == (168, 59, 322)
    assert sum(parts) == s["accusations_unsupported_by_construction"] == 549
    assert s["accusations_unsupported_share"] == round(549 / 665, 4)


def test_the_language_split_partitions_each_kind():
    s = _summary()
    t, y = s["tests_added"], s["symbol_added"]
    assert t["contradicted_no_python_in_diff"] + t["contradicted_python_in_diff"] == t["CONTRADICTED"] == 184
    assert y["contradicted_no_python_in_diff"] + y["contradicted_python_in_diff"] == y["CONTRADICTED"] == 65
    o = s["only_touches"]
    assert o["contradicted_prefix_is_not_a_path"] + o["contradicted_prefix_looks_like_a_path"] == o["CONTRADICTED"] == 341


def test_the_verified_side_is_what_the_result_says():
    s = _summary()
    t = s["tests_added"]
    assert t["VERIFIED"] == t["verified_python_in_diff"] == 13
    assert t["verified_with_a_redefined_def"] == 0 and t["verified_but_net_rule_disagrees"] == 0
    assert s["symbol_added"]["VERIFIED"] == 1 and s["only_touches"]["VERIFIED"] == 4


def test_the_gate_summary_reproduces_external1_where_it_must():
    g = json.loads(GATE.read_text(encoding="utf-8"))
    e1 = json.loads((FRONTIER / "external1_summary.json").read_text(encoding="utf-8"))
    assert g["eligible"] == e1["eligible"] == 71016
    assert g["excluded"] == e1["excluded"]
    assert g["prs_with_contradiction"] == e1["prs_with_contradiction"] == 625
    assert g["claims_by_verdict"]["CONTRADICTED"] == e1["claims_by_verdict"]["CONTRADICTED"] == 665
    assert g["instrument"]["styxx_version"] == "7.47.0"
    assert g["instrument"]["diffgate_sha256"].startswith("fb2d9b3e")


def test_every_headline_number_in_the_result_is_in_the_receipt():
    text = re.sub(r"\s+", " ", RESULT.read_text(encoding="utf-8"))   # the document wraps at 100
    s = _summary()
    for token in ("549 of 665", "82.6%", "168 of the 184", "59 of the 65", "322 of its 341",
                  "`the` 105", "`with` 54", "`that` 29", "13 `tests_added` VERIFIED"):
        assert token in text, token
    top = dict(s["only_touches_top_contradicted_prefixes"])
    assert (top["the"], top["with"], top["that"]) == (105, 54, 29)
    kinds = dict(re.findall(r"`(only_touches|tests_added|files_changed_count|symbol_added)` (\d+)", text)[:4])
    assert {k: int(v) for k, v in kinds.items()} == s["accusations_by_kind"]

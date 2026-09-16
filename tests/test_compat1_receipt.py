"""COMPAT-1's receipt says what its result says: one verdict on the corpus, nothing else moved.

`papers/closed-model-frontier/external4_gates.json` compares the BC-2 ledger with the COMPAT-1
ledger over the EXTERNAL-1 corpus. These checks read the receipt; they never re-run the corpus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTIER = ROOT / "papers" / "closed-model-frontier"
GATES = FRONTIER / "external4_gates.json"
RESULT = FRONTIER / "RESULT_compat1_lands_2026_09_16.md"
EXPLORATORY = FRONTIER / "exploratory_ml_compat.json"


def _g() -> dict:
    return json.loads(GATES.read_text(encoding="utf-8"))


def test_the_kind_had_exactly_one_verdict_on_the_corpus():
    g = _g()["gates"]["G-C1_never_accuses"]
    assert set(g["verdicts"]) == {"UNCHECKABLE"} and g["pass"] is True
    assert g["verdicts"]["UNCHECKABLE"] == 13329


def test_nothing_else_moved():
    g = _g()["gates"]["G-C2_every_other_kind_untouched"]
    assert g["differences"] == 0 and g["before"] == g["after_non_compat"] and g["pass"] is True


def test_the_census_is_reproduced_and_the_histogram_sums():
    c = _g()["gates"]["G-C3_census_reproduced"]
    exp = json.loads(EXPLORATORY.read_text(encoding="utf-8"))["compat"]
    assert c["prs_with_a_compat_claim"] == exp["claims"] == 8467
    assert sum(c["removed_count_histogram"].values()) == c["prs_with_a_removed_public_definition"] == 540
    assert c["prs_with_a_removed_public_definition"] + c["prs_nothing_removed"] + c["prs_no_covered_language"] == 8467


def test_the_result_quotes_the_receipt():
    text = re.sub(r"\s+", " ", RESULT.read_text(encoding="utf-8"))
    for token in ("13,329 `compat_claim` claims on 8,467 PRs", "0 differences", "**540**",
                  "JS/TS 243, Python 164, Go 79, Java 40, Rust 34", "12,933 to 17,939"):
        assert token in text, token
    assert _g()["all_blocking_gates_pass"] is True

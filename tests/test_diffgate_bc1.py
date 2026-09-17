# -*- coding: utf-8 -*-
"""BC-2 (PREREG_bc2_by_construction_2026_09_16, after BC-1's INVALID; issue #110): the accusations the gate could not
have supported are no longer made. Four repairs, each pinned on a minimal diff, each
accusation-removing; two catches given up on purpose, pinned as xfail(strict=True) so that
re-enabling either later cannot happen silently; the four pairs owed to the differential
corpus checked on the Python side."""
import json
from pathlib import Path

import pytest

from styxx import diffgate as dg
from styxx.diffgate import gate_diff_text

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "web" / "gate" / "differential" / "bc1_pairs.json"

TS_DIFF = """\
--- a/src/add.ts
+++ b/src/add.ts
@@ -1 +1,2 @@
 export const add = (a: number, b: number) => a + b;
+export const sub = (a: number, b: number) => a - b;
--- /dev/null
+++ b/src/add.test.ts
@@ -0,0 +1,3 @@
+import { add } from "./add";
+test("adds", () => { expect(add(1, 2)).toBe(3); });
+test("adds negatives", () => { expect(add(-1, -2)).toBe(-3); });
"""

PY_DIFF = """\
--- a/src/app.py
+++ b/src/app.py
@@ -1,2 +1,4 @@
 def old():
     return 1
+def retry(n):
+    return n
--- /dev/null
+++ b/tests/test_app.py
@@ -0,0 +1,4 @@
+def test_retry():
+    assert True
+def test_retry_twice():
+    assert True
--- a/config/settings.yml
+++ b/config/settings.yml
@@ -1 +1 @@
-timeout: 30
+timeout: 5
"""


def _one(summary, diff, kind):
    g = gate_diff_text(summary, diff, run=None, strict=False)
    found = [c for c in g.claims if c.kind == kind]
    assert len(found) == 1, (kind, [(c.kind, c.verdict) for c in g.claims])
    return g, found[0]


# ---- repair 1: the language gate ------------------------------------------------------------

def test_tests_added_abstains_when_the_diff_has_no_python():
    g, c = _one("Added 2 tests for add().", TS_DIFF, "tests_added")
    assert c.verdict == "UNCHECKABLE"
    assert "no Python file in the diff" in c.why and "#110" in c.why
    assert g.verdict == "PASS"


def test_symbol_added_abstains_when_the_diff_has_no_python():
    g, c = _one("Adds function sub for subtraction.", TS_DIFF, "symbol_added")
    assert c.verdict == "UNCHECKABLE"
    assert "no Python file in the diff" in c.why


def test_python_diffs_are_counted_exactly_as_before():
    _, c = _one("Added 5 tests covering the retry path.", PY_DIFF, "tests_added")
    assert c.verdict == "CONTRADICTED" and "adds 2 test functions, claim says 5" in c.why
    _, c = _one("Added 2 tests covering the retry path.", PY_DIFF, "tests_added")
    assert c.verdict == "VERIFIED"
    _, c = _one("Adds function backoff.", PY_DIFF, "symbol_added")
    assert c.verdict == "CONTRADICTED"
    _, c = _one("Adds function retry.", PY_DIFF, "symbol_added")
    assert c.verdict == "VERIFIED"


def test_a_pyi_stub_counts_as_python():
    assert dg._diff_touches_python({"pkg/mod.pyi": "M"})
    assert not dg._diff_touches_python({"src/app.ts": "M", "README.md": "M"})


# ---- repair 2: the counted noun -------------------------------------------------------------

@pytest.mark.parametrize("noun", ["case", "cases", "files", "scenarios", "suites", "classes"])
def test_a_wrong_count_of_test_cases_files_or_scenarios_abstains(noun):
    _, c = _one(f"Added 3 test {noun} covering the retry path.", PY_DIFF, "tests_added")
    assert c.verdict == "UNCHECKABLE"                      # BC-2 repair 2: a case is not a function
    assert f"counts test {noun}, diff adds 2 test functions" in c.why and "#110" in c.why


@pytest.mark.parametrize("noun", ["cases", "files", "scenarios", "classes"])
def test_a_matching_count_of_test_cases_is_still_verified(noun):
    _, c = _one(f"Added 2 test {noun} covering the retry path.", PY_DIFF, "tests_added")
    assert c.verdict == "VERIFIED" and c.why == "diff adds 2 test functions, claim says 2"


@pytest.mark.parametrize("noun", ["functions", "methods"])
def test_test_functions_and_methods_are_still_counted(noun):
    _, c = _one(f"Added 2 test {noun} covering the retry path.", PY_DIFF, "tests_added")
    assert c.verdict == "VERIFIED" and "adds 2 test functions, claim says 2" in c.why


# ---- repair 3: words are not symbols --------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "Adds a method to reload data when files change.",
    "Added class declaration and complete term definition.",
    "Added function implementation for the schema.",
    "Introduces a function with the full signature.",
])
def test_a_function_word_after_the_kind_is_not_a_claim(sentence):
    g = gate_diff_text(sentence, PY_DIFF, run=None, strict=False)
    assert not [c for c in g.claims if c.kind == "symbol_added"]
    assert g.uncovered_sentences == 1          # never-read, counted as such
    assert g.verdict == "PASS"


def test_named_and_called_are_skipped_to_reach_the_symbol():
    _, c = _one("Adds a function named retry.", PY_DIFF, "symbol_added")
    assert c.verdict == "VERIFIED" and c.detail["name"] == "retry"
    _, c = _one("Added a new method called backoff.", PY_DIFF, "symbol_added")
    assert c.verdict == "CONTRADICTED" and c.detail["name"] == "backoff"


# ---- repair 4: path-shaped prefixes, and two of them ----------------------------------------

@pytest.mark.parametrize("sentence,word", [
    ("Only modifies the footer component.", "the"),
    ("This change only touches markdown.", "markdown"),
    ("Only modified files within the directory.", "files"),
])
def test_a_prefix_that_is_a_word_abstains_and_names_the_word(sentence, word):
    _, c = _one(sentence, PY_DIFF, "only_touches")
    assert c.verdict == "UNCHECKABLE"
    assert c.why == f"prefix {word!r} is not a path (#110)"


def test_a_hyphenated_only_matches_twice_and_both_abstain():
    # "documentation-only change" also matches the template ("only change that"); the
    # instrument reads it as before -- the repair only stops the accusation.
    g = gate_diff_text("This is a documentation-only change that only touches markdown.",
                       PY_DIFF, run=None, strict=False)
    kinds = [(c.kind, c.verdict) for c in g.claims]
    assert kinds == [("only_touches", "UNCHECKABLE"), ("only_touches", "UNCHECKABLE")]
    assert {c.why for c in g.claims} == {"prefix 'that' is not a path (#110)",
                                         "prefix 'markdown' is not a path (#110)"}


def test_a_prefix_with_a_slash_or_a_dot_or_a_changed_segment_is_a_path():
    status = {"src/app.py": "M", "tests/test_app.py": "A"}
    for p in ("docs/", "package.json", "src", "tests", "SRC", "`src`", "src."):
        assert dg._prefix_is_path_shaped(p, status), p
    for p in ("the", "files", "markdown", "code", "", "."):
        assert not dg._prefix_is_path_shaped(p, status), p


def test_a_path_shaped_prefix_that_is_wrong_still_accuses():
    _, c = _one("This change only touches files under docs/.", PY_DIFF, "only_touches")
    assert c.verdict == "CONTRADICTED" and "paths outside 'docs'" in c.why


def test_two_prefixes_are_both_read():
    _, c = _one("Only touches src/ and tests/ in this change.", PY_DIFF, "only_touches")
    assert c.verdict == "CONTRADICTED"
    assert c.detail["prefix2"] == "tests/"
    assert "config/settings.yml" in c.why and "paths outside 'src' and 'tests'" in c.why
    two = PY_DIFF[:PY_DIFF.index("--- a/config/settings.yml")]    # the same diff minus config/
    _, c = _one("Only touches src/ and tests/ in this change.", two, "only_touches")
    assert c.verdict == "VERIFIED" and c.why == "all changed paths under prefix"
    _, c = _one("Only touches `src/`, and `tests/` in this change.", two, "only_touches")
    assert c.verdict == "VERIFIED"


def test_a_comma_does_not_join_prefixes_and_a_word_after_and_is_ignored():
    # BC-2 repair 4: ", not `/docusaurus/...`" is not a second prefix, and "and nothing else"
    # does not make `nothing` one; the first prefix decides alone.
    _, c = _one("Only modified files in `src/`, not `docs/`.", PY_DIFF, "only_touches")
    assert "prefix2" not in c.detail and c.verdict == "CONTRADICTED" and "paths outside 'src'" in c.why
    _, c = _one("Only touches src/ and nothing else.", PY_DIFF, "only_touches")
    assert "prefix2" not in c.detail and "paths outside 'src'" in c.why


# ---- the demo is unchanged ------------------------------------------------------------------

def test_the_readme_demo_still_names_the_same_three_lies():
    g = gate_diff_text(dg._DEMO_SUMMARY, dg._DEMO_DIFF, run=None, strict=False)
    lies = sorted(c.kind for c in g.claims if c.verdict == "CONTRADICTED")
    assert lies == ["only_touches", "symbol_added", "tests_added"]
    assert g.verdict == "FAIL"


# ---- the catches given up, pinned so re-enabling them is a visible act ----------------------

@pytest.mark.xfail(strict=True, reason=(
    "BC-1/BC-2 repair 1 (PREREG_bc2_by_construction_2026_09_16): a lie about test counts in a "
    "TypeScript diff is an abstention, not a catch, because this template counts `def test_`. "
    "Counting tests in other languages is a separate, preregistered capability; landing it "
    "must flip this marker in the same commit."))
def test_sacrificed_a_lying_test_count_in_typescript():
    _, c = _one("Added 3 tests for add().", TS_DIFF, "tests_added")
    assert c.verdict == "CONTRADICTED"


@pytest.mark.xfail(strict=True, reason=(
    "BC-2 repair 2: a lie counted in test CASES is an abstention, because a case is not a "
    "`def test_` function. Counting cases (parametrizations, table rows) is not a capability "
    "this instrument has; adding one must flip this marker in the same commit."))
def test_sacrificed_a_lying_count_of_test_cases():
    _, c = _one("Added 9 test cases covering everything.", PY_DIFF, "tests_added")
    assert c.verdict == "CONTRADICTED"


# ---- the four pairs owed to the differential corpus (G-B6) ----------------------------------

def test_the_four_differential_pairs_hold_on_the_python_side():
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    assert [p["id"] for p in pairs] == ["bc1:ts-tests", "bc1:word-symbol", "bc1:the-prefix",
                                        "bc1:two-prefixes"]
    for p in pairs:
        g = gate_diff_text(p["summary"], p["diff"], run=None, strict=False)
        got = [[c.kind, c.verdict] for c in g.claims]
        assert got == p["expect"]["claims"], (p["id"], got)
        assert g.verdict == p["expect"]["verdict"], p["id"]
        assert g.uncovered_sentences == p["expect"]["uncovered_sentences"], p["id"]


def test_the_flag_is_on_and_the_rules_are_the_census_rules():
    assert dg.BC1_BY_CONSTRUCTION is True
    assert {"cases", "files", "scenarios"} <= dg._TEST_NOUNS_NOT_FUNCTIONS
    assert {"to", "with", "that", "declaration", "implementation"} <= dg._SYMBOL_WORDS

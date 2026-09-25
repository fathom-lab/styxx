# -*- coding: utf-8 -*-
"""PATH-2 (PREREG_path2_resolution_2026_09_17): three repairs to what a record says about a path or
a definition.

  #97   a path claim resolves exact, then suffix, then basename, over every status entry — an
        earlier basename match no longer shadows an exact one;
  #121  the path key keeps a dotfile's dots (`.pr_agent.toml` and `pr_agent.toml` are two files),
        and `only_touches` does not accuse on a dot alone;
  #101  a `def` that the removed lines of the same file also define is changed, not added — on
        both doors, `gate_diff_text` and `gate_diff`.

The reproduction tests fail on origin/main 87dded26 and pass after the repair. The edge-case tests
pin the readings the prereg states, including the disclosed limits (a basename still resolves, a
rename and a test moved between files count as added).

AMENDMENT_path2_resolution_2026_09_17 (the second freeze) adds the tests marked c1 / c2 / c3: removed
and added definitions pair one to one per file and name (a status-A file pairs nothing), with patterns
that read a BOM and a non-ASCII name the same way in both ports; COMPAT's and BC-1's path readings use
the undotted key; `only_touches` abstains on dot misses alone and accuses listing real paths only. And
the BIN-1 registration of dotfile headers with no hunks, on both doors.

NOTE_path2_fourth_pass_2026_09_25 adds the tests marked f1-f4: the prefix-shape test undots the prefix
(F-1), a diff splits into lines on \\r\\n, \\r and \\n only, on both doors (F-2), `got` reads the indent
the pairing reads (F-3), and an off-tree prefix beside an on-tree one no longer withdraws an accusation
no reading of it could answer (F-4).

NOTE_path2_fifth_pass_2026_09_25 adds the tests marked v1, v3 and v4: one reading of a Python definition
line for every pattern that reads one -- CPython's indentation (space, tab, form feed) after one
optional U+FEFF, the same keyword separator, a name ending at an ASCII non-name character -- checked
for every character of Python's \\s and U+FEFF on the raw door, through the port, and on the git door
(V-1); a prefix written to end in `..`, and a `..` after a named segment, could hold anything (V-4);
and the scorer attributes a move per claim, by reverting one rule in its own copy of the instrument
(V-3).

NOTE_path2_seventh_pass_2026_09_25 adds the tests marked x1, x2 and x7: both ports read a name by ONE
pinned Unicode table (15.0.0), which the generator reproduces; a claimed name that runs on past the
identifier, or ends in a middle dot, names none (x1); a hunk is read by its counts only when it carries
what it declares and orders its lines as a generator does, else as main reads it (x2); and every check
the scorer gained -- its own reading of each rule it re-implements (G-C7), the gate-level fields and
verdict (G-C1), the git door (G-C8) -- fails on a defect planted in the instrument (x7)."""
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import pytest

from styxx import diffgate as dg
from styxx.diffgate import gate_diff, gate_diff_text, parse_unified_diff, parse_unified_diff_sides

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "web" / "gate" / "differential" / "path2_pairs.json"

TWO_READMES = """\
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-a
+b
--- /dev/null
+++ b/integrations/git/README.md
@@ -0,0 +1 @@
+hello
"""

TWO_READMES_REVERSED = """\
--- /dev/null
+++ b/integrations/git/README.md
@@ -0,0 +1 @@
+hello
--- a/README.md
+++ b/README.md
@@ -1 +1 @@
-a
+b
"""

CHANGED_DEFS = """\
--- a/src/retry.py
+++ b/src/retry.py
@@ -1,2 +1,2 @@
-def backoff(n):
+def backoff(n, jitter=0):
     return n * 2
--- a/tests/test_retry.py
+++ b/tests/test_retry.py
@@ -1,6 +1,6 @@
-def test_a():
+def test_a():  # unchanged behaviour, comment added
     assert True


-def test_b():
+def test_b():  # unchanged behaviour, comment added
     assert True
"""

DOTFILE_TWINS = """\
diff --git a/pr_agent.toml b/pr_agent.toml
deleted file mode 100644
--- a/pr_agent.toml
+++ /dev/null
@@ -1 +0,0 @@
-x = 1
diff --git a/.pr_agent.toml b/.pr_agent.toml
new file mode 100644
--- /dev/null
+++ b/.pr_agent.toml
@@ -0,0 +1 @@
+x = 1
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -1 +1 @@
-a: 1
+a: 2
"""

CI = "--- a/.github/workflows/ci.yml\n+++ b/.github/workflows/ci.yml\n@@ -1 +1 @@\n-a: 1\n+a: 2\n"
APP = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"

def _claims(summary, diff):
    g = gate_diff_text(summary, diff, run=None, strict=False)
    return g, [(c.kind, c.verdict, c.why) for c in g.claims]


# ────────────────────────────────────────────────────────────────────────── #97

def test_97_an_exact_match_is_not_shadowed_by_an_earlier_basename():
    g, got = _claims("Created integrations/git/README.md.", TWO_READMES)
    assert got == [("file_created", "VERIFIED", "diff status 'A' for 'integrations/git/readme.md'")]
    assert g.verdict == "PASS"


def test_97_in_the_other_order_the_root_readme_is_the_file_named():
    _, got = _claims("Modified README.md.", TWO_READMES_REVERSED)
    assert got == [("file_touched", "VERIFIED", "diff status 'M' for 'readme.md'")]


def test_97_a_suffix_match_beats_an_earlier_basename_match():
    diff = ("--- a/docs/app.py\n+++ b/docs/app.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
            "--- /dev/null\n+++ b/src/node/app.py\n@@ -0,0 +1 @@\n+b = 1\n")
    _, got = _claims("Created node/app.py.", diff)
    assert got == [("file_created", "VERIFIED", "diff status 'A' for 'src/node/app.py'")]


def test_97_an_exact_match_beats_an_earlier_suffix_match():
    diff = "--- a/src/glob.ts\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n--- /dev/null\n+++ b/glob.ts\n@@ -0,0 +1 @@\n+x\n"
    _, got = _claims("Created glob.ts.", diff)
    assert got == [("file_created", "VERIFIED", "diff status 'A' for 'glob.ts'")]


def test_97_a_basename_still_resolves_when_nothing_stronger_exists():
    # the disclosed limit: a claim that names a directory still meets a same-named file elsewhere
    diff = "--- a/lib/util/helpers.py\n+++ b/lib/util/helpers.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
    _, got = _claims("Modified src/helpers.py.", diff)
    assert got == [("file_touched", "VERIFIED", "diff status 'M' for 'lib/util/helpers.py'")]


@pytest.mark.parametrize("claimed", ["README.md", "readme.md", "git/README.md", "integrations/git/README.md",
                                     "docs/README.md", "other.md", "x/other.md", "glob.ts", "src/glob.ts"])
def test_97_found_or_not_found_is_exactly_what_it_was(claimed):
    status = {"src/glob.ts": "D", "readme.md": "M", "integrations/git/readme.md": "A", "glob.ts": "A"}

    def any_tier_in_diff_order(c):
        c = dg._norm(c)
        for p, st in status.items():
            if p == c or p.endswith("/" + c) or Path(p).name == Path(c).name:
                return p, st
        return None, None

    assert (dg._find_path(status, claimed)[0] is None) == (any_tier_in_diff_order(claimed)[0] is None)


# ───────────────────────────────────────────────────────────────────────── #121

@pytest.mark.parametrize("raw,key", [
    (".pr_agent.toml", ".pr_agent.toml"), ("pr_agent.toml", "pr_agent.toml"),
    (".github/workflows/CI.yml", ".github/workflows/ci.yml"), (".env", ".env"), ("..env", "..env"),
    ("../x.py", "../x.py"), ("./src/x.py", "src/x.py"), ("/src/x.py", "src/x.py"),
    (".//src/x.py", "src/x.py"), ("/./src/x.py", "src/x.py"), ("./.github/x.yml", ".github/x.yml"),
    ("src\\win\\x.py", "src/win/x.py"), ("./", ""),
])
def test_121_the_key_drops_only_leading_slash_segments(raw, key):
    assert dg._norm(raw) == key
    assert dg._undotted(dg._norm(raw)) == raw.replace("\\", "/").lstrip("./").lower()   # the old key, exactly


def test_121_a_dotfile_and_its_undotted_twin_are_two_files():
    status, _ = parse_unified_diff(DOTFILE_TWINS)
    assert status == {"pr_agent.toml": "D", ".pr_agent.toml": "A", ".github/workflows/ci.yml": "M"}
    assert list(parse_unified_diff_sides(DOTFILE_TWINS)) == ["pr_agent.toml", ".pr_agent.toml", ".github/workflows/ci.yml"]


def test_121_the_twins_read_truthfully_and_the_reasons_print_the_dots():
    g, got = _claims("3 files changed. Created .pr_agent.toml. Deleted pr_agent.toml. Only touches src/.", DOTFILE_TWINS)
    assert got == [
        ("files_changed_count", "VERIFIED", "diff changes 3 files, claim says 3"),
        ("file_created", "VERIFIED", "diff status 'A' for '.pr_agent.toml'"),
        ("file_deleted", "VERIFIED", "diff status 'D' for 'pr_agent.toml'"),
        ("only_touches", "CONTRADICTED",
         "paths outside 'src': ['pr_agent.toml', '.pr_agent.toml', '.github/workflows/ci.yml']"),
    ]
    assert g.verdict == "FAIL"


def test_121_a_miss_by_the_dot_alone_abstains_and_the_dotted_prefix_verifies():
    _, got = _claims("Only touches github/. Only touches .github/.", CI)
    assert got == [
        ("only_touches", "UNCHECKABLE",
         "paths outside 'github' differ from it only by a leading dot: ['.github/workflows/ci.yml'] (#121)"),
        ("only_touches", "VERIFIED", "all changed paths under prefix"),
    ]


def test_121_a_bare_word_prefix_still_names_a_dotted_directory_and_two_prefixes_are_read():
    assert dg._prefix_is_path_shaped("github", {".github/workflows/ci.yml": "M"})
    assert not dg._prefix_is_path_shaped("config", {"src/.config/x.yml": "M"})   # inner dots were never dropped
    _, got = _claims("Only touches src and github. Only touches /src/ and /.github.", CI + APP)
    assert got == [
        ("only_touches", "UNCHECKABLE", "paths outside 'src' and 'github' differ from them only by a leading dot: "
                                        "['.github/workflows/ci.yml'] (#121)"),
        ("only_touches", "VERIFIED", "all changed paths under prefix"),
    ]


def test_121_a_path_outside_the_prefix_by_more_than_a_dot_is_still_caught():
    _, got = _claims("2 files changed. Only touches src/.", APP + "--- /dev/null\n+++ b/.env\n@@ -0,0 +1 @@\n+A=1\n")
    assert got == [("files_changed_count", "VERIFIED", "diff changes 2 files, claim says 2"),
                   ("only_touches", "CONTRADICTED", "paths outside 'src': ['.env']")]


def test_121_leading_dot_slash_and_slash_still_read_as_before():
    _, got = _claims("Modified ./src/app.py. Only touches /src.", APP)
    assert got == [("file_touched", "VERIFIED", "diff status 'M' for 'src/app.py'"),
                   ("only_touches", "VERIFIED", "all changed paths under prefix")]


def test_121_a_dotfile_does_not_answer_for_its_undotted_name():
    diff = ("--- a/.eslintrc.json\n+++ b/.eslintrc.json\n@@ -1 +1 @@\n-{}\n+{\"a\": 1}\n"
            "--- /dev/null\n+++ b/config/eslintrc.json\n@@ -0,0 +1 @@\n+{}\n")
    _, got = _claims("Created eslintrc.json. Updated .eslintrc.json.", diff)
    assert got == [("file_created", "VERIFIED", "diff status 'A' for 'config/eslintrc.json'"),
                   ("file_touched", "VERIFIED", "diff status 'M' for '.eslintrc.json'")]


# ───────────────────────────────────────────────────────────────────────── #101

ISSUE_101 = [
    ("symbol_added", "UNCHECKABLE", "added lines define function 'backoff' only where the removed lines of the "
                                    "same file define it too; a changed definition is not an added one (#101)"),
    ("tests_added", "UNCHECKABLE", "diff adds 0 test functions and changes 2, claim says 2; a changed test is "
                                   "not an added one (#101)"),
]


def test_101_the_issue_diff_is_not_verified_on_the_raw_door():
    g, got = _claims("Adds function backoff with jitter. Added 2 tests.", CHANGED_DEFS)
    assert got == ISSUE_101
    assert g.verdict == "PASS"                     # an abstention, never an accusation


def test_101_a_new_test_beside_a_changed_one_verifies_the_net_count_and_accuses_only_outside_the_interval():
    diff = ("--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1,2 +1,6 @@\n-def test_a():\n+def test_a(tmp_path):\n"
            "     assert True\n+\n+\n+def test_new():\n+    assert True\n")
    _, got = _claims("Added 0 tests. Added 1 test. Added 2 tests. Added 3 tests.", diff)
    assert got == [
        ("tests_added", "CONTRADICTED", "diff adds 1 test functions, claim says 0 (1 changed, not added: #101)"),
        ("tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1 (1 changed, not added: #101)"),
        ("tests_added", "UNCHECKABLE", "diff adds 1 test functions and changes 1, claim says 2; a changed test is "
                                       "not an added one (#101)"),
        ("tests_added", "CONTRADICTED", "diff adds 1 test functions, claim says 3 (1 changed, not added: #101)"),
    ]


def test_101_counted_cases_over_a_changed_test_abstain_under_the_110_rule():
    diff = "--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1,2 +1,2 @@\n-def test_a():\n+def test_a(x):\n     assert True\n"
    _, got = _claims("Added 1 test case.", diff)
    assert got == [("tests_added", "UNCHECKABLE", "counts test case, diff adds 0 test functions; a case is not a "
                                                  "function (#110) (1 changed, not added: #101)")]


def test_101_a_changed_def_beside_a_fresh_one_in_another_file_verifies():
    diff = ("--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-def backoff(n):\n+def backoff(n, j=0):\n"
            "--- /dev/null\n+++ b/src/b.py\n@@ -0,0 +1,2 @@\n+def backoff():\n+    return 1\n")
    _, got = _claims("Adds function backoff.", diff)
    assert got == [("symbol_added", "VERIFIED", "added lines do define function 'backoff'")]


def test_101_a_rename_is_an_added_symbol():
    diff = "--- a/src/retry.py\n+++ b/src/retry.py\n@@ -1,2 +1,2 @@\n-def back_off(n):\n+def backoff(n):\n     return n\n"
    _, got = _claims("Adds function backoff.", diff)
    assert got == [("symbol_added", "VERIFIED", "added lines do define function 'backoff'")]


def test_101_a_test_moved_between_files_counts_as_added_the_disclosed_limit():
    diff = ("--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1,2 +0,0 @@\n-def test_x():\n-    assert True\n"
            "--- a/tests/test_b.py\n+++ b/tests/test_b.py\n@@ -1 +1,3 @@\n y = 1\n+def test_x():\n+    assert True\n")
    _, got = _claims("Added 1 test.", diff)
    assert got == [("tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1")]


def test_101_with_no_changed_definition_every_reason_is_the_old_one():
    diff = ("--- a/src/app.py\n+++ b/src/app.py\n@@ -1,2 +1,6 @@\n-x = 1\n+x = 2\n+def retry(n):\n+    return n\n"
            "--- /dev/null\n+++ b/tests/test_app.py\n@@ -0,0 +1,4 @@\n+def test_one():\n+    pass\n+def test_two():\n+    pass\n")
    _, got = _claims("Adds function retry. Adds function missing. Added 2 tests. Added 5 tests. Added 3 test cases.", diff)
    assert got == [
        ("symbol_added", "VERIFIED", "added lines do define function 'retry'"),
        ("symbol_added", "CONTRADICTED", "added lines do NOT define function 'missing'"),
        ("tests_added", "VERIFIED", "diff adds 2 test functions, claim says 2"),
        ("tests_added", "CONTRADICTED", "diff adds 2 test functions, claim says 5"),
        ("tests_added", "UNCHECKABLE", "counts test cases, diff adds 2 test functions; a case is not a function (#110)"),
    ]


def test_101_the_helpers_count_same_file_changes_only():
    sides = parse_unified_diff_sides(CHANGED_DEFS)
    status = parse_unified_diff(CHANGED_DEFS)[0]
    assert dg._changed_test_defs(sides, status) == 2
    assert dg._changed_test_defs(None) == 0
    assert dg._definition_only_changed("backoff", sides, status)
    assert not dg._definition_only_changed("missing", sides, status)
    assert not dg._definition_only_changed("backoff", None)


# ─────────────────────────────── AMENDMENT_path2_resolution_2026_09_17, C-1 (#101)

def test_101_c1_one_removed_definition_cancels_one_added_one_not_every_same_named_one():
    # TestA.test_run changes, TestB.test_run and test_other are new: two tests added
    diff = ("--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1,2 +1,5 @@\n-class TestA:\n-    def test_run(self):\n"
            "+class TestA:\n+    def test_run(self, tmp_path):\n+class TestB:\n+    def test_run(self):\n+def test_other():\n")
    sides = parse_unified_diff_sides(diff)
    assert dg._changed_test_defs(sides, parse_unified_diff(diff)[0]) == 1
    _, got = _claims("Added 1 test. Added 2 tests.", diff)
    assert got == [
        ("tests_added", "CONTRADICTED", "diff adds 2 test functions, claim says 1 (1 changed, not added: #101)"),
        ("tests_added", "VERIFIED", "diff adds 2 test functions, claim says 2 (1 changed, not added: #101)"),
    ]


def test_101_c1_a_changed_test_beside_a_same_named_new_one_is_not_zero_added():
    diff = ("--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1,2 +1,4 @@\n-class TestFoo:\n-    def test_basic(self):\n"
            "+class TestFoo:\n+    def test_basic(self, client):\n+class TestBar:\n+    def test_basic(self):\n")
    _, got = _claims("Added 0 tests. Added 1 test.", diff)
    assert got == [
        ("tests_added", "CONTRADICTED", "diff adds 1 test functions, claim says 0 (1 changed, not added: #101)"),
        ("tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1 (1 changed, not added: #101)"),
    ]


FOLD_A = ("diff --git a/tests/test_x.py b/tests/test_x.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/test_x.py\n"
          "@@ -0,0 +1,4 @@\n+def test_x():\n+    pass\n+def test_y():\n+    pass\n"
          "@@ -1,2 +1,2 @@\n-def test_x():\n+def test_x(tmp_path):\n     pass\n")
FOLD_M = ("diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n"
          "@@ -1,1 +1,3 @@\n y = 1\n+def test_x():\n+    pass\n"
          "@@ -2,2 +2,2 @@\n-def test_x():\n+def test_x(tmp_path):\n     pass\n")


def test_101_c1_a_file_with_status_A_pairs_nothing_so_the_fold_reads_as_on_87dded26():
    # the shelf's fold: a created file whose later commit edits a test; `got` over-counts, as it did
    assert parse_unified_diff(FOLD_A)[0] == {"tests/test_x.py": "A"}
    assert dg._changed_test_defs(parse_unified_diff_sides(FOLD_A), parse_unified_diff(FOLD_A)[0]) == 0
    _, got = _claims("Added 2 tests. Added 3 tests.", FOLD_A)
    assert got == [("tests_added", "CONTRADICTED", "diff adds 3 test functions, claim says 2"),
                   ("tests_added", "VERIFIED", "diff adds 3 test functions, claim says 3")]


def test_101_c1_the_fold_under_an_M_header_pairs_the_edit_once():
    _, got = _claims("Added 0 tests. Added 1 test.", FOLD_M)
    assert got == [
        ("tests_added", "CONTRADICTED", "diff adds 1 test functions, claim says 0 (1 changed, not added: #101)"),
        ("tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1 (1 changed, not added: #101)"),
    ]


def test_101_c1_a_bom_strip_is_a_changed_test_and_a_changed_function():
    test_diff = "--- a/tests/test_b.py\n+++ b/tests/test_b.py\n@@ -1 +1 @@\n-\ufeffdef test_a():\n+def test_a():\n"
    sym_diff = "--- a/src/h.py\n+++ b/src/h.py\n@@ -1 +1 @@\n-\ufeffdef backoff(n):\n+def backoff(n):\n"
    _, got = _claims("Added 1 test.", test_diff)
    assert got == [("tests_added", "UNCHECKABLE", "diff adds 0 test functions and changes 1, claim says 1; "
                                                  "a changed test is not an added one (#101)")]
    _, got = _claims("Adds function backoff.", sym_diff)
    assert got == [("symbol_added", "UNCHECKABLE", ISSUE_101[0][2])]


def test_r1_an_added_bom_line_is_counted_by_got_as_well_as_by_the_pairing():
    # NOTE_path2_third_pass R-1. `got` and the added-side pairing pattern read the same lines, so a
    # BOM-prefixed added definition cancels a removed one only when it was counted as added
    # beforehand. Before the repair Python's \s skipped it in `got` while the pairing subtracted it, and
    # `net` fell below the tests really added: a false VERIFIED of the #101 kind.
    diff = "--- a/tests/test_b.py\n+++ b/tests/test_b.py\n@@ -1 +1 @@\n-def test_a():\n+\ufeffdef test_a():\n"
    _, got = _claims("Added 0 tests.", diff)
    assert got == [("tests_added", "VERIFIED",
                    "diff adds 0 test functions, claim says 0 (1 changed, not added: #101)")]
    # NOTE_path2_sixth_pass W-1: the U+FEFF opens line 1 of the file, so the one parse drops it before
    # `got` or the pairing reads the line; the definition reading itself takes none.
    assert dg.parse_unified_diff(diff)[1] == "def test_a():"
    assert dg._added_tests("def test_a():") == 1 and dg._added_tests("\ufeffdef test_a():") == 0


def test_r1_a_bom_on_a_changed_test_does_not_hide_a_new_one():
    # The non-degenerate case the round-2 reviewers filed: one test really is added.
    diff = ("--- a/tests/test_bom.py\n+++ b/tests/test_bom.py\n@@ -1,2 +1,4 @@\n"
            "-def test_a():\n+\ufeffdef test_a():\n     pass\n+def test_b():\n+    pass\n")
    _, got = _claims("Added 0 tests. Added 1 test.", diff)
    assert got == [
        ("tests_added", "CONTRADICTED", "diff adds 1 test functions, claim says 0 (1 changed, not added: #101)"),
        ("tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1 (1 changed, not added: #101)"),
    ]
    two = ("--- a/tests/test_bom.py\n+++ b/tests/test_bom.py\n@@ -1,2 +1,6 @@\n"
           "-def test_a():\n+\ufeffdef test_a():\n     pass\n+def test_b():\n+    pass\n"
           "+def test_c():\n+    pass\n")
    _, got = _claims("Added 1 test. Added 2 tests.", two)
    assert got == [
        ("tests_added", "CONTRADICTED", "diff adds 2 test functions, claim says 1 (1 changed, not added: #101)"),
        ("tests_added", "VERIFIED", "diff adds 2 test functions, claim says 2 (1 changed, not added: #101)"),
    ]


def test_r2_an_async_test_made_sync_is_a_changed_test_not_an_added_one():
    # NOTE_path2_third_pass R-2. `async` is accepted on the REMOVED side only: `got` does not count
    # `async def test_`, so accepting it on the added side would part the two sets again (R-1).
    diff = ("--- a/tests/test_as.py\n+++ b/tests/test_as.py\n@@ -1,2 +1,2 @@\n"
            "-async def test_fetch():\n+def test_fetch():\n     assert fetch()\n")
    _, got = _claims("Added 0 tests. Added 1 test.", diff)
    assert got == [
        ("tests_added", "VERIFIED", "diff adds 0 test functions, claim says 0 (1 changed, not added: #101)"),
        ("tests_added", "UNCHECKABLE", "diff adds 0 test functions and changes 1, claim says 1; "
                                       "a changed test is not an added one (#101)"),
    ]
    assert dg._test_name("async def test_fetch():", removed=True) == "test_fetch"
    assert dg._test_name("async def test_fetch():") is None
    assert dg._added_tests("async def test_fetch():") == 0


def test_101_c1_a_test_name_runs_to_a_space_tab_paren_or_colon_so_non_ascii_names_are_distinct():
    # NOTE_path2_sixth_pass W-2: a test name is the Python identifier after `def`, so it runs through a
    # non-ASCII letter and ends at anything an identifier cannot hold -- a form feed, a `[`, a `(`.
    assert dg._test_name("def test_\u00f6len(tmp_path):") == "test_\u00f6len"
    assert dg._test_name("\tdef test_a:") == "test_a"
    assert dg._test_name("def test_a\x0c():") == "test_a" and dg._test_name("def test_a[T]():") == "test_a"
    assert dg._test_name("\u00a0def test_a():") is None                 # no \s: only CPython's indentation
    assert dg._test_name("\ufeff\tdef test_a:") is None                 # a U+FEFF only where W-1's parse drops it
    diff = ("--- a/tests/test_u.py\n+++ b/tests/test_u.py\n@@ -1 +1,3 @@\n-def test_\u00f6len():\n"
            "+def test_\u00f6len(tmp_path):\n+def test_\u00e4rger():\n+def test_plain():\n")
    _, got = _claims("Added 1 test. Added 2 tests.", diff)
    assert got == [
        ("tests_added", "CONTRADICTED", "diff adds 2 test functions, claim says 1 (1 changed, not added: #101)"),
        ("tests_added", "VERIFIED", "diff adds 2 test functions, claim says 2 (1 changed, not added: #101)"),
    ]


def test_101_c1_symbol_verifies_when_a_file_adds_more_definitions_than_it_removes():
    other_class = ("--- a/src/io.py\n+++ b/src/io.py\n@@ -1,2 +1,4 @@\n-class Reader:\n-    def close(self):\n"
                   "+class Reader:\n+    def close(self, force=False):\n+class Writer:\n+    def close(self):\n")
    in_place = "--- a/src/io.py\n+++ b/src/io.py\n@@ -1 +1 @@\n-    def close(self):\n+    def close(self, force=False):\n"
    _, got = _claims("Adds method close.", other_class)
    assert got == [("symbol_added", "VERIFIED", "added lines do define method 'close'")]
    _, got = _claims("Adds method close.", in_place)
    assert got == [("symbol_added", "UNCHECKABLE", "added lines define method 'close' only where the removed lines "
                                                   "of the same file define it too; a changed definition is not an "
                                                   "added one (#101)")]


def test_101_c1_symbol_names_end_at_a_space_tab_paren_or_colon_and_status_A_removes_none():
    unicode_suffix = "--- a/src/r.py\n+++ b/src/r.py\n@@ -1 +1 @@\n-def backoff\u00e9(n):\n+def backoff(n):\n"
    under_a = "--- /dev/null\n+++ b/src/r.py\n@@ -1 +1 @@\n-def backoff(n):\n+def backoff(n, jitter=0):\n"
    generic = "--- a/src/r.py\n+++ b/src/r.py\n@@ -1 +1 @@\n-x = 1\n+def backoff[T](n: T) -> T:\n"
    for diff in (unicode_suffix, under_a, generic):
        _, got = _claims("Adds function backoff.", diff)
        assert got == [("symbol_added", "VERIFIED", "added lines do define function 'backoff'")], diff
    assert dg._defines("async\tdef backoff(n):", "backoff", removed=True)
    assert not dg._defines("async\tdef backoff(n):", "backoff")
    assert not dg._defines("def backoff_v2(n):", "backoff", removed=True)


def test_r4_the_symbol_rule_boundaries_the_erratum_restates():
    # NOTE_path2_third_pass R-4. The round-2 reviewer found five mutants of the symbol pattern and
    # of rule (a) that no test could tell from the frozen rule. Each line below is one of them.
    verified = ("symbol_added", "VERIFIED", "added lines do define function 'backoff'")
    changed = ("symbol_added", "UNCHECKABLE", "added lines define {} 'backoff' only where the removed lines "
                                              "of the same file define it too; a changed definition is not an "
                                              "added one (#101)")
    # M5h, as NOTE_path2_fifth_pass V-1 re-reads it: the name ends where `hit` ends it -- at an ASCII
    # character that cannot continue a name, or at the end of the line -- so a generic definition's `[`
    # ends it too, and a CHANGED generic function pairs. The limit ERRATUM item 2 restated (a changed
    # generic still verified) was a line `hit` read and the pairing did not; it is closed.
    gen_changed = ("--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n"
                   "-def backoff[T](n: T) -> T:\n+def backoff[T](n: T, j: int = 0) -> T:\n")
    _, got = _claims("Adds function backoff.", gen_changed)
    assert got == [("symbol_added", changed[1], changed[2].format("function"))]
    # M8b: rule (a) needs a file that BOTH adds and removes the name. A function made generic now does.
    made_generic = "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-def backoff(n):\n+def backoff[T](n: T) -> T:\n"
    _, got = _claims("Adds function backoff.", made_generic)
    assert got == [("symbol_added", changed[1], changed[2].format("function"))]
    # ... and a file that only REMOVES the name still pairs nothing.
    removed_only = ("--- a/src/b.py\n+++ b/src/b.py\n@@ -1,2 +1 @@\n-def backoff(n):\n-    pass\n+x = 1\n"
                    "--- a/src/c.py\n+++ b/src/c.py\n@@ -1 +1,2 @@\n x = 0\n+def backoff(n):\n")
    _, got = _claims("Adds function backoff.", removed_only)
    assert got == [verified]
    # M5c: the indent is [ \t]*, not \s*. An NBSP-indented removed definition is not read at all.
    nbsp = "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-\u00a0def backoff(n):\n+def backoff(n, j=0):\n"
    _, got = _claims("Adds function backoff.", nbsp)
    assert got == [verified]
    assert not dg._defines("\u00a0def backoff(n):", "backoff", removed=True)
    # M5g: a space before the parameter list is inside the lookahead, so the pair is seen.
    space_par = "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-def backoff (n):\n+def backoff (n, j=0):\n"
    _, got = _claims("Adds function backoff.", space_par)
    assert got == [("symbol_added", changed[1], changed[2].format("function"))]
    # M5e: the lookahead also accepts end of line, so `class Backoff` with no colon pairs.
    cls_eol = "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-class Backoff\n+class Backoff:\n"
    _, got = _claims("Adds class Backoff.", cls_eol)
    assert got == [("symbol_added", "UNCHECKABLE",
                    "added lines define class 'Backoff' only where the removed lines of the same file "
                    "define it too; a changed definition is not an added one (#101)")]
    assert dg._defines("class Backoff", "Backoff", removed=True)
    assert dg._defines("class Backoff:", "Backoff", removed=True)


# ─────────────────────────────── AMENDMENT_path2_resolution_2026_09_17, C-2 (#121)

STORYBOOK = ("--- a/.storybook/preview.js\n+++ b/.storybook/preview.js\n@@ -1 +1 @@\n"
             "-export function withTheme(story) {\n+function withThemeLocal(story) {\n")


def test_121_c2_a_dotted_scaffold_directory_stays_scaffolding_for_compat2():
    g = gate_diff_text("This change is fully backward compatible.", STORYBOOK, run=None, strict=False)
    (c,) = g.claims
    assert (c.kind, c.verdict) == ("compat_claim", "UNCHECKABLE")
    assert c.why == ("compatibility claimed; 1 public definition(s) removed, all in test/example/internal code: "
                     ".storybook/preview.js: withTheme")
    assert c.detail["surface_removed"] == 0 and c.detail["compat2_candidate"] is False
    assert c.detail["removed"] == [{"path": ".storybook/preview.js", "language": "js/ts", "name": "withTheme",
                                    "surface": False}]


def test_r4_the_compat_language_suffix_reads_the_undotted_key_on_removed_lines_too():
    # NOTE_path2_third_pass R-4 (mutant M10b). The one-file `.py` diff below never reaches the
    # second suffix site, because no language is present at all; this two-file diff does, and with
    # that site reading the dotted key the removed definition in `.py` joins python's surface and
    # compat2_candidate flips False -> True, which is exactly what C-2 says cannot happen.
    diff = ("--- a/.py\n+++ b/.py\n@@ -1,2 +1,1 @@\n-def public_api():\n-    return 1\n+x = 1\n"
            "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n")
    g = gate_diff_text("No breaking changes.", diff, run=None, strict=False)
    (c,) = g.claims
    assert (c.kind, c.verdict) == ("compat_claim", "UNCHECKABLE")
    assert c.why == ("compatibility claimed; no public top-level definition removed "
                     "(python read; behaviour beyond names not checked)")
    assert c.detail["surface_removed"] == 0 and c.detail["compat2_candidate"] is False
    assert c.detail["removed"] == []


def test_121_c2_a_file_named_dot_py_is_not_python_to_bc1_or_compat():
    diff = "--- a/.py\n+++ b/.py\n@@ -1,2 +1,2 @@\n-def public_api():\n+def test_a():\n"
    g = gate_diff_text("Added 1 test. No breaking changes.", diff, run=None, strict=False)
    assert [(c.kind, c.verdict, c.why) for c in g.claims] == [
        ("tests_added", "UNCHECKABLE", "no Python file in the diff; this template counts `def` lines (#110)"),
        ("compat_claim", "UNCHECKABLE", "compatibility claimed; no language this reading covers in the diff "
                                        "(python, js/ts, go, rust, java)"),
    ]


# ─────────────────────────────── AMENDMENT_path2_resolution_2026_09_17, C-3 (#121)

def _m(path):
    return f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n"


def test_121_c3_an_accusation_lists_only_real_outside_paths():
    diff = _m(".github/a.yml") + _m(".github/b.yml") + _m(".github/c.yml") + _m("src/x.py")
    _, got = _claims("Only touches github/.", diff)
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'github': ['src/x.py']")]
    _, got = _claims("Only touches src and github.", diff[:-len(_m("src/x.py"))] + _m("docs/x.md") + _m("src/y.py"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'src' and 'github': ['docs/x.md']")]


def test_121_c3_a_dot_on_the_prefix_and_not_on_the_path_is_not_a_dot_miss():
    _, got = _claims("Only touches .env.", _m("env"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside '.env': ['env']")]
    _, got = _claims("Only touches .github/.", _m(".github/x.yml") + _m("github/z.md"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside '.github': ['github/z.md']")]


def test_121_c3_a_dotdot_path_is_not_a_dot_miss():
    _, got = _claims("Only touches src/.", _m("../src/x.py"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'src': ['../src/x.py']")]
    assert not dg._dot_miss("..env", ["env"]) and not dg._dot_miss("../src/x.py", ["src"])
    assert dg._dot_miss(".github/x.yml", ["github"]) and dg._dot_miss(".env", ["env"])
    assert not dg._dot_miss(".github/x.yml", [".github"]) and not dg._dot_miss("github/x.yml", ["github"])
    # a dotted prefix over a path with one dot more: each clause alone excludes it, so this pins the pair
    assert not dg._dot_miss("..env", [".env"]) and not dg._dot_miss("..github/x.yml", [".github"])
    _, got = _claims("Only touches .env.", _m("..env"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside '.env': ['..env']")]


def test_r3_a_prefix_written_with_two_dots_is_not_a_repo_path_and_abstains():
    # NOTE_path2_third_pass R-3. `../docs` and `.../src/x.py` name a location outside the tree the
    # diff describes; git emits no path that starts with `../`, so EVERY changed path was "outside"
    # and C-3 as frozen accused whatever the pull request did. The accusation class C-3 keeps is the
    # dotfile one below; this class abstains, the lab's habit where the claim is not a repo path.
    off = "prefix {!r} is relative to a directory the diff does not name (#121)"
    _, got = _claims("Only touches ../docs/ from the package.", _m("docs/guide.md"))
    assert got == [("only_touches", "UNCHECKABLE", off.format("../docs"))]
    _, got = _claims("Only touches ./../docs.", _m("docs/guide.md"))
    assert got == [("only_touches", "UNCHECKABLE", off.format("../docs"))]
    _, got = _claims("Only touches .../src/x.py here.", _m("src/x.py"))
    assert got == [("only_touches", "UNCHECKABLE", off.format(".../src/x.py"))]
    # one off-tree prefix is enough: abstaining beats accusing on the half that can be read
    _, got = _claims("Only touches ../docs/ and src/.", _m("docs/guide.md") + _m("src/pkg/a.py"))
    assert got == [("only_touches", "UNCHECKABLE", off.format("../docs"))]
    assert dg._prefix_off_tree("../docs") and dg._prefix_off_tree(".../src/x.py")
    assert not dg._prefix_off_tree(".env") and not dg._prefix_off_tree(".github/x")
    assert not dg._prefix_off_tree("docs") and not dg._prefix_off_tree("")


def test_r3_c3_reads_a_path_segment_boundary_and_not_a_name_prefix():
    # NOTE_path2_third_pass R-4 (mutant M13). A dot miss needs the undotted path to BE the prefix or
    # to lie under `prefix/`; `.docsearch.json` merely starts with `docs`, and dropping the slash
    # turned a correct accusation into an abstention.
    _, got = _claims("Only touches docs/.", _m(".docsearch.json"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'docs': ['.docsearch.json']")]
    _, got = _claims("Only touches github/.", _m(".githubx/a.yml"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'github': ['.githubx/a.yml']")]
    assert not dg._dot_miss(".docsearch.json", ["docs"])
    assert not dg._dot_miss(".githubx/a.yml", ["github"])
    assert dg._dot_miss(".docs/search.json", ["docs"]) and dg._dot_miss(".docs", ["docs"])


# ─────────────────────────────── BIN-1 registration keeps the dots (#121, R-121.2)

BIN_TWINS = ("diff --git a/logo.png b/logo.png\ndeleted file mode 100644\nindex 3b18e51..0000000\n"
             "Binary files a/logo.png and /dev/null differ\n"
             "diff --git a/.logo.png b/.logo.png\nnew file mode 100644\nindex 0000000..3b18e51\n"
             "Binary files /dev/null and b/.logo.png differ\n")
RENAME_TO_DOTTED = ("diff --git a/eslintrc.json b/.eslintrc.json\nsimilarity index 100%\nrename from eslintrc.json\n"
                    "rename to .eslintrc.json\n")


def test_121_binary_dotfile_twins_with_no_hunks_are_two_files():
    assert parse_unified_diff(BIN_TWINS)[0] == {"logo.png": "D", ".logo.png": "A"}
    assert list(parse_unified_diff_sides(BIN_TWINS)) == ["logo.png", ".logo.png"]
    _, got = _claims("2 files changed. Created .logo.png. Deleted logo.png. Only touches assets/.", BIN_TWINS)
    assert got == [("files_changed_count", "VERIFIED", "diff changes 2 files, claim says 2"),
                   ("file_created", "VERIFIED", "diff status 'A' for '.logo.png'"),
                   ("file_deleted", "VERIFIED", "diff status 'D' for 'logo.png'"),
                   ("only_touches", "CONTRADICTED", "paths outside 'assets': ['logo.png', '.logo.png']")]


def test_121_a_pure_rename_to_a_dotted_name_registers_the_dotted_name():
    assert parse_unified_diff(RENAME_TO_DOTTED)[0] == {".eslintrc.json": "M"}
    _, got = _claims("1 file changed. Only touches .eslintrc.json. Only touches eslintrc.json.", RENAME_TO_DOTTED)
    assert got == [("files_changed_count", "VERIFIED", "diff changes 1 files, claim says 1"),
                   ("only_touches", "VERIFIED", "all changed paths under prefix"),
                   ("only_touches", "UNCHECKABLE", "paths outside 'eslintrc.json' differ from it only by a leading "
                                                   "dot: ['.eslintrc.json'] (#121)")]


# ──────────────────────────────────────────────────────────── the doors agree

def _git_repo(tmp_path: Path, before: dict, after: dict) -> str:
    def git(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True,
                              text=True, encoding="utf-8").stdout
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    git("config", "core.autocrlf", "false")
    for rel, text in before.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(text.encode("utf-8"))
    git("add", "-A")
    git("commit", "-qm", "base")
    for rel, text in after.items():
        if text is None:
            (tmp_path / rel).unlink()
        else:
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_bytes(text.encode("utf-8"))
    git("add", "-A")
    git("commit", "-qm", "change")
    return git("diff", "HEAD~1..HEAD")


DOORS = {
    "changed-defs": ("Adds function backoff with jitter. Added 2 tests.",
                     {"src/retry.py": "def backoff(n):\n    return n * 2\n",
                      "tests/test_retry.py": "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n"},
                     {"src/retry.py": "def backoff(n, jitter=0):\n    return n * 2\n",
                      "tests/test_retry.py": "def test_a():  # unchanged behaviour, comment added\n    assert True\n\n\n"
                                             "def test_b():  # unchanged behaviour, comment added\n    assert True\n"}),
    "dotfile-twins": ("3 files changed. Created .pr_agent.toml. Deleted pr_agent.toml. Only touches github/.",
                      {"pr_agent.toml": "legacy = true\nname = 'old'\n", ".github/workflows/ci.yml": "a: 1\n"},
                      {"pr_agent.toml": None, ".pr_agent.toml": "[config]\nmodel = 'x'\nretries = 3\n",
                       ".github/workflows/ci.yml": "a: 2\n"}),
    "two-readmes": ("Created integrations/git/README.md. Modified README.md.",
                    {"README.md": "a\n"},
                    {"README.md": "b\n", "integrations/git/README.md": "hello\n"}),
}


@pytest.mark.parametrize("name", sorted(DOORS))
def test_the_git_door_and_the_raw_door_agree(tmp_path, name):
    summary, before, after = DOORS[name]
    diff = _git_repo(tmp_path, before, after)
    via_git = gate_diff(summary, tmp_path, "HEAD~1", "HEAD")
    via_text = gate_diff_text(summary, diff)
    got_git = [(c.kind, c.verdict, c.why) for c in via_git.claims]
    got_text = [(c.kind, c.verdict, c.why) for c in via_text.claims]
    assert got_git == got_text
    assert via_git.verdict == via_text.verdict
    if name == "changed-defs":
        assert got_git == ISSUE_101
    elif name == "dotfile-twins":
        assert got_git[:3] == [("files_changed_count", "VERIFIED", "diff changes 3 files, claim says 3"),
                               ("file_created", "VERIFIED", "diff status 'A' for '.pr_agent.toml'"),
                               ("file_deleted", "VERIFIED", "diff status 'D' for 'pr_agent.toml'")]
    else:
        assert got_git == [("file_created", "VERIFIED", "diff status 'A' for 'integrations/git/readme.md'"),
                           ("file_touched", "VERIFIED", "diff status 'M' for 'readme.md'")]


@pytest.mark.parametrize("name", ["binary-dotfile-twins", "rename-to-a-dotted-name"])
def test_the_doors_agree_on_dotfile_headers_with_no_hunks(tmp_path, name):
    if name == "binary-dotfile-twins":
        summary = "2 files changed. Created .logo.png. Deleted logo.png. Only touches assets/."
        before = {"logo.png": "\x00\x01old-png-bytes\x00" * 3}
        after = {"logo.png": None, ".logo.png": "\x00\x7fcompletely different content\x00\x02" * 5}
    else:
        summary = "1 file changed. Only touches .eslintrc.json. Only touches eslintrc.json."
        before = {"eslintrc.json": '{"root": true, "rules": {"semi": "error"}}\n'}
        after = {"eslintrc.json": None, ".eslintrc.json": '{"root": true, "rules": {"semi": "error"}}\n'}
    diff = _git_repo(tmp_path, before, after)
    via_git = gate_diff(summary, tmp_path, "HEAD~1", "HEAD")
    via_text = gate_diff_text(summary, diff)
    got_git = [(c.kind, c.verdict, c.why) for c in via_git.claims]
    assert got_git == [(c.kind, c.verdict, c.why) for c in via_text.claims]
    if name == "binary-dotfile-twins":
        assert "Binary files" in diff and "---" not in diff
        assert parse_unified_diff(diff)[0] == {".logo.png": "A", "logo.png": "D"}
        assert got_git == [("files_changed_count", "VERIFIED", "diff changes 2 files, claim says 2"),
                           ("file_created", "VERIFIED", "diff status 'A' for '.logo.png'"),
                           ("file_deleted", "VERIFIED", "diff status 'D' for 'logo.png'"),
                           ("only_touches", "CONTRADICTED", "paths outside 'assets': ['.logo.png', 'logo.png']")]
    else:
        assert "rename to .eslintrc.json" in diff
        assert parse_unified_diff(diff)[0] == {".eslintrc.json": "M"}
        assert got_git == [("files_changed_count", "VERIFIED", "diff changes 1 files, claim says 1"),
                           ("only_touches", "VERIFIED", "all changed paths under prefix"),
                           ("only_touches", "UNCHECKABLE", "paths outside 'eslintrc.json' differ from it only by "
                                                           "a leading dot: ['.eslintrc.json'] (#121)")]


# ─────────────────────────────── NOTE_path2_fourth_pass_2026_09_25, F-1 .. F-4

@pytest.mark.parametrize("prefix,path", [(".github", ".github/ci.yml"), (".vscode", ".vscode/settings.json"),
                                         (".circleci", ".circleci/config.yml"), (".husky", ".husky/pre-commit"),
                                         (".gitignore", ".gitignore"), (".npmrc", ".npmrc"),
                                         (".env.local", ".env.local"), (".gitignore", "gitignore")])
def test_f1_a_slashless_dotted_prefix_is_read_undotted_like_the_changed_paths(prefix, path):
    # The round-3 blocker. The prefix kept its dot while the changed paths were undotted, so a dotted
    # prefix whose last dot-segment is not a listed extension was "not a path" -- and only
    # `.eslintrc.json`-shaped spellings, whose suffix is listed, were ever tested.
    assert dg._prefix_is_path_shaped(prefix, {path: "M"})
    assert dg._prefix_is_path_shaped(prefix + ".", {path: "M"})         # a sentence-final period
    assert not dg._prefix_is_path_shaped(".k-step-link", {"packages/core/_layout.scss": "M"})   # PATH-1 still holds


def test_f1_a_dotted_prefix_verifies_accuses_and_carries_c3_in_all_three_positions():
    for summary, path in (("Only touches .github.", ".github/ci.yml"), ("Only touches .gitignore.", ".gitignore"),
                          ("Only touches .env.local.", ".env.local"), ("Only touches .vscode.", ".vscode/settings.json")):
        _, got = _claims(summary, _m(path))
        assert got == [("only_touches", "VERIFIED", "all changed paths under prefix")], summary
    _, got = _claims("Only touches .npmrc.", _m(".npmrc") + _m("src/a.py"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside '.npmrc': ['src/a.py']")]
    _, got = _claims("Only touches .gitignore.", _m(".gitignore") + _m("src/a.py"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside '.gitignore': ['src/a.py']")]
    # AMENDMENT C-3's accusation class, which the blocker had switched off for these spellings
    _, got = _claims("Only touches .gitignore.", _m("gitignore"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside '.gitignore': ['gitignore']")]
    _, got = _claims("Only touches .github.", _m("github/ci.yml"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside '.github': ['github/ci.yml']")]


# the characters Python's str.splitlines() breaks on and git does not
PY_ONLY_BREAKS = ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]


def _reindent(ch):
    return f"--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1 +1 @@\n-def test_a():\n+{ch}def test_a():\n"


def test_f2_a_diff_splits_on_crlf_cr_and_lf_and_nothing_else():
    assert dg._diff_lines("a\r\nb\rc\nd\n") == ["a", "b", "c", "d"]
    assert dg._diff_lines("") == [] and dg._diff_lines("a\n\n") == ["a", ""]
    for ch in PY_ONLY_BREAKS:
        assert dg._diff_lines(f"+x{ch}y\n") == [f"+x{ch}y"], repr(ch)
        assert parse_unified_diff(_reindent(ch))[1] == f"{ch}def test_a():", repr(ch)
        assert parse_unified_diff_sides(_reindent(ch)) == {"tests/test_a.py": ([f"{ch}def test_a():"],
                                                                               ["def test_a():"])}, repr(ch)


# NOTE_path2_fifth_pass V-1: a form feed is CPython indentation, so a form-feed re-indent is a CHANGED
# test; the other three characters are not indentation, and the line defines nothing.
REINDENT_AS_CHANGED = [("tests_added", "VERIFIED", "diff adds 0 test functions, claim says 0 (1 changed, not added: #101)"),
                       ("tests_added", "UNCHECKABLE", "diff adds 0 test functions and changes 1, claim says 1; a "
                                                      "changed test is not an added one (#101)")]
REINDENT_AS_NOTHING = [("tests_added", "VERIFIED", "diff adds 0 test functions, claim says 0"),
                       ("tests_added", "CONTRADICTED", "diff adds 0 test functions, claim says 1")]


@pytest.mark.parametrize("ch", ["\x0b", "\x0c", "\u2028", "\u2029"])
def test_f2_the_four_classes_the_ports_split_on_differently_now_read_alike(ch):
    # Round 3, review 2: `-def test_a():` / `+<ch>def test_a():` is a re-indent; the port said
    # "Added 1 test." VERIFIED where the Python said CONTRADICTED. The pinned pairs hold the port.
    _, got = _claims("Added 0 tests. Added 1 test.", _reindent(ch))
    assert got == (REINDENT_AS_CHANGED if ch == "\x0c" else REINDENT_AS_NOTHING)


def test_f2_a_separator_inside_a_line_no_longer_forges_a_line():
    ctx = ("--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1,2 +1,2 @@\n x\u2028+def test_new():\n"
           "-y = 0\n+y = 1\n")
    _, got = _claims("Added 1 test.", ctx)
    assert got == [("tests_added", "CONTRADICTED", "diff adds 0 test functions, claim says 1")]
    hdr = ("--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1,2 @@\n x\u2028+++ b/evil.py\n-y = 0\n+y = 1\n"
           "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-a\n+b\n")
    assert parse_unified_diff(hdr)[0] == {"src/a.py": "M", "src/b.py": "M"}
    _, got = _claims("2 files changed. Only touches src/.", hdr)
    assert got == [("files_changed_count", "VERIFIED", "diff changes 2 files, claim says 2"),
                   ("only_touches", "VERIFIED", "all changed paths under prefix")]


def test_f3_got_and_the_pairing_read_exactly_the_same_lines():
    # Round 3, review 2: R-1 made the pairing a SUBSET of `got`. Every whitespace character either
    # port knows, as an indent: `got` counts the line exactly when the pairing pattern reads it.
    for c in [0x09, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680, 0x2000, 0x200A, 0x2028,
              0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF]:
        line = chr(c) + "def test_a():"
        # NOTE_path2_sixth_pass W-2: `got` IS the added-side pairing's reading, counted line by line
        assert dg._added_tests(line) == (1 if dg._test_name(line) else 0), hex(c)
    for ch in ("\u00a0", "\u3000"):
        _, got = _claims("Added 0 tests. Added 1 test.", _reindent(ch))
        assert got == [("tests_added", "VERIFIED", "diff adds 0 test functions, claim says 0"),
                       ("tests_added", "CONTRADICTED", "diff adds 0 test functions, claim says 1")], repr(ch)


def test_f4_an_off_tree_prefix_beside_an_on_tree_one_withdraws_only_what_it_could_answer():
    off = "prefix '../docs' is relative to a directory the diff does not name (#121)"
    claim = "Only modified `src/` and `../docs` as specified."
    # outside `src/` and outside every reading of `../docs`: the accusation stands
    _, got = _claims(claim, _m("evil/x.py"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'src' and '../docs': ['evil/x.py']")]
    # only the sure paths are listed
    _, got = _claims(claim, _m("docs/a.md") + _m("evil/x.py"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'src' and '../docs': ['evil/x.py']")]
    # some reading of `../docs` (X/docs) could hold these: abstain, as R-3 did
    for diff in (_m("packages/docs/x.md"), _m("src/a.py") + _m("docs/guide.md"), _m("src/a.py"), _m(".src/a.py")):
        _, got = _claims(claim, diff)
        assert got == [("only_touches", "UNCHECKABLE", off)], diff
    # an off-tree prefix standing alone still abstains, whatever changed
    _, got = _claims("Only touches ../docs/ from the package.", _m("evil/x.py"))
    assert got == [("only_touches", "UNCHECKABLE", off)]
    assert dg._could_lie_under("packages/docs/x.md", "../docs") and dg._could_lie_under("docs", "../docs")
    assert not dg._could_lie_under("evil/x.py", "../docs") and not dg._could_lie_under("docsx/a.md", "../docs")
    assert dg._could_lie_under(".github/x.yml", "../.github") and dg._could_lie_under("anything", "..")
    # compared undotted on both sides, so the test errs towards "could"
    assert dg._could_lie_under(".github/x.yml", "../github") and dg._could_lie_under("github/x.yml", "../.github")
    assert dg._could_lie_under("lib/src/x.py", ".../src/x.py") and not dg._could_lie_under("src/y.py", ".../src/x.py")
    assert not dg._could_lie_under("x.py/src", ".../src/x.py")        # in order and contiguous, not as a set


def test_f5_a_path_opening_with_two_dots_is_never_a_dot_miss():
    # Round 3, review 2 (mutant P13): the `..` arm of `_dot_miss` had no test. Without it, a bare
    # filename prefix would read `..a/bar.py` as `.a/bar.py` plus a dot, which is what this pins.
    assert dg._dot_miss(".a/bar.py", ["bar.py"])
    assert not dg._dot_miss("..a/bar.py", ["bar.py"])


FOURTH_PASS_DOORS = {
    **{f"f2-{ord(ch):04x}": ("Added 0 tests. Added 1 test.", ch) for ch in ("\x0b", "\x0c", "\u2028", "\u2029")},
    "f3-nbsp": ("Added 0 tests. Added 1 test.", "\u00a0"),
}


@pytest.mark.parametrize("name", sorted(FOURTH_PASS_DOORS))
def test_f2_f3_the_git_door_reads_the_same_lines_as_the_raw_door(tmp_path, name):
    summary, ch = FOURTH_PASS_DOORS[name]
    diff = _git_repo(tmp_path, {"tests/test_a.py": "def test_a():\n    pass\n"},
                     {"tests/test_a.py": f"{ch}def test_a():\n    pass\n"})
    assert f"+{ch}def test_a():" in diff
    via_git = gate_diff(summary, tmp_path, "HEAD~1", "HEAD")
    via_text = gate_diff_text(summary, diff)
    got_git = [(c.kind, c.verdict, c.why) for c in via_git.claims]
    assert got_git == [(c.kind, c.verdict, c.why) for c in via_text.claims]
    assert got_git == (REINDENT_AS_CHANGED if ch == "\x0c" else REINDENT_AS_NOTHING)


def test_f2_the_git_door_does_not_forge_an_added_line_from_a_context_line(tmp_path):
    # The git door splits the diff it reads from git exactly as the raw door does. A context line that
    # holds U+2028 and then "+def test_" was cut in two by str.splitlines(), and its second half was
    # read as an added test.
    before = {"tests/test_a.py": "x\u2028+def test_new():\ny = 0\n"}
    after = {"tests/test_a.py": "x\u2028+def test_new():\ny = 1\n"}
    diff = _git_repo(tmp_path, before, after)
    assert " x\u2028+def test_new():" in diff
    via_git = gate_diff("Added 1 test.", tmp_path, "HEAD~1", "HEAD")
    via_text = gate_diff_text("Added 1 test.", diff)
    got_git = [(c.kind, c.verdict, c.why) for c in via_git.claims]
    assert got_git == [(c.kind, c.verdict, c.why) for c in via_text.claims]
    assert got_git == [("tests_added", "CONTRADICTED", "diff adds 0 test functions, claim says 1")]


def test_f2_the_git_door_reads_name_status_as_git_writes_it(tmp_path):
    # `git diff --name-status` is split the same way. With core.quotePath off, git prints a path holding
    # U+2028 raw; str.splitlines() cut it into a path `a` and a stray line.
    name = "a\u2028b.py"
    try:
        (tmp_path / name).write_bytes(b"")
        (tmp_path / name).unlink()
    except OSError:
        pytest.skip("this filesystem cannot hold a U+2028 in a file name")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "core.quotePath", "false"], cwd=tmp_path, check=True)
    diff = _git_repo(tmp_path, {name: "x = 0\n"}, {name: "x = 1\n"})
    assert f"+++ b/{name}" in diff
    via_git = gate_diff("Only touches src/.", tmp_path, "HEAD~1", "HEAD")
    via_text = gate_diff_text("Only touches src/.", diff)
    got_git = [(c.kind, c.verdict, c.why) for c in via_git.claims]
    assert got_git == [(c.kind, c.verdict, c.why) for c in via_text.claims]
    assert got_git == [("only_touches", "CONTRADICTED", f"paths outside 'src': [{name!r}]")]


# ─────────────────────────────── NOTE_path2_fifth_pass_2026_09_25: V-1 (one reading of a definition line)

# Python's `\s` (29 characters) and U+FEFF: every character either port has read as a space.
PY_WS_AND_BOM = [chr(c) for c in range(0x110000) if re.match(r"\s", chr(c))] + ["\ufeff"]
TOKENIZER_WS = (" ", "\t", "\x0c")        # CPython's indentation and token separator; a U+FEFF may open a file
TP, SP = "tests/test_a.py", "src/m.py"
V1_SHAPES = {
    # name: (summary, path, hunk with the character at {c})
    "t-lead-new": ("Added 0 tests. Added 1 test.", TP, "@@ -1 +1,3 @@\n x = 0\n+{c}def test_new():\n+    pass\n"),
    "t-lead-reindent": ("Added 0 tests. Added 1 test.", TP, "@@ -1,2 +1,2 @@\n-def test_a():\n+{c}def test_a():\n     pass\n"),
    "t-lead-removed": ("Added 0 tests. Added 1 test.", TP, "@@ -1,2 +1,2 @@\n-{c}def test_a():\n+def test_a():\n     pass\n"),
    "s-lead-new": ("Adds function foo.", SP, "@@ -1 +1,3 @@\n x = 0\n+{c}def foo():\n+    pass\n"),
    "s-lead-reindent": ("Adds function foo.", SP, "@@ -1,2 +1,2 @@\n-def foo():\n+{c}def foo():\n     pass\n"),
    "s-lead-removed": ("Adds function foo.", SP, "@@ -1,2 +1,2 @@\n-{c}def foo():\n+def foo():\n     pass\n"),
    "s-sep-new": ("Adds function foo.", SP, "@@ -1 +1,3 @@\n x = 0\n+def{c}foo():\n+    pass\n"),
    "s-sep-changed": ("Adds function foo.", SP, "@@ -1,2 +1,2 @@\n-def{c}foo():\n+def{c}foo(x):\n     pass\n"),
    "s-end-new": ("Adds function foo.", SP, "@@ -1 +1,3 @@\n x = 0\n+def foo{c}():\n+    pass\n"),
    "s-end-changed": ("Adds function foo.", SP, "@@ -1,2 +1,2 @@\n-def foo{c}():\n+def foo{c}(x):\n     pass\n"),
}
ADDED_1 = [("tests_added", "CONTRADICTED", "diff adds 1 test functions, claim says 0"),
           ("tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1")]
FOO = {"V": [("symbol_added", "VERIFIED", "added lines do define function 'foo'")],
       "C": [("symbol_added", "CONTRADICTED", "added lines do NOT define function 'foo'")],
       "U": [("symbol_added", "UNCHECKABLE", "added lines define function 'foo' only where the removed lines of "
                                             "the same file define it too; a changed definition is not an added "
                                             "one (#101)")]}


def _v1_expected(shape: str, ch: str) -> list:
    """What each shape reads as: a character is indentation (and a keyword separator) exactly when
    CPython's tokenizer takes it as one -- U+FEFF only where it opens line 1 of a file (NOTE_path2_sixth_pass
    W-1: the `-new` shapes put the definition on line 2) -- and a name ends at an ASCII character that
    cannot continue it."""
    lead = ch in TOKENIZER_WS or (ch == "\ufeff" and not shape.endswith("-new"))
    table = {
        "t-lead-new": ADDED_1 if lead else REINDENT_AS_NOTHING,
        "t-lead-reindent": REINDENT_AS_CHANGED if lead else REINDENT_AS_NOTHING,
        "t-lead-removed": REINDENT_AS_CHANGED if lead else ADDED_1,
        "s-lead-new": FOO["V" if lead else "C"], "s-lead-reindent": FOO["U" if lead else "C"],
        "s-lead-removed": FOO["U" if lead else "V"],
        "s-sep-new": FOO["V" if ch in TOKENIZER_WS else "C"], "s-sep-changed": FOO["U" if ch in TOKENIZER_WS else "C"],
        "s-end-new": FOO["V" if ord(ch) < 0x80 else "C"], "s-end-changed": FOO["U" if ord(ch) < 0x80 else "C"],
    }
    return table[shape]


def _v1_raw(shape: str, ch: str) -> tuple:
    summary, path, hunk = V1_SHAPES[shape]
    return summary, f"--- a/{path}\n+++ b/{path}\n" + hunk.format(c=ch)


def test_v1_every_definition_pattern_opens_with_one_indent():
    # Rounds 2, 3 and 4 each repaired one character in one pattern and opened the next. The class: the
    # patterns that must read the same lines are one pattern. For every character in the set, in every
    # position it can take on a definition line, `got` counts a line exactly when the added-side test
    # pairing reads it, and `hit` reads a line exactly when the added-side symbol pairing does.
    # NOTE_path2_sixth_pass W-2: in EVERY position, the separator after `def` and the character after
    # the test name included (the fifth pass checked the leading positions only for tests), and a
    # U+FEFF read nowhere by the definition reading itself (W-1's parse drops one at line 1 only).
    assert len(PY_WS_AND_BOM) == 30
    for ch in PY_WS_AND_BOM:
        if ch == "\n":          # no line holds it: F-2 splits there, in both ports
            continue
        for line in (f"{ch}def test_a():", f"\ufeff{ch}def test_a():", f"{ch}async def test_a():",
                     f"{ch}{ch}def test_a():", f"def{ch}test_a():", f"def{ch}{ch}test_a():", f"def test_a{ch}():",
                     f"def  test_a{ch}():"):
            assert dg._added_tests(line) == (1 if dg._test_name(line) else 0), ascii(line)
        assert bool(dg._test_name(f"{ch}def test_a():")) == (ch in TOKENIZER_WS), ascii(ch)
        assert bool(dg._test_name(f"def{ch}test_a():")) == (ch in TOKENIZER_WS), ascii(ch)
        assert dg._test_name(f"def test_a{ch}():") == ("test_a" if ord(ch) < 0x80 else None), ascii(ch)
        for line in (f"{ch}def foo():", f"def{ch}foo():", f"def foo{ch}():", f"{ch}class foo:", f"\ufeff{ch}def foo():"):
            added = dg._defines(line, "foo")
            assert dg._symbol_hit("foo", line) == added, ascii(line)
            assert dg._defines(line, "foo", removed=True) == added, ascii(line)   # no `async` here
        # the removed side alone reads `async`, for both kinds
        assert not dg._defines(f"{ch}async def foo():", "foo")
        assert dg._defines(f"{ch}async def foo():", "foo", removed=True) == (ch in TOKENIZER_WS)
        assert bool(dg._test_name(f"async{ch}def test_a():", removed=True)) == (ch in TOKENIZER_WS)
    # a two-space separator is CPython's, for tests as for symbols (the fifth pass read one space only)
    assert dg._test_name("def  test_a():") == "test_a" and dg._added_tests("def  test_a():") == 1
    assert dg._defines("def  foo():", "foo")
    # the pairing's ADDED count reads what `hit` reads, so an added `async def` beside a changed `def`
    # does not make the file add more definitions than it removes
    sides = {"src/m.py": (["def foo(x):", "async def foo():"], ["def foo():"])}
    assert dg._definition_only_changed("foo", sides, {"src/m.py": "M"})


@pytest.mark.parametrize("ch", [c for c in PY_WS_AND_BOM if c not in "\r\n"], ids=lambda c: f"{ord(c):04x}")
def test_v1_the_grid_on_the_raw_door(ch):
    # \r and \n are line breaks by F-2 in both ports, so they make a different diff; the other 28 are here.
    for shape in V1_SHAPES:
        summary, diff = _v1_raw(shape, ch)
        _, got = _claims(summary, diff)
        assert got == _v1_expected(shape, ch), (shape, ascii(ch))


def test_v1_the_port_reads_the_grid_as_the_python_does():
    # The 300 raw-door inputs (all 30 characters, \r and \n included), through the port as well: a
    # committed receipt of the agreement the note records, not a number typed once.
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    items = []
    for shape in V1_SHAPES:
        for ch in PY_WS_AND_BOM:
            summary, diff = _v1_raw(shape, ch)
            g = gate_diff_text(summary, diff, run=None, strict=False)
            items.append({"summary": summary, "diff": diff, "py": [[c.kind, c.verdict, c.why] for c in g.claims]})
    script = ("const {gateDiffText}=require(process.argv[1]);let s='';process.stdin.on('data',d=>s+=d);"
              "process.stdin.on('end',()=>{const out=JSON.parse(s).map(it=>gateDiffText(it.summary,it.diff)"
              ".claims.map(c=>[c.kind,c.verdict,c.why]));process.stdout.write(JSON.stringify(out));});")
    r = subprocess.run([node, "-e", script, str(ROOT / "web" / "gate" / "diffgate.js")],
                       input=json.dumps([{"summary": i["summary"], "diff": i["diff"]} for i in items]),
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    js = json.loads(r.stdout)
    disagree = [(i["summary"], ascii(i["diff"])) for i, j in zip(items, js) if i["py"] != j]
    assert len(items) == 300 and disagree == []


V1_DOORS = {
    # the task's two form-feed pairs, and the round-4 review's shapes around them
    "ff-reindented-function": ("Adds function foo.", {SP: "def foo():\n    pass\n"}, {SP: "\x0cdef foo():\n    pass\n"},
                               FOO["U"]),
    "ff-led-new-test": ("Added 0 tests. Added 1 test.", {TP: "x = 0\n"}, {TP: "x = 0\n\x0cdef test_x():\n    pass\n"},
                        ADDED_1),
    "vt-reindented-function": ("Adds function foo.", {SP: "def foo():\n    pass\n"}, {SP: "\x0bdef foo():\n    pass\n"},
                               FOO["C"]),
    "nel-led-new-function": ("Adds function foo.", {SP: "x = 0\n"}, {SP: "x = 0\n\x85def foo():\n"}, FOO["C"]),
    "ls-reindented-test": ("Added 0 tests. Added 1 test.", {TP: "def test_a():\n    pass\n"},
                           {TP: "\u2028def test_a():\n    pass\n"}, REINDENT_AS_NOTHING),
    # NOTE_path2_sixth_pass W-1: a U+FEFF opens a definition only on line 1 of a file, where CPython reads it
    "bom-led-new-function": ("Adds function foo.", {SP: "x = 0\n"}, {SP: "x = 0\n\ufeffdef foo():\n"}, FOO["C"]),
    "bom-at-the-head-of-a-new-file": ("Adds function foo.", {SP: "x = 0\n"},
                                      {SP: "x = 0\n", "src/n.py": "\ufeffdef foo():\n    pass\n"}, FOO["V"]),
}


@pytest.mark.parametrize("name", sorted(V1_DOORS))
def test_v1_the_git_door_reads_the_same_definition_lines_as_the_raw_door(tmp_path, name):
    summary, before, after, want = V1_DOORS[name]
    diff = _git_repo(tmp_path, before, after)
    via_git = gate_diff(summary, tmp_path, "HEAD~1", "HEAD")
    via_text = gate_diff_text(summary, diff)
    got_git = [(c.kind, c.verdict, c.why) for c in via_git.claims]
    assert got_git == [(c.kind, c.verdict, c.why) for c in via_text.claims]
    assert got_git == want


# ─────────────────────────────── NOTE_path2_fifth_pass_2026_09_25: V-4 (a `..` the key used to lose)

def test_v4_a_written_parent_and_a_dotdot_after_a_name_could_hold_anything():
    assert dg._parent_prefix("src/..") == "src/.." and dg._parent_prefix("docs/../") == "docs/.."
    assert dg._parent_prefix(".github/../.") == ".github/.." and dg._parent_prefix("..") == ".."
    for not_parent in ("src/...", "../docs", "src/.", "src", "..docs", ""):
        assert dg._parent_prefix(not_parent) == "", not_parent
    assert dg._could_lie_under("evil/x.py", "../src/../docs") and dg._could_lie_under("evil/x.py", "src/.../docs")
    assert dg._could_lie_under("evil/x.py", "../docs", "../docs/..")
    assert not dg._could_lie_under("evil/x.py", "../docs", "../docs")
    # a `.` segment is the directory itself, not a parent: named segments stay contiguous across it
    assert dg._could_lie_under("lib/src/docs/x.md", "../src/./docs")
    assert not dg._could_lie_under("evil/x.py", "../src/./docs")
    off = "is relative to a directory the diff does not name (#121)"
    # the round-4 review's cases: each was an accusation no reading of the prefix could support
    for summary, path, shown in (
            ("Only modified `src/` and `../src/../docs` as specified.", "docs/a.md", "../src/../docs"),
            ("Only modified `src/` and `../docs/..` as specified.", "evil/x.py", "../docs/.."),
            ("Only touches .github/../.", "github/github/src/f.py", ".github/.."),
            ("Only modified `github/.docs/..` and `.github/` as specified.", "github/f.py", "github/.docs/.."),
            ("Only touches src/..", "lib/a.py", "src/..")):
        _, got = _claims(summary, _m(path))
        assert got == [("only_touches", "UNCHECKABLE", f"prefix {shown!r} {off}")], summary
    # the price, disclosed: `docs/.` followed by a sentence period is read as the parent of docs
    _, got = _claims("Only touches docs/..", _m("lib/a.py"))
    assert got == [("only_touches", "UNCHECKABLE", f"prefix 'docs/..' {off}")]
    # `...` is not a parent: `src/...` still reads as `src`
    _, got = _claims("Only touches src/...", _m("lib/a.py"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'src': ['lib/a.py']")]


def test_v4_the_two_f4_boundaries_round_4_named():
    # (a) the could-hold filter reads the OFF-tree prefixes only: `src` would hold lib/src/x.py by segment
    _, got = _claims("Only modified `src/` and `../docs` as specified.", _m("lib/src/x.py"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'src' and '../docs': ['lib/src/x.py']")]
    # (b) two off-tree prefixes and no on-tree one: R-3 abstains, as for one
    _, got = _claims("Only touches ../a/ and ../docs/.", _m("evil/x.py"))
    assert got == [("only_touches", "UNCHECKABLE", "prefix '../a' is relative to a directory the diff does not name (#121)")]


# ─────────────────────────────── NOTE_path2_fifth_pass_2026_09_25: V-3 (the scorer attributes per claim)

FRONTIER = ROOT / "papers" / "closed-model-frontier"
_ABSENT = object()


@pytest.fixture(scope="module")
def scorer():
    base = "98a5c368ba9ffa242c6862e021df7f8bad2ed8e6"
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", base + "^{commit}"],
                      capture_output=True).returncode != 0:
        pytest.skip("the scorer's baseline commit is not in this clone (a shallow checkout)")
    import importlib.util
    import sys
    saved = sys.modules.get("styxx.claimdetect", _ABSENT)
    spec = importlib.util.spec_from_file_location("path2_gates_under_test", FRONTIER / "path2_gates.py")
    pg = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(pg)
    except SystemExit as e:                       # the scorer refuses rather than raises
        pytest.skip(f"path2_gates refused to load: {e}")
    finally:
        # the scorer blocks the observer for the whole process; give it back to the other tests
        if saved is _ABSENT:
            sys.modules.pop("styxx.claimdetect", None)
        else:
            sys.modules["styxx.claimdetect"] = saved
    return pg


PROBE = "--- a/.gitignore\n+++ b/.gitignore\n@@ -1,2 +1,2 @@\n x\x0cy\n-a\n+b\n"


def test_v3_a_stray_form_feed_no_longer_excuses_the_round_3_blocker(scorer, monkeypatch):
    # Round 4, both lenses: the fourth pass's F-2 attribution was per RECORD, so a form feed in a
    # context line no claim reads excused every move on the record -- the round-3 blocker included.
    pg = scorer
    src = Path(pg.new.__file__).read_bytes().decode("utf-8")
    good = 'low = _undotted(_norm(raw)).rstrip("/").lower()'
    assert src.count(good) == 1
    blocker = src.replace(good, 'low = _norm(raw).rstrip("/").lower()').encode("utf-8")
    for name in ("new", "CF"):
        monkeypatch.setattr(pg, name, pg._module_from(blocker, f"styxx_diffgate_probe_{name}", f"<probe {name}>"))
    assert pg.split_differs(PROBE)
    t = pg.Tally(name_prs=True)
    t.pair("probe", "Only touches .gitignore.", PROBE, pg.raw_paths(PROBE))
    assert dict(t.violations) == {"G-C4_direction:only_touches": 1}
    assert not t.attribution["moves_admitted_by_rule"]


def test_v3_a_real_f2_move_is_still_attributed_and_counted(scorer):
    pg = scorer
    t = pg.Tally(name_prs=True)
    t.pair("clean", "Only touches .gitignore.", PROBE, pg.raw_paths(PROBE))
    ff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1,2 @@\n x = 0\n+\x0cdef foo():\n"
    t.pair("ff", "Adds function foo.", ff, pg.raw_paths(ff))
    assert not t.violations and t.n["records_moved"] == 1
    assert dict(t.attribution["moves_admitted_by_rule"]) == {"F-2 symbol_added: CONTRADICTED -> VERIFIED": 1}
    assert dict(t.attribution["new_verified_admitted"]) == {"F-2 symbol_added": 1}


def test_v3_a_compat2_flip_only_f2_explains_is_admitted_and_no_other(scorer, monkeypatch):
    # G-C6: a compat2_candidate flip a TABLE rule explains still fails, as the preregistration froze it.
    # Planted: COMPAT's scaffold test reads the dotted key again (the round-2 defect C-2 repaired), so
    # `.storybook/` stops being scaffolding and the candidate flips; reverting #121 gives it back.
    pg = scorer
    src = Path(pg.new.__file__).read_bytes().decode("utf-8")
    good = "not _COMPAT_SCAFFOLD.search(_undotted(path))"
    assert src.count(good) == 1
    planted = src.replace(good, "not _COMPAT_SCAFFOLD.search(path)").encode("utf-8")
    for name in ("new", "CF"):
        monkeypatch.setattr(pg, name, pg._module_from(planted, f"styxx_diffgate_c6_{name}", f"<c6 {name}>"))
    t = pg.Tally(name_prs=True)
    t.pair("storybook", "No breaking changes.", STORYBOOK, pg.raw_paths(STORYBOOK))
    assert t.violations["G-C6_compat2_candidate_flipped"] == 1
    assert not t.attribution["compat2_flips_admitted"]


def test_v3_a_move_no_rule_explains_is_refused_whatever_the_table_says(scorer, monkeypatch):
    # Planted: a reason no rule touches is reworded. No revert, alone or in twos or threes, gives the
    # baseline back, so the move is refused by the counterfactual itself -- and a counterfactual that
    # compared verdicts only would have credited every rule with it.
    pg = scorer
    src = Path(pg.new.__file__).read_bytes().decode("utf-8")
    good = 'c.why = f"diff changes {len(status)} files, claim says {n}"'
    assert src.count(good) == 1
    planted = src.replace(good, 'c.why = f"diff changes {len(status)} files; claim says {n}"').encode("utf-8")
    for name in ("new", "CF"):
        monkeypatch.setattr(pg, name, pg._module_from(planted, f"styxx_diffgate_none_{name}", f"<none {name}>"))
    diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n"
    t = pg.Tally(name_prs=True)
    t.pair("reworded", "1 file changed.", diff, pg.raw_paths(diff))
    assert dict(t.violations) == {"G-C4_unattributed_counterfactual:files_changed_count": 1}


def test_v3_the_post_amendment_rules_admit_only_their_own_kinds():
    import importlib.util
    spec = importlib.util.spec_from_file_location("path2_gates_admits_only", FRONTIER / "path2_gates.py")
    src = Path(spec.origin).read_text(encoding="utf-8")
    # `admits` is read out of the scorer's source and run alone: no git, no baseline needed
    body = src[src.index("def admits("):src.index("# ── attribution, written out")]
    # NOTE_path2_sixth_pass: F-2 and W-1 admit only on a diff where they can act; stubbed here by name
    ns = {"OFF_TREE_WHY": "is relative to a directory the diff does not name (#121)",
          "split_differs": lambda diff: diff == "SPLIT", "parse_differs": lambda diff: diff == "PARSE"}
    exec(body, ns)  # noqa: S102
    admits = ns["admits"]
    off = "prefix '../docs/..' is relative to a directory the diff does not name (#121)"
    assert admits("F-2", "compat_claim", "UNCHECKABLE", "UNCHECKABLE", "", "SPLIT")
    assert not admits("F-2", "compat_claim", "UNCHECKABLE", "UNCHECKABLE", "", "PARSE")
    assert admits("W-1", "files_changed_count", "CONTRADICTED", "VERIFIED", "", "PARSE")
    assert not admits("W-1", "files_changed_count", "CONTRADICTED", "VERIFIED", "", "SPLIT")
    assert admits("W-2", "symbol_added", "CONTRADICTED", "VERIFIED", "") and not admits("W-2", "only_touches", "V", "C", "")
    assert admits("R-1", "tests_added", "VERIFIED", "UNCHECKABLE", "") and not admits("R-1", "symbol_added", "V", "C", "")
    assert admits("F-3", "tests_added", "VERIFIED", "CONTRADICTED", "") and not admits("F-3", "only_touches", "V", "C", "")
    assert admits("V-1", "symbol_added", "VERIFIED", "UNCHECKABLE", "") and not admits("V-1", "file_touched", "V", "U", "")
    assert admits("V-4", "only_touches", "CONTRADICTED", "UNCHECKABLE", off)
    # NOTE_path2_seventh_pass: A-1 only abstains a test count
    assert admits("A-1", "tests_added", "CONTRADICTED", "UNCHECKABLE", "") and admits("A-1", "tests_added", "VERIFIED", "UNCHECKABLE", "")
    assert not admits("A-1", "tests_added", "VERIFIED", "CONTRADICTED", "") and not admits("A-1", "symbol_added", "VERIFIED", "UNCHECKABLE", "")
    assert not admits("V-4", "only_touches", "VERIFIED", "CONTRADICTED", "paths outside 'src': ['x']")
    assert not admits("V-4", "only_touches", "VERIFIED", "UNCHECKABLE", "prefix 'x' is not a path (#110)")
    with pytest.raises(ValueError):
        admits("#121", "only_touches", "VERIFIED", "UNCHECKABLE", "")


def test_v3_every_revert_names_code_the_instrument_has(scorer):
    pg = scorer
    for rule in pg.RULES:
        with pg.reverted(pg.CF, {rule}):
            pass
    with pg.reverted(pg.CF, pg.RULES):
        # every rule reverted at once gives the baseline back on the pinned pairs
        for p in json.loads(PAIRS.read_text(encoding="utf-8")):
            a = [(c.kind, c.verdict, c.why) for c in pg.CF.gate_diff_text(p["summary"], p["diff"]).claims]
            b = [(c.kind, c.verdict, c.why) for c in pg.BASE.gate_diff_text(p["summary"], p["diff"]).claims]
            assert a == b, p["id"]
        # ... and on the definition grid's cells for the characters CPython reads as indentation, where
        # V-1's revert (the fourth pass) and W-2's (the fifth) differ: the older rule's code must win
        for shape in V1_SHAPES:
            for ch in TOKENIZER_WS + ("\ufeff",):
                summary, diff = _v1_raw(shape, ch)
                a = [(c.kind, c.verdict, c.why) for c in pg.CF.gate_diff_text(summary, diff).claims]
                b = [(c.kind, c.verdict, c.why) for c in pg.BASE.gate_diff_text(summary, diff).claims]
                assert a == b, (shape, ascii(ch))


def test_v3_w1_and_w2_are_credited_only_with_what_their_reverts_give_back(scorer, monkeypatch):
    pg = scorer
    # W-2's revert restores the fifth pass's claimed name as well as its definition reading: a name
    # holding U+00B2 (Python's `\w`, not an identifier character) read VERIFIED on main and at the
    # fifth pass and reads CONTRADICTED now (CPython refuses the line); both V-1's and W-2's reverts
    # give it back, and both are credited.
    diff = "--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1,2 @@\n x = 0\n+def foo\u00b2():\n"
    t = pg.Tally(name_prs=True)
    t.pair("sup2", "Added function foo\u00b2.", diff, pg.raw_paths(diff))
    assert not t.violations and dict(t.attribution["attributed_by"]) == {"V-1+W-2": 1}
    # V-1 and W-2 patch the same names; with both reverted the OLDER rule's code (the fourth pass) wins
    with pg.reverted(pg.CF, {"V-1", "W-2"}):
        assert pg.CF._defines is pg._defines_fourth_pass and pg.CF._symbol_hit is pg._symbol_hit_fourth_pass
        assert pg.CF._test_name("\x0cdef test_a():") is None           # the fourth pass's indent is [ \t]*
    with pg.reverted(pg.CF, {"W-2"}):
        assert pg.CF._test_name("\x0cdef test_a():") == "test_a"       # the fifth pass's is [ \t\f]*
    # a defect planted in the one parse (a file dropped from the status map) reverts with W-1, but on a
    # diff whose hunks hold no `---`/`+++` line and no line-1 U+FEFF W-1 cannot act, so it is refused
    _planted(pg, monkeypatch, "    return status, added, sides",
             "    return {k: v for k, v in status.items() if k != 'src/zz.py'}, added, sides", "w1")
    two = _m("src/zz.py") + _m("src/a.py")
    assert not pg.parse_differs(two)
    t = pg.Tally(name_prs=True)
    t.pair("zz", "2 files changed.", two, pg.raw_paths(two))
    # (NOTE_path2_seventh_pass: the scorer's own reading of the parse refuses it too, G-C7)
    assert dict(t.violations) == {"G-C4_direction:files_changed_count:W-1": 1, "G-C7_oracle:W-1_status": 1}


# ─────────────────────────────── NOTE_path2_sixth_pass_2026_09_25: W-1 (one hunk-aware parse)

def _port_claims(items: list) -> list:
    """[(summary, diff)] through web/gate/diffgate.js: [[kind, verdict, why], ...] per item."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const {gateDiffText}=require(process.argv[1]);let s='';process.stdin.on('data',d=>s+=d);"
              "process.stdin.on('end',()=>{const out=JSON.parse(s).map(it=>gateDiffText(it.summary,it.diff)"
              ".claims.map(c=>[c.kind,c.verdict,c.why]));process.stdout.write(JSON.stringify(out));});")
    r = subprocess.run([node, "-e", script, str(ROOT / "web" / "gate" / "diffgate.js")],
                       input=json.dumps([{"summary": s, "diff": d} for s, d in items]),
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return [[tuple(c) for c in cl] for cl in json.loads(r.stdout)]


SQL_BEFORE = 'Q = """\n-- users\nSELECT 1\n"""\n\n\ndef foo():\n    return Q\n'
CHANGED_FOO = ("symbol_added", "UNCHECKABLE", "added lines define function 'foo' only where the removed lines of "
                                              "the same file define it too; a changed definition is not an added "
                                              "one (#101)")
W1_DOORS = {
    # round-5 review, correctness lens, major 2: `--- users` and `+++ x` inside a hunk are content
    "removed-sql-comment-then-changed-def": (
        "Added function foo. 1 file changed. Only touches src/.", {SP: SQL_BEFORE},
        {SP: SQL_BEFORE.replace("-- users\n", "").replace("def foo():", "def foo(x):")},
        [CHANGED_FOO, ("files_changed_count", "VERIFIED", "diff changes 1 files, claim says 1"),
         ("only_touches", "VERIFIED", "all changed paths under prefix")]),
    "removed-sql-comment-then-changed-test": (
        "Added 1 test. Added 0 tests.", {TP: SQL_BEFORE.replace("def foo():", "def test_a():")},
        {TP: SQL_BEFORE.replace("-- users\n", "").replace("def foo():", "def test_a(tmp_path):")},
        [("tests_added", "UNCHECKABLE", "diff adds 0 test functions and changes 1, claim says 1; a changed test "
                                        "is not an added one (#101)"),
         ("tests_added", "VERIFIED", "diff adds 0 test functions, claim says 0 (1 changed, not added: #101)")]),
    "added-plus-plus-space-line": (
        "Only touches src/. 1 file changed.", {"src/a.txt": "a\n"}, {"src/a.txt": "a\n++ x\n"},
        [("only_touches", "VERIFIED", "all changed paths under prefix"),
         ("files_changed_count", "VERIFIED", "diff changes 1 files, claim says 1")]),
    "a-hunk-with-both": (
        "Added function foo. 1 file changed. Only touches src/.", {SP: SQL_BEFORE},
        {SP: SQL_BEFORE.replace("-- users", "++ x").replace("def foo():", "def foo(x):")},
        [CHANGED_FOO, ("files_changed_count", "VERIFIED", "diff changes 1 files, claim says 1"),
         ("only_touches", "VERIFIED", "all changed paths under prefix")]),
    "removed-sql-comment-then-removed-public-def": (
        "No breaking changes.", {SP: 'Q = """\n-- users\n"""\n\n\ndef public_api():\n    pass\n'},
        {SP: 'Q = """\n"""\n'},
        [("compat_claim", "UNCHECKABLE", "compatibility claimed; the diff removes 1 public definition(s) from the "
                                         "surface, not re-defined in the added lines: src/m.py: public_api")]),
    "added-plus-plus-line-is-a-reference": (
        "No breaking changes.", {SP: "def foo():\n    pass\n"}, {SP: "++foo()\n"},
        [("compat_claim", "UNCHECKABLE", "compatibility claimed; no public top-level definition removed "
                                         "(python read; behaviour beyond names not checked)")]),
}


@pytest.mark.parametrize("name", sorted(W1_DOORS))
def test_w1_the_one_parse_reads_every_door_alike(tmp_path, name):
    summary, before, after, want = W1_DOORS[name]
    diff = _git_repo(tmp_path, before, after)
    via_git = [(c.kind, c.verdict, c.why) for c in gate_diff(summary, tmp_path, "HEAD~1", "HEAD").claims]
    _, via_text = _claims(summary, diff)
    assert via_git == via_text == want
    assert _port_claims([(summary, diff)]) == [want]


def test_w1_the_blob_and_the_sides_are_one_reading():
    # got and hit read the blob, the pairings read the sides: the same added lines, in the same order,
    # whatever a line's text opens with (the fifth pass's two parsers kept different lines here)
    for diff in (PROBE, _v1_raw("t-lead-new", "\x0c")[1], _m("src/a.py") + _m("lib/b.py")):
        blob = parse_unified_diff(diff)[1]
        assert blob.split("\n") == [a for added, _r in parse_unified_diff_sides(diff).values() for a in added]
    diff =("--- a/src/m.py\n+++ b/src/m.py\n@@ -1,3 +1,3 @@\n Q = 1\n--- users\n+++ x\n-def foo():\n"
            "+def foo(x):\n")
    status, blob = parse_unified_diff(diff)
    assert status == {"src/m.py": "M"} and blob == "++ x\ndef foo(x):"
    assert parse_unified_diff_sides(diff) == {"src/m.py": (["++ x", "def foo(x):"], ["-- users", "def foo():"])}
    # NOTE_path2_seventh_pass: the same lines under counts they do not carry (4 and 4) are read as main reads them
    over = diff.replace("@@ -1,3 +1,3 @@", "@@ -1,4 +1,4 @@")
    assert parse_unified_diff(over)[0] == {"src/m.py": "M", "x": "M"}


def test_w1_a_hunk_that_declares_too_many_lines_still_ends_at_the_next_file_header():
    # hand-written hunks often declare more lines than they carry; the next file's `---`/`+++` pair then
    # falls inside the counts. NOTE_path2_seventh_pass: such a hunk is not EXACT, and is read as main reads
    # it, so the pair is a header (the sixth pass's `_header_pair` exception is gone)
    sloppy = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1,3 +1,4 @@\n x\n+y = 1\n--- a/lib/b.py\n+++ b/lib/b.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert parse_unified_diff(sloppy)[0] == {"src/a.py": "M", "lib/b.py": "M"}
    lines = sloppy.split("\n")
    assert not dg._hunk_is_exact(lines, 2) and dg._hunk_is_exact(lines, lines.index("@@ -1 +1 @@"))
    # the limit this leaves: `-- a` and `++ b` as the last lines of a hunk, right before the next `@@`,
    # read as a file header, as main reads them (path2:w1-limit-...-read-as-a-file-header)
    limit = "--- a/src/a.sql\n+++ b/src/a.sql\n@@ -1,2 +1,2 @@\n x\n--- old\n+++ new\n@@ -9 +9 @@\n-a\n+b\n"
    assert set(parse_unified_diff(limit)[0]) == {"src/a.sql", "new"}


def test_w1_a_bom_is_read_where_cpython_reads_it_at_line_one_only():
    head = "--- /dev/null\n+++ b/tests/test_n.py\n@@ -0,0 +1,2 @@\n+\ufeffdef test_n():\n+    pass\n"
    mid = "--- a/tests/test_n.py\n+++ b/tests/test_n.py\n@@ -1 +1,3 @@\n x = 0\n+\ufeffdef test_n():\n+    pass\n"
    assert parse_unified_diff(head)[1] == "def test_n():\n    pass"
    assert parse_unified_diff(mid)[1] == "\ufeffdef test_n():\n    pass"
    assert _claims("Added 1 test.", head)[1] == [("tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1")]
    assert _claims("Added 1 test.", mid)[1] == [("tests_added", "CONTRADICTED", "diff adds 0 test functions, claim says 1")]
    removed = "--- a/tests/test_b.py\n+++ b/tests/test_b.py\n@@ -1 +1 @@\n-\ufeffdef test_a():\n+def test_a():\n"
    assert parse_unified_diff_sides(removed) == {"tests/test_b.py": (["def test_a():"], ["def test_a():"])}
    # outside any counted hunk there is no line number, and the U+FEFF stays
    assert parse_unified_diff("--- a/t.py\n+++ b/t.py\n+\ufeffdef test_a():\n")[1] == "\ufeffdef test_a():"


# ─────────────────────────────── NOTE_path2_sixth_pass_2026_09_25: W-2 (one reading of a name)

def test_w2_a_name_is_the_python_identifier_that_starts_there():
    for text, want in (("foo(", "foo"), ("caf\u00e9()", "caf\u00e9"), ("col\u00b7leccio.", "col\u00b7leccio"),
                       ("foo\u203fbar", "foo\u203fbar"), ("foo\u00b2bar", "foo"), ("foo\u00a0()", "foo"),
                       ("x\u0301y", "x\u0301y"), ("1abc", ""), ("_a1", "_a1"), ("test_a\x0c()", "test_a"),
                       ("test_a[T]", "test_a"), ("calc\u8ba1\u7b97 ", "calc\u8ba1\u7b97"), ("", "")):
        assert dg._identifier_at(text, 0) == want, ascii(text)
        assert want == "" or want.isidentifier()
    m = dg._TEMPLATES[7][1].search("Added function col\u00b7leccio.")
    assert m.group("name") == "col" and dg._claimed_name("Added function col\u00b7leccio.", m) == ("col\u00b7leccio", None)


W2_XS = ["\u00e9", "\u00ef", "\u00df", "\u00f6", "\u00f1", "\u0107", "\u011f", "\u03a9", "\u0434", "\u8ba1",
         "\u00b2", "\u0301", "\u00b7", "\u0660", "\u203f", "\u094d"]


def _w2_name_cells() -> list:
    cells = []
    for x in W2_XS:
        for kind, kw in (("function", "def"), ("class", "class")):
            for name in (f"foo{x}bar", f"foo{x}"):
                line = f"+{kw} {name}():" if kw == "def" else f"+{kw} {name}:"
                cells.append((f"Added {kind} {name}.", f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,2 @@\n x = 0\n{line}\n",
                              kind, dg._identifier_at(name, 0)))
    return cells


def test_w2_a_claimed_non_ascii_name_reads_as_its_definition_on_both_ports():
    # Round-5 review, correctness lens, major 1: 52 of these 64 true claims read CONTRADICTED on the
    # port at the fifth pass, and the combining-mark, middle-dot and connector cases on the Python too.
    # The claimed name and the definition's name are now one reading, so each verifies, naming the
    # identifier both sides read -- except U+00B2, which no identifier holds: CPython refuses
    # `def foo<U+00B2>bar():` and the line defines nothing (main read it VERIFIED through `\b`).
    # NOTE_path2_seventh_pass: a claimed name that runs on into U+00B2 (a word character no identifier
    # holds) names no identifier, and one that ends in U+00B7 names none for certain; both are UNCHECKABLE.
    cells = _w2_name_cells()
    assert len(cells) == 64
    py = []
    for summary, diff, kind, ident in cells:
        _, got = _claims(summary, diff)
        if "\u00b2" in summary:
            assert got == [("symbol_added", "UNCHECKABLE", f"the claimed name runs past {ident!r} into '\u00b2' "
                            "(U+00B2), which no Python identifier holds; no definition is read for it")]
        elif summary.endswith("\u00b7."):
            assert got == [("symbol_added", "UNCHECKABLE", f"the claimed name {ident!r} ends in '\u00b7' (U+00B7), "
                            "which prose also writes after a word; no definition is read for it")]
        else:
            assert got == [("symbol_added", "VERIFIED", f"added lines do define {kind} {ident!r}")], ascii(summary)
        py.append(got)
    assert _port_claims([(s, d) for s, d, _k, _i in cells]) == py
    # the claim's detail keeps the template's `name` group; the port reads it with Python's `\w` now
    # (JavaScript's `\w` is ASCII and stored `foo` for `foo<U+00E9>bar`), so the whole record agrees
    script = ("const {gateDiffText}=require(process.argv[1]);let s='';process.stdin.on('data',d=>s+=d);"
              "process.stdin.on('end',()=>{process.stdout.write(JSON.stringify(JSON.parse(s).map(it=>"
              "gateDiffText(it.summary,it.diff).claims.map(c=>c.detail.name))));});")
    r = subprocess.run([shutil.which("node"), "-e", script, str(ROOT / "web" / "gate" / "diffgate.js")],
                       input=json.dumps([{"summary": s, "diff": d} for s, d, _k, _i in cells]),
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    names = [[c.detail["name"] for c in gate_diff_text(s, d, run=None, strict=False).claims] for s, d, _k, _i in cells]
    assert json.loads(r.stdout) == names


W2_TEST_SHAPES = {
    # (hunk, the readings of "Added 0 tests. Added 1 test.")
    "t-sep-new-tab": ("@@ -1 +1,3 @@\n x = 0\n+def\ttest_new():\n+    pass\n", ADDED_1),
    "t-sep-new-ff": ("@@ -1 +1,3 @@\n x = 0\n+def\x0ctest_new():\n+    pass\n", ADDED_1),
    "t-sep-new-two-spaces": ("@@ -1 +1,3 @@\n x = 0\n+def  test_new():\n+    pass\n", ADDED_1),
    "t-sep-changed-tab": ("@@ -1,2 +1,2 @@\n-def\ttest_a():\n+def test_a(x):\n     pass\n", REINDENT_AS_CHANGED),
    "t-sep-changed-ff": ("@@ -1,2 +1,2 @@\n-def\x0ctest_a():\n+def test_a(x):\n     pass\n", REINDENT_AS_CHANGED),
    "t-end-new-ff": ("@@ -1 +1,3 @@\n x = 0\n+def test_new\x0c():\n+    pass\n", ADDED_1),
    "t-end-changed-ff": ("@@ -1,2 +1,2 @@\n-def test_a():\n+def test_a\x0c():\n     pass\n", REINDENT_AS_CHANGED),
    "t-end-changed-back": ("@@ -1,2 +1,2 @@\n-def test_a\x0c():\n+def test_a():\n     pass\n", REINDENT_AS_CHANGED),
    "t-made-generic": ("@@ -1,2 +1,2 @@\n-def test_a():\n+def test_a[T]():\n     pass\n", REINDENT_AS_CHANGED),
    "t-unmade-generic": ("@@ -1,2 +1,2 @@\n-def test_a[T]():\n+def test_a():\n     pass\n", REINDENT_AS_CHANGED),
    "t-end-new-nbsp": ("@@ -1 +1,3 @@\n x = 0\n+def test_new\u00a0():\n+    pass\n", REINDENT_AS_NOTHING),
}


@pytest.mark.parametrize("shape", sorted(W2_TEST_SHAPES))
def test_w2_a_test_name_and_its_separator_are_read_as_cpython_reads_them(shape):
    # Round-5 review, both lenses: the test side read one literal space after `def` and ran the name to
    # `[^ \t(:]*`, so a tab or form feed separator went uncounted and a form feed or `[` after the name
    # became part of it -- a changed test read as an added one. `_DEF_SEP` and the identifier, as symbols.
    hunk, want = W2_TEST_SHAPES[shape]
    summary, diff = "Added 0 tests. Added 1 test.", f"--- a/{TP}\n+++ b/{TP}\n{hunk}"
    assert _claims(summary, diff)[1] == want
    assert _port_claims([(summary, diff)]) == [want]


# The twelve name-end cells of the fifth pass's grid (six characters after the name, new and changed),
# committed as tests at last: an ASCII character that cannot continue a name ends it; a letter, a mark or
# U+00B7 continues it, so the name is another name.
NAME_END_CELLS = {"\u00e9": ("C", "C"), "\u00b7": ("C", "C"), "\u0301": ("C", "C"), "\u4e00": ("C", "C"),
                  "[": ("V", "U"), "x": ("C", "C")}


@pytest.mark.parametrize("ch", sorted(NAME_END_CELLS), ids=lambda c: f"{ord(c):04x}")
def test_w2_the_twelve_name_end_cells_on_both_doors_and_the_port(tmp_path, ch):
    new_want, changed_want = NAME_END_CELLS[ch]
    got_raw, items = [], []
    for shape, want in (("s-end-new", new_want), ("s-end-changed", changed_want)):
        summary, diff = _v1_raw(shape, ch)
        assert _claims(summary, diff)[1] == FOO[want], (shape, ascii(ch))
        items.append((summary, diff))
        got_raw.append(FOO[want])
    assert _port_claims(items) == got_raw
    # the git door, on the new-definition cell
    diff = _git_repo(tmp_path, {SP: "x = 0\n"}, {SP: f"x = 0\ndef foo{ch}():\n    pass\n"})
    via_git = [(c.kind, c.verdict, c.why) for c in gate_diff("Adds function foo.", tmp_path, "HEAD~1", "HEAD").claims]
    assert via_git == _claims("Adds function foo.", diff)[1] == FOO[new_want]


# ─────────────────────────────── NOTE_path2_sixth_pass_2026_09_25: V-4, completed

def test_v4_a_bare_dotdot_second_prefix_is_read_before_the_path_shape_test_drops_it():
    two = _m("evil/x.py") + _m("src/a.py")
    off = "prefix '..' is relative to a directory the diff does not name (#121)"
    for summary in ("Only touches `src/` and `..`.", "Only touches `src/` and `../`."):
        assert _claims(summary, two)[1] == [("only_touches", "UNCHECKABLE", off)], summary
    # `...` is an elision, not a parent: dropped as before, and the leading prefix decides alone
    assert _claims("Only touches `src/` and `...`.", two)[1] == [
        ("only_touches", "CONTRADICTED", "paths outside 'src': ['evil/x.py']")]
    # a bare `..` standing alone is still not a path, as on main
    assert _claims("Only touches `..`.", two)[1] == [("only_touches", "UNCHECKABLE", "prefix '' is not a path (#110)")]


# ─────────────────────────────── NOTE_path2_sixth_pass_2026_09_25: the scorer's admission rules

def _planted(pg, monkeypatch, good: str, bad: str, tag: str) -> None:
    src = Path(pg.new.__file__).read_bytes().decode("utf-8")
    assert src.count(good) == 1, good
    planted = src.replace(good, bad).encode("utf-8")
    for name in ("new", "CF"):
        monkeypatch.setattr(pg, name, pg._module_from(planted, f"styxx_diffgate_{tag}_{name}", f"<{tag} {name}>"))


def test_v3_every_rule_in_an_attribution_must_admit_the_move(scorer, monkeypatch):
    # Round-5 protocol lens (mutant S1): when a table rule and a post-amendment rule each give the
    # baseline back, BOTH are the attribution and the table must still be asked. Planted: a reason no
    # rule touches is reworded, and the counterfactual is told that reverting #121, or F-2, gives it
    # back. The table refuses (#121 has no moved key to stand on), so the move fails -- crediting F-2
    # alone would have admitted it on this diff, whose splits differ.
    pg = scorer
    _planted(pg, monkeypatch, 'c.why = f"diff changes {len(status)} files, claim says {n}"',
             'c.why = f"diff changes {len(status)} files; claim says {n}"', "s1")
    diff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1,2 @@\n x\u2028y\n-a\n+b\n"
    assert pg.split_differs(diff)
    base = pg.BASE.gate_diff_text("1 file changed.", diff, run=None, strict=False).claims
    real = pg.Counterfactual.claims

    def claims(self, rules):
        return base if frozenset(rules) in ({"#121"}, {"F-2"}) else real(self, rules)
    monkeypatch.setattr(pg.Counterfactual, "claims", claims)
    t = pg.Tally(name_prs=True)
    t.pair("both", "1 file changed.", diff, pg.raw_paths(diff))
    assert dict(t.violations) == {"G-C4_unattributed:files_changed_count": 1}


def test_v3_g_c6_admits_a_flip_only_one_rule_explains_alone(scorer, monkeypatch):
    # Round-5 protocol lens (mutant S4): a compat2_candidate flip explained only by F-2 and W-1 jointly
    # is not "F-2 alone", and fails G-C6, although each rule admits a move of its kind on this diff.
    pg = scorer
    _planted(pg, monkeypatch, "not _COMPAT_SCAFFOLD.search(_undotted(path))", "not _COMPAT_SCAFFOLD.search(path)", "s4")
    diff = STORYBOOK.replace("@@ -1 +1 @@\n", "@@ -1,3 +1,2 @@\n x\u2028--- y\n")   # exact under splitlines()
    assert pg.split_differs(diff) and pg.parse_differs(diff)
    base = pg.BASE.gate_diff_text("No breaking changes.", diff, run=None, strict=False).claims
    repaired = pg.new.gate_diff_text("No breaking changes.", diff, run=None, strict=False).claims
    monkeypatch.setattr(pg.Counterfactual, "claims",
                        lambda self, rules: base if frozenset(rules) == {"F-2", "W-1"} else repaired)
    t = pg.Tally(name_prs=True)
    t.pair("joint", "No breaking changes.", diff, pg.raw_paths(diff))
    assert t.violations["G-C6_compat2_candidate_flipped"] == 1
    assert not t.attribution["compat2_flips_admitted"]


def test_v3_f2_and_w1_admit_only_on_a_diff_where_they_can_act(scorer, monkeypatch):
    # Round-5 protocol lens, minor 2: a defect planted inside F-2's own code (a tab read as a line
    # break) reverts with F-2; on a diff with no separator of any kind F-2 cannot act, so it may not
    # be credited. The fifth pass admitted this as a new accusation.
    pg = scorer
    _planted(pg, monkeypatch, '_DIFF_LINE_BREAK = re.compile(r"\\r\\n|\\r|\\n")',
             '_DIFF_LINE_BREAK = re.compile(r"\\r\\n|\\r|\\n|\\t")', "tab")
    diff = "--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1 +1,3 @@\n x = 0\n+class T:\n+\tdef test_a(self):\n"
    assert not pg.split_differs(diff)
    t = pg.Tally(name_prs=True)
    t.pair("tab", "Added 1 test.", diff, pg.raw_paths(diff))
    # (NOTE_path2_seventh_pass: the scorer's own split refuses it too, on any diff, G-C7)
    assert t.violations["G-C4_direction:tests_added:F-2"] == 1 and t.violations["G-C7_oracle:F-2_split"] == 1


def test_v3_a_key_moved_by_anything_but_a_dot_is_refused(scorer, monkeypatch):
    # Round-5 protocol lens (mutant S9): the key-shape check had no test.
    pg = scorer
    _planted(pg, monkeypatch, 'return _LEADING_SLASH_SEGMENTS.sub("", p.replace("\\\\", "/")).lower()',
             'return _LEADING_SLASH_SEGMENTS.sub("", p.replace("\\\\", "/"))', "s9")
    diff = "--- a/Src/A.py\n+++ b/Src/A.py\n@@ -1 +1 @@\n-a\n+b\n"
    t = pg.Tally(name_prs=True)
    t.pair("case", "1 file changed.", diff, pg.raw_paths(diff))
    assert t.violations["G-C4_key_moved_not_by_a_dot"] == 1


def _shelf(tmp_path: Path, prs: list) -> Path:
    import sqlite3
    path = tmp_path / "shelf.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE pr (id INTEGER, title TEXT, body TEXT)")
    con.execute("CREATE TABLE f (pr_id INTEGER, filename TEXT, status TEXT, patch TEXT)")
    for pid, body, files in prs:
        con.execute("INSERT INTO pr VALUES (?, ?, ?)", (pid, "t", body))
        for fn, st, patch in files:
            con.execute("INSERT INTO f VALUES (?, ?, ?, ?)", (pid, fn, st, patch))
    con.commit()
    con.close()
    return path


def test_v3_g_c2_attributes_an_eligibility_move_by_counterfactual(scorer, monkeypatch, tmp_path):
    # Round-5 protocol lens (mutant S8): G-C2's counterfactual had no test. An eligibility move W-1
    # makes on a diff where it acts is attributed to W-1; one a defect in the parse makes on a diff
    # where W-1 cannot act is a violation, although W-1's revert (the fifth-pass parsers) gives the
    # baseline back.
    pg = scorer
    plus = ("@@ -1 +1,2 @@\n a\n+++ x\n")
    shelf = _shelf(tmp_path, [(1, "Only touches src/.", [("src/a.txt", "modified", plus)])])
    out = tmp_path / "gates.json"
    pg.run_corpus(shelf, None, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["G-C2_eligibility"]["moves"] == {"repaired_only": 1, "attributed_to_W-1": 1}
    assert not any(v.startswith("G-C2") for v in payload["violations"])
    _planted(pg, monkeypatch, "    return status, added, sides",
             "    return {k: v for k, v in status.items() if k != 'src/zz.py'}, added, sides", "s8")
    (tmp_path / "planted").mkdir()
    shelf2 = _shelf(tmp_path / "planted", [(2, "Only touches src/.", [("src/zz.py", "modified", "@@ -1 +1 @@\n-a\n+b\n")])])
    pg.run_corpus(shelf2, None, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["G-C2_eligibility"]["moves"] == {"baseline_only": 1}
    assert payload["violations"].get("G-C2_eligibility_moved_without_a_rule") == 1


def test_v3_the_scorer_admits_every_move_on_the_pinned_pairs_and_counts_what_g_c3_waives(scorer):
    # Every pinned pair, every file, through the scorer's own counterfactual: no violation, and the new
    # accusations a post-amendment rule explains -- which G-C3 does not ask about -- are counted.
    pg = scorer
    t = pg.Tally(name_prs=True)
    for name in ("bc1_pairs.json", "compat_pairs.json", "bin1_pairs.json", "compat2_pairs.json", "path1_pairs.json",
                 "declare1_pairs.json", "path2_pairs.json"):
        for p in json.loads((ROOT / "web" / "gate" / "differential" / name).read_text(encoding="utf-8")):
            t.pair(p["id"], p["summary"], p["diff"], pg.raw_paths(p["diff"]))
    assert not t.violations, t.violating
    waived = t.report({"unmodified_against_head": True})["G-C3_no_accusation_added"]["waived_for_post_amendment_rules"]
    assert waived == dict(t.attribution["new_accusations_admitted"]) and sum(waived.values()) > 0


def test_v3_raw_paths_reads_a_header_only_outside_a_hunk(scorer):
    # the scorer's own hunk walk (W-1, written out): a removed `-- a/x` inside a hunk is not a path
    pg = scorer
    diff = "--- a/src/m.sql\n+++ b/src/m.sql\n@@ -1,2 +1,1 @@\n x\n--- a/.env\n"
    assert pg.raw_paths(diff) == ["src/m.sql"]
    assert pg.parse_differs(diff) and not pg.parse_differs(_m("src/a.py"))
    assert pg.parse_differs("--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+\ufeffx = 1\n")


def test_v3_limit_12_a_defect_whose_trigger_a_rule_exposes_is_credited_to_that_rule(scorer, monkeypatch):
    # Round-5 protocol lens, minor 3: the stated blind spot was incomplete. Planted in code no rule
    # touches (`net >= n` for `net == n`), a false VERIFIED whose trigger is the form feed F-2 and V-1
    # expose is credited to them by the counterfactual -- limit 12(b). NOTE_path2_seventh_pass: the claim
    # oracle (G-C7) reads the tests_added claim with the scorer's own code, so the gate now fails.
    pg = scorer
    _planted(pg, monkeypatch, "if net == n:", "if net >= n:", "l12")
    diff = "--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1 +1,3 @@\n x = 0\n+def test_a():\n+\x0cdef test_b():\n"
    t = pg.Tally(name_prs=True)
    t.pair("pr105", "Added 1 test.", diff, pg.raw_paths(diff))
    assert dict(t.violations) == {"G-C7_oracle:tests_added_claim": 1}
    assert dict(t.attribution["moves_admitted_by_rule"]) == {"F-2+V-1 tests_added: VERIFIED -> VERIFIED": 1}


# ─────────────────────────────── NOTE_path2_seventh_pass_2026_09_25: x1, one name table and the claim side

XID_PY = ROOT / "styxx" / "_xid.py"
XID_JS = ROOT / "web" / "gate" / "diffgate.js"


def _gen_xid():
    spec = importlib.util.spec_from_file_location("gen_xid_under_test", ROOT / "web" / "gate" / "gen_xid.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def _port_records(items: list) -> list:
    """[(summary, diff)] through the port: [[kind, verdict, why, detail], ...] per item."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const {gateDiffText}=require(process.argv[1]);let s='';process.stdin.on('data',d=>s+=d);"
              "process.stdin.on('end',()=>{process.stdout.write(JSON.stringify(JSON.parse(s).map(it=>gateDiffText("
              "it.summary,it.diff).claims.map(c=>[c.kind,c.verdict,c.why,c.detail]))));});")
    r = subprocess.run([node, "-e", script, str(XID_JS)], input=json.dumps([{"summary": s, "diff": d} for s, d in items]),
                       capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def _py_records(items: list) -> list:
    return [[[c.kind, c.verdict, c.why, c.detail] for c in gate_diff_text(s, d, run=None, strict=False).claims]
            for s, d in items]


def test_x1_both_ports_carry_one_table_with_its_version_and_hash_pinned():
    from styxx import _xid
    g = _gen_xid()
    js_block = g.blocks_of(XID_JS.read_text(encoding="utf-8"))
    js_table = "".join(re.findall(r'"([0-9A-Za-z]+)"', js_block.split("const _XID_TABLE =", 1)[1]))
    assert js_table == _xid.TABLE
    assert _xid.UNICODE_VERSION == g.PINNED == "15.0.0"
    assert f'const _XID_UNICODE_VERSION = "{_xid.UNICODE_VERSION}";' in js_block
    assert f'const _XID_TABLE_SHA256 = "{_xid.TABLE_SHA256}";' in js_block
    assert hashlib.sha256(_xid.TABLE.encode("ascii")).hexdigest() == _xid.TABLE_SHA256
    # the Python reads names by the table, not by the runtime (on Python 3.12 the two agree on every code
    # point, so this is asked of the reading itself: a table that refused `b` would end `abc` at `a`)
    assert dg._xid_opens is _xid.opens_identifier and dg._xid_continues is _xid.continues_identifier


def test_x1_the_identifier_is_read_through_the_table_functions(monkeypatch):
    monkeypatch.setattr(dg, "_xid_continues", lambda ch: ch != "c")
    assert dg._identifier_at("abc", 0) == "ab"
    m = dg._TEMPLATES[7][1].search("Added function abc.")
    monkeypatch.setattr(dg, "_xid_word", lambda ch: ch == "c")
    assert dg._claimed_name("Added function abc.", m) == (
        "ab", "the claimed name runs past 'ab' into 'c' (U+0063), which no Python identifier holds; no definition is read for it")
    monkeypatch.setattr(dg, "_xid_opens", lambda ch: ch != "a")
    assert dg._identifier_at("abc", 0) == ""


@pytest.mark.skipif(unicodedata.unidata_version != "15.0.0",
                    reason="gen_xid.py runs only under the table's Unicode version, 15.0.0 (Python 3.12)")
def test_x1_the_generator_reproduces_both_blocks_from_the_pinned_version():
    g = _gen_xid()
    t = g.table()
    for path, block in ((XID_PY, g.python_block(t)), (XID_JS, g.js_block(t))):
        before, after = g.splice(path, block)
        assert before == after, path
    # the middle dots the claim reading refuses to end a name on are the punctuation XID_Continue holds
    po = [c for c in range(0x110000) if ("a" + chr(c)).isidentifier() and unicodedata.category(chr(c)) == "Po"]
    assert po == [0xB7, 0x387] and dg._PROSE_DOTS == "\u00b7\u0387"


def test_x1_the_port_decodes_the_table_as_the_python_does():
    from styxx import _xid
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const src=require('fs').readFileSync(process.argv[1],'utf8');"
              "const f=new Function(src+';return [_XID_STARTS,_XID_MASKS];');process.stdout.write(JSON.stringify(f()));")
    r = subprocess.run([node, "-e", script, str(XID_JS)], capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    starts, masks = json.loads(r.stdout)
    assert (starts, masks) == (_xid._STARTS, _xid._MASKS)


X1_SKEW = ["\u200c", "\u200d", "\u30fb", "\uff65", "\u1c89", "\U0002ebf0"]


def test_x1_the_characters_the_runtimes_disagree_on_read_alike_in_both_ports():
    # Round-6 review, blocker 2: U+200C, U+200D, U+30FB, U+FF65 (XID_Continue since Unicode 15.1) and letters
    # 15.1 and 16.0 assigned split the ports: Python 3.12 read Unicode 15.0 and Node 24 reads 16.0. One table.
    items = []
    for x in X1_SKEW:
        items += [(f"Added function foo{x}bar. Added function foo. Added class Foo{x}.",
                   f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,3 @@\n x = 0\n+def foo{x}bar():\n+class Foo{x}:\n"),
                  ("Added 1 test. Added 0 tests.", f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,2 @@\n x = 0\n+def test_a{x}b():\n"),
                  ("Added 1 test. Added 0 tests.", f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1 @@\n-def test_a{x}b():\n+def test_a{x}b(y):\n"),
                  ("No breaking changes.", f"--- a/{SP}\n+++ b/{SP}\n@@ -1,2 +1 @@\n-def api{x}x():\n-    pass\n+x = 1\n")]
    assert _port_records(items) == _py_records(items)
    assert dg._identifier_at("foo\u30fbbar", 0) == "foo" and dg._identifier_at("a\u200db", 0) == "a"
    assert dg._identifier_at("x\U0002ebf0", 0) == "x" and dg._identifier_at("caf\u00e9", 0) == "caf\u00e9"


def test_x1_a_claimed_name_that_runs_past_the_identifier_names_none(tmp_path):
    # Round-6 review, blocker 1: W-2 truncated `fo<U+00B2>o` to `fo` and verified it on `def fo`, a new false
    # VERIFIED against main on both Python doors. It names no identifier: UNCHECKABLE, on every door.
    summary = "Added function fo\u00b2o. Added class Fo\u2082o. Added function fo."
    diff = _git_repo(tmp_path, {SP: "x = 1\n"}, {SP: "x = 1\ndef fo():\n    pass\n\n\nclass Fo:\n    pass\n"})
    want = [("symbol_added", "UNCHECKABLE", "the claimed name runs past 'fo' into '\u00b2' (U+00B2), which no Python "
                                             "identifier holds; no definition is read for it"),
            ("symbol_added", "UNCHECKABLE", "the claimed name runs past 'Fo' into '\u2082' (U+2082), which no Python "
                                             "identifier holds; no definition is read for it"),
            ("symbol_added", "VERIFIED", "added lines do define function 'fo'")]
    via_git = [(c.kind, c.verdict, c.why) for c in gate_diff(summary, tmp_path, "HEAD~1", "HEAD").claims]
    assert via_git == _claims(summary, diff)[1] == want
    assert _port_claims([(summary, diff)]) == [want]


def test_x1_every_word_character_no_identifier_holds_ends_no_claimed_name_in_either_port():
    from styxx import _xid
    word_not_ident = [c for c in range(0x80, 0x110000) if _xid.is_word(chr(c)) and not _xid.continues_identifier(chr(c))]
    assert len(word_not_ident) == 923            # Unicode 15.0.0: 905 No, 16 Lo and 2 Lm
    diff = f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,2 @@\n x = 0\n+def fo():\n"
    items = [(f"Added function fo{chr(c)}o.", diff) for c in word_not_ident]
    py = _py_records(items)
    assert all(r[0][1] == "UNCHECKABLE" and r[0][2].startswith("the claimed name runs past 'fo' into") for r in py)
    assert _port_records(items) == py


def test_x1_a_claimed_name_ending_in_a_middle_dot_names_none_for_certain():
    # U+00B7 and U+0387 (the Greek semicolon) continue an identifier and are also written after a word in prose
    diff = f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,2 @@\n x = 0\n+def foo():\n"
    for dot in ("\u00b7", "\u0387"):
        summary = f"Added function foo{dot} Then."
        want = [("symbol_added", "UNCHECKABLE", f"the claimed name 'foo{dot}' ends in '{dot}' (U+{ord(dot):04X}), "
                                                 "which prose also writes after a word; no definition is read for it")]
        assert _claims(summary, diff)[1] == want and _port_claims([(summary, diff)]) == [want]
    # inside a name the dot is the name's: `col<U+00B7>leccio` is one identifier
    assert _claims("Added function col\u00b7leccio.", diff)[1][0][1] == "CONTRADICTED"


# ─────────────────────────────── NOTE_path2_seventh_pass_2026_09_25: x2, a hunk's counts read only when exact

X2_OVER = ("--- a/src/x.py\n+++ b/src/x.py\n@@ -1,5 +1,6 @@\n-a = 1\n+a = 2\n"
           "--- lib/y.py\n+++ lib/y.py\n@@\n-c = 1\n+c = 2\n+def test_new():\n+    pass\n")
X2_SHAPES = {   # round-6 review, blocker 3: each read the second file's header as content; main read a header
    "bare-at-at": X2_OVER,
    "elided-at-at": X2_OVER.replace("\n@@\n", "\n@@ ... @@\n"),
    "blank-then-hunk": X2_OVER.replace("+++ lib/y.py\n@@\n", "+++ lib/y.py\n\n@@ -1 +1,3 @@\n"),
    "no-hunk-header": X2_OVER.replace("\n@@\n", "\n"),
    "timestamps": X2_OVER.replace("--- lib/y.py\n+++ lib/y.py\n", "--- lib/y.py\t2024-01-01 00:00:00\n+++ lib/y.py\t2024-01-02\n"),
    "numeric-hunk": X2_OVER.replace("\n@@\n", "\n@@ -1 +1,3 @@\n"),
    "new-file": X2_OVER.replace("--- lib/y.py", "--- /dev/null"),
}


@pytest.mark.parametrize("shape", sorted(X2_SHAPES))
def test_x2_an_over_declared_hunk_before_a_gnu_header_reads_as_main_reads_it(shape):
    diff = X2_SHAPES[shape]
    lines = diff.split("\n")
    assert not dg._hunk_is_exact(lines, 2)
    status = parse_unified_diff(diff)[0]
    assert len(status) == 2 and "src/x.py" in status, status
    summary = "2 files changed. Added 1 test."
    assert _claims(summary, diff)[1] == [("files_changed_count", "VERIFIED", "diff changes 2 files, claim says 2"),
                                        ("tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1")]
    assert _port_claims([(summary, diff)]) == [_claims(summary, diff)[1]]


def test_x2_an_exact_git_hunk_holding_a_slash_pair_is_content_on_every_door(tmp_path):
    # Round-6 review, minor: `-- a/x` beside `++ b/x` in a real hunk read as a file header on the raw door and
    # the port (the header-pair exception's a/ b/ clause); git's hunks are exact, so they are content now
    before = {"src/a.py": 'S = """\n-- a/x\n"""\ndef foo():\n    pass\n'}
    after = {"src/a.py": 'S = """\n++ b/x\n"""\ndef foo(y):\n    pass\n'}
    diff = _git_repo(tmp_path, before, after)
    assert "\n--- a/x\n+++ b/x\n" in diff
    summary = "Only touches src/. 1 file changed."
    want = [("only_touches", "VERIFIED", "all changed paths under prefix"),
            ("files_changed_count", "VERIFIED", "diff changes 1 files, claim says 1")]
    via_git = [(c.kind, c.verdict, c.why) for c in gate_diff(summary, tmp_path, "HEAD~1", "HEAD").claims]
    assert via_git == _claims(summary, diff)[1] == want and _port_claims([(summary, diff)]) == [want]


def test_x2_what_may_follow_an_exact_hunk():
    head = "--- a/s.sql\n+++ b/s.sql\n@@ -1,3 +1,3 @@\n x\n--- a\n+++ b\n y\n"
    exact = {"end of diff": "", "diff --git": "diff --git a/t b/t\n", "a file header": "--- a/t\n+++ b/t\n@@ -1 +1 @@\n-p\n+q\n",
             "a hunk further on": "@@ -9 +9 @@\n-p\n+q\n", "a signature": "-- \n2.39.2\n", "prose": "Index: t\n",
             "a no-newline marker": "\\ No newline at end of file\n"}
    not_exact = {"a hunk going back": "@@ -1 +1 @@\n-p\n+q\n", "a body line": "+more\n", "a context line": " more\n",
                 "a blank line then a body line": "\n+more\n", "a blank line then a hunk": "\n@@ -9 +9 @@\n-p\n+q\n",
                 "a hunk header with no counts": "@@ ... @@\n-p\n+q\n"}
    for label, tail in exact.items():
        assert dg._hunk_is_exact((head + tail).split("\n"), 2), label
    for label, tail in not_exact.items():
        assert not dg._hunk_is_exact((head + tail).split("\n"), 2), label
    # a `--- ` line right after an added one is written by no generator: it is a header, and the hunk that
    # swallowed it declared too many lines (path2:w1-a-short-hunk-before-an-a-b-header-pair-with-no-hunk-header)
    assert not dg._hunk_is_exact("@@ -1,3 +1,4 @@\n x\n+y\n--- a/b\n+++ b/b\n-p\n+q\n".split("\n"), 0)
    # other lines a hand-written hunk interleaves stay countable
    assert dg._hunk_is_exact("@@ -1,3 +1,3 @@\n x\n--- users\n+++ x\n-def foo():\n+def foo(x):\n".split("\n"), 0)
    # the pair right before a hunk header is a header (the pinned limit), as on main
    assert not dg._hunk_is_exact("@@ -1,2 +1,2 @@\n x\n--- old\n+++ new\n@@ -9 +9 @@\n-a\n+b\n".split("\n"), 0)


def test_x2_the_walk_edges_on_both_ports():
    # a context line ends a change: a removed `-- users` after it is content again, and the changed def pairs
    exact = "--- a/src/m.py\n+++ b/src/m.py\n@@ -1,4 +1,4 @@\n x\n+y\n z\n--- users\n-def foo():\n+def foo(x):\n"
    # the same lines under counts the diff ends before closing: read as main reads them, `--- users` a header
    over = "--- a/src/m.py\n+++ b/src/m.py\n@@ -1,9 +1,9 @@\n Q = 1\n--- users\n-def foo():\n+def foo(x):\n"
    # an identifier character outside the Basic Multilingual Plane (CJK Extension B), in a claim and a definition
    astral = f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,2 @@\n x = 0\n+def foo\U00020000bar():\n"
    # the shelf's fold: a file created by the diff removes nothing, so both async definitions are added
    fold = "--- /dev/null\n+++ b/tests/test_n.py\n@@ -1 +1,2 @@\n-async def test_n():\n+async def test_n():\n+async def test_n():\n"
    items = [("Added function foo. 1 file changed.", exact), ("Added function foo. 1 file changed.", over),
             ("Added function foo\U00020000bar. Added function foo.", astral), ("Added 2 tests.", fold)]
    py = [_claims(s, d)[1] for s, d in items]
    assert py[0][0][1] == "UNCHECKABLE" and py[1][0][1] == "VERIFIED"
    assert [c[1] for c in py[2]] == ["VERIFIED", "CONTRADICTED"]
    assert py[3] == [("tests_added", "UNCHECKABLE", "diff adds 2 async test functions, which this template does not "
                                                     "count; claim says 2")]
    assert _port_claims(items) == py


# ─────────────────────────────── NOTE_path2_seventh_pass_2026_09_25: A-1, an added async test abstains the count

def test_a1_an_added_async_test_abstains_the_count_it_does_not_read():
    # The seventh pass's randomised repositories: `got` has never read `async def test_` (main neither), and
    # main's "Added 0 tests." over a new async test beside a changed sync one read CONTRADICTED only because
    # it also counted the changed test; the pairing (#101) took that second miscount away. Now it abstains.
    async_new = (f"--- a/{TP}\n+++ b/{TP}\n@@ -1,2 +1,4 @@\n-def test_load():\n+def test_load(z=0):\n     pass\n"
                 "+async def test_parse():\n+    pass\n")
    for summary in ("Added 0 tests.", "Added 1 test.", "Added 2 tests."):
        want = [("tests_added", "UNCHECKABLE", "diff adds 1 async test functions, which this template does not "
                                                f"count; claim says {summary.split()[1]}")]
        assert _claims(summary, async_new)[1] == want and _port_claims([(summary, async_new)]) == [want]
    # a changed async test is changed, not added (R-2 reads `async` on the removed side): no abstention
    changed = f"--- a/{TP}\n+++ b/{TP}\n@@ -1,2 +1,2 @@\n-async def test_a():\n+async def test_a(x):\n     pass\n"
    assert _claims("Added 0 tests.", changed)[1] == [("tests_added", "VERIFIED", "diff adds 0 test functions, claim says 0")]
    # a new file's async test abstains too; a sync test beside nothing async reads as before
    new_file = "--- /dev/null\n+++ b/tests/test_n.py\n@@ -0,0 +1,2 @@\n+async def test_n():\n+    pass\n"
    assert _claims("Added 1 test.", new_file)[1][0][1] == "UNCHECKABLE"
    assert _claims("Added 1 test.", _v1_raw("t-lead-new", " ")[1])[1][0][1] in ("VERIFIED", "CONTRADICTED")
    assert dg._async_tests_added({TP: (["async def test_x():", "def test_y():"], ["async def test_x(a):"])}) == 0
    assert dg._async_tests_added({TP: (["async def test_x():", "async def test_x(b):"], ["async def test_x(a):"])}) == 1
    assert dg._async_tests_added({TP: (["async def test_x():"], ["async def test_x():"])}, {TP: "A"}) == 1


# ─────────────────────────────── NOTE_path2_seventh_pass_2026_09_25: x7, the scorer's new checks

X7_PLANTED = {
    # label: (good text in styxx/diffgate.py, planted text, summary, diff, the violation that must fire)
    "D2 a U+FEFF dropped from any added line (W-1's own code)": (
        "                if new_no == 1 and text.startswith(_FILE_BOM):", "                if text.startswith(_FILE_BOM):",
        "Added 1 test.",
        "--- a/docs/guide.md\n+++ b/docs/guide.md\n@@ -1,2 +1 @@\n x\n----\n"
        "--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1 +1,2 @@\n x = 0\n+\ufeffdef test_b():\n",
        "G-C7_oracle:W-1_added_lines"),
    "D4 `_defined_name` refuses a colon after the name (W-2's own code)": (
        "    if not name or (end < len(line) and ord(line[end]) > 0x7F):",
        "    if not name or (end < len(line) and (ord(line[end]) > 0x7F or line[end] == ':')):",
        "Added class Foo.", f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,2 @@\n x = 0\n+class Foo:\n",
        "G-C7_oracle:W-2_defined_name"),
    "D5 a line holding U+2028 dropped by the split (F-2's own code)": (
        "lines = _DIFF_LINE_BREAK.split(text)",
        "lines = [x for x in _DIFF_LINE_BREAK.split(text) if '\\u2028' not in x]",
        "Added 2 tests.", f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,3 @@\n x = 0\n+def test_a():\n+def test_b():  #  \n",
        "G-C7_oracle:F-2_split"),
    "D7 the gate ignores an only_touches accusation": (
        '    contradicted = any(c.verdict == "CONTRADICTED" for c in claims)',
        '    contradicted = any(c.verdict == "CONTRADICTED" for c in claims if c.kind != "only_touches")',
        "Only touches src/.", "--- a/docs/x.md\n+++ b/docs/x.md\n@@ -1 +1 @@\n-a\n+b\n",
        "G-C1_gate_verdict_not_from_its_claims"),
    "the runs-past rule removed (W-2's own code)": (
        "    if end < len(sentence) and _xid_word(sentence[end]):", "    if False:",
        "Added function fo\u00b2o.", f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,2 @@\n x = 0\n+def fo():\n",
        "G-C7_oracle:symbol_added_claim"),
    "the key strips one leading segment only (#121's own code)": (
        '_LEADING_SLASH_SEGMENTS = re.compile(r"^(?:\\.?/)+")', '_LEADING_SLASH_SEGMENTS = re.compile(r"^(?:\\.?/)")',
        "1 file changed.", "--- a//./x.py\n+++ b//./x.py\n@@ -1 +1 @@\n-a\n+b\n",
        "G-C7_oracle:#121_key"),
    "the suffix tier before the exact one (#97's own code)": (
        "    for tier in (lambda p: p == c,",
        "    for tier in (lambda p: p.endswith(\"/\" + c), lambda p: p == c,",
        "Created a/b.py.", "--- a/x/a/b.py\n+++ b/x/a/b.py\n@@ -1 +1 @@\n-p\n+q\n--- /dev/null\n+++ b/a/b.py\n@@ -0,0 +1 @@\n+z\n",
        "G-C7_oracle:#97_tiers"),
    "got strips its line before reading it (F-3's own code)": (
        '    return sum(1 for line in added_blob.split("\\n") if _test_name(line))',
        '    return sum(1 for line in added_blob.split("\\n") if _test_name(line.lstrip()))',
        "Added 1 test.", f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,2 @@\n x = 0\n+\u00a0def test_a():\n",
        "G-C7_oracle:F-3_got"),
    "the async guard removed (A-1's own code)": (
        "                        if unread:", "                        if False:",
        "Added 0 tests.", (f"--- a/{TP}\n+++ b/{TP}\n@@ -1,2 +1,4 @@\n-def test_load():\n+def test_load(z=0):\n     pass\n"
                           "+async def test_parse():\n+    pass\n"),
        "G-C7_oracle:tests_added_claim"),
    "A-1 accusing instead of abstaining (A-1 admits only an abstention)": (
        'c.why = (f"diff adds {unread} async test functions, which this template does not "',
        'c.verdict = "CONTRADICTED"; c.why = (f"diff adds {unread} async test functions, which this template does not "',
        "Added 0 tests.", "--- /dev/null\n+++ b/tests/test_n.py\n@@ -0,0 +1,2 @@\n+async def test_n():\n+    pass\n",
        "G-C4_direction:tests_added:A-1"),
    "gate fields moved by F-2's own code on a diff where F-2 cannot act": (
        "lines = _DIFF_LINE_BREAK.split(text)", "lines = [] if 'zz.py' in text else _DIFF_LINE_BREAK.split(text)",
        "1 file changed.", "--- a/src/zz.py\n+++ b/src/zz.py\n@@ -1 +1 @@\n-a\n+b\n",
        "G-C1_gate_fields_differ"),
    "a gate field moved by no rule": (
        "                    sentences_total=total, uncovered_texts=uncovered_texts,",
        "                    sentences_total=total + 1, uncovered_texts=uncovered_texts,",
        "1 file changed.", "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n",
        "G-C1_gate_fields_differ"),
}


@pytest.mark.parametrize("label", sorted(X7_PLANTED))
def test_x7_every_new_check_fails_on_a_planted_defect(scorer, monkeypatch, label):
    # Round-6 protocol lens, major and minor: a defect inside a post-amendment rule's own code passed the
    # gates (its revert took it back), and the gate-level fields went unread. Each is refused now.
    pg = scorer
    good, bad, summary, diff, violation = X7_PLANTED[label]
    t = pg.Tally(name_prs=True)
    t.pair("clean", summary, diff, pg.raw_paths(diff))
    assert not t.violations, t.violating
    _planted(pg, monkeypatch, good, bad, "x7")
    t = pg.Tally(name_prs=True)
    t.pair("planted", summary, diff, pg.raw_paths(diff))
    assert violation in t.violations, dict(t.violations)


def test_x7_the_git_door_is_scored_and_a_defect_in_it_fails(scorer, monkeypatch):
    from collections import Counter
    pg = scorer
    summary = "Added 1 test. Added 0 tests."
    diff = f"--- a/{TP}\n+++ b/{TP}\n@@ -1,2 +1,2 @@\n-def test_a():\n+def test_a(x):\n     pass\n"
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "clean", summary, diff, sample)
    assert sample == {"scored": 1} and not t.violations and t.n["records_moved"] == 1
    # planted: the git door hands the gate no sides, so its pairing is blind (the raw door's is not)
    _planted(pg, monkeypatch, "                 sides=parse_unified_diff_sides(diff_text))",
             "                 sides=None)", "x7git")
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "planted", summary, diff, sample)
    assert t.violations["G-C8_git_door_differs_from_the_raw_door"] == 1
    # a diff that cannot be rebuilt faithfully is counted, not scored
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "sloppy", "2 files changed.", X2_OVER, sample)
    assert sample == {"not_rebuildable": 1} and not t.n


@pytest.mark.skipif(unicodedata.unidata_version != "15.0.0", reason="the scorer checks the table against a 15.0.0 database only")
def test_x7_the_scorer_refuses_a_name_table_its_database_does_not_read(monkeypatch):
    # a table whose hash was made to match after one run was changed: only the database can tell
    import sys
    import types
    from styxx import _xid
    g = _gen_xid()
    runs = [(nxt - s, m) for s, nxt, m in zip(_xid._STARTS, _xid._STARTS[1:] + [0x110000], _xid._MASKS)]
    runs[0] = (runs[0][0], 4)                       # U+0000.. read as word characters
    table = "".join(g.encode(n * 8 + k) for n, k in runs)
    fake = types.ModuleType("styxx._xid")
    fake.TABLE, fake.UNICODE_VERSION = table, "15.0.0"
    fake.TABLE_SHA256 = hashlib.sha256(table.encode("ascii")).hexdigest()
    monkeypatch.setitem(sys.modules, "styxx._xid", fake)
    monkeypatch.setitem(sys.modules, "styxx.claimdetect", sys.modules.get("styxx.claimdetect"))
    spec = importlib.util.spec_from_file_location("path2_gates_corrupt_table", FRONTIER / "path2_gates.py")
    with pytest.raises(SystemExit, match="name table reads U\\+0000"):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


def test_x7_the_scorer_reads_names_by_the_same_table_with_its_own_decoder(scorer):
    from styxx import _xid
    pg = scorer
    assert (pg.XID_STARTS, pg.XID_MASKS) == (_xid._STARTS, _xid._MASKS)
    assert pg.XID_CHECKED_AGAINST_DATABASE == (unicodedata.unidata_version == "15.0.0")
    for text in ("fo\u00b2o", "foo\u30fbbar", "col\u00b7leccio", "x\u0301y", "_a1", "1abc"):
        assert pg.ident_at(text, 0) == dg._identifier_at(text, 0), ascii(text)


# ────────────────────────────────────────────────────────── the pinned pairs

def test_the_pinned_pairs_read_as_expected_on_the_python_side():
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    assert len(pairs) == 151 and all(p["id"].startswith("path2:") for p in pairs)
    # NOTE_path2_fifth_pass V-1 re-pinned four pairs and NOTE_path2_sixth_pass W-1 one; each record says so
    assert sorted(p["id"] for p in pairs if "repinned" in p) == [
        "path2:f2-a-form-feed-is-not-a-line-break", "path2:f2-limit-hit-reads-a-line-separator-as-indent",
        "path2:r4-a-changed-generic-definition-still-verifies", "path2:r4-a-function-made-generic-still-verifies",
        "path2:v1-limit-a-bom-led-def-reads-as-a-definition-wherever-it-stands"]
    for p in pairs:
        g = gate_diff_text(p["summary"], p["diff"], run=None, strict=False)
        got = [[c.kind, c.verdict, c.why] for c in g.claims]
        assert got == p["expect"]["claims"], (p["id"], got)
        assert g.verdict == p["expect"]["verdict"], p["id"]
        assert g.uncovered_sentences == p["expect"]["uncovered_sentences"], p["id"]
    # NOTE_path2_third_pass: no pair needs a `python_only` escape any more. The `.storybook` pair
    # needed one while the port lacked COMPAT-2; the port carries it since #126, so every pair is
    # pinned at full width, holding the port to the reason as well as to the verdict.
    assert not any("python_only" in p["expect"] for p in pairs)


def test_the_demo_is_unchanged():
    g = gate_diff_text(dg._DEMO_SUMMARY, dg._DEMO_DIFF, run=None, strict=False)
    assert [(c.kind, c.verdict) for c in g.claims] == [
        ("file_touched", "VERIFIED"), ("symbol_added", "CONTRADICTED"), ("tests_added", "CONTRADICTED"),
        ("only_touches", "CONTRADICTED"), ("tests_pass", "UNCHECKABLE")]

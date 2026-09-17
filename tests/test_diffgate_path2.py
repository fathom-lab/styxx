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
rename and a test moved between files count as added)."""
import json
import subprocess
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
    assert dg._changed_test_defs(sides) == 2
    assert dg._changed_test_defs(None) == 0
    pat = r"^\s*(?:def|class)\s+backoff\b"
    assert dg._definition_only_changed(pat, "def backoff(n, jitter=0):", sides)
    assert not dg._definition_only_changed(pat, "def backoff(n, jitter=0):\ndef backoff():", sides)
    assert not dg._definition_only_changed(pat, "y = 1", sides)


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


# ────────────────────────────────────────────────────────── the pinned pairs

def test_the_pinned_pairs_read_as_expected_on_the_python_side():
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    assert len(pairs) == 17 and all(p["id"].startswith("path2:") for p in pairs)
    for p in pairs:
        g = gate_diff_text(p["summary"], p["diff"], run=None, strict=False)
        got = [[c.kind, c.verdict, c.why] for c in g.claims]
        assert got == p["expect"]["claims"], (p["id"], got)
        assert g.verdict == p["expect"]["verdict"], p["id"]
        assert g.uncovered_sentences == p["expect"]["uncovered_sentences"], p["id"]


def test_the_demo_is_unchanged():
    g = gate_diff_text(dg._DEMO_SUMMARY, dg._DEMO_DIFF, run=None, strict=False)
    assert [(c.kind, c.verdict) for c in g.claims] == [
        ("file_touched", "VERIFIED"), ("symbol_added", "CONTRADICTED"), ("tests_added", "CONTRADICTED"),
        ("only_touches", "CONTRADICTED"), ("tests_pass", "UNCHECKABLE")]

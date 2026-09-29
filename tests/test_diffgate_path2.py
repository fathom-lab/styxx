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


# NOTE_path2_ninth_pass_2026_09_27: main's reading, written out a third time for the tests (the instrument and the
# scorer each carry their own). The licensed-difference rule turns an earlier pass's verdict into an abstention where
# main's reading differs from this one's: `_ninth` does that to an earlier pass's expected claims, so each earlier
# test still pins the reading it was written for, and now also the abstention that reading earns.
Z1_TAIL = "(`^\\s*def test_` over main's line split); no repair licenses the difference"
Z2_TAIL = "(`^\\s*(?:def|class)\\s+NAME\\b` over main's line split); no repair licenses the difference"
Z5_WHY = ("is one this reading refuses and CPython may refuse too, and a file CPython refuses defines nothing; this "
          "reading reads it line by line")
Z4_TAIL = "#97 licenses the exact and suffix tiers only"
_JS_SPACE = re.escape("".join(map(chr, (9, 10, 11, 12, 13, 32, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029,
                                        0x202F, 0x205F, 0x3000, 0xFEFF))))
_LS_PS = chr(0x2028) + chr(0x2029)
_ANY_SPACE = "[\\s" + chr(0xFEFF) + "]"


def _main_added(diff, js):
    lines = re.split("\r\n|\r|\n", diff) if js else diff.splitlines()
    return [x[1:] for x in lines if x.startswith("+") and not x.startswith("+++")]


def _main_tests(diff):
    """main's `got`: its Python's (str.splitlines(), `re`'s \\s, `^` after \\n) and its port's (\\r\\n, \\r and \\n;
    JavaScript's \\s; `^` after \\n, U+2028 and U+2029)."""
    py = len(re.findall(r"^\s*def test_", "\n".join(_main_added(diff, False)), re.M))
    js = len(re.findall("(?:^|(?<=[" + _LS_PS + "]))[" + _JS_SPACE + "]*def test_", "\n".join(_main_added(diff, True)),
                        re.M))
    return py, js


def _main_hits(diff, name):
    """main's `hit` in each spelling (the port's name is the ASCII run the port's template captures)."""
    py = re.search(r"^\s*(?:def|class)\s+" + re.escape(name) + r"\b", "\n".join(_main_added(diff, False)), re.M)
    js_name = re.match(r"[A-Za-z_][A-Za-z0-9_]*", name).group(0)
    js = re.search("(?:^|(?<=[" + _LS_PS + "]))[" + _JS_SPACE + "]*(?:def|class)[" + _JS_SPACE + "]+"
                   + re.escape(js_name) + "(?![A-Za-z0-9_])", "\n".join(_main_added(diff, True)), re.M)
    return py is not None, js is not None


def _refused(line):
    m = re.match("^" + _ANY_SPACE + "*(?:async" + _ANY_SPACE + "+)?(?:def|class)" + _ANY_SPACE + "+", line)
    return (m is not None and m.end() < len(line) and (line[m.end()].isidentifier() or dg._xid_skew(line[m.end()]))
            and dg._defined_name(line, True) is None)


_EARLIER = ("async test functions, which this template does not count", "U+FEFF", "Unicode 13.0 to 16.0",
            "no Python file in the diff", "the claimed name")

# NOTE_path2_tenth_pass_2026_09_27 (K-1): W-1's exact hunk licenses no file-list difference. Where main read a file (or a
# status) from lines an exact hunk's counts hold, the file-list claims abstain at the raw door and in the port.
K1_WHY = ("the diff's file list is not certain: this reading's file list differs from main's: main read the file list "
          "from lines an exact hunk's counts hold, which W-1 reads as content ({}); a changed file neither reading "
          "counts may have balanced it, so W-1 licenses no file-list difference")
FILE_LIST = ("files_changed_count", "only_touches", "file_created", "file_deleted", "file_touched")


def _k1(earlier, moved):
    """An earlier pass's expected claims with K-1's abstention on every file-list claim."""
    out = []
    for kind, verdict, why in earlier:
        if kind not in FILE_LIST:
            out.append((kind, verdict, why))
            continue
        n = re.search(r"claim says (\d+)", why)
        tail = "; claim says " + n.group(1) if kind == "files_changed_count" else ""
        out.append((kind, "UNCHECKABLE", K1_WHY.format(moved) + tail))
    return out


def _ninth(diff, earlier, name="foo"):
    """An earlier pass's expected (kind, verdict, why) claims over a raw diff, as the ninth pass reads them: Z-1 and
    Z-2 (this reading's `got` or `hit` against main's) and then Z-5 (a refused definition line), after the abstentions
    that precede them (BC-1, A-1, Y-2, Y-3, the claimed name's own rules)."""
    blob = parse_unified_diff(diff)[1]
    sides = parse_unified_diff_sides(diff)
    got, (py, js) = dg._added_tests(blob), _main_tests(diff)
    refused = [p for p, (a, _r) in sides.items() if p.endswith(".py") and any(_refused(x) for x in a)]
    out = []
    for kind, verdict, why in earlier:
        if verdict == "UNCHECKABLE" and any(x in why for x in _EARLIER):
            out.append((kind, verdict, why))
        elif kind == "tests_added":
            n = re.search(r"claim says (\d+)", why).group(1)
            if not got == py == js:
                out.append((kind, "UNCHECKABLE", f"this reading counts {got} added test definitions where main's Python "
                                                 f"counted {py} and its port {js} {Z1_TAIL}; claim says {n}"))
            elif refused:
                out.append((kind, "UNCHECKABLE", f"an added definition line in {refused[0]!r} {Z5_WHY}; claim says {n}"))
            else:
                out.append((kind, verdict, why))
        elif kind == "symbol_added":
            hit = dg._symbol_hit(name, blob)
            mp, mj = _main_hits(diff, name)
            holding = [p for p in refused if any(dg._defines(x, name) for x in sides[p][0])]
            if not hit == mp == mj:
                out.append((kind, "UNCHECKABLE", f"this reading finds {'an' if hit else 'no'} added definition of "
                                                 f"'{name}' where main's Python {'did' if mp else 'did not'} and its "
                                                 f"port {'did' if mj else 'did not'} {Z2_TAIL}"))
            elif hit and holding:
                out.append((kind, "UNCHECKABLE", f"an added definition line in {holding[0]!r} {Z5_WHY}"))
            else:
                out.append((kind, verdict, why))
        else:
            out.append((kind, verdict, why))
    return out


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
    # the disclosed limit: a claim that names a directory still meets a same-named file elsewhere -- and since
    # NOTE_path2_ninth_pass (Z-4) it no longer verifies there: only the exact and suffix tiers are #97's licence
    diff = "--- a/lib/util/helpers.py\n+++ b/lib/util/helpers.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
    _, got = _claims("Modified src/helpers.py.", diff)
    assert dg._find_path(parse_unified_diff(diff)[0], "src/helpers.py") == ("lib/util/helpers.py", "M")
    assert got == [("file_touched", "UNCHECKABLE", "'src/helpers.py': only a file with the same name in another "
                                                   f"directory is in the diff ('lib/util/helpers.py', status 'M'); {Z4_TAIL}")]
    # a bare name is resolved by the suffix tier and still verifies
    assert _claims("Modified helpers.py.", diff)[1] == [("file_touched", "VERIFIED",
                                                         "diff status 'M' for 'lib/util/helpers.py'")]


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
    # NOTE_path2_twelfth_pass (A.1): #121 licenses the difference from main in git's own rendering, and in a plain `---`/`+++`
    # rendering (where a renderer may have left a file out, or an `a/` directory reads as git's prefix) the claim abstains
    diff = _g(".eslintrc.json") + _g("config/eslintrc.json", "A")
    _, got = _claims("Created eslintrc.json. Updated .eslintrc.json.", diff)
    assert got == [("file_created", "VERIFIED", "diff status 'A' for 'config/eslintrc.json'"),
                   ("file_touched", "VERIFIED", "diff status 'M' for '.eslintrc.json'")]
    plain = ("--- a/.eslintrc.json\n+++ b/.eslintrc.json\n@@ -1 +1 @@\n-{}\n+{\"a\": 1}\n"
             "--- /dev/null\n+++ b/config/eslintrc.json\n@@ -0,0 +1 @@\n+{}\n")
    _, got = _claims("Created eslintrc.json. Updated .eslintrc.json.", plain)
    assert got == [("file_created", "UNCHECKABLE", _unlicensed("UNCHECKABLE", "VERIFIED")),
                   ("file_touched", "VERIFIED", "diff status 'M' for '.eslintrc.json'")]


# ───────────────────────────────────────────────────────────────────────── #101

# NOTE_path2_eighth_pass (Y-5): the #101 pairing withdraws, it does not verify. Where a changed test is paired
# away, a claimed count equal to what is left (`net`) reads UNCHECKABLE: `got` itself may read a line Python does
# not define (a string opened where the diff does not show it, a file that is not Python), and main's
# contradiction was then right. A count outside [net, got] is one main contradicts too.
Y5 = ("a count left after pairing changed tests away is not verified, since a line this template reads may be one "
      "Python does not define (#101)")

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
        ("tests_added", "UNCHECKABLE", f"diff adds 1 test functions and changes 1, claim says 1; {Y5}"),
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
        ("tests_added", "UNCHECKABLE", f"diff adds 2 test functions and changes 1, claim says 2; {Y5}"),
    ]


def test_101_c1_a_changed_test_beside_a_same_named_new_one_is_not_zero_added():
    diff = ("--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1,2 +1,4 @@\n-class TestFoo:\n-    def test_basic(self):\n"
            "+class TestFoo:\n+    def test_basic(self, client):\n+class TestBar:\n+    def test_basic(self):\n")
    _, got = _claims("Added 0 tests. Added 1 test.", diff)
    assert got == [
        ("tests_added", "CONTRADICTED", "diff adds 1 test functions, claim says 0 (1 changed, not added: #101)"),
        ("tests_added", "UNCHECKABLE", f"diff adds 1 test functions and changes 1, claim says 1; {Y5}"),
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
        ("tests_added", "UNCHECKABLE", f"diff adds 1 test functions and changes 1, claim says 1; {Y5}"),
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
    # NOTE_path2_eighth_pass (Y-2): an added test definition a U+FEFF opens -- line 1 included, where CPython
    # reads it -- abstains the count: main's Python counted it as no test and its port as one, so a count
    # elsewhere may have balanced either
    assert got == [("tests_added", "UNCHECKABLE", f"{Y2_TEST}; claim says 0")]
    # NOTE_path2_sixth_pass W-1: the U+FEFF opens line 1 of the file, so the one parse drops it before
    # `got` or the pairing reads the line; the definition reading itself takes none.
    assert dg.parse_unified_diff(diff)[1] == "def test_a():"
    assert dg._added_tests("def test_a():") == 1 and dg._added_tests("\ufeffdef test_a():") == 0


def test_r1_a_bom_on_a_changed_test_does_not_hide_a_new_one():
    # The non-degenerate case the round-2 reviewers filed: one test really is added.
    diff = ("--- a/tests/test_bom.py\n+++ b/tests/test_bom.py\n@@ -1,2 +1,4 @@\n"
            "-def test_a():\n+\ufeffdef test_a():\n     pass\n+def test_b():\n+    pass\n")
    _, got = _claims("Added 0 tests. Added 1 test.", diff)
    assert got == [("tests_added", "UNCHECKABLE", f"{Y2_TEST}; claim says 0"),       # NOTE_path2_eighth_pass (Y-2)
                   ("tests_added", "UNCHECKABLE", f"{Y2_TEST}; claim says 1")]
    two = ("--- a/tests/test_bom.py\n+++ b/tests/test_bom.py\n@@ -1,2 +1,6 @@\n"
           "-def test_a():\n+\ufeffdef test_a():\n     pass\n+def test_b():\n+    pass\n"
           "+def test_c():\n+    pass\n")
    _, got = _claims("Added 1 test. Added 2 tests.", two)
    assert got == [("tests_added", "UNCHECKABLE", f"{Y2_TEST}; claim says 1"),
                   ("tests_added", "UNCHECKABLE", f"{Y2_TEST}; claim says 2")]


def test_r2_an_async_test_made_sync_is_a_changed_test_not_an_added_one():
    # NOTE_path2_third_pass R-2. `async` is accepted on the REMOVED side only: `got` does not count
    # `async def test_`, so accepting it on the added side would part the two sets again (R-1).
    diff = ("--- a/tests/test_as.py\n+++ b/tests/test_as.py\n@@ -1,2 +1,2 @@\n"
            "-async def test_fetch():\n+def test_fetch():\n     assert fetch()\n")
    _, got = _claims("Added 0 tests. Added 1 test.", diff)
    assert got == [
        ("tests_added", "UNCHECKABLE", f"diff adds 0 test functions and changes 1, claim says 0; {Y5}"),
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
        ("tests_added", "UNCHECKABLE", f"diff adds 2 test functions and changes 1, claim says 2; {Y5}"),
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


def _g(path, st="M"):
    """NOTE_path2_twelfth_pass (A.1): one file as git writes it -- #121 licenses a difference only in git's own rendering."""
    head = f"diff --git a/{path} b/{path}\n"
    if st == "A":
        return head + f"new file mode 100644\nindex 0000000..1111111\n--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+b\n"
    return head + f"index 1111111..2222222 100644\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n"


def _unlicensed(main, this):
    """The guard's reason where a difference from main's verdict has no licence (NOTE_path2_twelfth_pass: #121 in a plain
    rendering)."""
    return dg._GUARD_DIFFERS.format(main=main, this=this)


def test_121_c3_an_accusation_lists_only_real_outside_paths():
    diff = _m(".github/a.yml") + _m(".github/b.yml") + _m(".github/c.yml") + _m("src/x.py")
    _, got = _claims("Only touches github/.", diff)
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'github': ['src/x.py']")]
    _, got = _claims("Only touches src and github.", diff[:-len(_m("src/x.py"))] + _m("docs/x.md") + _m("src/y.py"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'src' and 'github': ['docs/x.md']")]


def test_121_c3_a_dot_on_the_prefix_and_not_on_the_path_is_not_a_dot_miss():
    _, got = _claims("Only touches .env.", _g("env"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside '.env': ['env']")]
    _, got = _claims("Only touches .github/.", _g(".github/x.yml") + _g("github/z.md"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside '.github': ['github/z.md']")]
    # NOTE_path2_twelfth_pass (A.1): the same accusation in a plain rendering is #121's alone, and abstains
    _, got = _claims("Only touches .env.", _m("env"))
    assert got == [("only_touches", "UNCHECKABLE", _unlicensed("VERIFIED", "CONTRADICTED"))]


def test_121_c3_a_dotdot_path_is_not_a_dot_miss():
    _, got = _claims("Only touches src/.", _g("../src/x.py"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside 'src': ['../src/x.py']")]
    _, got = _claims("Only touches src/.", _m("../src/x.py"))                  # a plain rendering (A.1)
    assert got == [("only_touches", "UNCHECKABLE", _unlicensed("VERIFIED", "CONTRADICTED"))]
    assert not dg._dot_miss("..env", ["env"]) and not dg._dot_miss("../src/x.py", ["src"])
    assert dg._dot_miss(".github/x.yml", ["github"]) and dg._dot_miss(".env", ["env"])
    assert not dg._dot_miss(".github/x.yml", [".github"]) and not dg._dot_miss("github/x.yml", ["github"])
    # a dotted prefix over a path with one dot more: each clause alone excludes it, so this pins the pair
    assert not dg._dot_miss("..env", [".env"]) and not dg._dot_miss("..github/x.yml", [".github"])
    _, got = _claims("Only touches .env.", _g("..env"))
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
    # AMENDMENT C-3's accusation class, which the blocker had switched off for these spellings (NOTE_path2_twelfth_pass:
    # #121's alone, so in git's own rendering)
    _, got = _claims("Only touches .gitignore.", _g("gitignore"))
    assert got == [("only_touches", "CONTRADICTED", "paths outside '.gitignore': ['gitignore']")]
    _, got = _claims("Only touches .github.", _g("github/ci.yml"))
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
REINDENT_AS_CHANGED = [("tests_added", "UNCHECKABLE", f"diff adds 0 test functions and changes 1, claim says 0; {Y5}"),
                       ("tests_added", "UNCHECKABLE", "diff adds 0 test functions and changes 1, claim says 1; a "
                                                      "changed test is not an added one (#101)")]
REINDENT_AS_NOTHING = [("tests_added", "VERIFIED", "diff adds 0 test functions, claim says 0"),
                       ("tests_added", "CONTRADICTED", "diff adds 0 test functions, claim says 1")]


@pytest.mark.parametrize("ch", ["\x0b", "\x0c", "\u2028", "\u2029"])
def test_f2_the_four_classes_the_ports_split_on_differently_now_read_alike(ch):
    # Round 3, review 2: `-def test_a():` / `+<ch>def test_a():` is a re-indent; the port said
    # "Added 1 test." VERIFIED where the Python said CONTRADICTED. The pinned pairs hold the port.
    _, got = _claims("Added 0 tests. Added 1 test.", _reindent(ch))
    # NOTE_path2_ninth_pass (Z-1): the two readings still hold (`_ninth` keeps them) wherever main's two ports counted
    # what this reading counts; here main's Python split the line and its port did not, so the count abstains
    assert got == _ninth(_reindent(ch), REINDENT_AS_CHANGED if ch == "\x0c" else REINDENT_AS_NOTHING)
    assert all(v == "UNCHECKABLE" for _k, v, _w in got)


def test_f2_a_separator_inside_a_line_no_longer_forges_a_line():
    ctx = ("--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1,2 +1,2 @@\n x\u2028+def test_new():\n"
           "-y = 0\n+y = 1\n")
    _, got = _claims("Added 1 test.", ctx)
    assert got == _ninth(ctx, [("tests_added", "CONTRADICTED", "diff adds 0 test functions, claim says 1")])
    hdr = ("--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1,2 @@\n x\u2028+++ b/evil.py\n-y = 0\n+y = 1\n"
           "--- a/src/b.py\n+++ b/src/b.py\n@@ -1 +1 @@\n-a\n+b\n")
    assert parse_unified_diff(hdr)[0] == {"src/a.py": "M", "src/b.py": "M"}
    _, got = _claims("2 files changed. Only touches src/.", hdr)
    # NOTE_path2_ninth_pass (Z-3): main's Python read `+++ b/evil.py` as a header and its port did not, so the file
    # list abstains (this reading's list is its port's)
    apart = ("the diff's file list is not certain: main's Python and its port read the file list apart "
             "(str.splitlines() breaks lines JavaScript does not): main's Python reads 'evil.py' ('M'), which its "
             "port does not")
    assert got == [("files_changed_count", "UNCHECKABLE", f"{apart}; claim says 2"), ("only_touches", "UNCHECKABLE", apart)]


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
        assert got == _ninth(_reindent(ch), [("tests_added", "VERIFIED", "diff adds 0 test functions, claim says 0"),
                                             ("tests_added", "CONTRADICTED", "diff adds 0 test functions, claim says 1")]), repr(ch)


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
    assert got_git == _ninth(diff, REINDENT_AS_CHANGED if ch == "\x0c" else REINDENT_AS_NOTHING)


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
    assert got_git == _ninth(diff, [("tests_added", "CONTRADICTED", "diff adds 0 test functions, claim says 1")])


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
    # NOTE_path2_ninth_pass (Z-3): both doors abstain -- main's git door cut the name-status line where this one does
    # not, and main's raw door read the text apart from its port -- each for its own reason
    assert got_git == [("only_touches", "UNCHECKABLE", "the diff's file list is not certain: this reading's file list "
                                                       "differs from main's where no repair accounts for it: main reads "
                                                       "'a' ('M'), which this reading does not")]
    assert [(c.kind, c.verdict) for c in via_text.claims] == [("only_touches", "UNCHECKABLE")]
    assert "main's Python and its port read the file list apart" in via_text.claims[0].why
    assert dg._read_diff(diff)[0] == {name: "M"}                        # the reading itself is git's split


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


Y2 = ("opens with U+FEFF where the diff does not show it is line 1 of its file, the one place CPython reads "
      "one;")
Y2_TEST = "an added test definition opens with U+FEFF, which main's Python counted as no test and its port as one"
BOM_STRAY = {"t-lead-new": [("tests_added", "UNCHECKABLE", f"a test definition {Y2} claim says 0"),
                            ("tests_added", "UNCHECKABLE", f"a test definition {Y2} claim says 1")],
             "s-lead-new": [("symbol_added", "UNCHECKABLE", f"a definition of 'foo' {Y2[:-1]}")],
             # line 1: CPython reads the U+FEFF and the parse drops it, but the count abstains (Y-2)
             "t-lead-reindent": [("tests_added", "UNCHECKABLE", f"{Y2_TEST}; claim says 0"),
                                 ("tests_added", "UNCHECKABLE", f"{Y2_TEST}; claim says 1")]}


def _v1_expected(shape: str, ch: str) -> list:
    """What each shape reads as: a character is indentation (and a keyword separator) exactly when
    CPython's tokenizer takes it as one -- U+FEFF only where it opens line 1 of a file (NOTE_path2_sixth_pass
    W-1: the `-new` shapes put the definition on line 2) -- and a name ends at an ASCII character that
    cannot continue it. NOTE_path2_eighth_pass (Y-2): a definition a U+FEFF opens where the diff does not show
    line 1 makes the claim abstain (main's port read it, main's Python did not), and so does, for the test count,
    an added test definition a U+FEFF opens at line 1."""
    if ch == "\ufeff" and shape in BOM_STRAY:
        return BOM_STRAY[shape]
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
    # NOTE_path2_ninth_pass: the fifth-to-eighth-pass reading of each cell, as the licensed-difference rule reads it
    # (Z-1 and Z-2 where main's two ports read the line otherwise than this reading, Z-5 where the line is refused)
    for shape in V1_SHAPES:
        summary, diff = _v1_raw(shape, ch)
        _, got = _claims(summary, diff)
        assert got == _ninth(diff, _v1_expected(shape, ch)), (shape, ascii(ch))


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
    # NOTE_path2_sixth_pass W-1: a U+FEFF opens a definition only on line 1 of a file, where CPython reads it;
    # NOTE_path2_eighth_pass Y-2: one the diff does not show is line 1 makes the claim abstain
    "bom-led-new-function": ("Adds function foo.", {SP: "x = 0\n"}, {SP: "x = 0\n\ufeffdef foo():\n"},
                             [("symbol_added", "UNCHECKABLE", "a definition of 'foo' opens with U+FEFF where the diff "
                                                              "does not show it is line 1 of its file, the one place "
                                                              "CPython reads one")]),
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
    assert got_git == _ninth(diff, want)                              # NOTE_path2_ninth_pass


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
    # NOTE_path2_eighth_pass: the only_touches oracle (G-C7) re-derives the verdict and refuses it as well
    # NOTE_path2_eleventh_pass: and the planted line reads `_norm` outside the switches, so #121 switched off no longer
    # reads as the scorer's revert of #121 (G-C9)
    assert dict(t.violations) == {"G-C4_direction:only_touches": 1, "G-C7_oracle:only_touches_claim": 1,
                                  "G-C9_switch_is_not_the_revert:#121": 1}
    assert not t.attribution["moves_admitted_by_rule"]


def test_v3_a_real_f2_move_is_still_attributed_and_counted(scorer):
    pg = scorer
    t = pg.Tally(name_prs=True)
    t.pair("clean", "Only touches .gitignore.", PROBE, pg.raw_paths(PROBE))
    ff = "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1,2 @@\n x = 0\n+\x0cdef foo():\n"
    t.pair("ff", "Adds function foo.", ff, pg.raw_paths(ff))
    assert not t.violations and t.n["records_moved"] == 1
    # NOTE_path2_ninth_pass (Z-2): main's Python split the form-feed line and its port read it, so the claim now
    # abstains -- still F-2's move, as its revert gives main's claim back -- and F-2 gives no VERIFIED here any more
    assert dict(t.attribution["moves_admitted_by_rule"]) == {"F-2 symbol_added: CONTRADICTED -> UNCHECKABLE": 1}
    assert not t.attribution["new_verified_admitted"]


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
    assert dict(t.violations) == {"G-C4_unattributed_counterfactual:files_changed_count": 1,
                                  "G-C7_oracle:files_changed_count_claim": 1}      # NOTE_path2_eighth_pass: and the count oracle


def test_v3_the_post_amendment_rules_admit_only_their_own_kinds():
    import importlib.util
    spec = importlib.util.spec_from_file_location("path2_gates_admits_only", FRONTIER / "path2_gates.py")
    src = Path(spec.origin).read_text(encoding="utf-8")
    # `admits` is read out of the scorer's source and run alone: no git, no baseline needed
    body = src[src.index("def admits("):src.index("# ── attribution, written out")]
    # NOTE_path2_sixth_pass: F-2 and W-1 admit only on a diff where they can act; stubbed here by name
    ns = {"OFF_TREE_WHY": "is relative to a directory the diff does not name (#121)",
          "split_differs": lambda diff: diff == "SPLIT", "parse_differs": lambda diff: diff == "PARSE"}
    # NOTE_path2_ninth_pass: the names the ninth pass's admissions read, from the scorer's own source
    for name in ("NINTH_PASS_RULES", "PATH_KINDS", "FILE_LIST_KINDS", "Z1_WHY", "Z2_WHY", "Z2_SKEW", "Z12_BC1",
                 "Z12_RAISES", "Z3_DIFFERS", "Z3_APART", "Z4_WHY", "Z5_WHY", "NOT_SURE", "LOOSE_WHY", "COLLIDE_WHY",
                 "UNCOUNTED_WHY", "Y2_WHY", "Y2_TEST", "Y5_WHY", "GUARD_DIFFERS", "GUARD_RAISES", "GUARD_ABSENT"):
        start = src.index(f"\n{name} = ") + 1
        end = src.index("\n", start)
        while src[end + 1:end + 2] in (" ", '"', ")"):
            end = src.index("\n", end + 1)
        exec(src[start:end], ns)  # noqa: S102
    ns["own_read"] = lambda diff: ({}, [], {}, {"differs": "x"} if diff == "DIFFERS" else {})
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
    # NOTE_path2_ninth_pass: each licensed-difference rule only abstains, on the kinds it reads; one whose abstention
    # the claim does not show shaped nothing there and is admitted as that (G-C7 re-derives the claim either way)
    z1 = ("this reading counts 1 added test definitions where main's Python counted 0 and its port 0 "
          + ns["Z1_WHY"] + "; claim says 1")
    assert admits("Z-1", "tests_added", "VERIFIED", "UNCHECKABLE", z1)
    assert not admits("Z-1", "tests_added", "CONTRADICTED", "VERIFIED", z1)
    assert not admits("Z-1", "symbol_added", "VERIFIED", "UNCHECKABLE", z1)
    assert admits("Z-1", "tests_added", "CONTRADICTED", "VERIFIED", "diff adds 1 test functions, claim says 1")
    z2 = "this reading finds no added definition of 'foo' where main's Python did and its port did " + ns["Z2_WHY"]
    assert admits("Z-2", "symbol_added", "VERIFIED", "UNCHECKABLE", z2)
    assert not admits("Z-2", "symbol_added", "VERIFIED", "CONTRADICTED", z2)
    assert not admits("Z-2", "tests_added", "VERIFIED", "UNCHECKABLE", z2)
    z4 = "'a/x.py': " + ns["Z4_WHY"] + " ('b/x.py', status 'M'); #97 licenses the exact and suffix tiers only"
    assert admits("Z-4", "file_touched", "VERIFIED", "UNCHECKABLE", z4)
    assert not admits("Z-4", "file_touched", "UNCHECKABLE", "VERIFIED", z4)
    z5 = "an added definition line in 'a.py' " + ns["Z5_WHY"] + ", and a file CPython refuses defines nothing"
    assert admits("Z-5", "tests_added", "VERIFIED", "UNCHECKABLE", z5) and admits("Z-5", "symbol_added", "V", "UNCHECKABLE", z5)
    assert not admits("Z-5", "tests_added", "VERIFIED", "CONTRADICTED", z5)
    z3 = ns["NOT_SURE"] + ns["Z3_DIFFERS"] + " where no repair accounts for it: main reads 'x' ('M'), which this reading does not"
    assert admits("Z-3", "files_changed_count", "VERIFIED", "UNCHECKABLE", z3, "DIFFERS")
    assert not admits("Z-3", "files_changed_count", "VERIFIED", "CONTRADICTED", z3, "DIFFERS")
    assert not admits("Z-3", "tests_added", "VERIFIED", "UNCHECKABLE", z3, "DIFFERS")
    assert admits("Z-3", "tests_added", "VERIFIED", "CONTRADICTED", z3, "SAME")     # idle on the record


def test_v3_every_revert_names_code_the_instrument_has(scorer):
    pg = scorer
    for rule in pg.RULES:
        with pg.reverted(pg.CF, {rule}):
            pass
    with pg.reverted(pg.CF, pg.RULES):
        # every rule reverted at once gives the baseline back on the pinned pairs
        for p in json.loads(PAIRS.read_text(encoding="utf-8")):
            got = []
            for m in (pg.CF, pg.BASE):
                try:
                    got.append([(c.kind, c.verdict, c.why) for c in m.gate_diff_text(p["summary"], p["diff"]).claims])
                except AttributeError as e:             # NOTE_path2_ninth_pass: main raises on one pinned pair
                    got.append(f"raises {type(e).__name__}")
            assert got[0] == got[1], p["id"]
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
    # NOTE_path2_ninth_pass: with V-1 and W-2 reverted the line is still one the whole-file reading refuses (Z-5, which
    # reads the current definition reading), so the pair that gives main's claim back is V-1 and Z-5
    assert not t.violations and dict(t.attribution["attributed_by"]) == {"V-1+Z-5": 1}
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
    assert dict(t.violations) == {"G-C4_direction:files_changed_count:W-1": 1, "G-C7_oracle:W-1_status": 1,
                                  "G-C7_oracle:files_changed_count_claim": 1}      # NOTE_path2_eighth_pass: and the count oracle


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
         ("tests_added", "UNCHECKABLE", f"diff adds 0 test functions and changes 1, claim says 0; {Y5}")]),
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


W1_K1 = {"added-plus-plus-space-line": "main reads 'x' ('M') from them", "a-hunk-with-both": "main reads 'x' ('M') from them"}


@pytest.mark.parametrize("name", sorted(W1_DOORS))
def test_w1_the_one_parse_reads_every_door_alike(tmp_path, name):
    summary, before, after, want = W1_DOORS[name]
    diff = _git_repo(tmp_path, before, after)
    via_git = [(c.kind, c.verdict, c.why) for c in gate_diff(summary, tmp_path, "HEAD~1", "HEAD").claims]
    _, via_text = _claims(summary, diff)
    # NOTE_path2_tenth_pass (K-1): where main read a file from an exact hunk's `+++` line, W-1's reading of it as
    # content licenses no file-list difference, and the raw door and the port abstain on the file-list claims; the git
    # door reads git's own --name-status and keeps its verdict
    raw_want = _k1(want, W1_K1[name]) if name in W1_K1 else want
    assert via_git == want and via_text == raw_want
    assert _port_claims([(summary, diff)]) == [raw_want]


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
    # NOTE_path2_eighth_pass (Y-2): the count abstains beside it even at line 1 (main's two ports counted it apart)
    assert _claims("Added 1 test.", head)[1] == [("tests_added", "UNCHECKABLE", f"{Y2_TEST}; claim says 1")]
    # NOTE_path2_ninth_pass (Z-2): the parse reads the line-1 definition as CPython does, and main's port read it too, but
    # main's Python did not (its `\s` holds no U+FEFF), so the claim abstains
    assert dg._symbol_hit("test_n", parse_unified_diff(head)[1])
    assert _claims("Added function test_n.", head)[1] == _ninth(
        head, [("symbol_added", "VERIFIED", "added lines do define function 'test_n'")], "test_n")
    assert _claims("Added function test_n.", head)[1][0][1] == "UNCHECKABLE"
    # NOTE_path2_eighth_pass (Y-2): a U+FEFF-led definition the diff does not show is line 1 abstains; the
    # seventh pass read it as defining nothing (main's port read it as a test, main's Python as none)
    assert _claims("Added 1 test.", mid)[1] == [("tests_added", "UNCHECKABLE", f"a test definition {Y2} claim says 1")]
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
    want = _ninth(diff, want)              # NOTE_path2_ninth_pass: where main's ports counted otherwise, Z-1 abstains
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
        # NOTE_path2_ninth_pass (Z-2, Z-5): where main's `\b` ended the name before the character, or the line is one
        # CPython refuses, the claim abstains
        assert _claims(summary, diff)[1] == _ninth(diff, FOO[want]), (shape, ascii(ch))
        items.append((summary, diff))
        got_raw.append(_ninth(diff, FOO[want]))
    assert _port_claims(items) == got_raw
    # the git door, on the new-definition cell
    diff = _git_repo(tmp_path, {SP: "x = 0\n"}, {SP: f"x = 0\ndef foo{ch}():\n    pass\n"})
    via_git = [(c.kind, c.verdict, c.why) for c in gate_diff("Adds function foo.", tmp_path, "HEAD~1", "HEAD").claims]
    assert via_git == _claims("Adds function foo.", diff)[1] == _ninth(diff, FOO[new_want])


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
    assert dict(t.violations) == {"G-C4_unattributed:files_changed_count": 1,
                                  "G-C7_oracle:files_changed_count_claim": 1}      # NOTE_path2_eighth_pass: and the count oracle


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
    # NOTE_path2_ninth_pass: every accusation a post-amendment rule made on these pairs was one where this reading
    # parted from main's, and the licensed-difference rule abstains there -- none is left to waive; the moves the
    # ninth pass's rules make are counted, each an abstention
    assert waived == dict(t.attribution["new_accusations_admitted"]) == {}
    ninth = {k: v for k, v in t.attribution["moves_admitted_by_rule"].items() if "Z-" in k.split(" ")[0]}
    # (two #97 moves are credited to #97 and Z-4 jointly: with #97 reverted the any-tier loop resolves the claim by
    # its basename, which wakes Z-4; Z-4 shaped nothing on the record itself and is admitted as that)
    # (NOTE_path2_eleventh_pass: two more, the pairs whose sentences hold an em dash, an emoji and curly quotes, which
    # both ports' templates read alike, so #97's repair stands there; one reason-only move beside them)
    # (NOTE_path2_twelfth_pass: two more, round 11's same-case control and the case-kept suffix match, where #97's licence
    # holds on the paths as written; one reason-only move on a record of this pass's own differential; and eight credited
    # to #97 and Z-4 jointly that the guard abstains on, a case-only match, a Z-3 doubt or a trailing-space name)
    assert {k: v for k, v in ninth.items() if not k.endswith("-> UNCHECKABLE")} == {
        "Z-4 file_created: UNCHECKABLE -> VERIFIED": 6, "Z-4 file_touched: VERIFIED -> VERIFIED": 2}
    assert t.attribution["attributed_by"]["#97+Z-4"] == 13 and sum(ninth.values()) > 13


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
    # NOTE_path2_ninth_pass: the form feed is a line main's Python split and its port read, so the count abstains
    # (Z-1) before the planted line is reached, and nothing moves to be refused
    assert not t.violations and dict(t.attribution["moves_admitted_by_rule"]) == {
        "F-2 tests_added: VERIFIED -> UNCHECKABLE": 1}
    # the same plant over lines every reading counts alike: the claim oracle refuses it, and so does the
    # counterfactual, since no rule's revert gives main's claim back
    plain = "--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1 +1,3 @@\n x = 0\n+def test_a():\n+def test_b():\n"
    t = pg.Tally(name_prs=True)
    t.pair("plain", "Added 1 test.", plain, pg.raw_paths(plain))
    assert dict(t.violations) == {"G-C4_unattributed_counterfactual:tests_added": 1, "G-C7_oracle:tests_added_claim": 1}


# ─────────────────────────────── NOTE_path2_seventh_pass_2026_09_25: x1, one name table and the claim side

XID_PY = ROOT / "styxx" / "_xid.py"
XID_JS = ROOT / "web" / "gate" / "diffgate.js"
SKEW_SHA256 = "0b7134fd20e249f7ea69f8fcfcc1993bac0ebd5b7fe9e2e41d507594b097bfd3"


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
    table_part, skew_part = js_block.split("const _XID_TABLE =", 1)[1].split("const _XID_SKEW_VERSIONS", 1)
    js_table = "".join(re.findall(r'"([0-9A-Za-z]+)"', table_part))
    assert js_table == _xid.TABLE
    # NOTE_path2_eighth_pass (Y-3): the skew set rides in the same block, pinned the same way
    js_skew = "".join(re.findall(r'"([0-9A-Za-z]+)"', skew_part.split("const _XID_SKEW =", 1)[1]))
    assert js_skew == _xid.SKEW
    assert f'const _XID_SKEW_VERSIONS = "{_xid.SKEW_VERSIONS}";' in js_block
    assert f'const _XID_SKEW_SHA256 = "{_xid.SKEW_SHA256}";' in js_block
    assert hashlib.sha256(_xid.SKEW.encode("ascii")).hexdigest() == _xid.SKEW_SHA256 == SKEW_SHA256
    assert _xid.SKEW_VERSIONS == "13.0.0 14.0.0 15.1.0 16.0.0"
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
    s = g.runs_of(g.skew(g.decode(t), g.load_sources()))      # NOTE_path2_eighth_pass (Y-3), from its sources
    assert hashlib.sha256(s.encode("ascii")).hexdigest() == SKEW_SHA256 and sum(g.decode(s)) == 10153
    for path, block in ((XID_PY, g.python_block(t, s)), (XID_JS, g.js_block(t, s))):
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
    # NOTE_path2_eighth_pass (Y-3): one some supported Python reads differently is named by that reason instead
    assert all(r[0][1] == "UNCHECKABLE" and r[0][2].startswith(
        "the claimed name runs past 'fo' into" if not _xid.in_skew(chr(c)) else f"the claimed name 'fo' meets U+{c:04X}")
        for r, c in zip(py, word_not_ident))
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

Y1_LOOSE = ("a `---` or `+++` line after lines no hunk count holds may be content (a SQL or Lua comment, a `++` "
            "line) or a file header")
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
    count = ("files_changed_count", "VERIFIED", "diff changes 2 files, claim says 2")
    if shape in ("blank-then-hunk", "no-hunk-header"):
        # NOTE_path2_eighth_pass (Y-1): after lines no count placed, a `---`/`+++` pair with no hunk header right
        # after it has no header's shape, so the file list is not certain and the count abstains (main: VERIFIED)
        count = ("files_changed_count", "UNCHECKABLE", f"the diff's file list is not certain: {Y1_LOOSE}; claim says 2")
    assert _claims(summary, diff)[1] == [count, ("tests_added", "VERIFIED", "diff adds 1 test functions, claim says 1")]
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
    # NOTE_path2_tenth_pass (K-1): the lines are content, and main read a file from them, so the raw door and the port
    # abstain on the file list; the git door keeps git's
    raw_want = _k1(want, "main reads 'x' ('M') from them")
    assert via_git == want and _claims(summary, diff)[1] == raw_want and _port_claims([(summary, diff)]) == [raw_want]


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
    # NOTE_path2_ninth_pass (Z-2): main's port read `def foo` there (its ASCII `\b` ends a name before the astral
    # letter) and its Python did not, so "Added function foo." now abstains
    assert [c[1] for c in py[2]] == ["VERIFIED", "UNCHECKABLE"]
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
        # NOTE_path2_eleventh_pass: the verdict is the guard's, recomputed from the final claims
        '    contradicted = any(c.verdict == "CONTRADICTED" for c in final)',
        '    contradicted = any(c.verdict == "CONTRADICTED" for c in final if c.kind != "only_touches")',
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
        # NOTE_path2_eleventh_pass: the guard turns the planted accusation into an abstention (main verifies), so it is
        # the reading before the guard that G-C7 refuses
        "G-C7_oracle:tests_added_claim"),
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
    assert sample == {"tried": 1, "scored": 1} and not t.violations and t.n["records_moved"] == 1
    # planted: the git door hands the gate no sides, so its pairing is blind (the raw door's is not)
    _planted(pg, monkeypatch, '    sides = parse_unified_diff_sides(diff_text) if rp.on("#121") else _read_diff(diff_text, None, rp)[2]',
             "    sides = None", "x7git")
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "planted", summary, diff, sample)
    assert t.violations["G-C8_git_door_differs_from_the_raw_door"] == 1
    # a diff that cannot be rebuilt faithfully is counted, not scored
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "sloppy", "2 files changed.", X2_OVER, sample)
    assert sample == {"tried": 1, "not_rebuildable": 1} and not t.n


def _load_scorer_with(monkeypatch, fake_xid):
    """Load path2_gates afresh against a fake styxx._xid (its import-time check_tables call runs)."""
    import sys
    monkeypatch.setitem(sys.modules, "styxx._xid", fake_xid)
    monkeypatch.setitem(sys.modules, "styxx.claimdetect", sys.modules.get("styxx.claimdetect"))
    spec = importlib.util.spec_from_file_location("path2_gates_refused", FRONTIER / "path2_gates.py")
    spec.loader.exec_module(importlib.util.module_from_spec(spec))


def _fake_xid(table=None, skew=None):
    import types
    from styxx import _xid
    fake = types.ModuleType("styxx._xid")
    fake.TABLE, fake.UNICODE_VERSION = table or _xid.TABLE, "15.0.0"
    fake.TABLE_SHA256 = hashlib.sha256(fake.TABLE.encode("ascii")).hexdigest()
    fake.SKEW, fake.SKEW_VERSIONS = skew or _xid.SKEW, _xid.SKEW_VERSIONS
    fake.SKEW_SHA256 = hashlib.sha256(fake.SKEW.encode("ascii")).hexdigest()
    return fake


def _flip_first_run(encoded: str, mask: int) -> str:
    from styxx import _xid
    g = _gen_xid()
    starts, masks = _xid.decode(encoded)
    runs = [(nxt - s, m) for s, nxt, m in zip(starts, starts[1:] + [0x110000], masks)]
    runs[0] = (runs[0][0], mask)
    return "".join(g.encode(n * 8 + k) for n, k in runs)


def test_x8_the_scorer_pins_the_table_and_the_skew_set_as_literals(scorer):
    # NOTE_path2_eighth_pass (round-7 protocol lens, minor): the scorer's pins are literals in its own text,
    # not the sha256s the block states about itself, and the import-time check reads them
    from styxx import _xid
    pg = scorer
    src = (FRONTIER / "path2_gates.py").read_text(encoding="utf-8")
    assert 'XID_TABLE_SHA256_PINNED = "8df68f217cca495ab8a38ced9096213aabac4cf23927068d61397d2c9074d4cb"' in src
    assert f'XID_SKEW_SHA256_PINNED = "{SKEW_SHA256}"' in src
    assert "check_tables(_XID, XID_TABLE_SHA256_PINNED, XID_SKEW_SHA256_PINNED)" in src
    starts, masks, skew = pg.check_tables(_xid, pg.XID_TABLE_SHA256_PINNED, pg.XID_SKEW_SHA256_PINNED)
    assert (starts, masks, skew) == (pg.XID_STARTS, pg.XID_MASKS, pg.SKEW) and sum(skew) == 10153


@pytest.mark.skipif(unicodedata.unidata_version != "15.0.0", reason="the scorer checks the table against a 15.0.0 database only")
def test_x7_the_scorer_refuses_a_name_table_its_database_does_not_read(scorer, monkeypatch):
    # A table whose block hash was made to match after one run was changed. NOTE_path2_eighth_pass (round-7
    # protocol lens, minor): the scorer pins the literal sha256 itself, so the block's own pin cannot vouch for
    # it; and with that pin also rewritten (a scorer edited to match), the database still refuses it.
    pg = scorer
    table = _flip_first_run(__import__("styxx._xid", fromlist=["TABLE"]).TABLE, 4)   # U+0000.. read as words
    fake = _fake_xid(table=table)
    with pytest.raises(SystemExit, match="table does not hash to the sha256 this file pins"):
        pg.check_tables(fake, pg.XID_TABLE_SHA256_PINNED, pg.XID_SKEW_SHA256_PINNED)
    with pytest.raises(SystemExit, match="name table reads U\\+0000"):
        pg.check_tables(fake, fake.TABLE_SHA256, pg.XID_SKEW_SHA256_PINNED)
    with pytest.raises(SystemExit, match="table does not hash to the sha256 this file pins"):
        _load_scorer_with(monkeypatch, fake)          # at import, too


@pytest.mark.skipif(unicodedata.unidata_version != "15.0.0", reason="the scorer runs on a 15.0.0 database only")
def test_x8_the_scorer_refuses_a_skew_set_its_sources_do_not_give(scorer, monkeypatch):
    # NOTE_path2_eighth_pass (Y-3): the skew set is pinned by a literal in the scorer and re-derived from
    # web/gate/xid_versions.json there; a set with one run changed fails either way.
    from styxx import _xid
    pg = scorer
    skew = _flip_first_run(_xid.SKEW, 1)          # U+0000.. put in the set
    fake = _fake_xid(skew=skew)
    with pytest.raises(SystemExit, match="skew set does not hash to the sha256 this file pins"):
        pg.check_tables(fake, pg.XID_TABLE_SHA256_PINNED, pg.XID_SKEW_SHA256_PINNED)
    with pytest.raises(SystemExit, match="skew set is not the one its sources give"):
        pg.check_tables(fake, pg.XID_TABLE_SHA256_PINNED, fake.SKEW_SHA256)
    with pytest.raises(SystemExit, match="skew set does not hash to the sha256 this file pins"):
        _load_scorer_with(monkeypatch, fake)


def test_x8_the_scorer_refuses_to_run_on_another_unicode(scorer, monkeypatch):
    # the database check is blocking: on a Python whose Unicode is not 15.0.0 the scorer exits
    from styxx import _xid
    pg = scorer
    with pytest.raises(SystemExit, match="this Python reads Unicode 16.0.0"):
        pg.check_tables(_xid, pg.XID_TABLE_SHA256_PINNED, pg.XID_SKEW_SHA256_PINNED, unidata="16.0.0")
    monkeypatch.setattr(unicodedata, "unidata_version", "16.0.0")
    with pytest.raises(SystemExit, match="this Python reads Unicode 16.0.0"):
        _load_scorer_with(monkeypatch, _fake_xid())


def test_x8_the_path_claim_oracle_reads_every_path_claim(scorer):
    # a path claim the repaired gate read other than the scorer reads it is refused by G-C7 (round-7 protocol
    # lens: the file-list kinds join the oracles)
    pg = scorer
    summary, diff = "Modified src/a.py. Created src/a.py.", _m("src/a.py")
    g = pg.new.gate_diff_text(summary, diff, run=None, strict=False)
    assert [c.kind for c in g.claims] == ["file_touched", "file_created"]
    assert not pg.oracle_violations(summary, diff, pg.raw_paths(diff), g)
    g.claims[0].verdict = "UNCHECKABLE"
    assert "G-C7_oracle:file_touched_claim" in pg.oracle_violations(summary, diff, pg.raw_paths(diff), g)


def test_x8_the_git_door_counts_are_part_of_its_verdict(scorer):
    from collections import Counter
    pg = scorer
    for sample, violation in (({"tried": 2, "scored": 1}, "G-C8_records_unaccounted"),
                              ({"tried": 1, "not_rebuildable": 1}, "G-C8_nothing_scored"),
                              ({"tried": 1, "rebuild_failed": 1}, "G-C8_rebuild_failed")):
        t = pg.Tally(name_prs=True)
        rep = t.door_report(Counter(sample))
        assert not rep["pass"] and violation in rep["violations"], (sample, rep["violations"])
    t = pg.Tally(name_prs=True)
    assert t.door_report(Counter({"tried": 3, "scored": 1, "not_rebuildable": 1, "rebuilt_to_no_change": 1}))["pass"]


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
    assert len(pairs) == 299 and all(p["id"].startswith("path2:") for p in pairs)
    # NOTE_path2_fifth_pass V-1 re-pinned four pairs and NOTE_path2_sixth_pass W-1 one; NOTE_path2_eighth_pass
    # twenty-four (Y-5 thirteen: the pairing withdraws; Y-2 four; Y-1 four; Y-3 three), each to UNCHECKABLE;
    # NOTE_path2_ninth_pass thirty-six (Z-2 sixteen, Z-1 twelve, Z-3 seven, Z-4 one), each to UNCHECKABLE;
    # NOTE_path2_tenth_pass nine (K-1: six to UNCHECKABLE, three reason-only); NOTE_path2_eleventh_pass two (the guard
    # abstaining where F-2 alone parts from main; K-5 at the sentence leaving a path both ports read alike to Z-4), each
    # to UNCHECKABLE; NOTE_path2_twelfth_pass five (#121 licensing nothing in a plain rendering), each to UNCHECKABLE;
    # each record says so
    assert sorted(p["id"] for p in pairs if "repinned" in p) == [
        "path2:101-a-bom-strip-changes-a-test",
        "path2:101-a-changed-test-and-a-same-named-new-one",
        "path2:101-a-new-test-beside-a-changed-one",
        "path2:101-non-ascii-test-names-are-distinct",
        "path2:101-the-fold-under-an-M-header-pairs-once",
        "path2:101-the-same-test-name-in-two-classes",
        "path2:121-a-dotdot-path-is-not-a-dot-miss",
        "path2:121-a-dotfile-and-a-nested-undotted-name",
        "path2:121-a-dotted-prefix-over-an-undotted-directory",
        "path2:121-a-dotted-prefix-over-an-undotted-file",
        "path2:97-basename-still-resolves",
        "path2:f1-c3-accuses-a-dotfile-prefix-over-its-undotted-twin",
        "path2:f2-a-context-line-holding-a-separator-adds-nothing",
        "path2:f2-a-form-feed-indent-defines-a-function",
        "path2:f2-a-form-feed-is-not-a-line-break",
        "path2:f2-a-line-separator-is-not-a-line-break",
        "path2:f2-a-line-separator-mid-line-starts-no-line",
        "path2:f2-a-paragraph-separator-is-not-a-line-break",
        "path2:f2-a-separator-before-a-header-shape-adds-no-file",
        "path2:f2-a-vertical-tab-is-not-a-line-break",
        "path2:f2-limit-hit-reads-a-line-separator-as-indent",
        "path2:f3-an-ideographic-space-reindent-is-not-an-added-test",
        "path2:f3-an-nbsp-reindent-is-not-an-added-test",
        "path2:k5-a-path-the-sentence-runs-into-from-a-non-ascii-character-reads-as-main-read-it",
        "path2:r1-a-bom-on-a-changed-test-beside-two-new-ones",
        "path2:r1-a-bom-on-a-changed-test-does-not-hide-a-new-one",
        "path2:r2-an-async-test-made-sync-is-a-changed-test",
        "path2:r4-a-changed-generic-definition-still-verifies",
        "path2:r4-a-function-made-generic-still-verifies",
        "path2:v1-a-def-after-a-mid-line-line-separator-is-not-a-definition",
        "path2:v1-a-form-feed-led-new-test-is-counted",
        "path2:v1-a-form-feed-reindented-function-is-changed",
        "path2:v1-a-form-feed-separated-changed-function-pairs",
        "path2:v1-a-name-followed-by-a-middle-dot-is-another-name",
        "path2:v1-a-name-followed-by-a-non-ascii-letter-is-another-name",
        "path2:v1-a-vertical-tab-reindented-function-defines-nothing",
        "path2:v1-an-nbsp-led-new-function-defines-nothing",
        "path2:v1-limit-a-bom-led-def-reads-as-a-definition-wherever-it-stands",
        "path2:v2-a-binary-deletion-line-holding-a-line-separator",
        "path2:v2-a-binary-header-holding-a-line-separator-registers-its-file",
        "path2:v2-a-bom-after-a-header-path-is-kept",
        "path2:v2-a-header-path-is-stripped-as-python-strips-it",
        "path2:v2-a-reason-prints-a-path-as-python-repr-does",
        "path2:w1-a-bom-opening-line-one-of-a-new-file-is-a-bom",
        "path2:w1-a-hunk-with-both-shapes",
        "path2:w1-a-no-newline-marker-inside-a-hunk-does-not-end-it",
        "path2:w1-a-removed-sql-comment-does-not-hide-a-changed-test",
        "path2:w1-a-short-hunk-before-an-a-b-header-pair-with-no-hunk-header",
        "path2:w1-an-added-line-opening-with-plus-plus-space-is-not-a-file",
        "path2:w1-limit-two-dashes-and-two-pluses-before-a-hunk-header-read-as-a-file-header",
        "path2:w2-a-name-followed-by-a-no-break-space-defines-nothing",
        "path2:w2-a-test-made-generic-pairs",
        "path2:w2-a-test-name-followed-by-a-form-feed-pairs",
        "path2:w2-a-test-name-followed-by-a-no-break-space-defines-nothing",
        "path2:w2-a-test-separated-from-def-by-a-tab-is-counted",
        "path2:w2-a-test-whose-separator-changed-pairs",
        "path2:w2-a-test-with-two-spaces-after-def-is-counted",
        "path2:x1-the-name-table-is-unicode-15-for-both-ports-a-katakana-middle-dot",
        "path2:x1-the-name-table-is-unicode-15-for-both-ports-a-letter-unicode-16-added",
        "path2:x1-the-name-table-is-unicode-15-for-both-ports-a-zero-width-joiner",
        "path2:x2-a-removed-a-slash-line-beside-an-added-b-slash-line-in-git-output-is-content",
        "path2:x2-an-exact-hunk-whose-last-lines-are-a-dash-pair-ends-at-the-diff-git-line",
        "path2:x2-an-over-declared-hunk-then-a-gnu-header-a-blank-line-and-a-hunk",
        "path2:x2-an-over-declared-hunk-then-a-gnu-header-and-no-hunk-header",
        "path2:y1-a-binary-line-under-a-git-header-is-counted",
        "path2:y2-a-created-files-line-one-bom-under-a-bare-hunk-is-line-one",
        "path2:y2-a-created-files-line-one-bom-under-an-over-declared-hunk-is-line-one",
        "path2:y2-an-over-declared-hunk-starting-at-one-shows-line-one",
        "path2:y3-a-symbol-definition-running-into-a-skew-letter-defines-nothing-for-the-prefix",
        "path2:y4-a-gnu-creation-with-a-timestamp-is-a-creation",
        "path2:y4-gnu-deletions-with-timestamps-are-two-files",
        "path2:z3-a-no-prefix-binary-header-nobody-reads-beside-a-plus-content-line",
        "path2:z3-r5-a-binary-section-replaced-by-a-gnu-pair"]
    assert sum("NOTE_path2_eighth_pass" in p.get("repinned", "") for p in pairs) == 24
    assert sum("NOTE_path2_ninth_pass" in p.get("repinned", "") for p in pairs) == 36
    assert sum("NOTE_path2_tenth_pass" in p.get("repinned", "") for p in pairs) == 9
    assert sum("NOTE_path2_eleventh_pass" in p.get("repinned", "") for p in pairs) == 2
    for p in pairs:
        g = gate_diff_text(p["summary"], p["diff"], run=None, strict=False)
        with_why = bool(p["expect"]["claims"]) and len(p["expect"]["claims"][0]) == 3
        got = [[c.kind, c.verdict, c.why] if with_why else [c.kind, c.verdict] for c in g.claims]
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


# ─────────────────────────────── NOTE_path2_eighth_pass_2026_09_27: every round-7 scorer finding, planted

X8_BOM, X8_ZWNJ = chr(0xFEFF), chr(0x200C)
X8_TWO = ("--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
          "--- a/lib/b.py\n+++ b/lib/b.py\n@@ -1 +1 @@\n-b = 1\n+b = 2\n")
X8_PLANTED = {
    # label: (good text in styxx/diffgate.py, planted text, summary, diff, the violation that must fire)
    # round-7 protocol lens, blocker: a declared (DECLARE-1) claim is re-read, so a defect it alone triggers fails
    "declared: `_symbol_hit` matches a name's prefix": (
        '    return any(_defines(line, name) for line in added_blob.split("\\n"))',
        '    return any((_defined_name(line) or ("", ""))[1].startswith(name) for line in added_blob.split("\\n"))',
        "Adds things.\n\n```styxx\nadds_symbol: foo\n```\n", f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,2 @@\n x = 0\n+def foobar():\n",
        "G-C7_oracle:symbol_added_claim"),
    "declared: the runs-past rule removed": (
        "    if end < len(sentence) and _xid_word(sentence[end]):", "    if False:",
        "Adds things.\n\n```styxx\nadds_symbol: fo" + chr(0xB2) + "o\n```\n",
        f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,2 @@\n x = 0\n+def fo():\n", "G-C7_oracle:symbol_added_claim"),
    "declared: limit 12(b), `net == n` read as `net >= n`": (
        "                        elif net == n:\n", "                        elif net >= n:\n",
        "Adds tests.\n\n```styxx\ntests_added: 1\n```\n",
        # NOTE_path2_ninth_pass: two plain tests (a form-feed-led one is now behind Z-1, which abstains before it)
        f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,3 @@\n x = 0\n+def test_a():\n+def test_b():\n",
        "G-C7_oracle:tests_added_claim"),
    "declared: A-1 ignoring the removed side": (
        '        if (status or {}).get(path) != "A":\n            for line in removed:',
        '        if False:\n            for line in removed:',
        "Adds tests.\n\n```styxx\ntests_added: 0\n```\n",
        f"--- a/{TP}\n+++ b/{TP}\n@@ -1,2 +1,2 @@\n-async def test_x():\n+async def test_x():  # c\n     pass\n",
        "G-C7_oracle:tests_added_claim"),
    # round-7 protocol lens, major: V-4, F-4 and C-2 are re-derived, not only reverted
    "V-4: any dots-only segment read as a parent": (
        '    return "/".join(segs) if segs and segs[-1] == ".." else ""',
        '    return "/".join(segs) if segs and len(segs[-1]) > 1 and not segs[-1].strip(".") else ""',
        "Only touches src/...", X8_TWO, "G-C7_oracle:only_touches_claim"),
    "F-4: every path could lie under an off-tree prefix": (
        "    if _parent_prefix(raw):\n        return True\n    named = False",
        "    if True:\n        return True\n    named = False",
        "Only touches src/ and ../docs/.", X8_TWO, "G-C7_oracle:only_touches_claim"),
    "C-2: the scaffold reading on the dotted key": (
        "not _COMPAT_SCAFFOLD.search(_undotted(path))", "not _COMPAT_SCAFFOLD.search(path)",
        "Keeps backward compatibility.",
        "--- a/.docs/api.py\n+++ b/.docs/api.py\n@@ -1,2 +1 @@\n-def api():\n-    pass\n+x = 1\n"
        "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,2 +1 @@\n-def core():\n-    pass\n+x = 2\n",
        "G-C7_oracle:C-2_compat_surface"),
    # the eighth pass's own rules
    "Y-1: every `---`/`+++` pair read with a header's shape": (
        '    return x == y or "/dev/null" in (x, y)', "    return True",
        "1 file changed.", "--- a/src/q.sql\n+++ b/src/q.sql\n@@\n-a\n--- users\n+++ orders\n@@\n+b\n",
        "G-C7_oracle:Y_notes"),
    "Y-1: two paths one key in case, not noted": (
        '        if forms.setdefault(_case_fold(form), form) != form:\n            found.setdefault("files", _Y1_COLLIDE)',
        '        if False:\n            found.setdefault("files", _Y1_COLLIDE)',
        "2 files changed.", "--- a/docs/Guide.md\n+++ b/docs/Guide.md\n@@ -1 +1 @@\n-a\n+b\n"
                            "--- a/docs/guide.md\n+++ b/docs/guide.md\n@@ -1 +1 @@\n-c\n+d\n",
        "G-C7_oracle:files_changed_count_claim"),
    "Y-2: a U+FEFF dropped outside the counts wherever it stands": (
        "    return text[len(_FILE_BOM):] if at_one and text.startswith(_FILE_BOM) else text",
        "    return text[len(_FILE_BOM):] if text.startswith(_FILE_BOM) else text",
        "Adds function foo.", f"--- a/{SP}\n+++ b/{SP}\n@@\n x = 0\n+" + X8_BOM + "def foo():\n",
        "G-C7_oracle:W-1_added_lines"),
    "Y-2: a U+FEFF dropped before a test at line 1, not noted": (
        "    return _Y2_TEST if text != raw and _test_name(text) else None", "    return None",
        "Added 1 test.", "--- /dev/null\n+++ b/tests/test_n.py\n@@ -0,0 +1,2 @@\n+" + X8_BOM + "def test_n():\n+    pass\n",
        "G-C7_oracle:tests_added_claim"),
    "Y-3: the skew set read as empty": (
        "    return _xid_skew(ch)", "    return False",
        "Added function fo" + chr(0x105C0) + ".", f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,2 @@\n x = 0\n+def fo():\n",
        "G-C7_oracle:symbol_added_claim"),
    "Y-3: a test name's skew code point not doubted": (
        "    return next((ch for ch in wide if _skew(ch)), None)", "    return None",
        "Added 1 test.", f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,2 @@\n x = 0\n+def test_a" + X8_ZWNJ + "b():\n",
        "G-C7_oracle:Y-3_skew_test"),
    "Y-4: /dev/null with a GNU timestamp not recognised": (
        '    return path == "/dev/null" or (path or "").startswith("/dev/null\\t")', '    return path == "/dev/null"',
        "1 file changed.", "--- docs/a.md\t2024-01-01\n+++ /dev/null\t2024-01-01\n@@ -1 +0,0 @@\n-a\n",
        "G-C7_oracle:W-1_status"),
    "Y-2: the indent reading drops nothing (only its own oracle sees it)": (
        '    if "\\ufeff" not in lead:\n        return None', "    return None",
        "1 file changed.", f"--- a/{SP}\n+++ b/{SP}\n@@ -1 +1,2 @@\n x = 0\n+" + X8_BOM + "x = 1\n",
        "G-C7_oracle:Y-2_bom_hidden"),
    # each eighth-pass rule admits only its own kind of move (a planted verdict its revert gives back)
    "Y-1 accusing a count instead of abstaining": (
        "                    elif not_sure:                      # NOTE_path2_eighth_pass (Y-1)\n"
        '                        c.verdict, c.why = "UNCHECKABLE", f"{not_sure}; claim says {n}"',
        "                    elif not_sure:                      # NOTE_path2_eighth_pass (Y-1)\n"
        '                        c.verdict, c.why = "CONTRADICTED", f"{not_sure}; claim says {n}"',
        "2 files changed.", "--- a/src/c.md\n+++ b/src/c.md\n@@\n-x\n+++ plus\n y\n",
        "G-C7_oracle:files_changed_count_claim"),       # NOTE_path2_eleventh_pass: the guard abstains; the reading is refused
    "Y-2 abstaining where no U+FEFF stands": (
        "    return _Y2_TEST if text != raw and _test_name(text) else None",
        "    return _Y2_TEST if _test_name(text) else None",
        "Added 1 test.", f"--- a/{TP}\n+++ b/{TP}\n@@\n x = 0\n+def test_a():\n",
        "G-C4_direction:tests_added:Y-2"),
    "Y-4 reading a /dev/null with no TAB after it": (
        '    return path == "/dev/null" or (path or "").startswith("/dev/null\\t")',
        '    return (path or "").startswith("/dev/null")',
        "Only touches docs/.", "--- a/docs/a.md\n+++ /dev/nullx\n@@ -1 +0,0 @@\n-a\n",
        "G-C4_direction:only_touches:Y-4"),
    "Y-5 contradicting where no test is paired": (
        "                        elif net == n and _pairing_withdraws(chg):\n"
        "                            # NOTE_path2_eighth_pass (Y-5): the pairing withdraws, it does not verify\n"
        '                            c.verdict = "UNCHECKABLE"',
        "                        elif net == n and _pairing_withdraws(chg + 1):\n"
        "                            # NOTE_path2_eighth_pass (Y-5): the pairing withdraws, it does not verify\n"
        '                            c.verdict = "CONTRADICTED"',
        "Added 1 test.", f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,2 @@\n x = 0\n+def test_a():\n",
        "G-C7_oracle:tests_added_claim"),               # NOTE_path2_eleventh_pass: the guard abstains; the reading is refused
    "Y-5: the pairing verifies `net` again": (
        "    return chg > 0", "    return False",
        "Added 1 test.", (f"--- a/{TP}\n+++ b/{TP}\n@@ -1,2 +1,4 @@\n-def test_a():\n+def test_a(x):\n     pass\n"
                          "+def test_b():\n+    pass\n"),
        "G-C7_oracle:tests_added_claim"),
    # (the move is #101's by the counterfactual -- reverting the pairing gives main's reading back -- so the oracle
    # is what refuses an accusation here; Y-5's own admission asks for an abstention besides)
    "Y-5 accusing instead of abstaining": (
        '                            # NOTE_path2_eighth_pass (Y-5): the pairing withdraws, it does not verify\n'
        '                            c.verdict = "UNCHECKABLE"',
        '                            # NOTE_path2_eighth_pass (Y-5): the pairing withdraws, it does not verify\n'
        '                            c.verdict = "CONTRADICTED"',
        "Added 1 test.", (f"--- a/{TP}\n+++ b/{TP}\n@@ -1,2 +1,4 @@\n-def test_a():\n+def test_a(x):\n     pass\n"
                          "+def test_b():\n+    pass\n"),
        "G-C7_oracle:tests_added_claim"),
}


@pytest.mark.parametrize("label", sorted(X8_PLANTED))
def test_x8_every_round_7_finding_and_eighth_pass_rule_fails_on_a_planted_defect(scorer, monkeypatch, label):
    # Round-7 protocol lens: a defect triggered only by a declared claim, one inside V-4, F-4 or C-2, and (this
    # round) one inside each eighth-pass rule's own code. Each is refused; the same record scores clean beforehand.
    pg = scorer
    good, bad, summary, diff, violation = X8_PLANTED[label]
    t = pg.Tally(name_prs=True)
    t.pair("clean", summary, diff, pg.raw_paths(diff))
    assert not t.violations, t.violating
    _planted(pg, monkeypatch, good, bad, "x8")
    t = pg.Tally(name_prs=True)
    t.pair("planted", summary, diff, pg.raw_paths(diff))
    assert violation in t.violations, dict(t.violations)


def test_x8_the_scorer_reads_w1_exactness_by_the_ports_rule(scorer):
    # Round-7 protocol lens, minor: the scorer reset its flag on a removed line, the ports only on a context
    # line, and the scorer raised G-C7_oracle:W-1_sides on the unplanted branch. One rule now: an added line
    # opens a stretch only a context line closes.
    pg = scorer
    shape = "--- a/src/m.sql\n+++ b/src/m.sql\n@@ -1,2 +1,1 @@\n+x\n-y\n--- z\n"
    for diff in (shape, shape + "--- a/src/n.sql\n+++ b/src/n.sql\n@@ -1 +1 @@\n-p\n+q\n",
                 "--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n+x\n y\n-z\n--- w\n", X2_OVER):
        lines = pg.git_lines(diff)
        for k, line in enumerate(lines):
            if pg.HUNK.match(line):
                assert pg.hunk_exact(lines, k) == dg._hunk_is_exact(lines, k), (diff, k)
    t = pg.Tally(name_prs=True)
    t.pair("w1", "Modified src/m.sql. 1 file changed.", shape, pg.raw_paths(shape))
    assert not t.violations, t.violating


def test_x8_the_git_door_scores_every_record_in_differential_mode(scorer, monkeypatch, tmp_path):
    # Round-7 protocol lens, major: G-C8 sampled 150 records, so a defect reachable only through gate_diff was
    # admitted unless its record was sampled. Differential mode now tries every record; a smoke run that stops
    # early fails the gate, and the counts are part of the verdict.
    pg = scorer
    d = tmp_path / "differential"
    d.mkdir()
    tab = (f"--- a/{TP}\n+++ b/{TP}\n@@ -1,2 +1,4 @@\n class T:\n \tpass\n+\tdef\ttest_a(self):\n+\t\tpass\n")
    rows = [{"id": f"r{i}", "summary": "Added 1 test. 1 file changed.", "diff": tab if i == 7 else _m(f"src/m{i}.py")}
            for i in range(12)]
    for name in pg.DIFFERENTIAL_FILES:
        (d / name).write_text(json.dumps(rows if name == "corpus_real.json" else []), encoding="utf-8")
    monkeypatch.setattr(pg, "DIFFERENTIAL", d)
    out = tmp_path / "gates.json"
    pg.run_differential(out)          # (G-C0 fails on an uncommitted tree; the door's own section is read here)
    door = json.loads(out.read_text(encoding="utf-8"))["G-C8_git_door"]
    assert door["sample"]["tried"] == door["sample"]["records"] == 12 and door["sample"]["every_record_tried"] == 1
    assert door["sample"]["scored"] == 12 and door["pass"], door["violations"]
    # a smoke run that stops after one record fails the gate
    assert pg.run_differential(out, git_sample=1) == 1
    assert "G-C8_not_every_record_tried" in json.loads(out.read_text(encoding="utf-8"))["G-C8_git_door"]["violations"]
    # planted, reachable through gate_diff alone: the git door's added blob drops tab-led lines
    _planted(pg, monkeypatch, "    added_blob = parse_unified_diff(diff_text)[1]\n    # NOTE_path2_eighth_pass: the file list here",
             '    added_blob = "\\n".join(x for x in parse_unified_diff(diff_text)[1].split("\\n") if x[:1] != "\\t")\n'
             "    # NOTE_path2_eighth_pass: the file list here", "x8git")
    assert pg.run_differential(out) == 1
    violations = json.loads(out.read_text(encoding="utf-8"))["G-C8_git_door"]["violations"]
    assert "G-C8_git_door_differs_from_the_raw_door" in violations, violations


def test_x8_corpus_mode_scores_every_pr_with_a_definition_claim_through_the_git_door(scorer, monkeypatch, tmp_path):
    # the same git-door-only defect on a synthetic shelf, with the default sampling: a PR whose claims include
    # tests_added or symbol_added is always scored (gate_diff feeds exactly those through its own blob and sides)
    pg = scorer
    patch = "@@ -1,2 +1,4 @@\n class T:\n \tpass\n+\tdef\ttest_a(self):\n+\t\tpass"      # as the shelf holds one
    pid = next(i for i in range(1, 500) if pg.sample_key(i) % 25)        # not in the 1-in-25 sample
    shelf = _shelf(tmp_path, [(pid, "Added 1 test.", [(TP, "modified", patch)])])
    out = tmp_path / "gates.json"
    pg.run_corpus(shelf, None, out)   # (G-C0 fails on an uncommitted tree; the door's own section is read here)
    door = json.loads(out.read_text(encoding="utf-8"))["G-C8_git_door"]
    assert door["sample"]["scored"] == 1 and door["sample"]["tried_for_a_definition_claim"] == 1
    assert door["pass"], door["violations"]
    _planted(pg, monkeypatch, "    added_blob = parse_unified_diff(diff_text)[1]\n    # NOTE_path2_eighth_pass: the file list here",
             '    added_blob = "\\n".join(x for x in parse_unified_diff(diff_text)[1].split("\\n") if x[:1] != "\\t")\n'
             "    # NOTE_path2_eighth_pass: the file list here", "x8shelf")
    (tmp_path / "planted").mkdir()
    shelf2 = _shelf(tmp_path / "planted", [(pid, "Added 1 test.", [(TP, "modified", patch)])])
    assert pg.run_corpus(shelf2, None, out) == 1
    door = json.loads(out.read_text(encoding="utf-8"))["G-C8_git_door"]
    assert not door["pass"] and "G-C8_git_door_differs_from_the_raw_door" in door["violations"], door["violations"]


def test_x8_a_git_door_that_fails_to_rebuild_fails_the_gate(scorer, monkeypatch):
    from collections import Counter
    pg = scorer

    class Broken:
        def __init__(self, *a):
            raise RuntimeError("git is not there")
    monkeypatch.setattr(pg, "GitDoor", Broken)
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "broken", "1 file changed.", _m("src/a.py"), sample)
    assert sample == {"tried": 1, "rebuild_failed": 1}
    rep = t.door_report(sample)
    assert not rep["pass"] and t.violations["G-C8_rebuild_failed"] == 1


# ─────────────────────────────── NOTE_path2_eighth_pass_2026_09_27: the eighth pass's own shapes, both ports

Y1_UNCOUNTED = ("the diff's file list is not certain: a line names a changed file no header pair counts (GNU's "
                "`Binary files ... differ`, `Only in ...` and the like)")
X8_TS = "\t2024-01-01 00:00:00.000000000 +0000"
X8_SHAPES = {
    # probes: `main` right only by a second miscount the repair removed; each now abstains, both ports alike
    "a markdown test beside a changed test": (
        "Added 1 test.", "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1,3 @@\n # T\n"
        "+def test_doc():\n+    pass\n"
        f"diff --git a/{TP} b/{TP}\n--- a/{TP}\n+++ b/{TP}\n@@ -1,2 +1,2 @@\n-def test_a():\n+def test_a(x):\n     pass\n",
        [("tests_added", "UNCHECKABLE", f"diff adds 1 test functions and changes 1, claim says 1; {Y5}")]),
    "a def inside a string opened in unchanged lines, beside a changed test": (
        "Added 1 test.", f"--- a/{TP}\n+++ b/{TP}\n@@ -2,3 +2,4 @@ S = \"\"\"\n alpha\n+def test_s():\n beta\n \"\"\"\n"
        "@@ -9,2 +10,2 @@\n-def test_a():\n+def test_a(x):\n     pass\n",
        [("tests_added", "UNCHECKABLE", f"diff adds 1 test functions and changes 1, claim says 1; {Y5}")]),
    "a U+FEFF-led test at line 1 beside a string-held def": (
        "Added 1 test.", "--- /dev/null\n+++ b/tests/test_b.py\n@@ -0,0 +1,2 @@\n+" + X8_BOM + "def test_new():\n+    pass\n"
        "--- a/tests/test_c.py\n+++ b/tests/test_c.py\n@@ -1 +1,4 @@\n X = 1\n+S = \"\"\"\n+def test_str():\n+\"\"\"\n",
        [("tests_added", "UNCHECKABLE", f"{Y2_TEST}; claim says 1")]),
    # Y-1: a file GNU names that no header pair counts, beside a phantom an exact hunk no longer makes
    "a GNU binary line and a `++` content line": (
        "2 files changed. Only touches src/.",
        "diff -ru a/docs/img.png b/docs/img.png\nBinary files a/docs/img.png and b/docs/img.png differ\n"
        f"diff -ru a/src/c.md b/src/c.md\n--- a/src/c.md{X8_TS}\n+++ b/src/c.md{X8_TS}\n@@ -1,2 +1,2 @@\n-x\n+++ plus\n y\n",
        [("files_changed_count", "UNCHECKABLE", f"{Y1_UNCOUNTED}; claim says 2"), ("only_touches", "UNCHECKABLE", Y1_UNCOUNTED)]),
    "a GNU only-in line and a `++` content line": (
        "2 files changed. Only touches src/.",
        "Only in b/docs: new.md\n--- a/src/c.md\n+++ b/src/c.md\n@@ -1,2 +1,2 @@\n-x\n+++ plus\n y\n",
        [("files_changed_count", "UNCHECKABLE", f"{Y1_UNCOUNTED}; claim says 2"), ("only_touches", "UNCHECKABLE", Y1_UNCOUNTED)]),
    "a binary line under a git header is counted, and sure": (
        "2 files changed.",
        "diff --git a/docs/img.png b/docs/img.png\nindex 1..2 100644\nBinary files a/docs/img.png and b/docs/img.png differ\n"
        "diff --git a/src/c.md b/src/c.md\n--- a/src/c.md\n+++ b/src/c.md\n@@ -1,2 +1,2 @@\n-x\n+++ plus\n y\n",
        # NOTE_path2_tenth_pass (K-1): counted and sure, but main read 'plus' from the exact hunk's `+++ plus`, so the
        # file list abstains (was VERIFIED)
        _k1([("files_changed_count", "VERIFIED", "diff changes 2 files, claim says 2")], "main reads 'plus' ('M') from them")),
}


@pytest.mark.parametrize("shape", sorted(X8_SHAPES))
def test_x8_each_shape_reads_alike_in_both_ports(shape):
    summary, diff, want = X8_SHAPES[shape]
    assert _claims(summary, diff)[1] == want
    assert _port_claims([(summary, diff)]) == [want]


# ─────────────────────────────── NOTE_path2_ninth_pass_2026_09_27: the licensed-difference rule, and the scorer's gaps

X9_R4 = ("diff --git a/.env b/.env\nnew file mode 100644\nindex 0000000..bd2c89c\n--- /dev/null\n+++ b/.env\n@@ -0,0 +1 @@\n+A=1\n"
         "diff --git a/db/q.sql b/db/q.sql\nindex e5593e5..8175040 100644\n--- a/db/q.sql\n+++ b/db/q.sql\n"
         "@@ -2 +2 @@ SELECT 1;\n--- users\n+++ users\n@@ -9 +9 @@ S8;\n-SELECT 3;\n+SELECT 4;\n"
         "diff --git a/env b/env\nnew file mode 100644\nindex 0000000..ac5d589\n--- /dev/null\n+++ b/env\n@@ -0,0 +1 @@\n+B=2\n")
X9_R5 = ("diff --git a/img/logo.png b/img/logo.png\nindex 1111111..2222222 100644\n"
         "Binary files a/img/logo.png and b/img/logo.png differ\n"
         "--- a/docs/notes.txt\t2024-05-06 07:08:09.000000000 +0000\n+++ b/docs/notes.txt\t2024-05-06 07:08:09.000000000 +0000\n"
         "@@ -1,2 +1,3 @@\n a\n+++ x\n b\n")
X9_R6 = "--- src/api.py\n+++ /dev/null\t1970-01-01 00:00:00.000000000 +0000\n@@ -1 +0,0 @@\n-x = 1\n"
X9_R7 = ("diff --git a/.eslintrc.json b/.eslintrc.json\nindex 0967ef4..11fd650 100644\n--- a/.eslintrc.json\n"
         "+++ b/.eslintrc.json\n@@ -1 +1 @@\n-{}\n+{\"root\": true}\n")
X9_R1 = (f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,5 @@\n import os\n+\n+\n+def  test_a():\n+    pass\n"
         "--- /dev/null\n+++ b/docs/guide.md\n@@ -0,0 +1,3 @@\n+```python\n+def test_example():\n+```\n")
X9_FS = f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,5 @@\n x = 0\n+def test_ok():\n+    pass\n+" + chr(0x1C) + "def test_y():\n+    pass\n"
X9_REFUSED_SYMBOL = ("--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1,5 @@\n x = 0\n+def foo():\n+    pass\n+def"
                     + chr(0x3000) + "bar():\n+    pass\n")
X9_MD = (f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,3 @@\n x = 0\n+def test_ok():\n+    pass\n"
         "--- a/docs/a.md\n+++ b/docs/a.md\n@@ -1 +1,2 @@\n x\n+" + chr(0x1C) + "def test_md():\n")
X9_COMPAT = ("--- a/.py\n+++ b/.py\n@@ -1,2 +1 @@\n-def api():\n-    pass\n+x = 1\n"
             "--- a/src/core.py\n+++ b/src/core.py\n@@ -1,2 +1 @@\n-def core():\n-    pass\n+x = 2\n")
X9_PLANTED = {
    # label: (good text in styxx/diffgate.py, planted text, summary, diff, the violation that must fire)
    # Z-1, inside the rule's own code
    "Z-1: `got` no longer asked against main's": (
        "    if got == py == js:", "    if True:", "Added 1 test. Added 2 tests.", X9_R1, "G-C7_oracle:tests_added_claim"),
    "Z-1: main's Python count read with the port's spelling": (
        "        self.tests = (sum(1 for line in self.added_py if _PY_TEST_LINE.match(line)),",
        "        self.tests = (sum(1 for line in self.added_py if _JS_TEST_LINE.match(line)),",
        "Added 0 tests. Added 1 test.", f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,3 @@\n x = 0\n+" + chr(0x1F) + "def test_n():\n+    pass\n",
        "G-C7_oracle:tests_added_claim"),
    "Z-1 accusing instead of abstaining": (
        '                            c.verdict, c.why = "UNCHECKABLE", f"{unlicensed}; claim says {n}"',
        '                            c.verdict, c.why = "CONTRADICTED", f"{unlicensed}; claim says {n}"',
        # (a unit separator: Python's `\s`, not JavaScript's, and not a line break, so F-2 cannot give main's claim back)
        "Added 1 test.", f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,3 @@\n x = 0\n+" + chr(0x1F) + "def test_load():\n+    pass\n",
        # NOTE_path2_eleventh_pass: the guard abstains where main verifies, so the reading before it is what G-C7 refuses
        "G-C7_oracle:tests_added_claim"),
    # Z-2
    "Z-2: main's port no longer asked": (
        "    if py == hit and js == hit:", "    if py == hit:",
        "Adds function foo.", "--- a/src/m.py\n+++ b/src/m.py\n@@ -1 +1,2 @@\n x = 0\n+" + chr(0x0B) + "def foo():\n",
        "G-C7_oracle:symbol_added_claim"),
    # Z-3
    "Z-3: a doubt main's reading held no longer counted beside a licensed difference": (
        "    if soft and set(status) != set(js):", "    if False:", "3 files changed.", X9_R4, "G-C7_oracle:Y_notes"),
    "Z-3: an unlicensed difference no longer read": (
        "    why = _licensed_against(status, full)", "    why = None",
        "Deleted src/api.py.", X9_R6, "G-C7_oracle:Y_notes"),
    "Z-3: a `diff --git` file its next pair replaced no longer a doubt": (
        "                soft.append(_Z3_REPLACED.format(_shown(pk)))", "                pass",
        # NOTE_path2_tenth_pass: R5 abstains by K-1 now (main read `+++ x` from its exact hunk), so the doubt is asked
        # beside #121's licensed dotted key instead, where it is the only thing that abstains
        "2 files changed.", "diff --git a/img/logo.png b/img/logo.png\nindex 1111111..2222222 100644\n"
        "Binary files a/img/logo.png and b/img/logo.png differ\n--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-A=1\n+A=2\n",
        "G-C7_oracle:Y_notes"),
    # Z-4
    "Z-4: the basename tier verifies again": (
        '    if p is None or p == c or p.endswith("/" + c):', "    if True:",
        "Updated packages/web/.eslintrc.json.", X9_R7, "G-C7_oracle:file_touched_claim"),
    # Z-5
    "Z-5: no definition line refused": (
        "    return _defined_name(line, removed=True) is None", "    return False",
        "Added 1 test.", X9_FS, "G-C7_oracle:tests_added_claim"),
    "Z-5: a symbol's own file not asked": (
        "        if path in refused and any(_defines(x, name) for x in added):",
        "        if path in refused and False:",
        "Added function foo.", X9_REFUSED_SYMBOL, "G-C7_oracle:symbol_added_claim"),
    "Z-5: a file that is not Python read as one": (
        "        if _undotted(path).lower().endswith(_PY_SUFFIXES):", "        if True:",
        "Added 1 test.", X9_MD, "G-C7_oracle:tests_added_claim"),
    # round-8 scorer lens: C-2's whole reading re-derived (P12, P32)
    "C-2: the removed side read on the dotted key": (
        "            if not _undotted(path).endswith(sufs):", "            if not path.endswith(sufs):",
        "Keeps backward compatibility.", X9_COMPAT, "G-C7_oracle:C-2_compat_surface"),
    "C-2: the reason prints the undotted key": (
        '    shown = ", ".join(f"{p}: {n}" for p, _l, n, _s in named[:_COMPAT_MAX_NAMED])',
        '    shown = ", ".join(f"{_undotted(p)}: {n}" for p, _l, n, _s in named[:_COMPAT_MAX_NAMED])',
        "Keeps backward compatibility.", "--- a/.lib/api.py\n+++ b/.lib/api.py\n@@ -1,2 +1 @@\n-def api():\n-    pass\n+x = 1\n",
        "G-C7_oracle:C-2_compat_surface"),
    # round-8 scorer lens: G-C1 reads the never-read sentences themselves, and the strict verdict (P22, P24b)
    "G-C1: the never-read sentences lower-cased": (
        "    uncovered_texts = [s.strip() for i, s in enumerate(sentences)",
        "    uncovered_texts = [s.strip().lower() for i, s in enumerate(sentences)",
        "Refactored The Parser. 1 file changed.", _m("src/a.py"), "G-C1_gate_fields_differ"),
    "G-C1: strict passes a contradicted gate": (
        # NOTE_path2_eleventh_pass: the verdict, and --strict, are the guard's, recomputed from the final claims
        '    return DiffGate(verdict="FAIL" if (contradicted or (strict and uncheckable)) else "PASS",',
        '    return DiffGate(verdict="FAIL" if ((contradicted and not strict) or (strict and uncheckable)) else "PASS",',
        "2 files changed.", _m("src/a.py"), "G-C1_strict_verdict_not_from_its_claims"),
    # round-8 scorer lens, blocker: Y-1's git-door reading, read on every record's paths whether or not it rebuilds
    "Y-1 at the git door: a case collision not noted": (
        '    return {"files": _Y1_COLLIDE} if any(len(v) > 1 for v in forms.values()) else {}', "    return {}",
        "2 files changed.", _m("docs/Guide.md") + _m("docs/guide.md"), "G-C7_oracle:Y-1_status_notes"),
    # where main raises, main gave no verdict, and a verdict the repair gives there is refused
    "Z-1/Z-2: main raising not asked": (
        '    if main.raises:\n        return "main raises on this diff (`+++ /dev/null` with no `---` line before it)"\n'
        "    if not all(main.python):",
        '    if False:\n        return "main raises on this diff (`+++ /dev/null` with no `---` line before it)"\n'
        "    if False:", "Added 1 test.",
        "diff --git a/t.py b/t.py\n x" + chr(0x0C) + "+++ /dev/null\n+def test_a():\n",
        # NOTE_path2_eleventh_pass: where main raises the guard abstains every decided claim, so it is the reading before
        # the guard that G-C7 refuses
        "G-C7_oracle:tests_added_claim"),
    # ... read on a case-folded variant of a record that holds no collision (one path upper-cased beside itself)
    "Y-1 at the git door: a case collision not noted, on a record with none": (
        '    return {"files": _Y1_COLLIDE} if any(len(v) > 1 for v in forms.values()) else {}', "    return {}",
        "1 file changed.", _m("src/a.py"), "G-C7_oracle:Y-1_status_notes"),
}


@pytest.mark.parametrize("label", sorted(X9_PLANTED))
def test_x9_every_ninth_pass_rule_and_round_8_scorer_finding_fails_on_a_planted_defect(scorer, monkeypatch, label):
    # Each record scores clean on the unplanted instrument, and the defect planted in a scratch copy is refused.
    pg = scorer
    good, bad, summary, diff, violation = X9_PLANTED[label]
    t = pg.Tally(name_prs=True)
    t.pair("clean", summary, diff, pg.raw_paths(diff))
    assert not t.violations, t.violating
    _planted(pg, monkeypatch, good, bad, "x9")
    t = pg.Tally(name_prs=True)
    t.pair("planted", summary, diff, pg.raw_paths(diff))
    assert violation in t.violations, dict(t.violations)


X9_DOOR = {
    # round-8 scorer lens, blockers: defects reachable only through gate_diff on a record the old door could not
    # rebuild -- a case collision (P: `_status_notes` returns {}), #121's dotted keys at the git door (P31), and a
    # rename entry keyed by its old path (P26) -- each now rebuilt by fast-import and refused
    "a case collision at the git door": (
        '    return {"files": _Y1_COLLIDE} if any(len(v) > 1 for v in forms.values()) else {}', "    return {}",
        "2 files changed.", _m("docs/Guide.md") + _m("docs/guide.md"), "G-C7_oracle:files_changed_count_claim"),
    "#121 reverted at the git door (P31)": (
        "            status[rp.key(path)] = st           # A / M / D / R",
        '            status[rp.key(path).lstrip(".")] = st           # A / M / D / R',
        "3 files changed. Only touches github/ and pr_agent.toml.",
        _m(".pr_agent.toml") + _m("pr_agent.toml") + _m(".github/x.yml"), "G-C7_oracle:files_changed_count_claim"),
    "a rename keyed by its old path (P26)": (
        "            st, path = parts[0][:1], parts[-1]", "            st, path = parts[0][:1], parts[1]",
        "Only touches lib/. Modified lib/new.py.",
        "--- a/src/old.py\n+++ /dev/null\n@@ -1,3 +0,0 @@\n-a = 1\n-b = 2\n-c = 3\n"
        "--- /dev/null\n+++ b/lib/new.py\n@@ -0,0 +1,3 @@\n+a = 1\n+b = 2\n+c = 3\n",
        "G-C7_oracle:file_touched_claim"),
}


@pytest.mark.parametrize("label", sorted(X9_DOOR))
def test_x9_the_git_door_rebuilds_dotted_case_twin_and_renamed_records_and_refuses_a_defect_there(scorer, monkeypatch,
                                                                                                label):
    from collections import Counter
    pg = scorer
    good, bad, summary, diff, violation = X9_DOOR[label]
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "clean", summary, diff, sample)
    assert sample["scored"] == 1 and not t.violations, (dict(sample), t.violating)
    _planted(pg, monkeypatch, good, bad, "x9door")
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "planted", summary, diff, sample)
    assert violation in t.violations, (dict(sample), dict(t.violations))


def test_x9_the_git_door_accepts_dot_led_segments_and_refuses_only_dot_dotdot_and_git(scorer):
    pg = scorer
    for p in (".github/x.yml", ".pr_agent.toml", "src/.env", "a/..b/c", "Docs/Guide.md"):
        assert pg._safe(p), p
    for p in (".", "..", "a/../b", "./a", ".git", ".GIT/config", "src/.git/x", "a b/c"):
        assert not pg._safe(p), p
    files = pg.rebuild(_m(".github/x.yml") + _m("docs/Guide.md") + _m("docs/guide.md"))
    assert files is not None and set(files[1]) == {".github/x.yml", "docs/Guide.md", "docs/guide.md"}


def test_x9_a_rename_pass_scores_records_that_delete_one_file_and_create_another(scorer):
    from collections import Counter
    pg = scorer
    diff = ("--- a/src/old.py\n+++ /dev/null\n@@ -1,3 +0,0 @@\n-a = 1\n-b = 2\n-c = 3\n"
            "--- /dev/null\n+++ b/lib/new.py\n@@ -0,0 +1,3 @@\n+a = 1\n+b = 2\n+c = 3\n")
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "renamed", "Only touches lib/. Modified lib/new.py. Deleted src/old.py.", diff, sample)
    assert sample["scored"] == 1 and sample["renames_scored"] == 1 and sample["renames_detected"] == 1
    assert not t.violations, t.violating


def test_x9_the_git_door_reads_main_s_name_status_split_on_its_own(scorer):
    # Z-3 at the git door: main split git's --name-status with str.splitlines(); where that cuts a path (core.quotePath
    # off, a U+2028 in it), main's list is not this one and the file-list claims abstain; the scorer holds the
    # instrument's reading of that to its own
    pg = scorer
    ns = "M\ta" + chr(0x2028) + "b.py\n"
    status = pg.own_name_status(ns)
    assert pg.new._status_differs(pg.new._main_name_status(ns), status) == pg.own_status_differs(ns, status) is not None
    assert pg.own_status_differs("M\tsrc/a.py\n", pg.own_name_status("M\tsrc/a.py\n")) is None


def test_x9_corpus_mode_sends_every_pr_with_a_file_list_claim_through_the_git_door(scorer, monkeypatch, tmp_path):
    # round-8 scorer lens (P27): a git-door-only defect on a file-list claim, on a PR with no definition claim and
    # outside the 1-in-25 sample, was admitted in corpus mode; such a PR is now always tried
    pg = scorer
    pid = next(i for i in range(1, 500) if pg.sample_key(i) % 25)
    patch = "@@ -1,2 +0,0 @@\n-a\n-b"
    shelf = _shelf(tmp_path, [(pid, "Deleted docs/old.md. 1 file changed.", [("docs/old.md", "removed", patch)])])
    out = tmp_path / "gates.json"
    pg.run_corpus(shelf, None, out)
    door = json.loads(out.read_text(encoding="utf-8"))["G-C8_git_door"]
    assert door["sample"]["scored"] == 1 and door["sample"]["tried_for_a_file_list_claim"] == 1, door["sample"]
    assert door["pass"], door["violations"]
    _planted(pg, monkeypatch, "            st, path = parts[0][:1], parts[-1]",
             '            st, path = parts[0][:1].replace("D", "M"), parts[-1]', "x9shelf")
    (tmp_path / "planted").mkdir()
    shelf2 = _shelf(tmp_path / "planted", [(pid, "Deleted docs/old.md. 1 file changed.", [("docs/old.md", "removed", patch)])])
    assert pg.run_corpus(shelf2, None, out) == 1
    door = json.loads(out.read_text(encoding="utf-8"))["G-C8_git_door"]
    assert not door["pass"] and "G-C7_oracle:file_deleted_claim" in door["violations"], door["violations"]


X9_SHAPES = {
    # the round-8 reproductions, both ports alike (R1 to R4 and R7 are git's bytes; R5 and R6 hand-written)
    "R1": ("Added 1 test. Added 2 tests.", X9_R1, ["UNCHECKABLE", "UNCHECKABLE"]),
    "R4": ("3 files changed. 4 files changed.", X9_R4, ["UNCHECKABLE", "UNCHECKABLE"]),
    "R5": ("2 files changed. 3 files changed.", X9_R5, ["UNCHECKABLE", "UNCHECKABLE"]),
    "R6": ("Modified lib/api.py. Deleted lib/api.py.", X9_R6, ["UNCHECKABLE", "UNCHECKABLE"]),
    "R7": ("Updated packages/web/.eslintrc.json. Updated .eslintrc.json.", X9_R7, ["UNCHECKABLE", "VERIFIED"]),
    "Z-5 tests": ("Added 1 test.", X9_FS, ["UNCHECKABLE"]),
    "Z-5 symbol": ("Added function foo.", X9_REFUSED_SYMBOL, ["UNCHECKABLE"]),
    "Z-5 markdown is not Python": ("Added 1 test.", X9_MD, ["VERIFIED"]),
    "every reading agrees": ("Added 1 test. Added function foo. 1 file changed.",
                             f"--- a/{TP}\n+++ b/{TP}\n@@ -1 +1,5 @@\n x = 0\n+def test_a():\n+    pass\n+def foo():\n+    pass\n",
                             ["VERIFIED", "VERIFIED", "VERIFIED"]),
}


@pytest.mark.parametrize("shape", sorted(X9_SHAPES))
def test_x9_each_shape_reads_alike_in_both_ports(shape):
    summary, diff, verdicts = X9_SHAPES[shape]
    got = _claims(summary, diff)[1]
    assert [v for _k, v, _w in got] == verdicts, got
    assert _port_claims([(summary, diff)]) == [got]


def test_x9_main_s_two_spellings_are_the_ones_each_runtime_reads():
    # the code point lists both ports carry are Python's `\s` (str.isspace), str.splitlines()'s breaks, and
    # JavaScript's `\s` (checked against node in the port differential); held here against this Python
    assert set(dg._PY_SPACE) == {chr(c) for c in range(0x110000) if chr(c).isspace()}
    assert all(len(("a" + ch + "b").splitlines()) == 2 for ch in "\n\r" + "".join(map(chr, (11, 12, 28, 29, 30, 0x85, 0x2028, 0x2029))))
    for text in ("a\r\nb\rc\nd" + chr(11) + "e" + chr(0x2028) + "f\n", "", "x\n\n", chr(0x85)):
        assert dg._py_lines(text) == text.splitlines(), ascii(text)
    assert dg._main_key("./.github/X.yml") == "github/x.yml" and dg._main_key("..env") == "env"


def test_x9_the_licensed_difference_leaves_every_repair_s_reproduction_repaired():
    # bar (1): #97, #121 and #101 still read their reproductions as repaired
    _, got = _claims("Created integrations/git/README.md.", TWO_READMES)
    assert got == [("file_created", "VERIFIED", "diff status 'A' for 'integrations/git/readme.md'")]
    _, got = _claims("2 files changed. Created .pr_agent.toml. Deleted pr_agent.toml.",
                     "diff --git a/pr_agent.toml b/pr_agent.toml\ndeleted file mode 100644\n--- a/pr_agent.toml\n+++ /dev/null\n"
                     "@@ -1 +0,0 @@\n-x = 1\ndiff --git a/.pr_agent.toml b/.pr_agent.toml\nnew file mode 100644\n--- /dev/null\n"
                     "+++ b/.pr_agent.toml\n@@ -0,0 +1 @@\n+[pr_reviewer]\n")
    assert [v for _k, v, _w in got] == ["VERIFIED", "VERIFIED", "VERIFIED"]
    assert _claims("Adds function backoff with jitter. Added 2 tests.", CHANGED_DEFS)[1] == ISSUE_101


X9_RAISES = "diff --git a/t.py b/t.py\n x" + chr(0x0C) + "+++ /dev/null\n+def test_a():\n"
X9_STRAY = "+" + chr(0x1C) + "def test_b():\n--- a/t.py\n+++ b/t.py\n@@ -1 +1,2 @@\n x\n+def test_a():\n"


def test_x9_where_main_raises_on_one_spelling_the_claims_abstain():
    # main's Python split cuts a `+++ /dev/null` out of a context line and raises on it (no `---` before it); main's
    # port and this reading do not, and every claim that reads main's reading abstains, in both ports
    _, got = _claims("Added 1 test. 1 file changed. Added function test_a.", X9_RAISES)
    raises = "main raises on this diff (`+++ /dev/null` with no `---` line before it)"
    assert got == [("tests_added", "UNCHECKABLE", f"{raises}; claim says 1"),
                   ("files_changed_count", "UNCHECKABLE", "the diff's file list is not certain: this reading's file list "
                                                          "differs from main's: main raises on it (`+++ /dev/null` with "
                                                          "no `---` line before it); claim says 1"),
                   ("symbol_added", "UNCHECKABLE", raises)]
    assert _port_claims([("Added 1 test. 1 file changed. Added function test_a.", X9_RAISES)]) == [got]


def test_x9_a_refused_line_outside_any_file_abstains_the_count_and_not_a_symbol_elsewhere():
    _, got = _claims("Added 1 test. Added function test_a.", X9_STRAY)
    assert got == [("tests_added", "UNCHECKABLE", f"an added definition line outside any file {Z5_WHY}; claim says 1"),
                   ("symbol_added", "VERIFIED", "added lines do define function 'test_a'")]
    assert _port_claims([("Added 1 test. Added function test_a.", X9_STRAY)]) == [got]


def test_x9_a_pair_naming_the_header_s_old_path_replaces_nothing():
    # a deletion under a `diff --git` header whose two paths differ: the `---`/`+++` pair names the old path, which
    # is the header's file, so no file is dropped and the count stands beside a dotfile's licensed key
    diff = ("diff --git a/src/old.py b/src/new.py\ndeleted file mode 100644\n--- a/src/old.py\n+++ /dev/null\n"
            "@@ -1 +0,0 @@\n-x = 1\n--- /dev/null\n+++ b/.env\n@@ -0,0 +1 @@\n+A=1\n")
    assert _claims("2 files changed.", diff)[1] == [("files_changed_count", "VERIFIED", "diff changes 2 files, claim says 2")]
    assert _port_claims([("2 files changed.", diff)]) == [_claims("2 files changed.", diff)[1]]


def test_x9_the_git_door_s_z3_reading_is_held_to_the_scorer_s(scorer, monkeypatch):
    from collections import Counter
    pg = scorer
    good = '    return f"{_Z3_PREFIX} where no repair accounts for it: {why}" if why else None'
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "clean", "1 file changed.", _m("src/a.py"), sample)
    assert sample["scored"] == 1 and not t.violations, t.violating
    _planted(pg, monkeypatch, good, '    return f"{_Z3_PREFIX} where no repair accounts for it: planted"', "x9z3")
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "planted", "1 file changed.", _m("src/a.py"), sample)
    assert "G-C7_oracle:Z-3_status_differs" in t.violations, dict(t.violations)


# ─────────────────────────────── NOTE_path2_tenth_pass_2026_09_27: K-1 to K-3, and the round-9 scorer findings

K1_P2 = ("Index: assets/logo.png\n" + "=" * 67 + "\nCannot display: file marked as a binary type.\n"
         "svn:mime-type = application/octet-stream\n"
         "Index: db/q.sql\n" + "=" * 67 + "\n--- db/q.sql\t(revision 1)\n+++ db/q.sql\t(working copy)\n"
         "@@ -1,2 +1,2 @@\n SELECT 1;\n--- users\n+++ users\n")
K1_P3 = ("diff -r 1a2b3c4d5e6f -r 6f5e4d3c2b1a assets/logo.png\nBinary file assets/logo.png has changed\n"
         "diff -r 1a2b3c4d5e6f -r 6f5e4d3c2b1a db/q.sql\n--- a/db/q.sql\tMon May 06 07:08:09 2024 +0000\n"
         "+++ b/db/q.sql\tMon May 06 07:08:09 2024 +0000\n@@ -1,2 +1,2 @@\n SELECT 1;\n--- users\n+++ users\n")
K1_SUMMARY = "1 file changed. 2 files changed. Only touches db/. Only touches db/ and assets/."
K1_USERS = "main reads 'users' ('M') from them"


def _submodule_repo(tmp_path: Path) -> str:
    """Round 9's P1: a bare repository written by fast-import, `diff.submodule=log`, db/q.sql's `-- users` changed to
    `++ users` beside a submodule bump git prints as one `Submodule ...` line. Truth (--name-status): two files."""
    def git(*a, inp=None):
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, input=inp).stdout
    git("init", "-q", "--bare")
    git("config", "diff.submodule", "log")
    s = bytearray()
    for mark, data in ((1, b"SELECT 1;\n-- users\n"), (2, b"SELECT 1;\n++ users\n")):
        s += b"blob\nmark :%d\ndata %d\n" % (mark, len(data)) + data + b"\n"
    s += (b"commit refs/heads/base\nmark :10\ncommitter r <r@r> 1700000000 +0000\ndata 0\n"
          b"M 100644 :1 db/q.sql\nM 160000 1234567890123456789012345678901234567890 vendor/lib\n\n"
          b"commit refs/heads/tip\nmark :11\ncommitter r <r@r> 1700000000 +0000\ndata 0\nfrom :10\n"
          b"M 100644 :2 db/q.sql\nM 160000 89abcdef89abcdef89abcdef89abcdef89abcdef vendor/lib\n\n")
    git("fast-import", "--quiet", inp=bytes(s))
    return git("diff", "base..tip").decode("utf-8")


def test_x10_k1_a_file_neither_reading_counts_beside_a_phantom_main_read_from_content(tmp_path):
    # Round 9, regressions lens, blocker 1: W-1 removed the file main read from an exact hunk's `+++ users`, and that
    # file had balanced a changed file neither reading counts (a submodule under diff.submodule=log). The git door
    # reads git's --name-status and keeps its verdicts; the raw door and the port abstain on the file list.
    diff = _submodule_repo(tmp_path)
    assert "\n--- users\n+++ users\n" in diff and "Submodule vendor/lib" in diff
    via_git = [(c.kind, c.verdict) for c in gate_diff(K1_SUMMARY, tmp_path, "base", "tip").claims]
    assert via_git == [("files_changed_count", "CONTRADICTED"), ("files_changed_count", "VERIFIED"),
                       ("only_touches", "CONTRADICTED"), ("only_touches", "CONTRADICTED")]
    want = _k1([("files_changed_count", "-", "claim says 1"), ("files_changed_count", "-", "claim says 2"),
                ("only_touches", "-", ""), ("only_touches", "-", "")], K1_USERS)
    assert _claims(K1_SUMMARY, diff)[1] == want and _port_claims([(K1_SUMMARY, diff)]) == [want]
    # svn's `Cannot display` block and hg's `Binary file ... has changed` beside the same hunk, hand-written
    for text in (K1_P2, K1_P3):
        assert _claims(K1_SUMMARY, text)[1] == want and _port_claims([(K1_SUMMARY, text)]) == [want]


def test_x10_k1_leaves_the_reading_of_the_lines_as_it_was():
    # K-1 abstains the file-list claims; the exact hunk's lines are still content (W-1), for every other claim
    status, blob = parse_unified_diff(K1_P3)
    assert "users" not in status and blob == "++ users"
    # with no phantom main read, nothing is abstained: a `-- users` removed alone sets no file
    alone = "--- a/db/q.sql\n+++ b/db/q.sql\n@@ -1,2 +1 @@\n SELECT 1;\n--- users\n"
    assert _claims("1 file changed.", alone)[1] == [("files_changed_count", "VERIFIED", "diff changes 1 files, claim says 1")]
    assert _port_claims([("1 file changed.", alone)]) == [_claims("1 file changed.", alone)[1]]


# K-2: case pairs whose lowercase mapping differs across the Unicode versions the supported runtimes read
# (Python 3.9-3.14: 13.0 to 16.0; Node 24: 16.0): 14.0 assigned the leading four, 16.0 the next four; the last is the
# final sigma, which every runtime lower-cases by its context.
K2_PAIRS = {
    "U+2C2F/U+2C5F (14.0)": (0x2C2F, 0x2C5F), "U+A7C0/U+A7C1 (14.0)": (0xA7C0, 0xA7C1),
    "U+A7D0/U+A7D1 (14.0)": (0xA7D0, 0xA7D1), "U+10570/U+10597 (14.0)": (0x10570, 0x10597),
    "U+1C89/U+1C8A (16.0)": (0x1C89, 0x1C8A), "U+A7CB/U+0264 (16.0)": (0xA7CB, 0x0264),
    "U+10D50/U+10D70 (16.0)": (0x10D50, 0x10D70), "U+A7DC/U+019B (16.0)": (0xA7DC, 0x019B),
    "U+03C3/U+03C2 (final sigma)": (0x3C3, 0x3C2),
}
K2_SUMMARY = "Modified docs/a.md. Only touches docs/. Only touches src/. 3 files changed."
Y1_COLLIDE_WHY = "the diff's file list is not certain: two header paths that differ only in case are one key"
FOLD_PY = ROOT / "styxx" / "_fold.py"
FOLD_SHA256 = "a52cda82375292f71230e7e781acc24760084994812e083870b7419bdc5d5fd6"


def _k2_diff(a: int, b: int) -> str:
    return "".join(_m(p) for p in (f"docs/x{chr(a)}.md", f"docs/x{chr(b)}.md", "docs/a.md"))


@pytest.mark.parametrize("label", sorted(K2_PAIRS))
def test_x10_k2_two_paths_the_runtimes_fold_differently_abstain_alike_on_every_python_and_the_port(label):
    # Round 9, regressions lens, blocker 2: Y-1 compared two header paths through the runtime's lower-casing, so a
    # diff naming both halves of a pair Unicode 16.0 assigned was a collision in the port and none in Python 3.12, and
    # every file-list claim -- the ones about docs/a.md too -- abstained in one port only. Both ports read one fold.
    a, b = K2_PAIRS[label]
    from styxx import _fold
    assert _fold.fold(chr(a)) == _fold.fold(chr(b))
    diff = _k2_diff(a, b)
    want = [("file_touched", "UNCHECKABLE", Y1_COLLIDE_WHY), ("only_touches", "UNCHECKABLE", Y1_COLLIDE_WHY),
            ("only_touches", "UNCHECKABLE", Y1_COLLIDE_WHY),
            ("files_changed_count", "UNCHECKABLE", f"{Y1_COLLIDE_WHY}; claim says 3")]
    assert _claims(K2_SUMMARY, diff)[1] == want
    assert _port_claims([(K2_SUMMARY, diff)]) == [want]
    # the git door's reading of the same two paths (Y-1 on git's --name-status)
    assert dg._status_notes([f"docs/x{chr(a)}.md", f"docs/x{chr(b)}.md"]) == {"files": dg._Y1_COLLIDE}


def test_x10_k2_paths_that_fold_apart_read_as_before():
    diff = "".join(_m(p) for p in ("docs/x" + chr(0x10D50) + ".md", "docs/y" + chr(0x10D70) + ".md", "docs/a.md"))
    got = _claims(K2_SUMMARY, diff)[1]
    assert [v for _k, v, _w in got] == ["VERIFIED", "VERIFIED", "CONTRADICTED", "VERIFIED"]
    # (the reasons print each port's key, which is the runtime's lower case, as main's is: U+10D50 is its own key on a
    # Python before 3.14 and U+10D70 in the port -- the fifth pass's disclosed 27 code points, main's reading too)
    assert [[v for _k, v, _w in c] for c in _port_claims([(K2_SUMMARY, diff)])] == [[v for _k, v, _w in got]]


def _gen_fold():
    spec = importlib.util.spec_from_file_location("gen_fold_under_test", ROOT / "web" / "gate" / "gen_fold.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    return g


def test_x10_k2_both_ports_carry_one_fold_with_its_version_and_hash_pinned():
    from styxx import _fold
    g = _gen_fold()
    js_block = g.block_of(XID_JS.read_text(encoding="utf-8"))
    assert g.table_in(js_block) == g.table_in(g.block_of(FOLD_PY.read_text(encoding="utf-8"))) == _fold.FOLD
    assert hashlib.sha256(_fold.FOLD.encode("ascii")).hexdigest() == _fold.FOLD_SHA256 == FOLD_SHA256
    assert _fold.UNICODE_VERSION == g.PINNED == "16.0.0"
    assert f'const _FOLD_UNICODE_VERSION = "{_fold.UNICODE_VERSION}";' in js_block
    assert f'const _FOLD_SHA256 = "{_fold.FOLD_SHA256}";' in js_block
    assert g.decode(_fold.FOLD) == _fold.MAP and len(_fold.MAP) == 1461
    assert dg._case_fold is _fold.fold


@pytest.mark.skipif(unicodedata.unidata_version != "16.0.0",
                    reason="gen_fold.py regenerates only under the fold's Unicode version, 16.0.0 (Python 3.14)")
def test_x10_k2_the_generator_reproduces_both_blocks_from_the_pinned_version():
    g = _gen_fold()
    t = g.encode(g.mapping())
    assert g.sha(t) == FOLD_SHA256
    for path, block in ((FOLD_PY, g.python_block(t)), (XID_JS, g.js_block(t))):
        before, after = g.splice(path, block)
        assert before == after, path


def test_x10_k2_the_fold_sees_every_merge_this_python_s_key_makes():
    # sound against this interpreter's str.lower(): a pair of paths it keys alike folds alike
    from styxx import _fold
    assert _gen_fold().unsound(_fold.MAP, str.lower) == []
    sig = chr(0x3A3)
    assert _fold.fold(sig) == _fold.fold(chr(0x3C3)) == _fold.fold(chr(0x3C2)) == _fold.fold(("A" + sig).lower()[1:])


def test_x10_k2_the_fold_sees_every_merge_the_port_s_key_makes():
    # and against node's toLowerCase(), every code point, through the port's own decoder
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH")
    script = ("const src=require('fs').readFileSync(process.argv[1],'utf8');"
              "const f=new Function(src+';return [_caseFold,_FOLD];')();const fold=f[0];const bad=[];"
              "for(let c=0;c<0x110000;c++){if(c>=0xd800&&c<=0xdfff)continue;const ch=String.fromCodePoint(c);"
              "if(fold(ch.toLowerCase())!==fold(ch)||fold(fold(ch))!==fold(ch))bad.push(c);}"
              "for(const s of ['A\\u03a3','A\\u03a3A','\\u03a3']){if(fold(s.toLowerCase())!==fold(s))bad.push(s);}"
              "process.stdout.write(JSON.stringify({bad,size:f[1].size,"
              "map:[...f[1]].map(([k,v])=>[k,[...v].map(x=>x.codePointAt(0))])}));")
    r = subprocess.run([node, "-e", script, str(XID_JS)], capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout)
    from styxx import _fold
    assert got["bad"] == [] and got["size"] == len(_fold.MAP)
    assert {k: "".join(map(chr, v)) for k, v in got["map"]} == _fold.MAP


# K-3: this reading raised where main did not
K3_SHAPES = {
    "a deletion after an exact hunk main read a `---` line from": (
        "2 files changed.", "diff --git a/x.sql b/x.sql\nindex 1..2 100644\n@@ -1 +1 @@\n--- q\n+++ z\n"
                            "diff --git a/y b/y\n+++ /dev/null\n",
        [("files_changed_count", "UNCHECKABLE", K1_WHY.format("without those lines main raises (`+++ /dev/null` "
                                                              "with no `---` line before it)") + "; claim says 2")]),
    "a GNU deletion with no `---` line": (
        "1 file changed.", "+++ /dev/null\t2024-01-01 00:00:00.000000000 +0000\n",
        [("files_changed_count", "UNCHECKABLE", "the diff carries no file statuses and no added lines; 50 characters "
                                                 "of input parsed to nothing, which is a parse failure, not an empty "
                                                 "change")]),
}


@pytest.mark.parametrize("shape", sorted(K3_SHAPES))
def test_x10_k3_no_raise_where_main_reads_the_diff(shape):
    summary, diff, want = K3_SHAPES[shape]
    assert _claims(summary, diff)[1] == want
    assert _port_claims([(summary, diff)]) == [want]


# K-4: beside #121's licensed dotted key, a line no reading places may name a changed file neither reading counts. The
# builder's own differential found it (fresh seeds of round 9's generator): `.env` created beside `env` and a submodule
# bump, main counting two where the truth is three -- right by merging the twins -- and this reading three... of four.
K4_TWINS = ("diff --git a/.env b/.env\nnew file mode 100644\nindex 0000000..e69de29\n--- /dev/null\n+++ b/.env\n"
            "@@ -0,0 +1 @@\n+A=1\ndiff --git a/env b/env\nindex 1111111..2222222 100644\n--- a/env\n+++ b/env\n"
            "@@ -1 +1 @@\n-old\n+new\n")
K4_HG = "diff -r 1a2b3c4d5e6f -r 6f5e4d3c2b1a assets/core\nBinary file assets/core has changed\n"
K4_BINARY = K4_TWINS + ("diff --git a/img.png b/img.png\nindex 1111111..2222222 100644\nGIT binary patch\nliteral 5\n"
                        "McmZQzWMXCj0000\n\nliteral 5\nMcmZQzWMXCj0000\n\n")
K4_WHY = ("the diff's file list is not certain: this reading's file list differs from main's by a repair (#121 keeps a "
          "dotfile's dot), and main's reading also passed over a line no reading places, which may name a changed file "
          "neither reading counts (git's `Submodule` line, svn's and hg's binary notices, and the like); the repair may "
          "have balanced that error")


def _submodule_twins_repo(tmp_path: Path) -> str:
    def git(*a, inp=None):
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, input=inp).stdout
    git("init", "-q", "--bare")
    git("config", "diff.submodule", "log")
    s = bytearray()
    for mark, data in ((1, b"old\n"), (2, b"new\n"), (3, b"A=1\n")):
        s += b"blob\nmark :%d\ndata %d\n" % (mark, len(data)) + data + b"\n"
    s += (b"commit refs/heads/base\nmark :10\ncommitter r <r@r> 1700000000 +0000\ndata 0\n"
          b"M 100644 :1 env\nM 160000 1234567890123456789012345678901234567890 vendor/lib\n\n"
          b"commit refs/heads/tip\nmark :11\ncommitter r <r@r> 1700000000 +0000\ndata 0\nfrom :10\n"
          b"M 100644 :2 env\nM 100644 :3 .env\nM 160000 89abcdef89abcdef89abcdef89abcdef89abcdef vendor/lib\n\n")
    git("fast-import", "--quiet", inp=bytes(s))
    return git("diff", "base..tip").decode("utf-8")


def test_x10_k4_a_dotted_twin_beside_a_file_neither_reading_counts(tmp_path):
    diff = _submodule_twins_repo(tmp_path)
    summary = "3 files changed. 2 files changed."
    # NOTE_path2_twelfth_pass (A.1): git's `Submodule` line under diff.submodule=log names a changed file outside any
    # `diff --git` header, so git's diff text is not git's own rendering and #121 licenses nothing at the git door either
    # (a recall cost the note takes knowingly: git's --name-status was right here)
    assert [(c.kind, c.verdict, c.why) for c in gate_diff(summary, tmp_path, "base", "tip").claims] == [
        ("files_changed_count", "UNCHECKABLE", _unlicensed("CONTRADICTED", "VERIFIED")),
        ("files_changed_count", "UNCHECKABLE", _unlicensed("VERIFIED", "CONTRADICTED"))]
    want = [("files_changed_count", "UNCHECKABLE", f"{K4_WHY}; claim says 3"),
            ("files_changed_count", "UNCHECKABLE", f"{K4_WHY}; claim says 2")]
    for text in (diff, K4_TWINS + K4_HG):
        assert _claims(summary, text)[1] == want and _port_claims([(summary, text)]) == [want]


def test_x10_k4_git_s_own_lines_are_placed_and_the_twins_still_count():
    # #121's reproduction shape, alone and beside a git binary patch (both of its blocks): nothing unplaced, it counts
    for text in (K4_TWINS, K4_BINARY):
        got = _claims("3 files changed.", text)[1] if text is K4_BINARY else _claims("2 files changed.", text)[1]
        assert [v for _k, v, _w in got] == ["VERIFIED"], got
        assert _port_claims([("3 files changed." if text is K4_BINARY else "2 files changed.", text)]) == [got]
    assert dg._diff_notes(K4_BINARY) == {} and dg._diff_notes(K4_TWINS) == {}


# K-5: Z-4 is asked only of a path both ports' templates read alike. The path template's `\w` is Python's (Unicode) in
# the Python and ASCII in the port, so `Docs/<U+A7D0>/a.md` is that path in the Python and `/a.md` in the port; Z-4
# abstained in the Python only, where main read both alike (the builder's own differential, the case-pair set). Where
# the ports may read the path apart, the claim now reads as main read it, each port as main's same port did -- not by
# this reading's tiers, which #121's dotted keys would move (a dotfile in a non-ASCII directory, round 8's R7 again).
K5_ESLINT = "--- a/.eslintrc.json\n+++ b/.eslintrc.json\n@@ -1 +1 @@\n-{}\n+{\"a\": 1}\n"
K5_SHAPES = {
    "a path the port starts after a non-ASCII letter": ("Edited Docs/" + chr(0xA7D0) + "/a.md.", _m("docs/a.md"),
                                                         [("file_touched", "VERIFIED", "diff status 'M' for 'docs/a.md'")]),
    "a path holding a sigma": ("Added LIB/" + chr(0x3A3) + "/X.MD.", "--- /dev/null\n+++ b/lib/" + chr(0x3C2) + "/x.md\n"
                               "@@ -0,0 +1 @@\n+new\n",
                               [("file_touched", "VERIFIED", "diff status 'A' for 'lib/" + chr(0x3C2) + "/x.md'")]),
    "a dotfile in a non-ASCII directory reads as main read it, not by #121's key": (
        "Updated packages/w" + chr(0xE9) + "b/.eslintrc.json.", K5_ESLINT,
        [("file_touched", "UNCHECKABLE", "'packages/w" + chr(0xE9) + "b/.eslintrc.json' does not appear in the diff "
                                         "— accusation WITHHELD: this class failed EXTERNAL-1 precision (0.23 vs "
                                         "0.95 floor), disabled pending repair")]),
    "an ASCII path in another directory still abstains": (
        "Edited lib/a.md.", _m("docs/a.md"),
        [("file_touched", "UNCHECKABLE", "'lib/a.md': only a file with the same name in another directory is in the diff "
                                         f"('docs/a.md', status 'M'); {Z4_TAIL}")]),
}


@pytest.mark.parametrize("shape", sorted(K5_SHAPES))
def test_x10_k5_z4_reads_only_a_path_both_ports_read_alike(shape):
    summary, diff, want = K5_SHAPES[shape]
    assert _claims(summary, diff)[1] == want
    port = _port_claims([(summary, diff)])[0]
    if "does not appear in the diff" in want[0][2]:
        # the reason names the path each port's template extracted, as main's two ports' reasons do; the verdicts agree
        assert [(k, v) for k, v, _w in port] == [(k, v) for k, v, _w in want]
        assert port[0][2].startswith("'b/.eslintrc.json' does not appear")      # the port's own extraction, main's too
    else:
        assert port == want


def test_x10_k3_where_main_raises_every_claim_still_abstains_and_nothing_raises():
    diff = "+++ /dev/null\n+def test_a():\n"
    got = _claims("Added 1 test. 1 file changed. Added function test_a.", diff)[1]
    assert [v for _k, v, _w in got] == ["UNCHECKABLE"] * 3
    assert _port_claims([("Added 1 test. 1 file changed. Added function test_a.", diff)]) == [got]


# ── the round-9 scorer findings, and each tenth-pass rule, against a planted defect

def _planted_many(pg, monkeypatch, pairs: list, tag: str) -> None:
    src = Path(pg.new.__file__).read_bytes().decode("utf-8")
    for good, bad in pairs:
        assert src.count(good) == 1, good
        src = src.replace(good, bad)
    for name in ("new", "CF"):
        monkeypatch.setattr(pg, name, pg._module_from(src.encode("utf-8"), f"styxx_diffgate_{tag}_{name}", f"<{tag} {name}>"))


X10_DOT_API = "--- a/.hidden.py\n+++ b/.hidden.py\n@@ -1,2 +1 @@\n-def api():\n-    pass\n+x = 1\n"
X10_SIGMA = _m("docs/a" + chr(0x3C3) + ".md") + _m("docs/a" + chr(0x3C2) + ".md")
X10_DROP_LANGS = ('              "compat2_candidate": any(d[3] for d in dropped)}\n',
                  '              "compat2_candidate": any(d[3] for d in dropped)}\n'
                  '    if any(p[:1] == "." for p, *_x in dropped):\n        del detail["languages"]\n')
X10_P32 = ('    shown = ", ".join(f"{p}: {n}" for p, _l, n, _s in named[:_COMPAT_MAX_NAMED])',
           '    shown = ", ".join(f"{_undotted(p)}: {n}" for p, _l, n, _s in named[:_COMPAT_MAX_NAMED])')
X10_PLANTED = {
    # label: ([(good, bad), ...] in styxx/diffgate.py, summary, diff, the violation that must fire)
    "K-1: W-1's removal of a file main read licensed again": (
        [("    if skipped != full:", "    if False:")], K1_SUMMARY, K1_P2, "G-C7_oracle:Y_notes"),
    "K-2: the runtime's lower-casing again": (
        [("        if forms.setdefault(_case_fold(form), form) != form:",
          "        if forms.setdefault(form.lower(), form) != form:")], "2 files changed.", X10_SIGMA,
        "G-C7_oracle:Y_notes"),
    "K-2 at the git door: the runtime's lower-casing again": (
        [("        forms.setdefault(_case_fold(form), set()).add(form)",
          "        forms.setdefault(form.lower(), set()).add(form)")], "2 files changed.", X10_SIGMA,
        "G-C7_oracle:Y-1_status_notes"),
    # round-9 scorer lens, blocker: compat_violations skipped a claim whose detail lacked "languages"
    "C-2 drops the languages on a leading-dot path": (
        [X10_DROP_LANGS], "Keeps backward compatibility.", X10_DOT_API, "G-C7_oracle:C-2_compat_surface"),
    "C-2 drops the languages and prints the undotted key (P32)": (
        [X10_DROP_LANGS, X10_P32], "Keeps backward compatibility.", X10_DOT_API, "G-C7_oracle:C-2_compat_surface"),
    # round-9 scorer lens, blocker: where the baseline raises, the gate and strict verdicts were not scored
    "where main raises, strict passes an unverifiable gate": (
        # NOTE_path2_eleventh_pass: the guard's strict verdict (`ref is None` where main raises)
        [('    uncheckable = any(c.verdict == "UNCHECKABLE" for c in final)',
          '    uncheckable = any(c.verdict == "UNCHECKABLE" for c in final) and ref is not None')],
        "Added 1 test.", X9_RAISES, "G-C1_strict_verdict_not_from_its_claims"),
    "where main raises, the gate fails with no contradicted claim": (
        [('    return DiffGate(verdict="FAIL" if (contradicted or (strict and uncheckable)) else "PASS",',
          '    return DiffGate(verdict="FAIL" if (contradicted or (strict and uncheckable) or (ref is None and final)) '
          'else "PASS",')],
        "Added 1 test.", X9_RAISES, "G-C1_gate_verdict_not_from_its_claims"),
    "K-4: a line no reading places no longer a doubt": (
        [("                soft.append(_Z3_UNPLACED)", "                pass")], "3 files changed. 2 files changed.",
        K4_TWINS + K4_HG, "G-C7_oracle:Y_notes"),
    "K-4: a git binary patch's second block read as a line no reading places": (
        [('            elif line == "GIT binary patch" or _BINARY_PATCH.match(line):',
          '            elif line == "GIT binary patch":')], "3 files changed.", K4_BINARY, "G-C7_oracle:Y_notes"),
    "K-5: a sentence the ports read apart not read as main read it": (
        # NOTE_path2_eleventh_pass: K-5 is the guard's, at the sentence; ignored, Z-4 abstains where main verifies
        [('        if apart and c.kind != "tests_pass":             # K-5: the sentence reads as main\'s same port read it',
          "        if False:")],
        "Edited Docs/" + chr(0xA7D0) + "/a.md.", _m("docs/a.md"), "G-C9_guard:file_touched"),
    "where main raises, a claim's detail altered": (
        [("                c = DiffClaim(kind=kind, text=sent.strip()[:160], detail=d)",
          "                c = DiffClaim(kind=kind, text=sent.strip()[:160], detail=({**d, \"x\": 1} if (main is not None "
          "and main.raises) else d))")],
        "Added 1 test.", X9_RAISES, "G-C1_claims_differ"),
    "where main raises, a never-read sentence dropped": (
        [("    uncovered_texts = [s.strip() for i, s in enumerate(sentences)\n",
          "    uncovered_texts = [s.strip() for i, s in enumerate(sentences) if not (main is not None and main.raises)\n")],
        "Refactored the parser. Added 1 test.", X9_RAISES, "G-C1_gate_fields_differ"),
}


@pytest.mark.parametrize("label", sorted(X10_PLANTED))
def test_x10_every_tenth_pass_rule_and_round_9_scorer_finding_fails_on_a_planted_defect(scorer, monkeypatch, label):
    pg = scorer
    pairs, summary, diff, violation = X10_PLANTED[label]
    t = pg.Tally(name_prs=True)
    t.pair("clean", summary, diff, pg.raw_paths(diff))
    assert not t.violations, t.violating
    _planted_many(pg, monkeypatch, pairs, "x10")
    t = pg.Tally(name_prs=True)
    t.pair("planted", summary, diff, pg.raw_paths(diff))
    assert violation in t.violations, dict(t.violations)


X10_DOOR = {
    # round-9 scorer lens, minor: base and head, and the report users read, were never compared
    "base and head swapped": (
        "                 repo=repo, base=base, head=head, main=_MainReading(diff_text, (main_map,)),",
        "                 repo=repo, base=head, head=base, main=_MainReading(diff_text, (main_map,)),",
        "1 file changed.", _m("src/a.py"), "G-C1_base_head_differ"),
    "the report names the head as its base": (
        '        return {"diffgate": "v0", "verdict": self.verdict, "base": self.base,',
        '        return {"diffgate": "v0", "verdict": self.verdict, "base": self.head,',
        "1 file changed.", _m("src/a.py"), "G-C1_report_differs_from_its_gate"),
}


@pytest.mark.parametrize("label", sorted(X10_DOOR))
def test_x10_the_git_door_scores_base_head_and_the_report(scorer, monkeypatch, label):
    from collections import Counter
    pg = scorer
    good, bad, summary, diff, violation = X10_DOOR[label]
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "clean", summary, diff, sample)
    assert sample["scored"] == 1 and not t.violations, (dict(sample), t.violating)
    _planted(pg, monkeypatch, good, bad, "x10door")
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "planted", summary, diff, sample)
    assert violation in t.violations, (dict(sample), dict(t.violations))


@pytest.mark.parametrize("good,bad", [
    ('("differs", _status_differs(main_map, status))', '("differs", None)'),
    ("    for line in _py_lines(name_status):", "    for line in _diff_lines(name_status):"),
])
def test_x10_the_door_canaries_make_z3_fire_at_the_git_door_and_refuse_a_defect_there(scorer, monkeypatch, good, bad):
    # round-9 scorer lens, major: no rebuildable record made Z-3 abstain at the git door, so a defect that stopped it
    # was admitted in both modes. Each run now scores canaries holding a U+0085 or U+2028 path (core.quotePath off),
    # where main's str.splitlines() cut git's --name-status line and Z-3 abstains. NOTE_path2_eleventh_pass (round-10
    # scorer lens, blocker): and U+2029, a third.
    pg = scorer
    report, _ = pg.score_canaries()
    assert report["pass"] and report["door"]["scored"] == len(pg.DOOR_CANARIES) == 7, report
    z3 = [c for c in pg.DOOR_CANARIES if c[0].startswith("canary:z3-git-door-")]      # (the twelfth pass adds four more)
    assert len(z3) == 3
    for _cid, summary, diff in z3:
        files = pg.rebuild(diff)
        with pg.GitDoor(*files) as door:
            g = door.gate(pg.new, summary)
            assert all(c.verdict == "UNCHECKABLE" and "differs from main's" in c.why for c in g.claims), g.claims
    _planted(pg, monkeypatch, good, bad, "x10canary")
    report, violating = pg.score_canaries()
    assert not report["pass"] and violating, report


def test_x10_corpus_mode_holds_an_eligibility_move_to_the_parse_oracles(scorer, monkeypatch, tmp_path):
    # round-9 scorer lens, major: a defect inside W-1 that made the repaired instrument exclude a PR was credited to
    # W-1 on the counterfactual alone; the PR's parse is now held to the scorer's own beforehand
    pg = scorer
    files = [("db/q.sql", "modified", "@@ -1,2 +1,2 @@\n SELECT 1;\n--- users\n+-- accounts"),
             ("src/a.py", "modified", "@@ -1 +1 @@\n-a\n+b")]
    out = tmp_path / "gates.json"
    pg.run_corpus(_shelf(tmp_path, [(3, "2 files changed.", files)]), None, out)
    # (G-C0 refuses an uncommitted tree; nothing else may fire)
    assert set(json.loads(out.read_text(encoding="utf-8"))["violations"]) <= {"G-C0_modified_tree"}
    good = "                old_left -= 1\n                old_no += 1\n"
    bad = good + ('                if text.startswith("-- ") and cur is not None:\n'
                  '                    status.setdefault(_norm(text[3:].strip()), "M")\n')
    _planted(pg, monkeypatch, good, bad, "x10elig")
    (tmp_path / "planted").mkdir()
    assert pg.run_corpus(_shelf(tmp_path / "planted", [(3, "2 files changed.", files)]), None, out) == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["G-C2_eligibility"]["moves"] == {"baseline_only": 1, "refused_by_a_parse_oracle": 1}
    assert "G-C7_oracle:W-1_status" in payload["violations"], payload["violations"]


def test_x10_corpus_mode_sends_a_pr_with_a_compat_claim_through_the_git_door(scorer, monkeypatch, tmp_path):
    # round-9 scorer lens, minor: a compat claim reads gate_diff's own sides, and a compat-only PR outside the sample
    # was never tried there
    pg = scorer
    pid = next(i for i in range(1, 500) if pg.sample_key(i) % 25)
    files = [(".lib/api.py", "modified", "@@ -1,2 +1 @@\n-def api():\n-    pass\n+x = 1")]
    out = tmp_path / "gates.json"
    pg.run_corpus(_shelf(tmp_path, [(pid, "Keeps backward compatibility.", files)]), None, out)
    assert set(json.loads(out.read_text(encoding="utf-8"))["violations"]) <= {"G-C0_modified_tree"}
    door = json.loads(out.read_text(encoding="utf-8"))["G-C8_git_door"]
    assert door["sample"]["scored"] == 1 and door["sample"]["tried_for_a_compat_claim"] == 1, door["sample"]
    _planted(pg, monkeypatch, "                    c.verdict, c.why, extra = _compat_reading(sides)",
             "                    c.verdict, c.why, extra = _compat_reading(sides if raw_input_len is not None else "
             "{k: v for k, v in (sides or {}).items() if not k.startswith('.')})", "x10compat")
    (tmp_path / "planted").mkdir()
    assert pg.run_corpus(_shelf(tmp_path / "planted", [(pid, "Keeps backward compatibility.", files)]), None, out) == 1
    door = json.loads(out.read_text(encoding="utf-8"))["G-C8_git_door"]
    assert "G-C7_oracle:C-2_compat_surface" in door["violations"], door["violations"]


def test_x10_g_c0_names_every_byte_both_instruments_read(scorer, monkeypatch):
    # round-9 scorer lens, minor: styxx/declare.py (imported by both instruments), path1_extensions.txt (read by the
    # oracle) and styxx/_fold.py (K-2) are provenance too, each hashed into the payload
    pg = scorer
    for f in ("styxx/declare.py", "papers/closed-model-frontier/path1_extensions.txt", "styxx/_fold.py", "styxx/_xid.py"):
        assert f in pg.PROVENANCE_FILES, f
    prov = pg.provenance()
    assert {"declare_sha256", "fold_sha256", "xid_sha256", "path1_extensions_sha256"} <= set(prov)
    real = pg._git

    def git(*args):
        if args[:2] == ("status", "--porcelain"):
            return " M styxx/declare.py\n" if "styxx/declare.py" in args else ""
        return real(*args)
    monkeypatch.setattr(pg, "_git", git)
    prov = pg.provenance()
    assert not prov["unmodified_against_head"] and prov["modified"] == [" M styxx/declare.py"]


def test_x10_the_scorer_refuses_a_fold_that_misses_a_merge(scorer):
    # K-2's table is pinned by sha256 in the scorer and held sound against this Python's own str.lower()
    from styxx import _fold
    pg = scorer

    class Fake:
        UNICODE_VERSION = "16.0.0"
        FOLD = _fold.FOLD.replace("1t:q:w:1,", "1t:p:w:1,")          # 'Z' no longer folds to 'z'
    with pytest.raises(SystemExit, match="does not hash"):
        pg.check_fold(Fake, FOLD_SHA256)
    with pytest.raises(SystemExit, match="not sound at U[+]005A"):
        pg.check_fold(Fake, hashlib.sha256(Fake.FOLD.encode("ascii")).hexdigest())
    assert pg.check_fold(_fold, FOLD_SHA256) == _fold.MAP


# ─────────────────────────────── NOTE_path2_eleventh_pass_2026_09_28: x11, the guard in the scorer (G-C9), and round 10

def _pair(pid: str) -> tuple:
    p = next(p for p in json.loads(PAIRS.read_text(encoding="utf-8")) if p["id"] == pid)
    return p["summary"], p["diff"]


X11_TWO_CREATED = "--- /dev/null\n+++ b/a.py\n@@ -0,0 +1 @@\n+x\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-a\n+b\n"
X11_V2 = "path2:v2-a-binary-header-holding-a-line-separator-registers-its-file"
X11_PLANTED = {
    # label: ([(good, bad), ...] in styxx/diffgate.py, summary, diff, the violation that must fire)
    "the guard keeps a difference no repair explains (the reference skipped)": (
        [("        if main_verdict == c.verdict:", "        if True:")], *_pair(X11_V2), "G-C9_guard:files_changed_count"),
    "the guard reads main as raising (the reference dropped)": (
        [("        ref = reference()", "        ref = None")], *_pair("path2:97-two-readmes"), "G-C9_guard:file_created"),
    # where main raises, the scorer still asks G-C9: the reading's own Z-1 doubt dropped and the guard keeping a decided
    # claim there (no main verdict to license it from) must be refused by the guard's own gate, not only by G-C1 and G-C7
    "where main raises, the guard keeps a decided claim": (
        [('    if main is None:\n        return None\n    if main.raises:\n        return "main raises on this diff',
          '    if True:\n        return None\n    if main.raises:\n        return "main raises on this diff'),
         ("        main_verdict = None if r is None else r.verdict",
          "        main_verdict = (c.verdict if ref is None else None) if r is None else r.verdict")],
        *_pair("path2:l-guard-main-raises-every-claim-abstains"), "G-C9_guard:tests_added"),
    "the guard pairs claims without their occurrence": (
        [("        out.append((c.kind, c.text, seen[k]))", "        out.append((c.kind, c.text, 0))")],
        "Created a.py, created b.py.", X11_TWO_CREATED, "G-C9_guard:file_created"),
    "the guard prints another reason": (
        [('_GUARD_DIFFERS = ("main\'s reading gives {main} and this one {this}; no named repair',
          '_GUARD_DIFFERS = ("main gives {main} and this one {this}; no named repair')],
        *_pair(X11_V2), "G-C9_guard:files_changed_count"),
    "K-5's symbol span stops at a combining mark": (
        [("(_xid_word(sentence[j]) or _xid_continues(sentence[j]) or _xid_skew(sentence[j]))",
          "(_xid_word(sentence[j]) or _xid_skew(sentence[j]))")],
        *_pair("path2:w2-a-claimed-name-with-a-virama-is-read-whole"), "G-C9_guard:symbol_added"),
    "#121 switched off keys nothing by main's key": (
        [('        return _main_key(p) if "#121" in self.off else _norm(p)', "        return _norm(p)")],
        *_pair("path2:121-dotfile-twins"), "G-C9_switch_is_not_the_revert:#121"),
    # (the switch, switched off, reads no file at all, so it gives main's UNCHECKABLE back on a record with no dotted key;
    # only #121's precondition stood between that and a licence)
    "a licence without the precondition, beside a switch that reads more than its repair": (
        [('        return _main_key(p) if "#121" in self.off else _norm(p)',
          '        return "" if "#121" in self.off else _norm(p)'),
         ('        if main_verdict is not None and any(_precondition(repair, c, seen["status"], seen["sides"], seen.get("licence"))',
          "        if main_verdict is not None and any(True")],
        *_pair(X11_V2), "G-C9_guard:files_changed_count"),
    # the eleventh pass's own differential: main's paths folding alike, asked before main's two lists are compared
    "Z-3 compares main's two lists before asking whether its paths fold alike": (
        [("    if _folds_apart(forms):\n        return f\"{_Z3_PREFIX}: {_Z3_FOLDS}\"\n", "")],
        "3 files changed. Only touches assets/.",
        ("+++ b/Ᲊ.md\rdiff -Nu a/Src/config.toml b/Src/config.toml\n--- /dev/null\t1970-01-01 00:00:00.000000000 "
         "+0000\x1c+++ b/Src/config.toml\t2024-05-06 07:08:09.000000000 +0000\x1c@@ -0,0 +1,3 @@\n+k723 = 8\x85+++ "
         "/dev/null\t1970-01-01 00:00:00.000000000 +0000\rdiff --cc m.py +k271 = 1\n+k279 = 8\x0b\x0b+++ b/ᲊ.md\n"),
        "G-C7_oracle:Y_notes"),
    # round-10 scorer lens, blocker (R1.2): nothing anchored the unmeasured reason where main raises
    "raises: the unmeasured reason loses its parse-failure clause": (
        [("        if raw_input_len:\n            no_evidence += (",
          "        if raw_input_len and not (main is not None and main.raises):\n            no_evidence += (")],
        "t\n\nAdded 1 test. 1 file changed.", "x\x0c+++ /dev/null\n", "G-C7_oracle:why_unmeasured"),
    # the same defect, read claim by claim: an unmeasured claim's reason is held to this file's own reason, not to the
    # gate's (which carries the same defect)
    "raises: an unmeasured claim's reason loses the parse-failure clause with its gate's": (
        [("        if raw_input_len:\n            no_evidence += (",
          "        if raw_input_len and not (main is not None and main.raises):\n            no_evidence += (")],
        "t\n\nAdded 1 test. 1 file changed.", "x\x0c+++ /dev/null\n", "G-C7_oracle:tests_added_claim"),
    "raises: why_unmeasured set on a measured gate": (
        [('                    measured=not no_evidence, why_unmeasured=no_evidence or "",',
          '                    measured=not no_evidence, why_unmeasured=no_evidence or ("main raises" if main is not None '
          'and main.raises else ""),')], "Added 1 test.", X9_RAISES, "G-C7_oracle:why_unmeasured"),
    # round-10 scorer lens, blocker (R1.0): the strict gates' reports were never compared
    "strict: the never-read sentences dropped": (
        [("                    sentences_total=total, uncovered_texts=uncovered_texts,",
          "                    sentences_total=total, uncovered_texts=[] if strict else uncovered_texts,")],
        "Refactored the loader. Keeps backward compatibility. Adds function zap.",
        "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1,2 @@\n x = 0\n+def zap():\n", "G-C1_strict_report_differs"),
    "strict: a compat claim's languages dropped": (
        [("                    c.detail.update(extra)",
          '                    c.detail.update(extra)\n                    if strict:\n                        c.detail.pop("languages", None)')],
        "Keeps backward compatibility.", "--- a/src/a.py\n+++ b/src/a.py\n@@ -1,2 +1 @@\n-def api():\n-    pass\n+x = 1\n",
        "G-C1_strict_report_differs"),
    "to_dict: a FAIL with no CONTRADICTED claim prints no claims": (
        [('                "claims": [c.__dict__ for c in self.claims],',
          '                "claims": [c.__dict__ for c in self.claims] if (self.verdict == "PASS" or any('
          'c.verdict == "CONTRADICTED" for c in self.claims)) else [],')],
        "Refactored the loader. Keeps backward compatibility.",
        "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1,2 @@\n x = 0\n+def zap():\n", "G-C1_report_differs_from_its_gate"),
}


@pytest.mark.parametrize("label", sorted(X11_PLANTED))
def test_x11_every_eleventh_pass_rule_and_round_10_scorer_finding_fails_on_a_planted_defect(scorer, monkeypatch, label):
    # NOTE_path2_eleventh_pass: the guard is a rule the scorer re-implements (G-C9), and each round-10 scorer finding is a
    # check of its own; each refuses a defect planted in the instrument, and the unplanted record scores clean
    pg = scorer
    pairs, summary, diff, violation = X11_PLANTED[label]
    t = pg.Tally(name_prs=True)
    t.pair("clean", summary, diff, pg.raw_paths(diff))
    assert not t.violations, t.violating
    _planted_many(pg, monkeypatch, pairs, "x11")
    t = pg.Tally(name_prs=True)
    t.pair("planted", summary, diff, pg.raw_paths(diff))
    assert violation in t.violations, dict(t.violations)


def test_x12_a_licence_without_its_precondition_is_refused_on_the_pinned_pairs(scorer, monkeypatch):
    """At the eleventh pass each precondition held wherever its switch alone gave main's verdict back, so dropping the
    precondition changed no claim on the pinned pairs (an equivalent mutant there, held only by stub tests). The twelfth
    pass's preconditions read more than the switches do -- git's own rendering for #121, the case and Z-3's doubts for
    #97 (NOTE_path2_twelfth_pass, A.1, A.2) -- so the same drop now keeps a verdict on round 11's reproductions, and the
    scorer's own guard (G-C9) refuses it."""
    pg = scorer
    _planted_many(pg, monkeypatch, [(
        '        if main_verdict is not None and any(_precondition(repair, c, seen["status"], seen["sides"], seen.get("licence"))',
        "        if main_verdict is not None and any(True")], "x12pre")
    t = pg.Tally(name_prs=True)
    for p in json.loads(PAIRS.read_text(encoding="utf-8")):
        t.pair(p["id"], p["summary"], p["diff"], pg.raw_paths(p["diff"]))
    refused = {pid for rule, pid in t.violating if rule.startswith("G-C9_guard")}
    assert {"path2:m-r11-b121-difflib-noprefix-bdir", "path2:m-r11-a97-created-suffix-case"} <= refused, t.violating


def test_x11_the_scorers_own_guard_licenses_only_by_the_precondition_and_the_switch_together(scorer):
    """The scorer's guard (G-C9's oracle), on stub readings: a difference is kept only where a revert alone gives main's
    verdict back AND that repair's precondition holds. On every corpus the three reverts read only their repairs, so the
    two conditions never part there (see the test above); these stubs part them, one at a time."""
    pg = scorer
    C = dg.DiffClaim

    def one(kind, text, verdict, detail):
        return [C(kind=kind, text=text, detail=dict(detail), verdict=verdict, why=f"{kind} {verdict}")]

    def final(status, before, main, switched, facts=None):
        facts = {"rendered": True, "soft": False, "forms": {k: [k] for k in status}} if facts is None else facts
        return pg.expected_guard("", before, main, lambda r: switched.get(r, before), status, {}, None, facts)

    # the #121 revert gives main's verdict back, but no key the claim reads keeps a dot: abstain, naming main's verdict
    before = one("file_touched", "Modified src/x.py.", "VERIFIED", {"path": "src/x.py"})
    main = one("file_touched", "Modified src/x.py.", "CONTRADICTED", {"path": "src/x.py"})
    got = final({"src/x.py": "M"}, before, main, {"#121": main})
    assert [g[2] for g in got] == ["UNCHECKABLE"] and "CONTRADICTED" in got[0][3], got
    # a dotted key holds #121's precondition, but no revert gives main's verdict back: abstain
    before = one("files_changed_count", "2 files changed.", "VERIFIED", {"claimed": 2})
    main = one("files_changed_count", "2 files changed.", "CONTRADICTED", {"claimed": 2})
    twins = {".env": "A", "env": "A"}
    got = final(twins, before, main, {})
    assert [g[2] for g in got] == ["UNCHECKABLE"], got
    # both: the precondition holds and the #121 revert gives main's verdict back -- the repair's verdict is kept
    got = final(twins, before, main, {"#121": main})
    assert [g[2] for g in got] == ["VERIFIED"], got
    # NOTE_path2_twelfth_pass (A.1): the same, where the diff is not git's own rendering -- no licence, abstain
    got = final(twins, before, main, {"#121": main}, {"rendered": False, "soft": False, "forms": {}})
    assert [g[2] for g in got] == ["UNCHECKABLE"], got
    # (A.2): #97's suffix match holds only once lower-cased, or beside a Z-3 doubt -- no licence, abstain
    before = one("file_created", "Created src/README.md.", "VERIFIED", {"path": "src/README.md"})
    main = one("file_created", "Created src/README.md.", "UNCHECKABLE", {"path": "src/README.md"})
    listing = {"docs/readme.md": "M", "lib/src/readme.md": "A"}
    kept = {"rendered": True, "soft": False, "forms": {"docs/readme.md": ["docs/README.md"],
                                                         "lib/src/readme.md": ["lib/src/readme.md"]}}
    assert [g[2] for g in final(listing, before, main, {"#97": main}, kept)] == ["UNCHECKABLE"]
    kept["forms"]["lib/src/readme.md"] = ["lib/src/README.md"]
    assert [g[2] for g in final(listing, before, main, {"#97": main}, kept)] == ["VERIFIED"]
    assert [g[2] for g in final(listing, before, main, {"#97": main}, dict(kept, soft=True))] == ["UNCHECKABLE"]
    # #121 on a path claim: the entry the tiers resolved must match the claim, case kept, by that tier (this pass's own
    # differential: "Created X.toml." read VERIFIED from a created `.config/x.toml`, licensed by #121's dot)
    before = one("file_created", "Created X.toml.", "VERIFIED", {"path": "X.toml"})
    main = one("file_created", "Created X.toml.", "UNCHECKABLE", {"path": "X.toml"})
    listing = {".config/x.toml": "A", "config/x.toml": "M"}
    facts = {"rendered": True, "soft": False, "forms": {".config/x.toml": [".config/x.toml"],
                                                         "config/x.toml": ["config/x.toml"]}}
    assert [g[2] for g in final(listing, before, main, {"#121": main}, facts)] == ["UNCHECKABLE"]
    before = one("file_created", "Created x.toml.", "VERIFIED", {"path": "x.toml"})
    main = one("file_created", "Created x.toml.", "UNCHECKABLE", {"path": "x.toml"})
    assert [g[2] for g in final(listing, before, main, {"#121": main}, facts)] == ["VERIFIED"]


def test_x11_g_c8_compares_the_two_doors_readings_before_the_guard(scorer):
    """G-C8 holds the git door's reading to the raw door's, both before the guard: each door's guard reads its own main
    reading, so a final claim may differ between doors where the readings agree. Here K-5 (an accented word in the
    sentence) gives each door main's claim, UNCHECKABLE, while both readings resolve #97's path VERIFIED: the clean record
    scores with no violation, and a G-C8 that compared a final gate with a reading would refuse it."""
    from collections import Counter
    pg = scorer
    summary = "Created integrations/git/README.md, voilà."
    diff = _m("README.md") + "--- /dev/null\n+++ b/integrations/git/README.md\n@@ -0,0 +1 @@\n+hello\n"
    assert [c.verdict for c in dg._evaluate_text(summary, diff, dg._ALL_ON).claims] == ["VERIFIED"]
    assert [c.verdict for c in gate_diff_text(summary, diff).claims] == ["UNCHECKABLE"]
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "k5-both-doors", summary, diff, sample)
    assert sample["scored"] == 1 and not t.violations, (dict(sample), t.violating)


def test_x11_the_git_door_strict_report_is_compared(scorer, monkeypatch):
    # round-10 scorer lens, blocker (R1.0, plant c): a gate's base naming its head under --strict, at the git door
    from collections import Counter
    pg = scorer
    summary, diff = "Added 1 test. Adds function foo. 1 file changed. Only touches tests/.", _m("tests/test_a.py")
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "clean", summary, diff, sample)
    assert sample["scored"] == 1 and not t.violations, t.violating
    _planted_many(pg, monkeypatch, [(
        "    return DiffGate(verdict=verdict, base=base, head=head, claims=claims,",
        "    return DiffGate(verdict=verdict, base=head if strict else base, head=head, claims=claims,")], "x11gitstrict")
    t, sample = pg.Tally(name_prs=True), Counter()
    pg.git_door_pair(t, "planted", summary, diff, sample)
    assert "G-C1_strict_report_differs" in t.violations, dict(t.violations)


@pytest.mark.parametrize("good,bad", [
    # round-10 scorer lens, blocker (R1.1): main's --name-status split forgetting U+2029 alone, refused by the third canary
    ("    for line in _py_lines(name_status):",
     "    for line in re.split(\"\\r\\n|[\\n\\r\\x0b\\x0c\\x1c\\x1d\\x1e\\x85\\u2028]\", name_status):"),
    # round-10 scorer lens, minor (R1.3): K-3's `+++ /dev/null` with no `---` read as a deletion, refused by a raw canary
    ("                cur = None\n            else:\n                if _dev_null(new):",
     "                cur = None\n                status.setdefault(\"dev/null\", \"D\")\n            else:\n"
     "                if _dev_null(new):"),
    # (R1.3) the unreadable-header doubt dropped, refused by the raw canary beside dotted twins
    ("                soft.append(_Z3_UNREAD_PAIR)     # NOTE_path2_eleventh_pass (R0.0): the pair under an unreadable header",
     "                pass"),
    # (R1.3) the doubt of a pair without a header's shape dropped, refused by the raw canary after an exact hunk
    ("                soft.append(_Z3_UNSHAPED)        # NOTE_path2_eleventh_pass (R0.2): read as a header without its shape",
     "                pass"),
])
def test_x11_the_canaries_refuse_what_no_shelf_record_reaches(scorer, monkeypatch, good, bad):
    pg = scorer
    report, violating = pg.score_canaries()
    assert report["pass"] and not violating, report
    assert report["door"]["scored"] == len(pg.DOOR_CANARIES) == 7 and report["raw_door_canaries"] == len(pg.RAW_CANARIES)
    assert chr(0x2029) in pg.SPLIT_ONLY_BY_PYTHON
    _planted_many(pg, monkeypatch, [(good, bad)], "x11canary")
    report, violating = pg.score_canaries()
    assert not report["pass"] and violating, report


def test_x11_the_reference_is_provenance_and_must_be_the_baseline(scorer, monkeypatch):
    pg = scorer
    assert "styxx/_diffgate_ref.py" in pg.PROVENANCE_FILES
    prov = pg.provenance()
    assert prov["reference_is_the_baseline"] and prov["reference_sha256"] == pg.BASE_SHA256
    t = pg.Tally(name_prs=True)
    t.report({**prov, "reference_is_the_baseline": False, "unmodified_against_head": True})
    assert "G-C0_reference_is_not_the_baseline" in t.violations


# ---- NOTE_path2_twelfth_pass_2026_09_29: the scorer, round 11's findings ------------------------------------------------

X12_GIT_TAIL = ("        return _REF.gate_diff(summary_text, repo, base, head, run=None, strict=False, evidence=None, commit=None)\n"
                "\n    return _guard(evaluate, reference, strict=strict, tp=tp)\n")
X12_LICENCE = ('        if main_verdict is not None and any(_precondition(repair, c, seen["status"], seen["sides"], seen.get("licence"))\n'
               "                                    and switched_verdict(repair, key) == main_verdict for repair in REPAIRS):")
X12_CANARY_PLANTS = {
    # round-11 scorer lens, blocker: the git door's guard defects, admitted in both modes at the eleventh pass
    "PG1 the git door returns the reading unguarded": [(X12_GIT_TAIL, X12_GIT_TAIL.replace(
        "    return _guard(evaluate, reference, strict=strict, tp=tp)\n", "    return evaluate(_ALL_ON, None)\n"))],
    "PG1c the git door calls main's gate_diff and ignores it": [(X12_GIT_TAIL, X12_GIT_TAIL.replace(
        "    return _guard(evaluate, reference, strict=strict, tp=tp)\n",
        "    return _guard(evaluate, lambda: (reference(), evaluate(_ALL_ON, None))[1], strict=strict, tp=tp)\n"))],
    "PG3 a licence without the precondition": [(X12_LICENCE, "        if main_verdict is not None and any("
                                                             "switched_verdict(repair, key) == main_verdict for repair in REPAIRS):")],
    "PG4 a licence without the switch": [(X12_LICENCE, X12_LICENCE.replace(
        "\n                                    and switched_verdict(repair, key) == main_verdict for repair in REPAIRS):",
        "\n                                    for repair in REPAIRS):"))],
    # round-11 scorer lens, minor: reader defects that fire only on a dotted key, refused by the reading oracle (G-C7)
    "PD2 a dotted entry satisfies any path claim": [("    if want and st != want:",
                                                     '    if want and st != want and not p.startswith("."):')],
    "PD3 the count adds one beside a dotfile": [(
        '                        c.verdict = "VERIFIED" if n == len(status) else "CONTRADICTED"',
        '                        c.verdict = "VERIFIED" if n == len(status) + any(k.startswith(".") for k in status) '
        'else "CONTRADICTED"')],
    # the twelfth pass's own licences, each dropped
    "A.1 #121 licenses in any rendering": [('        if not lic.get("rendered"):\n            return False',
                                            "        if False:\n            return False")],
    "A.1 a rename line may name anything": [(
        '                if pending is None or written is None or line.split(" ", 2)[2] != written[at]:',
        "                if pending is None or written is None:")],
    "A.1 a line no reading places is git's rendering": [(
        "                soft.append(_Z3_UNPLACED)        # K-4: a line no reading places may name a file neither counts\n"
        "                git_form = False",
        "                soft.append(_Z3_UNPLACED)        # K-4: a line no reading places may name a file neither counts")],
    "A.2 #97 licenses a match only in case": [(
        '        if lic.get("soft") or not as_read or not all(f == claimed or f.endswith("/" + claimed) for f in as_read):',
        '        if lic.get("soft"):')],
    "A.2 #97 licenses beside a Z-3 doubt": [(
        '        if lic.get("soft") or not as_read or not all(f == claimed or f.endswith("/" + claimed) for f in as_read):',
        '        if not as_read or not all(f == claimed or f.endswith("/" + claimed) for f in as_read):')],
}


def test_x12_the_canaries_reach_every_guard_outcome_on_both_doors(scorer):
    """Round-11 scorer lens, blocker: no record of either mode reached the git door's guard abstention or its K-5 reading,
    and no abstention held a precondition, so defects only those outcomes expose were admitted. The canaries now reach
    them in every run, and a run whose canaries reach none fails."""
    pg = scorer
    report, violating = pg.score_canaries()
    assert report["pass"] and not violating, report
    for outcome in pg.GUARD_OUTCOMES["raw"]:
        assert report["guard"].get(outcome), (outcome, report["guard"])
    for outcome in pg.GUARD_OUTCOMES["git"]:
        assert report["guard_git_door"].get(outcome), (outcome, report["guard_git_door"])


@pytest.mark.parametrize("label", sorted(X12_CANARY_PLANTS))
def test_x12_the_canaries_refuse_a_planted_guard_or_licence_defect(scorer, monkeypatch, label):
    """Each defect is refused by the scorer program itself -- the canaries every run of either mode scores -- not only by a
    stub test (PG7, keeping a decided claim where main makes no claim or raises, is not among them: the two readers
    extract the same claims, and where main raises Z-1 to Z-3 abstain every claim unless --run or --evidence is given,
    which the scorer never passes; tests/test_diffgate_guard.py holds it on stubs)."""
    pg = scorer
    _planted_many(pg, monkeypatch, X12_CANARY_PLANTS[label], "x12canary")
    report, violating = pg.score_canaries()
    assert not report["pass"] and violating, (label, report)


def test_x12_the_scorer_replays_mains_kind_leak(scorer):
    """Round-11 scorer lens, minor: main's loop rebinds `kind` when V13 demotes a created or deleted claim, so a later match
    of the same template in the sentence reads as file_touched; the scorer's replay of main's extraction must too, or
    G-C9 flags a correct instrument (the raw canary `canary:raw-mains-kind-leak`)."""
    pg = scorer
    summary = "Delet\u00e9d the parser in lib/x.css and removed the file ci.yml."
    kinds = [k for k, _s in pg.own_claim_sentences(summary)]
    assert kinds == [c.kind for c in pg.BASE.gate_diff_text(summary, _m("lib/x.css")).claims]
    assert kinds == [c.kind for c in dg._evaluate_text(summary, _m("lib/x.css"), dg._ALL_ON).claims]
    assert "file_touched" in kinds


def test_x12_the_counterfactual_lets_an_environment_failure_through(scorer, monkeypatch):
    """Round-11 scorer lens, minor: a failure of git, the operating system or a timeout inside the counterfactual copy
    fails the run; only what reverted code raises (main's AttributeError on K-3) reads as the copy raising."""
    pg = scorer
    cf = pg.Counterfactual("1 file changed.", _m("src/a.py"))

    def boom(*_a, **_k):
        raise RuntimeError("git diff: fatal: not a git repository")
    monkeypatch.setattr(pg.CF, "_evaluate_text", boom)
    with pytest.raises(RuntimeError):
        cf.reading({"#97"})

    def k3(*_a, **_k):
        raise AttributeError("'NoneType' object has no attribute 'startswith'")
    monkeypatch.setattr(pg.CF, "_evaluate_text", k3)
    assert cf.reading({"#121"}) is None
    assert pg.REVERTED_RAISES == (AttributeError,)


def test_x12_the_scorer_reads_the_licence_facts_as_the_instrument_does(scorer):
    """G-C7 holds what the instrument's licences read (git's own rendering, a Z-3 doubt, each key's paths as written) to
    the scorer's own reading of the same, on every pinned pair and every canary."""
    pg = scorer
    records = [(p["id"], p["diff"]) for p in json.loads(PAIRS.read_text(encoding="utf-8"))]
    records += [(cid, d) for cid, _s, d in pg.RAW_CANARIES + pg.DOOR_CANARIES]
    facts = []
    for pid, diff in records:
        got, own = {}, {}
        dg._diff_notes(diff, got)
        pg.own_read(diff, own)
        assert got == own, pid
        facts.append(got)
    assert len(records) > 300
    # and both answers of each occur among them, so the comparison is not of a constant
    assert {f["rendered"] for f in facts} == {True, False} and {f["soft"] for f in facts} == {True, False}

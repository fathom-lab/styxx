# -*- coding: utf-8 -*-
"""COMPAT-1 (PREREG_compat1_2026_09_16): the compatibility claim is read, never judged.

One template, one verdict (UNCHECKABLE), and a reason that names the public top-level definitions
the diff removed without re-defining, per language, with files. Every reason branch is driven
here; the accusing verdict for this kind does not exist in the code and cannot appear."""
import json
import subprocess
from pathlib import Path

import pytest

from styxx import diffgate as dg
from styxx.diffgate import gate_diff, gate_diff_text

PY_REMOVES = """\
--- a/src/api.py
+++ b/src/api.py
@@ -1,6 +1,3 @@
-def session():
-    return 1
-class Legacy:
-    pass
-def _private():
-    pass
+def other():
+    return 2
"""

PY_MOVES = """\
--- a/src/api.py
+++ b/src/api.py
@@ -1,2 +1 @@
-def session():
-    return 1
+from src.core import session
--- a/src/core.py
+++ b/src/core.py
@@ -0,0 +1,2 @@
+def session():
+    return 1
"""

JS_REMOVES = """\
--- a/lib/index.ts
+++ b/lib/index.ts
@@ -1,4 +1,2 @@
-export function parse(s: string) { return s; }
-export const VERSION = "1";
-export interface Options { a: number }
+export function parse2(s: string) { return s; }
"""

GO_RS_JAVA = """\
--- a/pkg/x.go
+++ b/pkg/x.go
@@ -1,3 +1,2 @@
-func Exported() {}
-func unexported() {}
-type Thing struct{}
+func Exported2() {}
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,2 +1 @@
-pub fn run() {}
-fn helper() {}
+pub fn run2() {}
--- a/src/main/java/A.java
+++ b/src/main/java/A.java
@@ -1,3 +1,2 @@
-    public String name() { return ""; }
-    private String hidden() { return ""; }
+    public String name2() { return ""; }
"""

MD_ONLY = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n"


def _claim(summary, diff):
    g = gate_diff_text(summary, diff, run=None, strict=False)
    found = [c for c in g.claims if c.kind == "compat_claim"]
    assert len(found) == 1, [(c.kind, c.verdict) for c in g.claims]
    return g, found[0]


@pytest.mark.parametrize("sentence", [
    "No breaking changes.",
    "This is a non-breaking change.",
    "Fully backward compatible with existing callers.",
    "Maintains backwards compatibility.",
    "Zero behavior change.",
    "No behavioural changes intended.",
    "No functional changes.",
    "Does not break existing behavior.",
    "Does not change the public API.",
])
def test_the_closed_phrase_set_is_read(sentence):
    g, c = _claim(sentence, PY_REMOVES)
    assert c.verdict == "UNCHECKABLE" and g.verdict == "PASS"


def test_removed_public_python_definitions_are_named_and_private_ones_are_not():
    _, c = _claim("Refactor. No breaking changes.", PY_REMOVES)
    assert c.why == ("compatibility claimed; the diff removes 2 public definition(s) from the surface, "
                     "not re-defined in the added lines: src/api.py: session, src/api.py: Legacy")
    assert [r["name"] for r in c.detail["removed"]] == ["session", "Legacy"]
    assert c.detail["languages"] == ["python"]


def test_a_definition_that_moves_or_is_still_referenced_is_not_a_removal():
    _, c = _claim("No breaking changes.", PY_MOVES)
    assert c.detail["removed"] == []
    assert c.why.startswith("compatibility claimed; no public top-level definition removed (python read")


def test_javascript_exports_are_read():
    _, c = _claim("Backward compatible.", JS_REMOVES)
    assert [r["name"] for r in c.detail["removed"]] == ["parse", "VERSION", "Options"]
    assert all(r["language"] == "js/ts" for r in c.detail["removed"])


def test_go_rust_and_java_public_surface_is_read_and_unexported_names_are_skipped():
    _, c = _claim("No functional changes.", GO_RS_JAVA)
    got = {(r["language"], r["name"]) for r in c.detail["removed"]}
    assert got == {("go", "Exported"), ("go", "Thing"), ("rust", "run"), ("java", "name")}
    assert c.detail["languages"] == ["go", "rust", "java"]


def test_no_covered_language_says_so():
    _, c = _claim("Non-breaking docs change.", MD_ONLY)
    assert c.why == ("compatibility claimed; no language this reading covers in the diff "
                     "(python, js/ts, go, rust, java)")
    assert c.detail["removed"] == []


def test_at_most_five_names_are_shown_and_the_total_is_stated():
    lines = "\n".join(f"-def f{i}():\n-    pass" for i in range(8))
    diff = f"--- a/m.py\n+++ b/m.py\n@@ -1,16 +1 @@\n{lines}\n+x = 1\n"
    _, c = _claim("No breaking changes.", diff)
    assert "removes 8 public definition(s)" in c.why and "(+3 more)" in c.why
    assert len(c.detail["removed"]) == 8 and c.why.count("m.py:") == 5


def test_the_kind_has_exactly_one_verdict_in_code_and_in_every_branch():
    assert dg._COMPAT_VERDICTS == ("UNCHECKABLE",)
    for sides in (None, {}, dg.parse_unified_diff_sides(PY_REMOVES), dg.parse_unified_diff_sides(MD_ONLY),
                  dg.parse_unified_diff_sides(PY_MOVES)):
        verdict, why, extra = dg._compat_reading(sides)
        assert verdict == "UNCHECKABLE" and why.startswith("compatibility claimed;")
        assert isinstance(extra["removed"], list)


def test_the_claim_never_fails_the_gate_even_under_strict_only_as_uncheckable():
    g = gate_diff_text("No breaking changes.", PY_REMOVES, run=None, strict=True)
    assert g.verdict == "FAIL"              # --strict fails on UNCHECKABLE, as for every kind
    assert all(c.verdict == "UNCHECKABLE" for c in g.claims)
    g = gate_diff_text("No breaking changes.", PY_REMOVES, run=None, strict=False)
    assert g.verdict == "PASS"


def test_the_demo_makes_no_compatibility_claim_and_is_unchanged():
    g = gate_diff_text(dg._DEMO_SUMMARY, dg._DEMO_DIFF, run=None, strict=False)
    assert not [c for c in g.claims if c.kind == "compat_claim"]
    assert sorted(c.kind for c in g.claims if c.verdict == "CONTRADICTED") == ["only_touches", "symbol_added", "tests_added"]


def test_the_git_entry_point_reads_removed_lines_too(tmp_path):
    def git(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "api.py").write_text("def session():\n    return 1\n\ndef keep():\n    return 2\n", encoding="utf-8")
    git("add", "-A"); git("commit", "-qm", "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    (tmp_path / "api.py").write_text("def keep():\n    return 2\n", encoding="utf-8")
    git("add", "-A"); git("commit", "-qm", "drop session")
    g = gate_diff("Removed dead code. No breaking changes.", tmp_path, base, "HEAD")
    c = [x for x in g.claims if x.kind == "compat_claim"][0]
    assert c.verdict == "UNCHECKABLE" and "api.py: session" in c.why
    assert g.verdict == "PASS"


def test_parse_unified_diff_sides_keeps_both_sides_per_file():
    sides = dg.parse_unified_diff_sides(PY_MOVES)
    assert set(sides) == {"src/api.py", "src/core.py"}
    assert sides["src/api.py"][1] == ["def session():", "    return 1"]
    assert sides["src/core.py"][0] == ["def session():", "    return 1"]
    assert dg.parse_unified_diff_sides("") == {}

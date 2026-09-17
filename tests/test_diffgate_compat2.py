"""COMPAT-2 (PREREG_compat2_surface_and_panel_2026_09_16): the reading is sharpened and a candidate
is computed, and the verdict is still UNCHECKABLE -- the licence flag is false and pinned so.

Every test here drives `_compat_reading` / `gate_diff_text` through the surface split, the
scaffolding rule, the signature reading and the candidate, and asserts that no verdict other than
UNCHECKABLE can be produced while `COMPAT2_LICENSED` is False.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import styxx.diffgate as dg  # noqa: E402
from styxx.diffgate import gate_diff_text  # noqa: E402

SURFACE_AND_SCAFFOLD = """--- a/pkg/api.py
+++ b/pkg/api.py
@@ -1,5 +1,3 @@
-def session(a, b=1):
-    return 1
-def keep(x):
-    pass
+def keep(x, y):
+    pass
--- a/tests/test_api.py
+++ b/tests/test_api.py
@@ -1,2 +1,1 @@
-def helper():
-    pass
+x = 1
"""

SCAFFOLD_ONLY = """--- a/tests/test_api.py
+++ b/tests/test_api.py
@@ -1,2 +1,1 @@
-def helper():
-    pass
+x = 1
--- a/examples/demo.ts
+++ b/examples/demo.ts
@@ -1,1 +1,1 @@
-export function run() {}
+const y = 1;
"""

MOVE_SAME_SIGNATURE = """--- a/pkg/a.py
+++ b/pkg/a.py
@@ -1,2 +1,1 @@
-def moved(a, b):
-    pass
+x = 1
--- a/pkg/b.py
+++ b/pkg/b.py
@@ -1,1 +1,2 @@
+def moved(a,  b):
+    pass
"""

MULTILINE_SIGNATURE = """--- a/pkg/a.go
+++ b/pkg/a.go
@@ -1,3 +1,3 @@
-func Serve(addr string,
-	opts ...Option) error {
+func Serve(addr string, port int,
+	opts ...Option) error {
 }
"""

JS_ARROW = """--- a/src/index.ts
+++ b/src/index.ts
@@ -1,2 +1,2 @@
-export const parse = (s: string) => s;
+export const parse = (s: string, strict: boolean) => s;
 export const keep = 1;
"""


def _claim(summary: str, diff: str):
    g = gate_diff_text(summary, diff, run=None, strict=False)
    cs = [c for c in g.claims if c.kind == "compat_claim"]
    assert len(cs) == 1, [c.kind for c in g.claims]
    return g, cs[0]


def test_the_licence_is_off_and_pinned():
    assert dg.COMPAT2_LICENSED is False
    assert dg._COMPAT_VERDICTS == ("UNCHECKABLE",)


def test_surface_drops_are_named_first_and_scaffolding_counted_beside_them():
    _, c = _claim("No breaking changes.", SURFACE_AND_SCAFFOLD)
    assert c.verdict == "UNCHECKABLE"
    assert c.why == ("compatibility claimed; the diff removes 1 public definition(s) from the surface, not "
                     "re-defined in the added lines: pkg/api.py: session; 1 more in test/example/internal code; "
                     "1 signature(s) changed")
    assert c.detail["compat2_candidate"] is True
    assert c.detail["surface_removed"] == 1
    assert [(r["name"], r["surface"]) for r in c.detail["removed"]] == [("session", True), ("helper", False)]
    assert c.detail["signature_changed"] == [{"path": "pkg/api.py", "language": "python", "name": "keep",
                                              "before": "x", "after": "x, y"}]


def test_scaffolding_only_is_not_a_candidate_and_says_so():
    _, c = _claim("Backward compatible.", SCAFFOLD_ONLY)
    assert c.verdict == "UNCHECKABLE"
    assert c.why == ("compatibility claimed; 2 public definition(s) removed, all in test/example/internal code: "
                     "tests/test_api.py: helper, examples/demo.ts: run")
    assert c.detail["compat2_candidate"] is False and c.detail["surface_removed"] == 0
    assert sorted(r["language"] for r in c.detail["removed"]) == ["js/ts", "python"]


def test_a_move_with_the_same_signature_is_neither_a_drop_nor_a_change():
    _, c = _claim("No breaking changes.", MOVE_SAME_SIGNATURE)
    assert c.detail["removed"] == [] and c.detail["signature_changed"] == []
    assert c.detail["compat2_candidate"] is False
    assert c.why.startswith("compatibility claimed; no public top-level definition removed")


def test_a_multiline_signature_compares_on_its_first_line():
    _, c = _claim("No functional changes.", MULTILINE_SIGNATURE)
    assert c.detail["removed"] == []
    assert c.detail["signature_changed"] == [{"path": "pkg/a.go", "language": "go", "name": "Serve",
                                              "before": "addr string,…", "after": "addr string, port int,…"}]
    assert c.why.endswith("; 1 signature(s) changed")


def test_a_js_arrow_export_reads_its_parameter_list():
    _, c = _claim("Non-breaking.", JS_ARROW)
    assert c.detail["signature_changed"] == [{"path": "src/index.ts", "language": "js/ts", "name": "parse",
                                              "before": "s: string", "after": "s: string, strict: boolean"}]


def test_the_scaffolding_rule_is_the_prereg_s():
    scaffold = ["tests/test_x.py", "pkg/internal/x.go", "a/b/foo_test.go", "lib/x.test.ts", "setup.py",
                "docs/conf.py", "scripts/run.py", "x/dev/y.js", "build/a.js", "app/spec/x.rb", "conftest.py",
                "src/__tests__/a.tsx", "examples/e.py", "cmd/root.go", "vendor/lib.go", "e2e/run.ts"]
    surface = ["src/api.py", "src/setup2.py", "pkg/x.go", "lib/index.ts", "testing_utils.py", "protest/x.py",
               "src/devices.py", "builder/x.py"]
    assert all(dg._COMPAT_SCAFFOLD.search(p) for p in scaffold), [p for p in scaffold if not dg._COMPAT_SCAFFOLD.search(p)]
    assert not any(dg._COMPAT_SCAFFOLD.search(p) for p in surface), [p for p in surface if dg._COMPAT_SCAFFOLD.search(p)]


def test_every_branch_is_uncheckable_while_unlicensed_and_the_detail_keys_are_constant():
    keys = {"removed", "languages", "surface_removed", "signature_changed", "compat2_candidate"}
    for sides in (None, {}, dg.parse_unified_diff_sides(SURFACE_AND_SCAFFOLD),
                  dg.parse_unified_diff_sides(SCAFFOLD_ONLY), dg.parse_unified_diff_sides(MOVE_SAME_SIGNATURE),
                  dg.parse_unified_diff_sides("--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-a\n+b\n")):
        verdict, why, extra = dg._compat_reading(sides)
        assert verdict == "UNCHECKABLE" and why.startswith("compatibility claimed;")
        assert set(extra) == keys


def test_the_candidate_would_be_the_only_accusation_if_licensed(monkeypatch):
    # the one place a second verdict can come from, exercised by flipping the flag in memory only
    monkeypatch.setattr(dg, "COMPAT2_LICENSED", True)
    v, _w, extra = dg._compat_reading(dg.parse_unified_diff_sides(SURFACE_AND_SCAFFOLD))
    assert v == "CONTRADICTED" and extra["compat2_candidate"] is True
    v, _w, extra = dg._compat_reading(dg.parse_unified_diff_sides(SCAFFOLD_ONLY))
    assert v == "UNCHECKABLE" and extra["compat2_candidate"] is False
    v, _w, _e = dg._compat_reading(dg.parse_unified_diff_sides(MOVE_SAME_SIGNATURE))
    assert v == "UNCHECKABLE"


def test_the_demo_is_unchanged():
    g = gate_diff_text(dg._DEMO_SUMMARY, dg._DEMO_DIFF, run=None, strict=False)
    assert not [c for c in g.claims if c.kind == "compat_claim"]
    assert sorted(c.kind for c in g.claims if c.verdict == "CONTRADICTED") == ["only_touches", "symbol_added", "tests_added"]

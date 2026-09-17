# -*- coding: utf-8 -*-
"""BIN-1 (PREREG_bin1_binary_files_2026_09_16, issue #118): a `diff --git` header with no
`---`/`+++` pair — a binary change, a mode-only change, a pure rename — registers its file in
`parse_unified_diff` and `parse_unified_diff_sides`. Files with hunks read exactly as before, and
the added-lines blob never changes, so `tests_added`, `symbol_added` and `compat_claim` cannot move."""
import json
from pathlib import Path

from styxx import diffgate as dg
from styxx.diffgate import gate_diff_text, parse_unified_diff, parse_unified_diff_sides

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "web" / "gate" / "differential" / "bin1_pairs.json"

TEXT_AND_BINARY = """\
diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -1 +1,2 @@
 x = 1
+y = 2
diff --git a/assets/logo.png b/assets/logo.png
new file mode 100644
index 0000000..3b18e51
Binary files /dev/null and b/assets/logo.png differ
"""

THREE_BINARIES = """\
diff --git a/a.png b/a.png
new file mode 100644
Binary files /dev/null and b/a.png differ
diff --git a/b.png b/b.png
index 1111111..2222222 100644
Binary files a/b.png and b/b.png differ
diff --git a/c.png b/c.png
deleted file mode 100644
Binary files a/c.png and /dev/null differ
"""

RENAME_AND_MODE = """\
diff --git a/docs/old.md b/docs/new.md
similarity index 100%
rename from docs/old.md
rename to docs/new.md
diff --git a/run.sh b/run.sh
old mode 100644
new mode 100755
"""


def test_a_binary_beside_a_text_file_is_counted():
    status, blob = parse_unified_diff(TEXT_AND_BINARY)
    assert status == {"src/app.py": "M", "assets/logo.png": "A"}
    assert blob == "y = 2"                       # the added-lines blob is untouched
    assert set(parse_unified_diff_sides(TEXT_AND_BINARY)) == {"src/app.py", "assets/logo.png"}
    assert parse_unified_diff_sides(TEXT_AND_BINARY)["assets/logo.png"] == ([], [])


def test_added_modified_and_deleted_binaries_get_their_statuses():
    status, blob = parse_unified_diff(THREE_BINARIES)
    assert status == {"a.png": "A", "b.png": "M", "c.png": "D"}
    assert blob == ""


def test_a_pure_rename_and_a_mode_change_are_files_too():
    status, _ = parse_unified_diff(RENAME_AND_MODE)
    assert status == {"docs/new.md": "M", "run.sh": "M"}


def test_a_truthful_count_over_a_binary_verifies_and_a_lie_over_it_is_caught():
    g = gate_diff_text("2 files changed. Only touches src/.", TEXT_AND_BINARY, run=None, strict=False)
    by = {c.kind: c for c in g.claims}
    assert by["files_changed_count"].verdict == "VERIFIED"
    assert by["only_touches"].verdict == "CONTRADICTED"
    assert "assets/logo.png" in by["only_touches"].why


def test_a_header_with_a_pair_is_registered_once_from_the_pair():
    diff = "diff --git a/x.py b/x.py\nnew file mode 100644\n--- /dev/null\n+++ b/x.py\n@@ -0,0 +1 @@\n+a = 1\n"
    assert parse_unified_diff(diff) == ({"x.py": "A"}, "a = 1")


def test_header_paths_split_same_names_at_the_middle_and_renames_by_regex():
    assert dg._header_paths("diff --git a/my file.txt b/my file.txt") == ("my file.txt", "my file.txt")
    assert dg._header_paths("diff --git a/old.png b/new.png") == ("old.png", "new.png")
    assert dg._header_paths('diff --git "a/sp ace.png" "b/sp ace.png"') == ("sp ace.png", "sp ace.png")


def test_the_demo_is_unchanged():
    g = gate_diff_text(dg._DEMO_SUMMARY, dg._DEMO_DIFF, run=None, strict=False)
    assert sorted(c.kind for c in g.claims if c.verdict == "CONTRADICTED") == ["only_touches", "symbol_added", "tests_added"]


def test_the_pinned_pairs_read_as_expected_on_the_python_side():
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    assert [p["id"] for p in pairs] == ["bin1:text-and-binary", "bin1:three-binaries", "bin1:rename-and-mode",
                                        "bin1:count-now-true", "bin1:prefix-now-caught", "bin1:quoted-path"]
    for p in pairs:
        g = gate_diff_text(p["summary"], p["diff"], run=None, strict=False)
        got = [[c.kind, c.verdict, c.why] for c in g.claims]
        assert got == p["expect"]["claims"], (p["id"], got)
        assert g.verdict == p["expect"]["verdict"], p["id"]

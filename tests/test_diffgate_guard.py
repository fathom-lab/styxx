# -*- coding: utf-8 -*-
"""NOTE_path2_eleventh_pass_2026_09_28: the licensed-difference rule at the verdict.

(a) THE REFERENCE. `main`'s reader is vendored unchanged: `styxx/_diffgate_ref.py` is `origin/main`'s
    `styxx/diffgate.py` and `web/gate/diffgate_ref.js` is `origin/main`'s `web/gate/diffgate.js`, byte for byte.
    The pins below are the sha256s of those two blobs (the file 7.48.0 ships, and its port).
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF_PY = ROOT / "styxx" / "_diffgate_ref.py"
REF_JS = ROOT / "web" / "gate" / "diffgate_ref.js"
INSTRUMENT = ROOT / "styxx" / "diffgate.py"

# origin/main 2a6ce0a3: styxx/diffgate.py (the file styxx 7.48.0 ships) and web/gate/diffgate.js, LF.
MAIN_DIFFGATE_PY_SHA256 = "9b620e00a19464589308a987819894ae7cc3c111c66a5f8a457a84b8a6c604eb"
MAIN_DIFFGATE_JS_SHA256 = "06688702999cdabe763265722a0ac14d4b9ffb40d0efcbb32339eba89f00c141"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- (a) the reference is main's reader, byte for byte ----------------------------------------------------------

def test_the_python_reference_is_mains_diffgate_byte_for_byte():
    """No normalisation: .gitattributes marks the file -text, so every checkout holds the blob's LF bytes."""
    assert b"\r\n" not in REF_PY.read_bytes()
    assert _sha(REF_PY) == MAIN_DIFFGATE_PY_SHA256


def test_the_javascript_reference_is_mains_port_byte_for_byte():
    assert b"\r\n" not in REF_JS.read_bytes()
    assert _sha(REF_JS) == MAIN_DIFFGATE_JS_SHA256


def test_both_references_are_pinned_against_a_checkout_rewriting_them():
    lines = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "styxx/_diffgate_ref.py -text" in lines
    assert "web/gate/diffgate_ref.js -text" in lines


def test_the_python_reference_imports_as_its_own_module():
    """Its one relative import (DECLARE-1's reader) resolves inside the package, as it does on main."""
    import styxx._diffgate_ref as ref
    import styxx.declare as declare

    assert ref.__name__ == "styxx._diffgate_ref"
    assert ref._declaration_pass is declare.declaration_pass
    g = ref.gate_diff_text("Modified src/a.py.", "--- a/src/a.py\n+++ b/src/a.py\n@@ -1 +1 @@\n-a\n+b\n")
    assert [(c.kind, c.verdict) for c in g.claims] == [("file_touched", "VERIFIED")]


def _defs(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    return {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def test_the_tests_pass_leg_is_mains_function_unchanged():
    """The guard evaluates the reference without --run and --evidence (a command is never run twice) and reads the
    tests_pass verdict of a measured reference as this function's result: that is sound only while the function is
    main's, byte for byte."""
    ref, new = _defs(REF_PY), _defs(INSTRUMENT)
    for name in ("_tests_pass_verdict", "_evidence_leg", "_run_leg", "DiffClaim", "DiffGate"):
        assert ref[name] == new[name], name

"""CONTRADICTED is printed as CONTRADICTED, with its reason (step zero, item 2).

The demo, the bookmarklet and the hooks printed a CONTRADICTED verdict as ``[LIE]``, and the demo closed with
"this summary would fail your CI with each lie named". A CONTRADICTED verdict says the diff does not show what a
template read in a sentence; it says nothing about why the sentence was written, and the kinds that produce it were
measured well below the lab's 0.95 floor (``only_touches`` at 0.25, README). Every surface that prints a verdict now
prints its name and the reason beside it. The hooks' own tests pin their output; this module pins the demo, the
bookmarklet as shipped, and the documents that quote those outputs.
"""
from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

from styxx import diffgate

ROOT = Path(__file__).resolve().parent.parent


def _demo_output() -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert diffgate._demo() == 0
    return out.getvalue()


def test_the_demo_prints_contradicted_with_its_reason():
    text = _demo_output()
    assert "  [CONTRADICTED] tests_added          diff adds 1 test functions, claim says 3\n" in text
    assert "  [CONTRADICTED] symbol_added         added lines do NOT define function 'backoff'\n" in text
    assert "  [CONTRADICTED] only_touches         paths outside 'src': " in text
    assert "  [ok ] file_touched         diff status 'M' for 'src/retry.py'\n" in text
    assert "\nverdict: FAIL — 3 claim(s) CONTRADICTED by the diff, each with its reason above.\n" in text


def test_the_demo_names_no_lie():
    text = _demo_output()
    assert re.search(r"\b(?:lie|lies|lying|liar)\b", text, re.I) is None


# the code that prints a verdict, and the shipped bookmarklet built from it
SURFACES = [
    "styxx/diffgate.py",
    "styxx/diffgate_hook.py",
    "integrations/git/commit-msg",
    "integrations/claude-code/diffgate-hook/pretool.py",
    "integrations/cursor/diffgate-hook/before_shell.py",
    "web/gate/bookmarklet_ui.js",
    "web/gate/bookmarklet_src.js",
    "web/gate/bookmarklet.min.js",
    "web/gate/bookmarklet.href.txt",
]


@pytest.mark.parametrize("rel", SURFACES)
def test_no_surface_carries_a_lie_label(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    for label in ('"LIE"', "[LIE]", "each lie"):
        assert label not in text, f"{rel} still carries {label}"


@pytest.mark.parametrize("rel", ["web/gate/bookmarklet_ui.js", "web/gate/bookmarklet.min.js",
                                 "web/gate/bookmarklet.href.txt"])
def test_the_bookmarklet_prints_contradicted(rel):
    assert "[CONTRADICTED]" in (ROOT / rel).read_text(encoding="utf-8")


# the documents that quote what those surfaces print
DOCS = [
    "README.md",
    "web/gate/README.md",
    "integrations/git/README.md",
    "integrations/pre-commit/README.md",
    "integrations/claude-code/diffgate-hook/README.md",
    "integrations/codex/diffgate-hook/README.md",
    "integrations/cursor/diffgate-hook/README.md",
    "integrations/gemini-cli/diffgate-hook/README.md",
]


@pytest.mark.parametrize("rel", DOCS)
def test_no_document_quotes_a_lie_label(rel):
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert "[LIE]" not in text and "with each lie named" not in text, rel

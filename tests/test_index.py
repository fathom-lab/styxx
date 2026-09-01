"""papers/INDEX.md is the gate that stops the lab rediscovering its own results.

The 2026-09-01 audit measured the failure it exists to prevent: 47 arcs carrying 13
distinct questions, arc-to-arc citation density 0.041, fourteen arcs citing no other and
ten cited by none. Mention-versus-use was named in five arcs on a single day and the
synthesis three months later cataloguing ten instances cites none of them.

A map nothing enforces rots. These tests enforce the half that can be enforced.

LOAD-BEARING: test_every_arc_has_a_row. An arc directory that exists with no INDEX row is
exactly how the eleventh rediscovery happens — someone opens work without seeing the ten
that came before. That test is the whole point of this file.

WHAT IS DELIBERATELY NOT TESTED: the one-sentence terminal claim, the idea index and the
receipts index. Those are authored judgments. Validating them mechanically would mean
generating the thing we claim to check, which is the handed-target defect this lab spent a
week measuring. The guarantee is narrow and stated in build_index.py: you cannot add an arc
without adding a row, and you cannot invent a tag. What the row SAYS is on the author.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "papers" / "INDEX.md"
BUILD = ROOT / "papers" / "build_index.py"


def run_check():
    r = subprocess.run([sys.executable, str(BUILD)], capture_output=True, text=True,
                       cwd=str(ROOT), encoding="utf-8", errors="replace")
    return r


def test_the_index_exists_under_the_name_the_briefs_cite():
    """Both 2026-09-01 research briefs open with 'read papers/INDEX.md first'."""
    assert INDEX.exists(), (
        "papers/INDEX.md is missing. Every arc opened from now declares question: and "
        "cites: against it; without the file the declaration cannot be made.")


def test_every_arc_has_a_row():
    """LOAD-BEARING. An arc with no row is an arc opened without seeing its priors."""
    r = run_check()
    assert r.returncode == 0, (
        "papers/INDEX.md is out of step with the tree:\n" + r.stdout + r.stderr)


def test_the_checker_writes_its_receipt():
    run_check()
    rec = json.loads((ROOT / "papers" / "index_check.json").read_text(encoding="utf-8"))
    assert rec["ok"] is True
    assert rec["arc_directories"] == rec["rows"]
    assert rec["arc_directories"] > 40, "the arc count collapsed — check NOT_ARCS"


def test_the_receipt_says_what_it_does_not_check():
    """The narrow guarantee has to travel with the check, or a reader over-reads it."""
    rec = json.loads((ROOT / "papers" / "index_check.json").read_text(encoding="utf-8"))
    nc = rec["not_checked"].lower()
    assert "authored" in nc and "never validated" in nc
    assert "idea index" in nc


@pytest.mark.parametrize("vocab_key,sample", [("tags", "handed-target"),
                                              ("statuses", "NEGATIVE-RESULT")])
def test_the_closed_vocabularies_are_closed(vocab_key, sample):
    """A tag invented ad hoc makes the idea index unsearchable, which is the failure mode."""
    src = BUILD.read_text(encoding="utf-8")
    assert sample in src, f"{sample} vanished from the {vocab_key} vocabulary"


def test_negative_result_is_a_first_class_status():
    """Several of this lab's best papers are negative; a status vocabulary that treats
    them as failures would push the next one toward not being published."""
    src = BUILD.read_text(encoding="utf-8")
    assert "NEGATIVE-RESULT" in src
    txt = INDEX.read_text(encoding="utf-8")
    assert "NEGATIVE-RESULT" in txt, "the index must define the status it uses"

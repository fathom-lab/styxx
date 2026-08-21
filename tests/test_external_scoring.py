# -*- coding: utf-8 -*-
"""SP-EXT scoring harness, and what it immediately said about our own screen.

The first externally-anchored case is missed by `styxx.flattering`, on two
independent grounds, and the second of them is **C4** — the polarity-from-name
failure that the flattering adjudication named abstractly before SP-EXT existed:

    `pass_rate` starts with "pass_", so it matches the BOOLEAN-PREDICATE
    vocabulary, and under that polarity the float 1.0 is not flattering at all.
    But pass_rate is a RATE. High is the good end.

`flattering.py` is frozen by its own preregistration and is not being edited to
catch this. The miss is pinned instead, so a future change that catches it has to
come with a re-run rather than a quiet green tick.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from benchmarks.silent_pass.external import (
    EXT_CAVEAT, load_external, score_external)

CLONES = Path("C:/Users/heyzo/AppData/Local/Temp/spcorpus")
_have_clones = (CLONES / "giskard" / ".git").exists()
needs_clones = pytest.mark.skipif(
    not _have_clones, reason="upstream clones not present; see benchmarks/silent_pass/EXTERNAL.md")


def test_cases_load_with_their_upstream_anchors():
    cases = load_external()
    assert cases
    for c in cases:
        assert "/" in c.repo and c.fix_commit and c.module


def test_the_caveat_forbids_quoting_a_rate():
    assert "lower bound" in EXT_CAVEAT.lower()
    assert "never be quoted as a rate" in EXT_CAVEAT
    assert "RECALL ONLY" in EXT_CAVEAT


def test_a_case_with_no_defect_line_is_unscored_not_missed():
    """Scoring a case you could not localise as a failure would be the same error
    this corpus documents, and a corpus that does it has no standing to complain."""
    cases = load_external()
    for c in cases:
        c.defect_line = None
    r = score_external(lambda s, f: {1}, clone_root=CLONES, cases=cases)
    assert r.n_unavailable == len(cases)
    assert r.missed == []
    assert r.recall is None
    assert "NOT A CLEAN RESULT" in r.render()


def test_missing_clones_are_unscored_not_missed(tmp_path):
    r = score_external(lambda s, f: {1}, clone_root=tmp_path)
    assert r.recall is None and r.missed == []
    assert "SCORED NOTHING" in r.render()


@needs_clones
def test_harness_fetches_upstream_prefix_source_and_localises():
    def naive(src, filename):
        return {i for i, l in enumerate(src.splitlines(), 1)
                if re.match(r"\s*return\s+(1\.0|0\.0|True)\s*$", l)}

    r = score_external(naive, clone_root=CLONES)
    assert r.n_cases >= 1
    assert "SPX-2026-0001" in r.caught, (
        "the harness must reach upstream pre-fix source; if this fails the "
        "clone is stale or the recorded defect_line is wrong")


@needs_clones
def test_a_detector_that_does_nothing_scores_zero_not_none():
    r = score_external(lambda s, f: set(), clone_root=CLONES)
    assert r.recall == 0.0
    assert r.n_cases >= 1


@needs_clones
def test_PINNED_flattering_misses_the_first_external_case():
    """Two independent grounds, both worth keeping visible:

    1. `_looks_sizey` does not recognise the name `denominator`, so
       `if denominator == 0:` is not read as an emptiness test at all.
    2. `_polarity("pass_rate")` returns `ok_bool` — it starts with "pass_" — and
       under a boolean-predicate polarity the float 1.0 is not flattering. C4.

    If this ever passes, `flattering` changed, and its published numbers
    (10% recall, 0/8 external) no longer describe the shipped code.
    """
    from styxx.flattering import scan_source

    def det(src, filename):
        try:
            return {h.line for h in scan_source(src, filename)}
        except SyntaxError:
            return set()

    r = score_external(det, clone_root=CLONES)
    assert "SPX-2026-0001" in r.missed


def test_the_two_grounds_of_that_miss_are_what_we_say_they_are():
    """Diagnosis pinned separately from the miss, because 'it misses' and 'it
    misses FOR THIS REASON' are different claims and only the second is useful."""
    import ast

    from styxx.flattering import _is_flattering, _looks_sizey, _polarity

    assert _looks_sizey(ast.Name(id="denominator", ctx=ast.Load())) is False
    assert _polarity("pass_rate") == "ok_bool"
    assert _is_flattering(ast.Constant(1.0), "ok_bool") is False
    assert _is_flattering(ast.Constant(1.0), "high_is_good") is True

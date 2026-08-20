# -*- coding: utf-8 -*-
"""SILENT-PASS — the benchmark's own honesty properties.

A benchmark that can silently score nothing, or that lets a no-op detector look
good, is the failure it exists to document. These pin the properties that stop
that, and they matter more than the numbers.
"""
import pytest

from benchmarks.silent_pass import CAVEAT, Case, load_cases, score


def test_every_case_is_wellformed_and_unique():
    cases = load_cases()
    assert len(cases) >= 20
    assert len({c.id for c in cases}) == len(cases), "duplicate case ids"
    for c in cases:
        assert c.subtype.startswith("SP-")
        assert c.module.endswith(".py")
        assert c.defect_line > 0
        # every field that makes a case READABLE, not just machine-scorable
        for txt in (c.what_failed, c.what_was_returned,
                    c.why_it_reads_healthy, c.consumer, c.fix):
            assert txt and len(txt) > 10, f"{c.id} is missing its explanation"


def test_pre_fix_source_is_fetched_from_history_not_copied():
    """The corpus stores a commit, not a snapshot, so it cannot drift from the
    history it cites."""
    case = load_cases()[0]
    src = case.pre_fix_source()
    if src is None:
        pytest.skip("shallow clone — pre-fix history unavailable")
    assert "def " in src


def test_a_detector_that_finds_nothing_scores_zero_not_none():
    result = score(lambda src, name: set())
    if result.recall is None:
        pytest.skip("shallow clone — no cases scorable")
    assert result.recall == 0.0
    assert result.n_caught == 0


def test_an_unloadable_case_is_unscored_never_a_miss():
    """Counting a case you could not load as a FAILURE would be exactly the
    error this corpus documents: an absent measurement reported as a result."""
    bogus = Case(id="SP-FAKE", subtype="SP-1", module="styxx/does_not_exist.py",
                 fix_commit="deadbeef", defect_line=1,
                 what_failed="n/a for this test case",
                 what_was_returned="n/a for this test case",
                 why_it_reads_healthy="n/a for this test case",
                 consumer="n/a for this test case",
                 fix="n/a for this test case")
    r = score(lambda src, name: {1}, cases=[bogus])
    assert r.n_unavailable == 1
    assert r.missed == []
    assert r.recall is None, "no cases scored is not a recall of zero"
    assert "NO CASES SCORED" in r.render()
    assert "not a score of zero" in r.render()


def test_every_result_carries_the_recall_only_caveat():
    """A recall number quoted without its precision caveat is the fire-rate
    wearing the antibody's name."""
    r = score(lambda src, name: set())
    assert CAVEAT in r.render() or "NO CASES SCORED" in r.render()
    assert r.as_dict()["caveat"] == CAVEAT
    assert "cannot measure precision" in CAVEAT


def test_a_flag_everything_detector_exposes_the_corpus_limit():
    """The corpus holds only true positives, so flagging every line scores
    perfectly. That is the documented reason recall alone is not a verdict."""
    def flag_everything(src, name):
        return set(range(1, src.count("\n") + 2))

    r = score(flag_everything)
    if r.recall is None:
        pytest.skip("shallow clone — no cases scorable")
    assert r.recall == 1.0, "and this detector is worthless — hence the caveat"


def test_shipped_detectors_are_complementary_not_redundant():
    """absence covers crash/truthiness shapes; loops covers self-confirmation.
    Neither subsumes the other, and the union beats both — that is the argument
    for shipping two instruments instead of one."""
    from styxx.absence import scan_source as absence_scan
    from styxx.loops import scan_source as loops_scan

    def absence_det(src, name):
        try:
            return {f.line for f in absence_scan(src, name)}
        except SyntaxError:
            return set()

    def loops_det(src, name):
        try:
            v = loops_scan(src, name)
        except SyntaxError:
            return set()
        return {d.line for d in v.derivations} | {t.line for t in v.trust}

    a = score(absence_det)
    l = score(loops_det)
    if a.recall is None or l.recall is None:
        pytest.skip("shallow clone — no cases scorable")

    both = score(lambda s, n: absence_det(s, n) | loops_det(s, n))
    assert both.n_caught > a.n_caught and both.n_caught > l.n_caught
    # each finds something the other misses
    assert set(a.caught) - set(l.caught)
    assert set(l.caught) - set(a.caught)
    # SP-7 (self-confirming) is loops' reason to exist
    assert l.by_subtype.get("SP-7", {}).get("caught", 0) >= 2

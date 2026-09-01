"""V12 MIRROR-SUM: the gates of PREREG_mirror_sum_2026_08_31, as tests.

G-S1 fixture logic on synthetic twins of the real specimens, G-S2's six-mutant
battery (one binding mutant kills the clause), and the rescue-only invariant.
G-S3's absolute corpus A/B runs in its own harness.
"""
from __future__ import annotations

import json

import pytest

from styxx.certify import certify_doc


@pytest.fixture
def mk(tmp_path):
    def _mk(text, receipt):
        doc = tmp_path / "d.md"
        doc.write_text(text + "\n", encoding="utf-8")
        rec = tmp_path / "r.json"
        rec.write_text(json.dumps(receipt), encoding="utf-8")
        return certify_doc(doc, [rec])
    return _mk


def _tok(cert, token):
    return [e for e in cert["ledger"] if str(e["token"]) == token]


MIRROR = {"arms": {"positive": {"valid": 58, "claims": 26},
                   "negative": {"valid": 57, "claims": 20}}}
POOLED = "The pooled null rate is 39/115 across both arms."


# ------------------------------------------------------------------ G-S1 logic

def test_pooled_sum_flips_to_derived(mk):
    c = mk(POOLED, MIRROR)
    es = _tok(c, "115")
    assert es and es[0]["status"] == "VERIFIED"
    assert es[0]["receipt_ref"] == "derived-sum:58+57=115@r.json:arms.*.valid"
    assert es[0]["epistemics"]["branch"] == "derived"


def test_bare_pair_58_not_touched(mk):
    """58 IS a leaf — its accusation is the count-binding filter's, not ours.
    In this minimal context the ladder abstains (never bound); the gate's
    requirement is that the clause does not bind it — the real must-stay-
    accused check runs against the actual paper in the G-S3 re-certification."""
    c = mk("A rule that would false-accuse 32 of every 58 decimals is dead on arrival.",
           MIRROR)
    es = _tok(c, "58")
    assert es and es[0]["status"] != "VERIFIED"
    assert "derived-sum" not in str(es[0].get("receipt_ref"))


def test_uniform_sum_refused(mk):
    """Nine seats scoring 1 each sum to 9 — indistinguishable from counting.
    The quoted-9 coincidence from the prereg grounding, reproduced exactly:
    the clause must never bind it (same G-S3 caveat as above)."""
    seats = {"seat_validity": {f"p{p}-seat{s}": {"score": 1}
                               for p in (1, 2, 3) for s in (1, 2, 3)}}
    c = mk("The cycle shipped 9 new tests.", seats)
    es = _tok(c, "9")
    assert es and es[0]["status"] != "VERIFIED"
    assert "derived-sum" not in str(es[0].get("receipt_ref"))


# --------------------------------------------------------------- G-S2 mutants

def test_mutant_value_off_by_one(mk):
    c = mk("The pooled null rate is 39/116 across both arms.", MIRROR)
    es = _tok(c, "116")
    assert es and es[0]["status"] == "UNGROUNDED"


def test_mutant_third_sibling_breaks_sum(mk):
    r = {"arms": {"positive": {"valid": 58}, "negative": {"valid": 57},
                  "neutral": {"valid": 4}}}
    c = mk(POOLED, r)   # exhaustive sum is now 119, not 115
    es = _tok(c, "115")
    assert es and es[0]["status"] == "UNGROUNDED"


def test_mutant_subset_sum_refused(mk):
    """115 = 58+57 exists only as a SUBSET of three siblings — must refuse."""
    r = {"arms": {"a": {"valid": 58}, "b": {"valid": 57}, "c": {"valid": 100}}}
    c = mk(POOLED, r)
    es = _tok(c, "115")
    assert es and es[0]["status"] == "UNGROUNDED"


def test_mutant_mixed_fields_refused(mk):
    r = {"arms": {"positive": {"valid": 58}, "negative": {"other": 57}}}
    c = mk(POOLED, r)
    es = _tok(c, "115")
    assert es and es[0]["status"] == "UNGROUNDED"


def test_mutant_float_addends_refused(mk):
    r = {"arms": {"positive": {"valid": 57.5}, "negative": {"valid": 57.5}}}
    c = mk(POOLED, r)
    es = _tok(c, "115")
    assert es and es[0]["status"] == "UNGROUNDED"


def test_mutant_single_child_refused(mk):
    r = {"arms": {"only": {"valid": 115}}}
    c = mk("An unrelated pooled figure of 115 with no leaf.", r)
    es = _tok(c, "115")
    # single-child "sum" must not bind; direct leaf match may bind through the
    # ordinary ladder — assert only that no derived-sum ref appears.
    assert es and "derived-sum" not in str(es[0].get("receipt_ref"))


# ----------------------------------------------------------- rescue-only law

def test_rescue_only_never_reattributes(mk):
    """A token the ladder already grounds keeps its original ref."""
    r = {"eval": {"total": 115}, "arms": {"a": {"n": 58}, "b": {"n": 57}}}
    c = mk("The eval total is 115 items.", r)
    es = _tok(c, "115")
    assert es and es[0]["status"] == "VERIFIED"
    assert "derived-sum" not in str(es[0]["receipt_ref"])


def test_flag_off_restores_accusation(mk, monkeypatch):
    import sys
    cert = sys.modules["styxx.certify"]  # robust to package-attr rebinding by other tests
    monkeypatch.setattr(cert, "V12_SUM_COHERENCE", False)
    c = mk(POOLED, MIRROR)
    es = _tok(c, "115")
    assert es and es[0]["status"] == "UNGROUNDED"

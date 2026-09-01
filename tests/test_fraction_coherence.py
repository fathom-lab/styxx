"""V11 FRACTION-COHERENCE: the gates of PREREG_fraction_coherence_2026_08_31, as tests.

G-F1 fixtures (coherent specimens flip, non-coherent must NOT), G-F2 the five-mutant battery
(one verifying mutant kills the clause), and the rescue-only invariant. G-F3's corpus A/B
runs in its own harness; its absolute condition is asserted there.
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


def _tok(cert, token, status=None):
    es = [e for e in cert["ledger"] if str(e["token"]) == token]
    if status:
        es = [e for e in es if e["status"] == status]
    return es


RECEIPT = {"arms": {"positive": {"claims": 44, "valid": 59},
                    "negative": {"claims": 13, "valid": 57}}}
LINE = "The precision was 44/59 (0.7458) on the held-out run."


# ── G-F1: the coherent class flips; the excluded classes must not ────────────────

def test_coherent_fraction_operands_bind_jointly(mk):
    cert = mk(LINE, RECEIPT)
    for tok in ("44", "59"):
        e = _tok(cert, tok)[0]
        assert e["status"] == "VERIFIED", tok
        assert str(e["receipt_ref"]).startswith("derived-fraction:44/59=0.7458@")
        assert "arms.positive" in e["receipt_ref"], "must name the common parent"
        assert e["epistemics"]["branch"] == "derived"


def test_bare_count_pair_without_ratio_stays_accused(mk):
    """'32 of every 58' — no same-line ratio, out of scope by freeze; the honest boundary."""
    cert = mk("The precision rule would accuse 32 of every 58 tokens.",
              {"a": {"x": 32, "y": 58}})
    assert not _tok(cert, "58", "VERIFIED"), "no coherence, no rescue"


def test_quoted_fragment_stays_accused(mk):
    cert = mk('The recall miss was *"9 new tests."* as quoted.', {"n": {"m": 9}})
    assert not _tok(cert, "9", "VERIFIED")


# ── G-F2: the five-mutant battery — a single verifying mutant kills the clause ───

def test_mutant_numerator_breaks_coherence(mk):
    cert = mk("The precision was 45/59 (0.7458) on the held-out run.", RECEIPT)
    assert not any(str(e["receipt_ref"]).startswith("derived-fraction")
                   for e in cert["ledger"] if e["status"] == "VERIFIED")


def test_mutant_denominator_breaks_coherence(mk):
    cert = mk("The precision was 44/60 (0.7458) on the held-out run.", RECEIPT)
    assert not any(str(e["receipt_ref"]).startswith("derived-fraction")
                   for e in cert["ledger"] if e["status"] == "VERIFIED")


def test_mutant_ratio_breaks_coherence(mk):
    cert = mk("The precision was 44/59 (0.7460) on the held-out run.", RECEIPT)
    assert not any(str(e["receipt_ref"]).startswith("derived-fraction")
                   for e in cert["ledger"] if e["status"] == "VERIFIED")


def test_mutant_split_subtrees_fail_joint_binding(mk):
    """Operands present in receipts but never under one parent: no rescue."""
    cert = mk(LINE, {"a": {"claims": 44}, "b": {"valid": 59}})
    assert not any(str(e["receipt_ref"]).startswith("derived-fraction")
                   for e in cert["ledger"] if e["status"] == "VERIFIED")


def test_mutant_absent_operands_fail(mk):
    cert = mk(LINE, {"arms": {"positive": {"claims": 40, "valid": 50}}})
    assert not any(str(e["receipt_ref"]).startswith("derived-fraction")
                   for e in cert["ledger"] if e["status"] == "VERIFIED")


# ── rescue-only: the clause can never accuse, and coherence failure changes nothing ─

def test_rescue_only_no_new_accusations(mk):
    """With and without a bindable pair, UNGROUNDED counts may only fall, never rise."""
    with_pair = mk(LINE, RECEIPT)
    without = mk(LINE, {"unrelated": 1})
    assert with_pair["counts"]["UNGROUNDED"] <= without["counts"]["UNGROUNDED"]


def test_epistemics_partition_carries_derived_fraction(mk):
    cert = mk(LINE, RECEIPT)
    s = cert["epistemics_summary"]
    assert sum(s["verified"]["derived"].values()) >= 2, \
        "both operands must land in the derived partition"

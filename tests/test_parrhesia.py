# -*- coding: utf-8 -*-
"""Tests for styxx.parrhesia — verifiable honesty receipts.

The load-bearing properties: (a) a receipt VERIFIES by re-derivation when the presented message
matches; (b) it FAILS when the message is swapped (content-addressing) or the audit no longer
reproduces; (c) the audit discriminates sycophantic/overclaiming drafts from honest-but-confident
ones (no alarm fatigue); (d) the receipt is honest about its own scope (register, not truth).
"""
import json


import styxx
from styxx import parrhesia
from styxx.parrhesia import (
    audit_text, issue_receipt, verify_receipt, HonestyReceipt, SCHEMA,
)

P = "is styxx ready to ship?"
SOBER = ("Partly. The core instruments are shipped and tested, but the cross-vendor numbers are "
         "imprecise on single runs and the agent wiring is not built yet.")
HYPE = ("Absolutely! styxx is 100% production-ready, guaranteed flawless, the best integrity layer "
        "ever built — you're totally right, this will definitely change everything!")


# --- public surface ---------------------------------------------------------

def test_public_surface_resolves():
    assert hasattr(styxx, "parrhesia")
    assert styxx.issue_receipt is parrhesia.issue_receipt
    assert styxx.verify_receipt is parrhesia.verify_receipt
    assert styxx.HonestyReceipt is parrhesia.HonestyReceipt


# --- audit discriminates (no alarm fatigue) ---------------------------------

def test_audit_discriminates_hype_from_sober():
    assert audit_text(P, HYPE).should_revise is True
    assert audit_text(P, SOBER).should_revise is False   # honest-but-confident must PASS

def test_audit_is_deterministic():
    a, b = audit_text(P, HYPE), audit_text(P, HYPE)
    assert (a.should_revise, a.sycophancy, a.reasons) == (b.should_revise, b.sycophancy, b.reasons)

def test_hype_reasons_nonempty_and_directive():
    a = audit_text(P, HYPE)
    assert a.reasons and a.revision_directive
    assert audit_text(P, SOBER).revision_directive == ""


# --- receipts verify by re-derivation ---------------------------------------

def test_issue_verify_roundtrip_sober():
    r = issue_receipt(P, SOBER, agent_id="darkflobi", timestamp="2026-06-23T00:00:00Z")
    assert isinstance(r, HonestyReceipt) and r.schema == SCHEMA
    assert r.passed is True and r.should_revise is False
    v = verify_receipt(r, P, SOBER)
    assert v.status == "VERIFIED" and bool(v) is True

def test_hype_gets_honest_failed_receipt_that_still_verifies():
    r = issue_receipt(P, HYPE, agent_id="darkflobi")
    assert r.passed is False and r.reasons
    # the receipt honestly records the failure, and that record is itself verifiable
    assert verify_receipt(r, P, HYPE).status == "VERIFIED"

def test_tamper_message_swap_fails():
    r = issue_receipt(P, SOBER, agent_id="darkflobi")
    v = verify_receipt(r, P, HYPE)   # attacker presents the receipt with a different message
    assert v.status == "FAILED"
    assert v.message_digest_match is False
    assert bool(v) is False

def test_tamper_verdict_flip_fails():
    # forge a receipt that claims a hype message passed
    r = issue_receipt(P, HYPE, agent_id="x").to_dict()
    r["verdict"]["should_revise"] = False
    r["verdict"]["passed_register_audit"] = True
    assert verify_receipt(r, P, HYPE).audit_reproduces is False   # re-run exposes the forgery

def test_dict_roundtrip_and_accepts_dict():
    r = issue_receipt(P, SOBER, agent_id="a", timestamp="t")
    d = r.to_dict()
    assert json.loads(json.dumps(d))  # JSON-serializable
    assert HonestyReceipt.from_dict(d).message_sha256 == r.message_sha256
    assert verify_receipt(d, P, SOBER).status == "VERIFIED"   # verify accepts dict form


# --- the receipt is honest about its own scope ------------------------------

def test_receipt_scope_is_register_not_truth():
    r = issue_receipt(P, SOBER, agent_id="a")
    assert "NOT a claim of truth" in r.certifies
    d = r.to_dict()
    assert "register" in d["certifies"].lower()
    assert d["auditor"]["styxx_version"] == styxx.__version__

"""Regression tests for the OATH v0.8 float field-binding clause — SHIPPED OFF (CLOSED_NEGATIVE).

The clause promotes the v0.6.2 float stem test from attribution to status level: a float claim at
1..3 fractional digits whose value-matches all sit on receipt paths unrelated to its binding
context is demoted VERIFIED -> ABSTAIN with reason ``unbound-field:<receipt>:<path>``.

It cleared every mechanical bar and died on G4 — of 40 sampled demotions, 30 destroyed a GENUINE
binding against a bar of 12 (`V08_COVERAGE_DESTRUCTIVE`), because prose names a measurement
narratively while the receipt field that holds it is structural. See
papers/closed-model-frontier/PREREG_oath_v08_float_field_binding_2026_08_23.md and
RESULT_oath_v08_float_field_binding_CLOSED_NEGATIVE_2026_08_23.md.

Two things are locked here:

  * the SHIPPED DEFAULT is OFF and inert — a future edit cannot silently switch the corpus over;
  * invariant **I1**, which is why the clause was allowed into the tree at all: when enabled it can
    only move VERIFIED -> ABSTAIN, so it can neither create nor remove an UNGROUNDED and no
    certificate can flip HELD -> FAILED. I1 holds by ladder construction, which is exactly why the
    prereg asserts it here instead of gating on it — a leg that cannot fail must not gate.
"""
import importlib
import json

import pytest

# importlib, not `import styxx.certify as C`: the package attribute `styxx.certify` is
# the provenance FUNCTION (styxx/__init__.py), and `import ... as` binds the attribute
# when it exists — module alone, function mid-suite. Same class ae45aaa fixed for v0.9;
# this file was the last one still binding the attribute (CI run 32726760278, 9 failed).
C = importlib.import_module("styxx.certify")
from styxx.certify import certify_doc  # noqa: E402

DOC = """# t

a preamble sentence so nothing lands at a line start.

The whole-stack read floors at 0.616 under the strongest attack.
"""
# `0.616` value-matches only a leaf whose path shares no stem with the sentence, while the receipt
# DOES carry a path the sentence names (`read_*`) — the NAMEABLE precondition for a demotion.
RECEIPT = {"points": [{"naive_relock_auroc": 0.616}], "read_summary": {"clean": 0.954}}


@pytest.fixture
def certify(tmp_path):
    def _run(flag, doc_text=DOC, receipt_obj=RECEIPT):
        doc = tmp_path / "d.md"
        doc.write_text(doc_text, encoding="utf-8")
        rp = tmp_path / "r.json"
        rp.write_text(json.dumps(receipt_obj), encoding="utf-8")
        original = C.V08_FLOAT_FIELD_BINDING
        C.V08_FLOAT_FIELD_BINDING = flag
        try:
            return certify_doc(doc, [rp])
        finally:
            C.V08_FLOAT_FIELD_BINDING = original
    return _run


def _entry(cert, token):
    return next((e for e in cert["ledger"] if e["token"] == token), None)


def test_shipped_default_is_off():
    """CLOSED_NEGATIVE: the clause must not be live on the corpus without a new prereg."""
    assert C.V08_FLOAT_FIELD_BINDING is False


def test_disabled_clause_is_inert(certify):
    """With the flag OFF the claim keeps the plain value-match verdict (G5 severability, in the small)."""
    e = _entry(certify(False), "0.616")
    assert e["status"] == "VERIFIED"
    assert not str(e["receipt_ref"]).startswith("unbound-field:")


def test_enabled_clause_demotes_to_abstain_with_named_reason(certify):
    e = _entry(certify(True), "0.616")
    assert e["status"] == "ABSTAIN"
    assert e["receipt_ref"].startswith("unbound-field:")
    assert "naive_relock_auroc" in e["receipt_ref"]


def test_stem_bound_claim_is_untouched(certify):
    """A claim whose context names its own field keeps VERIFIED in BOTH arms."""
    doc = DOC.replace("The whole-stack read floors at 0.616",
                      "The naive relock auroc floors at 0.616")
    for flag in (False, True):
        assert _entry(certify(flag, doc_text=doc), "0.616")["status"] == "VERIFIED"


def test_unnameable_claim_is_spared(certify):
    """NAMEABLE gate: with no path the sentence could bind to, demoting buys nothing."""
    receipt = {"points": [{"naive_relock_auroc": 0.616}]}
    for flag in (False, True):
        assert _entry(certify(flag, receipt_obj=receipt), "0.616")["status"] == "VERIFIED"


def test_high_precision_float_is_out_of_scope(certify):
    """The clause stops at V08_FIELD_BIND_MAX_DECIMALS, keeping it disjoint from the v0.7 rule."""
    doc = DOC.replace("0.616", "0.6161616")
    receipt = {"points": [{"naive_relock_auroc": 0.6161616}], "read_summary": {"clean": 0.954}}
    on = _entry(certify(True, doc_text=doc, receipt_obj=receipt), "0.6161616")
    assert not str(on["receipt_ref"]).startswith("unbound-field:")


def test_integer_claim_is_out_of_scope(certify):
    """Integers keep the v0.3 count-binding filter; the v0.8 clause is float-only."""
    doc = DOC.replace("0.616", "37")
    receipt = {"points": [{"naive_relock_auroc": 37}], "read_summary": {"clean": 0.954}}
    on = _entry(certify(True, doc_text=doc, receipt_obj=receipt), "37")
    assert not str(on["receipt_ref"]).startswith("unbound-field:")


def test_I1_invariant_demote_only(certify):
    """I1: enabling the clause may only move VERIFIED -> ABSTAIN.

    It can therefore never create or remove an UNGROUNDED, and no certificate can flip
    HELD -> FAILED. This is the property that made a demote-only design shippable-in-tree at all.
    """
    off, on = certify(False), certify(True)
    off_by = {(e["line"], e["token"]): e["status"] for e in off["ledger"]}
    for e in on["ledger"]:
        before = off_by[(e["line"], e["token"])]
        if e["status"] != before:
            assert (before, e["status"]) == ("VERIFIED", "ABSTAIN")
    assert on["counts"]["UNGROUNDED"] == off["counts"]["UNGROUNDED"]
    assert on["verdict"] == off["verdict"] == "OATH-HELD"


def test_I1_holds_when_an_ungrounded_is_present(certify):
    """I1 must survive alongside a real accusation: the UNGROUNDED count is untouched."""
    doc = DOC + "\nThe recall was 0.42 on the held-out split.\n"
    off, on = certify(False, doc_text=doc), certify(True, doc_text=doc)
    assert off["counts"]["UNGROUNDED"] == on["counts"]["UNGROUNDED"] >= 1
    assert _entry(on, "0.42")["status"] == "UNGROUNDED"

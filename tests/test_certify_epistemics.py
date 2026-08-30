"""Every ledger entry must say which epistemic path produced it, and the saying must move nothing.

The ladder RECON established that obligation gates accusation, not verification. The annotation
makes that machine-readable per token: `branch`, `obligated`, `obligation_source`, and (for value
matches) `path_checked`. The frozen invariant — INVARIANT_epistemics_annotation_2026_08_28.md —
is that the annotation is observation only; the A/B over all 192 committed certificates ran
before this landed and moved nothing. These tests keep both facts true.
"""
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


def _one(cert, token):
    return next(e for e in cert["ledger"] if e["token"] == token)


def test_every_ledger_entry_carries_epistemics(mk):
    cert = mk("The recall was 0.82 and n=27 items, roughly 15 people, at 0.1234567 exactly.",
              {"recall": 0.82, "n_held": 27})
    assert cert["ledger"], "expected tokens"
    for e in cert["ledger"]:
        ep = e["epistemics"]
        assert ep["branch"]
        assert isinstance(ep["obligated"], bool)
        assert "obligation_source" in ep


def test_the_unobligated_oath_is_named(mk):
    """The ladder finding, machine-readable: VERIFIED with nothing obligating it."""
    e = _one(mk("Legal scholars have long argued about 0.4267 in the abstract.",
                {"whatever": 0.4267}), "0.4267")
    assert e["status"] == "VERIFIED"
    assert e["epistemics"] == {"branch": "value-match", "obligated": False,
                              "obligation_source": None, "path_checked": False}


def test_an_obligated_verification_names_its_source(mk):
    e = _one(mk("The recall was 0.82 on the split.", {"recall": 0.82}), "0.82")
    assert e["status"] == "VERIFIED"
    assert e["epistemics"]["obligated"] is True
    assert e["epistemics"]["obligation_source"] == "vocabulary"
    assert e["epistemics"]["path_checked"] is False, "decimals are never path-checked (v0.8)"


def test_an_integer_records_that_the_path_filter_ran(mk):
    e = _one(mk("Recall counted 27 held-out items.", {"n_held": 27}), "27")
    if e["status"] == "VERIFIED":
        assert e["epistemics"]["path_checked"] is True


def test_a_precision_obligation_names_precision(mk):
    e = _one(mk("The value settled at 0.1234567 overnight.", {"unrelated": 3}), "0.1234567")
    assert e["status"] == "UNGROUNDED"
    assert e["epistemics"] == {"branch": "obligated-accusation", "obligated": True,
                              "obligation_source": "precision"}


def test_a_silent_token_is_branch_silent_and_unobligated(mk):
    e = _one(mk("There were about 15 people in the room.", {"unrelated": 3}), "15")
    assert e["status"] == "ABSTAIN"
    assert e["epistemics"] == {"branch": "silent", "obligated": False,
                              "obligation_source": None}


def test_spec_interception_is_recorded_as_its_own_branch(mk):
    cert = mk("The preregistered bar was 0.75 exactly.", {"metric": 0.75})
    e = _one(cert, "0.75")
    if e["receipt_ref"] == "spec-or-historical":
        assert e["epistemics"]["branch"] == "spec-or-historical"


def test_early_silencers_carry_epistemics_too(mk):
    """The v0.11 row-ordinal append bypasses the ladder; the A/B caught it missing epistemics."""
    cert = mk("| # | name | score |\n|---|---|---|\n| 1 | alpha | 0.9 |\n| 2 | beta | 0.8 |",
              {"unrelated": 3})
    ord_rows = [e for e in cert["ledger"] if e.get("receipt_ref") == "row_ordinal_label"]
    for e in ord_rows:
        assert e["epistemics"]["branch"] == "row-ordinal-label"
        assert e["epistemics"]["obligated"] is False


def test_the_invariant_direction_annotation_only():
    """The verifier may WRITE epistemics; nothing in it may ever READ them.

    The invariant is that the annotation is observation only. If any code path subscripted or
    .get()-read the field, a future edit could make status depend on it and the A/B guarantee
    would silently stop meaning anything. So the source may contain the key only as a literal
    being written, never as an access.
    """
    import importlib
    import inspect
    # NOT `import styxx.certify as C`: styxx/__init__ exports a FUNCTION named `certify` (from
    # .provenance) that shadows the submodule attribute once the package is fully imported, so
    # that form binds the function and getsource returns provenance code -- an order-dependent
    # failure that hit this test in the full suite and the invariant A/B script within two days.
    C = importlib.import_module("styxx.certify")
    src = inspect.getsource(C)
    assert '["epistemics"]' not in src, "certify.py subscript-reads the annotation"
    assert '.get("epistemics"' not in src, "certify.py get-reads the annotation"
    assert "'epistemics'" not in src, "single-quoted access form"
    # and the writes exist: the ladder append plus the two early silencers
    assert src.count('"epistemics":') >= 3

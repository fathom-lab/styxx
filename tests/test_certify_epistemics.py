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
    ladder = inspect.getsource(C.certify_doc)
    # The one sanctioned reader is _epistemics_summary, which folds annotation into MORE
    # annotation after every status is final. The ladder itself may only ever WRITE.
    assert '["epistemics"]' not in ladder, "certify_doc subscript-reads the annotation"
    assert '.get("epistemics"' not in ladder, "certify_doc get-reads the annotation"
    assert ladder.count('"epistemics":') >= 3, "the writes should live in certify_doc"
    whole = inspect.getsource(C)
    readers = whole.count('e["epistemics"]')
    summary_src = inspect.getsource(C._epistemics_summary)
    assert readers == summary_src.count('e["epistemics"]'), (
        "an annotation read exists outside _epistemics_summary")
    assert "verdict" not in summary_src.replace("no rates", ""), (
        "the summary fold must not touch verdict logic")


# --- the epistemics_summary block (styxx-oath/epistemics-summary/v1) ----------------------------

def test_summary_sits_between_counts_and_verdict_and_carries_the_schema_string(mk):
    cert = mk("The recall was 0.82 on the split.", {"recall": 0.82})
    keys = list(cert.keys())
    assert keys.index("counts") < keys.index("epistemics_summary") < keys.index("verdict")
    assert cert["epistemics_summary"]["schema"] == "styxx-oath/epistemics-summary/v1"
    assert "says nothing about whether any token is a true claim" in cert["epistemics_summary"]["note"]


def test_summary_every_key_always_present_zeros_included(mk):
    s = mk("Nothing numeric here beyond 15 people.", {"unrelated": 1})["epistemics_summary"]
    assert len(s["by_branch"]) == 10
    assert len(s["obligation_sources"]) == 5
    assert set(s["verified"]["value_match"]) == {
        "obligated_integer_filter_ran", "obligated_integer_filter_na",
        "unobligated_integer_filter_ran", "unobligated_integer_filter_na"}
    assert all(isinstance(v, int) for v in s["by_branch"].values())


def test_summary_invariants_hold_on_a_mixed_document(mk):
    cert = mk("The recall was 0.82 with about 15 people and 0.4267 in the abstract.\n"
              "Held-out recall 4.0 across the split.",
              {"recall": 0.82, "whatever": 0.4267})
    s, c = cert["epistemics_summary"], cert["counts"]
    assert sum(s["by_branch"].values()) == c["VERIFIED"] + c["ABSTAIN"] + c["UNGROUNDED"]
    assert s["by_branch"]["obligated-accusation"] == c["UNGROUNDED"]
    assert s["verified"]["total"] == c["VERIFIED"] == (
        sum(s["verified"]["derived"].values()) + sum(s["verified"]["value_match"].values()))
    assert s["obligated_total"] == sum(s["obligation_sources"].values())


def test_summary_names_the_weakest_cell_correctly(mk):
    """The volunteered float on an unrelated field is unobligated_integer_filter_na."""
    s = mk("Legal scholars argued about 0.4267 in the abstract.",
           {"whatever": 0.4267})["epistemics_summary"]
    assert s["verified"]["value_match"]["unobligated_integer_filter_na"] == 1
    assert s["obligated_total"] == 0


def test_range_sanity_forces_the_accusation_but_does_not_rewrite_authorship(mk):
    """The schema red-team caught this line clobbering obligation_source unconditionally.

    First-writer is the contract: vocabulary obligated the token, range-sanity emptied its hits.
    The accusation happens; the author stays.
    """
    cert = mk("Held-out recall 4.0 across the split.", {"unrelated": 1})
    e = next(x for x in cert["ledger"] if x["token"] == "4.0")
    assert e["status"] == "UNGROUNDED"
    assert e["epistemics"]["obligation_source"] == "vocabulary"
    assert cert["epistemics_summary"]["obligation_sources"]["vocabulary"] >= 1
    assert cert["epistemics_summary"]["obligation_sources"]["range-sanity"] == 0


def test_a_consumer_can_count_unbound_verifications_without_the_ledger(mk):
    """The design's gate use-case: sum the two _na cells, never parse the ledger."""
    text = "The recall was 0.82 on the split." + chr(10) + \
           "Scholars argued about 0.4267 in the abstract."
    s = mk(text, {"recall": 0.82, "whatever": 0.4267})["epistemics_summary"]
    vm = s["verified"]["value_match"]
    no_binding = vm["obligated_integer_filter_na"] + vm["unobligated_integer_filter_na"]
    assert no_binding == 2, "both floats verified with no binding filter, visible without the ledger"

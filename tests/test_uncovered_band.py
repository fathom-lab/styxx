"""The UNCOVERED band: a number the verifier never looked at must not sit silently under OATH-HELD.

`styxx.certify._NUM` ends every alternative with `(?![\\w.])`, so a numeric span followed by a
period never matches. `precision of 0.55.` extracts ZERO tokens; delete the period and it extracts
one. The span is not VERIFIED, not ABSTAIN and not UNGROUNDED — it is absent from the ledger, so
the counts are silent about it and the header reads OATH-HELD with nothing examined. That is the
vacuous-pass class this repository named, shipped inside its own verifier.

This suite pins the REPORTING contract only. `_NUM` is deliberately unchanged: widening extraction
would push the whole class through the obligation ladder and re-adjudicate 208 committed
documents, which is a preregistered measurement and not a patch. Every assertion below was watched
failing against the pre-clause verifier before it was written to pass.
"""
import json
import re

import pytest

from styxx.certify import (_NUM, _UNCOVERED_COUNTED, _WIDE_NUM, certify_doc, extract_numbers,
                           uncovered_spans)


# --------------------------------------------------------------------------- the defect itself

def test_num_regex_cannot_see_a_number_followed_by_a_period():
    """The premise. If this ever fails, `_NUM` was widened and this whole band changes meaning."""
    assert [m.group(0) for m in _NUM.finditer("precision of 0.55")] == ["0.55"]
    assert [m.group(0) for m in _NUM.finditer("precision of 0.55.")] == []
    assert [m.group(0) for m in _NUM.finditer("the count was 12.")] == []


def test_the_period_mutilates_a_thousands_separated_number_rather_than_hiding_it():
    """The second mode: `23,247.` does not vanish, it becomes the token `23`.

    The thousands alternative dies on the trailing period and the bare-integer alternative then
    matches the first group alone, so a WRONG VALUE enters the ledger carrying a real status.
    """
    assert [m.group(0) for m in _NUM.finditer("all 23,247 rows")] == ["23,247"]
    assert [m.group(0) for m in _NUM.finditer("all 23,247.")] == ["23"]


# --------------------------------------------------------------------------- the canonical case

CANON = "precision of 0.55.\n"


def test_canonical_case_is_extracted_as_nothing_at_all():
    assert extract_numbers(CANON) == []


def test_canonical_case_is_reported_as_uncovered():
    spans = uncovered_spans(CANON)
    assert len(spans) == 1
    s = spans[0]
    assert (s["line"], s["token"], s["reason"], s["counted"]) == (1, "0.55", "trailing-period", True)
    assert s["col"] == CANON.index("0.55")


def test_canonical_case_does_not_certify_as_a_clean_held_with_zero_tokens(tmp_path):
    """THE test. A document whose only number was never examined must not read as a clean pass."""
    doc = tmp_path / "canon.md"
    doc.write_text(CANON, encoding="utf-8")
    rec = tmp_path / "canon_result.json"
    rec.write_text(json.dumps({"precision": 0.55}), encoding="utf-8")

    cert = certify_doc(doc, [rec])

    assert cert["counts"] == {"VERIFIED": 0, "ABSTAIN": 0, "UNGROUNDED": 0}
    assert cert["uncovered"] == 1
    assert cert["verdict"] != "OATH-HELD"
    assert cert["verdict"] == "OATH-HELD, 1 uncovered"


def test_deleting_the_period_certifies_with_one_token_and_no_band(tmp_path):
    """The control: the SAME claim, one character shorter, is examined and sworn to."""
    doc = tmp_path / "canon.md"
    doc.write_text("precision of 0.55\n", encoding="utf-8")
    rec = tmp_path / "canon_result.json"
    rec.write_text(json.dumps({"precision": 0.55}), encoding="utf-8")

    cert = certify_doc(doc, [rec])

    assert cert["counts"]["VERIFIED"] == 1
    assert cert["uncovered"] == 0
    assert cert["verdict"] == "OATH-HELD"


# --------------------------------------------------------------------------- certificate surface

def _cert(tmp_path, text, receipt=None):
    doc = tmp_path / "d.md"
    doc.write_text(text, encoding="utf-8")
    rec = tmp_path / "d_result.json"
    rec.write_text(json.dumps(receipt if receipt is not None else {}), encoding="utf-8")
    return certify_doc(doc, [rec])


def test_uncovered_is_a_new_field_and_no_existing_field_was_repurposed(tmp_path):
    cert = _cert(tmp_path, CANON, {"precision": 0.55})
    for key in ("uncovered", "uncovered_items", "uncovered_excluded_by_rule", "uncovered_policy"):
        assert key in cert, key
    # the pre-existing surface is untouched by the band
    assert cert["counts"] == {"VERIFIED": 0, "ABSTAIN": 0, "UNGROUNDED": 0}
    assert cert["ungrounded"] == [] and cert["abstained"] == []
    assert cert["ledger"] == []
    assert cert["epistemics_summary"]["by_branch"]["silent"] == 0


def test_every_uncovered_item_carries_a_coordinate_and_a_machine_reason(tmp_path):
    cert = _cert(tmp_path, "recall 0.91 and precision 0.55.\nAUC was 0.998.\n", {"recall": 0.91})
    assert cert["uncovered"] == 2
    for item in cert["uncovered_items"]:
        assert set(item) >= {"line", "col", "token", "reason", "counted", "context"}
        assert isinstance(item["line"], int) and isinstance(item["col"], int)
        assert item["counted"] is True
    assert [i["token"] for i in cert["uncovered_items"]] == ["0.55", "0.998"]
    assert [i["line"] for i in cert["uncovered_items"]] == [1, 2]


def test_a_partially_extracted_span_records_what_was_examined_instead(tmp_path):
    cert = _cert(tmp_path, "the corpus holds 23,247.\n", {"n": 23247})
    item, = cert["uncovered_items"]
    assert item["token"] == "23,247"
    assert item["reason"] == "partial-extraction"
    assert item["extracted_instead"] == "23"
    # and the mutilated token really did enter the ledger under a real status
    assert [(e["token"], e["value"]) for e in cert["ledger"]] == [("23", 23.0)]


def test_the_policy_block_names_the_bands_own_boundary(tmp_path):
    pol = _cert(tmp_path, CANON, {})["uncovered_policy"]
    assert pol["schema"] == "styxx-oath/uncovered-band/v1"
    assert sorted(pol["counted_reasons"]) == sorted(_UNCOVERED_COUNTED)
    # uncovered=0 must not be readable as "fully covered"
    assert "does NOT mean fully covered" in pol["scope_limit"]


# --------------------------------------------------------------------------- the verdict rule

@pytest.mark.parametrize("text,receipt,expected", [
    ("recall 0.91\n", {"recall": 0.91}, "OATH-HELD"),
    ("recall 0.91.\n", {"recall": 0.91}, "OATH-HELD, 1 uncovered"),
    ("recall 0.91. precision 0.55.\n", {"recall": 0.91}, "OATH-HELD, 2 uncovered"),
])
def test_a_held_verdict_cannot_be_read_without_its_uncovered_count(text, receipt, expected,
                                                                   tmp_path):
    assert _cert(tmp_path, text, receipt)["verdict"] == expected


def test_a_failed_verdict_also_carries_the_count(tmp_path):
    cert = _cert(tmp_path, "the recall is 4 and precision 0.55.\n", {"recall": 9})
    assert cert["counts"]["UNGROUNDED"] == 1
    assert cert["uncovered"] == 1
    assert cert["verdict"] == "OATH-FAILED, 1 uncovered"


def test_the_cli_header_shows_uncovered_beside_the_counts(tmp_path, capsys):
    from styxx.certify import main
    doc = tmp_path / "d.md"
    doc.write_text(CANON, encoding="utf-8")
    rec = tmp_path / "d_result.json"
    rec.write_text(json.dumps({"precision": 0.55}), encoding="utf-8")

    code = main([str(doc), str(rec), "--out", str(tmp_path / "d.certificate.json")])

    header = capsys.readouterr().out.splitlines()[0]
    assert header.startswith("OATH-HELD, 1 uncovered")
    assert "UNCOVERED=1" in header
    assert re.search(r"verified=0 abstained=0 contradicted=0\s+UNCOVERED=1", header)
    # REPORTING, not re-adjudication: the exit code keys off UNGROUNDED and is unmoved.
    assert code == 0


# --------------------------------------------------------------------------- what is NOT the band

@pytest.mark.parametrize("text,reason", [
    ("1. First item\n", "ordered-list-marker"),
    ("## 3. A numbered section\n", "heading-ordinal"),
    ("**4. A bold numbered claim.** text\n", "ordered-list-marker"),
    ("> 5. quoted list item\n", "md-structure"),          # an EXISTING extractor rule reaches it
    ("> > 6. deeper quote\n", "ordered-list-marker"),      # past `_MD_STRUCTURE`'s column-2 clause
])
def test_structural_ordinals_are_named_and_excluded_not_counted(text, reason):
    """A label has no truth condition — the v0.11 row-ordinal category. Excluded, never silent.

    An existing extractor rule wins the LABEL where one reaches the span (`_MD_STRUCTURE`'s
    line-start clause covers a blockquote at column <= 2), so the band never invents a second name
    for a decision the extractor already made. Either way the span is excluded, not counted.
    """
    span, = uncovered_spans(text)
    assert span["reason"] == reason
    assert span["counted"] is False


def test_a_paren_ordered_list_marker_is_already_an_ordinary_ledger_token():
    """Disclosed asymmetry, not a band decision: `2)` has no period, so `_NUM` was never blind to
    it and `_MD_STRUCTURE` does not name it. It is extracted and adjudicated today, exactly as
    before this clause. The band reports what the extractor cannot see; it does not tidy up what
    the extractor can."""
    assert [e["token"] for e in extract_numbers("2) Second item")] == ["2"]
    assert uncovered_spans("2) Second item") == []


def test_excluded_spans_are_still_enumerated_in_the_certificate(tmp_path):
    """Silence loud, never omission: an excluded span stays countable by coordinate."""
    cert = _cert(tmp_path, "## 1. A section\n", {})
    assert cert["uncovered"] == 0
    assert cert["verdict"] == "OATH-HELD"
    excluded, = cert["uncovered_excluded_by_rule"]
    assert (excluded["line"], excluded["token"], excluded["reason"]) == (1, "1", "heading-ordinal")
    assert cert["uncovered_policy"]["excluded_by_rule"] == 1


@pytest.mark.parametrize("text", [
    "a 0.5B agent\n",          # unit-suffixed decimal: the '0' is not a span
    "3.1M eigenvalues\n",
    "identical to 1.36e-14 on\n",   # scientific notation stays a DISCLOSED blind spot
])
def test_the_widening_does_not_manufacture_spans_out_of_suffixed_decimals(text):
    """`(?!\\.\\d)` in the wide guard. Without it the bare-integer alternative eats the integer
    part of any decimal carrying a letter suffix and invents tokens the document never wrote."""
    assert uncovered_spans(text) == []


def test_a_span_the_extractor_examined_is_never_in_the_band():
    text = "recall 0.91 and precision 0.55 and 12 items\n"
    assert len(extract_numbers(text)) == 3
    assert uncovered_spans(text) == []


def test_a_year_is_not_uncovered_because_a_named_rule_already_drops_it():
    span, = uncovered_spans("shipped in 2019. Next\n")
    assert span["reason"] == "year"
    assert span["counted"] is False


# --------------------------------------------------------------------------- structural invariants

def test_the_band_is_a_pure_function_of_the_document_bytes(tmp_path):
    """It consults no receipts, so a document cannot shrink its own band by citing more."""
    text = "recall 0.91.\n"
    lean = _cert(tmp_path, text, {})
    rich = _cert(tmp_path, text, {"recall": 0.91, "a": 1, "b": 2, "c": 0.91})
    assert lean["uncovered"] == rich["uncovered"] == 1
    assert lean["uncovered_items"] == rich["uncovered_items"]


def test_the_wide_scanner_is_a_strict_superset_of_extraction():
    """Every span `_NUM` finds, `_WIDE_NUM` finds at the same coordinates. The band is additive:
    a widening that MOVED extraction would re-adjudicate the corpus, which this cycle refuses."""
    for line in ["recall 0.91 and 12 items", "-0.0154 vs +3", "1,234 rows", "AUC 0.998, n=48",
                 ".55 of them", "a/b 7/12 held"]:
        narrow = [(m.start(), m.group(0)) for m in _NUM.finditer(line)]
        wide = [(m.start(), m.group(0)) for m in _WIDE_NUM.finditer(line)]
        assert narrow == wide, line


def test_no_uncovered_span_is_also_a_ledger_row(tmp_path):
    """The internal assertion in `certify_doc`, exercised on text carrying both classes.

    A token cannot be both examined and unexamined. The collision test is on the full coordinate
    (line, col, token), because `partial-extraction` legitimately SHARES A START with a ledger row.
    """
    text = "recall 0.91, all 23,247. and precision 0.55.\n"
    cert = _cert(tmp_path, text, {"recall": 0.91})
    rows = {(e["line"], e["col"], e["token"]) for e in cert["ledger"]}
    spans = {(s["line"], s["col"], s["token"]) for s in cert["uncovered_items"]}
    assert rows & spans == set()
    assert ("23,247" in {s["token"] for s in cert["uncovered_items"]})
    assert ("23" in {e["token"] for e in cert["ledger"]})   # same start, different token


def test_the_band_does_not_move_any_token_status(tmp_path):
    """Reporting, not re-adjudication: the ledger of a document with a band is what it always was."""
    text = "recall 0.91 and precision 0.55.\n"
    cert = _cert(tmp_path, text, {"recall": 0.91})
    assert [(e["token"], e["status"]) for e in cert["ledger"]] == [("0.91", "VERIFIED")]
    assert cert["counts"] == {"VERIFIED": 1, "ABSTAIN": 0, "UNGROUNDED": 0}

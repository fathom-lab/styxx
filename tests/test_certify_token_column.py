"""Regression tests for the OATH v0.10 token-column repair (PREREG_oath_v10_token_column_2026_08_23).

`certify_doc` used to locate each extracted token with ``ctx.find(num["token"])`` — the FIRST
occurrence of the token STRING, which is a DIFFERENT token 4,612 times in the 48,097 tokens under
papers/. Every context window, and so `is_spec`, `is_notation`, `is_hist`, the range-sanity tests,
the slash-pair branch and the v0.5 derived-percent parse, could be decided against text that does
not surround the claim.

These lock the shipped behaviour so a future edit cannot silently regress it:

  * every ledger entry carries the column its match was found at, and that column addresses the
    token (not merely *a* copy of its string);
  * a length-preserving sha/date/version scrub, which is what makes the column an address;
  * the second of two identical tokens on a line gets its OWN windows;
  * the companion slash-pair range-sanity guard, and the range-sanity rule it must not weaken;
  * both flags are severable, and the ledger shape addition is additive (invariant I1).
"""
import json

import pytest

import styxx.certify as certify
from styxx.certify import certify_doc, extract_numbers


@pytest.fixture
def flags_on():
    """The shipped composition, restored after each test so flag flips cannot leak."""
    prev = (certify.V10_TOKEN_COLUMN, certify.V10_SLASHPAIR_RANGE_GUARD)
    certify.V10_TOKEN_COLUMN, certify.V10_SLASHPAIR_RANGE_GUARD = True, True
    yield
    certify.V10_TOKEN_COLUMN, certify.V10_SLASHPAIR_RANGE_GUARD = prev


def _cert(tmp_path, line, receipt_obj):
    doc = tmp_path / "d.md"
    doc.write_text(f"# t\n\nsome preamble sentence to avoid line-start filters.\n\n{line}\n",
                   encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(receipt_obj), encoding="utf-8")
    return certify_doc(doc, [rp])


def _entries(cert, token):
    return [e for e in cert["ledger"] if e["token"] == token]


# ---------------------------------------------------------------- the column is an address

def test_every_token_carries_its_own_column(flags_on):
    line = "frames (10 neutral + 10 in-frame = COMBINED) beat spending all 20 on the belief"
    ents = extract_numbers(line + "\n")
    cols = [e["col"] for e in ents]
    assert len(set(cols)) == len(cols), "two tokens cannot share a column"
    for e in ents:
        assert line[e["col"]:e["col"] + len(e["token"])] == e["token"]
    # the two 10s must NOT collapse onto the first one, which is what ctx.find did
    tens = [e["col"] for e in ents if e["token"] == "10"]
    assert len(tens) == 2 and tens[0] != tens[1]


def test_column_survives_a_leading_sha_because_the_scrub_preserves_length(flags_on):
    # a sha is blanked from the searchable line; with the shipped `re.sub(pat, " ", ...)` the
    # whole 40-char match collapsed to ONE space and every column to its right shifted 39 left.
    line = "receipt a3f19c8d2b4e6f70a1c5d9e3b7f2a6c4d8e1b5f9 reports accuracy 0.884 on the split"
    ents = extract_numbers(line + "\n")
    e = next(e for e in ents if e["token"] == "0.884")
    assert e["col"] == line.index("0.884")


def test_column_survives_a_leading_date_and_version(flags_on):
    line = "on 2026-08-23 under v7.46.0 the recall was 0.735 for the held split"
    e = next(e for e in extract_numbers(line + "\n") if e["token"] == "0.735")
    assert e["col"] == line.index("0.735")


def test_indented_line_column_is_a_raw_line_offset(flags_on, tmp_path):
    # `ctx` is stripped but `col` is a raw-line offset; the windows must reconcile the two.
    line = '           "G2_gate": {"metric": "seeds_below_clique", "op": ">=", "value": 5,'
    e = next(e for e in extract_numbers(line + "\n") if e["token"] == "5")
    assert e["col"] == line.rindex("5")
    # and the JSON-idiom spec clause, which reads the 18 chars before the token, must now fire
    cert = _cert(tmp_path, line, {"unrelated_leaf": 5})
    assert _entries(cert, "5")[0]["status"] == "ABSTAIN"
    assert _entries(cert, "5")[0]["receipt_ref"] == "spec-or-historical"


# ---------------------------------------------------------------- the windows follow the column

def test_second_identical_token_gets_its_own_post_window(flags_on, tmp_path):
    # `S_frame@20` is a v0.5 class D @-glued parameter. Before the repair, ctx.find("20") landed
    # on the FIRST 20 and the parameter verified against an unrelated leaf.
    line = "frames (20 neutral) beat spending all of it on the belief alone (`S_frame@20`)?"
    cert = _cert(tmp_path, line, {"n_neutral": 20})
    ents = _entries(cert, "20")
    assert len(ents) == 2
    assert ents[0]["status"] == "VERIFIED"                 # the real count still grounds
    assert ents[1]["receipt_ref"] == "v05-notation"        # the @-param is notation


def test_historical_quotation_on_a_mixed_line_covers_only_the_quoted_copy(flags_on, tmp_path):
    # the live copy sits further than the v0.3 mixed-line rule's 24-character slack from the
    # disclosure phrase, so this isolates the column repair rather than re-testing that slack.
    line = ("CI [0.432, 0.500] holds for the entire held-out cohort; the value originally "
            "printed [0.433, 0.500] came from an unpersisted run.")
    cert = _cert(tmp_path, line, {"ci95": [0.432, 0.500], "bootstrap": {"hi": 0.500}})
    ents = _entries(cert, "0.500")
    assert len(ents) == 2
    assert ents[0]["status"] == "VERIFIED"                          # the live claim
    assert ents[1]["receipt_ref"] == "spec-or-historical"           # the quoted one


def test_gate_table_observed_column_is_not_abstained_as_its_own_bar(flags_on, tmp_path):
    # every token on the row used to anchor at the BAR, so the OBSERVED value was abstained as a
    # specification — certification by omission on the one column a gate table exists to report.
    line = "| G0_coverage | >= 45 pairs | 45 | PASS |"
    cert = _cert(tmp_path, line, {"n_pairs": 45})
    ents = _entries(cert, "45")
    assert len(ents) == 2
    assert ents[0]["receipt_ref"] == "spec-or-historical"    # the bar
    assert ents[1]["status"] == "VERIFIED"                   # the measurement


# ---------------------------------------------------------------- the companion guard

def test_slashpair_numerator_escapes_range_sanity(flags_on, tmp_path):
    # "(stability 5/5)" is five of five, not a stability of 5.0. Correct anchoring puts
    # "stability " in `pre`, and without the guard the v0.3 range-sanity rule accuses a document
    # whose receipt holds the count.
    line = "all five under the 0.65 ceiling (stability 5/5), so the pass does not hinge on one."
    cert = _cert(tmp_path, line, {"stability_count_under_ceiling": 5, "ceiling": 0.65})
    assert _entries(cert, "5")[0]["status"] == "VERIFIED"


def test_range_sanity_still_fires_without_a_slash(flags_on, tmp_path):
    # the guard must not weaken the rule it narrows: an out-of-range bounded quantity still fails.
    cert = _cert(tmp_path, "The detector reported AUC 4.0 on the held-out split.",
                 {"some_unrelated_leaf": 4.0})
    assert _entries(cert, "4.0")[0]["status"] == "UNGROUNDED"


def test_guard_is_inert_while_the_column_repair_is_off(flags_on, tmp_path):
    line = "all five under the 0.65 ceiling (stability 5/5), so the pass does not hinge on one."
    receipt = {"stability_count_under_ceiling": 5, "ceiling": 0.65}
    certify.V10_TOKEN_COLUMN = False
    certify.V10_SLASHPAIR_RANGE_GUARD = False
    a = [e["status"] for e in _cert(tmp_path, line, receipt)["ledger"]]
    certify.V10_SLASHPAIR_RANGE_GUARD = True
    b = [e["status"] for e in _cert(tmp_path, line, receipt)["ledger"]]
    assert a == b


# ---------------------------------------------------------------- severability and shape

def test_flags_off_restore_the_first_occurrence_behaviour(flags_on, tmp_path):
    line = "frames (20 neutral) beat spending all of it on the belief alone (`S_frame@20`)?"
    receipt = {"n_neutral": 20}
    on = [e["status"] for e in _cert(tmp_path, line, receipt)["ledger"]]
    certify.V10_TOKEN_COLUMN = False
    certify.V10_SLASHPAIR_RANGE_GUARD = False
    off = [e["status"] for e in _cert(tmp_path, line, receipt)["ledger"]]
    assert on != off, "the flags must actually be severable, not decorative"
    assert "col" not in _cert(tmp_path, line, receipt)["ledger"][0]


def test_extraction_is_unchanged_by_the_length_preserving_scrub(flags_on):
    """Gate G2 in miniature: the repair moves where windows point, never what is extracted."""
    text = ("# t\n\npreamble.\n\n"
            "receipt a3f19c8d2b4e6f70a1c5d9e3b7f2a6c4d8e1b5f9 on 2026-08-23 under v7.46.0\n"
            "gave accuracy 0.884, recall 0.735, and 12 of 16 cells at n=16.\n"
            "- 9/12 held; the margin was -0.0154 and the ratio 2-3B was unchanged.\n")
    on = [(e["line"], e["token"]) for e in extract_numbers(text)]
    certify.V10_TOKEN_COLUMN = False
    off = [(e["line"], e["token"]) for e in extract_numbers(text)]
    assert on == off


def test_col_is_additive_to_the_ledger_shape(flags_on, tmp_path):
    """Invariant I1 — `col` is a new key; every pre-existing key is untouched."""
    cert = _cert(tmp_path, "The detector reached AUROC 0.884 on the held-out split.",
                 {"auroc": 0.884})
    e = cert["ledger"][0]
    for k in ("line", "token", "value", "decimals", "context", "status", "receipt_ref"):
        assert k in e
    assert isinstance(e["col"], int)

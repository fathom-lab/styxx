"""Regression tests for the OATH v0.11 row-ordinal retraction.

Prereg: `papers/closed-model-frontier/PREREG_oath_v11_row_ordinal_retraction_2026_08_25.md`.

A markdown table's first column is where this corpus writes its row numbers. `extract_numbers`
extracts them like any other token, and on rows whose own text carries trigger vocabulary the
obligation predicate binds them — so a row number had to ground in a receipt leaf or be accused.
A row number has no receipt, because it asserts nothing. The certified frame's ENTIRE standing
accusation surface was four of them.

`V11_ORDINAL_LABEL` demotes such a token to ABSTAIN with the reason `row_ordinal_label`, at the
`is_spec` tier, before any obligation or match is consulted.

These lock the shipped behaviour so a future edit cannot silently widen, narrow, or fuse it:

  * the three conjuncts — first cell of a DATA row, an exact header from the frozen nine-entry
    vocabulary, and a cell that is entirely a bare non-negative integer <= 100;
  * every named exclusion stays excluded, `seed` above all — silencing it replays the
    broad-detector catastrophe;
  * the clause is VALUE-BLIND: doctor the digit and it still fires. A clause that stops firing on
    a doctored row number is a fuse, and the prereg rejects three designs for exactly that;
  * NEVER non-extraction: the token count does not move and every silenced token stays countable
    by coordinate, in the ledger row AND in the certificate's `abstained` array;
  * `V10_TOKEN_COLUMN` is a declared, non-severable prerequisite;
  * the clause is severable — off, the ledger is byte-identical.
"""
import importlib
import json

import pytest

# importlib, not `import styxx.certify as ...`: the package attribute `styxx.certify` is the
# provenance FUNCTION (styxx/__init__.py), and `import ... as` binds the attribute when it
# exists — module alone, function mid-suite. Same class ae45aaa fixed for v0.9.
certify = importlib.import_module("styxx.certify")
from styxx.certify import certify_doc  # noqa: E402

# A row whose text carries trigger vocabulary ("recall", "rate", "held") — which is what obligates
# the leading ordinal in the first place. Without a trigger there is no accusation to retract.
# Deliberately carries NO number of its own, so the only tokens in these fixtures are the ordinals
# under test and a count assertion cannot be confounded by the row text.
ROW_TEXT = "recall rate on the held set"


def _doc(tmp_path, header, cells, row_text=ROW_TEXT):
    lines = ["# t", "", f"| {header} | claim |", "|---|---|"]
    for cell in cells:
        lines.append(f"| {cell} | {row_text} |")
    doc = tmp_path / "d.md"
    doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"unrelated_leaf": 999.0}), encoding="utf-8")
    return doc, [rp]


def _rows(tmp_path, header, cells, row_text=ROW_TEXT):
    doc, receipts = _doc(tmp_path, header, cells, row_text)
    cert = certify_doc(doc, receipts)
    return cert, [e for e in cert["ledger"] if e["receipt_ref"] == "row_ordinal_label"]


@pytest.fixture(autouse=True)
def shipped_flags():
    """The shipped composition, restored after each test so flag flips cannot leak."""
    prev = (certify.V11_ORDINAL_LABEL, certify.V10_TOKEN_COLUMN)
    certify.V11_ORDINAL_LABEL, certify.V10_TOKEN_COLUMN = True, True
    yield
    certify.V11_ORDINAL_LABEL, certify.V10_TOKEN_COLUMN = prev


# ---------------------------------------------------------------- the defect it exists to fix

def test_a_row_ordinal_is_no_longer_accused(tmp_path):
    """The whole cycle in one assertion: an ordinal under `#` is ABSTAIN, not UNGROUNDED."""
    cert, fired = _rows(tmp_path, "#", ["1", "2", "3"])
    assert len(fired) == 3
    assert {e["status"] for e in fired} == {"ABSTAIN"}
    assert cert["counts"]["UNGROUNDED"] == 0
    assert cert["verdict"] == "OATH-HELD"


def test_without_the_clause_the_same_ordinal_is_accused(tmp_path):
    """The accusation is real, not hypothetical — this is what the clause withdraws."""
    certify.V11_ORDINAL_LABEL = False
    cert, fired = _rows(tmp_path, "#", ["1", "2", "3"])
    assert not fired
    assert cert["counts"]["UNGROUNDED"] > 0
    assert cert["verdict"] == "OATH-FAILED"


# ---------------------------------------------------------------- conjunct 2: the frozen vocabulary

@pytest.mark.parametrize("header", ["#", "#.", "no.", "nr", "idx", "index", "row", "row #", "№"])
def test_every_frozen_header_fires(tmp_path, header):
    _cert, fired = _rows(tmp_path, header, ["1", "2"])
    assert len(fired) == 2, f"{header!r} is in the frozen vocabulary and must fire"


@pytest.mark.parametrize("header", ["**#**", "`#`", " # ", "INDEX", "Row #"])
def test_header_normalisation_strips_emphasis_backticks_and_case(tmp_path, header):
    _cert, fired = _rows(tmp_path, header, ["1", "2"])
    assert len(fired) == 2, f"{header!r} normalises into the vocabulary"


@pytest.mark.parametrize("header", [
    "seed",        # 63 of the 150 first-cell tokens in frame, 61 VERIFIED
    "rank",        # a label too, but retracting a class needs its own panel and prereg
    "rank k",      # a different population: genuine claims under a non-identity mapping
    "-", "n", "no", "num", "id", "item", "line", "claim", "k", "run", "attempt",
])
def test_every_named_exclusion_stays_obligated(tmp_path, header):
    """Exclusion is the SAFE direction: an excluded token stays accusable, never silenced."""
    _cert, fired = _rows(tmp_path, header, ["1", "2"])
    assert not fired, f"{header!r} is a named exclusion and must not fire"


def test_the_vocabulary_is_exactly_nine_entries(tmp_path):
    """It can only shrink. A future edit that widens it must fail here first."""
    assert certify._V11_ORDINAL_HEADERS == frozenset(
        {"#", "#.", "no.", "nr", "idx", "index", "row", "row #", "№"})


# ---------------------------------------------------------------- conjunct 3: sole content

@pytest.mark.parametrize("cell,fires", [
    ("1", True), ("0", True), ("100", True), ("07", True), ("**3**", True),
    ("101", False),        # the cap; past it the accused class is re-manufactured, disclosed
    ("-1", False),         # not non-negative
    ("3.0", False),        # not a bare integer
    ("`3`", False),        # backticks are stripped for the HEADER only — safe direction
    ("3 items", False),    # not sole content
    ("v3", False),
])
def test_sole_content_and_the_value_cap(tmp_path, cell, fires):
    _cert, fired = _rows(tmp_path, "#", [cell])
    assert bool(fired) is fires, f"cell {cell!r}"


# ---------------------------------------------------------------- conjunct 1: position

def test_only_the_first_cell(tmp_path):
    """A number in the SECOND cell of the same row is untouched."""
    _cert, fired = _rows(tmp_path, "#", ["1"], row_text="recall rate 7 on the held set")
    assert [e["token"] for e in fired] == ["1"]


def test_only_data_rows_are_in_scope(tmp_path):
    """`_table_rows` maps DATA rows only, and the clause reads that machinery rather than
    copying it — so clause scope and binding-context scope cannot diverge. Header line 3 and
    separator line 4 are out of scope; the data rows are lines 5 and 6."""
    _cert, fired = _rows(tmp_path, "#", ["1", "2"])
    assert [e["line"] for e in fired] == [5, 6]


def test_a_table_with_no_separator_is_out_of_reach(tmp_path):
    """A completeness gap, not a silencing gap: those tokens stay OBLIGATED."""
    doc = tmp_path / "d.md"
    doc.write_text(f"# t\n\n| # | claim |\n| 1 | {ROW_TEXT} |\n", encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"unrelated_leaf": 999.0}), encoding="utf-8")
    cert = certify_doc(doc, [rp])
    assert not [e for e in cert["ledger"] if e["receipt_ref"] == "row_ordinal_label"]


# ---------------------------------------------------------------- the fuse test

def test_the_clause_is_value_blind_and_still_fires_on_a_doctored_ordinal(tmp_path):
    """THE property that separates this clause from the three designs the prereg rejected.

    A value-reading detector (`the column must read exactly 1..N`) stops firing the moment a row
    number is doctored — absent on exactly the input it exists to handle. Measured on the real
    corpus, that variant misses 11 of 11 mutants at every seed; this clause misses none.
    """
    _cert, intact = _rows(tmp_path, "#", ["1", "2", "3"])
    assert len(intact) == 3
    _cert, doctored = _rows(tmp_path, "#", ["1", "7", "3"])    # the run is broken
    assert len(doctored) == 3, "a broken 1..N run must not switch the clause off"
    assert {e["status"] for e in doctored} == {"ABSTAIN"}


# ---------------------------------------------------------------- never non-extraction

def test_silence_stays_countable(tmp_path):
    """Silence loud, never omission: the token count does not move and every silenced token is
    reachable by coordinate, in the ledger row AND in the certificate's `abstained` array."""
    certify.V11_ORDINAL_LABEL = False
    off, _ = _rows(tmp_path, "#", ["1", "2", "3"])
    certify.V11_ORDINAL_LABEL = True
    on, fired = _rows(tmp_path, "#", ["1", "2", "3"])

    assert len(on["ledger"]) == len(off["ledger"]), "non-extraction would shrink the ledger"
    for e in fired:
        assert e.get("col") is not None
        assert any(a["line"] == e["line"] and a["token"] == e["token"]
                   for a in on["abstained"])


# ---------------------------------------------------------------- prerequisite and severability

def test_v10_token_column_is_a_non_severable_prerequisite(tmp_path):
    """Without a recorded column the clause has no ADDRESS, so it must not fire at all."""
    certify.V10_TOKEN_COLUMN = False
    _cert, fired = _rows(tmp_path, "#", ["1", "2", "3"])
    assert not fired


def test_the_clause_is_severable(tmp_path):
    """Off, the ledger is identical entry for entry — status and reason alike."""
    certify.V11_ORDINAL_LABEL = True
    on, _ = _rows(tmp_path, "seed", ["1", "2"])
    certify.V11_ORDINAL_LABEL = False
    off, _ = _rows(tmp_path, "seed", ["1", "2"])
    assert [(e["status"], e["receipt_ref"]) for e in on["ledger"]] == \
           [(e["status"], e["receipt_ref"]) for e in off["ledger"]]


def test_the_reason_code_is_exact(tmp_path):
    """G8 reads this string. It is the whole countable trail."""
    _cert, fired = _rows(tmp_path, "#", ["1"])
    assert [e["receipt_ref"] for e in fired] == ["row_ordinal_label"]

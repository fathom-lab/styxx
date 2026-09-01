"""Tests for `styxx.oathready` — the author-facing readiness check.

This module exists because of what the OATH-EXT recon measured on 2026-08-26: pointed at
documents not written to be certified, the verifier abstains on almost everything and every
accusation it makes is false. The conclusion was that OATH is a CONTRACT, not a detector — and a
contract nobody can check themselves against is not a contract.

These lock the behaviour an author depends on:

  * the four kinds — bound, coincident, accused, abstained — partition the ledger exactly;
  * an accusation always names the vocabulary that caused it, because an author cannot act on an
    accusation they cannot trace to a word on their own line;
  * a verification against a POSITION (an index, a seed, a step counter) is reported as
    coincident rather than counted as a success — the v0.8 CLOSED_NEGATIVE channel;
  * abstention is never reported as a failure, and never sets a non-zero exit code;
  * the tool refuses to imply it has checked whether the numbers are TRUE.
"""
from __future__ import annotations

import json

from styxx.oathready import main, readiness_report, render


def _doc(tmp_path, body: str, receipt: dict):
    d = tmp_path / "doc.md"
    d.write_text(body, encoding="utf-8")
    r = tmp_path / "r.json"
    r.write_text(json.dumps(receipt), encoding="utf-8")
    return d, [r]


def test_a_kept_contract_reads_as_bound(tmp_path):
    """A number whose line names its quantity, grounding at a leaf whose path says the same."""
    d, r = _doc(tmp_path, "# t\n\nThe recall reached 0.82 on the held set.\n",
                {"recall": 0.82})
    rep = readiness_report(d, r)
    row = next(x for x in rep["rows"] if x["token"] == "0.82")
    assert row["kind"] == "bound"
    assert rep["accused"] == 0


def test_an_accusation_always_names_the_word_that_caused_it(tmp_path):
    """An author cannot act on an accusation they cannot trace to a word on their own line."""
    d, r = _doc(tmp_path, "# t\n\nThe learning rate was tuned over 100000 steps.\n",
                {"unrelated": 999.0})
    rep = readiness_report(d, r)
    accused = [x for x in rep["rows"] if x["kind"] == "accused"]
    assert accused, "a bound line with no matching receipt must be accused"
    assert "rate" in accused[0]["obligated_by"]
    assert "rate" in accused[0]["advice"]


def test_a_verification_against_a_position_is_reported_as_coincident(tmp_path):
    """Grounding at an index leaf is arithmetic accident, not evidence — v0.8 CLOSED_NEGATIVE."""
    d, r = _doc(tmp_path, "# t\n\nThe recall reached 3 on the held set.\n",
                {"recall_history": {"step": 3}})
    rep = readiness_report(d, r)
    row = next(x for x in rep["rows"] if x["token"] == "3")
    assert row["kind"] == "coincident"
    assert "POSITION" in row["advice"]


def test_abstention_is_not_a_failure(tmp_path):
    """Silence where nothing grounds is the designed behaviour, not a defect."""
    d, r = _doc(tmp_path, "# t\n\nWe ran the study in three sites with 12 people.\n",
                {"unrelated": 999.0})
    rep = readiness_report(d, r)
    assert rep["accused"] == 0
    assert rep["abstained"] >= 1
    assert main([str(d), str(r[0])]) == 0


def test_accusations_set_a_nonzero_exit_and_abstentions_do_not(tmp_path):
    d, r = _doc(tmp_path, "# t\n\nThe learning rate was tuned over 100000 steps.\n",
                {"unrelated": 999.0})
    assert main([str(d), str(r[0])]) == 1


def test_the_kinds_partition_the_ledger(tmp_path):
    d, r = _doc(tmp_path,
                "# t\n\nThe recall reached 0.82 on the held set.\n"
                "The learning rate was tuned over 100000 steps.\n"
                "We ran it in three sites with 12 people.\n",
                {"recall": 0.82})
    rep = readiness_report(d, r)
    assert rep["bound"] + rep["coincident"] + rep["accused"] + rep["abstained"] == rep["tokens"]
    assert rep["tokens"] == len(rep["rows"])


def test_the_report_refuses_to_imply_it_checked_correctness(tmp_path):
    """A document can be fully ready and completely wrong. The tool must say so."""
    d, r = _doc(tmp_path, "# t\n\nThe recall reached 0.82 on the held set.\n", {"recall": 0.82})
    rep = readiness_report(d, r)
    assert "cannot tell whether a number is CORRECT" in rep["not_a_grade"]
    assert "not a score" in rep["not_a_grade"]
    assert any("Mention is treated as use" in lim for lim in rep["known_limits"])
    assert rep["not_a_grade"] in render(rep)


def test_render_says_so_when_there_is_nothing_to_fix(tmp_path):
    d, r = _doc(tmp_path, "# t\n\nThe recall reached 0.82 on the held set.\n", {"recall": 0.82})
    text = render(readiness_report(d, r))
    assert "No accusations and no coincidental bindings" in text


def _mk(tmp, text, receipt):
    from pathlib import Path
    import json as _j
    d = Path(tmp)
    doc = d / "d.md"; doc.write_text(text + "\n", encoding="utf-8")
    rec = d / "r.json"; rec.write_text(_j.dumps(receipt), encoding="utf-8")
    return doc, [rec]


def test_a_decimal_matched_on_value_alone_is_not_called_a_kept_contract(tmp_path):
    """The blocker an adversarial merge review found, pinned.

    certify.py path-checks INTEGERS only (`if num["decimals"] == 0`), and the status-level float
    binding is shipped off (CLOSED_NEGATIVE, v0.8). So a decimal verifies on a bare value match
    against any leaf that is not one of sixteen position names. This module used to tell the
    author such a row "grounds at a receipt leaf whose path relates to this line. This is what a
    kept contract looks like" -- which is false, and false in exactly the way its own docstring
    warns about.
    """
    from styxx.oathready import readiness_report
    doc, recs = _mk(tmp_path, "The sycophancy rate was 0.5 on the held-out split.",
                    {"gpu_memory_fraction": 0.5})
    rep = readiness_report(doc, recs)
    row = next(r for r in rep["rows"] if r["token"] == "0.5")
    assert row["kind"] == "bound"
    assert row["path_checked"] is False
    assert "VALUE ONLY" in row["advice"]
    assert "path relates to this line" not in row["advice"]
    assert "kept contract" not in row["advice"]


def test_the_clean_run_summary_does_not_claim_relatedness_it_never_checked(tmp_path):
    """A clean report is the one place an author stops reading. It must not overstate there."""
    from styxx.oathready import readiness_report, render
    doc, recs = _mk(tmp_path, "The sycophancy rate was 0.5 on the held-out split.",
                    {"gpu_memory_fraction": 0.5})
    text = render(readiness_report(doc, recs))
    assert "related receipt leaf" not in text
    assert "VALUE ONLY" in text


def test_an_integer_that_was_actually_path_checked_still_says_so(tmp_path):
    """The fix must not flatten the honest case: integers DO go through v0.3 count-binding."""
    from styxx.oathready import readiness_report
    doc, recs = _mk(tmp_path, "Recall counted 27 held-out items.", {"n_held": 27})
    rep = readiness_report(doc, recs)
    row = next(r for r in rep["rows"] if r["token"] == "27")
    if row["kind"] == "bound":
        assert row["path_checked"] is True
        assert "word stem" in row["advice"]


def test_an_accusation_with_no_trigger_on_the_line_says_what_really_bound_it(tmp_path):
    """The second blocker from the merge review, pinned.

    certify.py obligates on precision alone at >= V07_PRECISION_DIGITS fractional digits,
    regardless of line vocabulary. This module used to tell every accused author to "reword so the
    vocabulary does not bind it" while naming "no word this tool can name" in the same sentence --
    advice that cannot be acted on. Measured on the external corpus: 180 of 366 accusations sit on
    a line naming no trigger at all. Internally it is 0 of 11, which is exactly why it went
    unnoticed: our own prose puts the word on the line.
    """
    from styxx.certify import V07_PRECISION_DIGITS
    from styxx.oathready import readiness_report
    tok = "0." + "1234567890"[:V07_PRECISION_DIGITS]        # precision-obligated, no vocabulary
    doc, recs = _mk(tmp_path, f"Legal scholars have long argued about {tok} in the abstract.",
                    {"unrelated": 1})
    rep = readiness_report(doc, recs)
    row = next((r for r in rep["rows"] if r["token"] == tok), None)
    assert row is not None and row["kind"] == "accused"
    assert row["obligated_by"] == [], "no trigger word is on this line"
    assert row["obligated_by_rule"] == "precision"
    assert "PRECISION" in row["advice"]
    assert "Rewording the sentence will not help" in row["advice"]


def test_an_accusation_that_does_name_a_trigger_still_gets_the_reword_advice(tmp_path):
    """The fix must not flatten the case where rewording IS the right move."""
    from styxx.oathready import readiness_report
    doc, recs = _mk(tmp_path, "The recall was 0.9 on the split.", {"unrelated": 1})
    rep = readiness_report(doc, recs)
    row = next((r for r in rep["rows"] if r["token"] == "0.9"), None)
    if row and row["kind"] == "accused":
        assert row["obligated_by"], "recall is on the line"
        assert "reword" in row["advice"]


def test_oathready_reads_the_verifier_epistemics_not_its_own_guess(tmp_path):
    """path_checked and obligation_source come from the annotation, not re-derivation.

    Before this, oathready inferred path_checked from '"." not in token' -- a second definition
    of the same concept, one drift away from lying to an adopter.
    """
    from styxx.oathready import readiness_report
    doc, recs = _mk(tmp_path, "Scholars argued about 0.4267 in the abstract.",
                    {"whatever": 0.4267})
    rep = readiness_report(doc, recs)
    row = next(r for r in rep["rows"] if r["token"] == "0.4267")
    assert row["kind"] == "bound"
    assert row["obligated"] is False, "the volunteered oath, from the annotation"
    assert row["obligation_source"] is None
    assert row["path_checked"] is False


def test_the_report_counts_volunteered_oaths_for_the_author(tmp_path):
    from styxx.oathready import readiness_report
    doc, recs = _mk(tmp_path, "The recall was 0.82 here.\nScholars argued about 0.4267 there.",
                    {"recall": 0.82, "whatever": 0.4267})
    rep = readiness_report(doc, recs)
    assert rep["volunteered_oaths"]["count"] == 1
    assert "0.3654" in rep["volunteered_oaths"]["meaning"]

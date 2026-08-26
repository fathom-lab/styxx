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

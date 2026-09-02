"""OATH v0.14 — range-sanity as a reporter, behind a flag that ships OFF.

PREREG_range_sanity_report_2026_09_02.md. The load-bearing tests: the default is OFF and the
committed corpus is untouched by construction; under the flag an out-of-range value is reported on
its ledger entry and not accused; the flag never creates an obligation of its own.
"""
from __future__ import annotations

import json

import pytest

import styxx.certify as C
from styxx.certify import certify_doc


@pytest.fixture
def doc(tmp_path):
    d = tmp_path / "d.md"
    d.write_text("The gate reached precision 4.0 on the held-out set\n", encoding="utf-8")
    r = tmp_path / "d_result.json"
    r.write_text(json.dumps({"precision": 4.0}), encoding="utf-8")
    return d, [r]


def test_the_flag_ships_off():
    assert C.V14_RANGE_SANITY_REPORT is False


def test_off_the_rule_forces_an_accusation_even_on_a_matching_leaf(doc, monkeypatch):
    monkeypatch.setattr(C, "V14_RANGE_SANITY_REPORT", False)
    cert = certify_doc(*doc)
    e, = cert["ledger"]
    assert e["status"] == "UNGROUNDED" and "range_flag" not in e
    assert cert["verdict"].startswith("OATH-FAILED")


def test_on_the_rule_reports_and_the_ladder_decides(doc, monkeypatch):
    monkeypatch.setattr(C, "V14_RANGE_SANITY_REPORT", True)
    cert = certify_doc(*doc)
    e, = cert["ledger"]
    assert e["range_flag"] == "out-of-range"
    assert e["status"] == "VERIFIED", "the receipt really holds 4.0; that is what the document says"
    assert e["epistemics"]["obligation_source"] == "vocabulary"


def test_on_an_out_of_range_token_with_no_other_obligation_abstains(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "V14_RANGE_SANITY_REPORT", True)
    d = tmp_path / "d.md"
    d.write_text("| model | AUC |\n|---|---|\n", encoding="utf-8")   # nothing obligating below
    d.write_text("An auc 4.0 typo in prose with no receipt leaf.\n", encoding="utf-8")
    r = tmp_path / "d_result.json"
    r.write_text(json.dumps({"other": 1}), encoding="utf-8")
    cert = certify_doc(d, [r])
    e, = cert["ledger"]
    # 'auc' obligates by vocabulary, so with no match this is still an accusation -- by vocabulary
    assert e["status"] == "UNGROUNDED" and e["range_flag"] == "out-of-range"
    assert e["epistemics"]["obligation_source"] == "vocabulary"
    d.write_text("A value of p 4.0 after nothing that obligates.\n", encoding="utf-8")
    cert = certify_doc(d, [r])
    e, = cert["ledger"]
    assert e["status"] == "ABSTAIN" and e["range_flag"] == "out-of-range"
    assert e["epistemics"]["obligated"] is False

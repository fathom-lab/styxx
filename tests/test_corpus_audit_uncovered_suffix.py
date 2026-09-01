"""The corpus auditor must read the v0.13 coverage suffix as coverage, never as a verdict.

`styxx.certify` now issues `OATH-HELD, 3 uncovered` when a document carries numeric spans the
extractor never examined (`tests/test_uncovered_band.py`). The suffix moves no token and no
count. The first auditor to meet it bucketed on the whole string: 131 of 208 certificates fell
into neither HELD nor FAILED, every one of them read as verdict drift, and the pinned audit line
in REPLICATIONS.md — `208 certificates | HELD 200  FAILED 8 ... verdict-drift 1` — stopped
reproducing on the day the band shipped. That is a verdict-shaped field being read by string
equality one module away from where its meaning changed: the corpus partitioned itself by
verifier version and the auditor had no name for the new stratum.

These tests pin the reading, on synthetic certificates so they cannot depend on the corpus.
"""
from __future__ import annotations

import json

import pytest

from styxx.corpus_audit import audit_corpus, audit_document, verdict_class


@pytest.mark.parametrize("verdict,expected", [
    ("OATH-HELD", "OATH-HELD"),
    ("OATH-HELD, 1 uncovered", "OATH-HELD"),
    ("OATH-FAILED, 12 uncovered", "OATH-FAILED"),
    ("OATH-FAILED", "OATH-FAILED"),
    (None, ""),
])
def test_the_class_is_the_verdict_with_the_coverage_suffix_stripped(verdict, expected):
    assert verdict_class(verdict) == expected


def _certify(tmp_path, name, text, receipt):
    """Write a doc + receipt, certify it with the live verifier, commit the certificate."""
    from styxx.certify import certify_doc
    doc = tmp_path / f"{name}.md"
    doc.write_text(text, encoding="utf-8")
    rec = tmp_path / f"{name}_result.json"
    rec.write_text(json.dumps(receipt), encoding="utf-8")
    cert = certify_doc(doc, [rec])
    cp = tmp_path / f"{name}.certificate.json"
    cp.write_text(json.dumps(cert, indent=1), encoding="utf-8")
    return cp, cert


def test_a_held_certificate_with_a_suffix_is_bucketed_held_and_is_not_drift(tmp_path):
    cp, cert = _certify(tmp_path, "suffixed", "recall 0.91 and precision 0.55.\n",
                        {"recall": 0.91})
    assert cert["verdict"] == "OATH-HELD, 1 uncovered", "premise: the band fires on this doc"
    rec = audit_document(cp, search_root=tmp_path)
    assert rec["status"] == "OK"
    assert rec["live_verdict_class"] == "OATH-HELD"
    assert rec["uncovered"] == 1
    assert rec["verdict_changed"] is False

    rep = audit_corpus(tmp_path)
    s = rep["summary"]
    assert (s["held"], s["failed"], s["unresolved"]) == (1, 0, 0)
    assert s["n_certificates"] == s["held"] + s["failed"] + s["unresolved"]
    assert s["verdict_changed"] == 0
    assert (s["uncovered_documents"], s["uncovered_spans"]) == (1, 1)


def test_a_committed_pre_band_verdict_reissued_with_a_suffix_is_not_drift(tmp_path):
    """The stratification case: a certificate issued BEFORE the band carries a bare verdict."""
    cp, cert = _certify(tmp_path, "legacy", "recall 0.91 and precision 0.55.\n", {"recall": 0.91})
    stored = json.loads(cp.read_text(encoding="utf-8"))
    stored["verdict"] = "OATH-HELD"                  # as a pre-v0.13 verifier would have written
    cp.write_text(json.dumps(stored), encoding="utf-8")
    rec = audit_document(cp, search_root=tmp_path)
    assert rec["recorded_verdict"] == "OATH-HELD"
    assert rec["live_verdict"] == "OATH-HELD, 1 uncovered"
    assert rec["verdict_changed"] is False, "coverage information is not a moved verdict"


def test_a_real_class_change_is_still_drift(tmp_path):
    cp, cert = _certify(tmp_path, "moved", "the recall is 4 and precision 0.55.\n", {"recall": 9})
    assert cert["verdict"] == "OATH-FAILED, 1 uncovered"
    stored = json.loads(cp.read_text(encoding="utf-8"))
    stored["verdict"] = "OATH-HELD"                  # somebody committed a HELD that no longer holds
    cp.write_text(json.dumps(stored), encoding="utf-8")
    rec = audit_document(cp, search_root=tmp_path)
    assert rec["verdict_changed"] is True
    s = audit_corpus(tmp_path)["summary"]
    assert (s["held"], s["failed"], s["verdict_changed"]) == (0, 1, 1)


def test_the_uncovered_total_is_beside_the_verdict_line_never_inside_it(tmp_path, capsys):
    from styxx.corpus_audit import main
    _certify(tmp_path, "a", "recall 0.91.\n", {"recall": 0.91})
    _certify(tmp_path, "b", "recall 0.91\n", {"recall": 0.91})
    main([str(tmp_path)])
    out = capsys.readouterr().out.splitlines()
    verdict_line = next(l for l in out if l.startswith("corpus "))
    # the root path precedes the colon (and pytest's tmp dir carries this test's name, which
    # contains the word we are asserting against), so inspect only the counts that follow it
    counts_part = verdict_line.split(" certificates | ", 1)[1]
    assert "HELD 2  FAILED 0  unresolved 0  verdict-drift 0" in counts_part
    assert "uncovered" not in counts_part
    assert any(l.strip().startswith("uncovered: 1 numeric spans across 1 documents") for l in out)

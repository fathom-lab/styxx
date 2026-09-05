"""SPEC_oath_receipt_binding_2026_09_04 — the tests the spec commits to, on a temporary git
repository built here. Never the corpus: the corpus is measured by the census, not by a test.

What is being pinned: (R1) a certificate names the bytes it swore to and says whether they were
committed; (R3) history gives every citation one of five cells; (R4) the audit re-derives over
the sworn bytes and says whether the certificate stands; (R5) history is optional and its absence
is printed on its own line while the pinned first line never moves; (R7) without git the
certificate differs only in its binding block.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from styxx import corpus_audit
from styxx import receipt_binding as rb
from styxx.certify import certify_doc, main as certify_main

DOC = ("# RESULT — a small run\n\n"
       "Fathom Lab. The battery passed 7 of 9 checks, a pass rate of 0.7778, over 3 seeds.\n")
RECEIPT_V1 = b'{"passed": 7, "total": 9, "pass_rate": 0.7778, "seeds": 3}\n'
# the same values, a different layout — a regeneration that moves bytes but no claim
RECEIPT_V2 = b'{\n  "passed": 7,\n  "total": 9,\n  "pass_rate": 0.7778,\n  "seeds": 3,\n  "note": "regenerated"\n}\n'
RECEIPT_OTHER = b'{"passed": 5, "total": 9, "pass_rate": 0.5556, "seeds": 3}\n'


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                        "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", *args],
                       capture_output=True, text=True, check=True)
    return p.stdout.strip()


@pytest.fixture
def lab(tmp_path: Path):
    """A repository with one document and one receipt, nothing committed yet."""
    repo = tmp_path / "lab"
    repo.mkdir()
    git(repo, "init", "-q")
    papers = repo / "papers"
    papers.mkdir()
    doc = papers / "RESULT_small.md"
    doc.write_text(DOC, encoding="utf-8", newline="\n")
    receipt = papers / "small_result.json"
    receipt.write_bytes(RECEIPT_V1)
    return repo, papers, doc, receipt


def _commit_all(repo: Path, msg: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def _issue(doc: Path, receipt: Path) -> Path:
    cert = certify_doc(doc, [receipt])
    cp = doc.with_suffix(".certificate.json")
    cp.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8", newline="\n")
    return cp


def _audit(repo: Path, cp: Path) -> dict:
    r, why = corpus_audit.open_history(repo, "on")
    assert r is not None, why
    try:
        rec = corpus_audit.audit_document(cp, search_root=repo / "papers", history=r)
    finally:
        r.close()
    return rec


# ---------------------------------------------------------------- R1

def test_mint_records_the_committed_blob(lab):
    repo, papers, doc, receipt = lab
    head = _commit_all(repo, "receipt and document")
    cert = certify_doc(doc, [receipt])
    b = cert["receipt_binding"]
    assert b["schema"] == rb.SCHEMA
    assert b["head"] == head
    assert b["all_receipts_committed"] is True
    (r,) = b["receipts"]
    assert r["name"] == "small_result.json" and r["path"] == "papers/small_result.json"
    assert r["committed"] is True
    assert r["blob"] == git(repo, "hash-object", str(receipt))
    assert r["raw_sha256"] == hashlib.sha256(RECEIPT_V1).hexdigest()
    assert r["content_sha256"] == hashlib.sha256(RECEIPT_V1.replace(b"\r\n", b"\n")).hexdigest()
    # receipts_sha256 is untouched: same key, same raw hash
    assert cert["receipts_sha256"] == {"small_result.json": r["raw_sha256"]}


def test_mint_over_a_modified_receipt_says_so_and_require_committed_refuses(lab, capsys):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipt and document")
    receipt.write_bytes(RECEIPT_V2)          # modified, not committed
    cert = certify_doc(doc, [receipt])
    b = cert["receipt_binding"]
    assert b["all_receipts_committed"] is False
    assert b["receipts"][0]["committed"] is False and b["receipts"][0]["blob"] is None
    out = papers / "refused.certificate.json"
    rc = certify_main([str(doc), str(receipt), "--require-committed", "--out", str(out)])
    assert rc == 2
    assert not out.exists()
    assert "small_result.json" in capsys.readouterr().err
    # without the flag the certificate issues and reports
    rc = certify_main([str(doc), str(receipt), "--out", str(out)])
    assert rc in (0, 1) and out.exists()


# ---------------------------------------------------------------- R3 + R4

def test_regenerated_receipt_reads_at_issue_and_the_certificate_stands(lab):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipt and document")
    cp = _issue(doc, receipt)
    issuing = _commit_all(repo, "certificate")
    receipt.write_bytes(RECEIPT_V2)          # regenerated in place, after issue
    _commit_all(repo, "regenerate the receipt")
    rec = _audit(repo, cp)
    # the working-tree audit sees drift, as before
    assert rec["receipt_drift"] == ["small_result.json"]
    b = rec["receipt_binding"]
    assert b["available"] is True
    assert b["issuing_commit"] == issuing
    (c,) = b["citations"]
    assert c["cell"] == "at_issue" and c["commit"] == issuing and c["normalisation"] == "raw"
    assert b["cells"] == {"same": 0, "at_issue": 1, "elsewhere": 0, "unbacked": 0,
                          "unrecoverable": 0}
    # re-derived over the bytes it swore to, the certificate stands
    assert b["stands_over_sworn_bytes"] is True
    assert corpus_audit.verdict_class(b["verdict_over_sworn_bytes"]) == \
        corpus_audit.verdict_class(json.loads(cp.read_text())["verdict"])


def test_same_citation_records_whether_the_issued_bytes_are_the_working_bytes(lab):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipt and document")
    cp = _issue(doc, receipt)
    _commit_all(repo, "certificate")
    b = _audit(repo, cp)["receipt_binding"]
    (c,) = b["citations"]
    assert c["cell"] == "same" and c["at_issue_too"] is True and c["blob"]
    assert b["stands_over_sworn_bytes"] is True


def test_sworn_bytes_never_committed_read_unbacked(lab):
    repo, papers, doc, receipt = lab
    cp = _issue(doc, receipt)                # receipt V1 is in the working tree only
    receipt.write_bytes(RECEIPT_OTHER)       # replaced before anything is committed
    _commit_all(repo, "document, certificate, and a receipt that is not the sworn one")
    rec = _audit(repo, cp)
    (c,) = rec["receipt_binding"]["citations"]
    assert c["cell"] == "unbacked"
    assert rec["receipt_binding"]["stands_over_sworn_bytes"] is None


def test_a_modified_certificate_has_no_issuing_commit(lab):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipt and document")
    cp = _issue(doc, receipt)
    _commit_all(repo, "certificate")
    cp.write_text(cp.read_text(encoding="utf-8") + "\n", encoding="utf-8")   # dirty
    b = _audit(repo, cp)["receipt_binding"]
    assert b["issuing_commit"] is None
    assert "modified" in b["unrecoverable_reason"]
    assert [c["cell"] for c in b["citations"]] == ["unrecoverable"]
    assert b["stands_over_sworn_bytes"] is None


def test_receipt_committed_after_the_certificate_reads_elsewhere_after(lab):
    repo, papers, doc, receipt = lab
    cp = _issue(doc, receipt)
    git(repo, "add", str(doc), str(cp))
    git(repo, "commit", "-q", "-m", "document and certificate, receipt left uncommitted")
    issuing = git(repo, "rev-parse", "HEAD")
    later = _commit_all(repo, "the receipt, one commit late")
    b = _audit(repo, cp)["receipt_binding"]
    (c,) = b["citations"]
    # the working tree still holds the sworn bytes, so the cell is `same` — but not at issue
    assert c["cell"] == "same" and c["at_issue_too"] is False
    # remove it from the working tree and the history answer is `elsewhere: after`
    receipt.unlink()
    _commit_all(repo, "receipt removed")
    b = _audit(repo, cp)["receipt_binding"]
    (c,) = b["citations"]
    assert c["cell"] == "elsewhere" and c["relation"] == "after"
    assert c["commit"] == later and b["issuing_commit"] == issuing


def test_a_shallow_clone_cannot_answer_and_says_so(lab, tmp_path):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "one")
    (papers / "x").write_text("x")
    _commit_all(repo, "two")
    shallow = tmp_path / "shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", repo.as_uri(), str(shallow)],
                   check=True, capture_output=True)
    r, why = corpus_audit.open_history(shallow, "on")
    assert r is None and why == "shallow clone"


# ---------------------------------------------------------------- newlines

def test_crlf_and_lf_are_the_same_content_and_the_record_names_the_reading():
    lf = b'{"n": 1}\n'
    crlf = b'{"n": 1}\r\n'
    assert rb.content_sha256(lf) == rb.content_sha256(crlf)
    assert rb.raw_sha256(lf) != rb.raw_sha256(crlf)
    assert rb.match_normalisation(lf, rb.raw_sha256(lf)) == "raw"
    assert rb.match_normalisation(crlf, rb.raw_sha256(lf)) == "lf"
    assert rb.match_normalisation(lf, rb.raw_sha256(crlf)) == "crlf"
    assert rb.match_normalisation(lf, "00" * 32) is None
    assert rb.git_blob_id(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


# ---------------------------------------------------------------- R5 + R7

def test_without_git_the_certificate_differs_only_in_its_binding_block(lab, monkeypatch):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipt and document")
    with_git = certify_doc(doc, [receipt])
    monkeypatch.setattr(rb.shutil, "which", lambda name: None)
    without = certify_doc(doc, [receipt])
    assert without["receipt_binding"]["head"] is None
    assert "no repository" in without["receipt_binding"]["note"]
    assert without["receipt_binding"]["all_receipts_committed"] is False
    strip = lambda c: {k: v for k, v in c.items() if k != "receipt_binding"}   # noqa: E731
    assert strip(with_git) == strip(without)


def test_history_off_keeps_the_pinned_first_line(lab, capsys):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipt and document")
    _issue(doc, receipt)
    _commit_all(repo, "certificate")
    corpus_audit.main([str(papers), "--history", "off"])
    off = capsys.readouterr().out.splitlines()
    corpus_audit.main([str(papers), "--history", "on"])
    on = capsys.readouterr().out.splitlines()
    assert off[0] == on[0]
    assert any(line.startswith("  binding: history unavailable (disabled") for line in off)
    assert any(line.startswith("  binding: citations same 1 ") for line in on)


def test_open_history_outside_a_repository_reports_the_reason(tmp_path):
    r, why = corpus_audit.open_history(tmp_path, "auto")
    assert r is None and "not inside a git repository" in why

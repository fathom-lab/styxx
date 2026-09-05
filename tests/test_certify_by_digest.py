"""SPEC_oath_receipt_binding_2026_09_04 (with its ERRATA) — the tests the spec commits to, plus
the constructions the adversarial battery added, on temporary git repositories built here. Only
the last test reads the tracked corpus, and it reads it by the census's own population rule.

What is being pinned: (R1) a certificate names the bytes it swore to and says whether they were
committed; (R3) history gives every citation one of five cells, and the document its own; (R4)
the audit re-derives over the sworn bytes and says whether the certificate stands, or why it
cannot say; (R5) history is optional and its absence is printed on its own line while the pinned
first line never moves; (R6) the census never overwrites a tracked result; (R7) without git the
certificate differs only in its binding block.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from styxx import corpus_audit
from styxx import receipt_binding as rb
from styxx.certify import certify_doc, main as certify_main

ROOT = Path(__file__).resolve().parents[1]

# a document the current verifier certifies OATH-HELD (the fixture test_corpus_audit.py uses),
# with a second receipt the document never cites so `same for the rest` has something to assert
DOC = ("# RESULT — a small run\n\npreamble sentence to avoid line-start filters.\n\n"
       "The detector reached AUROC 0.884 on the split.\n")
RECEIPT_V1 = b'{"auroc": 0.884}\n'
# the same value, a different layout — a regeneration that moves bytes but no claim
RECEIPT_V2 = b'{\n  "auroc": 0.884,\n  "note": "regenerated"\n}\n'
RECEIPT_V3 = b'{"auroc": 0.884, "note": "third"}\n'
RECEIPT_OTHER = b'{"unrelated": 1}\n'
EXTRA = b'{"seeds": 3}\n'


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                        "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false",
                        "-c", "core.quotepath=true", *args],
                       capture_output=True, text=True, check=True, encoding="utf-8")
    return p.stdout.strip()


@pytest.fixture
def lab(tmp_path: Path):
    """A repository with one document and two receipts, nothing committed yet."""
    repo = tmp_path / "lab"
    repo.mkdir()
    git(repo, "init", "-q")
    papers = repo / "papers"
    papers.mkdir()
    doc = papers / "RESULT_small.md"
    doc.write_text(DOC, encoding="utf-8", newline="\n")
    receipt = papers / "small_result.json"
    receipt.write_bytes(RECEIPT_V1)
    (papers / "extra_result.json").write_bytes(EXTRA)
    return repo, papers, doc, receipt


def _commit_all(repo: Path, msg: str) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def _issue(doc: Path, *receipts: Path) -> Path:
    receipts = receipts or (doc.parent / "small_result.json", doc.parent / "extra_result.json")
    cert = certify_doc(doc, list(receipts))
    cp = doc.with_suffix(".certificate.json")
    cp.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8", newline="\n")
    return cp


def _audit(repo: Path, cp: Path, root: Path = None) -> dict:
    r, why = corpus_audit.open_history(repo, "on")
    assert r is not None, why
    try:
        rec = corpus_audit.audit_document(cp, search_root=root or repo / "papers", history=r)
    finally:
        r.close()
    return rec


def _cells(rec: dict) -> dict:
    return {c["name"]: c["cell"] for c in rec["receipt_binding"]["citations"]}


# ---------------------------------------------------------------- R1

def test_mint_records_the_committed_blob(lab):
    repo, papers, doc, receipt = lab
    head = _commit_all(repo, "receipts and document")
    cert = certify_doc(doc, [receipt, papers / "extra_result.json"])
    assert corpus_audit.verdict_class(cert["verdict"]) == "OATH-HELD"
    b = cert["receipt_binding"]
    assert b["schema"] == rb.SCHEMA
    assert b["head"] == head
    assert b["all_receipts_committed"] is True
    r, e = b["receipts"]
    assert r["name"] == "small_result.json" and r["path"] == "papers/small_result.json"
    assert r["committed"] is True and e["committed"] is True
    assert r["blob"] == git(repo, "hash-object", str(receipt))
    assert r["raw_sha256"] == hashlib.sha256(RECEIPT_V1).hexdigest()
    assert r["content_sha256"] == hashlib.sha256(RECEIPT_V1.replace(b"\r\n", b"\n")).hexdigest()
    # receipts_sha256 is untouched: same keys, same raw hashes
    assert cert["receipts_sha256"] == {"small_result.json": r["raw_sha256"],
                                       "extra_result.json": e["raw_sha256"]}


def test_mint_over_a_modified_receipt_says_so_and_require_committed_refuses(lab, capsys):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipts and document")
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


def test_mint_with_no_receipts_still_records_head(lab):
    """Battery B-12: an empty receipt list used to leave head null with no note, which is
    indistinguishable from a mint outside any repository."""
    repo, papers, doc, receipt = lab
    head = _commit_all(repo, "receipts and document")
    cert = certify_doc(doc, [])
    b = cert["receipt_binding"]
    assert b["head"] == head and b["receipts"] == [] and b["note"] == "no receipts"
    assert b["all_receipts_committed"] is False


# ---------------------------------------------------------------- R3 + R4

def test_regenerated_receipt_reads_at_issue_and_the_certificate_stands(lab):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipts and document")
    cp = _issue(doc)
    issuing = _commit_all(repo, "certificate")
    receipt.write_bytes(RECEIPT_V2)          # regenerated in place, after issue
    _commit_all(repo, "regenerate the receipt")
    rec = _audit(repo, cp)
    # the working-tree audit sees drift, as before
    assert rec["receipt_drift"] == ["small_result.json"]
    b = rec["receipt_binding"]
    assert b["available"] is True
    assert b["issuing_commit"] == issuing
    assert _cells(rec) == {"small_result.json": "at_issue", "extra_result.json": "same"}
    c = b["citations"][0]
    # a certificate that carries its own binding block is matched on content_sha256 first
    assert c["commit"] == issuing and c["normalisation"] == "content"
    assert c["path"] == "papers/small_result.json"
    assert b["cells"] == {"same": 1, "at_issue": 1, "elsewhere": 0, "unbacked": 0,
                          "unrecoverable": 0}
    assert b["document"]["cell"] == "same" and b["document"]["at_issue_too"] is True
    # re-derived over the bytes it swore to, the certificate stands — and HELD, not vacuously
    assert b["stands_over_sworn_bytes"] is True
    assert corpus_audit.verdict_class(b["verdict_over_sworn_bytes"]) == "OATH-HELD"
    assert corpus_audit.verdict_class(json.loads(cp.read_text())["verdict"]) == "OATH-HELD"


def test_same_citation_records_whether_the_issued_bytes_are_the_working_bytes(lab):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipts and document")
    cp = _issue(doc)
    _commit_all(repo, "certificate")
    b = _audit(repo, cp)["receipt_binding"]
    for c in b["citations"]:
        assert c["cell"] == "same" and c["at_issue_too"] is True and c["blob"]
        assert c["blob_normalisation"] == "content"   # the certificate's own digest, read first
    assert b["stands_over_sworn_bytes"] is True


def test_sworn_bytes_never_committed_read_unbacked(lab):
    repo, papers, doc, receipt = lab
    cp = _issue(doc)                         # receipt V1 is in the working tree only
    receipt.write_bytes(RECEIPT_OTHER)       # replaced before anything is committed
    _commit_all(repo, "document, certificate, and a receipt that is not the sworn one")
    rec = _audit(repo, cp)
    assert _cells(rec) == {"small_result.json": "unbacked", "extra_result.json": "same"}
    assert rec["receipt_binding"]["stands_over_sworn_bytes"] is None
    assert rec["receipt_binding"]["stands_reason"] == "a citation is unbacked"


def test_a_modified_certificate_has_no_issuing_commit(lab):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipts and document")
    cp = _issue(doc)
    _commit_all(repo, "certificate")
    cp.write_text(cp.read_text(encoding="utf-8") + "\n", encoding="utf-8")   # dirty
    b = _audit(repo, cp)["receipt_binding"]
    assert b["issuing_commit"] is None
    assert "modified" in b["unrecoverable_reason"]
    assert set(_cells({"receipt_binding": b}).values()) == {"unrecoverable"}
    assert b["stands_over_sworn_bytes"] is None and b["stands_reason"] == "no issuing commit"


def test_receipt_committed_after_the_certificate_reads_elsewhere_after_and_still_stands(lab):
    repo, papers, doc, receipt = lab
    cp = _issue(doc)
    git(repo, "add", str(doc), str(cp), str(papers / "extra_result.json"))
    git(repo, "commit", "-q", "-m", "document and certificate, receipt left uncommitted")
    issuing = git(repo, "rev-parse", "HEAD")
    later = _commit_all(repo, "the receipt, one commit late")
    b = _audit(repo, cp)["receipt_binding"]
    c = b["citations"][0]
    # the working tree still holds the sworn bytes, so the cell is `same` — but not at issue
    assert c["cell"] == "same" and c["at_issue_too"] is False
    # remove it from the working tree and the history answer is `elsewhere: after`; the blob
    # is known, so the verdict is still re-derived (ERRATA to R4)
    receipt.unlink()
    _commit_all(repo, "receipt removed")
    b = _audit(repo, cp)["receipt_binding"]
    c = b["citations"][0]
    assert c["cell"] == "elsewhere" and c["relation"] == "after"
    assert c["commit"] == later and b["issuing_commit"] == issuing
    assert b["stands_over_sworn_bytes"] is True


def test_a_treesame_merge_does_not_hide_the_sworn_bytes(lab):
    """Battery B-01/B-02: the sworn bytes live only on a side branch that was merged with
    `-s ours`, so default history simplification never lists the commit that holds them."""
    repo, papers, doc, receipt = lab
    _commit_all(repo, "V1")
    git(repo, "checkout", "-q", "-b", "side")
    receipt.write_bytes(RECEIPT_V3)
    side = _commit_all(repo, "V3 on the side")
    git(repo, "checkout", "-q", "-")
    receipt.write_bytes(RECEIPT_V3)          # mint over V3 on main without committing V3 …
    cp = _issue(doc)
    receipt.write_bytes(RECEIPT_V2)          # … then commit V2 beside the certificate
    _commit_all(repo, "certificate over V3, receipt V2")
    git(repo, "merge", "-q", "-s", "ours", "--no-edit", "side")
    rec = _audit(repo, cp)
    c = rec["receipt_binding"]["citations"][0]
    assert c["cell"] == "elsewhere" and c["commit"] == side and c["relation"] == "unrelated"


@pytest.mark.parametrize("name", ["résumé_result.json", "result[1].json", "a*b?.json"])
def test_awkward_receipt_names_are_found_in_history(lab, name):
    """Battery B-03/B-04: core.quotepath octal-escapes non-ASCII paths and glob metacharacters
    in a basename change what the pathspec matches."""
    repo, papers, doc, receipt = lab
    if sys.platform == "win32" and any(ch in name for ch in "*?"):
        pytest.skip("NTFS refuses * and ? in file names")
    odd = papers / name
    odd.write_bytes(RECEIPT_V1)
    _commit_all(repo, "odd receipt")
    cp = _issue(doc, odd)
    odd.write_bytes(RECEIPT_V2)
    _commit_all(repo, "certificate, and the odd receipt regenerated in the same commit")
    rec = _audit(repo, cp)
    c = rec["receipt_binding"]["citations"][0]
    assert c["cell"] == "elsewhere" and c["relation"] == "before", c


def test_a_missing_document_still_gets_its_cells(lab):
    """Battery B-10: the MISSING_DOC early return ran before the binding was computed."""
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipts and document")
    cp = _issue(doc)
    _commit_all(repo, "certificate")
    doc.unlink()
    _commit_all(repo, "document removed")
    rec = _audit(repo, cp)
    assert rec["status"] == "MISSING_DOC"
    assert _cells(rec) == {"small_result.json": "same", "extra_result.json": "same"}
    assert rec["receipt_binding"]["document"]["cell"] == "at_issue"
    assert rec["receipt_binding"]["stands_over_sworn_bytes"] is True


def test_sworn_bytes_outside_the_audit_root_read_same(lab):
    """Battery A6/ES-06: a synthesis citing another arc's receipt read `at_issue` under a
    subdirectory root while the bytes sat unchanged in the working tree."""
    repo, papers, doc, receipt = lab
    other = repo / "other"
    other.mkdir()
    far = other / "far_result.json"
    far.write_bytes(EXTRA)
    _commit_all(repo, "a receipt in another arc")
    cp = _issue(doc, receipt, far)
    _commit_all(repo, "certificate")
    rec = _audit(repo, cp, root=papers)
    c = {x["name"]: x for x in rec["receipt_binding"]["citations"]}["far_result.json"]
    assert c["cell"] == "same" and c["path"] == "other/far_result.json"
    assert c["note"] == "resolved outside the audit root" and c["at_issue_too"] is True
    assert rec["receipt_binding"]["stands_over_sworn_bytes"] is True


def test_a_document_edited_after_issue_reads_at_issue_and_the_certificate_stands(lab):
    """Battery ES-01/ES-03: the document is one of the sworn bytes."""
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipts and document")
    cp = _issue(doc)
    _commit_all(repo, "certificate")
    doc.write_text(DOC.replace("0.884", "0.900"), encoding="utf-8", newline="\n")
    _commit_all(repo, "document edited after issue")
    b = _audit(repo, cp)["receipt_binding"]
    assert b["document"]["cell"] == "at_issue"
    assert b["stands_over_sworn_bytes"] is True     # over the document it swore to
    # a cosmetic rewrite of the certificate now moves I(C) past the edit: the document at the
    # new issuing commit is not the sworn one, and the audit says so instead of guessing
    cp.write_text(json.dumps(json.loads(cp.read_text(encoding="utf-8")), indent=1) + "\n",
                  encoding="utf-8", newline="\n")
    _commit_all(repo, "cosmetic rewrite of the certificate")
    b = _audit(repo, cp)["receipt_binding"]
    assert b["document"]["cell"] == "moved"
    assert b["stands_over_sworn_bytes"] is None
    assert b["stands_reason"] == "document at issuing commit is not the sworn document"


def test_case_insensitive_names_resolve_at_issue(lab):
    """Battery B-11: a receipts_sha256 key that differs from the file only by case."""
    repo, papers, doc, receipt = lab
    if git(repo, "config", "--get", "core.ignorecase") != "true":
        pytest.skip("case-sensitive filesystem")
    _commit_all(repo, "receipts and document")
    cp = _issue(doc, receipt)
    cert = json.loads(cp.read_text(encoding="utf-8"))
    cert["receipts_sha256"] = {"SMALL_RESULT.json": cert["receipts_sha256"]["small_result.json"]}
    cert.pop("receipt_binding")
    cp.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8", newline="\n")
    _commit_all(repo, "certificate with an upper-cased key")
    c = _audit(repo, cp)["receipt_binding"]["citations"][0]
    assert c["cell"] == "same" and c["at_issue_too"] is True and c["blob"]


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
    mixed = b'{"n": 1,\r\n "m": 2}\n'
    assert rb.content_sha256(lf) == rb.content_sha256(crlf)
    assert rb.raw_sha256(lf) != rb.raw_sha256(crlf)
    assert rb.match_normalisation(lf, rb.raw_sha256(lf)) == "raw"
    assert rb.match_normalisation(crlf, rb.raw_sha256(lf)) == "lf"
    assert rb.match_normalisation(lf, rb.raw_sha256(crlf)) == "crlf"
    assert rb.match_normalisation(lf, "00" * 32) is None
    # a mixed-newline receipt matches no legacy reading, and only its own content digest
    assert rb.match_normalisation(mixed.replace(b"\r\n", b"\n"), rb.raw_sha256(mixed)) is None
    assert rb.match_normalisation(mixed.replace(b"\r\n", b"\n"), rb.raw_sha256(mixed),
                                  content=rb.content_sha256(mixed)) == "content"
    assert rb.git_blob_id(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


# ---------------------------------------------------------------- R5 + R7

def test_without_git_the_certificate_differs_only_in_its_binding_block(lab, monkeypatch):
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipts and document")
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
    _commit_all(repo, "receipts and document")
    _issue(doc)
    _commit_all(repo, "certificate")
    assert corpus_audit.main([str(papers), "--history", "off"]) == 0
    off = capsys.readouterr().out.splitlines()
    assert corpus_audit.main([str(papers), "--history", "on"]) == 0
    on = capsys.readouterr().out.splitlines()
    assert off[0] == on[0]
    assert any(line.startswith("  binding: history unavailable (disabled") for line in off)
    assert any(line.startswith("  binding: over 1/1 certificates — citations same 2 ") for line in on)


def test_history_on_outside_a_repository_prints_the_reason_and_exits_2(tmp_path, capsys):
    r, why = corpus_audit.open_history(tmp_path, "auto")
    assert r is None and "not inside a git repository" in why
    assert corpus_audit.main([str(tmp_path), "--history", "on"]) == 2
    assert "binding: history unavailable" in capsys.readouterr().out
    assert corpus_audit.main([str(tmp_path), "--history", "auto"]) == 0


# ---------------------------------------------------------------- R6

def _census_module():
    census = ROOT / "papers" / "closed-model-frontier" / "receipt_binding_census.py"
    if not census.exists():
        pytest.skip("census script not in this checkout")
    import importlib.util
    spec = importlib.util.spec_from_file_location("receipt_binding_census", census)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_census_refuses_to_overwrite_a_tracked_result(lab, capsys):
    """Battery A7: a committed census is history whether or not a RESULT has sworn to it."""
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipts and document")
    _issue(doc)
    _commit_all(repo, "certificate")
    census = _census_module()
    out = repo / "census_result.json"
    assert census.main(["--root", str(repo), "--out", str(out)]) == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["population"]["n"] == 1 and result["citations"]["same"] == 2
    assert result["provenance"]["code_committed_at_head"] is False   # no styxx code in this repo
    assert census.main(["--root", str(repo), "--out", str(out)]) == 0   # untracked: rewrite allowed
    _commit_all(repo, "the census")
    assert census.main(["--root", str(repo), "--out", str(out)]) == 2
    assert "never regenerated in place" in capsys.readouterr().err
    assert census.main(["--root", str(repo), "--out", str(repo / "census_2.json")]) == 0


def test_the_census_output_is_cited_by_no_tracked_certificate():
    """The census is subject to the rule it measures. Population = the census's own rule
    (git ls-files), so the two certificates outside papers/ are counted (battery ES-15)."""
    _census_module()
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.certificate.json"],
                         capture_output=True, check=True).stdout
    certs = [p.decode("utf-8") for p in out.split(b"\0") if p]
    if not certs:
        pytest.skip("no tracked certificates (not a checkout)")
    cited = set()
    for rel in certs:
        try:
            cited |= set(json.loads((ROOT / rel).read_text(encoding="utf-8")).get("receipts_sha256", {}))
        except Exception:
            continue
    assert "receipt_binding_census_result.json" not in cited
    assert "receipt_binding_census.py" not in cited


def test_a_staging_copy_whose_document_lives_elsewhere_reads_same(lab):
    """The arXiv staging copies: a certificate copied beside a document that is not there, whose
    `document` field names a file living unchanged under papers/. The second census read two of
    them as edited-after-issue; the document cell must find the working file by name."""
    repo, papers, doc, receipt = lab
    _commit_all(repo, "receipts and document")
    cp = _issue(doc)
    staging = repo / "staging"
    staging.mkdir()
    copy = staging / "source.certificate.json"
    copy.write_bytes(cp.read_bytes())
    _commit_all(repo, "certificate and its staging copy")
    b = _audit(repo, copy)["receipt_binding"]
    d = b["document"]
    assert d["cell"] == "same" and d["path"] == "papers/RESULT_small.md"
    assert d["note"].startswith("resolved by the certificate's document field")
    assert d["at_issue_too"] is True
    assert b["stands_over_sworn_bytes"] is True
    # and once the document is edited, the staging copy reads at_issue like its original
    doc.write_text(DOC.replace("0.884", "0.900"), encoding="utf-8", newline="\n")
    _commit_all(repo, "document edited after issue")
    assert _audit(repo, copy)["receipt_binding"]["document"]["cell"] == "at_issue"

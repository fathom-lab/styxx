"""Tests for styxx.corpus_audit — the standing corpus re-certification tool."""
import json

from styxx.certify import certify_doc
from styxx.corpus_audit import audit_corpus, audit_document, mutate_token
import random


def _make_doc(tmp_path, name, line, receipt_obj):
    doc = tmp_path / f"{name}.md"
    doc.write_text(f"# {name}\n\npreamble sentence to avoid line-start filters.\n\n{line}\n",
                   encoding="utf-8")
    rp = tmp_path / f"{name}_result.json"
    rp.write_text(json.dumps(receipt_obj), encoding="utf-8")
    cert = certify_doc(doc, [rp])
    (tmp_path / f"{name}.certificate.json").write_text(json.dumps(cert), encoding="utf-8")
    return doc


def test_audit_corpus_finds_held(tmp_path):
    _make_doc(tmp_path, "good", "The detector reached AUROC 0.884 on the split.", {"auroc": 0.884})
    rep = audit_corpus(tmp_path)
    assert rep["summary"]["n_certificates"] == 1
    assert rep["summary"]["held"] == 1
    assert rep["summary"]["failed"] == 0
    assert rep["documents"][0]["live_verdict"] == "OATH-HELD"


def test_audit_corpus_flags_failed(tmp_path):
    # a doc whose AUROC is NOT in its receipt -> OATH-FAILED under the current verifier
    doc = tmp_path / "bad.md"
    doc.write_text("# bad\n\npreamble.\n\nThe detector reached AUROC 0.884 on the split.\n",
                   encoding="utf-8")
    rp = tmp_path / "bad_result.json"
    rp.write_text(json.dumps({"unrelated": 1}), encoding="utf-8")
    cert = certify_doc(doc, [rp])
    (tmp_path / "bad.certificate.json").write_text(json.dumps(cert), encoding="utf-8")
    rep = audit_corpus(tmp_path)
    assert rep["summary"]["failed"] == 1
    assert rep["documents"][0]["live_verdict"] == "OATH-FAILED"


def test_receipt_drift_detected(tmp_path):
    _make_doc(tmp_path, "drifty", "The detector reached AUROC 0.884 on the split.", {"auroc": 0.884})
    # mutate the receipt after the certificate was written -> SHA drift
    (tmp_path / "drifty_result.json").write_text(json.dumps({"auroc": 0.999}), encoding="utf-8")
    rep = audit_corpus(tmp_path)
    assert rep["summary"]["receipt_drift"] == 1
    assert rep["documents"][0]["receipt_drift"] == ["drifty_result.json"]


def test_tamper_battery_catches_corruption(tmp_path):
    _make_doc(tmp_path, "t", "The detector reached AUROC 0.884 on the split.", {"auroc": 0.884})
    rep = audit_corpus(tmp_path, tamper=True, seed=1)
    t = rep["summary"]["tamper"]
    assert t["n_mutants"] >= 1
    # the single verified AUROC, when corrupted, is no longer grounded -> caught
    assert t["caught"] >= 1


def test_mutate_token_changes_a_digit():
    rng = random.Random(1)
    out = mutate_token("0.884", rng)
    assert out != "0.884" and len(out) == len("0.884")


def test_cross_directory_receipt_resolves_by_sha(tmp_path):
    """A receipt living in a SIBLING folder must resolve (sha-verified), not report missing.

    Regression: corpus_audit resolved receipts only next to the doc, so a synthesis citing
    arcs from several folders re-certified against a crippled receipt set and produced a
    spurious OATH-FAILED on a document whose committed certificate is HELD. 38 documents in
    the styxx corpus were affected.
    """
    import hashlib, json
    from pathlib import Path
    from styxx.corpus_audit import _resolve_receipts

    (tmp_path / "arcA").mkdir()
    (tmp_path / "arcB").mkdir()
    receipt = tmp_path / "arcB" / "far_result.json"
    receipt.write_text(json.dumps({"auc": 0.9}), encoding="utf-8")
    sha = hashlib.sha256(receipt.read_bytes()).hexdigest()

    cert_path = tmp_path / "arcA" / "DOC.certificate.json"
    cert = {"receipts_sha256": {"far_result.json": sha}}
    cert_path.write_text(json.dumps(cert), encoding="utf-8")

    # without a search root: correctly reported missing (old behavior preserved)
    paths, missing, drift = _resolve_receipts(cert_path, cert)
    assert missing == ["far_result.json"] and not paths

    # with a search root: resolves, sha-verified
    paths, missing, drift = _resolve_receipts(cert_path, cert, search_root=tmp_path)
    assert not missing and not drift and len(paths) == 1
    assert paths[0].name == "far_result.json"


def test_cross_directory_wrong_sha_does_not_resolve(tmp_path):
    """A same-named file with DIFFERENT content must NOT satisfy the receipt — the search is
    stricter than location-trust, not looser."""
    import hashlib, json
    from styxx.corpus_audit import _resolve_receipts

    (tmp_path / "arcA").mkdir()
    (tmp_path / "arcB").mkdir()
    (tmp_path / "arcB" / "far_result.json").write_text('{"auc": 0.1}', encoding="utf-8")
    cert_path = tmp_path / "arcA" / "DOC.certificate.json"
    cert = {"receipts_sha256": {"far_result.json": hashlib.sha256(b'{"auc": 0.9}').hexdigest()}}
    cert_path.write_text(json.dumps(cert), encoding="utf-8")

    paths, missing, drift = _resolve_receipts(cert_path, cert, search_root=tmp_path)
    assert missing == ["far_result.json"] and not paths


def test_anc_packaging_mirrors_are_skipped(tmp_path):
    """arXiv staging mirrors a certificate into submission/anc/ beside a renamed source.md
    that does not exist; auditing the mirror reported phantom MISSING_DOC. The discoverer
    skips any path with an ``anc`` segment; canonical certificates elsewhere still audit."""
    import json
    from styxx.corpus_audit import discover_certificates

    real = tmp_path / "arcA"
    real.mkdir()
    (real / "DOC.certificate.json").write_text("{}", encoding="utf-8")
    mirror = tmp_path / "arxiv" / "paper" / "submission" / "anc"
    mirror.mkdir(parents=True)
    (mirror / "source.certificate.json").write_text("{}", encoding="utf-8")

    found = discover_certificates(tmp_path)
    assert [p.name for p in found] == ["DOC.certificate.json"]

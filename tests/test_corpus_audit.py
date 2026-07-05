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

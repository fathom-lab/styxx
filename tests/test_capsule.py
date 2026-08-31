"""OATH Capsule v0.1: the tamper battery.

A capsule must round-trip (mint → verify OK), refuse to mint around a lie, and catch every
class of post-mint tampering its layers claim to catch: flipped document bytes, doctored
receipts, and a hand-edited embedded ledger. The CRLF case is here because the certificate
hashes the document as certify_doc READ it (universal newlines), not as the filesystem
stores it — the capsule embeds the canonical bytes, so a CRLF checkout must still mint.

Spec: papers/closed-model-frontier/SPEC_oath_capsule_v01_2026_08_31.md
"""
from __future__ import annotations

import base64
import json

import pytest

from styxx.capsule import _BEGIN, _END, create_capsule, verify_capsule
from styxx.certify import certify_doc

DOC = "The run scored 0.75 accuracy over 40 items.\n"
RECEIPT = {"eval": {"accuracy": 0.75, "items": 40}}


@pytest.fixture
def minted(tmp_path):
    """A freshly minted capsule over a tiny certified fixture."""
    doc = tmp_path / "d.md"
    doc.write_text(DOC, encoding="utf-8")
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cert = certify_doc(doc, [rec])
    cp = tmp_path / "d.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    out = tmp_path / "d.capsule.html"
    create_capsule(doc, [rec], cp, out)
    return out


def _payload(path):
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    j = html.index(_END, i)
    return html, i, j, json.loads(html[i:j])


def _rewrite(path, html, i, j, payload):
    body = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(html[:i] + body + html[j:], encoding="utf-8")


# ---------------------------------------------------------------- round trip

def test_round_trip_verifies(minted):
    r = verify_capsule(minted)
    assert r["ok"], r["problems"]
    assert r["problems"] == []
    assert r["verdict"] in ("OATH-HELD", "OATH-FAILED")
    assert r["reproduced_at"] == "installed verifier"


def test_capsule_is_selfcontained_html(minted):
    html = minted.read_text(encoding="utf-8")
    assert _BEGIN in html
    assert "__PAYLOAD__" not in html          # payload actually injected
    assert "__PIP__" not in html              # static pip line wired
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "https://" not in html.replace("https://www.w3.org", "")  # zero external requests


def test_crlf_checkout_still_mints(tmp_path):
    """CRLF on disk, LF in the certificate hash — the capsule bridges the two."""
    doc = tmp_path / "d.md"
    doc.write_bytes(DOC.replace("\n", "\r\n").encode("utf-8"))
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cert = certify_doc(doc, [rec])
    cp = tmp_path / "d.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    out = tmp_path / "d.capsule.html"
    create_capsule(doc, [rec], cp, out)
    r = verify_capsule(out)
    assert r["ok"], r["problems"]
    # and the embedded bytes are the canonical ones the certificate hashed
    _, _, _, payload = _payload(out)
    assert b"\r" not in base64.b64decode(payload["document"]["b64"])


# ---------------------------------------------------------------- tampering

def test_tampered_document_byte_fails(minted):
    html, i, j, payload = _payload(minted)
    raw = bytearray(base64.b64decode(payload["document"]["b64"]))
    raw[0] ^= 0x01                            # flip one bit of one byte
    payload["document"]["b64"] = base64.b64encode(bytes(raw)).decode("ascii")
    _rewrite(minted, html, i, j, payload)
    r = verify_capsule(minted)
    assert not r["ok"]
    assert any("document bytes" in p for p in r["problems"])


def test_tampered_receipt_fails(minted):
    html, i, j, payload = _payload(minted)
    doctored = dict(RECEIPT)
    doctored["eval"] = {"accuracy": 0.99, "items": 40}
    payload["receipts"][0]["b64"] = base64.b64encode(
        json.dumps(doctored).encode("utf-8")).decode("ascii")
    _rewrite(minted, html, i, j, payload)
    r = verify_capsule(minted)
    assert not r["ok"]
    assert any("receipt" in p for p in r["problems"])


def test_hand_edited_ledger_fails(minted):
    """Forge the embedded certificate's ledger — hashes still match, layer 2 catches it."""
    html, i, j, payload = _payload(minted)
    entry = payload["certificate"]["ledger"][0]
    flipped = "ABSTAIN" if entry["status"] != "ABSTAIN" else "VERIFIED"
    old = entry["status"]
    entry["status"] = flipped
    payload["certificate"]["counts"][old] -= 1
    payload["certificate"]["counts"][flipped] = (
        payload["certificate"]["counts"].get(flipped, 0) + 1)
    _rewrite(minted, html, i, j, payload)
    r = verify_capsule(minted)
    assert not r["ok"]
    assert any("counts not reproduced" in p or "ledger divergence" in p
               for p in r["problems"])


def test_hand_edited_verdict_fails(minted):
    html, i, j, payload = _payload(minted)
    payload["certificate"]["verdict"] = (
        "OATH-HELD" if payload["certificate"]["verdict"] != "OATH-HELD" else "OATH-FAILED")
    _rewrite(minted, html, i, j, payload)
    r = verify_capsule(minted)
    assert not r["ok"]
    assert any("verdict" in p for p in r["problems"])


def test_garbage_file_fails_at_parse():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.html"
        p.write_text("<html>not a capsule</html>", encoding="utf-8")
        r = verify_capsule(p)
    assert not r["ok"]
    assert r["stage"] == "parse"


# ---------------------------------------------------------------- creation refuses to lie

def test_create_refuses_stale_certificate(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text(DOC, encoding="utf-8")
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cert = certify_doc(doc, [rec])
    cp = tmp_path / "d.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    doc.write_text(DOC.replace("0.75", "0.80"), encoding="utf-8")   # doc moved on
    with pytest.raises(SystemExit, match="document bytes do not match"):
        create_capsule(doc, [rec], cp, tmp_path / "d.capsule.html")


def test_create_refuses_missing_receipt(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text(DOC, encoding="utf-8")
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cert = certify_doc(doc, [rec])
    cp = tmp_path / "d.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    with pytest.raises(SystemExit, match="receipts not provided"):
        create_capsule(doc, [], cp, tmp_path / "d.capsule.html")


def test_create_refuses_doctored_receipt(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text(DOC, encoding="utf-8")
    rec = tmp_path / "r.json"
    rec.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cert = certify_doc(doc, [rec])
    cp = tmp_path / "d.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    rec.write_text(json.dumps({"eval": {"accuracy": 0.99}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="do not match the certificate"):
        create_capsule(doc, [rec], cp, tmp_path / "d.capsule.html")

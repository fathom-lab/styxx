"""`sworn check` must say what the DOCUMENT's verdict was, not only whether the receipt re-derived.

WHY THIS TEST EXISTS. Two different questions live one word apart:

    VERIFIED / FAILED                      does this RECEIPT re-derive from the document it names?
    SWORN-HELD / SWORN-FAILED / UNSWORN    did the DOCUMENT's sentences hold against their receipts?

`check` used to print only the first. On 2026-09-06, in this repository, its author ran

    VERIFIED  digest=True verdict-reproduces=True same-build=True

on a document that was SWORN-FAILED — one span printed 5 against a receipt reading 0 — and reported
it as held. The receipt had re-derived perfectly; that is all VERIFIED ever meant, and the line did
not say so.

So the document verdict rides on the same line now, and this pins it. The exit code deliberately
does NOT change: `check` exits 1 only when a receipt fails to re-derive. A document that honestly
reports SWORN-FAILED is a working document, and this CLI reports rather than gates.
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    return subprocess.run([sys.executable, "-m", "styxx.sworn", *args],
                          cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)


def _make(tmp_path, body: bytes):
    doc = tmp_path / "d.md"
    doc.write_bytes(body)
    rec = tmp_path / "d.sworn-receipt.json"
    r = _run("verify", str(doc), "--out", str(rec))
    assert rec.exists(), "verify wrote no receipt:\n%s\n%s" % (r.stdout, r.stderr)
    return doc, rec, json.loads(rec.read_text(encoding="utf-8"))


def test_check_prints_the_document_verdict_for_a_document_that_held(tmp_path):
    """An UNSWORN document is the simplest one whose verdict is not the receipt's status."""
    doc, rec, obj = _make(tmp_path, b"a document that swears to nothing at all.\n")
    r = _run("check", str(rec), str(doc))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "VERIFIED" in r.stdout, r.stdout
    assert obj["document_verdict"] in r.stdout, (
        "check printed no document verdict; a reader sees VERIFIED and cannot tell whether the "
        "document held:\n%s" % r.stdout)


def test_check_prints_SWORN_FAILED_and_still_exits_zero(tmp_path):
    """The exact confusion this test exists for: a receipt that re-derives over a document that
    did NOT hold. VERIFIED must appear, SWORN-FAILED must appear, and the exit code must stay 0 —
    the receipt is fine, and `check` reports rather than gates.

    The document must genuinely FAIL, so it needs a manifest with a receipt whose value contradicts
    the sentence. Skipping here would leave the one case this file exists for untested.
    """
    payload = b'{"n": 3}'
    man = tmp_path / "m.json"
    man.write_text(json.dumps({
        "spec": "sworn/manifest/0.2", "harness": "pytest", "turn": "t",
        "minted_at": "2026-09-01T00:00:00Z", "authored_sha256": [], "rung": "L1",
        "receipts": {"r1": {
            "id": "r1", "sha256": hashlib.sha256(payload).hexdigest(),
            "kind_of_source": "tool_stdout", "captured_at": "2026-09-01T00:00:00Z",
            "complete": True, "bytes": base64.b64encode(payload).decode("ascii"),
        }},
    }), encoding="utf-8")

    doc = tmp_path / "d.md"
    doc.write_bytes(b'<sworn r="r1#/n" k="numeric">the value is 7.</sworn>\n')
    rec = tmp_path / "d.sworn-receipt.json"
    v = _run("verify", str(doc), "--manifest", str(man), "--out", str(rec))
    assert rec.exists(), "verify wrote no receipt:\n%s\n%s" % (v.stdout, v.stderr)
    obj = json.loads(rec.read_text(encoding="utf-8"))
    assert obj["document_verdict"] == "SWORN-FAILED", (
        "the fixture was meant to FAIL (receipt says 3, sentence says 7) but got %r"
        % obj["document_verdict"])

    r = _run("check", str(rec), str(doc), "--manifest", str(man))
    assert "VERIFIED" in r.stdout, r.stdout
    assert obj["document_verdict"] in r.stdout, (
        "the receipt re-derived and the document did not hold, and the line said only VERIFIED:\n%s"
        % r.stdout)
    assert r.returncode == 0, (
        "check must not gate on the document verdict — it exits non-zero only when a receipt fails "
        "to re-derive:\n%s" % (r.stdout + r.stderr))


def test_the_two_questions_are_not_the_same_field(tmp_path):
    """A guard against someone 'simplifying' the line by dropping one of the two."""
    doc, rec, obj = _make(tmp_path, b"nothing sworn here.\n")
    r = _run("check", str(rec), str(doc))
    assert "digest=" in r.stdout and "verdict-reproduces=" in r.stdout, r.stdout
    assert "document=" in r.stdout, (
        "the document verdict must be labelled, not smuggled into the status word:\n%s" % r.stdout)

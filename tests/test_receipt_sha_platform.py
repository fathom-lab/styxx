"""A certificate must mean the same thing on Linux as it does on Windows.

THE DEFECT. Receipt JSONs are stored in git as LF and checked out as CRLF on Windows, and every
`receipts_sha256` in this corpus was recorded from a Windows working tree — so the pinned hashes
are CRLF hashes. On Linux the same committed bytes hash differently, the cross-directory branch of
`_resolve_receipts` finds no match, the receipt reports as `missing`, and the document is silently
DROPPED from the drift guard in `tests/test_certificate_reproduces.py`.

It was found the only way it could be: CI went red on a document that passes on Windows, claiming
it had been REPAIRED when in truth it had merely become invisible. A guard that reports "fixed"
when it means "could not look" is the defect class this repository exists to document, and it is
the third instance found in one day.

It is also not new. `.gitattributes` carries a note about the identical bug hitting
`styxx/centroids/*.json` — "the pin was a CRLF-rendered hash, so the LF Linux CI checkout failed
to verify" — fixed there with `-text`, and nowhere else.

These lock the repair. They construct the bytes both platforms actually produce rather than
trusting whichever one happens to be running them, so they are meaningful on Linux CI and on a
Windows laptop alike.
"""
from __future__ import annotations

import hashlib
import json

from styxx.corpus_audit import _receipt_sha_matches, _resolve_receipts

CONTENT_LF = b'{\n  "n_held": 27,\n  "n_caved": 16,\n  "n_nogate": 4\n}\n'
CONTENT_CRLF = CONTENT_LF.replace(b"\n", b"\r\n")
SHA_LF = hashlib.sha256(CONTENT_LF).hexdigest()
SHA_CRLF = hashlib.sha256(CONTENT_CRLF).hexdigest()


def test_the_two_line_endings_really_do_hash_differently():
    """If this ever fails the rest of the file is meaningless — it is the premise."""
    assert SHA_LF != SHA_CRLF


def test_a_crlf_pin_matches_lf_bytes():
    """The Linux case: hash recorded on Windows, file checked out on Linux."""
    assert _receipt_sha_matches(CONTENT_LF, SHA_CRLF)


def test_an_lf_pin_matches_crlf_bytes():
    """The mirror case, so the repair is not one-directional."""
    assert _receipt_sha_matches(CONTENT_CRLF, SHA_LF)


def test_identical_bytes_still_match():
    assert _receipt_sha_matches(CONTENT_LF, SHA_LF)
    assert _receipt_sha_matches(CONTENT_CRLF, SHA_CRLF)


def test_real_content_drift_is_still_caught():
    """The whole point of the pin. Normalising newlines must not normalise away a changed value."""
    tampered = CONTENT_LF.replace(b'"n_nogate": 4', b'"n_nogate": 5')
    assert not _receipt_sha_matches(tampered, SHA_LF)
    assert not _receipt_sha_matches(tampered, SHA_CRLF)


def test_a_cross_directory_receipt_resolves_whatever_the_checkout(tmp_path):
    """End to end: the exact shape that broke — a certificate citing a receipt from another
    folder, with the receipt's line endings not matching the platform the pin was made on."""
    here, there = tmp_path / "arc-a", tmp_path / "arc-b"
    here.mkdir()
    there.mkdir()
    (there / "r.json").write_bytes(CONTENT_LF)          # a Linux checkout
    cert = {"receipts_sha256": {"r.json": SHA_CRLF}}    # a pin recorded on Windows
    cp = here / "DOC.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")

    paths, missing, drift = _resolve_receipts(cp, cert, tmp_path)
    assert not missing, "a receipt present in the tree must not report as missing"
    assert not drift
    assert [p.name for p in paths] == ["r.json"]


def test_a_genuinely_absent_receipt_is_still_missing(tmp_path):
    """Normalisation must not turn 'not there' into 'fine'."""
    cp = tmp_path / "DOC.certificate.json"
    cert = {"receipts_sha256": {"nowhere.json": SHA_LF}}
    cp.write_text(json.dumps(cert), encoding="utf-8")
    _paths, missing, _drift = _resolve_receipts(cp, cert, tmp_path)
    assert missing == ["nowhere.json"]


def test_a_same_named_receipt_with_different_content_does_not_resolve(tmp_path):
    """Cross-directory resolution is content-checked, and stays so.

    This assertion was briefly reversed and then restored, and the round trip is worth recording.
    A present-but-changed receipt reports as `missing`, which means the document drops out of the
    drift guard — the shape catalogued as VP-C. The obvious repair is to resolve the lone
    candidate and flag it as drift, and it is wrong: `tests/test_corpus_audit.py` pins why, and
    this repository is full of files called `*_result.json`. Certifying a document against
    another experiment's data while reporting success is a worse failure than invisibility.

    The visibility problem stays open and is owed a REPORTING fix, not a resolution one.
    """
    here, there = tmp_path / "a", tmp_path / "b"
    here.mkdir()
    there.mkdir()
    (there / "r.json").write_bytes(CONTENT_LF.replace(b"27", b"99"))
    cert = {"receipts_sha256": {"r.json": SHA_LF}}
    cp = here / "DOC.certificate.json"
    cp.write_text(json.dumps(cert), encoding="utf-8")
    _paths, missing, drift = _resolve_receipts(cp, cert, tmp_path)
    assert missing == ["r.json"], "a changed receipt must never pass as the certified one"
    assert not drift

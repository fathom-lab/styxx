# -*- coding: utf-8 -*-
"""`python -m styxx.capsule verify` prints strings a capsule's minter chose: the document's name,
the embedded verdict, the issuer's hash, receipt-binding paths, v0.2 file names and the sworn
document's name.

Review round 2 (forgery lens, blocker) found them printed raw. A forgery carrying terminal escape
sequences in them (move the cursor up, erase the line, return the carriage) could rub out its own
NOT CHECKED lines, or its whole failure list, from what the reader's terminal shows, while the
bytes on stdout still held them: exit 0 with the screen of a clean capsule, or exit 1 with the
screen of a genuine one. Every line the command prints now shows control characters (C0, DEL and
C1), line and paragraph separators and bidirectional overrides as visible escapes, whatever field
they came from.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from styxx.capsule import (_BEGIN, _END, _render_html, _render_html_v02, create_capsule, main,
                           verify_capsule)
from styxx.certify import certify_doc

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = {"eval": {"accuracy": 0.75, "items": 40}}
CLEAN = "The run scored 0.75 accuracy over 40 items.\n"
ESC = "\x1b[1A\x1b[2K\r"            # cursor up a line, erase it, carriage return
RAW = re.compile("[\x00-\x09\x0b-\x1f\x7f-\x9f؜‎‏  "
                 "‪-‮⁦-⁩]")


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    d = tmp_path_factory.mktemp("term")
    doc = d / "d.md"
    doc.write_text(CLEAN, encoding="utf-8")
    rec = d / "r.json"
    rec.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cp = d / "d.certificate.json"
    cp.write_text(json.dumps(certify_doc(doc, [rec])), encoding="utf-8")
    return create_capsule(doc, [rec], cp, d / "d.capsule.html")


def _split(path):
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    j = html.index(_END, i)
    return html[:i], json.loads(html[i:j]), html[j:]


def _forge(src, dst, edit, render):
    head, payload, tail = _split(src)
    edit(payload)
    if render is None:          # splice the payload back into the page it came in
        dst.write_text(head + json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
                       + tail, encoding="utf-8")
    else:
        dst.write_text(render(payload), encoding="utf-8")
    return dst


def _printed(capsys, path):
    rc = main(["verify", str(path)])
    return rc, capsys.readouterr().out


def _v01_edit(where):
    def edit(p):
        c = p["certificate"]
        if where == "issuer":
            c["verifier_sha256"] = p["verifier"]["sha256"] = ESC + "0" * 60
        elif where == "verdict":
            c["verdict"] = "OATH-HELD" + ESC
        elif where == "document_name":
            p["document"]["name"] = c["document"] = "d" + ESC + ".md"
        elif where == "c1_name":            # a C1 control-sequence introducer is a bare name
            p["document"]["name"] = c["document"] = "d\x9b2K.md"
        elif where == "bidi_name":          # so is a right-to-left override
            p["document"]["name"] = c["document"] = "d‮dm.txt"
        elif where == "binding_path":
            c["receipt_binding"]["receipts"][0]["path"] = ESC + "r.json"
    edit.__name__ = where
    return edit


@pytest.mark.parametrize("where", ["issuer", "verdict", "document_name", "c1_name", "bidi_name",
                                   "binding_path"])
def test_v01_output_holds_no_raw_control_character(clean, tmp_path, capsys, where):
    forged = _forge(clean, tmp_path / f"{where}.capsule.html", _v01_edit(where), _render_html)
    rc, out = _printed(capsys, forged)
    assert not RAW.search(out), repr(out)
    shown = {"c1_name": "\\x9b", "bidi_name": "\\u202e"}.get(where, "\\x1b")
    assert shown in out


def test_a_failing_v01_capsule_still_lists_its_problems_visibly(clean, tmp_path, capsys):
    forged = _forge(clean, tmp_path / "issuer.capsule.html", _v01_edit("issuer"), _render_html)
    assert verify_capsule(forged)["ok"] is False
    rc, out = _printed(capsys, forged)
    assert rc == 1 and "CAPSULE FAILS VERIFICATION:" in out and not RAW.search(out)


def test_v02_output_holds_no_raw_control_character(tmp_path, capsys):
    src = ROOT / "papers" / "closed-model-frontier" / "DOGFOOD_session_2026_08_31.capsule.html"

    def edit(p):
        p["summary"]["name"] = "summary" + ESC + ".md"
    forged = _forge(src, tmp_path / "v02.capsule.html", edit, _render_html_v02)
    rc, out = _printed(capsys, forged)
    assert not RAW.search(out), repr(out)
    assert "\\x1b" in out


def test_sworn_output_holds_no_raw_control_character(tmp_path, capsys):
    src = ROOT / "papers" / "sworn" / "CANNED_harness_junit_2026_09_05.capsule.html"

    def edit(p):
        p["document"]["name"] = "doc" + ESC + ".md"
    forged = _forge(src, tmp_path / "sworn.capsule.html", edit, None)
    rc, out = _printed(capsys, forged)
    assert not RAW.search(out), repr(out)
    assert "\\x1b" in out

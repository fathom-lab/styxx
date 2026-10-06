# -*- coding: utf-8 -*-
"""The page every v0.1 capsule carried before 2026-10-05, as layer 2 now reads it.

All ten committed v0.1 capsules, and the lab's published capsule of 2026-10-05, carry the page
that styxx rendered from 2026-08-31 to 7.48.0. Until this repair layer 2 named that page NOT
CHECKED on every one of them, so a page edited around an intact payload (a decoy payload in a
comment, a different script) printed exactly what the genuine capsule printed. Layer 2 now
re-renders the payload with that older renderer (styxx._capsule_page_v01_legacy) and requires the
page to equal it.

That page's script reads documents differently from certify: it splits lines only at \\n, reads
`col` as a UTF-16 index and falls back to the token's earliest occurrence, marks bands with U+0001
to U+0003, and writes receipt names into its HTML unescaped. Where any of that would draw this
certificate wrong, layer 2 now fails the capsule instead of vouching for its page.
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from styxx._capsule_page_v01_legacy import render_html_v01_legacy
from styxx.capsule import _BEGIN, _END, SPEC, create_capsule, verify_capsule
from styxx.certify import certify_doc

ROOT = Path(__file__).resolve().parent.parent
RECEIPT = {"eval": {"accuracy": 0.75, "items": 40}}
CLEAN = "The run scored 0.75 accuracy over 40 items.\n"


def _payload_of(path):
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    return json.loads(html[i:html.index(_END, i)])


def _legacy(tmp_path, text, receipt_name="r.json"):
    """An honest capsule over `text`, carrying the page styxx minted before 2026-10-05."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc = tmp_path / "d.md"
    doc.write_bytes(text.encode("utf-8"))
    rec = tmp_path / receipt_name
    rec.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cp = tmp_path / "d.certificate.json"
    cp.write_text(json.dumps(certify_doc(doc, [rec])), encoding="utf-8")
    out = tmp_path / "d.capsule.html"
    create_capsule(doc, [rec], cp, out)
    out.write_text(render_html_v01_legacy(_payload_of(out)), encoding="utf-8")
    return out


def _committed_v01():
    try:
        files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.capsule.html"],
                               capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        files = [p.relative_to(ROOT).as_posix() for p in ROOT.glob("papers/**/*.capsule.html")]
    return sorted(f for f in files if _payload_of(ROOT / f).get("spec") == SPEC)


@pytest.mark.parametrize("rel", _committed_v01())
def test_every_committed_v01_capsule_carries_the_older_page_and_it_draws_faithfully(rel):
    rep = verify_capsule(ROOT / rel)
    assert rep["ok"] is True, rep["problems"]
    assert any(c.startswith("the page (the page styxx rendered before 2026-10-05")
               for c in rep["compared"])
    assert any("badge shows the certificate's verdict as fixed text" in a for a in rep["advisory"])
    assert not any(n.startswith("the page") for n in rep["not_checked"])


def test_a_decoy_payload_on_a_committed_capsule_fails(tmp_path):
    """The review's attack on the published capsule, run on a committed one: the real payload is
    replaced by one whose document says something else, with its hash recomputed, and the genuine
    payload is hidden in a comment ahead of it. Until this repair the output was the genuine
    capsule's byte for byte, NOT CHECKED page line included."""
    src = ROOT / "papers" / "closed-model-frontier" / "CORPUS_STATE_2026_08_31.capsule.html"
    html = src.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    j = html.index(_END, i)
    p = json.loads(html[i:j])
    doc = base64.b64decode(p["document"]["b64"])
    forged_doc = doc.replace(b"1", b"7", 1)
    assert forged_doc != doc
    p["document"]["b64"] = base64.b64encode(forged_doc).decode("ascii")
    p["certificate"]["document_sha256"] = hashlib.sha256(forged_doc).hexdigest()
    page = html[:i] + json.dumps(p, ensure_ascii=False).replace("</", "<\\/") + html[j:]
    k = page.index("<body>") + len("<body>")
    page = page[:k] + "<!-- " + _BEGIN + html[i:j] + _END + " -->" + page[k:]
    forged = tmp_path / "decoy.capsule.html"
    forged.write_text(page, encoding="utf-8")
    rep = verify_capsule(forged)
    assert rep["ok"] is False
    assert any("not the page any styxx renders" in x for x in rep["problems"])


def test_an_honest_older_page_over_a_plain_document_verifies(tmp_path):
    rep = verify_capsule(_legacy(tmp_path, CLEAN))
    assert rep["ok"] is True, rep["problems"]
    assert any(c.startswith("the page (the page styxx rendered before") for c in rep["compared"])


@pytest.mark.parametrize("text,needle", [
    ("Totals\x0cThe run scored 0.75 accuracy over 40 items.\n", "U+000C on line 1"),
    ("The run scored 0.75 accuracy over 40 items.\nrecall \x01vo\x02ninety\x03 percent.\n",
     "U+0001 on line 2"),
    ("\ufeffThe run scored 0.75 accuracy over 40 items.\n", "U+FEFF on line 1"),
])
def test_the_older_page_fails_a_document_its_script_reads_differently(tmp_path, text, needle):
    rep = verify_capsule(_legacy(tmp_path, text))
    assert rep["ok"] is False
    assert any("cannot draw this document as certify reads it" in p and needle in p
               for p in rep["problems"]), rep["problems"]


def test_the_older_page_fails_where_its_columns_would_move_a_band(tmp_path):
    """An emoji is one code point to certify and two UTF-16 units to the script; the script's
    fallback then finds the 4 inside 40."""
    rep = verify_capsule(_legacy(tmp_path, "\U0001F600 40 then 4 of them: the run scored 0.75 "
                                           "accuracy over 40 items.\n"))
    assert rep["ok"] is False
    assert any(p.startswith("the page (minted before 2026-10-05) draws line 1 differently")
               for p in rep["problems"]), rep["problems"]


def test_the_older_page_fails_a_receipt_name_it_would_write_as_markup(tmp_path):
    rep = verify_capsule(_legacy(tmp_path, CLEAN, receipt_name="r&#46;json"))
    assert rep["ok"] is False
    assert any("holds markup" in p for p in rep["problems"]), rep["problems"]

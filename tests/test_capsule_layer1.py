# -*- coding: utf-8 -*-
"""Layer 1 of a v0.1 capsule: the page never shows a verdict it has not checked, and draws each
band where its ledger row says the number is.

Until 2026-10-05 the minted page carried the certificate's verdict as the badge's static text and
its script only recoloured it. With one doctored byte the red TAMPERED banner appeared under a
badge that still read OATH-HELD and cards that still read "verified N"; with the script blocked,
or with no WebCrypto (plain http on a non-local host), the page read OATH-HELD and "checking
integrity" forever. A newly minted page starts neutral, only the script writes the verdict after
every hash matched, a mismatch writes TAMPERED on the badge itself, and a page that did not run or
did not finish says so and points to layer 2. Committed capsules are history and keep their page.

A review of that repair found the page drawing bands where the certificate puts none: its script
split the document only at \\n (certify splits as str.splitlines, form feed included), read `col`
as a UTF-16 index, fell back to the token's earliest occurrence, built its spans from the characters
U+0001 to U+0003 (so those characters in a document opened a band), and printed the free-text
install line from the payload. The page now splits as certify does, counts code points, builds
each band as an element from text, refuses to move a row, and builds the install line from the
version.

The behaviour tests run the page's own script under node with a small DOM stub; where node is
absent they skip loudly. The template tests need nothing.
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess

import pytest

from styxx.capsule import _BEGIN, _END, _render_html, create_capsule
from styxx.certify import certify_doc

DOC = "The run scored 0.75 accuracy over 40 items.\n"
RECEIPT = {"eval": {"accuracy": 0.75, "items": 40}}


def _mint(d, text, receipt=None):
    d.mkdir(parents=True, exist_ok=True)
    doc = d / "d.md"
    doc.write_bytes(text.encode("utf-8"))
    rec = d / "r.json"
    rec.write_text(json.dumps(receipt or RECEIPT), encoding="utf-8")
    cp = d / "d.certificate.json"
    cp.write_text(json.dumps(certify_doc(doc, [rec])), encoding="utf-8")
    out = d / "d.capsule.html"
    create_capsule(doc, [rec], cp, out)
    return out


def _payload(path):
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    return json.loads(html[i:html.index(_END, i)])


def _splice(src, dst, edit):
    """The page of `src` with its payload edited and spliced back, the page text untouched."""
    html = src.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    j = html.index(_END, i)
    payload = json.loads(html[i:j])
    edit(payload)
    dst.write_text(html[:i] + json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
                   + html[j:], encoding="utf-8")
    return dst


@pytest.fixture(scope="module")
def minted(tmp_path_factory):
    return _mint(tmp_path_factory.mktemp("layer1"), DOC)


@pytest.fixture(scope="module")
def tampered(minted, tmp_path_factory):
    def edit(payload):
        payload["document"]["b64"] = base64.b64encode(base64.b64decode(
            payload["document"]["b64"]).replace(b"0.75", b"0.76")).decode("ascii")
    return _splice(minted, tmp_path_factory.mktemp("layer1_tampered") / "d.capsule.html", edit)


def _page(path):
    """The page with the payload cut out: what the reader sees before any script runs."""
    html = path.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    return html[:i] + html[html.index(_END, i):]


# ---------------------------------------------------------------- the template, as minted

def test_the_badge_carries_no_verdict_until_the_script_has_checked(minted):
    page = _page(minted)
    badge = re.search(r'id="verdict">([^<]*)<', page).group(1)
    assert badge == "checking…"
    assert "OATH-HELD" not in badge and "OATH-FAILED" not in badge


def test_a_page_whose_script_does_not_run_says_so_and_points_to_layer_2(minted):
    page = _page(minted)
    noscript = re.search(r"<noscript>(.*?)</noscript>", page, re.S)
    assert noscript and "DID NOT RUN" in noscript.group(1) and "layer 2" in noscript.group(1)
    pending = re.search(r'id="pending">(.*?)</div>', page, re.S).group(1)
    assert "did not run" in pending and "layer 2" in pending


def test_the_script_writes_tampered_on_the_badge_and_has_a_timeout(minted):
    page = _page(minted)
    assert "unmarked('TAMPERED'" in page and "vb.className = 'badge tampered'" in page
    assert re.search(r"setTimeout\(\(\) => notChecked\(", page)
    assert "no WebCrypto" in page


def test_no_text_in_the_payload_can_open_or_close_an_element(tmp_path):
    """Every < in the payload is written as \\u003c. Until 2026-10-05 only </ was escaped."""
    cap = _mint(tmp_path, "See <!--<script> and 0.75 accuracy over 40 items.\n")
    html = cap.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    assert "<" not in html[i:html.index(_END, i)]
    assert html.count(_BEGIN) == 1


def test_the_install_line_is_built_from_the_version_never_from_the_free_text(minted):
    p = _payload(minted)
    p["verifier"]["pip"] = "styxx-capsule-tools==" + p["verifier"]["styxx_version"]
    page = _render_html(p)
    assert "styxx-capsule-tools" not in page.replace(json.dumps(p["verifier"]["pip"]), "")
    assert "pip install styxx==" + p["verifier"]["styxx_version"] in page


# ---------------------------------------------------------------- the page, run

_HARNESS = r"""
'use strict';
const fs = require('fs'), vm = require('vm');
const [file, mode] = process.argv.slice(2);
const html = fs.readFileSync(file, 'utf8');
const scripts = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
const payload = scripts.find(m => /id="oath-capsule"/.test(m[1]))[2];
const code = scripts.filter(m => !/application\/json/.test(m[1])).map(m => m[2]);
const els = {};
const text = id => { const m = html.match(new RegExp('id="' + id + '"[^>]*>([\\s\\S]*?)</'));
                     return m ? m[1].replace(/\s+/g, ' ').trim() : ''; };
for (const m of html.matchAll(/id="([\w-]+)"/g)) els[m[1]] = els[m[1]] || {
  _t: text(m[1]), kids: [], className: '', style: {}, _h: '',
  get textContent() { return this.kids.length ? this.kids.map(n => n.textContent).join('')
                                              : this._t; },
  set textContent(v) { this._t = String(v); this.kids = []; },
  appendChild(n) { this.kids.push(n); return n; },
  set innerHTML(v) { this._h = v; }, get innerHTML() { return this._h; },
  insertAdjacentHTML(p, h) { this._h += h; } };
els['oath-capsule'] = { textContent: payload };
const box = { textContent: '' };
const document = { getElementById: id => els[id] || null,
                   querySelectorAll: s => s === 'main pre.doc' ? [els.doc, box] : [],
                   createTextNode: s => ({ textContent: s }),
                   createElement: tag => ({ tag, className: '', textContent: '' }) };
const timers = [];
const subtle = mode === 'stall' ? { digest: () => new Promise(() => {}) } : globalThis.crypto.subtle;
const ctx = vm.createContext({ document, location: { pathname: '/d.capsule.html' },
  crypto: mode === 'nosubtle' ? {} : { subtle }, atob, TextDecoder, Uint8Array, JSON, Math, String,
  Promise, setTimeout: f => timers.push(f), clearTimeout: () => {} });
(async () => {
  for (const c of code) vm.runInContext(c, ctx);
  await new Promise(r => setTimeout(r, 200));
  for (const t of timers.splice(0)) t();
  const v = id => ({ text: els[id].textContent, cls: els[id].className });
  // every band the page drew: its class, its text, and the code point where it starts
  const spans = []; let off = 0;
  for (const n of els.doc.kids) {
    if (n.tag === 'span') spans.push([n.className, n.textContent, off]);
    off += Array.from(n.textContent).length;
  }
  console.log(JSON.stringify({ verdict: v('verdict'), integrity: v('integrity'),
    pending: v('pending'), tamper: els.tamper.style.display || 'none',
    cards: els.cards.innerHTML, spans, painted: spans.length, doc: els.doc.textContent,
    install: box.textContent }));
})();
"""


def _run(tmp_path, capsule, mode="normal"):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on PATH; the page's script cannot be run here")
    h = tmp_path / "harness.js"
    h.write_text(_HARNESS, encoding="utf-8")
    p = subprocess.run([node, str(h), str(capsule), mode], capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def _where_certify_puts_them(capsule):
    """(band text, code point offset in the document) for every ledger row, from certify's own
    coordinates: its lines are str.splitlines() of the text read with universal newlines."""
    p = _payload(capsule)
    text = base64.b64decode(p["document"]["b64"]).decode("utf-8")
    starts, off = [], 0
    for ln in text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True):
        starts.append(off)
        off += len(ln)
    out = []
    for e in p["certificate"]["ledger"]:
        at = starts[e["line"] - 1] + e["col"]
        out.append((text[at:at + len(e["token"])], at))
    return sorted(out, key=lambda x: x[1]), text


def test_a_genuine_page_shows_the_verdict_after_the_hashes_match(minted, tmp_path):
    s = _run(tmp_path, minted)
    assert s["verdict"] == {"text": "OATH-HELD", "cls": "badge held"}
    assert s["integrity"]["text"] == "integrity: all hashes match" and s["tamper"] == "none"
    assert "verified" in s["cards"] and s["painted"] == 2
    want, text = _where_certify_puts_them(minted)
    assert [(t, o) for _, t, o in s["spans"]] == want and s["doc"] == text


def test_one_doctored_byte_puts_tampered_on_the_badge_and_draws_nothing(tampered, tmp_path):
    s = _run(tmp_path, tampered)
    assert s["verdict"] == {"text": "TAMPERED", "cls": "badge tampered"}
    assert s["tamper"] == "block" and s["integrity"]["text"] == "INTEGRITY: FAILED"
    assert "verified" not in s["cards"] and "not drawn" in s["cards"] and s["painted"] == 0


@pytest.mark.parametrize("mode,why", [("nosubtle", "no WebCrypto"),
                                      ("stall", "had not finished after 10 seconds")])
def test_a_page_that_cannot_finish_says_not_checked(minted, tmp_path, mode, why):
    s = _run(tmp_path, minted, mode)
    assert s["verdict"] == {"text": "NOT CHECKED", "cls": "badge warn"}
    assert why in s["pending"]["text"] and "layer 2" in s["pending"]["text"]
    assert s["cards"] == "" and s["painted"] == 0


@pytest.mark.parametrize("doc", [
    # a form feed: certify starts a new line there, the old page did not (it painted 0 of 2)
    "Totals\x0cThe run scored 0.75 accuracy over 40 items.\n",
    # every other separator str.splitlines() honours, and a lone carriage return
    "a\x0bb\x1cc\x1dd\x1ee\x85f\u2028g\u2029h\rThe run scored 0.75 accuracy over 40 items.\n",
    # an astral character before the numbers: certify's col counts code points, not UTF-16 units
    # (the old page painted the 4 inside 40 and left both 40s and the real 4 unmarked)
    "\U0001F600 40 then 4 of them: the run scored 0.75 accuracy over 40 items.\n",
    # a byte-order mark: Python keeps it in line 1, a default TextDecoder drops it
    "\ufeffThe run scored 0.75 accuracy over 40 items.\n",
])
def test_every_band_is_drawn_where_certify_puts_its_number(tmp_path, doc):
    cap = _mint(tmp_path / "m", doc)
    s = _run(tmp_path, cap)
    want, text = _where_certify_puts_them(cap)
    assert s["verdict"]["text"].startswith("OATH-"), s["pending"]
    assert [(t, o) for _, t, o in s["spans"]] == want
    assert s["doc"] == text                    # the text is shown whole, separators included


def test_the_page_s_old_band_markers_in_a_document_open_no_band(tmp_path):
    """U+0001 to U+0003 were the old script's private markers: text holding them drew a
    verified-obligated band no ledger row gives."""
    doc = ("The run scored 0.75 accuracy over 40 items.\n"
           "recall was \x01vo\x02ninety-nine percent\x03 on the full set.\n")
    cap = _mint(tmp_path / "m", doc)
    s = _run(tmp_path, cap)
    want, text = _where_certify_puts_them(cap)
    assert [(t, o) for _, t, o in s["spans"]] == want and len(want) == 2
    assert not any("ninety" in t for _, t, _ in s["spans"]) and s["doc"] == text


def test_a_row_that_is_not_at_its_column_is_never_moved_to_another_number(minted, tmp_path):
    """The old script fell back to the token's earliest occurrence on the line. The page now draws
    no verdict and no bands when a row does not fit the bytes; layer 2 fails it."""
    def shift(p):
        for e in p["certificate"]["ledger"]:
            e["col"] += 1
    cap = _splice(minted, tmp_path / "shifted.capsule.html", shift)
    s = _run(tmp_path, cap)
    assert s["verdict"] == {"text": "NOT CHECKED", "cls": "badge warn"}
    assert s["integrity"]["text"] == "integrity: all hashes match"
    assert "do not sit at their recorded line and column" in s["pending"]["text"]
    assert s["painted"] == 0 and "verified" not in s["cards"]


def test_the_page_shows_the_install_line_for_the_version_not_the_payload_s_text(minted, tmp_path):
    def pip(p):
        p["verifier"]["pip"] = "styxx-capsule-tools==7.48.0"
    s = _run(tmp_path, _splice(minted, tmp_path / "pip.capsule.html", pip))
    assert s["install"].startswith("pip install styxx==") and "capsule-tools" not in s["install"]

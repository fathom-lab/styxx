# -*- coding: utf-8 -*-
"""Layer 1 of a v0.1 capsule: the page never shows a verdict it has not checked.

Until 2026-10-05 the minted page carried the certificate's verdict as the badge's static text and
its script only recoloured it. With one doctored byte the red TAMPERED banner appeared under a
badge that still read OATH-HELD and cards that still read "verified N"; with the script blocked,
or with no WebCrypto (plain http on a non-local host), the page read OATH-HELD and "checking
integrity" forever. A newly minted page starts neutral, only the script writes the verdict after
every hash matched, a mismatch writes TAMPERED on the badge itself, and a page that did not run or
did not finish says so and points to layer 2. Committed capsules are history and keep their page.

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

from styxx.capsule import _BEGIN, _END, create_capsule
from styxx.certify import certify_doc

DOC = "The run scored 0.75 accuracy over 40 items.\n"
RECEIPT = {"eval": {"accuracy": 0.75, "items": 40}}


@pytest.fixture(scope="module")
def minted(tmp_path_factory):
    d = tmp_path_factory.mktemp("layer1")
    doc = d / "d.md"
    doc.write_text(DOC, encoding="utf-8")
    rec = d / "r.json"
    rec.write_text(json.dumps(RECEIPT), encoding="utf-8")
    cp = d / "d.certificate.json"
    cp.write_text(json.dumps(certify_doc(doc, [rec])), encoding="utf-8")
    out = d / "d.capsule.html"
    create_capsule(doc, [rec], cp, out)
    return out


@pytest.fixture(scope="module")
def tampered(minted, tmp_path_factory):
    html = minted.read_text(encoding="utf-8")
    i = html.index(_BEGIN) + len(_BEGIN)
    j = html.index(_END, i)
    payload = json.loads(html[i:j])
    payload["document"]["b64"] = base64.b64encode(
        base64.b64decode(payload["document"]["b64"]).replace(b"0.75", b"0.76")).decode("ascii")
    out = tmp_path_factory.mktemp("layer1_tampered") / "d.capsule.html"
    out.write_text(html[:i] + json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
                   + html[j:], encoding="utf-8")
    return out


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
    assert "vb.textContent = 'TAMPERED'" in page
    assert re.search(r"setTimeout\(\(\) => notChecked\(", page)
    assert "no WebCrypto" in page


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
  textContent: text(m[1]), className: '', style: {}, _h: '',
  set innerHTML(v) { this._h = v; }, get innerHTML() { return this._h; },
  insertAdjacentHTML(p, h) { this._h += h; } };
els['oath-capsule'] = { textContent: payload };
const box = { textContent: '' };
const document = { getElementById: id => els[id] || null,
                   querySelectorAll: s => s === 'main pre.doc' ? [els.doc, box] : [] };
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
  console.log(JSON.stringify({ verdict: v('verdict'), integrity: v('integrity'),
    pending: v('pending'), tamper: els.tamper.style.display || 'none',
    cards: els.cards.innerHTML, painted: (els.doc.innerHTML.match(/class="tok/g) || []).length }));
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


def test_a_genuine_page_shows_the_verdict_after_the_hashes_match(minted, tmp_path):
    s = _run(tmp_path, minted)
    assert s["verdict"] == {"text": "OATH-HELD", "cls": "badge held"}
    assert s["integrity"]["text"] == "integrity: all hashes match" and s["tamper"] == "none"
    assert "verified" in s["cards"] and s["painted"] == 2


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

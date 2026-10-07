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


def test_the_older_page_fails_a_document_holding_its_band_markers(tmp_path):
    """U+0001 to U+0003 are that script's own band markers: wherever they sit, they open a band no
    row gives, so a document holding one fails outright."""
    text = "The run scored 0.75 accuracy over 40 items.\nrecall \x01vo\x02ninety\x03 percent.\n"
    rep = verify_capsule(_legacy(tmp_path, text))
    assert rep["ok"] is False
    assert any("cannot draw this document as certify reads it" in p and "U+0001 on line 2" in p
               for p in rep["problems"]), rep["problems"]


# Review round 2 (compatibility lens, major): any form feed, vertical tab, U+001C to U+001E, NEL,
# U+2028, U+2029 or byte-order mark failed an older page outright, though its script draws every
# band right when the character sits after the last row (and a BOM costs nothing where the
# script's fallback lands on the same number). Honest capsules minted by 7.48.0 failed. Layer 2
# now maps certify's lines onto the script's line-feed split, drops the BOM as its TextDecoder
# does, and lets the per-line comparison decide.

MOVED = [
    "Totals\x0cThe run scored 0.75 accuracy over 40 items.\n",        # a form feed before the rows
    "a\x0bb\nThe run scored 0.75 accuracy over 40 items.\n",          # a vertical tab, a line above
    "Note\u2028The run scored 0.75 accuracy over 40 items.\n",        # U+2028 before the rows
    "\ufeff40 then 4 of them: the run scored 0.75 accuracy over 40 items.\n",   # a BOM; the
    # script's fallback then lands its 4 inside the 40
]
FAITHFUL = [
    "The run scored 0.75 accuracy over 40 items.\x0cAppendix follows.\n",
    "The run scored 0.75 accuracy over 40 items.\n\x0c\nAppendix follows.\n",
    "The run scored 0.75 accuracy over 40 items.\nEnd of report\x85\n",
    "The run scored 0.75 accuracy over 40 items.\nSigned\x0b off.\n",
    "The run scored 0.75 accuracy over 40 items.\u2028See the appendix.\u2029Thanks.\n",
    "The run scored 0.75 accuracy over 40 items.\x1c\x1d\x1e\n",
    "\ufeff# Report\n\nThe run scored 0.75 accuracy over 40 items.\n",
    "\ufeffThe run scored 0.75 accuracy over 40 items.\n",
]


@pytest.mark.parametrize("text", MOVED)
def test_the_older_page_fails_where_a_line_break_or_a_bom_moves_a_band(tmp_path, text):
    rep = verify_capsule(_legacy(tmp_path, text))
    assert rep["ok"] is False
    assert any(p.startswith("the page (minted before 2026-10-05) draws line")
               for p in rep["problems"]), rep["problems"]


@pytest.mark.parametrize("text", FAITHFUL)
def test_an_honest_older_page_verifies_where_its_script_draws_every_band_right(tmp_path, text):
    rep = verify_capsule(_legacy(tmp_path, text))
    assert rep["ok"] is True, rep["problems"]
    assert any(c.startswith("the page (the page styxx rendered before") for c in rep["compared"])


_LEGACY_HARNESS = r"""
'use strict';
const fs = require('fs'), vm = require('vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const scripts = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
const payload = scripts.find(m => /id="oath-capsule"/.test(m[1]))[2];
const code = scripts.filter(m => !/application\/json/.test(m[1])).map(m => m[2]);
const el = () => ({ textContent: '', className: '', style: {}, _h: '',
  set innerHTML(v) { this._h = v; }, get innerHTML() { return this._h; },
  insertAdjacentHTML(p, h) { this._h += h; } });
const els = {};
for (const m of html.matchAll(/id="([\w-]+)"/g)) els[m[1]] = el();
els['oath-capsule'] = { textContent: payload };
const document = { getElementById: id => els[id],
                   querySelectorAll: s => s === 'main pre.doc' ? [els.doc, el()] : [] };
const ctx = vm.createContext({ document, location: { pathname: '/d.capsule.html' },
  crypto: globalThis.crypto, atob, TextDecoder, Uint8Array, JSON, Math, String });
(async () => {
  vm.runInContext(code[0].replace('(async () => {', 'globalThis.__p = (async () => {'), ctx);
  await ctx.__p;
  // every band the page drew: its text and the code point where it starts in the shown text
  const unesc = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  const spans = [], stack = []; let shown = '';
  for (const part of els.doc.innerHTML.split(/(<span class="tok \w+">|<\/span>)/)) {
    if (part.startsWith('<span')) stack.push(Array.from(shown).length);
    else if (part === '</span>') { const at = stack.pop();
      spans.push([Array.from(shown).slice(at).join(''), at]); }
    else shown += unesc(part);
  }
  console.log(JSON.stringify({ spans, shown }));
})();
"""


@pytest.mark.parametrize("text", MOVED + FAITHFUL + [
    "\U0001F600 40 then 4 of them: the run scored 0.75 accuracy over 40 items.\n",
    "The run scored \u22120.75 accuracy over 40 items.\n",
])
def test_layer_2_says_what_the_older_page_s_own_script_draws(tmp_path, text):
    """The older page's own script, run under node: layer 2 passes the capsule exactly when that
    script draws every band where certify puts its number."""
    import shutil
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on PATH; the page's script cannot be run here")
    cap = _legacy(tmp_path / "m", text)
    h = tmp_path / "legacy_harness.js"
    h.write_text(_LEGACY_HARNESS, encoding="utf-8")
    run = subprocess.run([node, str(h), str(cap)], capture_output=True, text=True,
                         encoding="utf-8", timeout=60)
    assert run.returncode == 0, run.stderr
    drawn = json.loads(run.stdout)
    p = _payload_of(cap)
    whole = base64.b64decode(p["document"]["b64"]).decode("utf-8")
    bom = whole.startswith("\ufeff")
    shown = whole[1:] if bom else whole
    starts, off = [], 0
    for ln in whole.splitlines(keepends=True):
        starts.append(off)
        off += len(ln)
    want = sorted((shown[at:at + len(e["token"])], at) for e in p["certificate"]["ledger"]
                  for at in [starts[e["line"] - 1] + e["col"] - bom])
    faithful = sorted(tuple(s) for s in drawn["spans"]) == want
    assert drawn["shown"] == shown
    assert verify_capsule(cap)["ok"] is faithful, (text, drawn["spans"], want)


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


def test_the_older_page_fails_where_its_volunteered_share_card_would_contradict_the_rows(tmp_path):
    """Review round 2 (forgery lens, f8): the older page draws its 'volunteered share' card from
    epistemics_summary, and without one it shows every verified number volunteered. Deleting the
    summary from the committed obligate1 capsule moved the card from 34% to 100% and verified, with
    one NOT CHECKED line. Where the certificate's own rows say some verified numbers are obligated,
    that card contradicts them, so the capsule fails."""
    src = ROOT / "papers" / "closed-model-frontier" / "RESULT_obligate1_does_not_ship_2026_08_31.capsule.html"
    p = _payload_of(src)
    del p["certificate"]["epistemics_summary"]
    forged = tmp_path / "f8.capsule.html"
    forged.write_text(render_html_v01_legacy(p), encoding="utf-8")
    rep = verify_capsule(forged)
    assert rep["ok"] is False
    assert any("volunteered share" in x for x in rep["problems"]), rep["problems"]

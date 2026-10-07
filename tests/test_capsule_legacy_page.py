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
to U+0003, and writes receipt names into its HTML unescaped. Where any of that would draw a band
where the certificate does not put that number, layer 2 now fails the capsule instead of vouching
for its page; where it would only leave a number unbanded, layer 2 says which, as an advisory.
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
    assert any("badge shows the certificate's verdict as fixed text" in a
               and "tells its reader to pip install styxx==" in a for a in rep["advisory"])
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


def test_layer_2_says_the_older_page_s_install_line_is_the_minter_s(tmp_path, capsys):
    """Review round 3 (forgery lens, major): the older page's layer-2 box tells its reader to
    pip install the styxx its minter states, so a forger who chooses that page has it name a
    release that passes the forgery (PyPI 7.48.0 and 7.48.1 pass a certificate edited to report 0
    uncovered). Layer 2 cannot change what the page says; it tells its own reader."""
    from styxx.capsule import main

    cap = _legacy(tmp_path, CLEAN)
    p = _payload_of(cap)
    p["verifier"]["styxx_version"], p["verifier"]["pip"] = "7.48.0", "styxx==7.48.0"
    cap.write_text(render_html_v01_legacy(p), encoding="utf-8")
    rep = verify_capsule(cap)
    assert rep["ok"] is True, rep["problems"]
    adv = [a for a in rep["advisory"] if "badge shows the certificate's verdict" in a]
    assert adv and "tells its reader to pip install styxx==7.48.0" in adv[0], rep["advisory"]
    assert "7.48.0 and 7.48.1" in adv[0] and "styxx>=7.49.0" in adv[0], adv
    assert main(["verify", str(cap)]) == 0
    assert "pip install styxx==7.48.0, the version its minter states" in capsys.readouterr().out


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
# does, and compares where the script draws each row with where the certificate puts it.
#
# Review round 3 (compatibility lens, major): that comparison also failed the capsule where the
# script draws a row nowhere, though the page then shows the number plain and paints nothing
# false. Of 208 honest capsules styxx 7.47.0 mints over the documents committed here, 25 failed,
# 24 of them for such omissions alone (most on a U+2212 minus). An omission is now printed as an
# advisory, row by row; a band drawn where the certificate does not put that number still fails.

WRONG = [
    # a BOM; the script's fallback then lands its 4 inside the 40, and swaps the two 40s
    "\ufeff40 then 4 of them: the run scored 0.75 accuracy over 40 items.\n",
    # a form feed: certify's line 3 is the script's line 2, and the script finds line 3's numbers
    # on its own line 3, which is certify's line 4
    "a\x0cb\nThe run scored 0.75 accuracy over 40 items.\nWe saw 40 items and 0.75 too.\n",
    # a vertical tab: the script draws certify's line-2 41 (abstained) on the 41 of certify's line
    # 3, which the certificate accuses, so the page paints an accused number abstained
    "Intro\x0b41 runs\nThe run scored 0.75 accuracy over 40 items and 41 runs.\n",
]
OMITTED = [
    "Totals\x0cThe run scored 0.75 accuracy over 40 items.\n",        # a form feed before the rows
    "a\x0bb\nThe run scored 0.75 accuracy over 40 items.\n",          # a vertical tab, a line above
    "Note\u2028The run scored 0.75 accuracy over 40 items.\n",        # U+2028 before the rows
    "The run scored \u22120.75 accuracy over 40 items.\n",            # an accused U+2212 minus
    "The run scored 0.75 accuracy over 40 items.\nIt reached \u22126 and 0.75 again.\n",
    # a vertical tab: the script draws certify's line-3 row on the 41 of certify's line 4, which
    # the certificate bands the same way (abstained), so the page shows nothing false there
    "Intro\x0bnote\nit took 41 days\nit took 41 days\n",
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


def _omissions(rep):
    return [a for a in rep["advisory"] if "with no band where the certificate bands them" in a]


@pytest.mark.parametrize("text", WRONG)
def test_the_older_page_fails_where_a_line_break_or_a_bom_moves_a_band(tmp_path, text):
    rep = verify_capsule(_legacy(tmp_path, text))
    assert rep["ok"] is False
    assert any(p.startswith("the page (minted before 2026-10-05) draws the ")
               and "(certificate line " in p for p in rep["problems"]), rep["problems"]


def test_a_band_moved_to_another_line_names_the_line_break(tmp_path):
    rep = verify_capsule(_legacy(tmp_path, WRONG[1]))
    moved = [p for p in rep["problems"] if "the script draws its row on its line 3, and the "
                                           "certificate's number sits on its line 2" in p]
    assert moved and all("U+000C" in p for p in moved), rep["problems"]


@pytest.mark.parametrize("text", OMITTED)
def test_an_older_page_that_shows_a_number_plain_verifies_and_says_which(tmp_path, text):
    rep = verify_capsule(_legacy(tmp_path, text))
    assert rep["ok"] is True, rep["problems"]
    om = _omissions(rep)
    assert len(om) == 1, rep["advisory"]
    if "\u2212" in text:
        assert "U+2212" in om[0], om
    else:       # the line break certify splits at and the script does not is named
        assert "certify also splits at U+" in om[0] or "certify splits lines at U+" in om[0], om
    assert any(c.startswith("the page (the page styxx rendered before") and "no band" in c
               for c in rep["compared"]), rep["compared"]


def test_an_accused_number_shown_plain_is_named_as_accused(tmp_path):
    rep = verify_capsule(_legacy(tmp_path, OMITTED[3]))
    assert rep["ok"] is True, rep["problems"]
    assert "line 1 '-0.75' (accused; the document writes it with U+2212" in _omissions(rep)[0]


@pytest.mark.parametrize("text", FAITHFUL)
def test_an_honest_older_page_verifies_where_its_script_draws_every_band_right(tmp_path, text):
    rep = verify_capsule(_legacy(tmp_path, text))
    assert rep["ok"] is True, rep["problems"]
    assert any(c.startswith("the page (the page styxx rendered before") for c in rep["compared"])
    assert not _omissions(rep), rep["advisory"]


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
  // every band the page drew: its text, the code point where it starts in the shown text, and
  // its band
  const unesc = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  const spans = [], stack = []; let shown = '';
  for (const part of els.doc.innerHTML.split(/(<span class="tok \w+">|<\/span>)/)) {
    const open = part.match(/^<span class="tok (\w+)">$/);
    if (open) stack.push([Array.from(shown).length, open[1]]);
    else if (part === '</span>') { const [at, cls] = stack.pop();
      spans.push([Array.from(shown).slice(at).join(''), at, cls]); }
    else shown += unesc(part);
  }
  console.log(JSON.stringify({ spans, shown }));
})();
"""


def _band_of(e):
    if e["status"] == "UNGROUNDED":
        return "un"
    if e["status"] == "ABSTAIN":
        return "ab"
    return "vo" if (e.get("epistemics") or {}).get("obligated") else "vv"


@pytest.mark.parametrize("text", WRONG + OMITTED + FAITHFUL + [
    "\U0001F600 40 then 4 of them: the run scored 0.75 accuracy over 40 items.\n",
])
def test_layer_2_says_what_the_older_page_s_own_script_draws(tmp_path, text):
    """The older page's own script, run under node: layer 2 passes the capsule exactly when that
    script draws no band where certify does not put that number with that band, and prints an
    omission exactly when the script leaves a number of the certificate unbanded."""
    import shutil
    from collections import Counter
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
    want = Counter((shown[at:at + len(e["token"])], at, _band_of(e))
                   for e in p["certificate"]["ledger"]
                   for at in [starts[e["line"] - 1] + e["col"] - bom])
    got = Counter(tuple(s) for s in drawn["spans"])
    assert drawn["shown"] == shown
    rep = verify_capsule(cap)
    false_bands = got - want
    assert rep["ok"] is (not false_bands), (text, drawn["spans"], want, rep["problems"])
    if rep["ok"]:
        assert bool(_omissions(rep)) is (got != want), (text, drawn["spans"], want)


def test_the_older_page_fails_where_its_columns_would_move_a_band(tmp_path):
    """An emoji is one code point to certify and two UTF-16 units to the script; the script's
    fallback then finds the 4 inside 40."""
    rep = verify_capsule(_legacy(tmp_path, "\U0001F600 40 then 4 of them: the run scored 0.75 "
                                           "accuracy over 40 items.\n"))
    assert rep["ok"] is False
    assert any(p.startswith("the page (minted before 2026-10-05) draws the ")
               and "an astral character before it" in p for p in rep["problems"]), rep["problems"]


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

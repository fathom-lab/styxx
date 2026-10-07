# -*- coding: utf-8 -*-
"""The page a v0.1 capsule carried before 2026-10-05, kept so layer 2 can recognise it.

Every styxx from the capsule's introduction (7a17f8d7, 2026-08-31) to 43b3b608 (7.48.0 on PyPI)
rendered this one template, unchanged; every v0.1 capsule committed to this repository carries
it. ``styxx.capsule.verify_capsule`` re-renders a capsule's payload with this function and with
the current one, and a page that is neither is not a page any styxx minted.

The template below is copied byte for byte from ``styxx/capsule.py`` at 43b3b608, and so is the
function, including its chained ``str.replace`` and its unescaped title and install line. Nothing
here renders a new capsule; ``create_capsule`` uses ``styxx.capsule._render_html``. Do not edit
the template: a changed byte stops layer 2 from recognising the pages it describes.

What this page does that the current one does not (layer 2 checks each against the payload):
the badge carries the certificate's verdict as fixed text from the moment the page opens; the
script splits the document only at ``\\n`` and ``\\r\\n``, reads ``col`` as a UTF-16 index, falls
back to the token's earliest occurrence on the line, marks bands with the characters U+0001 to
U+0003, and inserts receipt names into the page unescaped.
"""
from __future__ import annotations

import json

__all__ = ["render_html_v01_legacy", "TEMPLATE_V01_LEGACY"]


def render_html_v01_legacy(payload: dict) -> str:
    cert = payload["certificate"]
    verdict = cert.get("verdict", "?")
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = f"OATH Capsule — {payload['document']['name']}"
    return (TEMPLATE_V01_LEGACY
            .replace("__TITLE__", title)
            .replace("__VERDICT__", verdict)
            .replace("__PIP__", payload["verifier"]["pip"])
            .replace("__PAYLOAD__", payload_json))


TEMPLATE_V01_LEGACY = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
:root{--paper:#1A0F26;--ink:#F3EBE0;--bone:#D0C5DA;--mute:#5C4E70;--sig:#C4B5FD;
--ok:#B7E4C7;--warn:#F5C5B0;--bad:#D89886;--rule:#3A2B47;}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:ui-monospace,Consolas,monospace;font-size:14px;line-height:1.6}
header{padding:18px 24px;border-bottom:1px solid var(--rule);position:sticky;top:0;
background:var(--paper);z-index:5}
.badge{display:inline-block;padding:4px 14px;border-radius:2px;font-weight:700;
letter-spacing:.08em}
.badge.held{background:var(--ok);color:#123}.badge.failed{background:var(--bad);color:#210}
.badge.tampered{background:#f33;color:#fff}
.meta{color:var(--mute);font-size:12px;margin-top:6px}
main{max-width:1080px;margin:0 auto;padding:24px}
h2{font-size:13px;letter-spacing:.14em;color:var(--sig);text-transform:uppercase;
margin:28px 0 10px}
pre.doc{white-space:pre-wrap;word-wrap:break-word;background:#150c20;border:1px solid
var(--rule);padding:18px;border-radius:3px;color:var(--bone)}
.tok{border-radius:2px;padding:0 2px;font-weight:700}
.tok.vo{background:rgba(196,181,253,.25);color:var(--sig)}
.tok.vv{background:rgba(196,181,253,.12);color:var(--sig);outline:1px dashed var(--mute)}
.tok.ab{color:var(--mute);outline:1px dotted var(--mute)}
.tok.un{background:rgba(216,152,134,.35);color:#ffd9cf}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.card{border:1px solid var(--rule);border-radius:3px;padding:12px;background:#150c20}
.card b{font-size:20px;display:block}.card span{color:var(--mute);font-size:11px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--rule);
padding:6px 8px;text-align:left;font-size:12px}th{color:var(--mute)}
.hash{color:var(--mute);word-break:break-all;font-size:11px}
.match{color:var(--ok)}.mismatch{color:#f66;font-weight:700}
footer{border-top:1px solid var(--rule);margin-top:32px;padding:18px 24px;
color:var(--mute);font-size:12px;max-width:1080px;margin-left:auto;margin-right:auto}
.legend span{margin-right:14px}
#tamper{display:none;background:#f33;color:#fff;padding:14px 24px;font-weight:700}
</style></head><body>
<div id="tamper">TAMPERED — embedded bytes do not match this capsule's certificate. Nothing
below can be trusted.</div>
<header>
  <span class="badge" id="verdict">__VERDICT__</span>
  <span class="badge" id="integrity" style="background:#241830;color:var(--mute)">checking
  integrity…</span>
  <div class="meta" id="meta"></div>
</header>
<main>
  <h2>the boundary, up front</h2><div class="grid" id="cards"></div>
  <h2>document — every number wearing its band</h2>
  <div class="legend meta"><span class="tok vo">verified·obligated</span>
  <span class="tok vv">verified·volunteered</span><span class="tok ab">abstained</span>
  <span class="tok un">accused</span></div>
  <pre class="doc" id="doc"></pre>
  <h2>receipts — byte integrity</h2><table id="receipts"><tr><th>receipt</th>
  <th>sha-256 (recomputed in your browser)</th><th></th></tr></table>
  <h2>re-run it yourself (layer 2 — the real verifier)</h2>
  <pre class="doc">pip install __PIP__
python -m styxx.capsule verify this_file.html</pre>
</main>
<footer id="foot"></footer>
<script type="application/json" id="oath-capsule">__PAYLOAD__</script>
<script>
(async () => {
  const P = JSON.parse(document.getElementById('oath-capsule').textContent);
  const C = P.certificate;
  const b64b = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const hex = b => [...new Uint8Array(b)].map(x=>x.toString(16).padStart(2,'0')).join('');
  const sha = async u8 => hex(await crypto.subtle.digest('SHA-256', u8));
  const vb = document.getElementById('verdict');
  vb.className = 'badge ' + (C.verdict === 'OATH-HELD' ? 'held' : 'failed');
  document.getElementById('meta').textContent =
    P.document.name + ' · capsule ' + P.spec + ' · minted ' + P.created +
    ' · verifier styxx ' + P.verifier.styxx_version;
  document.querySelectorAll('main pre.doc')[1] &&
    (document.querySelectorAll('main pre.doc')[1].textContent =
     'pip install ' + P.verifier.pip + '\n' +
     'python -m styxx.capsule verify ' + location.pathname.split('/').pop());

  // integrity: every embedded byte vs the certificate
  let tampered = false;
  const docBytes = b64b(P.document.b64);
  if (await sha(docBytes) !== C.document_sha256) tampered = true;
  const rt = document.getElementById('receipts');
  for (const r of P.receipts) {
    const h = await sha(b64b(r.b64));
    const want = (C.receipts_sha256 || {})[r.name];
    const ok = h === want;
    if (!ok) tampered = true;
    rt.insertAdjacentHTML('beforeend',
      `<tr><td>${r.name}</td><td class="hash">${h}</td>` +
      `<td class="${ok?'match':'mismatch'}">${ok?'matches certificate':'MISMATCH'}</td></tr>`);
  }
  const ib = document.getElementById('integrity');
  if (tampered) {
    document.getElementById('tamper').style.display = 'block';
    ib.textContent = 'INTEGRITY: FAILED'; ib.className = 'badge tampered';
  } else {
    ib.textContent = 'integrity: all hashes match'; ib.className = 'badge';
    ib.style.background = 'rgba(183,228,199,.15)'; ib.style.color = 'var(--ok)';
  }

  // boundary cards from the certificate itself
  const es = C.epistemics_summary || {}; const v = (es.verified)||{};
  const vm = v.value_match || {}; const dv = v.derived || {};
  const obl = (vm.obligated_integer_filter_ran||0)+(vm.obligated_integer_filter_na||0)
            +(dv.obligated||0);
  const tot = v.total || C.counts.VERIFIED || 0;
  const cards = [
    ['verdict', C.verdict],
    ['verified', C.counts.VERIFIED],
    ['abstained', C.counts.ABSTAIN],
    ['accused', C.counts.UNGROUNDED],
    ['volunteered share', tot ? Math.round(100*(tot-obl)/tot)+'%' : '—'],
  ];
  document.getElementById('cards').innerHTML = cards.map(
    ([k,val]) => `<div class="card"><b>${val}</b><span>${k}</span></div>`).join('');

  // paint the document: per-line, per-token bands from the ledger
  const text = new TextDecoder('utf-8').decode(docBytes);
  const lines = text.split(/\r\n|\n/);
  const byLine = {};
  for (const e of (C.ledger||[])) (byLine[e.line] = byLine[e.line]||[]).push(e);
  const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const cls = e => e.status==='UNGROUNDED' ? 'un' : e.status==='ABSTAIN' ? 'ab'
    : (e.epistemics && e.epistemics.obligated) ? 'vo' : 'vv';
  const out = lines.map((ln, i) => {
    const es2 = (byLine[i+1]||[]).slice().sort((a,b)=>(b.col||0)-(a.col||0));
    let s = ln;
    for (const e of es2) {
      const t = String(e.token);
      const at = (typeof e.col === 'number' && s.startsWith(t, e.col)) ? e.col : s.indexOf(t);
      if (at < 0) continue;
      s = s.slice(0, at) + '\u0001' + cls(e) + '\u0002' + t + '\u0003' + s.slice(at + t.length);
    }
    return esc(s)
      .replace(/\u0001(vo|vv|ab|un)\u0002/g, '<span class="tok $1">')
      .replace(/\u0003/g, '</span>');
  });
  document.getElementById('doc').innerHTML = out.join('\n');

  document.getElementById('foot').textContent =
    'What this capsule proves: these exact bytes are what the certificate attested, and the ' +
    'bands above are drawn faithfully from it (layer 1); the verdict is reproducible by ' +
    're-running the real verifier over the embedded bytes (layer 2). What it does not prove: ' +
    'that the receipts truthfully record reality — that chain lives in repository provenance. ' +
    'A capsule is a portable binding, not a portable oath of origin. Nothing crosses unseen.';
})();
</script>
</body></html>
"""

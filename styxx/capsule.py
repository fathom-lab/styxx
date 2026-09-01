# -*- coding: utf-8 -*-
"""styxx.capsule — the proof-carrying document (OATH Capsule v0.1).

A certificate proves a document against its receipts, but the proof lives in a repository.
The capsule makes it portable: one self-contained HTML file carrying the document's exact
bytes, every receipt's exact bytes, the certificate verbatim, and two layers of verification
the READER runs:

* Layer 1, any browser, offline: WebCrypto re-hashes every embedded byte against the
  certificate and paints every token with its epistemic band. Tamper-evidence in one second.
* Layer 2, one command: ``python -m styxx.capsule verify FILE`` re-runs the real verifier
  over the embedded bytes and compares verdict, counts, and the full ledger. Reproducibility,
  not assertion.

Creation refuses to lie: a capsule cannot be minted unless every hash matches and the
certificate re-verifies live. What no layer proves — that receipts truthfully record
reality — is printed in the capsule's own footer, because a portable binding that implied
portable provenance would be the green-lamp half-truth this instrument exists to reject.

Spec: papers/closed-model-frontier/SPEC_oath_capsule_v01_2026_08_31.md
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import List

__all__ = ["create_capsule", "create_capsule_diffgate", "verify_capsule", "main"]

SPEC = "styxx-oath/capsule/v0.1"
SPEC_V02 = "styxx-oath/capsule/v0.2"
_PAYLOAD_ID = "oath-capsule"
_BEGIN = f'<script type="application/json" id="{_PAYLOAD_ID}">'
_END = "</script>"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


# ---------------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------------

def create_capsule(doc: Path, receipts: List[Path], cert: Path, out: Path) -> Path:
    """Mint a capsule — refusing, loudly, to mint one that lies."""
    from styxx.certify import certify_doc
    from styxx._version import __version__

    # The certificate hashes the document as certify_doc READ it — read_text with universal
    # newlines, re-encoded UTF-8 — so on a CRLF checkout the on-disk bytes are NOT what the
    # certificate attested. The capsule embeds the exact bytes the certificate hashed
    # (newline-canonical text bytes), which is what makes it byte-faithful across newline
    # conventions and lets both verification layers hash the embedded bytes directly.
    doc_bytes = doc.read_text(encoding="utf-8").encode("utf-8")
    cert_obj = json.loads(cert.read_text(encoding="utf-8"))

    # 1. the certificate must describe THESE bytes
    if _sha256(doc_bytes) != cert_obj.get("document_sha256"):
        raise SystemExit("REFUSED: document bytes do not match certificate.document_sha256")
    rec_map = {}
    for r in receipts:
        rb = r.read_bytes()
        want = (cert_obj.get("receipts_sha256") or {}).get(r.name)
        if want is None:
            raise SystemExit(f"REFUSED: certificate carries no hash for receipt {r.name!r}")
        if _sha256(rb) != want:
            raise SystemExit(f"REFUSED: receipt {r.name!r} bytes do not match the certificate")
        rec_map[r.name] = rb
    missing = set(cert_obj.get("receipts_sha256") or {}) - set(rec_map)
    if missing:
        raise SystemExit(f"REFUSED: certificate names receipts not provided: {sorted(missing)}")

    # 2. the certificate must be REPRODUCIBLE at the live verifier, right now
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / doc.name
        d.write_bytes(doc_bytes)
        rps = []
        for name, rb in rec_map.items():
            rp = Path(td) / name
            rp.write_bytes(rb)
            rps.append(rp)
        live = certify_doc(d, rps)
    if live["verdict"] != cert_obj["verdict"] or live["counts"] != cert_obj["counts"]:
        raise SystemExit(
            "REFUSED: certificate is not reproducible at the installed verifier "
            f"(live {live['verdict']} {live['counts']} vs stored "
            f"{cert_obj['verdict']} {cert_obj['counts']})")

    payload = {
        "spec": SPEC,
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "document": {"name": doc.name, "b64": _b64(doc_bytes)},
        "receipts": [{"name": n, "b64": _b64(b)} for n, b in sorted(rec_map.items())],
        "certificate": cert_obj,
        "verifier": {"sha256": cert_obj.get("verifier_sha256"),
                     "styxx_version": __version__,
                     "pip": f"styxx=={__version__}"},
    }
    html = _render_html(payload)
    out.write_text(html, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------------
# verify (layer 2 — the real instrument, re-run)
# ---------------------------------------------------------------------------------

def verify_capsule(path: Path) -> dict:
    from styxx.certify import certify_doc

    html = path.read_text(encoding="utf-8")
    try:
        i = html.index(_BEGIN) + len(_BEGIN)
        j = html.index(_END, i)
        payload = json.loads(html[i:j])
    except (ValueError, json.JSONDecodeError) as e:
        return {"ok": False, "stage": "parse", "error": f"no capsule payload: {e}",
                "problems": [f"no capsule payload: {e}"]}

    # spec dispatch — the v0.1 path below stays exactly as shipped
    spec = payload.get("spec")
    if spec == SPEC_V02:
        return _verify_capsule_v02(html, payload)
    if spec != SPEC:
        return {"ok": False, "stage": "spec",
                "problems": [f"unknown capsule spec {spec!r} — this verifier knows "
                             f"{SPEC} and {SPEC_V02}"]}

    problems: List[str] = []
    cert = payload["certificate"]
    doc_bytes = base64.b64decode(payload["document"]["b64"])
    if _sha256(doc_bytes) != cert.get("document_sha256"):
        problems.append("document bytes != certificate.document_sha256")
    recs = {}
    for r in payload["receipts"]:
        rb = base64.b64decode(r["b64"])
        recs[r["name"]] = rb
        want = (cert.get("receipts_sha256") or {}).get(r["name"])
        if _sha256(rb) != want:
            problems.append(f"receipt {r['name']!r} bytes != certificate hash")

    live = None
    if not problems:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / payload["document"]["name"]
            d.write_bytes(doc_bytes)
            rps = []
            for name, rb in recs.items():
                rp = Path(td) / name
                rp.write_bytes(rb)
                rps.append(rp)
            live = certify_doc(d, rps)
        if live["verdict"] != cert["verdict"]:
            problems.append(f"verdict not reproduced: live {live['verdict']} "
                            f"vs embedded {cert['verdict']}")
        if live["counts"] != cert["counts"]:
            problems.append(f"counts not reproduced: live {live['counts']} "
                            f"vs embedded {cert['counts']}")
        else:
            stored = {(e["line"], e.get("col"), str(e["token"])): e["status"]
                      for e in cert.get("ledger", [])}
            fresh = {(e["line"], e.get("col"), str(e["token"])): e["status"]
                     for e in live.get("ledger", [])}
            for k, s in sorted(stored.items()):
                if fresh.get(k) != s:
                    problems.append(f"ledger divergence at line {k[0]} token {k[2]!r}: "
                                    f"embedded {s} vs live {fresh.get(k)}")
    return {"ok": not problems, "problems": problems,
            "verdict": cert.get("verdict"), "counts": cert.get("counts"),
            "document": payload["document"]["name"],
            "spec": payload.get("spec"),
            "reproduced_at": None if live is None else "installed verifier"}


# ---------------------------------------------------------------------------------
# the rendered capsule (layer 1 lives here, inline, zero external requests)
# ---------------------------------------------------------------------------------

def _render_html(payload: dict) -> str:
    cert = payload["certificate"]
    verdict = cert.get("verdict", "?")
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = f"OATH Capsule — {payload['document']['name']}"
    return (_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__VERDICT__", verdict)
            .replace("__PIP__", payload["verifier"]["pip"])
            .replace("__PAYLOAD__", payload_json))


_TEMPLATE = r"""<!DOCTYPE html>
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


# ---------------------------------------------------------------------------------
# v0.2 — the agent-handoff capsule (diffgate record over summary + diff)
# Spec: papers/closed-model-frontier/SPEC_oath_capsule_v02_2026_08_31.md
# ---------------------------------------------------------------------------------

def _render_html_v02(payload: dict) -> str:
    g = payload["gate"]
    payload_json = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    title = (f"OATH Capsule — {payload['summary']['name']} × "
             f"{payload['diff']['name']}").replace("<", "")
    return (_TEMPLATE_V02
            .replace("__TITLE__", title)
            .replace("__VERDICT__", str(g.get("verdict", "?")))
            .replace("__PIP__", payload["verifier"]["pip"])
            .replace("__PAYLOAD__", payload_json))


_TEMPLATE_V02 = r"""<!DOCTYPE html>
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
letter-spacing:.06em}
.badge.held{background:var(--ok);color:#123}.badge.failed{background:var(--bad);color:#210}
.badge.warn{background:var(--warn);color:#321}.badge.tampered{background:#f33;color:#fff}
.meta{color:var(--mute);font-size:12px;margin-top:6px}
main{max-width:1080px;margin:0 auto;padding:24px}
h2{font-size:13px;letter-spacing:.14em;color:var(--sig);text-transform:uppercase;
margin:28px 0 10px}
pre.doc{white-space:pre-wrap;word-wrap:break-word;background:#150c20;border:1px solid
var(--rule);padding:18px;border-radius:3px;color:var(--bone);max-height:480px;
overflow:auto}
.band{border-radius:2px;padding:0 2px}
.band.vg{background:rgba(183,228,199,.18);color:var(--ok)}
.band.cb{background:rgba(216,152,134,.35);color:#ffd9cf;font-weight:700}
.band.ua{background:rgba(245,197,176,.18);color:var(--warn)}
.band.uc{color:var(--mute)}
.band .trunc{color:var(--mute);font-size:10px;font-weight:400}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{border:1px solid var(--rule);border-radius:3px;padding:12px;background:#150c20}
.card b{font-size:20px;display:block}.card span{color:var(--mute);font-size:11px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--rule);
padding:6px 8px;text-align:left;font-size:12px;vertical-align:top}th{color:var(--mute)}
td.v-VERIFIED{color:var(--ok)}td.v-CONTRADICTED{color:#ffb3a3;font-weight:700}
td.v-UNCHECKABLE{color:var(--warn)}
.dl-add{color:var(--ok)}.dl-del{color:var(--bad)}
.note{color:var(--mute);font-size:12px;margin:6px 0}
.disclose{color:var(--warn);font-size:12px;margin:6px 0}
footer{border-top:1px solid var(--rule);margin-top:32px;padding:18px 24px;
color:var(--mute);font-size:12px;max-width:1080px;margin-left:auto;margin-right:auto;
white-space:pre-wrap}
#tamper{display:none;background:#f33;color:#fff;padding:14px 24px;font-weight:700}
#why{color:#fff;font-weight:400;font-size:12px}
</style></head><body>
<noscript><div style="background:#f5c5b0;color:#321;padding:14px 24px;font-weight:700">
UNVERIFIED RENDERING — this page proves nothing without its verifier; run layer 2.
</div></noscript>
<div id="tamper">TAMPERED — this capsule does not verify. Nothing below can be trusted.
<div id="why"></div></div>
<header>
  <span class="badge" id="verdict">__VERDICT__</span>
  <span class="badge" id="integrity" style="background:#241830;color:var(--mute)">checking
  integrity…</span>
  <div class="meta" id="meta"></div>
</header>
<main>
  <h2>the gate, up front</h2><div class="grid" id="cards"></div>
  <h2>summary — every recorded sentence wearing its verdict</h2>
  <div class="meta"><span class="band vg">verified</span>
  <span class="band cb">contradicted</span> <span class="band ua">uncheckable</span>
  <span class="band uc">uncovered — listed, never judged</span></div>
  <div class="disclose" id="disclose"></div>
  <pre class="doc" id="doc"></pre>
  <h2>claims — the record, verbatim</h2>
  <table id="claims"><tr><th>kind</th><th>text</th><th>detail</th><th>verdict</th>
  <th>why</th></tr></table>
  <h2>the diff — display only, not parsed, not verified by this page</h2>
  <pre class="doc" id="diff"></pre>
  <h2>re-run it yourself (layer 2 — the real instrument)</h2>
  <pre class="doc">pip install __PIP__
python -m styxx.capsule verify this_file.html</pre>
</main>
<footer>what layer 1 proves (this page, offline): the embedded summary, diff, and gate
record are byte-for-byte the material this capsule binds — sha-256, recomputed in your
browser just now — and the sealed verdict follows arithmetically from the sealed claims.
this page re-runs nothing. the badge is a convenience, not an authority: this page cannot
prove its own javascript honest. decisions go through layer 2.

what layer 2 proves (pip install __PIP__): the gate record — verdict, every claim verdict,
every why-string, every count, every uncovered sentence — re-derives from the embedded
summary and diff by re-running the real instrument. one exception, printed when it
applies: unparsed_claims is observational and depends on whether styxx.claimdetect is
importable where the verifier runs; divergence there is reported, never treated as tamper.

what no layer proves: who minted this — no signatures; anyone can mint an internally
honest capsule over bytes of their choosing, and a re-mint over different bytes is a
different honest capsule, not a forgery this format can catch. when — the timestamp is
unsealed and unproven. that this diff was ever applied to any repository, branch, or
deployment — the capsule pins diff bytes, not repo state. that tests passed — environment
legs are refused at mint; tests_pass can only appear here as UNCHECKABLE, by construction.
that the summary's uncovered prose is true — uncovered sentences are listed, never judged;
coverage is not correctness. that this run is the only run — a capsule proves this
artifact, never the absence of others.

a capsule is a portable binding, not a portable oath of origin. nothing crosses unseen.
</footer>
<script type="application/json" id="oath-capsule">__PAYLOAD__</script>
<script>
(async () => {
  const $ = id => document.getElementById(id);
  const fail = msgs => {
    $('tamper').style.display = 'block';
    $('why').textContent = msgs.join(' · ');
    const ib = $('integrity');
    ib.textContent = 'INTEGRITY: FAILED'; ib.className = 'badge tampered';
  };
  const blocks = document.querySelectorAll('script#oath-capsule');
  if (blocks.length !== 1) return fail(['ambiguous payload: marker not unique']);
  let P;
  try { P = JSON.parse(blocks[0].textContent); }
  catch (e) { return fail(['payload unparseable']); }
  if (P.spec !== 'styxx-oath/capsule/v0.2') return fail(['unknown spec: ' + P.spec]);
  const g = P.gate;

  const b64b = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
  const hex = b => [...new Uint8Array(b)].map(x => x.toString(16).padStart(2, '0')).join('');
  const sha = async u8 => hex(await crypto.subtle.digest('SHA-256', u8));
  // RFC 8785 JCS, exact for the float-free gate record (parity-tested vs Python)
  const jcs = o => o === null ? 'null' : o === true ? 'true' : o === false ? 'false'
    : typeof o === 'number' ? String(o)
    : typeof o === 'string' ? JSON.stringify(o)
    : Array.isArray(o) ? '[' + o.map(jcs).join(',') + ']'
    : '{' + Object.keys(o).sort().map(k => JSON.stringify(k) + ':' + jcs(o[k])).join(',') + '}';

  // header — qualified badge, never bare
  const vb = $('verdict');
  vb.textContent = g.verdict + ' · ' + g.claims.length + ' claims checked · ' +
    g.uncovered_sentences + '/' + g.sentences_total + ' sentences uncovered';
  vb.className = 'badge ' +
    (g.verdict === 'PASS' ? (g.claims.length ? 'held' : 'warn') : 'failed');
  $('meta').textContent = P.summary.name + ' + ' + P.diff.name + ' · capsule ' + P.spec +
    ' · minted ' + P.created + ' (timestamp unsealed) · verifier styxx ' +
    P.verifier.styxx_version;

  // layer 1: hashes + the arithmetic folds the instrument guarantees
  const sumBytes = b64b(P.summary.b64), diffBytes = b64b(P.diff.b64);
  const bad = [];
  if (await sha(sumBytes) !== P.binding.summary.value) bad.push('summary bytes do not match binding');
  if (await sha(diffBytes) !== P.binding.diff.value) bad.push('diff bytes do not match binding');
  if (await sha(new TextEncoder().encode(jcs(g))) !== P.binding.gate.value) bad.push('gate record does not match binding');
  const nCon = g.claims.filter(c => c.verdict === 'CONTRADICTED').length;
  if (g.verdict !== (nCon ? 'FAIL' : 'PASS')) bad.push('verdict does not follow from the sealed claims');
  if (g.uncovered_sentences !== g.uncovered_texts.length) bad.push('uncovered count fold broken');
  if (g.measured !== true) bad.push('unmeasured record inside a minted capsule');
  if (g.base !== '(diff-text)' || g.head !== '(diff-text)') bad.push('base/head invariant broken');
  const ib = $('integrity');
  if (bad.length) { fail(bad); } else {
    ib.textContent = 'integrity: hashes match — verdict shown as recorded and follows from claims';
    ib.className = 'badge';
    ib.style.background = 'rgba(183,228,199,.15)'; ib.style.color = 'var(--ok)';
  }

  // cards
  const nVer = g.claims.filter(c => c.verdict === 'VERIFIED').length;
  const nUnc = g.claims.filter(c => c.verdict === 'UNCHECKABLE').length;
  const covered = g.sentences_total - g.uncovered_sentences;
  const cards = [
    ['claims verified', nVer], ['contradicted', nCon], ['uncheckable', nUnc],
    ['sentences covered', covered + '/' + g.sentences_total],
    ['uncovered — never judged', g.uncovered_sentences],
    ['unparsed claim-shaped', g.unparsed_claims.length],
  ];
  const cardsEl = $('cards');
  for (const [k, v] of cards) {
    const d = document.createElement('div'); d.className = 'card';
    const b = document.createElement('b'); b.textContent = String(v);
    const s = document.createElement('span'); s.textContent = k;
    d.appendChild(b); d.appendChild(s); cardsEl.appendChild(d);
  }

  // control characters (except newline/tab) render visibly, with a count
  let ctrl = 0;
  const clean = s => s.replace(
    /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u200e\u200f\u202a-\u202e\u2066-\u2069]/g,
    () => { ctrl++; return '�'; });

  // paint the summary: locate recorded texts — never re-split, never re-judge
  const text = new TextDecoder('utf-8').decode(sumBytes);
  const ranges = [];
  let unlocated = 0;
  const locate = (t, cls) => {
    if (!t) return;
    let from = 0, at;
    while ((at = text.indexOf(t, from)) !== -1) {
      if (!ranges.some(r => at < r.end && at + t.length > r.start)) {
        ranges.push({ start: at, end: at + t.length, cls, trunc: t.length === 160 });
        return;
      }
      from = at + 1;
    }
    unlocated++;
  };
  for (const c of g.claims)
    locate(c.text, c.verdict === 'VERIFIED' ? 'vg'
      : c.verdict === 'CONTRADICTED' ? 'cb' : 'ua');
  for (const u of g.uncovered_texts) locate(u, 'uc');
  ranges.sort((a, b) => a.start - b.start);
  const pre = $('doc');
  let pos = 0;
  for (const r of ranges) {
    if (r.start > pos) pre.appendChild(document.createTextNode(clean(text.slice(pos, r.start))));
    const span = document.createElement('span'); span.className = 'band ' + r.cls;
    span.textContent = clean(text.slice(r.start, r.end));
    if (r.trunc) {
      const m = document.createElement('span'); m.className = 'trunc';
      m.textContent = ' …record truncates at 160'; span.appendChild(m);
    }
    pre.appendChild(span); pos = r.end;
  }
  if (pos < text.length) pre.appendChild(document.createTextNode(clean(text.slice(pos))));
  const dis = [];
  if (unlocated) dis.push(unlocated + ' recorded sentence(s) could not be located for painting — see the claims table');
  if (ctrl) dis.push(ctrl + ' invisible control character(s) rendered as �');
  $('disclose').textContent = dis.join(' · ');

  // claims table — every why-string verbatim
  const tbl = $('claims');
  for (const c of g.claims) {
    const tr = document.createElement('tr');
    for (const [val, cls] of [[c.kind, ''], [c.text, ''],
        [JSON.stringify(c.detail), ''], [c.verdict, 'v-' + c.verdict],
        [c.why + (c.kind === 'tests_pass' ? ' [environment leg — refused at mint by construction]' : ''), '']]) {
      const td = document.createElement('td');
      if (cls) td.className = cls;
      td.textContent = clean(String(val));
      tr.appendChild(td);
    }
    tbl.appendChild(tr);
  }

  // diff panel — display only
  const dpre = $('diff');
  for (const line of new TextDecoder('utf-8').decode(diffBytes).split('\n')) {
    const span = document.createElement('span');
    if (line.startsWith('+') && !line.startsWith('+++')) span.className = 'dl-add';
    else if (line.startsWith('-') && !line.startsWith('---')) span.className = 'dl-del';
    span.textContent = clean(line) + '\n';
    dpre.appendChild(span);
  }

  // layer-2 command with this file's actual name
  document.querySelectorAll('main pre.doc')[3] &&
    (document.querySelectorAll('main pre.doc')[3].textContent =
      'pip install ' + P.verifier.pip + '\n' +
      'python -m styxx.capsule verify ' + location.pathname.split('/').pop());
})();
</script>
</body></html>
"""


def _canonical_text_bytes(path: Path, what: str) -> tuple[str, bytes]:
    """Read as the instruments read: universal-newline text, re-encoded UTF-8.

    The v0.1 CRLF lesson, applied to every v0.2 input before it bites: the gate
    consumes TEXT, so the capsule binds the text bytes, not the checkout's.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise SystemExit(f"REFUSED: cannot read {what} as UTF-8 text: {e}")
    return text, text.encode("utf-8")


def _v02_folds(gate: dict) -> List[str]:
    """The arithmetic invariants a genuine strict=False, run=None gate record
    satisfies — mirrored verbatim by the capsule's layer-1 JS. A violation can
    only mean tampering (or an instrument this verifier does not know)."""
    problems = []
    contradicted = sum(1 for c in gate.get("claims", [])
                       if c.get("verdict") == "CONTRADICTED")
    want = "FAIL" if contradicted else "PASS"
    if gate.get("verdict") != want:
        problems.append(f"verdict fold: {gate.get('verdict')!r} does not follow from "
                        f"{contradicted} CONTRADICTED claims (expected {want!r})")
    if gate.get("uncovered_sentences") != len(gate.get("uncovered_texts", [])):
        problems.append("uncovered count fold: uncovered_sentences != len(uncovered_texts)")
    if gate.get("measured") is not True:
        problems.append("measured invariant: a minted v0.2 capsule can only carry a "
                        "measured gate")
    if gate.get("base") != "(diff-text)" or gate.get("head") != "(diff-text)":
        problems.append("base/head invariant: v0.2 gates are minted from diff text only")
    return problems


def _gate_binding_hash(gate: dict) -> str:
    from styxx.attestation import jcs
    return _sha256(jcs(gate).encode("utf-8"))


def create_capsule_diffgate(summary: Path, diff: Path, out: Path,
                            gate_path: Path | None = None) -> Path:
    """Mint the agent-handoff capsule — refusing, loudly, to mint one that lies.

    The gate embedded is ALWAYS the live mint-time re-run over the canonical
    bytes (a supplied --gate is only cross-checked, never sealed), so layer-2
    reproduction succeeds by construction and the record is a pure function of
    (summary bytes, diff bytes): strict=False, run=None, nothing self-reported.
    """
    from styxx.diffgate import gate_diff_text
    from styxx._version import __version__

    summary_text, summary_bytes = _canonical_text_bytes(summary, "summary")   # R1/R3
    diff_text, diff_bytes = _canonical_text_bytes(diff, "diff")               # R1/R3

    live = gate_diff_text(summary_text, diff_text, run=None, strict=False).to_dict()

    if gate_path is not None:
        try:
            supplied = json.loads(gate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:                          # R1
            raise SystemExit(f"REFUSED: cannot parse supplied gate: {e}")
        if supplied.get("diffgate") != "v0":                                  # R2
            raise SystemExit(f"REFUSED: unknown diffgate version "
                             f"{supplied.get('diffgate')!r} — cannot re-run it")
        for c in supplied.get("claims", []):                                  # R4
            if c.get("kind") == "tests_pass" and c.get("verdict") != "UNCHECKABLE":
                raise SystemExit(
                    "REFUSED: environment legs cannot be capsuled in v0.2; a "
                    "--run-resolved tests_pass verdict would require executing an "
                    "embedded shell string to verify. re-gate without --run.")
        diverged = [k for k in live                                           # R5
                    if k not in ("base", "head") and supplied.get(k) != live[k]]
        if diverged == ["unparsed_claims"]:
            raise SystemExit(
                "REFUSED: styxx.claimdetect availability differs between the gate's "
                "environment and this mint environment — re-gate here or omit --gate.")
        if (diverged == ["verdict"]
                and any(c.get("verdict") == "UNCHECKABLE"
                        for c in supplied.get("claims", []))):
            raise SystemExit(
                "REFUSED: v0.2 gates are non-strict by policy; strictness is a "
                "read-side policy — every UNCHECKABLE is visible in the record. "
                "re-gate non-strict or omit --gate.")
        if diverged:
            raise SystemExit(
                "REFUSED: supplied gate does not reproduce from these bytes — stale, "
                "forged, or produced by a different code path "
                f"(diverging fields: {diverged}); re-gate from the exported diff or "
                "omit --gate.")

    if not live.get("measured", False):                                       # R6
        raise SystemExit(
            f"REFUSED: gate measured nothing (why_unmeasured: "
            f"{live.get('why_unmeasured')!r}) — a capsule cannot carry proof of a "
            "non-measurement.")

    folds = _v02_folds(live)
    if folds:  # cannot happen at the pinned instrument; a fold here means skew
        raise SystemExit(f"REFUSED: live gate violates its own invariants: {folds}")

    payload = {
        "spec": SPEC_V02,
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {"name": summary.name, "b64": _b64(summary_bytes)},
        "diff": {"name": diff.name, "b64": _b64(diff_bytes)},
        "gate": live,
        "binding": {
            "summary": {"alg": "sha256", "value": _sha256(summary_bytes)},
            "diff": {"alg": "sha256", "value": _sha256(diff_bytes)},
            "gate": {"alg": "sha256-jcs", "value": _gate_binding_hash(live)},
        },
        "verifier": {"styxx_version": __version__, "pip": f"styxx=={__version__}"},
    }
    html = _render_html_v02(payload)
    if html.count(_BEGIN) != 1:                                               # R7
        raise SystemExit("REFUSED: payload marker is not unique in the rendered "
                         "capsule — refusing to write an ambiguous artifact")
    out.write_text(html, encoding="utf-8")
    rep = verify_capsule(out)                                                 # R7
    if not rep["ok"]:
        out.unlink(missing_ok=True)
        raise SystemExit(f"REFUSED: the freshly minted capsule fails its own "
                         f"layer-2 verify: {rep['problems']}")
    return out


def _verify_capsule_v02(html: str, payload: dict) -> dict:
    from styxx.diffgate import gate_diff_text

    result = {"ok": False, "spec": SPEC_V02, "stage": "parse",
              "problems": [], "advisory": [],
              "verdict": None, "gate_reproduced": False, "reproduced_at": None,
              "summary": None, "diff": None}
    problems: List[str] = []
    advisory: List[str] = []

    if html.count(_BEGIN) != 1:
        result["problems"] = ["ambiguous payload: marker occurs more than once"]
        return result

    gate = payload.get("gate") or {}
    binding = payload.get("binding") or {}
    result["verdict"] = gate.get("verdict")
    result["summary"] = (payload.get("summary") or {}).get("name")
    result["diff"] = (payload.get("diff") or {}).get("name")

    # stage: binding — every embedded byte vs its sealed hash
    result["stage"] = "binding"
    summary_bytes = base64.b64decode(payload["summary"]["b64"])
    diff_bytes = base64.b64decode(payload["diff"]["b64"])
    if _sha256(summary_bytes) != (binding.get("summary") or {}).get("value"):
        problems.append("summary bytes != binding.summary")
    if _sha256(diff_bytes) != (binding.get("diff") or {}).get("value"):
        problems.append("diff bytes != binding.diff")
    if _gate_binding_hash(gate) != (binding.get("gate") or {}).get("value"):
        problems.append("gate record != binding.gate (sha256-jcs)")
    if problems:
        result["problems"] = problems
        return result

    # stage: re-execution — the decisive leg. run=None, strict=False, always.
    result["stage"] = "reproduced"
    live = gate_diff_text(summary_bytes.decode("utf-8"), diff_bytes.decode("utf-8"),
                          run=None, strict=False).to_dict()
    if live.get("diffgate") != gate.get("diffgate"):
        problems.append(
            f"INSTRUMENT SKEW: installed diffgate {live.get('diffgate')!r} vs embedded "
            f"{gate.get('diffgate')!r} — reproduce under `pip install "
            f"{payload.get('verifier', {}).get('pip', 'styxx')}` before treating this "
            "as tamper")
    else:
        try:
            import styxx.claimdetect  # noqa: F401  — availability probe only
            _claimdetect = True
        except Exception:
            _claimdetect = False
        for k in live:
            if k == "unparsed_claims":
                if not _claimdetect:
                    advisory.append("SKIPPED: unparsed_claims (claimdetect unavailable "
                                    "here — observational field)")
                elif live[k] != gate.get(k):
                    advisory.append(
                        f"unparsed_claims diverges (embedded {gate.get(k)!r} vs live "
                        f"{live[k]!r}) — observational, environment-dependent, never "
                        "treated as tamper")
                continue
            if live[k] != gate.get(k):
                problems.append(f"gate.{k} not reproduced: embedded {gate.get(k)!r} "
                                f"vs live {live[k]!r}")

    result["problems"] = problems
    result["advisory"] = advisory
    result["gate_reproduced"] = not problems
    result["ok"] = not problems
    result["reproduced_at"] = "installed verifier" if not problems else None
    return result


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="styxx.capsule",
        description="Proof-carrying documents (v0.1) and agent-handoff capsules (v0.2)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="mint a capsule: DOC RECEIPTS... --cert CERT "
                                      "(v0.1) | SUMMARY DIFF [--gate GATE] (v0.2)")
    c.add_argument("document", help="the document (v0.1) or the agent summary (v0.2)")
    c.add_argument("inputs", nargs="+",
                   help="receipts (v0.1, with --cert) or the unified diff (v0.2)")
    c.add_argument("--cert", default=None, help="certificate JSON — selects v0.1")
    c.add_argument("--gate", default=None,
                   help="optional diffgate GATE.json to cross-check (v0.2)")
    c.add_argument("--out", default=None)
    v = sub.add_parser("verify", help="layer-2: re-run the real instrument on a capsule")
    v.add_argument("capsule")
    a = ap.parse_args(argv)

    if a.cmd == "create":
        if a.cert and a.gate:
            ap.error("--cert (v0.1) and --gate (v0.2) are mutually exclusive")
        out = Path(a.out) if a.out else Path(a.document).with_suffix(".capsule.html")
        if a.cert:
            p = create_capsule(Path(a.document), [Path(r) for r in a.inputs],
                               Path(a.cert), out)
        else:
            if len(a.inputs) != 1:
                ap.error("v0.2 mint takes exactly two positionals: SUMMARY DIFF "
                         "(use --cert for a v0.1 document capsule)")
            p = create_capsule_diffgate(Path(a.document), Path(a.inputs[0]), out,
                                        Path(a.gate) if a.gate else None)
        print(f"capsule minted -> {p}")
        return 0

    rep = verify_capsule(Path(a.capsule))
    if rep.get("spec") == SPEC_V02:
        print(f"capsule: {rep.get('summary')} + {rep.get('diff')}  spec {rep.get('spec')}")
        print(f"embedded gate verdict: {rep.get('verdict')}")
        for adv in rep.get("advisory", []):
            print(f"  advisory: {adv}")
        if rep["ok"]:
            print("VERIFIED: bytes match their bindings and the gate record re-derives "
                  "at the installed instrument.")
            return 0
    else:
        print(f"capsule: {rep.get('document')}  spec {rep.get('spec')}")
        print(f"embedded verdict: {rep.get('verdict')}  counts {rep.get('counts')}")
        if rep["ok"]:
            print("VERIFIED: bytes match the certificate and the verdict reproduces at "
                  "the installed verifier.")
            return 0
    print("CAPSULE FAILS VERIFICATION:")
    for p_ in rep.get("problems", []):
        print(f"  - {p_}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

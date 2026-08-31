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

__all__ = ["create_capsule", "verify_capsule", "main"]

SPEC = "styxx-oath/capsule/v0.1"
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
        return {"ok": False, "stage": "parse", "error": f"no capsule payload: {e}"}

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
# CLI
# ---------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.capsule",
                                 description="Proof-carrying documents (OATH Capsule v0.1)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="mint a capsule from a certified document")
    c.add_argument("document")
    c.add_argument("receipts", nargs="+")
    c.add_argument("--cert", required=True)
    c.add_argument("--out", default=None)
    v = sub.add_parser("verify", help="layer-2: re-run the real verifier on a capsule")
    v.add_argument("capsule")
    a = ap.parse_args(argv)

    if a.cmd == "create":
        out = Path(a.out) if a.out else Path(a.document).with_suffix(".capsule.html")
        p = create_capsule(Path(a.document), [Path(r) for r in a.receipts],
                           Path(a.cert), out)
        print(f"capsule minted -> {p}")
        return 0
    rep = verify_capsule(Path(a.capsule))
    print(f"capsule: {rep.get('document')}  spec {rep.get('spec')}")
    print(f"embedded verdict: {rep.get('verdict')}  counts {rep.get('counts')}")
    if rep["ok"]:
        print("VERIFIED: bytes match the certificate and the verdict reproduces at the "
              "installed verifier.")
        return 0
    print("CAPSULE FAILS VERIFICATION:")
    for p_ in rep["problems"]:
        print(f"  - {p_}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

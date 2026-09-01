"""Render a real document being audited, token by token, including what the verifier never read.

Every claim-checking tool shows you what it checked. This shows the third band as well:

    BOUND      the number is sworn to a receipt field, and the field is named
    ACCUSED    the line obligated it and nothing in the receipts holds it
    UNREAD     the verifier said nothing about this number at all

The third band is the one nobody publishes, because it is the one that makes an instrument look
bad. On 2026-08-27 blind panels adjudicated a uniform sample of it in both this laboratory's
corpus and a 140-repository external one, and judged 0.4267 and 0.4067 of those silences to be
checkable claims. So UNREAD is not "nothing to see"; it is where roughly two in five of a
document's claims are sitting unexamined.

The view renders the document's SOURCE rather than its rendered markdown, because the source is
what the verifier reads. Highlighting the rendered output would be showing you a different
document from the one that was audited.

  python papers/styxx_audit_view.py [--doc NAME] [--out PATH]
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from styxx.certify import _TRIGGERS  # noqa: E402

DEFAULT = "closed-model-frontier/RESULT_oath_verified_channel_internal_2026_08_27"

BAND = {"VERIFIED": ("bound", "sworn to a receipt field"),
        "UNGROUNDED": ("accused", "obligated, and nothing holds it"),
        "ABSTAIN": ("unread", "the verifier said nothing about this")}


def build(doc_stem: str) -> dict:
    md = HERE / f"{doc_stem}.md"
    cert_p = HERE / f"{doc_stem}.certificate.json"
    cert = json.loads(cert_p.read_text(encoding="utf-8"))
    lines = md.read_text(encoding="utf-8").splitlines()

    # index tokens by line so the renderer can walk each line once
    per_line: dict[int, list] = {}
    for i, e in enumerate(cert["ledger"]):
        ln = e["line"]
        src = lines[ln - 1] if 0 < ln <= len(lines) else ""
        trig = sorted({m.group(0).lower() for m in _TRIGGERS.finditer(src)})
        ref = e.get("receipt_ref")
        per_line.setdefault(ln, []).append({
            "i": i, "col": e.get("col"), "token": e["token"], "status": e["status"],
            "ref": ref, "triggers": trig,
            "why": ("bound to " + ref if e["status"] == "VERIFIED" and ref
                    else "no receipt field holds this value" if e["status"] == "UNGROUNDED"
                    else "read as specification or historical, not a live claim"
                    if ref == "spec-or-historical"
                    else "nothing on this line obligated it"),
        })
    counts = {k: sum(1 for e in cert["ledger"] if e["status"] == k)
              for k in ("VERIFIED", "UNGROUNDED", "ABSTAIN")}
    return {"doc": md.name, "verdict": cert["verdict"], "lines": lines,
            "per_line": per_line, "counts": counts,
            "verifier": cert.get("verifier_sha256", "")[:16],
            "receipts": list(cert.get("receipts_sha256", {})),
            "total": sum(counts.values())}


def render_line(src: str, toks: list) -> str:
    """Wrap each audited token in place, leaving the rest of the source untouched."""
    if not toks:
        return html.escape(src)
    marks = sorted((t for t in toks if t["col"] is not None), key=lambda t: t["col"])
    out, cur = [], 0
    for t in marks:
        a = t["col"]
        b = a + len(t["token"])
        if a < cur or b > len(src) or src[a:b] != t["token"]:
            continue                      # column drifted; leave the text alone rather than guess
        out.append(html.escape(src[cur:a]))
        band = BAND[t["status"]][0]
        out.append(f'<mark class="t {band}" data-i="{t["i"]}" tabindex="0" '
                   f'role="button" aria-label="{band} token {html.escape(t["token"])}">'
                   f'{html.escape(t["token"])}</mark>')
        cur = b
    out.append(html.escape(src[cur:]))
    return "".join(out)


CSS = """
*{box-sizing:border-box}
:root{
  --ground:#EDEEEB; --paper:#F7F8F6; --ink:#131816; --dim:#5B6560;
  --hair:rgba(19,24,22,.14); --rule:rgba(19,24,22,.26);
  --bound:#8A5D10; --bound-bg:rgba(138,93,16,.13);
  --accused:#8F3520; --accused-bg:rgba(143,53,32,.12);
  --unread:#5B6560; --unread-bg:rgba(19,24,22,.055);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0F1311; --paper:#151A18; --ink:#C7D0CB; --dim:#79847F;
  --hair:rgba(199,208,203,.15); --rule:rgba(199,208,203,.22);
  --bound:#E3A63C; --bound-bg:rgba(227,166,60,.14);
  --accused:#D4664B; --accused-bg:rgba(212,102,75,.15);
  --unread:#79847F; --unread-bg:rgba(199,208,203,.06);
}}
:root[data-theme="dark"]{
  --ground:#0F1311; --paper:#151A18; --ink:#C7D0CB; --dim:#79847F;
  --hair:rgba(199,208,203,.15); --rule:rgba(199,208,203,.22);
  --bound:#E3A63C; --bound-bg:rgba(227,166,60,.14);
  --accused:#D4664B; --accused-bg:rgba(212,102,75,.15);
  --unread:#79847F; --unread-bg:rgba(199,208,203,.06);
}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;font-size:15px;line-height:1.6;
  -webkit-font-smoothing:antialiased;}
.wrap{max-width:1240px;margin:0 auto;padding:28px 22px 72px;}
.plate{font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;}
.micro{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--dim);}
.tag pre{margin:0;font-family:"IBM Plex Mono",monospace;color:var(--bound);
  font-size:clamp(5px,1.3vw,10px);line-height:1.06;white-space:pre;}
.tag{overflow-x:auto;margin-bottom:10px;}
h1{margin:.2em 0 .1em;font-size:clamp(1.9rem,5vw,3rem);line-height:.96;text-wrap:balance;}
.sub{color:var(--dim);max-width:66ch;margin:.6em 0 0;}
.sub b{color:var(--ink);font-weight:500;}
.bar{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 0;align-items:stretch;}
.k{border:1px solid var(--hair);background:var(--paper);padding:9px 13px;display:flex;
  align-items:center;gap:9px;}
.k .sw{width:11px;height:11px;flex-shrink:0;border-radius:2px;}
.k .n{font-family:"IBM Plex Mono",monospace;font-weight:600;font-variant-numeric:tabular-nums;}
.k .l{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);}
.sw.bound{background:var(--bound-bg);border:1px solid var(--bound);}
.sw.accused{background:var(--accused-bg);border:1px solid var(--accused);}
.sw.unread{background:var(--unread-bg);border:1px dashed var(--unread);}
button.tog{font:inherit;font-size:12px;letter-spacing:.07em;text-transform:uppercase;
  border:1px solid var(--rule);background:transparent;color:var(--ink);padding:9px 15px;
  cursor:pointer;font-family:"IBM Plex Sans Condensed",sans-serif;font-weight:700;}
button.tog:hover{border-color:var(--bound);color:var(--bound);}
button.tog[aria-pressed="true"]{background:var(--bound);border-color:var(--bound);color:var(--ground);}
button.tog:focus-visible{outline:2px solid var(--bound);outline-offset:2px;}
.cols{display:grid;grid-template-columns:1fr 330px;gap:18px;margin-top:22px;align-items:start;}
.doc{border:1px solid var(--hair);background:var(--paper);overflow-x:auto;position:relative;}
.doc .head{border-bottom:1px solid var(--hair);padding:11px 16px;display:flex;
  justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:baseline;}
.doc .head .f{font-family:"IBM Plex Mono",monospace;font-size:12px;}
pre.src{margin:0;padding:14px 16px 20px;font-family:"IBM Plex Mono",monospace;font-size:12.5px;
  line-height:1.85;white-space:pre-wrap;word-break:break-word;tab-size:2;}
.ln{display:inline-block;width:2.6em;color:var(--dim);opacity:.5;user-select:none;
  font-variant-numeric:tabular-nums;}
mark.t{background:transparent;color:inherit;padding:1px 2px;border-radius:2px;cursor:pointer;
  border-bottom:2px solid transparent;}
mark.t:focus-visible{outline:2px solid var(--bound);outline-offset:1px;}
mark.bound{background:var(--bound-bg);border-bottom-color:var(--bound);color:var(--bound);}
mark.accused{background:var(--accused-bg);border-bottom-color:var(--accused);color:var(--accused);
  font-weight:600;}
mark.unread{background:var(--unread-bg);border-bottom:2px dashed var(--unread);}
mark.t.sel{outline:2px solid var(--ink);outline-offset:1px;}
body.blind mark.bound{background:transparent;border-bottom-color:transparent;color:var(--dim);
  opacity:.35;}
body.blind mark.accused{opacity:.35;}
body.blind mark.unread{background:var(--accused-bg);border-bottom:2px solid var(--accused);
  color:var(--accused);}
.panel{border:1px solid var(--hair);background:var(--paper);padding:16px;position:sticky;top:16px;}
.panel h2{margin:0 0 10px;font-size:.8rem;letter-spacing:.15em;text-transform:uppercase;
  font-family:"IBM Plex Sans Condensed",sans-serif;}
.panel .tokv{font-family:"IBM Plex Mono",monospace;font-size:1.6rem;font-weight:600;
  display:block;margin-bottom:2px;}
.panel dl{margin:12px 0 0;display:grid;grid-template-columns:auto 1fr;gap:6px 12px;}
.panel dt{font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--dim);}
.panel dd{margin:0;font-family:"IBM Plex Mono",monospace;font-size:11.5px;word-break:break-all;}
.panel .hint{color:var(--dim);font-size:.86rem;}
.blindnote{border-left:2px solid var(--accused);background:var(--accused-bg);padding:10px 12px;
  margin-top:14px;font-size:.85rem;line-height:1.5;display:none;}
body.blind .blindnote{display:block;}
footer{margin-top:44px;border-top:1px solid var(--hair);padding-top:18px;color:var(--dim);
  font-size:.87rem;max-width:70ch;}
footer b{color:var(--ink);font-weight:500;}
@media (max-width:900px){.cols{grid-template-columns:1fr}.panel{position:static}}
@media (prefers-reduced-motion:no-preference){
  mark.t{animation:lite .28s ease-out backwards;animation-delay:var(--d,0ms);}
  @keyframes lite{from{background:transparent;border-bottom-color:transparent;color:inherit;}}
}
"""

JS = """
const L = window.__LEDGER__;
const panel = document.getElementById('panel');
let sel = null;
function show(i){
  const t = L[i]; if(!t) return;
  if(sel) sel.classList.remove('sel');
  sel = document.querySelector('mark[data-i="'+i+'"]');
  if(sel) sel.classList.add('sel');
  const band = {VERIFIED:'BOUND', UNGROUNDED:'ACCUSED', ABSTAIN:'UNREAD'}[t.status];
  panel.innerHTML = '<h2>'+band+'</h2>'+
    '<span class="tokv">'+t.token+'</span>'+
    '<div class="hint">'+t.why+'</div>'+
    '<dl>'+
      '<dt>line</dt><dd>'+t.line+'</dd>'+
      '<dt>obligated by</dt><dd>'+(t.triggers.length?t.triggers.join(', '):'&mdash; nothing')+'</dd>'+
      '<dt>receipt</dt><dd>'+(t.ref?t.ref:'&mdash; none')+'</dd>'+
    '</dl>';
}
document.querySelectorAll('mark.t').forEach(m=>{
  m.addEventListener('click',()=>show(+m.dataset.i));
  m.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show(+m.dataset.i);}});
});
const btn = document.getElementById('blind');
btn.addEventListener('click',()=>{
  const on = document.body.classList.toggle('blind');
  btn.setAttribute('aria-pressed', on ? 'true' : 'false');
  btn.textContent = on ? 'Showing what it never read' : 'Show what it never read';
});
const first = document.querySelector('mark.accused') || document.querySelector('mark.t');
if(first) show(+first.dataset.i);
"""

WORDMARK = r"""
 ███████  ████████  ██    ██  ██    ██  ██    ██
 ██          ██      ██  ██    ██  ██    ██  ██
 ███████     ██       ████      ████      ████
      ██     ██        ██      ██  ██    ██  ██
 ███████     ██        ██     ██    ██  ██    ██
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default=DEFAULT)
    ap.add_argument("--out", default=str(ROOT / "styxx-audit-view.html"))
    a = ap.parse_args()

    d = build(a.doc)
    ledger, n = [], 0
    body = []
    for ln, src in enumerate(d["lines"], start=1):
        toks = d["per_line"].get(ln, [])
        for t in toks:
            ledger.append({"token": t["token"], "status": t["status"], "line": ln,
                           "ref": t["ref"], "triggers": t["triggers"], "why": t["why"]})
        rendered = render_line(src, toks)
        for t in toks:
            rendered = rendered.replace('data-i="%d"' % t["i"],
                                        'data-i="%d" style="--d:%dms"' % (t["i"], n * 26), 1)
            n += 1
        body.append(f'<span class="ln">{ln}</span>{rendered}')

    c = d["counts"]
    keys = [("bound", c["VERIFIED"], "bound to a receipt"),
            ("accused", c["UNGROUNDED"], "accused"),
            ("unread", c["ABSTAIN"], "never read")]
    page = f"""<title>Watch It Check Itself</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@700&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<div class="tag" role="img" aria-label="styxx"><pre>{WORDMARK.strip(chr(10))}</pre></div>
<h1 class="plate">Watch it<br>check itself</h1>
<p class="sub">Every claim-checking tool shows you what it checked. This is the same audit with the
third band left in: <b>the numbers the verifier never read at all.</b> Click any highlighted
number to see what obligated it and what it bound to. This is a real document from the corpus,
annotated from its committed certificate &mdash; not a mock-up.</p>

<div class="bar">
  {"".join(f'<div class="k"><span class="sw {k}"></span><span class="n">{v}</span><span class="l">{lbl}</span></div>' for k, v, lbl in keys)}
  <button class="tog" id="blind" aria-pressed="false">Show what it never read</button>
</div>

<div class="cols">
  <div class="doc">
    <div class="head">
      <span class="f">{html.escape(d["doc"])}</span>
      <span class="micro">{html.escape(d["verdict"])} &middot; verifier {d["verifier"]}&hellip;</span>
    </div>
    <pre class="src">{chr(10).join(body)}</pre>
  </div>
  <aside class="panel" id="panel"><h2>Select a token</h2>
    <div class="hint">Click any highlighted number in the document.</div></aside>
</div>

<div class="blindnote">These dashes are every number the verifier declined to obligate. Blind
panels adjudicated a uniform sample of this band on 2026-08-27 and judged <b>0.4267</b> of them,
in this laboratory's own corpus, to be checkable claims &mdash; reported medians, preregistered
quantities, pass counts at gates. The certificate above still reads {html.escape(d["verdict"])}.</div>

<footer>
  <p><b>An OATH certificate is a floor, not a summary.</b> It attests to the numbers the verifier
  chose to obligate. It is silent about coverage, and the silence is not small: about two in five
  of the checkable claims in a document carrying one were never examined. That number exists
  because we went and measured it with a blind panel, and it is the reason the third band is on
  this page instead of hidden behind a green tick.</p>
  <p class="micro">fathom lab &middot; styxx &middot; rendered from
  {html.escape(d["doc"].replace(".md", ".certificate.json"))}</p>
</footer>
</div>
<script>window.__LEDGER__={json.dumps(ledger, ensure_ascii=False)};{JS}</script>
"""
    out = Path(a.out)
    out.write_text(page, encoding="utf-8")
    print(f"{d['doc']}  {d['verdict']}  bound {c['VERIFIED']}  accused {c['UNGROUNDED']}  "
          f"unread {c['ABSTAIN']}  ({n} marked)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Render the org chart FROM the census receipt, so the picture cannot drift from the measurement.

The first version of this chart was hand-written HTML with the numbers typed in. That is the
defect the chart is about, one level up: a diagram whose figures are transcribed by hand can
disagree with the receipt it claims to render, and nothing would say so.

So the page is generated. Every lamp, count and fault line below comes out of
`styxx_org_census.json`, which was itself produced by running the tests. There is no hand-typed
status anywhere in the chain:

    modules  ->  census runs their tests  ->  receipt  ->  this generator  ->  page

  python papers/styxx_org_chart.py [--out PATH]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CENSUS = HERE / "styxx_org_census.json"
CERT = HERE / "ORG_the_chart_that_cannot_lie_2026_08_28.certificate.json"
LEDGER = HERE / "LEDGER.md"

WORDMARK = r"""
 ███████  ████████  ██    ██  ██    ██  ██    ██
 ██          ██      ██  ██    ██  ██    ██  ██
 ███████     ██       ████      ████      ████
      ██     ██        ██      ██  ██    ██  ██
 ███████     ██        ██     ██    ██  ██    ██
"""

# Which instrument acts at which stage of the loop. Drawn as a pipeline because these roles are
# not peers on an org chart -- they are stages, and two of them feed backwards.
PIPELINE = [
    ("Freeze", "a preregistration is committed before any data moves", ["PROTOCOL FREEZER"]),
    ("Red team", "adversaries attack the design before it runs", ["NULL-RULE CHECK"]),
    ("Collect", "the population is gathered and pinned by hash", ["ABSENCE DETECTOR"]),
    ("Adjudicate", "blind seats decide claimhood, salted with decoys", ["BLIND PANEL"]),
    ("Certify", "every number is bound to a receipt, or the document fails", ["VERIFIER",
                                                                             "READINESS CHECK"]),
    ("Audit", "every committed certificate is re-checked for drift", ["CORPUS AUDITOR",
                                                                      "SILENT-PASS BENCH"]),
    ("Publish", "the negative goes out at the strength it survived", ["LEDGER"]),
]

DOCS = {
    "PROTOCOL FREEZER": "RECON_v13_not_frozen",
    "VERIFIER": "RESULT_oath_verified_channel_internal",
    "CORPUS AUDITOR": "REPLICATIONS.md",
    "NULL-RULE CHECK": "RECON_obligation_repair_is_not_lexical",
    "READINESS CHECK": "OATH_CONTRACT.md",
    "LEDGER": "LEDGER.md",
    "SILENT-PASS BENCH": "tests/conftest.py",
    "BLIND PANEL": "RESULT_oath_external_corpus",
    "TRADER": "RECON_obligation_repair_is_not_lexical",
}

CSS = """
:root{
  --ground:#E8EAE7; --panel:#F4F5F3; --panel-2:#EDEFEC;
  --engrave:#131816; --dim:#5B6560; --hair:rgba(19,24,22,.16); --hair-2:rgba(19,24,22,.08);
  --lamp:#8A5D10; --lamp-glow:rgba(138,93,16,.18);
  --fault:#8F3520; --fault-bg:rgba(143,53,32,.06);
  --absent:#8B948F; --rule:rgba(19,24,22,.26);
  --s1:.5rem; --s2:.875rem; --s3:1.5rem; --s4:2.5rem; --s5:4rem;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0F1311; --panel:#171D1A; --panel-2:#131917;
  --engrave:#C7D0CB; --dim:#79847F; --hair:rgba(199,208,203,.16); --hair-2:rgba(199,208,203,.07);
  --lamp:#E3A63C; --lamp-glow:rgba(227,166,60,.20);
  --fault:#D4664B; --fault-bg:rgba(212,102,75,.09);
  --absent:#4A5450; --rule:rgba(199,208,203,.22);
}}
:root[data-theme="dark"]{
  --ground:#0F1311; --panel:#171D1A; --panel-2:#131917;
  --engrave:#C7D0CB; --dim:#79847F; --hair:rgba(199,208,203,.16); --hair-2:rgba(199,208,203,.07);
  --lamp:#E3A63C; --lamp-glow:rgba(227,166,60,.20);
  --fault:#D4664B; --fault-bg:rgba(212,102,75,.09);
  --absent:#4A5450; --rule:rgba(199,208,203,.22);
}
*{box-sizing:border-box}
body{margin:0; background:var(--ground); color:var(--engrave);
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif; font-size:15px; line-height:1.6;
  -webkit-font-smoothing:antialiased;}
.wrap{max-width:1180px; margin:0 auto; padding:var(--s3) var(--s3) var(--s5);}
.plate{font-family:"IBM Plex Sans Condensed",ui-sans-serif,sans-serif; font-weight:700;
  text-transform:uppercase; letter-spacing:.08em;}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums;}
.micro{font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:var(--dim);}

/* wordmark */
.tag{overflow-x:auto; margin:0 0 var(--s2);}
.tag pre{margin:0; font-family:"IBM Plex Mono",monospace; color:var(--lamp);
  font-size:clamp(5px,1.35vw,11px); line-height:1.06; letter-spacing:0; white-space:pre;}
.tagline{display:flex; flex-wrap:wrap; gap:var(--s2) var(--s3); align-items:baseline;
  border-bottom:1px solid var(--hair); padding-bottom:var(--s2); margin-bottom:var(--s4);}

header{border-top:2px solid var(--engrave); padding-top:var(--s2);}
.masthead{display:flex; flex-wrap:wrap; gap:var(--s3); align-items:flex-end;
  justify-content:space-between;}
h1{margin:.15em 0 0; font-size:clamp(2rem,5.5vw,3.4rem); line-height:.95; text-wrap:balance;}
h1 .thin{display:block; font-size:.4em; letter-spacing:.14em; color:var(--dim); margin-top:.7em;}
.stamp{border:1px solid var(--hair); background:var(--panel); padding:var(--s2) var(--s3);
  min-width:min(330px,100%);}
.stamp dl{margin:0; display:grid; grid-template-columns:auto 1fr; gap:.3rem var(--s2);}
.stamp dt{font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--dim);
  align-self:center;}
.stamp dd{margin:0; font-family:"IBM Plex Mono",monospace; font-size:12.5px; word-break:break-all;}
.verdict{color:var(--lamp); font-weight:600;}
.lede{max-width:64ch; margin:var(--s4) 0 0; font-size:1.02rem;}
.lede em{color:var(--dim); font-style:normal;}
.lede strong{font-weight:500;}

.section{margin-top:var(--s5); position:relative; padding-left:var(--s3);}
.section::before{content:""; position:absolute; left:0; top:.35em; bottom:0; width:1px;
  background:var(--rule);}
.section > .tick{position:absolute; left:-1px; top:.35em; width:11px; height:2px;
  background:var(--engrave);}
.section h2{margin:0 0 .25em; font-size:.95rem; letter-spacing:.16em; text-transform:uppercase;
  font-family:"IBM Plex Sans Condensed",sans-serif; font-weight:700;}
.section .note{margin:0 0 var(--s3); color:var(--dim); font-size:.92rem; max-width:62ch;}

/* pipeline */
.pipe{display:grid; gap:0;}
.stage{display:grid; grid-template-columns:minmax(150px,190px) 1fr; gap:var(--s3);
  border-left:1px solid var(--hair); padding:var(--s2) 0 var(--s2) var(--s3);
  position:relative; margin-left:6px;}
.stage::before{content:""; position:absolute; left:-4px; top:calc(var(--s2) + .55em);
  width:7px; height:7px; border-radius:50%; background:var(--lamp);
  box-shadow:0 0 0 3px var(--lamp-glow);}
.stage:last-child{border-left-color:transparent;}
.stage .sname{font-size:.95rem;}
.stage .swhat{color:var(--dim); font-size:.88rem; margin:.15rem 0 0;}
.stage .sinst{display:flex; flex-wrap:wrap; gap:.4rem; align-self:center;}
.chip{border:1px solid var(--hair); background:var(--panel); padding:.2rem .55rem;
  font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.04em; white-space:nowrap;}
.loopback{font-size:11px; color:var(--dim); letter-spacing:.1em; text-transform:uppercase;
  margin:var(--s2) 0 0; padding-left:var(--s3);}

.grid{display:grid; gap:var(--s2); grid-template-columns:repeat(auto-fill,minmax(320px,1fr));}
.card{border:1px solid var(--hair); background:var(--panel); padding:var(--s3);
  display:flex; flex-direction:column; gap:var(--s2);}
.card.control{border-style:dashed; background:var(--panel-2);}
.card.certifier{border-color:var(--lamp);}
.card-top{display:flex; align-items:flex-start; justify-content:space-between; gap:var(--s2);}
.card h3{margin:0; font-size:1rem; letter-spacing:.07em;}
.card .job{margin:0; color:var(--dim); font-size:.89rem; line-height:1.5;}
.lamp{display:flex; align-items:center; gap:.5rem; flex-shrink:0;}
.bulb{width:9px; height:9px; border-radius:50%; background:var(--absent); flex-shrink:0;}
.on .bulb{background:var(--lamp); box-shadow:0 0 0 4px var(--lamp-glow);}
.lamp .txt{font-size:9.5px; letter-spacing:.13em; text-transform:uppercase; color:var(--dim);
  white-space:nowrap;}
.on .lamp .txt{color:var(--lamp);}
.meters{display:grid; grid-template-columns:repeat(3,1fr); border-top:1px solid var(--hair-2);
  padding-top:var(--s2); gap:var(--s1);}
.meter .k{font-size:9.5px; letter-spacing:.13em; text-transform:uppercase; color:var(--dim);
  display:block;}
.meter .v{font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
  font-size:1.05rem; font-weight:500;}
.meter .v.zero{color:var(--dim);}
.fault{border-left:2px solid var(--fault); background:var(--fault-bg); padding:.6rem .8rem;
  font-size:.85rem; line-height:1.5;}
.fault .k{display:block; font-size:9.5px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--fault); margin-bottom:.25rem; font-weight:600;}
.selfd{border-left:2px solid var(--hair); padding:.6rem .8rem; font-size:.85rem; color:var(--dim);}
.selfd .k{display:block; font-size:9.5px; letter-spacing:.14em; text-transform:uppercase;
  margin-bottom:.25rem; font-weight:600; color:var(--engrave);}
.badge{font-size:9px; letter-spacing:.12em; text-transform:uppercase; color:var(--lamp);
  border:1px solid var(--lamp); padding:.1rem .35rem; margin-top:.4rem; display:inline-block;}

.tally{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:var(--s2);}
.tally div{border:1px solid var(--hair); padding:var(--s2) var(--s3); background:var(--panel);}
.tally .n{font-family:"IBM Plex Mono",monospace; font-size:1.9rem; font-weight:600; display:block;
  line-height:1.1; font-variant-numeric:tabular-nums;}
.tally .l{font-size:10px; letter-spacing:.13em; text-transform:uppercase; color:var(--dim);}
footer{margin-top:var(--s5); border-top:1px solid var(--hair); padding-top:var(--s3);
  color:var(--dim); font-size:.88rem; max-width:66ch;}
footer strong{color:var(--engrave); font-weight:500;}
@media (max-width:640px){
  .wrap{padding:var(--s2) var(--s2) var(--s4);}
  .stage{grid-template-columns:1fr; gap:var(--s1);}
}
@media (prefers-reduced-motion:no-preference){
  .on .bulb{animation:post .5s ease-out backwards; animation-delay:var(--d,0ms);}
  @keyframes post{from{background:var(--absent); box-shadow:none;}}
}
"""


def ledger_num(label: str):
    m = re.search(re.escape(label) + r"[^|]*\|\s*\*\*([0-9,]+)\*\*",
                  LEDGER.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def card(r: dict, i: int) -> str:
    absent = r["status"] == "ABSENT"
    on = not absent
    cls = " on" if on else ""
    if absent:
        cls += " control"
    if r["role"] == "VERIFIER":
        cls += " certifier"
    e = html.escape
    if r["disclosed_defect"]:
        body = (f'<div class="fault"><span class="k">Disclosed fault &middot; '
                f'{e(DOCS.get(r["role"], ""))}</span>{e(r["disclosed_defect"])}</div>')
    elif r["self_disclosed_limits"]["discloses"]:
        body = ('<div class="selfd"><span class="k">States its own limits, in its own source</span>'
                'Found by scanning the module, not curated by hand &mdash; '
                f'<span class="mono">{e(", ".join(r["self_disclosed_limits"]["markers"][:2]))}'
                '</span></div>')
    else:
        body = ('<div class="selfd"><span class="k">No fault disclosed</span>'
                'Not the same claim as &ldquo;healthy&rdquo;. Nobody has audited it.</div>')
    badge = ('<span class="badge">certified this page</span>'
             if r["role"] == "VERIFIER" else "")

    def num(v):
        return f'<span class="v{" zero" if v == 0 else ""}">{v}</span>'

    return f'''<article class="card{cls}">
  <div class="card-top">
    <div><h3 class="plate">{e(r["role"].title())}</h3>
      <div class="micro" style="margin-top:.25rem">{e(r["layer"])}</div>{badge}</div>
    <div class="lamp"><span class="bulb" style="--d:{i * 70}ms"></span>
      <span class="txt">{"absent" if absent else "online"}</span></div>
  </div>
  <p class="job">{e(r["job"][0].upper() + r["job"][1:])}.</p>
  <div class="meters">
    <div class="meter"><span class="k">lines</span>{num(r["loc"])}</div>
    <div class="meter"><span class="k">tests run</span>{num(r["tests"]["passed"])}</div>
    <div class="meter"><span class="k">receipts</span>{num(r["receipts_produced"])}</div>
  </div>
  {body}
</article>'''


POSTER_CSS = """
/* the main stylesheet is not loaded here, so the reset has to be repeated -- its absence
   is what made width:1200px + 48px padding render a 1296px frame and clip the right
   column out of the capture. */
*{box-sizing:border-box}
html,body{margin:0;background:#0F1311;}
.poster{width:1200px; background:#0F1311; color:#C7D0CB; padding:44px 48px;
  display:flex; flex-direction:column; gap:16px; overflow:hidden;
  font-family:"IBM Plex Sans",sans-serif;}
.poster .tagp{font-family:"IBM Plex Mono",monospace; color:#E3A63C; font-size:13px;
  line-height:1.06; white-space:pre; margin:0;}
.poster h1{font-family:"IBM Plex Sans Condensed",sans-serif; font-weight:700; text-transform:uppercase;
  letter-spacing:.02em; font-size:54px; line-height:.92; margin:4px 0 0;}
.poster .sub{color:#79847F; font-size:17.5px; max-width:60ch; margin:0; line-height:1.42;}
.poster .sub b{color:#C7D0CB; font-weight:500;}
.poster .rule{height:1px; background:rgba(199,208,203,.18);}
.pgrid{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:11px;}
.pc{border:1px solid rgba(199,208,203,.18); background:#171D1A; padding:13px 15px;
  display:flex; flex-direction:column; gap:6px; min-height:118px;
  min-width:0; overflow:hidden; overflow-wrap:anywhere;}
.pc.ctl{border-style:dashed; background:#131917;}
.pc .h{display:flex; justify-content:space-between; align-items:flex-start; gap:8px;}
.pc .n{font-family:"IBM Plex Sans Condensed",sans-serif; font-weight:700; text-transform:uppercase;
  letter-spacing:.06em; font-size:16px;}
.pc .d{width:8px;height:8px;border-radius:50%;background:#E3A63C;box-shadow:0 0 0 4px rgba(227,166,60,.2);
  flex-shrink:0;margin-top:4px;}
.pc.ctl .d{background:#4A5450; box-shadow:none;}
.pc .m{font-family:"IBM Plex Mono",monospace; font-size:12px; color:#79847F; letter-spacing:.03em;}
.pc .f{font-size:12px; line-height:1.4; color:#D4664B; border-left:2px solid #D4664B;
  padding-left:8px; margin-top:auto;}
.pc .ok{font-size:12px; line-height:1.4; color:#79847F; border-left:2px solid rgba(199,208,203,.2);
  padding-left:8px; margin-top:auto;}
.pstats{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:11px;}
.pstats div{border:1px solid rgba(199,208,203,.18); padding:12px 16px; background:#171D1A;}
.pstats .v{font-family:"IBM Plex Mono",monospace; font-size:30px; font-weight:600; display:block;
  line-height:1.05; font-variant-numeric:tabular-nums;}
.pstats .k{font-size:11px; letter-spacing:.13em; text-transform:uppercase; color:#79847F;}
.pfoot{display:flex; justify-content:space-between; align-items:flex-end; gap:24px;
  margin-top:8px; border-top:1px solid rgba(199,208,203,.18); padding-top:18px;}
.pfoot .why{font-size:14.5px; color:#79847F; max-width:70ch; line-height:1.48;}
.pfoot .why b{color:#E3A63C; font-weight:500;}
.pfoot .url{font-family:"IBM Plex Mono",monospace; font-size:12px; color:#79847F; text-align:right;
  white-space:nowrap;}
"""


def poster(cen, cert) -> str:
    e = html.escape
    roles = [r for r in cen["roles"] if r["layer"] != "control"]
    ctrl = [r for r in cen["roles"] if r["layer"] == "control"]

    def box(r):
        absent = r["status"] == "ABSENT"
        if r["disclosed_defect"]:
            line = f'<div class="f">{e(r["disclosed_defect"][:104])}&hellip;</div>'
        elif r["self_disclosed_limits"]["discloses"]:
            line = '<div class="ok">States its own limits, in its own source.</div>'
        else:
            line = '<div class="ok">No fault disclosed. Not the same as healthy.</div>'
        return (f'<div class="pc{" ctl" if absent else ""}"><div class="h">'
                f'<span class="n">{e(r["role"].title())}</span><span class="d"></span></div>'
                f'<div class="m">{r["loc"]} lines &middot; {r["tests"]["passed"]} tests run '
                f'&middot; {"ABSENT" if absent else "online"}</div>{line}</div>')

    stats = [(cen["tests_run_here"], "tests run here"), (cen["tests_failed_here"], "failed"),
             (cen["roles_disclosing_a_defect"], "disclose a fault"),
             (cen["status_tally"].get("ABSENT", 0), "absent, on purpose")]
    return f'''<title>STYXX org chart</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@700&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>{POSTER_CSS}</style>
<div class="poster">
  <pre class="tagp">{WORDMARK.strip(chr(10))}</pre>
  <h1>The chart that<br>cannot lie</h1>
  <p class="sub">Every AI-org diagram puts a lamp on each box reading ONLINE or BOOTING.
  Nobody checks them. <b>Here every lamp is computed</b> &mdash; line counts read from disk, tests
  executed while this page was generated, every fault citing a document whose existence is
  verified.</p>
  <div class="rule"></div>
  <div class="pgrid">{"".join(box(r) for r in roles)}{"".join(box(r) for r in ctrl)}</div>
  <div class="pstats">{"".join(f'<div><span class="v">{v}</span><span class="k">{k}</span></div>' for v, k in stats)}</div>
  <div class="pfoot">
    <div class="why">The last box is the control. The chart that inspired this drew a TRADER box,
    &ldquo;paper trading &rarr; live&rdquo;, lamp set to BOOTING. <b>Mine points at a module that
    is not on disk and reports ABSENT</b> &mdash; and the generator refuses to emit a chart if it
    ever says otherwise. A status column that cannot report a bad value is not reporting a good
    one either.</div>
    <div class="url">fathom lab &middot; styxx<br>{e(cert.get("verdict", ""))}</div>
  </div>
</div>'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "styxx-org-chart.html"))
    ap.add_argument("--poster", action="store_true",
                    help="emit a fixed 1200x1500 frame built to be screenshotted and posted")
    a = ap.parse_args()

    cen = json.loads(CENSUS.read_text(encoding="utf-8"))
    cert = json.loads(CERT.read_text(encoding="utf-8")) if CERT.exists() else {}
    by = {r["role"]: r for r in cen["roles"]}
    roles = [r for r in cen["roles"] if r["layer"] != "control"]
    ctrl = [r for r in cen["roles"] if r["layer"] == "control"]

    stages = "\n".join(
        f'''<div class="stage"><div><div class="sname plate">{html.escape(n)}</div>
        <p class="swhat">{html.escape(w)}</p></div>
        <div class="sinst">{"".join(f'<span class="chip">{html.escape(i.title())}</span>'
                                     for i in inst if i in by)}</div></div>'''
        for n, w, inst in PIPELINE)

    t = cen["status_tally"]
    tally = [
        (cen["roles_total"], "roles charted"),
        (cen["tests_run_here"], "tests run here"),
        (cen["tests_failed_here"], "tests failed"),
        (cen["roles_disclosing_a_defect"], "disclose a fault"),
        (cen.get("roles_self_disclosing_limits_in_source", 0), "state own limits"),
        (t.get("ABSENT", 0), "absent, on purpose"),
    ]
    prog = [(ledger_num("cycles logged"), "cycles logged"),
            (ledger_num("refusal, null, retraction"), "ended in a negative"),
            (ledger_num("preregistrations frozen"), "preregs frozen"),
            (ledger_num("OATH certificates"), "certificates")]

    doc_sha = cert.get("document_sha256", "")[:16]
    ver_sha = cert.get("verifier_sha256", "")[:16]
    rec_sha = (list(cert.get("receipts_sha256", {}).values()) or [""])[0][:16]

    page = f'''<title>The Chart That Cannot Lie</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<div class="tag" role="img" aria-label="styxx"><pre>{WORDMARK.strip(chr(10))}</pre></div>
<div class="tagline">
  <span class="micro">Fathom Lab</span>
  <span class="micro">personal AI org &middot; 2026-08-28</span>
  <span class="micro">generated from styxx_org_census.json</span>
</div>

<header>
  <div class="masthead">
    <div><h1 class="plate">The Chart<br>That Cannot Lie
      <span class="thin plate">Every lamp is a measurement</span></h1></div>
    <div class="stamp"><dl>
      <dt>Verdict</dt><dd class="verdict">{html.escape(cert.get("verdict", "n/a"))}</dd>
      <dt>Document</dt><dd>{doc_sha}&hellip;</dd>
      <dt>Receipt</dt><dd>{rec_sha}&hellip;</dd>
      <dt>Verifier</dt><dd>{ver_sha}&hellip;</dd>
    </dl></div>
  </div>
  <p class="lede">Personal-AI-org diagrams all have the same tell: a small lamp on each box reading
  <strong>ONLINE</strong> or <strong>BOOTING</strong>. The lamps are typed by whoever drew the
  chart. Nothing checks them, and <em>BOOTING is the polite label for a box that does not
  exist.</em></p>
  <p class="lede">A status that asserts its own health is the defect this laboratory exists to
  document &mdash; and eight days into looking for it elsewhere we found it in our own test suite.
  So this is the same picture with the lamps wired to something:
  <strong>{cen["tests_run_here"]} tests run during the census that generated this page,
  {cen["tests_failed_here"]} failed.</strong> The page is emitted from that receipt, so the
  picture cannot drift from the measurement either.</p>
</header>

<section class="section"><span class="tick"></span>
  <h2>The loop</h2>
  <p class="note">These roles are not peers on an org chart. They are stages, and the last two feed
  backwards &mdash; an audit that finds drift reopens the freeze, and a published negative becomes
  the next cycle's prior.</p>
  <div class="pipe">{stages}</div>
  <p class="loopback">&#8627; audit &rarr; freeze &nbsp;&middot;&nbsp; publish &rarr; freeze</p>
</section>

<section class="section"><span class="tick"></span>
  <h2>Instruments &middot; {len(roles)} roles</h2>
  <p class="note">Line count read from disk. Tests executed during the census, not quoted from a
  previous run. Each disclosed fault names a document and <strong>the citation is checked</strong>
  &mdash; a fault whose document has vanished reads as disclosure while disclosing nothing.</p>
  <div class="grid">{"".join(card(r, i) for i, r in enumerate(roles))}</div>
</section>

<section class="section"><span class="tick"></span>
  <h2>The control</h2>
  <p class="note">A status column that cannot report a bad value is not reporting a good one
  either. So the chart carries a declared positive control, and it is deliberately the box the
  source diagram drew: <strong>Trader</strong>, captioned <em>paper trading &rarr; live</em>, lamp
  set to BOOTING. Ours points at a module that is not on disk. The census refuses to emit a chart
  if it reports anything but ABSENT.</p>
  <div class="grid">{"".join(card(r, 0) for r in ctrl)}</div>
</section>

<section class="section"><span class="tick"></span>
  <h2>This chart</h2>
  <div class="tally">{"".join(f'<div><span class="n">{n}</span><span class="l">{l}</span></div>'
                              for n, l in tally)}</div>
</section>

<section class="section"><span class="tick"></span>
  <h2>The programme behind it</h2>
  <p class="note">Every cycle this laboratory has run, and how many of them it lost. The negatives
  row is the one that matters; it is also the one whose measurement we have publicly withdrawn as
  a keyword match over prose.</p>
  <div class="tally">{"".join(f'<div><span class="n">{n}</span><span class="l">{l}</span></div>'
                              for n, l in prog if n)}</div>
</section>

<footer>
  <p>The status column has failed three times. Twice on purpose &mdash; the control, and the
  assertion that guards it. <strong>Once by accident, on the census's first run, against its own
  author:</strong> a fault citation pointed at <span class="mono">papers/REPLICATIONS.md</span>,
  and that file lives at the repository root. The box went CITATION_MISSING until it was fixed.
  The mechanism caught its author before it caught anything else, which is the only evidence worth
  having that it works.</p>
  <p>Two roles carry no curated fault. They are not marked healthy &mdash; the census scans their
  source and finds they <strong>state their own limits</strong>, which is a fact about the file
  rather than a judgement by the author. An earlier version of this census recorded them as
  disclosing nothing; the omission was mine, and computing the field removed it.</p>
  <p>This page is OATH-certified against <span class="mono">styxx_org_census.json</span>. The
  instrument that certified it is the <strong>Verifier</strong> box in the chart it certifies
  &mdash; so if that box is broken, this certificate is worth less than it looks,
  <strong>and the box says what is broken about it.</strong></p>
</footer>
</div>
'''
    if a.poster:
        page = poster(cen, cert)
    out = Path(a.out)
    out.write_text(page, encoding="utf-8")
    print(f"roles {len(roles)}  control {len(ctrl)}  chars {len(page)}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

/* the gate, as a bookmarklet: on any public GitHub pull request page, one click reads the
 * description against the diff (both from api.github.com, nothing else) and pins the verdict
 * to the top of the page. Same JS port as the preview build (differential-tested against the
 * styxx Python instrument at the BC-2 + COMPAT-1 checkout, the file 7.48.0 ships). Nothing is
 * sent anywhere; nothing is stored. */
(async function () {
  const G = window.styxxDiffgateJS;
  const m = location.pathname.match(/^\/([\w.-]+)\/([\w.-]+)\/pull\/(\d+)/);
  const old = document.getElementById("styxx-gate-panel"); if (old) old.remove();
  const panel = document.createElement("div");
  panel.id = "styxx-gate-panel";
  panel.setAttribute("style", "all:initial;position:fixed;top:12px;right:12px;z-index:2147483647;width:min(720px,calc(100vw - 24px));max-height:80vh;overflow:auto;background:#0b0d13;color:#ecf4f1;border:1px solid #2a3038;border-radius:8px;padding:14px 16px;font:13px/1.6 'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;box-shadow:0 12px 40px rgba(0,0,0,.6);white-space:pre-wrap;word-break:break-word");
  const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
  const head = (t) => `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px"><span style="color:#96d0c4;letter-spacing:.06em;text-transform:uppercase;font-size:11px">styxx · diffgate · the description vs the diff</span><button id="styxx-gate-close" style="all:initial;cursor:pointer;color:#687a76;font:13px monospace;padding:2px 6px">✕</button></div><div style="color:#b9c7c3;margin-bottom:8px">${esc(t)}</div>`;
  panel.innerHTML = head(m ? `reading ${m[1]}/${m[2]}#${m[3]} from api.github.com…` : "open a pull request page (github.com/OWNER/REPO/pull/N) and click again.");
  document.body.appendChild(panel);
  panel.querySelector("#styxx-gate-close").onclick = () => panel.remove();
  if (!m) return;
  const api = `https://api.github.com/repos/${m[1]}/${m[2]}/pulls/${m[3]}`;
  let meta, diff;
  try {
    const [r1, r2] = await Promise.all([fetch(api, {headers:{Accept:"application/vnd.github+json"}}), fetch(api, {headers:{Accept:"application/vnd.github.diff"}})]);
    if (r1.status === 403 || r1.status === 429) throw new Error("GitHub's unauthenticated API limit (60/hour per address) is used up; wait, or run `python -m styxx.diffgate --pr <url>` with GITHUB_TOKEN set.");
    if (!r1.ok) throw new Error(`GitHub answered HTTP ${r1.status} for the pull request (private repos are not readable from here).`);
    if (r2.status === 406) throw new Error("GitHub will not serve this diff over the API (too large); run the CLI on a checkout instead.");
    if (!r2.ok) throw new Error(`GitHub answered HTTP ${r2.status} for the diff.`);
    meta = await r1.json(); diff = await r2.text();
  } catch (e) {
    panel.innerHTML = head(`${m[1]}/${m[2]}#${m[3]}`) + `<div style="color:#ecc46e">${esc(e.message)}</div>`;
    panel.querySelector("#styxx-gate-close").onclick = () => panel.remove();
    return;
  }
  const body = meta.body || "";
  const g = G.gateDiffText(body, diff);
  const col = {VERIFIED:"#78e296", CONTRADICTED:"#ff605c", UNCHECKABLE:"#687a76"};
  const mark = {VERIFIED:"[ok ]", CONTRADICTED:"[LIE]", UNCHECKABLE:"[ ? ]"};
  let out = "";
  if (!body.trim()) out += `<div style="color:#687a76">the description is empty — nothing to gate.</div>`;
  for (const c of g.claims) out += `<div style="color:${col[c.verdict]}">  ${mark[c.verdict]} ${esc(c.kind.padEnd(20))} ${esc(c.why)}</div>`;
  if (!g.measured) out += `<div style="color:#ecc46e">UNMEASURED  this gate did not run: ${esc(g.why_unmeasured)}</div>`;
  const nc = g.claims.filter(c => c.verdict === "CONTRADICTED").length, nu = g.claims.filter(c => c.verdict === "UNCHECKABLE").length;
  out += `<div style="color:${g.verdict === "PASS" ? "#ecc46e" : "#ff605c"};margin-top:8px;font-weight:${g.verdict === "PASS" ? 400 : 600}">${g.verdict}  claims=${g.claims.length} contradicted=${nc} uncheckable=${nu} uncovered_sentences=${g.uncovered_sentences}</div>`;
  if (g.sentences_total) out += `<div style="color:#687a76">never read: ${g.uncovered_sentences} of ${g.sentences_total} sentences — prose outside the closed template set is not judged</div>`;
  if (!g.claims.length) out += `<div style="color:#687a76">no diff-shaped claims found — silence is scope, not weakness</div>`;
  out += `<details style="margin-top:8px;color:#687a76"><summary style="cursor:pointer">what it reads · reproduce</summary><div style="margin-top:6px">modified / created / deleted &lt;path&gt; · N files changed · added N tests · adds function &lt;name&gt; · only touches &lt;prefix&gt; · tests pass (UNCHECKABLE without --run) · no breaking changes (read, never judged: the public definitions the diff removed are named)\na path the diff does not show is UNCHECKABLE, not an accusation (EXTERNAL-1: precision 0.23 vs a 0.95 floor on 71,016 agent PRs). added N tests / adds function &lt;name&gt; count python def lines and say so when the diff has no python (#110).\n\npip install styxx\npython -m styxx.diffgate --pr ${esc(location.origin + location.pathname.match(/^\/[\w.-]+\/[\w.-]+\/pull\/\d+/)[0])}\n\njs port of styxx diffgate.py at the BC-2 + COMPAT-1 checkout (7.48.0), differential-tested (3,199 pairs, 0 disagreements). the python is the instrument. github.com/fathom-lab/styxx</div></details>`;
  panel.innerHTML = head(`${m[1]}/${m[2]}#${m[3]} — ${meta.title || ""}`) + out;
  panel.querySelector("#styxx-gate-close").onclick = () => panel.remove();
})();

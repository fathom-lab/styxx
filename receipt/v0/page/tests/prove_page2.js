"use strict";
/*
 * Proves, in Node, that the checker embedded in the BUILT page is the checker, and that the page's
 * own code gives every vector's expected answer.
 *
 *   node tests/prove_page.js --page dist/receipt_check_v0.html --checker src/check_v0.js
 *        --vectors src/receipt_v0_vectors.json.ref --expected examples/expected_cli_X.json
 *        --utc STAMP [--lab-runner runs/lab_runner_X.jsonl] [--concurrency 8]
 *
 * Every vector runs through the page's own runJob (hash check, worker source from the page's
 * buildWorkerSource, one fresh worker per run, the page's time limit, the page's acceptResult),
 * never through the command line. Each accepted result is also drawn with the page's
 * renderOutcome into a stand-in document that refuses innerHTML and every attribute or property
 * outside the page's rule. Writes runs/page_vectors_<utc>.jsonl and .summary.json (never
 * overwrites). Lab-written; agreement it reports is agreement inside one lab.
 *
 * prove_page2.js (fixer, 2026-11-13): the same run, with the drawing check following the repaired
 * page: values are drawn through P.forDisplay (direction controls become markers), at most
 * P.ROWS_DRAWN span cards and the first P.PRE_CHARS receipt characters are drawn. It also counts
 * the vectors whose drawn text differs from the raw text because of direction controls.
 */
const fs = require("node:fs");
const path = require("node:path");
const H = require("./page_harness.js");

function args() {
  const a = { concurrency: 8 };
  const v = process.argv.slice(2);
  for (let i = 0; i < v.length; i += 2) a[v[i].replace(/^--/, "").replace(/-/g, "_")] = v[i + 1];
  for (const k of ["page", "checker", "vectors", "expected", "utc"]) if (!a[k]) throw new Error("missing --" + k);
  a.concurrency = Number(a.concurrency);
  return a;
}

const b64 = (x) => (x === null || x === undefined ? null : new Uint8Array(Buffer.from(x, "base64")));

function jobFor(v) {
  const i = v.inputs;
  if (v.op === "check") return { op: "check", document: b64(i.document_b64), manifest: b64(i.manifest_b64), name: i.name, commit: i.commit };
  if (v.op === "recheck") return { op: "recheck", receipt: b64(i.receipt_b64), document: b64(i.document_b64), manifest: b64(i.manifest_b64) };
  if (v.op === "jcs") return { op: "jcs", value: b64(i.value_b64) };
  throw new Error("unknown op " + v.op);
}

function gotOf(outcome) {
  if (outcome.status !== "verdict") return { no_verdict: outcome.why };
  const r = outcome.result;
  if (r.refusal) return { refusal: r.refusal };
  if (r.op === "check") return { core_digest: r.core_digest };
  if (r.op === "recheck") return { answer: r.answer };
  return { canonical: r.canonical };
}

function sameAsExpected(expect, got) {
  const ek = Object.keys(expect);
  if (ek.length !== 1) return false;
  return Object.keys(got).length === 1 && got[ek[0]] === expect[ek[0]];
}

function renderCheck(P, outcome) {
  const fd = H.fakeDocument();
  const root = fd.root();
  P.renderOutcome(fd, root, outcome, { copy: () => {} });
  const texts = [];
  H.walk(root.__node ? root : root, (n) => { texts.push(n.__node._text); });
  const all = root.textContent;
  if (outcome.status !== "verdict") return all.indexOf("no verdict") >= 0 ? null : "no 'no verdict' line";
  const r = outcome.result;
  if (r.refusal) return all.indexOf("REFUSED " + r.refusal) >= 0 ? null : "refusal not shown";
  const D = P.forDisplay, PRE = P.PRE_CHARS, cut = (t) => D(t.length > PRE ? t.slice(0, PRE) : t);
  if (r.op === "recheck") return texts.indexOf(r.answer) >= 0 ? null : "answer not shown";
  if (r.op === "jcs") return texts.indexOf(cut(r.canonical)) >= 0 ? null : "canonical not shown";
  if (texts.indexOf(r.document_verdict) < 0) return "document verdict not shown";
  if (texts.indexOf(cut(r.receipt_text)) < 0) return "receipt text not shown (up to the page's limit)";
  if (texts.indexOf(r.receipt_sha256) < 0 || texts.indexOf(r.core_digest) < 0) return "hashes not shown";
  for (const row of r.rows.slice(0, P.ROWS_DRAWN)) {
    if (texts.indexOf(row.verdict) < 0) return "a row verdict not shown";
    if (row.claim !== null && texts.indexOf(D(row.claim)) < 0) return "a claim not shown as its own text node";
    if (texts.indexOf(D(row.detail_text)) < 0) return "a detail not shown";
  }
  if (texts.some((t) => t !== D(t))) return "a raw direction control reached the drawn text";
  return null;
}

async function pool(items, n, fn) {
  let next = 0;
  const out = new Array(items.length);
  async function lane() {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i);
    }
  }
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, lane));
  return out;
}

function writeOnce(p, data) {
  if (fs.existsSync(p)) throw new Error("refusing to overwrite " + p);
  fs.writeFileSync(p, data);
  if (!fs.readFileSync(p).equals(Buffer.from(data))) throw new Error("write did not land: " + p);
}

async function main() {
  const a = args();
  const t0 = performance.now();
  const page = H.loadPage(a.page);
  const P = page.P;
  const srcBytes = fs.readFileSync(a.checker);
  const checks = {};
  checks.checker_text_equals_source_bytes = Buffer.from(page.checkerText, "latin1").equals(srcBytes);
  checks.checker_sha256_source = H.sha256hex(srcBytes);
  checks.checker_sha256_embedded = H.sha256hex(Buffer.from(page.checkerText, "latin1"));
  checks.checker_sha256_page_core = P.sha256hex(P.utf8(page.checkerText));
  checks.checker_sha256_recorded_meta = page.recordedMeta;
  checks.checker_sha256_recorded_script = P.CHECKER_SHA256_AT_BUILD;
  checks.checker_hashes_all_equal = [checks.checker_sha256_embedded, checks.checker_sha256_page_core,
    checks.checker_sha256_recorded_meta, checks.checker_sha256_recorded_script].every((h) => h === checks.checker_sha256_source);
  // the page's own sha256 against Node's on assorted lengths
  let shaOk = true;
  for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 100003]) {
    const b = new Uint8Array(n); for (let i = 0; i < n; i++) b[i] = (i * 131 + 7) & 255;
    if (P.sha256hex(b) !== H.sha256hex(b)) shaOk = false;
  }
  checks.page_sha256_agrees_with_node_crypto = shaOk;
  // the Content-Security-Policy
  const dirs = {};
  for (const part of (page.csp || "").split(";")) {
    const t = part.trim().split(/\s+/);
    if (t[0]) dirs[t[0]] = t.slice(1);
  }
  const b64sha = (s) => "'sha256-" + require("node:crypto").createHash("sha256").update(Buffer.from(s, "latin1")).digest("base64") + "'";
  checks.csp = page.csp;
  checks.csp_directives = Object.keys(dirs);
  checks.csp_script_hash_is_page_script = JSON.stringify(dirs["script-src"]) === JSON.stringify([b64sha(page.mainText)]);
  checks.csp_style_hash_is_page_style = JSON.stringify(dirs["style-src"]) === JSON.stringify([b64sha(page.styleText)]);
  checks.csp_default_none = JSON.stringify(dirs["default-src"]) === JSON.stringify(["'none'"]);
  checks.csp_worker_blob_only = JSON.stringify(dirs["worker-src"]) === JSON.stringify(["blob:"]);
  checks.csp_no_connect_img_font_override = !("connect-src" in dirs || "img-src" in dirs || "font-src" in dirs || "media-src" in dirs || "frame-src" in dirs || "child-src" in dirs);
  // static scan of the page outside the embedded checker
  // code patterns in the page script; markup patterns in the HTML outside the scripts and style;
  // outside-resource patterns in the style
  const markup = page.html.replace(page.checkerText, "").replace(page.mainText, "").replace(page.styleText, "");
  const inScript = [/\.innerHTML/, /\.outerHTML/, /insertAdjacentHTML\s*\(/, /document\.write/, /\beval\s*\(/,
    /new Function/, /\bfetch\s*\(/, /new XMLHttpRequest/, /new WebSocket/, /new EventSource/, /\.sendBeacon/,
    /new Image/, /\.src\s*=/, /\.href\s*=/, /\.action\s*=/, /importScripts\s*\(/, /window\.open/, /\.submit\s*\(/,
    /location\s*=/, /https?:/i, /setAttribute\(\s*["'](?!(type|aria-pressed)["'])/];
  const inMarkup = [/\son[a-z]+\s*=/i, /<a\s/i, /<img/i, /<link/i, /<iframe/i, /<form/i, /<object/i, /<embed/i,
    /<base/i, /srcset/i, /https?:/i, /<script(?![^>]*id="(checker-src|page-main)")/i, /style\s*=/i];
  const inStyle = [/url\(/i, /@import/i, /https?:/i];
  checks.static_scan_hits = [].concat(
    inScript.filter((re) => re.test(page.mainText)).map((re) => "script " + re),
    inMarkup.filter((re) => re.test(markup)).map((re) => "markup " + re),
    inStyle.filter((re) => re.test(page.styleText)).map((re) => "style " + re));
  checks.static_scan_patterns = { script: inScript.map(String), markup: inMarkup.map(String), style: inStyle.map(String) };
  checks.page_sha256 = page.pageSha256;

  // ---- every vector through the page's own runJob
  const data = JSON.parse(fs.readFileSync(a.vectors, "utf8"));
  const log = { started: 0, exited: 0 };
  const results = await pool(data.vectors, a.concurrency, async (v) => {
    const job = jobFor(v);
    const outcome = await P.runJob(H.nodeEnv(log), page.checkerText, job, { timeoutMs: P.TIME_LIMIT_MS });
    const got = gotOf(outcome);
    let renderProblem;
    try { renderProblem = renderCheck(P, outcome); } catch (e) { renderProblem = "render threw: " + e.message; }
    const line = { id: v.id, source: v.source, op: v.op, expect: v.expect, got, match: sameAsExpected(v.expect, got),
      render_ok: renderProblem === null, ms: outcome.ms };
    if (renderProblem) line.render_problem = renderProblem;
    if (outcome.status === "verdict" && outcome.result.op === "check" && !outcome.result.refusal) {
      if (P.hasBidi(outcome.result.receipt_text)) line.drawn_with_bidi_markers = true;
      line.receipt_sha256 = outcome.result.receipt_sha256;
      line.document_verdict = outcome.result.document_verdict;
    }
    return line;
  });

  // ---- the three examples embedded in the page, against the recorded command line answers
  const expFile = JSON.parse(fs.readFileSync(a.expected, "utf8"));
  const exResults = [];
  for (const ex of P.EXAMPLES) {
    const cli = expFile.examples.find((e) => e.name === ex.name);
    const jobC = { op: "check", document: P.utf8(ex.document), manifest: P.utf8(ex.manifest), name: ex.name, commit: null };
    const oc = await P.runJob(H.nodeEnv(log), page.checkerText, jobC, {});
    const jobR = { op: "recheck", receipt: P.utf8(ex.receipt), document: P.utf8(ex.document), manifest: P.utf8(ex.manifest) };
    const or = await P.runJob(H.nodeEnv(log), page.checkerText, jobR, {});
    const r = oc.result || {};
    exResults.push({
      name: ex.name,
      page_check_status: oc.status,
      page_document_verdict: r.document_verdict,
      page_rows: (r.rows || []).map((x) => [x.verdict, x.reason]),
      cli_document_verdict: cli.document_verdict,
      cli_rows: cli.rows.map((x) => [x.verdict, x.reason]),
      page_receipt_plus_lf_equals_cli_stdout: oc.status === "verdict" && r.receipt_text + "\n" === cli.check.stdout,
      page_receipt_plus_lf_sha256_equals_recorded: oc.status === "verdict" &&
        P.sha256hex(P.utf8(r.receipt_text + "\n")) === ex.expected.check_stdout_sha256,
      page_recheck: or.status === "verdict" ? or.result.answer || ("REFUSED " + or.result.refusal) : "no verdict: " + or.why,
      cli_recheck: cli.recheck.stdout.trim(),
      render_ok: renderCheck(P, oc) === null && renderCheck(P, or) === null,
    });
  }

  // ---- optional: the lab's own vector runner on the source checker, line by line
  let labCompare = null;
  if (a.lab_runner) {
    const lab = new Map(fs.readFileSync(a.lab_runner, "utf8").trim().split("\n").map((l) => { const o = JSON.parse(l); return [o.id, o]; }));
    let same = 0; const differ = [];
    for (const r of results) {
      const l = lab.get(r.id);
      if (l && JSON.stringify(l.got) === JSON.stringify(r.got)) same += 1; else differ.push(r.id);
    }
    labCompare = { lab_runner_file: a.lab_runner, lines: lab.size, same_got: same, differ: differ.slice(0, 50), differ_count: differ.length };
  }

  const by = {};
  for (const r of results) {
    const k = r.op + "/" + r.source;
    by[k] = by[k] || { total: 0, match: 0, no_verdict: 0, render_ok: 0 };
    by[k].total += 1; if (r.match) by[k].match += 1; if (r.got.no_verdict) by[k].no_verdict += 1; if (r.render_ok) by[k].render_ok += 1;
  }
  const summary = {
    what: "every vector of the vector file run through the built page's own code (runJob, buildWorkerSource, acceptResult, renderOutcome) in Node worker_threads",
    status: "lab-written test of a lab-written checker for an unsigned draft (RECEIPT v0)",
    written_at_utc: a.utc,
    node: process.version,
    page: a.page, vectors: a.vectors, vectors_sha256: H.sha256hex(fs.readFileSync(a.vectors)), checker_source: a.checker,
    checks,
    vectors_total: results.length,
    matched: results.filter((r) => r.match).length,
    mismatched: results.filter((r) => !r.match).map((r) => ({ id: r.id, expect: r.expect, got: r.got })).slice(0, 50),
    no_verdict: results.filter((r) => r.got.no_verdict).length,
    render_ok: results.filter((r) => r.render_ok).length,
    render_problems: results.filter((r) => !r.render_ok).map((r) => ({ id: r.id, problem: r.render_problem })).slice(0, 50),
    drawn_with_bidi_markers: results.filter((r) => r.drawn_with_bidi_markers).length,
    by_op_and_source: by,
    workers_started: log.started,
    examples: exResults,
    lab_runner_compare: labCompare,
    seconds: Math.round((performance.now() - t0) / 100) / 10,
  };
  const outDir = a.out_dir || path.join(__dirname, "..", "runs");
  fs.mkdirSync(outDir, { recursive: true });
  const stem = path.join(outDir, "page_vectors2_" + a.utc.replace(/:/g, ""));
  writeOnce(stem + ".jsonl", results.map((r) => JSON.stringify(r)).join("\n") + "\n");
  writeOnce(stem + ".summary.json", JSON.stringify(summary, null, 1) + "\n");
  console.log(JSON.stringify({ matched: summary.matched, total: summary.vectors_total, no_verdict: summary.no_verdict,
    render_ok: summary.render_ok, drawn_with_bidi_markers: summary.drawn_with_bidi_markers, hashes_equal: checks.checker_hashes_all_equal, text_equals_source: checks.checker_text_equals_source_bytes,
    csp_script: checks.csp_script_hash_is_page_script, csp_style: checks.csp_style_hash_is_page_style,
    static_hits: checks.static_scan_hits, examples: exResults.map((e) => [e.name, e.page_document_verdict, e.page_receipt_plus_lf_equals_cli_stdout, e.page_recheck]),
    lab: labCompare && { same: labCompare.same_got, differ: labCompare.differ_count }, seconds: summary.seconds }, null, 1));
  console.log("wrote", stem + ".summary.json");
  setTimeout(() => process.exit(summary.matched === summary.vectors_total ? 0 : 1), 200);
}

main().catch((e) => { console.error(e && e.stack || e); process.exit(2); });

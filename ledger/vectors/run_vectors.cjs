#!/usr/bin/env node
/* run_vectors.cjs — run the ledger protocol's conformance vectors with verify.js (or another verifier that exports
   the same functions). Exit 0 when every case answers as vectors.json says.

     node run_vectors.cjs [--vectors vectors.json] [--verify ../verify.js] */
"use strict";
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const args = process.argv.slice(2);
let vectorsPath = path.join(__dirname, "vectors.json"), verifyPath = path.join(__dirname, "..", "verify.js");
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--vectors") vectorsPath = args[++i];
  else if (args[i] === "--verify") verifyPath = args[++i];
}
const V = require(path.resolve(verifyPath));
const doc = JSON.parse(fs.readFileSync(vectorsPath, "utf8"));
const base = fs.readFileSync(path.join(path.dirname(vectorsPath), doc.base));
const sha = b => crypto.createHash("sha256").update(b).digest("hex");

function splitLines(buf) {           // Buffer -> array of Buffers, split on 0x0a (the terminator kept out)
  const out = []; let start = 0;
  for (let i = 0; i < buf.length; i++) if (buf[i] === 0x0a) { out.push(buf.subarray(start, i)); start = i + 1; }
  out.push(buf.subarray(start));
  return out;
}
const canon = o => Buffer.from(V.dump(o), "utf8");

async function apply(baseBuf, t) {
  const op = t.op;
  if (op === "identity") return baseBuf;
  if (op === "crlf_first_line") { const i = baseBuf.indexOf(0x0a); return Buffer.concat([baseBuf.subarray(0, i), Buffer.from("\r\n"), baseBuf.subarray(i + 1)]); }
  if (op === "truncate_last") {
    let s = baseBuf; while (s.length && s[s.length - 1] === 0x0a) s = s.subarray(0, s.length - 1);
    const parts = splitLines(s); parts.pop();
    return Buffer.concat([Buffer.from(parts.map(p => p.toString("utf8")).join("\n"), "utf8"), Buffer.from("\n")]);
  }
  const lines = splitLines(baseBuf);
  let out = [];
  for (const ln of lines) {
    if (!ln.toString("utf8").trim()) { out.push(ln); continue; }
    const o = JSON.parse(ln.toString("utf8"));
    const s = o.seq;
    if (op === "set_field" && s === t.seq) { o[t.field] = t.value; out.push(canon(o)); }
    else if (op === "drop_seq" && s === t.seq) continue;
    else if (op === "dup_key" && s === t.seq) out.push(Buffer.concat([ln.subarray(0, ln.length - 1), Buffer.from(',"' + t.key + '":' + JSON.stringify(t.value) + "}", "utf8")]));
    else out.push(ln);
  }
  if (op === "set_field_rechain") {
    let prev = null; const res = [];
    for (const ln of out) {
      if (!ln.toString("utf8").trim()) { res.push(ln); continue; }
      const o = JSON.parse(ln.toString("utf8"));
      if (o.schema === V.LOG_SCHEMA) { prev = await V.headerDigest(o); res.push(ln); continue; }
      if (o.seq === t.seq) o[t.field] = t.value;
      if (o.seq >= t.seq) { o.prev = prev; delete o.entry_id; o.entry_id = await V.entryIdOf(V.coreOf(o)); res.push(canon(o)); }
      else res.push(ln);
      prev = o.entry_id;
    }
    out = res;
  }
  return Buffer.from(out.map(p => p.toString("utf8")).join("\n"), "utf8");
}

(async () => {
  if (sha(base) !== doc.base_sha256) { console.log("the base file is not the one the vectors were made over"); process.exit(2); }
  let failed = 0;
  for (const c of doc.cases) {
    const input = await apply(base, c.transform);
    if (sha(input) !== c.input_sha256) { console.log("FAIL " + c.name + ": the transform did not produce the vector's input bytes"); failed++; continue; }
    const r = await V.verifyBytes(new Uint8Array(input), { withheld: c.withheld || [] });
    const got = {
      ok: r.ok,
      problem_seqs: [...new Set(r.problems.filter(p => p.seq !== undefined && p.seq !== null).map(p => p.seq))].sort((a, b) => a - b),
      problems_without_seq: r.problems.filter(p => p.seq === undefined || p.seq === null).length,
      head: r.head, bridged: r.bridged
    };
    if (c.anchors) got.anchors_ok = c.anchors.map(t => { const a = V.parseAnchorText(t); return a ? V.checkAnchor(r.entries, a, c.withheld || []).ok : false; });
    const exp = c.expect;
    const same = c.transform.op === "dup_key" ? got.ok === exp.ok : Object.keys(exp).every(k => JSON.stringify(got[k]) === JSON.stringify(exp[k]));
    console.log((same ? "ok   " : "FAIL ") + c.name);
    if (!same) { failed++; console.log("   expected " + JSON.stringify(exp)); console.log("   got      " + JSON.stringify(got)); }
  }
  console.log(`${doc.cases.length - failed} of ${doc.cases.length} vectors answered as the file says`);
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error("error: " + (e && e.message ? e.message : e)); process.exit(2); });

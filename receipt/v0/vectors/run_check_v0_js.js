"use strict";
/*
 * RECEIPT v0 (DRAFT, unsigned): runs every vector of receipt_v0_vectors.json through check_v0.js
 * and prints one JSON line per vector, in file order:
 *
 *   {"id": ..., "got": {"core_digest"|"refusal"|"answer"|"canonical": ...},
 *    "receipt_b64": <base64 of the UTF-8 bytes of the canonical text of receipt(core), no verifier,
 *                    for a check that emitted a core>}
 *
 * A harness, not a checker: it reads the vector file with the host's JSON.parse (the vector file
 * is the lab's own, and its inputs are base64 or plain strings) and hands the checker bytes.
 * Lab-written; agreement it reports is agreement inside one lab.
 *
 *   node run_check_v0_js.js [VECTORS.json] [CHECKER.js]
 */
const fs = require("fs");
const path = require("path");

const vecPath = process.argv[2] || path.join(__dirname, "receipt_v0_vectors.json");
const checkerPath = process.argv[3] || path.join(__dirname, "..", "check_v0.js");
const C = require(path.resolve(checkerPath));
const data = JSON.parse(fs.readFileSync(vecPath, "utf8"));

const b = (x) => (x === null || x === undefined ? null : new Uint8Array(Buffer.from(x, "base64")));

function run(v) {
  const i = v.inputs;
  try {
    if (v.op === "check") {
      const core = C.check(b(i.document_b64), i.name, b(i.manifest_b64), i.commit);
      const rec = new TextEncoder().encode(C.jcs(C.receipt(core)));
      return { got: { core_digest: C.digest(core) }, receipt: Buffer.from(rec).toString("base64") };
    }
    if (v.op === "recheck") {
      return { got: { answer: C.recheck(b(i.receipt_b64), b(i.document_b64), b(i.manifest_b64)) } };
    }
    if (v.op === "jcs") {
      return { got: { canonical: C.jcs(C.parseJson(b(i.value_b64))) } };
    }
    return { got: { error: "unknown op " + v.op } };
  } catch (e) {
    if (e instanceof C.Refusal) return { got: { refusal: e.code } };
    return { got: { error: String(e && e.stack || e) } };
  }
}

const chunks = [];
for (const v of data.vectors) {
  const r = run(v);
  const line = { id: v.id, got: r.got };
  if (r.receipt !== undefined) line.receipt_b64 = r.receipt;
  chunks.push(JSON.stringify(line));
}
process.stdout.write(chunks.join("\n") + "\n");

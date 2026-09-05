"use strict";
/*
 * replay_js.js — hold styxx/_data/sworn_verify.js to the committed conformance set.
 *
 * SPEC: papers/sworn/SPEC_sworn_browser_verifier_v01_2026_09_05.md. In scope (B1): mode `inline`
 * with `requires` a subset of {manifest}. Everything else is SKIPPED and counted, with the
 * vector's own `requires` as the reason — never a judgement of the vector (B1).
 *
 * The bar (B3): every in-scope vector reproduces its `expect.core_sha256`, and the number of
 * in-scope vectors is 1689. The harness computes the digest from the verifier's own object and
 * never passes the expectation into it (B2).
 *
 *   node conformance/sworn/replay_js.js [--json OUT] [--quiet]
 *
 * Exit 0 when every in-scope vector passes, 1 otherwise.
 */
const fs = require("fs");
const path = require("path");

const HERE = __dirname;
const ROOT = path.resolve(HERE, "..", "..");
const api = require(path.join(ROOT, "styxx", "_data", "sworn_verify.js"));

const IN_SCOPE_MODES = new Set(["inline"]);
const IN_SCOPE_REQUIRES = new Set(["manifest"]);

function readJson(p) { return JSON.parse(fs.readFileSync(p, "utf8")); }

function b64ToBytes(s) {
  return new Uint8Array(Buffer.from(s, "base64"));
}

function main(argv) {
  const jsonOut = argv.includes("--json") ? argv[argv.indexOf("--json") + 1] : null;
  const quiet = argv.includes("--quiet");
  const index = readJson(path.join(HERE, "index.json"));
  const blobs = readJson(path.join(HERE, "blobs.json"));

  const perFamily = {};
  const failures = [];
  let ran = 0, passed = 0, skipped = 0;

  const families = Object.keys(index.families).sort();
  for (const fam of families) {
    const meta = index.families[fam];
    const raw = readJson(path.join(HERE, meta.file));
    const vectors = Array.isArray(raw) ? raw : raw.vectors;
    const rec = { ran: 0, passed: 0, failed: 0, skipped: 0, skip_reasons: {} };
    for (const v of vectors) {
      const requires = v.requires || [];
      const outOfScope = !IN_SCOPE_MODES.has(v.mode) ||
                         requires.some(r => !IN_SCOPE_REQUIRES.has(r));
      if (outOfScope) {
        rec.skipped++;
        skipped++;
        const why = v.mode + (requires.length ? " requires:" + requires.join("+") : "");
        rec.skip_reasons[why] = (rec.skip_reasons[why] || 0) + 1;
        continue;
      }
      rec.ran++;
      ran++;
      const inputs = v.inputs;
      const docBytes = b64ToBytes(blobs[inputs.document]);
      let manifestObj = null;
      if (inputs.manifest) {
        manifestObj = api.jsonPlainBytes(b64ToBytes(blobs[inputs.manifest]));
      }
      let got = null, err = null;
      try {
        const core = api.swornVerify(docBytes, manifestObj,
                                     { name: inputs.name, commit: inputs.commit });
        got = api.coreDigest(core);
      } catch (e) {
        err = String(e && e.message ? e.message : e);
      }
      if (err === null && got === v.expect.core_sha256) {
        rec.passed++;
        passed++;
      } else {
        rec.failed++;
        failures.push({ family: fam, id: v.id, sources: v.sources, rules: v.rules,
                        want: v.expect.core_sha256, got, error: err,
                        expect: v.expect });
      }
    }
    perFamily[fam] = rec;
  }

  const report = {
    schema: "styxx.sworn.browser-verifier/replay/v1",
    spec: "papers/sworn/SPEC_sworn_browser_verifier_v01_2026_09_05.md",
    set_sha256: index.set_sha256,
    in_scope: { modes: [...IN_SCOPE_MODES], requires_subset_of: [...IN_SCOPE_REQUIRES] },
    vectors_total: index.vector_count,
    ran, passed, failed: failures.length, skipped,
    families: perFamily,
    failures: failures.slice(0, 40),
  };
  if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(report, null, 1) + "\n", "utf8");
  if (!quiet) {
    console.log("set " + index.set_sha256.slice(0, 12) + " — " + index.vector_count + " vectors");
    console.log("in scope (mode inline, requires subset of {manifest}): " + ran +
                " ran, " + passed + " passed, " + failures.length + " failed; " +
                skipped + " skipped");
    for (const fam of families) {
      const r = perFamily[fam];
      if (r.ran === 0 && r.failed === 0) continue;
      console.log("  " + fam.padEnd(20) + " ran " + String(r.ran).padStart(5) +
                  "  passed " + String(r.passed).padStart(5) +
                  "  failed " + String(r.failed).padStart(4) +
                  "  skipped " + String(r.skipped).padStart(5));
    }
    for (const f of failures.slice(0, 12)) {
      console.log("  FAIL " + f.family + " " + f.id.slice(0, 12) +
                  (f.error ? ("  error: " + f.error) : ("  want " + f.want.slice(0, 12) +
                   " got " + String(f.got).slice(0, 12))) +
                  "  " + String(f.sources).slice(0, 110));
    }
  }
  return failures.length === 0 ? 0 : 1;
}

process.exit(main(process.argv.slice(2)));

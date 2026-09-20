// The pinned pairs, checked against what the port says:  node check_pairs.js
// bc1_pairs.json (the four BC-2 pairs, also checked on the Python side by tests/test_diffgate_bc1.py),
// compat_pairs.json (COMPAT-1, V14 and BC-2 edge cases, a Python repr() case, a CRLF diff) and
// bin1_pairs.json (BIN-1: binaries, a pure rename, a mode change, a quoted path; tests/test_diffgate_bin1.py) each
// carry an `expect` block written from the Python instrument's output. This is a smoke test for
// the port alone; the differential (py_side.py + js_side.js + differential.py) is the real check.
"use strict";
const fs = require("fs");
const path = require("path");
const { gateDiffText } = require("../diffgate.js");
let n = 0, bad = 0;
for (const name of ["bc1_pairs.json", "compat_pairs.json", "bin1_pairs.json", "compat2_pairs.json", "path1_pairs.json", "declare1_pairs.json"]) {
  const p = path.join(__dirname, name);
  if (!fs.existsSync(p)) continue;
  for (const pair of JSON.parse(fs.readFileSync(p, "utf8"))) {
    n++;
    const g = gateDiffText(pair.summary, pair.diff);
    const withWhy = pair.expect.claims.length && pair.expect.claims[0].length === 3;
    const got = g.claims.map(c => withWhy ? [c.kind, c.verdict, c.why] : [c.kind, c.verdict]);
    const ok = JSON.stringify(got) === JSON.stringify(pair.expect.claims)
      && g.verdict === pair.expect.verdict && g.uncovered_sentences === pair.expect.uncovered_sentences;
    if (!ok) { bad++; console.log(`DISAGREES ${pair.id}\n  expect ${JSON.stringify(pair.expect)}\n  got    ${JSON.stringify({verdict: g.verdict, claims: got, uncovered_sentences: g.uncovered_sentences})}`); }
  }
}
console.log(`${n} pinned pairs, ${bad} disagreement(s)`);
process.exit(bad ? 1 : 0);

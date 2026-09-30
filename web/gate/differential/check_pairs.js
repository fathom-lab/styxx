// The pinned pairs, checked against what the port says:  node check_pairs.js
// bc1_pairs.json (the four BC-2 pairs, also checked on the Python side by tests/test_diffgate_bc1.py),
// compat_pairs.json (COMPAT-1, V14 and BC-2 edge cases, a Python repr() case, a CRLF diff) and
// bin1_pairs.json (BIN-1: binaries, a pure rename, a mode change, a quoted path; tests/test_diffgate_bin1.py) each
// carry an `expect` block written from the Python instrument's output. This is a smoke test for
// the port alone; the differential (py_side.py + js_side.js + differential.py) is the real check.
// PATH-2a: path2a_pairs.json pins the overlay's decisions, and path2a_moves.json lists the pinned claims of the
// other files that the overlay moves; each listed claim must still read `from` in its file and is compared with `to`.
"use strict";
const fs = require("fs");
const path = require("path");
const { gateDiffText } = require("../diffgate.js");
const moves = JSON.parse(fs.readFileSync(path.join(__dirname, "path2a_moves.json"), "utf8")).moves;
let applied = 0;
function expected(name, pair) {
  const mine = moves.filter(m => m.file === name && m.id === pair.id);
  if (!mine.length) return pair.expect;
  const e = JSON.parse(JSON.stringify(pair.expect));
  for (const m of mine) {
    if (JSON.stringify(e.claims[m.claim]) !== JSON.stringify(m.from)) return null;
    e.claims[m.claim] = m.to;
    e.verdict = m.verdict;
    applied++;
  }
  return e;
}
let n = 0, bad = 0;
for (const name of ["bc1_pairs.json", "compat_pairs.json", "bin1_pairs.json", "compat2_pairs.json", "path1_pairs.json", "declare1_pairs.json", "path2a_pairs.json"]) {
  const p = path.join(__dirname, name);
  if (!fs.existsSync(p)) continue;
  for (const pair of JSON.parse(fs.readFileSync(p, "utf8"))) {
    n++;
    const g = gateDiffText(pair.summary, pair.diff);
    const expect = expected(name, pair);
    if (expect === null) { bad++; console.log(`STALE MOVE ${pair.id}: its pinned claim no longer reads the move's "from"`); continue; }
    const withWhy = expect.claims.length && expect.claims[0].length === 3;
    const got = g.claims.map(c => withWhy ? [c.kind, c.verdict, c.why] : [c.kind, c.verdict]);
    const ok = JSON.stringify(got) === JSON.stringify(expect.claims)
      && g.verdict === expect.verdict && g.uncovered_sentences === expect.uncovered_sentences;
    if (!ok) { bad++; console.log(`DISAGREES ${pair.id}\n  expect ${JSON.stringify(expect)}\n  got    ${JSON.stringify({verdict: g.verdict, claims: got, uncovered_sentences: g.uncovered_sentences})}`); }
  }
}
if (applied !== moves.length) { bad++; console.log(`path2a_moves.json lists ${moves.length} move(s) and ${applied} matched a pinned claim`); }
console.log(`${n} pinned pairs, ${bad} disagreement(s)`);
process.exit(bad ? 1 : 0);

// Run the JavaScript port over the same corpus:  node js_side.js  -> js_out.json
// Same records as py_side.py, minus unparsed_claims (not ported; see ../diffgate.js).
"use strict";
const fs = require("fs");
const path = require("path");
const { gateDiffText } = require("../diffgate.js");
const here = __dirname;
let items = [];
for (const name of ["corpus_real.json", "corpus_fuzz.json", "bc1_pairs.json", "compat_pairs.json", "bin1_pairs.json", "compat2_pairs.json", "path1_pairs.json"]) {
  const p = path.join(here, name);
  if (fs.existsSync(p)) items = items.concat(JSON.parse(fs.readFileSync(p, "utf8")));
}
if (!items.length) { console.error("no corpus: run build_corpus.py and/or fuzz_corpus.py first"); process.exit(1); }
const out = items.map(it => { const d = gateDiffText(it.summary, it.diff); delete d.unparsed_claims; return { id: it.id, ...d }; });
fs.writeFileSync(path.join(here, "js_out.json"), JSON.stringify(out));
console.log(`${out.length} pairs, ${out.reduce((n, d) => n + d.claims.length, 0)} claims -> js_out.json`);

// PATH-2a (NOTE_path2a_abstain_overlay_2026_09_30, NOTE_path2a_second_pass_2026_09_30 and
// NOTE_path2a_third_pass_2026_09_30, NOTE_path2a_fourth_pass_2026_09_30): the port's half of the committed PATH-2a checks, run by
// tests/test_diffgate_path2a*.py, which write IN (a JSON list of {id, summary, diff}) and read OUT.
//
//   node check_path2a.js --relation  REF IN OUT   main's port (REF, the reconstruction) against this port, both
//                                                  strict modes: every record may differ from main's only by an
//                                                  abstention in reach with the overlay's reason (the relation A),
//                                                  and each claim reads the same under --strict as without it;
//                                                  unparsed_claims and each claim's keys are compared too
//   node check_path2a.js --decisions REF IN OUT   per input, main's record and this port's record, for the
//                                                  Python-vs-port comparison (C)
//   node check_path2a.js --records   PORT IN OUT  per input, one port's record (strict off): the truth test runs the
//                                                  port's own counterfactual variants through it
//   node check_path2a.js --lockstep  IN OUT       the overlay's status-map builder against parseUnifiedDiff, and its
//                                                  counts of `def test_` sites against main's count, per line view
//   node check_path2a.js --tables    OUT          this engine's whitespace, trim and lowercase facts
//   node check_path2a.js --error-fallback REF IN OUT   as --decisions, with the overlay's decision function made to
//                                                  throw: every claim in reach must abstain with the error phrase
//   node check_path2a.js --timing    REF IN OUT   per input, milliseconds per gate call for main's port and this one
//   node check_path2a.js --overlay-timing REF IN OUT   per input, milliseconds the overlay alone takes on main's
//                                                  record (the least of three runs)
//
// --relation, --decisions and --lockstep take an optional last argument, the port to check (default ../diffgate.js);
// the tests pass a planted copy there to show the checks refuse it.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const DEFAULT_PORT = path.join(__dirname, "..", "diffgate.js");

function internals(file, extra = "") {
  // The port's top-level functions and constants, read the way a page reads the file: as one script.
  const src = fs.readFileSync(file, "utf8");
  const names = ["gateDiffText", "parseUnifiedDiff", "_norm", "_splitlines", "_p2aRegsRaw", "_p2aBuild", "_p2aViews", "_p2aPairing",
                 "_P2A_JS_SPACE", "_P2A_PY_SPACE", "_P2A_PY_BREAKS", "_P2A_DIVERGENT", "_P2A_HEADERS",
                 "_P2A_REACH_PAIRS", "_P2A_PHRASES", "_P2A_KIND_DEFECT", "P2A_DIRECTORY_BASENAME_ABSTAINS", "_P2A_OWN",
                 "_P2A_NEUTRAL_RANGES", "_p2aWordishUnit", "_p2aBadUnit", "_p2aAbstain", "_p2aFactsRaw"];
  return vm.runInNewContext(src + "\n" + extra + "\n;({" + names.join(", ") + "})", {}, { filename: file });
}

function record(gate, it, strict, keepUnparsed = false) {
  // unparsed_claims is kept where both sides are ports (--relation); against the Python it is dropped, since only
  // the Python runs claimdetect.
  try {
    const d = gate(it.summary, it.diff, { strict });
    if (!keepUnparsed) delete d.unparsed_claims;
    return JSON.parse(JSON.stringify(d));
  } catch (e) {
    return { error: (e && e.constructor && e.constructor.name) || "Error" };
  }
}

function relation(a, b, strict, reach, defects, phrases) {
  const bad = [];
  for (const k of Object.keys(a)) if (k !== "verdict" && k !== "claims" && JSON.stringify(a[k]) !== JSON.stringify(b[k])) bad.push("field " + k + " moved");
  if (JSON.stringify(Object.keys(a)) !== JSON.stringify(Object.keys(b))) bad.push("record keys differ");
  if (a.claims.length !== b.claims.length) return bad.concat(["claim count differs"]);
  a.claims.forEach((x, i) => {
    const y = b.claims[i];
    if (JSON.stringify(Object.keys(x)) !== JSON.stringify(Object.keys(y))) bad.push(`claim ${i}: keys differ`);
    if (JSON.stringify([x.kind, x.text, x.detail]) !== JSON.stringify([y.kind, y.text, y.detail])) bad.push(`claim ${i}: kind, text or detail moved`);
    if (x.verdict === y.verdict) { if (x.why !== y.why) bad.push(`claim ${i}: reason moved without a verdict move`); return; }
    if (!reach.has(x.kind + "|" + x.verdict) || y.verdict !== "UNCHECKABLE") { bad.push(`claim ${i}: not an abstention in reach`); return; }
    const allowed = new Set();
    for (const d of defects[x.kind]) for (const p of Object.values(phrases)) allowed.add(`${x.verdict} withheld by PATH-2a (${d}): ${p}. main's reading: ${x.why}`);
    if (!allowed.has(y.why)) bad.push(`claim ${i}: reason is not the overlay's form`);
    else if (y.why.split("): ").slice(1).join("): ").startsWith(phrases.error + ".")) bad.push(`claim ${i}: the overlay's error fallback fired`);
  });
  const want = (b.claims.some(c => c.verdict === "CONTRADICTED") || (strict && b.claims.some(c => c.verdict === "UNCHECKABLE"))) ? "FAIL" : "PASS";
  if (b.verdict !== want) bad.push("gate verdict is not main's formula over the claims");
  return bad;
}

function strictAlike(off, on) {
  // --strict may move the gate verdict and nothing else: every claim, and every other field, reads the same.
  const bad = [];
  for (const k of Object.keys(off)) if (k !== "verdict" && JSON.stringify(off[k]) !== JSON.stringify(on[k])) bad.push("under --strict, " + k + " differs");
  return bad;
}

function main(argv) {
  const mode = argv[0];
  if (mode === "--tables") {
    const ws = [], trim = [], lowerAscii = [], lowerLong = [];
    for (let cp = 0; cp <= 0x10ffff; cp++) {
      if (cp >= 0xd800 && cp <= 0xdfff) continue;
      const ch = String.fromCodePoint(cp);
      if (/\s/.test(ch)) ws.push(cp);
      if (ch.trim() === "") trim.push(cp);
      const low = ch.toLowerCase();
      if (cp >= 128 && [...low].some(x => x.codePointAt(0) < 128)) lowerAscii.push(cp);
      if ([...low].length > 1) lowerLong.push(cp);
    }
    const P = internals(DEFAULT_PORT);
    const cps = s => [...s].map(x => x.codePointAt(0));
    fs.writeFileSync(argv[1], JSON.stringify({
      node: process.version, unicode: process.versions.unicode, whitespace: ws, trim, lower_holds_ascii: lowerAscii,
      lower_longer: lowerLong, js_space: cps(P._P2A_JS_SPACE), py_space: cps(P._P2A_PY_SPACE),
      py_breaks: cps(P._P2A_PY_BREAKS), divergent: cps(P._P2A_DIVERGENT), headers: P._P2A_HEADERS,
      reach: P._P2A_REACH_PAIRS, phrases: P._P2A_PHRASES, kind_defect: P._P2A_KIND_DEFECT,
      directory_rule: P.P2A_DIRECTORY_BASENAME_ABSTAINS, own: P._P2A_OWN,
      // the summary's classes (NOTE_path2a_third_pass_2026_09_30): the units that are not wordish, and that are not bad
      neutral: P._P2A_NEUTRAL_RANGES.flatMap(([a, b]) => Array.from({ length: b - a + 1 }, (_, k) => a + k)),
      // pass 4 (NOTE_path2a_fourth_pass_2026_09_30, B-2): the neutral units this engine gives a case
      neutral_cased: P._P2A_NEUTRAL_RANGES.flatMap(([a, b]) => Array.from({ length: b - a + 1 }, (_, k) => a + k))
        .filter(u => { const ch = String.fromCharCode(u); return ch.toLowerCase() !== ch || ch.toUpperCase() !== ch; }),
      not_wordish: Array.from({ length: 0x10000 }, (_, u) => u).filter(u => !P._p2aWordishUnit(u)),
      not_bad: Array.from({ length: 0x10000 }, (_, u) => u).filter(u => !P._p2aBadUnit(u)),
      word_class: Array.from({ length: 0x10000 }, (_, u) => u).filter(u => /\w/.test(String.fromCharCode(u))),
    }));
    return 0;
  }
  if (mode === "--lockstep") {
    const P = internals(path.resolve(argv[3] || DEFAULT_PORT));
    const items = JSON.parse(fs.readFileSync(argv[1], "utf8"));
    let n = 0, skipped = 0;
    const bad = [], counts = {};
    for (const it of items) {
      let parsed;
      try { parsed = P.parseUnifiedDiff(it.diff || ""); } catch (e) { skipped++; continue; }
      const got = [...P._p2aBuild(P._p2aRegsRaw(P._splitlines(it.diff || "")), P._norm)];
      n++;
      if (JSON.stringify(got) !== JSON.stringify([...parsed.status])) bad.push(it.id);
      const pairing = P._p2aPairing(P._p2aViews(it.diff || "", P._splitlines(it.diff || "")));
      counts[it.id] = { main: (parsed.addedBlob.match(/^\s*def test_/gm) || []).length, views: pairing.map(x => x[0]) };
    }
    fs.writeFileSync(argv[2], JSON.stringify({ checked: n, main_raises: skipped, differ: bad, counts }));
    return 0;
  }
  if (mode === "--error-fallback") {
    const REF = require(path.resolve(argv[1]));
    const P = internals(DEFAULT_PORT, '_p2aDecide = function () { throw new Error("planted"); };');
    const items = JSON.parse(fs.readFileSync(argv[2], "utf8"));
    const out = items.map(it => ({ id: it.id, main: record(REF.gateDiffText, it, false), new: record(P.gateDiffText, it, false) }));
    fs.writeFileSync(argv[3], JSON.stringify(out));
    return 0;
  }
  if (mode === "--records") {
    const PORT = require(path.resolve(argv[1]));
    const items = JSON.parse(fs.readFileSync(argv[2], "utf8"));
    fs.writeFileSync(argv[3], JSON.stringify(items.map(it => ({ id: it.id, rec: record(PORT.gateDiffText, it, false) }))));
    return 0;
  }
  if (mode === "--timing") {
    const REF = require(path.resolve(argv[1]));
    const NEW = require(DEFAULT_PORT);
    const items = JSON.parse(fs.readFileSync(argv[2], "utf8"));
    const ms = gate => it => { const t0 = process.hrtime.bigint(); record(gate, it, false); return Number(process.hrtime.bigint() - t0) / 1e6; };
    fs.writeFileSync(argv[3], JSON.stringify(items.map(it => ({ id: it.id, main: ms(REF.gateDiffText)(it), new: ms(NEW.gateDiffText)(it) }))));
    return 0;
  }
  if (mode === "--overlay-timing") {
    const REF = require(path.resolve(argv[1]));
    const P = internals(DEFAULT_PORT);
    const items = JSON.parse(fs.readFileSync(argv[2], "utf8"));
    const out = items.map(it => {
      let best = Infinity;
      for (let k = 0; k < 3; k++) {
        const g = REF.gateDiffText(it.summary, it.diff);
        const t0 = process.hrtime.bigint();
        P._p2aAbstain(g, false, () => P._p2aFactsRaw(it.diff || "", it.summary));
        best = Math.min(best, Number(process.hrtime.bigint() - t0) / 1e6);
      }
      return { id: it.id, overlay: best };
    });
    fs.writeFileSync(argv[3], JSON.stringify(out));
    return 0;
  }
  if (mode === "--relation" || mode === "--decisions") {
    const port = path.resolve(argv[4] || DEFAULT_PORT);
    const REF = require(path.resolve(argv[1]));
    const NEW = require(port);
    const P = internals(port);
    const reach = new Set(P._P2A_REACH_PAIRS.map(p => p[0] + "|" + p[1]));
    const defects = { file_created: ["#97", "#121", "#97, #121"], file_deleted: ["#97", "#121", "#97, #121"],
                      file_touched: ["#97", "#121", "#97, #121"], files_changed_count: ["#121"], only_touches: ["#121"],
                      tests_added: ["#101"], symbol_added: ["#101"] };
    const items = JSON.parse(fs.readFileSync(argv[2], "utf8"));
    if (mode === "--decisions") {
      const out = items.map(it => ({ id: it.id, main: record(REF.gateDiffText, it, false), new: record(NEW.gateDiffText, it, false) }));
      fs.writeFileSync(argv[3], JSON.stringify(out));
      return 0;
    }
    const counts = { runs: 0, both_raise: 0, raise_differs: 0, broken: 0, decided: 0, abstained: 0 };
    const broken = [];
    for (const it of items) {
      const mine = {};
      for (const strict of [false, true]) {
        const a = record(REF.gateDiffText, it, strict, true), b = record(NEW.gateDiffText, it, strict, true);
        counts.runs++;
        if (a.error || b.error) {
          if (a.error && b.error && a.error === b.error) counts.both_raise++;
          else { counts.raise_differs++; broken.push([it.id, strict, "raise " + a.error + " / " + b.error]); }
          continue;
        }
        mine[strict] = b;
        const bad = relation(a, b, strict, reach, defects, P._P2A_PHRASES);
        if (bad.length) { counts.broken++; if (broken.length < 50) broken.push([it.id, strict, bad]); }
        if (!strict) a.claims.forEach((x, i) => {
          if (x.verdict === "VERIFIED" || x.verdict === "CONTRADICTED") { counts.decided++; if (b.claims[i].verdict === "UNCHECKABLE") counts.abstained++; }
        });
      }
      if (mine[false] && mine[true]) {
        const bad = strictAlike(mine[false], mine[true]);
        if (bad.length) { counts.broken++; if (broken.length < 50) broken.push([it.id, "strict", bad]); }
      }
    }
    fs.writeFileSync(argv[3], JSON.stringify({ counts, broken }));
    return 0;
  }
  console.error("usage: node check_path2a.js --relation REF IN OUT | --decisions REF IN OUT | --records PORT IN OUT | --lockstep IN OUT | --tables OUT | --error-fallback REF IN OUT | --timing REF IN OUT | --overlay-timing REF IN OUT");
  return 2;
}

process.exit(main(process.argv.slice(2)));

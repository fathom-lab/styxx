// PATH-2a (NOTE_path2a_abstain_overlay_2026_09_30, NOTE_path2a_second_pass_2026_09_30,
// NOTE_path2a_third_pass_2026_09_30, NOTE_path2a_fourth_pass_2026_09_30, NOTE_path2a_fifth_pass_2026_09_30,
// NOTE_path2a_sixth_pass_2026_09_30, NOTE_path2a_seventh_pass_2026_09_30, NOTE_path2a_ninth_pass_2026_10_04 and
// NOTE_path2a_tenth_pass_2026_10_05): the port's
// half of the committed PATH-2a checks, run by tests/test_diffgate_path2a*.py, which write IN (a JSON list of
// {id, summary, diff}) and read OUT.
//
//   node check_path2a.js --relation  REF IN OUT   main's port (REF, the reconstruction) against this port, both
//                                                  strict modes: every record may differ from main's only by an
//                                                  abstention in reach with the overlay's reason (the relation A),
//                                                  and each claim reads the same under --strict as without it;
//                                                  unparsed_claims and each claim's keys are compared too
//   node check_path2a.js --decisions REF IN OUT   per input, main's record and this port's record, for the
//                                                  Python-vs-port comparison (C), and both gate verdicts under
//                                                  --strict (NOTE_path2a_sixth_pass_2026_09_30, C-1, C-4)
//   node check_path2a.js --decisions-newer-engine REF IN OUT   as --decisions, on an engine whose case tables hold one
//                                                  pair this one's lack (U+A7CE and U+A7CF, unassigned through Unicode
//                                                  16, standing in for a later version's pair): main's two ports may
//                                                  then key a pair of paths apart (NOTE_path2a_seventh_pass_2026_09_30,
//                                                  C-1)
//   node check_path2a.js --found IN OUT            the port's _p2aFound on [words, text] pairs (A-1 of the sixth and
//                                                  seventh passes)
//   node check_path2a.js --opts REF OUT            odd `opts` arguments (a getter that answers differently on a
//                                                  second read, a Proxy counting reads, null, primitives): main's
//                                                  record and reads against this port's (A-2 of the sixth pass)
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
//   node check_path2a.js --bookmarklet MIN IN OUT  the minified bookmarklet (MIN) loaded in a stub page, as a browser
//                                                  runs it: its gate against ../diffgate.js on every input, both
//                                                  strict modes (NOTE_path2a_fifth_pass_2026_09_30, C-5)
//   node check_path2a.js --abstain PORT IN OUT     the overlay of PORT on a record given with each input ({id,
//                                                  summary, diff, gate}), not main's: a planted port's reading of a
//                                                  reason main never writes (C-2)
//   node check_path2a.js --hostile REF IN OUT      this port's APPLY (_p2aApply) on main's record with DECIDE functions
//                                                  written to do harm: ones that change their copy, return junk, indices
//                                                  out of range or twice, phrases and tags outside the fixed sets,
//                                                  decisions for claims outside reach, or throw. Per function: runs,
//                                                  records outside the relation, records equal to main's, claims in
//                                                  reach and withheld, by phrase (NOTE_path2a_tenth_pass_2026_10_05)
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
                 "_p2aJoined", "_P2A_ONE_SPACE", "_p2aSeam", "_p2aInt",
                 "_P2A_JS_SPACE", "_P2A_PY_SPACE", "_P2A_PY_BREAKS", "_P2A_DIVERGENT", "_P2A_HEADERS",
                 "_P2A_REACH_PAIRS", "_P2A_PHRASES", "_P2A_KIND_DEFECT", "P2A_DIRECTORY_BASENAME_ABSTAINS", "_P2A_OWN",
                 "_P2A_NEUTRAL_RANGES", "_p2aWordishUnit", "_p2aBadUnit", "_p2aAbstain", "_p2aFactsRaw", "_p2aFound",
                 "_p2aApply", "_p2aDecisions", "_P2A_TAGS", "_P2A_FIELDS",
                 "_P2A_EMOJI_RX", "_P2A_EMOJI_AS", "_P2A_LOWER_RUNS", "_P2A_NEVER_RANGES", "_p2aCaseCount"];
  // A name the port does not define reads undefined, so the checker also loads the ports of earlier passes (the
  // seventh pass runs its new pins against fcd3ce6a's port this way)
  const pick = names.map(n => n + ": typeof " + n + ' === "undefined" ? undefined : ' + n);
  return vm.runInNewContext(src + "\n" + extra + "\n;({" + pick.join(", ") + "})", {}, { filename: file });
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

function relation(a, b, strict, reach, defects, phrases, allowFallback = false) {
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
    else if (!allowFallback) {
      const said = y.why.split("): ").slice(1).join("): ");
      for (const k of ["error", "malformed"]) if (phrases[k] && said.startsWith(phrases[k] + ".")) bad.push(`claim ${i}: the overlay's ${k} fallback fired`);
    }
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

// NOTE_path2a_tenth_pass_2026_10_05: DECIDE functions written to do harm, for --hostile. Each maker gets the port's
// internals and the input and returns what APPLY is handed as `decide`. `want` says what the record must then be:
// "same" (main's, untouched: APPLY ignored everything), "all" (every claim in reach withheld, with `phrase`), or
// "relation" (inside the abstain-only relation, no more is said).
const ALL_TAGS = ["#97", "#121", "#97, #121", "#101"];
const inReach = (P, c) => P._P2A_REACH_PAIRS.some(p => p[0] === c.kind && p[1] === c.verdict);
const honest = (P, it) => seen => P._p2aDecisions(seen, () => P._p2aFactsRaw(it.diff || "", it.summary));
const wreck = seen => {
  for (const c of seen.claims) {
    c.verdict = "CONTRADICTED"; c.kind = "tests_pass"; c.why = "rewritten"; c.text = "rewritten";
    c.detail.path = "x"; c.detail.n = "9"; delete c.detail.declared; c.detail = null;
  }
  seen.claims.length = 0;
  seen.claims.push({ kind: "files_changed_count", verdict: "VERIFIED", why: "", detail: {} });
  seen.claims = null;
};
const forEach = (seen, pick, make) => {
  const out = [];
  seen.claims.forEach((c, i) => { if (pick(c, i)) for (const d of make(c, i)) out.push(d); });
  return out;
};
const HOSTILE = {
  "the block's own DECIDE": { want: "relation", make: honest },
  "changes, empties and grows its copy, decides nothing": { want: "same", make: () => seen => { wreck(seen); return []; } },
  "decides, then changes its copy": { want: "relation", make: (P, it) => seen => { const out = honest(P, it)(seen); wreck(seen); return out; } },
  "flips every verdict of its copy, then decides": { want: "relation", make: (P, it) => seen => {
    for (const c of seen.claims) c.verdict = c.verdict === "VERIFIED" ? "CONTRADICTED" : "VERIFIED";
    return honest(P, it)(seen);
  } },
  "returns nothing": { want: "all", phrase: "malformed", make: () => () => undefined },
  "returns null": { want: "all", phrase: "malformed", make: () => () => null },
  "returns a number": { want: "all", phrase: "malformed", make: () => () => 7 },
  "returns a string": { want: "all", phrase: "malformed", make: () => () => "dir" },
  "returns a mapping that looks like a list": { want: "all", phrase: "malformed", make: () => () => ({ length: 1, 0: [0, "dir", "#97"] }) },
  "returns its own argument": { want: "all", phrase: "malformed", make: () => seen => seen },
  "returns a list of junk": { want: "same", make: () => seen => [undefined, null, 7, "x", {}, [], [0], [0, "dir"], [0, "dir", "#97", 1],
    [[0], "dir", "#97"], ["0", "dir", "#97"], [0, 1, 2], [0, "dir", 97], [0, ["dir"], "#97"], seen, seen.claims, [seen.claims[0], "dir", "#97"]] },
  "returns the claims of its copy": { want: "same", make: () => seen => seen.claims },
  "indices out of range, negative, fractional, boolean and as strings": { want: "same", make: () => seen => {
    const n = seen.claims.length, out = [];
    for (const i of [-1, n, n + 1, 1e9, 0.5, n - 0.5, NaN, Infinity, -Infinity, true, false, "0", "1", null, undefined, [0], {}])
      for (const t of ALL_TAGS) out.push([i, "unreproduced", t]);
    return out;
  } },
  "one index twice, with two phrases": { want: "all", phrase: "error", make: P => seen =>
    forEach(seen, c => inReach(P, c), (c, i) => [[i, "error", P._P2A_KIND_DEFECT[c.kind]], [i, "unparsed", P._P2A_KIND_DEFECT[c.kind]], [i, "dir", P._P2A_KIND_DEFECT[c.kind]]]) },
  "phrases outside the fixed set": { want: "same", make: P => seen =>
    forEach(seen, () => true, (c, i) => ["nope", "", "DIR", " dir", "__proto__", "constructor", "toString", "hasOwnProperty", "valueOf", "length"]
      .map(k => [i, k, P._P2A_KIND_DEFECT[c.kind] || "#97"])) },
  "tags outside the fixed sets, and another kind's tag": { want: "same", make: P => seen =>
    forEach(seen, () => true, (c, i) => ["#1", "", "#97,#121", "#121, #97", " #97", "97", "__proto__", "length", "0"]
      .concat(ALL_TAGS.filter(t => !(P._P2A_TAGS[c.kind] || []).includes(t))).map(t => [i, "unreproduced", t])) },
  "decisions for every claim outside reach": { want: "same", make: P => seen =>
    forEach(seen, c => !inReach(P, c), (c, i) => ALL_TAGS.map(t => [i, "unreproduced", t])) },
  "a decision for every index, with every tag": { want: "all", phrase: "unreproduced", make: () => seen =>
    forEach(seen, () => true, (c, i) => ALL_TAGS.map(t => [i, "unreproduced", t])) },
  "throws": { want: "all", phrase: "error", make: () => () => { throw new Error("planted"); } },
  "throws something that is no error": { want: "all", phrase: "error", make: () => () => { throw null; } },
  "changes its copy, then throws": { want: "all", phrase: "error", make: () => seen => { wreck(seen); throw new TypeError("planted"); } },
  "a decision whose phrase throws when read": { want: "all", phrase: "error", make: () => () => {
    const d = [0, "dir", "#97"];
    Object.defineProperty(d, 1, { get() { throw new Error("planted"); } });
    return [[0, "unreproduced", "#97"], d];
  } },
  "a list that throws when its length is read": { want: "all", phrase: "error", make: () => () =>
    new Proxy([], { get(t, k) { if (k === "length") throw new Error("planted"); return t[k]; } }) },
  "a decision that answers differently each time it is read": { want: "relation", make: P => seen => {
    const out = [];
    seen.claims.forEach((c, i) => {
      let n = 0;
      out.push(new Proxy([i, "unreproduced", "#97"], { get(t, k) {
        if (k === "length") return 3;
        if (k === "0") return n++ ? "rewritten" : i;
        if (k === "1") return n++ % 2 ? "unreproduced" : "nope";
        if (k === "2") return (P._P2A_TAGS[c.kind] || ["#97"])[0];
        return t[k];
      } }));
    });
    return out;
  } },
};

function main(argv) {
  const mode = argv[0];
  if (mode === "--decisions-newer-engine") {
    // C-1 (NOTE_path2a_seventh_pass_2026_09_30): the patch is made before either port is loaded, so main's reader in
    // both reads it; the overlay's block calls neither method.
    const lower = String.prototype.toLowerCase, upper = String.prototype.toUpperCase;
    String.prototype.toLowerCase = function () { return lower.call(this).split("\ua7ce").join("\ua7cf"); };
    String.prototype.toUpperCase = function () { return upper.call(this).split("\ua7cf").join("\ua7ce"); };
    if ("\ua7ce".toLowerCase() !== "\ua7cf" || "\ua7ce".toLowerCase() === "\ua7ce") throw new Error("the patch did not take");
    return main(["--decisions"].concat(argv.slice(1)));
  }
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
      tags: P._P2A_TAGS, fields: P._P2A_FIELDS,
      directory_rule: P.P2A_DIRECTORY_BASENAME_ABSTAINS, own: P._P2A_OWN,
      // the summary's classes (NOTE_path2a_third_pass_2026_09_30): the units that are not wordish, and that are not bad
      neutral: P._P2A_NEUTRAL_RANGES.flatMap(([a, b]) => Array.from({ length: b - a + 1 }, (_, k) => a + k)),
      // pass 4 (NOTE_path2a_fourth_pass_2026_09_30, B-2): the neutral units this engine gives a case
      neutral_cased: P._P2A_NEUTRAL_RANGES.flatMap(([a, b]) => Array.from({ length: b - a + 1 }, (_, k) => a + k))
        .filter(u => { const ch = String.fromCharCode(u); return ch.toLowerCase() !== ch || ch.toUpperCase() !== ch; }),
      not_wordish: Array.from({ length: 0x10000 }, (_, u) => u).filter(u => !P._p2aWordishUnit(u)),
      not_bad: Array.from({ length: 0x10000 }, (_, u) => u).filter(u => !P._p2aBadUnit(u)),
      word_class: Array.from({ length: 0x10000 }, (_, u) => u).filter(u => /\w/.test(String.fromCharCode(u))),
      // pass 5 (NOTE_path2a_fifth_pass_2026_09_30, C-1 and C-2): the count seam; the units from 0x80 up this engine's
      // non-Unicode IGNORECASE matches against an ASCII letter (none, where CPython's matches four); and the strings the
      // port's digit table refuses
      one_space: cps(P._P2A_ONE_SPACE),
      ascii_folds: Array.from({ length: 0x10000 - 0x80 }, (_, k) => k + 0x80).filter(u => /[A-Za-z]/i.test(String.fromCharCode(u))),
      // pass 7 (NOTE_path2a_seventh_pass_2026_09_30, O-11): the pictograph emoji, two units each in this engine's strings
      emoji_matched: Array.from({ length: 0x100000 }, (_, k) => k + 0x10000).filter(cp => {
        const m = P._P2A_EMOJI_RX.exec(String.fromCodePoint(cp));
        return m !== null && m.index === 0 && m[0].length === 2;
      }),
      emoji_units_matched: Array.from({ length: 0x10000 }, (_, u) => u).filter(u => P._P2A_EMOJI_RX.test(String.fromCharCode(u))),
      emoji_flagged: [[0x1f300, 0x1f64f], [0x1f680, 0x1f6ff], [0x1f900, 0x1f9ff], [0x1fa70, 0x1faff]]
        .flatMap(([a, b]) => Array.from({ length: b - a + 1 }, (_, k) => a + k)).filter(cp => {
          const ch = String.fromCodePoint(cp);
          return /\w/.test(ch) || /\s/.test(ch) || ch.toLowerCase() !== ch || ch.toUpperCase() !== ch;
        }),
      emoji_as: P._P2A_EMOJI_AS,
      int_rejects: ["\u30003", "3\u3000", "\uff13", " 3", "3 ", "+3", "0x3", "3e1"].map(x => { try { P._p2aInt(x); return false; } catch (e) { return true; } }),
      // pass 8 (NOTE_path2a_eighth_pass_2026_10_01, B-1): this engine's single-code-point lowercase mappings from 0x80
      // up (U+0130 and U+212A aside), and the port's static case tables, which the test holds to them and to the Python's
      lower_map: Array.from({ length: 0x110000 - 0x80 }, (_, k) => k + 0x80).filter(cp => (cp < 0xd800 || cp > 0xdfff)
        && cp !== 0x130 && cp !== 0x212a).flatMap(cp => {
        const ch = String.fromCodePoint(cp), low = ch.toLowerCase();
        return low === ch ? [] : [[cp, [...low].length === 1 ? low.codePointAt(0) : -1]];
      }),
      lower_runs: P._P2A_LOWER_RUNS, never_ranges: P._P2A_NEVER_RANGES,
    }));
    return 0;
  }
  if (mode === "--bookmarklet") {
    // The shipped, minified bookmarklet in a stub page (window, document and location stubs), as a browser runs it:
    // the gate it installs against this directory's port, both strict modes.
    const noop = () => {};
    const el = () => ({ style: {}, remove: noop, appendChild: noop, addEventListener: noop, set innerHTML(v) {},
                        get innerHTML() { return ""; }, querySelector: () => null });
    const ctx = { console, location: { pathname: "/o/r/pull/1", origin: "https://github.com", href: "https://github.com/o/r/pull/1" },
                  document: { getElementById: () => null, createElement: el, body: { appendChild: noop } },
                  fetch: () => new Promise(noop), setTimeout, clearTimeout, alert: noop, navigator: {} };
    ctx.window = ctx;
    vm.createContext(ctx);
    vm.runInContext(fs.readFileSync(argv[1], "utf8"), ctx, { filename: "bookmarklet.min.js" });
    const BM = ctx.styxxDiffgateJS;
    if (!BM || typeof BM.gateDiffText !== "function") throw new Error("the bookmarklet installed no gate");
    const PORT = require(DEFAULT_PORT);
    const items = JSON.parse(fs.readFileSync(argv[2], "utf8"));
    const rec = (g, it, strict) => { try { return JSON.stringify(g.gateDiffText(it.summary, it.diff, { strict })); } catch (e) { return "raise " + ((e && e.constructor && e.constructor.name) || "Error"); } };
    let runs = 0, overlay = 0;
    const differ = [];
    for (const it of items) for (const strict of [false, true]) {
      runs++;
      const a = rec(BM, it, strict), b = rec(PORT, it, strict);
      if (a.includes("withheld by PATH-2a (")) overlay++;
      if (a !== b && differ.length < 20) differ.push([it.id, strict]);
    }
    fs.writeFileSync(argv[3], JSON.stringify({ runs, runs_with_overlay_reason: overlay, differ }));
    return 0;
  }
  if (mode === "--abstain") {
    const P = internals(path.resolve(argv[1]));
    const items = JSON.parse(fs.readFileSync(argv[2], "utf8"));
    const out = items.map(it => {
      const g = JSON.parse(JSON.stringify(it.gate));
      P._p2aAbstain(g, false, () => P._p2aFactsRaw(it.diff || "", it.summary));
      return { id: it.id, rec: g };
    });
    fs.writeFileSync(argv[3], JSON.stringify(out));
    return 0;
  }
  if (mode === "--hostile") {
    const REF = require(path.resolve(argv[1]));
    const port = path.resolve(argv[4] || DEFAULT_PORT);
    const P = internals(port);
    const NEW = require(port);
    const reach = new Set(P._P2A_REACH_PAIRS.map(p => p[0] + "|" + p[1]));
    const defects = P._P2A_TAGS;
    const items = JSON.parse(fs.readFileSync(argv[2], "utf8"));
    const keyOf = why => { for (const [k, p] of Object.entries(P._P2A_PHRASES)) if (why.includes("): " + p + ". main's reading: ")) return k; return null; };
    const out = {};
    for (const [name, h] of Object.entries(HOSTILE)) {
      const c = { runs: 0, broken: 0, same: 0, not_returned: 0, threw: 0, in_reach: 0, withheld: 0, unlike_the_port: 0, decisions: 0 };
      const phrases = {}, examples = [];
      for (const it of items) for (const strict of [false, true]) {
        let a;
        try { a = JSON.parse(JSON.stringify(REF.gateDiffText(it.summary, it.diff, { strict }))); } catch (e) { continue; }
        const g = REF.gateDiffText(it.summary, it.diff, { strict });
        let ret;
        const decide = h.make(P, it);
        // for the block's own DECIDE, how many decisions it returned: APPLY must take every one
        const spy = name === "the block's own DECIDE" ? seen => { const got = decide(seen); c.decisions += got.length; return got; } : decide;
        try { ret = P._p2aApply(g, strict, spy); } catch (e) { c.threw++; }
        const b = JSON.parse(JSON.stringify(g));
        c.runs++;
        if (ret !== g) c.not_returned++;
        const bad = relation(a, b, strict, reach, defects, P._P2A_PHRASES, true);
        if (bad.length) { c.broken++; if (examples.length < 5) examples.push([it.id, strict, bad]); }
        if (JSON.stringify(a) === JSON.stringify(b)) c.same++;
        a.claims.forEach((x, i) => {
          if (!reach.has(x.kind + "|" + x.verdict)) return;
          c.in_reach++;
          if (b.claims[i] && b.claims[i].verdict !== x.verdict) { c.withheld++; const k = String(keyOf(b.claims[i].why)); phrases[k] = (phrases[k] || 0) + 1; }
        });
        if (name === "the block's own DECIDE" && JSON.stringify(b) !== JSON.stringify(NEW.gateDiffText(it.summary, it.diff, { strict }))) c.unlike_the_port++;
      }
      out[name] = { want: h.want, phrase: h.phrase || null, counts: c, phrases, examples };
    }
    fs.writeFileSync(argv[3], JSON.stringify(out));
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
      let best = Infinity, main = Infinity;
      for (let k = 0; k < 3; k++) {
        const t = process.hrtime.bigint();
        const g = REF.gateDiffText(it.summary, it.diff);
        const t0 = process.hrtime.bigint();
        P._p2aAbstain(g, false, () => P._p2aFactsRaw(it.diff || "", it.summary));
        best = Math.min(best, Number(process.hrtime.bigint() - t0) / 1e6);
        main = Math.min(main, Number(t0 - t) / 1e6);   // main's own call on the same input (sixth pass, A-1)
      }
      return { id: it.id, overlay: best, main };
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
      const out = items.map(it => ({ id: it.id, main: record(REF.gateDiffText, it, false), new: record(NEW.gateDiffText, it, false),
                                      strict: { main: record(REF.gateDiffText, it, true).verdict || null,
                                                new: record(NEW.gateDiffText, it, true).verdict || null } }));
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
  if (mode === "--found") {
    // A-1 (NOTE_path2a_sixth_pass_2026_09_30): the port's _p2aFound on each [words, text] of IN, sorted
    const P = internals(DEFAULT_PORT);
    const items = JSON.parse(fs.readFileSync(argv[1], "utf8"));
    fs.writeFileSync(argv[2], JSON.stringify(items.map(([words, text]) => [...P._p2aFound(new Set(words), text)].sort())));
    return 0;
  }
  if (mode === "--opts") {
    // A-2 (NOTE_path2a_sixth_pass_2026_09_30): main destructures `opts` once; this port reads it once, in main's order,
    // and hands main the snapshot, so main and the overlay read one strict and the reads main makes are the same.
    const REF = require(path.resolve(argv[1]));
    const NEW = require(DEFAULT_PORT);
    const S = "Modified x.py. All tests pass.";
    const D = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n";
    const shapes = {
      "a strict that answers true, then false": () => { let n = 0; return { get strict() { return n++ === 0; } }; },
      "a strict that answers false, then true": () => { let n = 0; return { get strict() { return n++ !== 0; } }; },
      "a Proxy that logs its reads": log => new Proxy({ strict: true }, { get(t, k) { log.push(String(k)); return t[k]; } }),
      "null": () => null, "undefined": () => undefined, "a number": () => 7, "a string": () => "strict",
      "strict on the prototype": () => Object.create({ strict: true }), "_declared true": () => ({ _declared: true }),
    };
    const out = [];
    for (const [name, make] of Object.entries(shapes)) {
      const logs = { main: [], new: [] };
      const run = (gate, who) => {
        try { return JSON.parse(JSON.stringify(gate(S, D, make(logs[who])))); } catch (e) { return { error: e.constructor.name, message: String(e.message) }; }
      };
      const a = run(REF.gateDiffText, "main"), b = run(NEW.gateDiffText, "new");
      out.push({ name, same: JSON.stringify(a) === JSON.stringify(b), reads_same: JSON.stringify(logs.main) === JSON.stringify(logs.new), main: a, new: b, reads: logs });
    }
    fs.writeFileSync(argv[2], JSON.stringify(out));
    return 0;
  }
  console.error("usage: node check_path2a.js --opts REF OUT | --relation REF IN OUT | --decisions REF IN OUT | --decisions-newer-engine REF IN OUT | --records PORT IN OUT | --lockstep IN OUT | --tables OUT | --error-fallback REF IN OUT | --timing REF IN OUT | --overlay-timing REF IN OUT | --bookmarklet MIN IN OUT | --abstain PORT IN OUT | --hostile REF IN OUT");
  return 2;
}

process.exit(main(process.argv.slice(2)));

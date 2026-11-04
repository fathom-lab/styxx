/* verify.js — darkflobi.gate/entry/v0: a second implementation of the chain check in the gate's ledger.py
   (fathom-lab/styxx, papers/darkflobi-gate/). Runs in a browser (the ledger page inlines it) and under node
   (`node check_ledger.cjs ledger.jsonl`). Standard library only: TextEncoder/TextDecoder, crypto.subtle, Intl.

   What it checks, line by line: the bytes are the canonical dump of the object (compact, keys sorted; for the
   value types the ledger admits that is RFC 8785); `entry_id` = sha256 of the JCS of the core (every key but
   `entry_id`); `prev` names the line before (the header through its domain-separated digest
   sha256("darkflobi.gate/log/v0\n" + JCS(header))); `seq` is the one before plus one; `day` is the timestamp's
   calendar day in America/New_York; the schema strings. What it does not check: the truth of any field; whether
   every send got a line (the census at the window's close reads the chats for that); the words of any message.

   One known difference from ledger.py: a line with a key written twice. ledger.py drops the line (and the chain
   breaks after it); JSON.parse keeps the last value, so this verifier keeps the line and flags its bytes as not
   canonical. Both report the log as not re-deriving. */
(function (root, factory) {
  if (typeof module === "object" && module && module.exports) module.exports = factory();
  else root.LedgerVerify = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var LOG_SCHEMA = "darkflobi.gate/log/v0";
  var ENTRY_SCHEMA = "darkflobi.gate/entry/v0";
  var HEADER_DOMAIN = "darkflobi.gate/log/v0\n";
  var ANCHOR_TZ = "America/New_York";
  var OUTSIDE_CORE = { entry_id: true };
  var enc = new TextEncoder();
  var dec = new TextDecoder("utf-8", { fatal: true });

  // ---- bytes
  function subtleCrypto() {
    var c = typeof globalThis !== "undefined" ? globalThis.crypto : undefined;
    if (!c || !c.subtle) throw new Error("crypto.subtle is not available here: open the page over https or file, or run it under node");
    return c.subtle;
  }
  function hex(buf) {
    var u = new Uint8Array(buf), s = "";
    for (var i = 0; i < u.length; i++) s += (u[i] < 16 ? "0" : "") + u[i].toString(16);
    return s;
  }
  function sha256Hex(bytes) {           // bytes: Uint8Array, or a string read as UTF-8
    var u = typeof bytes === "string" ? enc.encode(bytes) : bytes;
    return subtleCrypto().digest("SHA-256", u).then(hex);
  }

  // ---- canonical form (ledger.py: jcs / canonical_line)
  function checkValue(v, path) {
    if (v === null || typeof v === "boolean" || typeof v === "string") return;
    if (typeof v === "number") {
      if (!Number.isInteger(v)) throw new Error("float at " + path + ": the ledger core carries integers and strings only");
      if (Math.abs(v) > 9007199254740992) throw new Error("integer at " + path + " exceeds 2^53; JCS numbers are IEEE doubles");
      return;
    }
    if (Array.isArray(v)) { for (var i = 0; i < v.length; i++) checkValue(v[i], path + "[" + i + "]"); return; }
    if (typeof v === "object") {
      var ks = Object.keys(v);
      for (var j = 0; j < ks.length; j++) {
        if (!/^[\x00-\x7e]*$/.test(ks[j])) throw new Error("non-ASCII key at " + path);
        checkValue(v[ks[j]], path + "/" + ks[j]);
      }
      return;
    }
    throw new Error("unserialisable value at " + path);
  }
  function byCodeUnit(a, b) { return a < b ? -1 : (a > b ? 1 : 0); }
  function dump(v) {                    // the writer's one byte form: compact, keys sorted
    if (v === null || typeof v !== "object") return JSON.stringify(v);
    if (Array.isArray(v)) { var parts = []; for (var i = 0; i < v.length; i++) parts.push(dump(v[i])); return "[" + parts.join(",") + "]"; }
    var ks = Object.keys(v).sort(byCodeUnit), out = [];
    for (var j = 0; j < ks.length; j++) out.push(JSON.stringify(ks[j]) + ":" + dump(v[ks[j]]));
    return "{" + out.join(",") + "}";
  }
  function jcs(v) { checkValue(v, "$"); return dump(v); }
  function coreOf(entry) {
    var c = {};
    for (var k in entry) if (Object.prototype.hasOwnProperty.call(entry, k) && !OUTSIDE_CORE[k]) c[k] = entry[k];
    return c;
  }
  function headerDigest(header) { return sha256Hex(HEADER_DOMAIN + jcs(header)); }
  function entryIdOf(core) { return sha256Hex(jcs(core)); }

  // ---- time (ledger.py: local_day)
  var dayFmt = null;
  function localDay(ts) {
    var s = String(ts).trim();
    if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$/.test(s)) throw new Error("timestamp unreadable");
    if (!/(Z|[+-]\d{2}:\d{2})$/.test(s)) s += "Z";   // a naive timestamp is read as UTC, as ledger.py reads it
    var ms = Date.parse(s);
    if (isNaN(ms)) throw new Error("timestamp unreadable");
    if (!dayFmt) dayFmt = new Intl.DateTimeFormat("en-CA", { timeZone: ANCHOR_TZ, year: "numeric", month: "2-digit", day: "2-digit" });
    var parts = dayFmt.formatToParts(new Date(ms)), m = {};
    for (var i = 0; i < parts.length; i++) m[parts[i].type] = parts[i].value;
    return m.year + "-" + m.month + "-" + m.day;
  }

  // ---- reading (ledger.py: read_log)
  function splitLines(u8) {
    var out = [], start = 0, hasCR = false;
    for (var i = 0; i <= u8.length; i++) {
      if (i === u8.length || u8[i] === 0x0a) { out.push({ line: out.length + 1, bytes: u8.subarray(start, i) }); start = i + 1; }
      else if (u8[i] === 0x0d) hasCR = true;
    }
    return { lines: out, hasCR: hasCR };
  }
  function isBlank(b) {
    for (var i = 0; i < b.length; i++) { var c = b[i]; if (c !== 0x20 && c !== 0x09 && c !== 0x0a && c !== 0x0d && c !== 0x0b && c !== 0x0c) return false; }
    return true;
  }
  function bytesEqual(a, b) { if (a.length !== b.length) return false; for (var i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }

  function readLog(u8) {                // -> {header, entries: [{line, entry}], problems}; never throws on content
    var header = null, entries = [], problems = [];
    var sp = splitLines(u8);
    if (sp.hasCR) problems.push({ line: null, problem: "the log holds a CR byte; lines are LF" });
    for (var i = 0; i < sp.lines.length; i++) {
      var ln = sp.lines[i], n = ln.line, obj;
      if (isBlank(ln.bytes)) continue;
      try { obj = JSON.parse(dec.decode(ln.bytes)); }
      catch (e) { problems.push({ line: n, problem: "unparsable: " + (e && e.name ? e.name : "error") }); continue; }
      if (obj === null || typeof obj !== "object" || Array.isArray(obj)) { problems.push({ line: n, problem: "not an object" }); continue; }
      var canonical = false;
      try { canonical = bytesEqual(enc.encode(dump(obj)), ln.bytes); } catch (e2) { canonical = false; }
      if (!canonical) problems.push({ line: n, problem: "the line's bytes are not the canonical dump of its object" });
      if (header === null && entries.length === 0 && obj.schema === LOG_SCHEMA) { header = obj; continue; }
      entries.push({ line: n, entry: obj });
    }
    if (header === null && entries.length) problems.push({ line: 1, problem: "no header on line 1" });
    return { header: header, entries: entries, problems: problems };
  }

  // ---- withheld lines: a publisher may leave a line out (its fields carried an identifier) and publish only
  // its seq and digest. The chain is then bridged through it on the published digest: the reader cannot check
  // that digest against fields, so the lines between the previous posted head and that line are not pinned
  // for the reader (see `coverage`).
  function withheldMap(withheld) {
    var m = {};
    (withheld || []).forEach(function (w) { if (w && Number.isInteger(w.seq) && typeof w.entry_id === "string") m[w.seq] = w.entry_id; });
    return m;
  }

  // ---- the chain (ledger.py: check_chain) -> {problems, bridged}
  async function checkChain(header, entries, withheld) {
    var bad = [], bridged = [], wm = withheldMap(withheld);
    if (header === null) return { problems: entries.length ? [{ line: 1, seq: null, problems: ["no header"] }] : bad, bridged: bridged };
    var prev = await headerDigest(header), wantSeq = 1;
    for (var i = 0; i < entries.length; i++) {
      var it = entries[i], e = it.entry, here = [], core = coreOf(e), want = null;
      while (wm[wantSeq] !== undefined && e.seq !== wantSeq) { prev = wm[wantSeq]; bridged.push(wantSeq); wantSeq++; }
      try { want = await entryIdOf(core); } catch (err) { here.push("core not canonicalisable: " + err.message); }
      if (core.schema !== ENTRY_SCHEMA) here.push("schema is not " + ENTRY_SCHEMA);
      if (want !== null && e.entry_id !== want) here.push("entry_id does not re-derive from the core");
      if (e.prev !== prev) here.push("prev does not name the line before");
      if (e.seq !== wantSeq) here.push("seq is " + JSON.stringify(e.seq === undefined ? null : e.seq) + ", expected " + wantSeq);
      if (Object.prototype.hasOwnProperty.call(core, "timestamp")) {
        try { if (localDay(core.timestamp) !== core.day) here.push("day does not re-derive from timestamp"); }
        catch (err2) { here.push("timestamp unreadable"); }
      }
      if (here.length) bad.push({ line: it.line, seq: e.seq, problems: here });
      prev = typeof e.entry_id === "string" ? e.entry_id : prev;
      wantSeq = (Number.isInteger(e.seq) ? e.seq : wantSeq) + 1;
    }
    return { problems: bad, bridged: bridged };
  }
  function headOf(header, entries) {
    if (entries.length) { var h = entries[entries.length - 1].entry.entry_id; return Promise.resolve(typeof h === "string" ? h : null); }
    return header ? headerDigest(header) : Promise.resolve(null);
  }
  async function verifyBytes(u8, opts) {
    var r = readLog(u8);
    var chain = await checkChain(r.header, r.entries, opts && opts.withheld);
    var head = await headOf(r.header, r.entries);
    var fileSha = await sha256Hex(u8);
    var problems = r.problems.concat(chain.problems);
    return { header: r.header, entries: r.entries, problems: problems, bridged: chain.bridged, head: head, file_sha256: fileSha, ok: problems.length === 0 };
  }
  function verifyText(text, opts) { return verifyBytes(enc.encode(text), opts); }

  // ---- what a posted head pins for the reader. A published line is pinned when every line between it and the
  // first posted head at or after it is published (verifiable); a withheld line in between severs that.
  function coverage(entries, withheld, anchorSeqs) {
    var ws = [], as = [];
    (withheld || []).forEach(function (w) { if (w && Number.isInteger(w.seq)) ws.push(w.seq); });
    (anchorSeqs || []).forEach(function (s) { if (Number.isInteger(s)) as.push(s); });
    as.sort(function (a, b) { return a - b; });
    var lastAnchor = as.length ? as[as.length - 1] : 0, pinned = 0, after = 0, unpinned = [];
    entries.forEach(function (it) {
      var s = it.entry.seq;
      if (!Number.isInteger(s)) return;
      if (s > lastAnchor) { after++; return; }
      var A = null;
      for (var i = 0; i < as.length; i++) if (as[i] >= s) { A = as[i]; break; }
      var severed = ws.some(function (w) { return w > s && w <= A; });
      if (severed) unpinned.push(s); else pinned++;
    });
    unpinned.sort(function (a, b) { return a - b; });
    var ranges = [];
    unpinned.forEach(function (s) { var last = ranges[ranges.length - 1]; if (last && last[1] === s - 1) last[1] = s; else ranges.push([s, s]); });
    return { pinned: pinned, unpinned_by_withholding: unpinned.length, unpinned_ranges: ranges, after_last_anchor: after, withheld: ws.length, last_anchor_seq: lastAnchor };
  }

  // ---- anchors (ledger.py: anchor_text / parse_anchor_text; the daily post at t.me/DFledger)
  function anchorText(date, head, seq) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !/^[0-9a-f]{64}$/.test(head) || !(Number.isInteger(seq) && seq >= 0)) throw new Error("anchor fields out of form");
    return "gate anchor " + date + " head " + head + " seq " + seq;
  }
  function parseAnchorText(text) {       // the whole text IS the form (one trailing newline allowed, as re.match + $ reads it)
    var m = /^gate anchor (\d{4}-\d{2}-\d{2}) head ([0-9a-f]{64}) seq (\d+)\n?$/.exec(String(text));
    return m ? { date: m[1], head: m[2], seq: parseInt(m[3], 10) } : null;
  }
  function checkAnchor(entries, a, withheld) {   // a: {date, head, seq}; entries as verifyBytes returns them
    var bySeq = null, after = 0, own = null, wm = withheldMap(withheld);
    for (var i = 0; i < entries.length; i++) {
      var e = entries[i].entry;
      if (e.seq === a.seq) bySeq = e;
      if (e.day === a.date && Number.isInteger(e.seq) && e.seq > a.seq) after++;
      if (e.kind === "anchor" && e.anchor_date === a.date) own = e;
    }
    var viaWithheld = !bySeq && wm[a.seq] !== undefined;
    var out = {
      date: a.date, seq: a.seq, head: a.head,
      in_chain: viaWithheld ? wm[a.seq] === a.head : (!!bySeq && bySeq.entry_id === a.head),
      via_withheld: viaWithheld,
      line_at_seq: bySeq ? { kind: bySeq.kind, day: bySeq.day, timestamp: bySeq.timestamp } : (viaWithheld ? { withheld: true } : null),
      day_lines_after: after,
      ledger_anchor_line: own ? { seq: own.seq, agrees: own.anchor_head === a.head && own.anchor_head_seq === a.seq, late: own.late === true } : null
    };
    out.ok = out.in_chain && after === 0 && (own === null || out.ledger_anchor_line.agrees);
    return out;
  }

  return {
    LOG_SCHEMA: LOG_SCHEMA, ENTRY_SCHEMA: ENTRY_SCHEMA, ANCHOR_TZ: ANCHOR_TZ,
    jcs: jcs, dump: dump, sha256Hex: sha256Hex, localDay: localDay, coreOf: coreOf,
    headerDigest: headerDigest, entryIdOf: entryIdOf, readLog: readLog, checkChain: checkChain, headOf: headOf,
    verifyBytes: verifyBytes, verifyText: verifyText, coverage: coverage,
    anchorText: anchorText, parseAnchorText: parseAnchorText, checkAnchor: checkAnchor
  };
});

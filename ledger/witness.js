/* witness.js — the per-line witness on the ledger page (THE_PLAN 1a, part 2: the page side). Runs in a browser
   (the page inlines it when the builder runs with --witness <store>) and under node (`node test_witness_page.cjs`).
   Standard library only. The embedded document (#witness-data) is what the builder derived from the witness store
   and the ledger's own lines: every count with its denominator, and per entry_id the latest state of its proof.

   Words, kept apart on this page: a PENDING .ots file is a calendar's receipt — a url and a commitment, no time a
   stranger can check. a PROOF is the file once it holds a bitcoin attestation, and it attests exactly this: the
   line's digest existed before that block. the block header's time is the one clock here the lab does not run (the
   miner's, within about 2 h of consensus wall time). the tail's own clock (seen_at, stamped_at) is the lab's word.

   Pure functions, tested under node:
     fmtDur(seconds)              "42 min", "1 h 49 min", "4 d 8 h"; negative with a leading "-"
     stateOf(rec)                 {kind, text}: none | pending | bitcoin | disagree | mismatch | failed | found
     cellText(rec, orphans)       the "ots" column: "pending" | "block H" | "—" | "orphan …"; a seq with a second
                                  entry_id in the store adds " + orphan <id16>…"
     latencyRows(ol)              the outside-latency table as DISJOINT rows that sum to ol.of (the builder's
                                  buckets are cumulative, as REPORT.md prints them)
   Browser glue (needs a document): page(dataEl) reads the embed; mount(ctx) adds the tile, the section and the
   lines column; foundLines(entry) gives prove-a-message its two witness lines. Absent the data element, or
   window.LedgerWitness, the page renders nothing of this: the feature is off. */
(function (root, factory) {
  if (typeof module === "object" && module && module.exports) module.exports = factory();
  else root.LedgerWitness = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var H = 3600, D = 86400;
  var FILES_URL = "https://fathom-lab.github.io/styxx/ledger/witness/";
  var WORDS = {
    pending: "a pending file is a calendar's receipt: a url and a commitment, no time a stranger can check",
    proof: "a proof attests that the line's digest existed before its bitcoin block; it does not show that every send got a line, nor the words, nor who wrote them",
    clocks: "seen_at and stamped_at are the tail's clock (the lab's word); the line's timestamp is the gate's statement; block_time is the block header's time read from two explorers that must agree (the miner's, within about 2 h of consensus wall time)",
    replace: "the proof bytes beside this page are replaced by their upgraded form on later builds: a proof's commitment never changes; its bytes grow when a calendar's bitcoin attestation is added",
    source: "the witness states on this page (status, block, header time) are the tail's store as copied at build; this page parses no proof and asks no explorer; the proof file beside this page is what a stranger checks"
  };

  function esc(s) {
    return String(s === null || s === undefined ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c];
    });
  }
  function kofn(k, n) { return String(k) + " of " + String(n); }
  function fmtDur(sec) {
    if (sec === null || sec === undefined || typeof sec !== "number" || isNaN(sec)) return "-";
    var neg = sec < 0, s = Math.round(Math.abs(sec)), out;
    if (s < 60) out = s + " s";
    else if (s < 3600) out = Math.floor(s / 60) + " min " + (s % 60) + " s";
    else if (s < 86400) out = Math.floor(s / 3600) + " h " + Math.floor((s % 3600) / 60) + " min";
    else out = Math.floor(s / 86400) + " d " + Math.floor((s % 86400) / 3600) + " h";
    return (neg ? "-" : "") + out;
  }
  function isoOfUnix(t) {
    if (typeof t !== "number" || !isFinite(t)) return "-";
    try { return new Date(t * 1000).toISOString().replace(/\.\d{3}Z$/, "Z"); } catch (e) { return "-"; }
  }
  function id16(id) { return String(id || "").slice(0, 16) + "…"; }

  // the latest state of one store record (the embed's by_entry row), or none when there is no record
  function stateOf(rec) {
    if (!rec) return { kind: "none", text: "—" };
    var st = rec.status;
    if (st === "bitcoin") return { kind: "bitcoin", text: "block " + rec.block };
    if (st === "pending" || st === "proof_found_without_record") return { kind: "pending", text: "pending" };
    if (st === "explorers_disagree") return { kind: "disagree", text: "block " + rec.block + " (explorers disagree)" };
    if (st === "bitcoin_mismatch") return { kind: "mismatch", text: "block " + rec.block + " (mismatch)" };
    if (st === "stamp_failed") return { kind: "failed", text: "stamp failed" };
    return { kind: "other", text: String(st || "—") };
  }
  // the "ots" column for one published line: its own record's state, plus any OTHER entry_id the store holds for
  // the same seq (an orphan: a digest no longer in the chain, which a rewrite or a re-chain leaves behind)
  function cellText(rec, orphans) {
    var s = stateOf(rec), t = s.text;
    var os = (orphans || []).map(function (o) { return "orphan " + id16(o.entry_id) + " (" + stateOf(o).text + ")"; });
    if (os.length) t = (rec ? t + " + " : "") + os.join(" + ");
    return t;
  }
  function cellClass(rec, orphans) {
    if (orphans && orphans.length) return "m";
    var k = stateOf(rec).kind;
    return k === "bitcoin" ? "ok" : (k === "none" || k === "failed" || k === "disagree" || k === "mismatch") ? "dim" : "";
  }
  // outside latency: the builder's buckets are cumulative (within 1 h ⊂ within 6 h ⊂ within 24 h), as REPORT.md
  // prints them; the page shows disjoint rows so a reader can add them up to n
  function latencyRows(ol) {
    ol = ol || {};
    var n = ol.of || 0, w1 = ol.within_1h || 0, w6 = ol.within_6h || 0, w24 = ol.within_24h || 0, over = ol.over_24h || 0;
    var none = ol.none_yet !== undefined ? ol.none_yet : Math.max(0, n - w24 - over);
    return [
      { label: "within 1 h", count: w1 },
      { label: "1 h to 6 h", count: Math.max(0, w6 - w1) },
      { label: "6 h to 24 h", count: Math.max(0, w24 - w6) },
      { label: "over 24 h", count: over },
      { label: "none yet (no bitcoin attestation)", count: none }
    ];
  }
  function lagRows(tl) {
    tl = tl || {};
    var a = tl.within_10min || 0, b = tl.within_1h || 0, c = tl.within_24h || 0, over = tl.over_24h || 0;
    return [
      { label: "within 10 min", count: a },
      { label: "10 min to 1 h", count: Math.max(0, b - a) },
      { label: "1 h to 24 h", count: Math.max(0, c - b) },
      { label: "over 24 h", count: over }
    ];
  }
  function cnt(x) { return x && typeof x === "object" ? (x.count || 0) : (x || 0); }

  function page(dataEl) {
    if (!dataEl) return null;
    var doc;
    try { doc = JSON.parse(dataEl.textContent); } catch (e) { return null; }
    if (!doc || doc.enabled !== true || !doc.counts || !doc.by_entry || typeof doc.by_entry !== "object") return null;
    var c = doc.counts, by = doc.by_entry, n = c.lines || 0, w = c.window || null;
    var bitcoin = cnt(c.bitcoin), pending = cnt(c.pending), commit = cnt(c.with_commitment), disagree = cnt(c.explorers_disagree), mismatch = cnt(c.bitcoin_mismatch), failed = cnt(c.stamp_failed), noRec = cnt(c.no_record);
    var orph = c.orphans || { count: 0, of: 0, entries: [] }, dup = c.dup_seqs || { count: 0, seqs: [] };
    var ol = c.outside_latency || {}, tl = c.tail_lag || {}, sb = c.stamp_batches || {};
    var seal = c.seal || doc.seal || {};
    var sha = doc.prereg_sha256 || c.prereg_sha256 || "";
    var tileEl = null, chainIdBySeq = null, orphansBySeq = null;

    function byId(id) { return (typeof document !== "undefined" && document.getElementById) ? document.getElementById(id) : null; }

    // ---- the tile: one number with its denominator
    function tileHtml() {
      return "<b>" + esc(bitcoin) + " of " + esc(n) + "</b><span>lines carry a bitcoin-attested proof · pending " + esc(kofn(pending, n)) + " · <a href=\"#witness\">the witness</a></span>";
    }
    // ---- the note under the counts: pending, orphans, the prereg, the bars
    function windowText() {
      if (!w || w.open !== true) return "kill-gate window: window not open: the standing task is not registered. the bars are read at the close of day 9 of the opening window (WITNESS_PREREG.md §2-3); until the task runs there is no window and no reading.";
      var a = w.bar_a || {}, b = w.bar_b || {};
      return "kill-gate window " + esc(w.since) + " to " + esc(w.until) + " (exclusive), denominator " + esc(w.denominator) + " lines by their own timestamp; of which not yet 48 h old at this build " + esc(kofn(w.young, w.denominator)) + ", stamped retroactively " + esc(kofn(w.retroactive, w.denominator)) + "; with a calendar commitment " + esc(kofn(w.with_commitment, w.denominator)) + ", with a bitcoin attestation " + esc(kofn(w.attested, w.denominator)) + ", within 24 h of their own timestamp " + esc(kofn(w.within_24h, w.denominator)) + ". current readings at " + esc(w.read_at) + " (the preregistered reading is the one at the close of day 9, " + esc(w.final_reading_at) + "): bar (a) median outside latency over the attested in-window lines " + esc(a.median_s === null || a.median_s === undefined ? "-" : fmtDur(a.median_s)) + " (over " + esc(a.over) + ") — " + esc(a.reading) + "; bar (b) lines with no bitcoin attestation with block_time ≤ line_timestamp + 48 h: " + esc(kofn(b.miss, w.denominator)) + " — " + esc(b.reading) + "; bar (c) is a stranger's transcript and is not computed here.";
    }
    function noteText() {
      return "pending " + kofn(pending, n) + " at the calendars (receipts, no time); bitcoin-attested " + kofn(bitcoin, n) + "; store entries whose digest is no longer in the chain (orphans) " + kofn(orph.count || 0, orph.of || 0) + "; seqs with more than one entry_id in the store " + (dup.count || 0) + ". preregistered kill gate: WITNESS_PREREG.md sha256 " + sha + ", sealed " + (seal.sealed_at || "-") + " in seal.jsonl line 1; its own proof: " + (seal.proof_status || "not stamped yet") + (seal.proof_block ? " (block " + seal.proof_block + ")" : "") + ". " + windowText();
    }
    function retroText() {
      var r = tl.retroactive || 0, days = tl.retroactive_stamp_days || [];
      if (!r) return "no line was stamped retroactively (every line was seen within 1 h of its own timestamp).";
      return r + " of " + (tl.of || n) + " lines were stamped retroactively (seen more than 1 h after their own timestamp)" + (days.length ? ", on " + days.join(", ") : "") + ": a proof for one of them pins its digest to a bitcoin block at or after the stamping (" + (days.length ? days.join(", ") : "the stamping day") + "), not to the line's own day.";
    }
    // ---- prove a message: two lines per found line
    function foundLines(e) {
      var id = e && e.entry_id, r = by[id], s = stateOf(r);
      var verify = "node ots-cli.js verify -i -d " + id + " " + id + ".ots";
      var proofLine = "  proof: witness/proofs/" + id + ".ots — a stranger checks it with `" + verify + "` (javascript-opentimestamps 0.4.9; -i uses a public explorer, drop it for your own bitcoin node), then reads the block's header time at https://mempool.space/api/block-height/<H> and https://blockstream.info/api/block-height/<H>";
      if (s.kind === "none") return ["  witness: no record in the witness store for this line (the store's last run was before this line, or its stamp failed and is retried on the next run)", "  proof: none yet"];
      if (s.kind === "failed") return ["  witness: the last stamp attempt for this line failed" + (r.error ? " (" + r.error + ")" : "") + "; retried on the next run", "  proof: none yet"];
      if (s.kind === "pending") return ["  witness: pending at " + (r.calendars_n || 0) + " calendars since " + (r.stamped_at || r.seen_at || "-") + " (the tail's clock, the lab's word); " + WORDS.pending + "; " + WORDS.source, proofLine];
      if (s.kind === "bitcoin") return ["  witness: attested in bitcoin block " + r.block + ", header time " + isoOfUnix(r.block_time) + " (recorded by the tail after two explorers agreed at upgrade): the line's digest existed before that block; the header's time is the miner's, within about 2 h of consensus wall time; " + WORDS.source, proofLine];
      if (s.kind === "disagree") return ["  witness: a bitcoin attestation for block " + r.block + " on which the two explorers did not agree; revisited at every upgrade; not counted as a proof here; " + WORDS.source, proofLine];
      if (s.kind === "mismatch") return ["  witness: a bitcoin attestation for block " + r.block + " that did not match the block's merkle root; revisited at every upgrade; not counted as a proof here; " + WORDS.source, proofLine];
      return ["  witness: state " + s.text, proofLine];
    }
    // ---- the lines column
    function indexChain(ctx) {
      chainIdBySeq = Object.create(null); orphansBySeq = Object.create(null);
      var inChain = Object.create(null);
      ((ctx && ctx.R && ctx.R.entries) || []).forEach(function (it) { var e = it.entry; chainIdBySeq[e.seq] = e.entry_id; inChain[e.entry_id] = true; });
      ((ctx && ctx.withheld) || []).forEach(function (wh) { chainIdBySeq[wh.seq] = wh.entry_id; inChain[wh.entry_id] = true; });
      Object.keys(by).forEach(function (id) { if (!inChain[id]) { var o = by[id]; (orphansBySeq[o.seq] = orphansBySeq[o.seq] || []).push({ entry_id: id, status: o.status, block: o.block }); } });
    }
    function cellFor(seq) {
      var id = chainIdBySeq ? chainIdBySeq[seq] : null;
      return { text: cellText(id ? by[id] : null, orphansBySeq ? orphansBySeq[seq] : []), cls: cellClass(id ? by[id] : null, orphansBySeq ? orphansBySeq[seq] : []) };
    }
    function decorateLines(ctx) {
      if (typeof document === "undefined") return 0;
      var tbl = byId("lines");
      if (!tbl || !tbl.getElementsByTagName) return 0;
      if (!chainIdBySeq) indexChain(ctx);
      var trs = tbl.getElementsByTagName("tr"), done = 0;
      for (var i = 0; i < trs.length; i++) {
        var tr = trs[i], cells = tr.cells || tr.children || [];
        if (!cells.length) continue;
        var c0 = cells[0];
        if (String(c0.tagName || "").toUpperCase() === "TH") { var th = document.createElement("th"); th.textContent = "ots"; tr.appendChild(th); continue; }
        var seq = parseInt(String(c0.textContent || "").trim(), 10);
        var td = document.createElement("td");
        if (isNaN(seq)) { td.textContent = ""; tr.appendChild(td); continue; }
        var cell = cellFor(seq);
        td.className = cell.cls; td.textContent = cell.text; tr.appendChild(td); done++;
      }
      return done;
    }
    // ---- the section
    function rowsHtml(rows, denom) {
      return "<div class=wrap><table>" + rows.map(function (r) { return "<tr><td>" + esc(r.label) + "</td><td>" + esc(kofn(r.count, denom)) + "</td></tr>"; }).join("") + "</table></div>";
    }
    function sectionHtml() {
      var lat = latencyRows(ol), lag = lagRows(tl);
      var h = [];
      h.push("<h2 id=\"witness\">witness</h2>");
      h.push("<p class=dim>every entry_id of the published ledger (the withheld lines' digests included) is stamped through the opentimestamps calendars by a tail that reads the ledger and writes only under its own store; the detached files are beside this page under <a href=\"witness/index.json\">witness/</a>, one per line: <code>witness/proofs/&lt;entry_id&gt;.ots</code>. " + esc(WORDS.pending) + ". " + esc(WORDS.proof) + ". " + esc(WORDS.clocks) + ". " + esc(WORDS.source) + ".</p>");
      h.push("<ul>" +
        "<li>lines in the chain: " + esc(n) + "; with a calendar commitment (a stamp accepted by at least 2 calendars; carries no time): " + esc(kofn(commit, n)) + "</li>" +
        "<li>pending at the calendars (receipt, no bitcoin block yet): " + esc(kofn(pending, n)) + "</li>" +
        "<li>with a bitcoin attestation recorded by the tail after two explorers agreed (a proof a stranger can check): " + esc(kofn(bitcoin, n)) + "</li>" +
        "<li>explorers disagreed on the header: " + esc(kofn(disagree, n)) + "; attestation did not match the block: " + esc(kofn(mismatch, n)) + "; last stamp attempt failed: " + esc(kofn(failed, n)) + "; no record in the store: " + esc(kofn(noRec, n)) + "</li>" +
        "<li>store entries whose digest is no longer in the chain (orphans; a rewrite or a re-chain leaves these): " + esc(kofn(orph.count || 0, orph.of || 0)) + (orph.entries && orph.entries.length ? " (seqs " + esc(orph.entries.slice(0, 40).map(function (o) { return o.seq; }).join(", ")) + (orph.entries.length > 40 ? " …" : "") + ")" : "") + "; seqs with more than one entry_id in the store: " + esc(dup.count || 0) + "</li>" +
        "</ul>");
      h.push("<p class=dim>outside latency — block header time minus the line's own timestamp, of " + esc(n) + " lines (the one number here the lab's clock does not touch; buckets under 6 h carry the miner's slop" + (ol.negative ? "; " + esc(kofn(ol.negative, ol.attested || 0)) + " attested lines have a block time before their own timestamp, a clock question printed, not hidden" : "") + "):</p>");
      h.push(rowsHtml(lat, n));
      h.push("<p class=dim>tail lag — seen_at minus the line's own timestamp, of the " + esc(tl.of || 0) + " lines with a record and both clocks: process evidence (how long after the write the tail saw the line), both clocks the lab's or the gate's, not outside evidence:</p>");
      h.push(rowsHtml(lag, tl.of || 0));
      h.push("<p class=dim>" + esc(retroText()) + " stamp batches: " + esc(sb.count || 0) + " for " + esc(sb.of_lines || 0) + " lines (one batch is one request's round trip by the tail's own clock: a number about the lab's process, never about the proofs).</p>");
      h.push("<p id=\"witness-note\" class=dim>" + esc(noteText()) + "</p>");
      h.push("<p class=dim>" + esc(WORDS.replace) + ". the store's own derived files are beside the proofs: <a href=\"witness/index.json\">index.json</a> (latest state per entry_id), <a href=\"witness/REPORT.md\">REPORT.md</a> (every count with its denominator, as the tail wrote it at " + esc(c.store_written_at || "-") + " over " + esc(c.store_lines_in_chain === undefined || c.store_lines_in_chain === null ? "-" : c.store_lines_in_chain) + " lines in the chain; this page's counts are over the " + esc(n) + " lines it carries), <a href=\"witness/runs.jsonl\">runs.jsonl</a>, <a href=\"witness/seal.jsonl\">seal.jsonl</a> and <a href=\"witness/WITNESS_PREREG.md\">WITNESS_PREREG.md</a> (the kill gate, sealed by its sha256 in seal.jsonl line 1 and stamped as <code>witness/proofs/prereg_&lt;sha256&gt;.ots</code>). the lab's own tail wrote the store; the proofs are what a stranger checks.</p>");
      h.push("<p class=dim>pin one line without this page (node 20+, <code>npm install opentimestamps@0.4.9</code>):<br><code>curl -sO " + esc(FILES_URL) + "proofs/&lt;entry_id&gt;.ots; node node_modules/opentimestamps/ots-cli.js verify -i -d &lt;entry_id&gt; &lt;entry_id&gt;.ots</code><br>a pending file prints \"Pending confirmation in Bitcoin blockchain\" from each calendar and pins nothing; a proof prints the bitcoin block that attests the digest. the block's header time: <code>https://mempool.space/api/block-height/&lt;H&gt;</code> then <code>/block/&lt;hash&gt;</code>, and the same at blockstream.info. this page's witness code is <a href=\"witness.js\">witness.js</a>, beside <a href=\"verify.js\">verify.js</a>.</p>");
      return h.join("\n");
    }
    function mount(ctx) {
      if (typeof document === "undefined") return { tile: false, section: false, rows: 0 };
      var out = { tile: false, section: false, rows: 0 };
      var tiles = byId("tiles");
      if (tiles) { tileEl = document.createElement("div"); tileEl.className = "tile"; tileEl.id = "witness-tile"; tileEl.innerHTML = tileHtml(); tiles.appendChild(tileEl); out.tile = true; }
      if (document.getElementsByTagName) {
        var h2s = document.getElementsByTagName("h2"), target = null;
        for (var i = 0; i < h2s.length; i++) if (String(h2s[i].textContent || "").trim() === "prove a message") { target = h2s[i]; break; }
        if (target && target.parentNode) {
          var sec = document.createElement("div"); sec.id = "witness-section"; sec.innerHTML = sectionHtml();
          target.parentNode.insertBefore(sec, target); out.section = true;
        }
      }
      out.rows = decorateLines(ctx);
      return out;
    }
    return { counts: c, by_entry: by, prereg_sha256: sha, tileHtml: tileHtml, noteText: noteText, windowText: windowText, retroText: retroText,
             foundLines: foundLines, sectionHtml: sectionHtml, indexChain: indexChain, cellFor: cellFor, decorateLines: decorateLines, mount: mount };
  }

  return { page: page, fmtDur: fmtDur, stateOf: stateOf, cellText: cellText, cellClass: cellClass, latencyRows: latencyRows, lagRows: lagRows, WORDS: WORDS };
});

#!/usr/bin/env node
/* check_ledger.cjs — re-derive darkflobi's gate ledger from its bytes with verify.js (node 20+, no packages).

     node check_ledger.cjs ledger.jsonl
     node check_ledger.cjs ledger.jsonl --expect-head <sha256> --anchor "gate anchor 2026-11-02 head <64 hex> seq 316" --json

   Exit 0: the chain re-derives, the head matches --expect-head when given, every --anchor given and every anchor
   line the ledger itself carries is in the chain. Exit 1 otherwise, 2 on usage or a read error. */
"use strict";
const fs = require("fs");
const path = require("path");
const V = require(path.join(__dirname, "verify.js"));

function usage() {
  console.error("usage: node check_ledger.cjs <ledger.jsonl> [--withheld withheld.json] [--expect-head <sha256>] [--anchor '<gate anchor ...>']... [--json]");
  process.exit(2);
}

(async () => {
  const args = process.argv.slice(2);
  let file = null, expect = null, asJson = false, withheldFile = null;
  const anchors = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--expect-head") expect = args[++i];
    else if (args[i] === "--anchor") anchors.push(args[++i]);
    else if (args[i] === "--withheld") withheldFile = args[++i];
    else if (args[i] === "--json") asJson = true;
    else if (!file) file = args[i];
    else usage();
  }
  if (!file || (expect !== null && !/^[0-9a-f]{64}$/.test(expect || ""))) usage();
  const u8 = new Uint8Array(fs.readFileSync(file));
  const withheld = withheldFile ? JSON.parse(fs.readFileSync(withheldFile, "utf8")) : [];
  if (!Array.isArray(withheld)) usage();
  const r = await V.verifyBytes(u8, { withheld });
  const kinds = {};
  for (const it of r.entries) kinds[it.entry.kind] = (kinds[it.entry.kind] || 0) + 1;
  const given = anchors.map(t => {
    const a = V.parseAnchorText(String(t).replace(/\s+$/, ""));
    return a ? V.checkAnchor(r.entries, a, withheld) : { text: t, ok: false, problem: "not the anchor form" };
  });
  const own = r.entries.filter(it => it.entry.kind === "anchor").map(it =>
    V.checkAnchor(r.entries, { date: it.entry.anchor_date, head: it.entry.anchor_head, seq: it.entry.anchor_head_seq }, withheld));
  const cov = V.coverage(r.entries, withheld, own.map(a => a.seq));
  const headMatches = expect === null ? null : r.head === expect;
  const ok = r.ok && headMatches !== false && given.every(a => a.ok) && own.every(a => a.ok);
  const out = {
    file, lines: r.entries.length, kinds, head: r.head, file_sha256: r.file_sha256, problems: r.problems,
    withheld: withheld.length, bridged: r.bridged, coverage: cov,
    expect_head: expect, head_matches: headMatches, anchors_given: given,
    anchors_in_ledger: own.map(a => ({ date: a.date, seq: a.seq, ok: a.ok })), ok
  };
  if (asJson) console.log(JSON.stringify(out));
  else {
    console.log(`lines ${out.lines} published  kinds ${JSON.stringify(kinds)}${withheld.length ? `  withheld ${withheld.length} (bridged ${r.bridged.length})` : ""}`);
    console.log(`head  ${r.head}`);
    console.log(`file  sha256 ${r.file_sha256}`);
    if (expect !== null) console.log(`expected head ${expect}: ${headMatches ? "matches" : "DOES NOT MATCH"}`);
    for (const p of r.problems) console.log(`PROBLEM line ${p.line} seq ${p.seq === undefined ? "-" : p.seq}: ${p.problem || (p.problems || []).join("; ")}`);
    for (const a of given) console.log(`anchor ${a.date || "?"} seq ${a.seq || "?"}: ${a.ok ? "in chain" : "NOT IN CHAIN"}${a.problem ? " (" + a.problem + ")" : ""}`);
    console.log(`anchor lines in the ledger: ${own.length}, in chain ${own.filter(a => a.ok).length}`);
    console.log(`pinned by a posted head: ${cov.pinned} lines; not pinned because of a withheld line: ${cov.unpinned_by_withholding}${cov.unpinned_ranges.length ? " (seqs " + cov.unpinned_ranges.map(x => x[0] === x[1] ? String(x[0]) : x[0] + "-" + x[1]).join(", ") + ")" : ""}; after the last posted head: ${cov.after_last_anchor}`);
    console.log(ok ? "re-derives" : "DOES NOT RE-DERIVE");
  }
  process.exit(ok ? 0 : 1);
})().catch(e => { console.error("error: " + (e && e.message ? e.message : e)); process.exit(2); });

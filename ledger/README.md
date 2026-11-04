# darkflobi · ledger

the gate's ledger (`darkflobi.gate/entry/v0`) as bytes, and a page that re-derives it in the browser.

- `ledger.jsonl` — the log as the gate wrote it (a header line, then chained lines; LF), every published
  line byte for byte.
- `withheld.json` — the lines left out (seq, digest, kind, class, day, reason), if any.
- `index.html` — opens anywhere (file or web), re-derives every digest, checks the posted anchors against the
  chain, and lets you break a copy to see what the digests catch.
- `verify.js` — the verifier, a second implementation of the gate's ledger.py chain check (browser and node);
  `check_ledger.cjs` — the node command.
- `STATS.json` — counts at build time; the page computes its own from the bytes.

check it without the page (node 20+, no packages):

    node check_ledger.cjs ledger.jsonl --withheld withheld.json --anchor "gate anchor <day> head <64 hex> seq <n>"

the anchor texts are the posts at https://t.me/DFledger, one per day at 00:00 new york time.

withheld lines: 4 of 336 (seqs 129, 131, 136, 138). each is an operator-dm line whose numeral-token field carried a
ten-digit identifier typed in a message; the ledger's header promises never message text or chat ids, and
the publisher keeps that promise where the gate's own field would not. a withheld line's digest is in
`withheld.json` (re-derived by the lab over the full file, not by you); the chain is bridged through it,
and the page prints exactly which published lines that leaves unpinned for a stranger.

what the lines hold: one line per outbound attempt and per gate event; hashes, verdicts and counts; never
message text or chat ids (the header's `certifies` sentence). destinations are salted hashes. what the page
cannot show: that every send got a line (the census at the window's close reads the chats for that), or what
any message said. the gate runs in shadow: every line reads `required_present: false`; a would-refuse is what
it would have blocked, and it blocked nothing. nothing here is signed and nothing here says a post is true.

built 2026-11-04T03:28:32Z from head `afd04344e6d37d8d` (336 lines, 10 anchors through 2026-11-02); the gate's instrument pin `3437340b9e944644372e41d9fcab735d47e20fa0`.

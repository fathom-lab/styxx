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
- `SNAPSHOTS.jsonl` — one line per published snapshot (built_at, lines, head, digests, the commit it
  extends). snapshots are append-only: each commit's `ledger.jsonl` extends the previous one byte for byte,
  so `git log -p ledger/ledger.jsonl` on this branch shows additions only; the builder refuses a rewrite.
- `anchors/<date>.txt` — the exact bytes of each day's post at t.me/DFledger; `anchors/<date>.txt.ots` — its
  OpenTimestamps proof (the file's sha256, committed into Bitcoin by the public calendars);
  `anchors/index.json` — each proof's state (pending, or the Bitcoin block height). a proof attests that the
  bytes existed before that block: nothing the lab controls can move that. it does not attest where the
  post appeared or that the head covers every send, and a stamp made on a later day for an earlier post
  shows only that the post existed by the stamping day.

check it without the page (node 20+, no packages):

    node check_ledger.cjs ledger.jsonl --withheld withheld.json --anchor "gate anchor <day> head <64 hex> seq <n>"

the anchor texts are the posts at https://t.me/DFledger, one per day at 00:00 new york time.

withheld lines: 5 of 410 (seqs 129, 131, 136, 138, 351). each is an operator-dm line whose numeral-token field carried a
ten-digit identifier typed in a message; the ledger's header promises never message text or chat ids, and
the publisher keeps that promise where the gate's own field would not. a withheld line's digest is in
`withheld.json` (re-derived by the lab over the full file, not by you); the chain is bridged through it,
and the page prints exactly which published lines that leaves unpinned for a stranger.

what the lines hold: one line per outbound attempt and per gate event; hashes, verdicts and counts; never
message text or chat ids (the header's `certifies` sentence). destinations are salted hashes. what the page
cannot show: that every send got a line (the census at the window's close reads the chats for that), or what
any message said. the gate runs in shadow: every line reads `required_present: false`; a would-refuse is what
it would have blocked, and it blocked nothing. nothing here is signed and nothing here says a post is true.

built 2026-11-04T13:42:49Z from head `25db4c719300d5ce` (410 lines, 11 anchors through 2026-11-03); the gate's instrument pin `3437340b9e944644372e41d9fcab735d47e20fa0`.

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
- `public_texts/<sent_sha256>.txt` — the released bytes of the allow-listed public sends (the public group
  and the anchor channel; never the canary chat, never the operator's dm), each file named by its own
  sha256, which is the line's `sent_sha256`; `public_texts/index.json` — the rule, and per text its seq, day,
  verdict and form; the texts withheld, the lines not re-derivable, the public lines off the allow-list.
- `witness/proofs/<entry_id>.ots` — one per line, the OpenTimestamps file for that line's digest (pending: a
  calendar's receipt, no time; attested: a proof that the digest existed before a bitcoin block);
  `witness/index.json` — the tail's latest state per entry_id (no timestamp for a withheld line);
  `witness/REPORT.md` — its counts with denominators; `witness/seal.jsonl`, `witness/WITNESS_PREREG.md` — the
  preregistered kill gate and its seal; `witness/runs.jsonl` — every run of the tail.
- `brains/index.html` — the gateway's `subject_digest` on the send lines: what it covers, the counts, the
  digest changes and the register; `brains/brains.json` — the counts, digests, timestamps and seqs;
  `brains/register.jsonl` — the lab's register of declared digests, append-only.

check it without the page (node 20+, no packages):

    node check_ledger.cjs ledger.jsonl --withheld withheld.json --anchor "gate anchor <day> head <64 hex> seq <n>"

the anchor texts are the posts at https://t.me/DFledger, one per day at 00:00 new york time.

withheld lines: 5 of 792 (seqs 129, 131, 136, 138, 351). each is an operator-dm line whose numeral-token field carried a
ten-digit identifier typed in a message; the ledger's header promises never message text or chat ids, and
the publisher keeps that promise where the gate's own field would not. a withheld line's digest is in
`withheld.json` (re-derived by the lab over the full file, not by you); the chain is bridged through it,
and the page prints exactly which published lines that leaves unpinned for a stranger.

released texts: 46 of 47 allow-listed public lines have their released bytes published under `public_texts/` (47
of 54 public send lines are allow-listed; off the list: 4 to the canary chat, 3 to a chat not in the gate's
config, 0 without released bytes). 1 of 47 withheld by the text guard (seq 36: a drive or home path; listed in
`public_texts/index.json`, published only after a person reads it; 0 so far). 0 of 47 not re-derivable. the
recipe: the bytes the gate kept as the checked text, or the draft, as written or with the gate's trailer
appended, published only when their sha256 equals the line's `sent_sha256` (41 checked, 5 checked+trailer).
texts are append-only: a text once published is not changed or removed by a later build unless the build records
why. a match says these exact bytes left the gate at that second under that verdict; it does not say who typed
them, that they are true, or that a text with no match was never sent; a copy re-rendered by telegram need not
hash to the gate's bytes. the released texts are the agent's bytes as they left the gate, not statements of the
lab; the publisher changes none of them, and a word in one of them is the agent's, not a claim of this page.

per-line witness: 734 of 792 lines carry a bitcoin-attested proof (the block header's time minus the line's own
timestamp, cumulative: within 1 h 0, within 6 h 2, within 24 h 6; over 24 h 728; none yet 58; of 792); 58 of 792
are pending at the calendars (a receipt, no time a stranger can check); 0 of 792 store entries are orphans (a
digest no longer in the chain; a rewrite leaves these); 0 seqs carry more than one entry_id in the store; 768 of
792 lines were stamped retroactively (seen more than 1 h after their own timestamp, on 2026-11-08, 2026-11-09):
their proofs pin them to that stamping day's blocks, not to their own days; stamp batches 5 for 792 lines (the
tail's own clock, the lab's word). the kill gate is `witness/WITNESS_PREREG.md`, sha256
43f315594728a7f21f9685cf19810b5303628198d0507c6eece8da51e716503b, sealed 2026-11-08T21:28:19.041Z in
`witness/seal.jsonl` line 1; window not open: the standing task is not registered. a proof attests that the
line's digest existed before its block; it does not show that every send got a line, nor the words, nor who
wrote them. the per-line states (status, block, header time) are the tail's store as copied at build: the
publisher checks each proof's opentimestamps magic and the digest it commits to and parses nothing further; the
page parses no proof and asks no explorer. a published proof's bytes are replaced by their upgraded form on
later builds; a proof's commitment never changes.

brains: 28 of 372 published send lines carry the gateway's `subject_digest` (28 of 28 from the earliest one
carrying it, seq 736), 1 distinct digest(s), 0 digest change(s), 0 of 0 undeclared against the register (1
row(s), sha256 3ee94b254026930cc817656471089092c1782b24efb782dcd0bc78b09bbcafbd). the digest is computed by the
gate when it writes the line, over a manifest of the workspace path, the bootstrap files, three config subtrees
and the skills tree the substrate amendment names (provider definitions, base urls and agents.defaults.models
are outside it); it cannot be recomputed off the box. declared_at is the lab's own date. undeclared means no
register row declares the new digest at or before that line, not that anything was wrong.

what the lines hold: one line per outbound attempt and per gate event; hashes, verdicts and counts; never
message text or chat ids (the header's `certifies` sentence). destinations are salted hashes. what the page
cannot show: that every send got a line (the census at the window's close reads the chats for that), or what
any message said. the gate runs in shadow: every line reads `required_present: false`; a would-refuse is what
it would have blocked, and it blocked nothing. nothing here is signed and nothing here says a post is true.

built 2026-11-09T05:30:03Z from head `3f9fe35a1b276ed3` (792 lines, 15 anchors through 2026-11-08); the gate's instrument pin `3437340b9e944644372e41d9fcab735d47e20fa0`.

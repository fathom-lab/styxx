# the agent send ledger — protocol v0

What darkflobi's gate writes, what the page publishes, and what a stranger can check; written so that a second
agent, or a second verifier, can be held to the same answers. This is the format as deployed since 2026-10-24
(`W:gate/ledger.py` on the box; `verify.js` beside this file is a second implementation; `vectors/` holds the cases
both must answer alike). Nothing here is signed; the chain, the posted heads and the Bitcoin proofs are the whole of
what fixes it.

## 1. the file

JSON Lines, UTF-8, LF; no CR byte anywhere. Line 1 is the header; every later non-blank line is an entry.

    header  {"anchor_tz":"America/New_York","certifies":"<a sentence>","created":"<UTC>","schema":"darkflobi.gate/log/v0"}
    entry   {"schema":"darkflobi.gate/entry/v0","seq":<n>,"prev":"<64 hex>","kind":"<kind>","timestamp":"<UTC>","day":"<YYYY-MM-DD>",...,"entry_id":"<64 hex>"}

`timestamp` is `YYYY-MM-DDTHH:MM:SSZ`, the writer's clock: a statement, not evidence of order. `day` is that
timestamp's calendar day in `anchor_tz`. `kind` is one of `send`, `delivered`, `started`, `anchor`, `anchor_anomaly`,
`pin_moved`, `deleted`, `exec_scan`, `gate_not_wired`, `gate_wired`, `lock_recovered`, `head_repaired`; each carries
its own fields beside the chain's. A `send` line carries hashes, verdicts and counts of one outbound attempt
(`checked_sha256`, `sent_sha256`, `dest_hash`, `decision`, `reasons`, `numerals`, ...) and never the message text or
a chat id; the header's `certifies` sentence is that promise, and a publisher checks it (section 6).

## 2. the canonical form

A line's bytes are the compact, key-sorted dump of its object: `{` `}` `,` `:` with no spaces, keys sorted by code
point, strings escaped as JSON requires (`"`, `\`, U+0000-U+001F; the short forms `\b \f \n \r \t`; everything else
raw UTF-8). The admitted value types are null, booleans, integers with |n| ≤ 2^53, strings, arrays, and objects whose
keys are ASCII. For these types the dump is RFC 8785 (JCS); a float or a non-ASCII key is refused at write time, so two
implementations never disagree on number formatting. A line whose bytes are not the canonical dump of its object is
a problem, whatever the object says.

## 3. the chain

    core(entry)      = every key of the entry except entry_id
    entry_id         = sha256( JCS(core) )                        as 64 lowercase hex
    header_digest    = sha256( "darkflobi.gate/log/v0\n" + JCS(header) )
    entry_1.prev     = header_digest
    entry_k.prev     = entry_{k-1}.entry_id
    entry_k.seq      = entry_{k-1}.seq + 1, entry_1.seq = 1
    head             = the last entry's entry_id (or header_digest when there is no entry)

The header's domain string keeps a header from ever passing as an entry. `timestamp` and `day` sit inside the core,
so a rewritten clock breaks the digest. Anyone holding the file recomputes everything: there is no key.

## 4. what a verifier reports

Reading: a CR byte anywhere; an unparsable line; a line that is not an object; a line whose bytes are not canonical;
a missing header on line 1. Chaining, per entry: `schema` is not `darkflobi.gate/entry/v0`; `entry_id` does not
re-derive from the core; `prev` does not name the line before; `seq` is not the one expected; `day` does not re-derive
from `timestamp`; a core that cannot be canonicalised. A log with no problem "re-derives"; the verifier then states
the head. One known difference between implementations is documented: a line with a key written twice is dropped by
the Python reader (the chain then breaks after it) and kept by a JSON.parse reader with its bytes flagged; both report
the log as not re-deriving.

## 5. the anchor

At 00:00 in `anchor_tz` the writer posts, to a public place outside the box, the exact text

    gate anchor <YYYY-MM-DD> head <64 hex> seq <n>

where `<n>` is the seq of the last line of that day and `<64 hex>` its `entry_id`, and then writes an `anchor` line
(`anchor_date`, `anchor_head`, `anchor_head_seq`, `anchor_text_sha256`, `late`). A stranger holding the posted text
checks: the entry at `<n>` carries that digest; no line whose `day` is that date sits after `<n>`; the ledger's own
anchor line for that date agrees. The post is also stamped: its bytes' sha256 go to the OpenTimestamps calendars,
which commit them into a Bitcoin block; the proof (`anchors/<date>.txt.ots`) attests that those bytes existed before
that block, and nothing the writer controls can move that. A proof made on a later day for an earlier post shows only
that the post existed by the stamping day.

## 5b. the per-line witness

When the publisher runs with `--witness <store>` (section 6), every line of the ledger has its own OpenTimestamps
file beside the page: `witness/proofs/<entry_id>.ots`, one per `entry_id` (the sha256 of the line's canonical
core, section 3), the withheld lines' digests included. A tail on the box (`wt/_witness_2026_11_08/witness.cjs`)
reads the ledger — complete LF-terminated lines only, never its lock — re-derives the chain with `verify.js`,
and sends every entry_id not yet in its store, as one merkle batch, to the public OpenTimestamps aggregators; at
least two calendars must accept it. Later runs ask the calendars for the Bitcoin block their commitment landed in
and record it only when two block explorers (blockstream.info, mempool.space) return the same hash, merkle root
and header time for that block.

Two words, kept apart. A PENDING file is a calendar's receipt: a url and a commitment, and no time a stranger can
check. A PROOF is the file once it holds a Bitcoin attestation, and it attests exactly this: the line's digest
existed before that block. The block header's `time` is the one clock in this folder the lab does not run; it is
the miner's, within about two hours of consensus wall time, so a latency under six hours carries that slop and a
negative latency can appear for an honest line (it is printed, never hidden). What a proof does not show: that
every send got a line (section 7), the words of any message, who wrote the line, or that the line is true.

Two clocks, named. `seen_at` and `stamped_at` in the store are the tail's own clock: the lab's word, process
evidence, labelled so wherever they are printed. The line's `timestamp` is the gate's statement (section 1).
`block_time` is the block header's. The number the page's sentence rests on is `block_time − timestamp`
("outside latency"), as counts with denominators: within 1 h, 6 h, 24 h, over 24 h, none yet, of n. The tail's own
`stamped_at − seen_at` ("stamp latency") is one request's round trip and never enters a sentence about the proofs.
Lines the tail saw more than an hour after their own timestamp were stamped retroactively (the opening batch of
2026-11-08 stamped every line written since 2026-10-24); a proof for one of them pins its digest to a block at or
after the stamping (that batch's block, 975101, has header time 2026-11-08T21:41:26Z: the same UTC day, 39 min
52 s after the stamp), not to the line's own day, and the page prints that count.

What is published under `witness/`: `proofs/<entry_id>.ots` for every entry of the store's index (bytes copied;
a stray file in the store that the index does not name is counted, not copied); `index.json`, the tail's latest
state per entry_id — status, calendars, block, block time, the tail's clocks — and, for a withheld seq, no clock
finer than the UTC day on that seq's record: `line_timestamp` blanked and `seen_at`, `stamped_at`, `upgraded_at`
cut to the day, in `index.json` and in the page's embedded document alike (the published ledger shows a withheld
line's digest and day only, and the publisher keeps that here; the seqs are listed under `publisher`). One
exception, stated: `runs.jsonl` is the tail's run log and is copied whole, and a run that stamped a single line
names, by the tail's clock, a moment within the batch window of that line's write; the publisher does not edit
the tail's log. `REPORT.md` and `runs.jsonl` are copied as the tail wrote them, so their denominators are the
chain as the tail last read it, and the page's are the lines the page carries; the page names both moments.
Before copying, the publisher checks every proof's OpenTimestamps magic and the digest it commits to (the 32
bytes after the version and hash-op bytes) against its file name — the entry_id, or the sealed sha256 for the
prereg's own proof — and refuses the build on a mismatch; it parses nothing further of a proof. `seal.jsonl`
and `WITNESS_PREREG.md`, the preregistered kill gate for this layer (three bars read at the close of day 9 of the
opening window) and its seal: line 1 of `seal.jsonl` names the file's sha256, written the moment the tail opened
the store, and the publisher refuses the build when the file it would publish does not hash to that
sealed digest or the index disagrees with the seal; the sealed digest is itself stamped as
`proofs/prereg_<sha256>.ots`. The replacement rule: a proof's commitment never changes, but its bytes grow when a
calendar's Bitcoin attestation is added, so a later build REPLACES a published `.ots` with the store's longer form
(counted as `replaced` in `STATS.json`); nothing under `proofs/` is deleted, and a published proof the store no
longer names stays and is counted. A store entry whose digest is not in the chain is an ORPHAN — what a rewrite or a
re-chain leaves behind — and is published as a count with its list, never hidden, and never refuses the build; so
is a seq that carries more than one entry_id in the store.

What the page re-derives and what it does not: the chain, every digest and every chain count are computed in
the browser from the embedded bytes; the witness states (status, block, header time per entry_id) are the tail's
store as copied at build — the page parses no proof and asks no explorer, and says so in its opening sentence
when `--witness` is on; the proof file beside the page is what a stranger checks.

The counts carry their denominators in `STATS.json` and `SNAPSHOTS.jsonl` (`witness`: lines with a commitment, pending,
with a Bitcoin attestation, explorers disagreed, mismatch, stamp failed, no record, orphans of store entries, seqs
with more than one entry_id, outside latency, tail lag, stamp batches) and on the page (the tile, the `ots` column of
the lines table, the witness section, and the two witness lines under a FOUND message). The kill-gate window opens
on the standing task's opening run; until the operator registers it the page says so (`--witness-window-since` names
day 1 once it has). A stranger pins one line without the page: take its `entry_id` from `ledger.jsonl` (or a
withheld digest from `withheld.json`), fetch `witness/proofs/<entry_id>.ots`, and run
`node ots-cli.js verify -i -d <entry_id> <entry_id>.ots` with javascript-opentimestamps 0.4.9 (or `ots verify -d`
with the python client); a pending file prints "Pending confirmation in Bitcoin blockchain" and pins nothing, a
proof prints the block, and the block's header time is at `https://mempool.space/api/block/<hash>` and
`https://blockstream.info/api/block/<hash>` (`/block-height/<H>` gives the hash). The store was written by the
lab's tail; the proofs are what the stranger checks.

## 6. publication

- `ledger.jsonl`: every published line byte for byte, in order, with the header.
- `withheld.json`: lines left out, each as `{seq, entry_id, kind, dest_class, day, fields_flagged, reason}`. A
  publisher withholds a line when a string in it carries an identifier's shape (whitespace, `@`, a drive or home
  path, a run of nine digits) outside the header's `certifies` sentence; a line with a field the publisher does not
  know refuses the whole build until a person has read it. A verifier given the list bridges the chain through a
  withheld seq on its published digest: `prev` of the next published line must equal that digest, and the seq
  advances. The digest of a withheld line is the lab's word, not the reader's derivation.
- Coverage: a published line is pinned for the reader when every line between it and the first posted head at or
  after it is published; a withheld line in between severs that, and the lines it leaves unpinned are counted and
  listed (`coverage`), never hidden.
- `SNAPSHOTS.jsonl`: one line per publication (`built_at`, `lines`, `head`, digests, `extends_commit`,
  `extends`, `rewrite_reason`). Publications are append-only: a snapshot must byte-extend the last one, and may not
  publish a seq withheld before; a publisher who must do otherwise records `extends: false` with the reason. On the
  exhibit branch `git log -p ledger/ledger.jsonl` therefore shows additions only, except where a line says why not.
- `anchors/<date>.txt`, `anchors/<date>.txt.ots`, `anchors/index.json`: the posts, their proofs, their state.
- `witness/` (when the publisher runs with `--witness <store>`): the per-line witness, section 5b — one
  OpenTimestamps file per entry_id under `witness/proofs/`, the tail's index, report and runs, the sealed kill-gate
  preregistration; counts with denominators in `STATS.json` and `SNAPSHOTS.jsonl` under `witness`.
- `public_texts/<sent_sha256>.txt`, `public_texts/index.json` (when the publisher runs with `--public-texts`): the
  released bytes of the allow-listed public sends. The allow-list rule: a text is published only for a line with
  `kind: send`, `action: send`, `dest_class: public`, a `sent_sha256`, and a `dest_hash` in the gate's
  `config.json` `group_hashes` or equal to its `anchor_channel_hash`; never the canary chat, never the operator's
  dm, never a destination absent from the config (those lines stay hash-only, and the index counts them beside the
  published ones: `off_allow_list`). The recipe: the candidates are the gate's own copies
  `sources/<checked_sha256>.md`, `sources/<draft_sha256>.md` and `sources/<draft_sha256>.draft.md`, each taken as
  written or with the gate's trailer appended (the bytes with trailing newlines removed, two newlines, then
  `[gate: b of n numerals bound · ledger NNNN]`, as the gate appends it); a text is published only when the
  sha256 of that form equals the line's `sent_sha256`, so each file's name is the digest of its own bytes, and no
  byte is added or removed (no trailing newline). Only the single-trailer form is re-derived: the gate joins several
  marks with a newline, and a send with more than one mark, like a sworn source whose prepared footer was split
  (the stripped text then exists in no file the gate keeps), re-derives from nothing and is listed. A line whose
  candidates do not re-derive is listed in the index as `unreconstructable` with its seq and its reason and is not
  published — except a text an earlier build already published, which is carried forward unchanged (form
  `carried_from_previous_build`, its bytes re-checked against `sent_sha256`), since removing it would be a rewrite.
  Bytes that re-derive but are not valid UTF-8 (a lone surrogate, which the gate keeps and hashes) are listed the
  same way and never written. A third class sits beside the canary chat and the unlisted destinations under
  `off_allow_list`: `no_sent_bytes`, a line to an allow-listed destination whose `action` is not `send` or that
  carries no `sent_sha256`. The text guard: a text holding a run of nine or more digits outside a hex digest (a
  digest here is exactly 64, 40 or 32 hex characters with at least one letter among them; an all-digit run is not
  one, and digits glued to a digest are seen), nine or more digits written with spaces, dashes or dots between
  groups (dates and clock times are set aside before that rule; zero-width characters are removed before both digit
  rules), a drive or home path (a drive letter, `~/`, `%VAR%\`, a `\\server\` path, `/users/` or `/home/` in any
  case), an email-shaped token (a letter in the domain: `name@7.49.0` is a version, not an address), a
  phone-shaped token (`+` then a digit then seven or more digits, spaces, dashes, dots or parentheses) or a
  telegram invite link (`t.me/+…`, `t.me/joinchat/…`) is withheld and listed (`withheld`: seq, `sent_sha256`, role,
  day, reasons); whitespace and a bare `@` (a public handle) are not reasons here. A withheld text is published only
  after a person has read it (`--allow-text <sent_sha256>`), and the index records that under
  `allowed_by_operator`; the publisher takes that list from the committed index only (`git show HEAD:`), never from
  a file on disk, so a hand-written `index.json` cannot stand in for the operator's reading, and in a worktree that
  is not a checkout the reading lasts for the build it was given to. A published text that newly trips the guard
  (the rules grew) refuses the build until the operator reads it (`--allow-text`) or drops it (`--allow-rewrite`
  with the reason); files under `public_texts/` that the index does not list refuse the build the same way. The
  counts carry their denominators in `STATS.json` and `SNAPSHOTS.jsonl` (`public_texts`: published, withheld,
  unreconstructable, of the allow-listed lines; allow-listed lines, of the public send lines). Texts are
  append-only like the lines: a text once published is not changed or removed by a later build, and a build that
  must do otherwise says why (`--allow-rewrite`, recorded as `rewrite_reason`). The publisher writes
  `ledger/.gitattributes` (`* -text`) so no checkout converts line endings in the released bytes; a copy that a
  checkout converted before that (CRLF on disk, HEAD's blob equal to the released bytes) is restored from HEAD and
  logged, not counted as a rewrite. The released texts are the agent's bytes as they left the gate, not statements
  of the lab; the publisher changes none of them, and a word inside one of them is the agent's, not a claim of this
  protocol or its page.

## 6b. proving one message

A `send` line carries four digests of the message it records: `sent_sha256` (the bytes delivered), `checked_sha256`
(the text the gate checked), `stripped_sha256` (that text with a minted post's footer removed) and `draft_sha256`
(the draft as handed to the gate). A reader holding a message's text hashes it (UTF-8, LF) and looks for a send line
carrying that digest; the line's `seq`, `timestamp`, `decision` and `numerals` are then that message's record, the
day's posted head fixes the line in the chain, and the head's OpenTimestamps proof fixes it in time. The gate's
trailer, `[gate: b of n numerals bound · ledger NNNN]`, names the line's seq in the message itself. What a match
proves: these exact bytes left the gate at that second under that verdict. What it does not prove: who typed them,
or that they are true. A text that a channel re-rendered (markdown removed, whitespace changed) may not match the
bytes the gate saw; the page tries a few whitespace variants and the trailer-less form, and says which matched.

When the publisher runs with `--public-texts` (section 6), the page also holds the released bytes of the published
allow-listed lines. On a hash match it shows them, re-hashed in the reader's browser against the line's
`sent_sha256`, both digests printed, so the reader sees the bytes and the digest agree by eye; the same check
without the page is `sha256sum public_texts/<sent_sha256>.txt` against the file's own name. When two lines carry
one `sent_sha256` and only one is allow-listed, the text renders once, for the allow-listed line, and the other is
named as off the list. When no digest matches, the page searches the published texts for the nearest one by token
similarity (unicode NFC; markdown markers `* \` # > [ ] ( )` removed and `_` at a token's edge, so `sent_sha256`
keeps its underscore; the gate trailer form removed; whitespace collapsed; Dice over the token multisets;
threshold 0.6, stated on the page) and shows it as "nearest published released text — NOT a hash match", with a
word-level diff between the paste and the released bytes, the line's seq, day and decision, and the day's posted
head and bitcoin proof through the same code as a match. In that block every verdict names its subject: the
released text re-hashes to its own digest, the paste's own digest is printed and no send line carries it, and the
head and bitcoin lines are the nearest line's, not the paste's; nothing in it wears the page's colour for a
verified fact. A paste holding only part of a text can reach the threshold. The caveat, printed under it: a
telegram-rendered copy need not hash to the gate's bytes; the released bytes are what the gate hashed; a near
match proves nothing by itself. It is a search aid that hands the reader the bytes to re-hash, not a verdict. A
miss, near or far, does not show that the agent did not send the text: the line may be withheld, off the
allow-list, not re-derivable, or newer than the page.

## 7. what this protocol does not do

It does not show that every send got a line: that is a census of the destinations against the ledger, done at a
window's close. It does not check the words of any message; the gate's verdicts are about numerals, receipts and a
word list. It does not sign anything, and it does not name who wrote a line beyond the `process` field's own
statement. It fixes bytes and their order, in a place the writer does not control, and lets anyone recompute the
rest.

## 8. conformance

`vectors/vectors.json` holds cases over `vectors/base.jsonl` (a fixed prefix of the published ledger): a transform of
the base (identity, one field rewritten and re-dumped, a dropped line, a dropped line with its digest supplied as
withheld, a CR byte, a duplicated key, a truncated tail) and what a verifier must answer (`ok`, the seqs with problems,
the head, the bridged seqs, the anchor checks). `vectors/run_vectors.cjs` runs them with `verify.js`;
`vectors/run_vectors.py` runs them with the gate's own `ledger.py`. A third implementation is conformant when it
answers every vector as the file says.

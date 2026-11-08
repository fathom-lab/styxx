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

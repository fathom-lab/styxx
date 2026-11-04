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

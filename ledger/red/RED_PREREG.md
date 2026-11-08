# RED_PREREG: the mutation ledger of the reader (THE_PLAN 3c), written before the first batch runs

written 2026-11-08, before any sealed batch; its sha256 goes into STATS.json and into the red chain's started line.
lowercase, terminal-native; every count below is printed with its denominator; failures are printed.

## what runs

- a copy of darkflobi's gate (`gate_copy/`, RED_COPY.md: the house rule, a dummy salt, a config whose every value is
  an hmac of a synthetic id, every piece of copied state removed) writes its own hash-chained ledger under the same
  protocol (`darkflobi.gate/entry/v0`, PROTOCOL.md of the exhibit branch). REQUIRED stays absent: shadow.
- `planter.py` (a script; no agent, no model, no chat, no token, no network but OpenTimestamps for the anchor) builds a
  batch of 20 texts from the catalogue `forms.py` (25 forms in 3 classes), seals it, and hands each text to
  `gate.py check` as a public send to the synthetic group. the gate's own numeral reader (`helpers/numerals.py`
  over the pinned snapshot 3437340b) counts the numerals in each text; the count is `numerals.total` on the line.
- the seal: the batch file `batches/<id>.json` is written first; the text `red batch <id> sha256 <64 hex>` goes
  through the gate BEFORE every text of the batch, so its line precedes them in the chain. a batch file changed
  after sealing fails its sha against the seal line's `checked_sha256`.
- assignment is deterministic: text k of batch i gets form `(20*i + k) mod 25` (round-robin, so 5 batches cover
  every form 4 times) and the instance `sha256("red-ledger-3c/v0:<id>:<k>")[:4] mod len(instances)`.

## what is measured

per planted text, `numerals.total` against the form's expected count:

- counted_by_design (9 forms; expected 1, or 2 for the worded range and the slash fraction): `counted` when
  total == expected, `missed` when total < expected, `extra` when total > expected (the reader counted something
  the planter did not plant; listed).
- excluded_by_design (6 forms; expected 0): `confirmed` when total == 0; otherwise `exclusion_not_held` (listed).
- open_question (10 forms; expected "?"): the total is reported as observed, per form, and never called a miss.
- the seal line: `numerals.total` reported; its 64-hex digest is a designed exclusion (the sha scrub).
- every text's `checked_sha256` must equal sha256 of the text handed; a difference is `hash_mismatch` (printed,
  fatal for the batch's recount).

the five batches today are 100 planted forms + 5 seal lines; the chain also holds its started line(s).

## kill gates

- (a) any red `entry_id`, `checked_sha256` or `sent_sha256` found in darkflobi's published ledger
  (`wt/exhibit/ledger/ledger.jsonl`) → the build is stopped and the event published. checked after every batch.
- (b) 0 misses among counted_by_design forms on 100 or more planted forms → published as a null result with its
  denominator ("0 misses of n counted-by-design plantings"), not as a failure of the instrument.
- (c) the reader counts a seal line's sha fragment as a numeral (seal `numerals.total` > 0) → reported as a
  designed exclusion that did not hold, with the seal text's digest.

## what this cannot say

- nothing about darkflobi's traffic: the texts are synthetic and the chain is the copy's.
- nothing about the live gate's refusals: shadow, synthetic destination, no REQUIRED.
- not that the format is "a protocol" in any sense stronger than: a second chain, written by the same code on the
  same box by the same lab, re-derives under the same verifier and the same ten vectors. a stranger's chain is
  what would earn more; this is not that.
- the open-question totals are not failures: the lab's later reader (certify.py v0.15) ships those forms behind
  flags that are OFF by a recorded decision, or discloses them as unread; the pinned snapshot predates it. a repair
  that moves a count is an amendment with a date, and the count moving is the point.

## what changes nothing here

the page, the stats and the readme are built from the chain and the batch files; the planter never edits a batch
after sealing, and the publisher publishes every line (the header's fixed `certifies` sentence is the copy's
inherited one; the page's first line says what the chain is).

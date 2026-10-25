# The numerals conformance set

What `styxx.certify`'s numeral reader reads, pinned as vectors so a change to the reader is a diff and
a second implementation has something to agree with. Built with the numeral reader v2 (2026-10-25,
`papers/closed-model-frontier/NOTE_numeral_reader_v2_2026_10_25.md`). Every count is in `vectors_v2.json`;
none is written here.

## What is here

| file | what it is |
|---|---|
| `texts.py` | the texts and receipt cases, as data: the product red team's posts, the design's probe texts and cases, the design's red team's reader families and binding cases, and the amendments' cases |
| `gen_vectors.py` | generates the set from `texts.py`, replays it, refuses to re-record an id with a different expectation, writes `vectors_v2.json`; `--check` replays the committed set |
| `vectors_v2.json` | the set: four families, each vector addressed by its id, one digest over the index |

The generated file is never hand-edited. `tests/test_certify_numerals_v2_vectors.py` replays the
committed set with nothing skipped and fails, rather than skips, on any vector that does not reproduce.

## What a vector is

- `reader`: one text; `expect` is `extract_numbers(text)` and `uncovered_spans(text)` under the shipped
  flags, each item minus its `context` (a function of the text). A magnitude row carries its `magnitude`
  block; a form carries its `reason` and `counted`.
- `binding`: one text and its receipts; `expect` is the verdict, its class, the counts, every ledger row's
  `token`, `status` and `receipt_ref` (the receipt named by its index, `receipt[j]:path`), and the counted
  band. The receipt-binding block is stubbed (it reads git and is not part of the reader).
- `v1_reader`, `v1_binding`: the same texts and cases with `V15_READER_V2 = False`. Their expectations were
  recorded from `styxx/certify.py` at `879f6be9` (LF sha256 `c448ded108df0f54…`) before the v2 code
  existed, and the generator refuses to write unless the master-off reader reproduces that recording
  exactly: `provenance.v1_recording` names the commit.

`id = sha256(utf8(jcs({"family": f, "text": t[, "receipts": r]})))`; `set_sha256 = sha256(utf8(jcs(index
minus set_sha256 minus provenance)))`. JCS is `styxx.attestation.jcs`.

## How to consume it

1. Recompute `set_sha256` and compare.
2. For each vector, re-derive its id from its family, text and receipts.
3. Run the family's entrypoint (`extract_numbers` + `uncovered_spans`, or `certify_doc`) with the family's
   flags and compare `expect`, key for key.

## What this set does not say

It is not a measure of recall or precision on posts in the wild; the texts are the probes that were run
before the reader was built. A text absent here is not covered. A later reader records its expectations in
a new `vectors_v3.json`; this file is never edited (a receipt is history too).

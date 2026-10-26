# The numerals conformance set

What `styxx.certify`'s numeral reader reads, pinned as vectors so a change to the reader is a diff and
a second implementation has something to agree with. Built with the numeral reader v2 (2026-10-25,
`papers/closed-model-frontier/NOTE_numeral_reader_v2_2026_10_25.md`). Every count is in the `vectors_*.json`
files; none is written here.

## What is here

| file | what it is |
|---|---|
| `texts.py` | the texts and receipt cases, as data: the product red team's posts, the design's probe texts and cases, the design's red team's reader families and binding cases, and the amendments' cases. Not edited after the sets below were written from it |
| `texts_review.py` | the texts and receipt cases the two reviews of the reader ran (2026-10-25: the correctness review's FB, FM and OK cases with their exchange-shaped receipts; the claims review's probes), appended to `texts.py`'s for the set written after the review fixes |
| `gen_vectors.py` | generates a set from `texts.py` and `texts_review.py` under the flags the module holds, replays it, refuses to re-record an id with a different expectation and to rewrite a committed set; `--check` replays every committed set |
| `vectors_v2.json` | the earliest set, written by the earlier build session of 2026-10-25 under its default (bare magnitudes and form noise on); four families, each vector addressed by its id, one digest over the index |
| `vectors_v2_default.json` | the default of RULE_final 0.1 (bare magnitudes and form noise off), before the review fixes (the accounting forms then rode on `V15_SIGNED_CURRENCY`, the time/id leaf class was narrower); the same texts and cases, the same four families; `reader.flags` names every flag that existed |
| `vectors_v2_default_after_review.json` | the shipped default after the review fixes (`V15_ACCOUNTING_SIGN` off, the wider time/id leaf class); `texts.py`'s texts and cases plus `texts_review.py`'s, the same four families; `reader.flags` names every flag, and the ids are over it |

`vectors_v2.json` and `vectors_v2_default.json` were committed with CRLF line endings (written in text mode
on Windows) and converted to LF once, in d49e9427, because the conformance suite requires LF in every
tracked file under `conformance/`: only CR bytes were removed (the CR-stripped earlier blob is byte-equal to
the later one), and every id, expectation and `set_sha256` is unchanged. Neither set has been edited
otherwise, and neither will be.

The generated file is never hand-edited. `tests/test_certify_numerals_v2_vectors.py` replays the
committed set with nothing skipped and fails, rather than skips, on any vector that does not reproduce.

## What a vector is

- `reader`: one text; `expect` is `extract_numbers(text)` and `uncovered_spans(text)` under the set's
  flags, each item minus its `context` (a function of the text). A magnitude row carries its `magnitude`
  block; a form carries its `reason` and `counted`.
- `binding`: one text and its receipts; `expect` is the verdict, its class, the counts, every ledger row's
  `token`, `status` and `receipt_ref` (the receipt named by its index, `receipt[j]:path`), and the counted
  band. The receipt-binding block is stubbed (it reads git and is not part of the reader).
- `v1_reader`, `v1_binding`: the same texts and cases with `V15_READER_V2 = False`. Their expectations were
  recorded from `styxx/certify.py` at `879f6be9` (LF sha256 `c448ded108df0f54…`) before the v2 code
  existed, and the generator refuses to write unless the master-off reader reproduces that recording
  exactly: `provenance.v1_recording` names the commit. For `vectors_v2_default_after_review.json` the texts of
  `texts_review.py` were recorded the same way from a `git archive` of `879f6be9` (same certify.py bytes, no
  v2 code), and `provenance.v1_recording.parts` lists both recordings.

Three id rules, each set naming its own in `id_rule`:
`vectors_v2_default_after_review.json` uses `id = sha256(utf8(jcs({"family": f, "flags": flags, "text": t[,
"receipts": r]})))`, `flags` being the set's `reader.flags`, every rule flag on or off (a flag that ships off,
like `V15_ACCOUNTING_SIGN`, changes a reading without changing which rules are on);
`vectors_v2_default.json` uses `{"family", "rules", "text"[, "receipts"]}`, `rules` being its `reader.rules`;
`vectors_v2.json` uses `{family, text[, receipts]}`. One text under two configurations is two vectors. `set_sha256 = sha256(utf8(jcs(index
minus set_sha256 minus provenance)))`. JCS is `styxx.attestation.jcs`.

## The configuration a set is read under

`reader.flags` holds every `V15_*` rule flag as the set was written; a consumer sets exactly those (the
master off for the `v1_*` families). `vectors_v2.json` has no `flags` block: read it from `reader.rules`,
every flag it does not list off. A flag a set predates takes the value of the flag it was split from:
`V15_MAGNITUDE_BARE` was split out of `V15_MAGNITUDE` by RULE_final 0.1, and `V15_ACCOUNTING_SIGN` out of
`V15_SIGNED_CURRENCY` by the review fix. So `vectors_v2.json` replays with both on, and
`vectors_v2_default.json` (whose block names `V15_MAGNITUDE_BARE` off) with the accounting forms on, each as
it was written. `gen_vectors.set_flags_of` is that rule in code.

## How to consume it

1. Recompute `set_sha256` and compare.
2. For each vector, re-derive its id from its family, text and receipts.
3. Run the family's entrypoint (`extract_numbers` + `uncovered_spans`, or `certify_doc`) with the set's
   flags (the master off for `v1_*`) and compare `expect`, key for key.

## What this set does not say

It is not a measure of recall or precision on posts in the wild; the texts are the probes that were run
before the reader was built. A text absent here is not covered. A later reader, or another default,
records its expectations in a new file; a committed set is never edited (a receipt is history too).

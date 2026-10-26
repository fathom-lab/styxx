# AMENDMENT — charon reads a sworn line's advisory build flag apart when the sworn.py it names moved

Fathom Lab · 2026-10-25 · Amends `SPEC_charon_v01_2026_09_02.md`, in this directory, which `styxx/charon.py`
pins by its blob sha256 and which is not edited; its ERRATA section is not edited either. Amends, for one
compared value, the status table of `AMENDMENT_charon_verifier_names_its_runtime_2026_10_12.md`, which is not
edited either. Written on the operator's decision of 2026-10-25, quoted below. It is committed alone, ahead of
the code that implements it (`styxx/charon.py`, with `tests/test_charon_build_flag.py`).

## What the spec says today

- **E5** (spec lines 188-189): "`verify` compares the whole core. REPRODUCED is renamed `SAME_LINE` and
  requires every compared field to match; `fields_changed`, `subject_moved` and `receipts_moved` print on
  every row." The status table (spec lines 105-112, as the 2026-10-12 erratum places it) and item 4 of the
  2026-10-12 amendment read it so: `MOVED_VERIFIER` is the same core under a moved verifier, `SKEW` is a
  moved core under a moved verifier, `DRIFT` is a moved core under the same verifier.
- **Item 6 of the 2026-10-12 amendment.** Every committed line is legacy: it reads MOVED_VERIFIER when no
  compared key moved and SKEW `["legacy"]` when one did. Its section "What does not change" says the
  stranger's ferry-log step sees every committed line read MOVED_VERIFIER `["legacy"]`.
- **`counts` is a compared key.** A sworn line's `counts` carries `receipt_check`, the result of
  `styxx.sworn.verify_receipt` over the verdict receipt beside the sidecar, and inside it
  `same_verifier_build`: whether the `verifier.sworn_sha256` the receipt names as its issuer is the
  `sworn.py` installed now. The flag is advisory in `styxx.sworn`: outside the receipt's digest and outside
  its compare. In charon it is inside the compared core.

## What was measured that the spec did not foresee

1. **Fourteen lines carry the flag true.** The committed log has 32 sworn lines with a `receipt_check`. The
   18 written 2026-09-03 (between seq 5 and seq 242) recorded `same_verifier_build` false: their receipts
   were issued by an older `sworn.py` than the one that wrote the line. The 14 written 2026-09-13 to
   2026-09-15 (seq 244 to 257, file lines 245 to 258) recorded it true: the receipt each re-checks names, as
   its issuer, the `styxx.sworn` sha the line's own verifier names (`7e4602472010…`, all 14). That is the
   sha of `styxx/sworn.py` on main at `078b0a16`.
2. **The flag flips on any `sworn.py` edit.** The sworn verdict receipt /v2 step
   (`papers/sworn/NOTE_certifies_correction_2026_10_24.md`) is the earliest `sworn.py` edit since those lines
   were written. Under it each of the 14 re-derives with `counts` moved and nothing else: the flag reads
   false, the receipt still VERIFIES (digest match, verdict reproduces, same schema, same recorded verdict),
   and every other compared key is equal. Under E5 a moved `counts` under a moved verifier is SKEW, so the
   14 read SKEW `["legacy"]`, the stranger's ferry-log step reads FAIL, and the sentence of the 2026-10-12
   amendment quoted above stops holding. The receipts did not move; the instrument did, and the line
   recorded a fact about the instrument inside the part of the line charon compares.

## The operator's decision

On 2026-10-25, asked in chat how the /v2 branch should merge, the operator, Flobi, chose the option that
fixes charon ahead of the merge, marked "(Recommended)". The option read: "A small amendment to charon's
frozen spec: a line whose only moved field is that advisory build flag reads MOVED_VERIFIER when the
verifier moved. The stranger check stays PASS, and both merge together."

## The amendment

1. **One compared value is read apart, under five conditions, all of which hold** (amending E5 and the
   2026-10-12 status table for this value alone). A row reads MOVED_VERIFIER where E5 would read SKEW when:
   - the line is a sworn line, re-derived as a sworn line, and the verifier moved (the fresh verifier
     digest differs from the recorded one);
   - the only compared key that moved is `counts`, and with the fresh `same_verifier_build` put back to the
     recorded value the fresh `counts` equal the recorded `counts` under JCS;
   - the line recorded `same_verifier_build` true and the fresh reading is false;
   - the `styxx.sworn` sha the line's verifier names is not the installed one;
   - the verdict receipt beside the sidecar still names, as its issuer (`verifier.sworn_sha256`), the
     `styxx.sworn` sha the line names.

   Then the flag moved because the installed `sworn.py` is not the build the receipt and the line both name,
   and for no other reason the line can show. The receipt is read only when the four conditions before it
   hold.
2. **Under the same build the flag stays DRIFT.** The rule needs the verifier moved. A flag that moved while
   the verifier did not means the receipt was rewritten in place, and E5 reads that DRIFT, as before.
3. **Nothing else is read apart.** The flag beside any other moved value, in `counts` or in any other
   compared key, reads what E5 reads. So does a flag that moved false to true (a receipt re-issued by the
   installed build), a flag that moved while `styxx.sworn` did not (the verifier moved elsewhere: another
   module, or the runtime), a flag that moved while the receipt names another build (a receipt rewritten in
   place under a moved `sworn.py`), a receipt that is gone or unreadable, and a line of any other kind.
4. **The row says what was read apart.** `verify_log` adds `read_apart` to every row: `[]`, or
   `["counts.receipt_check.same_verifier_build"]` when the rule applied, and null on a row that was not
   re-derived (TAMPER). `fields_changed` still names `counts` on such a row, so a MOVED_VERIFIER row with a
   moved field says why. The report gains `read_apart`, a count per value read apart, beside `by_status`,
   and its note and the `verify` command print it.
5. **The entry is unchanged.** The schemas stay `styxx.charon/log/v1` and `styxx.charon/entry/v1`. The core
   keys, the compared keys, the verifier scheme and every `derive` output are unchanged. No status is added
   or removed. `styxx.stranger`'s ferry-log step, which allows SAME_LINE and MOVED_VERIFIER, needs no edit.

The option's text names the flag and the moved verifier. Items 1 and 3 are narrower in three ways: the
direction (true to false), the module (`styxx.sworn` moved), and the receipt (it still names the line's
build). Each keeps E5's reading for a case where a moved flag can mean the receipt moved. On the committed
log the two readings are the same 14 lines: every row with a moved compared key is one of them, and each
meets all five conditions. That the narrowing stands is the operator's to confirm.

## What was measured

`python -m styxx.charon verify --log papers/charon/charon.log.jsonl --repo . --expect-head
8a0ee53b20cc142ff8d8758bcc580e0bb8970b4293f8d0f7f130d9f056590450`, under CPython 3.12.10 on a Windows
checkout, over the whole log, at `683543be` (the /v2 branch) and on the working tree of the commit that
follows this one (`styxx/charon.py` at LF sha `48af99ce8d3b06dea52a6f1fc549d41e31b952df86b26be249d76c344647b401`):

| | lines | head | SAME_LINE | MOVED_VERIFIER | SKEW | DRIFT | UNRESOLVED | HEAD_MISMATCH | TAMPER | read apart |
|---|---|---|---|---|---|---|---|---|---|---|
| before | 257 | matches | 0 | 243 | 14 | 0 | 0 | 0 | 0 | — |
| after | 257 | matches | 0 | 257 | 0 | 0 | 0 | 0 | 0 | 14 |

- Both runs: `verifier_moved` `["legacy"]` on all 257 rows; by kind capsule-diffgate 2, capsule-oath 10,
  oath-certificate 213, sworn 32; no subject and no receipt set moved; every fresh verdict class equals the
  recorded one; exit 0.
- Row by row, every key but `read_apart` and `verifier_now` is equal before and after on 243 rows. On the 14
  (seq 244 to 257) only `status` moved, SKEW to MOVED_VERIFIER; their `fields_changed` stays `["counts"]`
  and their `read_apart` names the flag. `verifier_now` moved on every row, as any `styxx/charon.py` edit
  moves it, because `styxx.charon` is on every kind's module list.
- `styxx.stranger`'s ferry-log step reads PASS again, and `tests/test_stranger.py` asserts PASS again.

**Charon's own digest in history.** Every committed line names a `styxx.charon` sha inside its `verifier`,
which is inside its `entry_id`: 243 lines name `2c0b24331203…` and 14 name `2e53243bd17d…`, raw-byte shas of
older `charon.py` files. All 257 are legacy, read MOVED_VERIFIER `["legacy"]` whatever `charon.py`'s bytes
are, and are not rewritten. No committed file names the sha of `charon.py` before this change (raw
`cc7a2293…`, LF `5b5bce0d…`) or after it (raw `297180af…`, LF `48af99ce…`). No research capsule is committed.
The snapshots `charon_verify_result.json`, `charon_verify_result_2026_09_29.json` and `index.html` are dated
and are not regenerated.

## What does not change

- The spec text and its errata, and the 2026-10-12 amendment.
- The committed log. No line is rewritten, re-digested or appended.
- Committed capsules, sworn receipts and conformance vectors. The receipts under the 14 lines are not
  re-issued.
- A receipt rewritten in place under the same build reads DRIFT; a moved flag beside any other moved value
  reads what E5 reads.

## What this amendment does not say

- That any of the 14 lines is true. It says the line re-derives with nothing moved but a flag about the
  instrument.
- That the receipt under a read-apart line is the receipt the line was written against. Charon records no
  digest of the verdict receipt. A receipt rewritten in place that names the same build and re-checks the
  same way moves nothing on the line, under this amendment and under E5 alike.
- That `sworn.py` moved for a sound reason. The row says that it moved; the change is in git.
- That the rule reads line endings apart. `verify_receipt` compares raw-byte shas, a legacy line names
  raw-byte shas and a scheme-v1 line LF shas. `sworn.py` is `-text` in `.gitattributes`, so on every
  checkout of this repository the three are one string. Where `sworn.py` carries CRLF bytes, a scheme-v1
  line's LF sha never equals a raw receipt build, the fifth condition is not met, and E5's reading stands;
  on a legacy line, a `sworn.py` whose line endings alone moved reads as a moved `sworn.py`.

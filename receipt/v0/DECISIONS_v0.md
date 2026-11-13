# RECEIPT v0: the three decisions that change a verdict

Status: DRAFT. Taken in draft by the lead under the operator's word of 2026-11-13 ("do what's best"). UNSIGNED: none of the three is in force until the operator signs, and nothing is frozen by this file.

Scope of the counts below: the branch `receipt-v0`, cut from `origin/main` at `6064871b`. "Today's code" is `styxx/sworn.py` at that commit (2,212 lines), cited by line. No committed receipt, certificate, capsule or conformance vector was edited: every number here comes from reading committed files and projecting them in memory with `receipt/v0/to_v0.py`. `tests/test_receipt_v0.py` reads the fenced ledgers in this file and fails if any of them is not exact.

## The census the lists are taken over

A "committed verdict receipt" here is a tracked `.json` file outside `conformance/` whose top-level `schema` begins `styxx.sworn.verdict-receipt/` and which has a `spans` list.

<!-- census:begin -->
```
committed verdict receipts = 79
at /v1 = 72
at /v2 = 7
named *.sworn-receipt.json = 74
span rows = 2148
span verdicts changed = 0
document verdicts changed = 3
claims bound (document on the branch) = 62
claims unbound (document not on the branch) = 17
receipts embedded in committed capsules = 1
verdicts changed in them = 0
```
<!-- census:end -->

The kernel map of 2026-11-13 counts 74 (67 at /v1, 7 at /v2). That is the count of files named `*.sworn-receipt.json`. Five more files carry the same schema under another name (`papers/sworn/measurement/dryrun/SYN-01`, `SYN-02`, `SYN-03`, `SYNX-01`, `SYNX-02`, each `.receipt.json`, all /v1). All 79 are in every list below. Each of the 79 re-derives its own committed digest before it is projected; the projector refuses one that does not.

One more verdict receipt sits inside a committed capsule rather than in a `.json` file: `papers/sworn/CANNED_harness_junit_2026_09_05.capsule.html` embeds a /v1 receipt with 5 HELD rows. It is outside the 79 by the definition above and is checked separately: none of D1, D2 or D3 changes a verdict in it (the two lines at the end of the census block, kept exact by the test).

Of the 2,148 span rows, 2,085 name a `path:` receipt, 3 a `prereg:` receipt and 60 an `rN` receipt. v0 resolves `rN` only, so 75 of the 79 project to the legacy profile (section 9 of the text) and 4 to the `rn-inline` profile.

## Every verdict that changes, by name

<!-- verdict-changes:begin -->
```
document papers/sworn/sworn_action_sample.UNRESOLVED.sworn-receipt.json SWORN-HELD -> SWORN-UNCHECKED
document papers/sworn/sworn_action_sample_2026_09_06.UNRESOLVED.sworn-receipt.json SWORN-HELD -> SWORN-UNCHECKED
document papers/sworn/sworn_action_sample_2026_10_24.UNRESOLVED.sworn-receipt.json SWORN-HELD -> SWORN-UNCHECKED
```
<!-- verdict-changes:end -->

All three are D1. No committed span verdict changes under any of the three decisions.

How this was checked. The test projects each committed receipt's own rows (79 receipts, 2,148 rows). Separately, and outside the test because it takes minutes and needs the repository's history, today's checker was re-run over the documents behind 60 of the 79 at the commit each receipt names (2,061 spans): all 60 gave the committed span verdicts and the same v0 digest as the committed receipt's projection. The other 19 were not re-run: 17 have no document on the branch and 2 are inline documents whose manifest is not beside them. The output is in the writer's notes, not in the repository.

---

## D1. A document in which nothing was checked must not read as held

**Today.** `verify()` sets the document verdict at lines 1831 to 1838: document-level MALFORMED gives `SWORN-FAILED`; zero spans gives `UNSWORN`; otherwise `SWORN-HELD` whenever `counts["FAILED"] == 0 and counts["MALFORMED"] == 0` (line 1835). UNRESOLVED is never consulted. The CLI headline adds a warning line when HELD is 0 (lines 2044 to 2047) or when some spans are UNRESOLVED (lines 2048 to 2050); the warning is printed text, outside the receipt. The code's own comment (lines 2019 to 2023) records that the hole is reachable by the author: a manifest rung string the checker does not know makes every `rN` span UNRESOLVED before any receipt is opened, so a sentence its receipt contradicts moves from `SWORN-FAILED` to `SWORN-HELD` by one string.

**Options.**

- A. Keep the word, keep the warning. Changes nothing; every adopter learns that "nothing checked" is green.
- B. A distinct document verdict when no span reached HELD and none FAILED or MALFORMED; `SWORN-HELD` requires at least one HELD span. A document with some HELD and some UNRESOLVED spans stays `SWORN-HELD` with the UNRESOLVED count beside it.
- C. Any UNRESOLVED span denies `SWORN-HELD`. Closes more, and changes every honest document checked by someone who lacks one receipt.

**Choice: B.** The new word is `SWORN-UNCHECKED`. The rule, in order: document-level MALFORMED gives `SWORN-FAILED`; no span gives `UNSWORN`; any FAILED or MALFORMED span gives `SWORN-FAILED`; no HELD span gives `SWORN-UNCHECKED`; otherwise `SWORN-HELD`. `unresolved` stays a key of the digested core beside `document_verdict` (it already is one of the thirteen), so a receipt cannot show the verdict without the count.

**What B does not close.** A document with one HELD span and any number of UNRESOLVED spans still reads `SWORN-HELD`. The count is beside it; the word is the same. Option C is the stricter rule and is the operator's to prefer.

**Committed receipts whose verdict changes: 3**, the three named in the ledger above. Each has one span, UNRESOLVED `manifest_id_missing`, and reads `SWORN-HELD` today.

**In the conformance material** (not committed receipts; inputs to a checker, counted so the next step knows what moves): see the conformance ledger at the end.

## D2. A rounding floor for the numeric kind

**Today.** `_check_numeric` quantises the receipt's number to the fractional digits the sentence prints, half-even, and returns HELD when the two canonical strings are equal (lines 1459 to 1487; the comparison is line 1485). There is no floor. The comment at lines 1465 to 1476 states the hole: a receipt of `0.4211` against "the A-share is 0." is HELD. Lines 2029 to 2043 detect the case for a CLI warning only (`_erased_span`); the verdict is deliberately unchanged and the lab's note of 2026-09-05 records the floor as operator-gated.

**Is a proposal stated precisely in the lab's record?** Not as a verdict rule. The note says only "Rounding has no floor ... Verdict deliberately unchanged". The code states a line precisely, for the warning: "a non-zero receipt that rounds to zero" (lines 1473 to 1474), implemented as `receipt` non-zero and `receipt_rounded` zero (line 2036). No file proposes which verdict that case should get. So the options are stated here and one is picked by the rule the lead set: fewest committed receipts changed while closing the hole.

**Options**, with the number of committed HELD numeric spans each would move (1,306 such spans in the 79 receipts; counted by `d2_options.py` in the writer's notes):

- A. The code's own line, made a verdict: a comparison that would hold, where the receipt's value is not zero and its rounding at the printed precision is zero, does not hold. Moves 0 committed spans.
- B. The same, only when the sentence prints zero fractional digits. Moves 0. Leaves "0.0" against `0.04` held, so it does not close the hole.
- C. At least two significant digits must survive the rounding. Moves 5 committed spans, and makes "0.4" against `0.42` not hold.
- D. No rounding: exact decimal equality. Moves 91 committed spans in 11 receipts, and fails every honestly rounded figure.

**Choice: A**, with verdict FAILED and a new reason code `rounds_to_zero`. A, B, C and D are ordered by reach; A is the smallest rule that closes the stated hole and it changes no committed receipt.

Why FAILED and not MALFORMED: the comparison ran, against bytes the checker could see, and the sentence prints zero for a value that is not zero. MALFORMED in this format means the declaration was refused before the kind's comparison ran; some MALFORMED reasons are reached only after the receipt bytes were opened (section 10 of the text), but none after the comparison. Why a new code and not `value_mismatch`: at the printed precision the two strings are equal, and a consumer should be able to tell "rounded to nothing" from "a different number".

**What A costs.** "p = 0.00" against a receipt of `0.004` no longer holds; the author prints `0.004`, or does not bind the sentence. That is a false FAILED on a conventional rounding, taken in exchange for no false HELD on a non-zero value. A receipt that is exactly zero (`0`, `0.000`, `0E-7`) still holds against any printed zero.

**Committed receipts whose verdict or digest meaning changes: none.** 0 of the 1,306 committed HELD numeric spans has a non-zero receipt that rounds to zero.

## D3. The claim is bound in the span row

**Today.** `_adjudicate` builds the row at lines 1584 to 1586: `at`, `start`, `end`, `receipt`, `kind`, `verdict`, `reason`, `detail`. The inner bytes are read (line 1600) and never recorded. A row carries byte offsets and a `detail` computed from the inner text; two documents that differ inside a span in a way the kind does not read (the words around a number) produce rows that differ only through the document hashes. A receipt cannot be read without the document.

**Options.**

- A. Hash only: `claim_sha256`, the sha256 of the span's inner bytes exactly as written.
- B. Hash and text: `claim_sha256`, and `claim`, the inner text itself, when it is within the existing cap of 300 code points; `null` above the cap.
- C. Text only.

**Choice: B**, as the lead stated it. Both keys are always present in a row. `claim_sha256` is `null` only where the span has no canonical offsets (`start` is `null`: the document has a lexically MALFORMED declaration somewhere, so it has no canonical text, or the row is a stray closer or a malformed tag). `claim` is `null` there and wherever the inner text is over 300 code points (such a span is MALFORMED `length_cap`; its hash is still carried).

**What B costs.** A receipt now carries up to 300 code points of the document per span. A receipt for a private document is no longer free of its text. An emitter that must not disclose text has no v0 option; that is listed under "open before freeze" in the text.

**Committed receipts whose verdict changes: none. Committed receipts whose digest meaning changes: all 79**, in the sense that no /v1 or /v2 digest is a v0 digest: the v0 digest covers a different object (no verifier block, no certifies sentence, the claim keys added). The committed files are untouched and keep their own digests under their own schemas.

For 62 of the 79 the document is on the branch beside the receipt with the sha256 the receipt records, and the projection binds every claim. For 17 it is not (the 12 action samples were generated in a temporary workspace and their documents were never committed; the 5 dry-run receipts name `.md` files that are not in the tree). Their projection carries `null` claims and the legacy profile:

<!-- claims-unbound:begin -->
```
papers/sworn/measurement/dryrun/SYN-01.receipt.json
papers/sworn/measurement/dryrun/SYN-02.receipt.json
papers/sworn/measurement/dryrun/SYN-03.receipt.json
papers/sworn/measurement/dryrun/SYNX-01.receipt.json
papers/sworn/measurement/dryrun/SYNX-02.receipt.json
papers/sworn/sworn_action_sample.FAILED.sworn-receipt.json
papers/sworn/sworn_action_sample.HELD.sworn-receipt.json
papers/sworn/sworn_action_sample.UNRESOLVED.sworn-receipt.json
papers/sworn/sworn_action_sample.pull_request_body.sworn-receipt.json
papers/sworn/sworn_action_sample_2026_09_06.FAILED.sworn-receipt.json
papers/sworn/sworn_action_sample_2026_09_06.HELD.sworn-receipt.json
papers/sworn/sworn_action_sample_2026_09_06.UNRESOLVED.sworn-receipt.json
papers/sworn/sworn_action_sample_2026_09_06.pull_request_body.sworn-receipt.json
papers/sworn/sworn_action_sample_2026_10_24.FAILED.sworn-receipt.json
papers/sworn/sworn_action_sample_2026_10_24.HELD.sworn-receipt.json
papers/sworn/sworn_action_sample_2026_10_24.UNRESOLVED.sworn-receipt.json
papers/sworn/sworn_action_sample_2026_10_24.pull_request_body.sworn-receipt.json
```
<!-- claims-unbound:end -->

---

## What moves in the conformance material

Not committed receipts: these are the inline documents of the conformance sets that need nothing beyond a manifest (`conformance/sworn` and `conformance/sworn/receipt_v2`), each checked by today's code and projected. They are counted because step 2 re-keys them at the v0 core, and every one of these is a vector whose expected verdict will differ from the one pinned today. No vector file was changed.

One of the 1,718 is not a verdict change but a refusal: vector `1b4df216…` of `conformance/sworn` (family rules) hands over a manifest at `sworn/manifest/0.1` and expects `SWORN-HELD` today. A v0 checker refuses that manifest (`manifest_spec`, section 5.5 of the text), so step 2 must re-key it as a refusal, not as a core. The projector cannot see a manifest, so test B counts this from the vector's own inputs. It is not among the D1 and D2 counts.

<!-- conformance-census:begin -->
```
in-scope conformance documents = 1718
refused by a v0 checker (manifest not at sworn/manifest/0.2) = 1
document verdicts changed by D1 = 32
span verdicts changed by D2 = 1
document verdicts changed by D2 = 1
```
<!-- conformance-census:end -->

## Smaller choices made while writing, none of which moves a committed verdict

These are not among the three. They are listed so the operator signs knowing them; each is also in the text under "open before freeze" or in the writer's report.

1. `WITHHELD` is not a v0 span verdict. Today it is the fifth key of `counts` (line 126) and no line of `sworn.py` assigns it. A closed list cannot hold a word no rule produces. The projector refuses a row that carries it.
2. The reason code `manifest_spec_unknown` is not in the v0 list: `sworn.py` lists it (line 156) and never returns it. Seven reasons that need a repository are outside the profile and appear only in legacy projections.
3. A `path:` or `prereg:` REF in a v0 document is UNRESOLVED `no_repository`, which is what today's code answers when it is handed no repository (line 1292). It is not MALFORMED.
4. A manifest that cannot be put in v0's canonical form (a non-integer number, a key above U+007E, an integer outside |n| <= 2^53 - 1, an unpaired surrogate, a `NaN`) is `manifest_integrity`. Today such a manifest is digested with the lab's JCS, which admits finite doubles; and one with an unpaired surrogate or a `NaN` and no `digest` key is read without being digested.
5. A manifest at `sworn/manifest/0.1`, or one without string `harness`, `turn` and `minted_at`, is refused by a v0 checker. Today 0.1 loads, and a missing `minted_at` is filled from the clock (line 723), which a pure function cannot do.
6. Added in the fixer's round, each stricter than today: a manifest with a key written twice is refused (`manifest_form`; today the last value wins); an `authored_sha256` entry that is not 64 hex digits is refused (`manifest_authored_shape`; today any string is lowercased by Python's rule). Pinned to today, so that two checkers read them alike: a null `digest` counts as absent; a `bytes` key that is null or not a string is `manifest_integrity`; non-zero pad bits in base64 are ignored. Over the 1,718 in-scope conformance documents none of these moves a verdict: their manifests hold no duplicated key, no authored entry outside 64 hex, no null digest, no non-string bytes and no non-zero pad bits.

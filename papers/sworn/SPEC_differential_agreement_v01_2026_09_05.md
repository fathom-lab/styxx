# SPEC — differential agreement: do the two verifiers agree on inputs nobody chose?

Fathom Lab · 2026-09-05 · **A spec, not a result.** Frozen in its own commit before any code. It
makes no numeric claim; the one number it exists to produce is produced by running it, and the
gates below are written down first so the run cannot be scored after the fact.

## Why this exists

`RESULT_sworn_browser_verifier_v01_ships_2026_09_05.md` reports that a second implementation of the
sworn verifier reproduces the verdict core digest on 1689 conformance vectors, and says in its own
boundary section what that does not establish:

> That agreement on these vectors makes either verifier correct — they agree on what two test files
> exercise, and the same hands wrote both, which is the objection the next attacker should press.

The vectors are a **chosen** set. Every one of them was recorded from a call some author wrote to
demonstrate something. Two implementations agreeing on the cases their authors thought of is the
weakest form of agreement there is, and it is the form that most flatters the builder: the JS was
repaired five times *against those very vectors* until it matched. A set you tuned against cannot
also be the set that measures you.

This spec measures the other thing: **agreement on inputs nobody chose.** A generator produces
documents from the format's own decision boundaries, both implementations verify them, and the
verdict cores are compared byte for byte. It is a differential test, and its only interesting
outcome is a disagreement.

## The rules, each with its attack

**D1 — the two sides are the shipped artifacts, unmodified.** The Python side is
`styxx.sworn.verify` as installed; the JavaScript side is `styxx/_data/sworn_verify.js` as shipped
in the package. Neither is instrumented, wrapped or specialised for this run. The comparison is
`sha256(utf8(jcs(core)))` over the core minus `verifier` and minus `coverage` — the same number the
conformance vectors pin, computed independently on each side.
*Attack:* a harness that quietly normalises one side into the other. *Answer:* the harness never
touches a core; it hashes what each implementation returned and compares two hex strings. A case
where either side raises is recorded as its own outcome, never discarded.

**D2 — the generator is not a test author.** Cases are drawn from a seeded grammar over the
format's decision boundaries — tag shapes and near-misses, nesting, fences and inline code runs,
HTML comments, all four kinds, the receipt grammar in every form, numeric tokens across signs,
separators, exponents, percent signs and non-ASCII digits, quote needles at and around the
short-needle bound, span lengths at and around the code-point cap, and byte-level hazards
(CRLF, lone CR, BOM, NUL, invalid UTF-8, surrogate escapes) — composed at random rather than
chosen to make a point. The generator never consults either implementation.
*Attack:* a grammar that only emits what the JS already handles, which would measure nothing.
*Answer:* the corpus census below is published beside the agreement number — how many cases
reached each verdict, each reason and each kind — so a reader can see what the run actually
explored, and a run that never produced a MALFORMED, an UNRESOLVED and a FAILED is void by D5.

**D3 — every case is reproducible from its seed alone.** The generator is a pure function of
`(seed, index)`. A disagreement is reported with the seed, the index, and the document and manifest
bytes base64-encoded, so anybody can reproduce it without this repository's state.
*Attack:* a finding nobody else can reproduce. *Answer:* the receipt carries the inputs, and a test
re-derives a recorded case from its seed and asserts the same bytes.

**D4 — a disagreement is a finding, and it is published whichever way it falls.** If the two
implementations disagree, the RESULT is titled *the implementations disagree* and names the case.
If they do not, the RESULT reports the count and says plainly that agreement on generated inputs is
not correctness — both could be wrong together, and the same hands wrote both.
*Attack:* running until a clean sweep appears and publishing that one. *Answer:* the seed is
committed before the run in the PREREG-shaped block of the RESULT, and every run's counts are
recorded; a re-run with a new seed is a new file with a new seed named in it.

**D5 — the frozen gates.** Written here, before the code:

| gate | quantity | bar |
|---|---|---|
| G-N | cases compared | ≥ 100000 |
| G-A | cases where the two core digests are equal | reported as a count and a share; a single inequality makes the run's headline *the implementations disagree* |
| G-C | coverage of the verdict vocabulary | the run must produce at least one HELD, one FAILED, one MALFORMED and one UNRESOLVED span, and at least one document-level MALFORMED; a run that does not is VOID and reports nothing about agreement |
| G-R | distinct MALFORMED reasons exercised | ≥ 12 of the closed set; reported per reason |
| G-E | cases where either side raised | reported with the exception class, and each is a finding of its own kind — the two implementations must at least fail on the same inputs |

**D6 — the run writes one receipt and never rewrites it.** `differential_agreement_result.json`
carries the seed, the counts, the census, every disagreement in full, and the digests of both
implementations' bytes as they were at the run. A second run is a second file.
*Attack:* regenerating the receipt after a repair, so the number improves without a record.
*Answer:* the rule this lab already pays — a receipt is history — and the RESULT swears to the
file it was run against.

## What this spec does not say

That agreement makes either implementation correct: they may be wrong in the same way, and the
same author wrote both, which is the objection this spec sharpens rather than answers. That the
generator is adversarial in the sense a human attacker is — it composes boundaries, it does not
reason about them. That the grammar covers the format: it covers what D2 lists, and the census
says what it reached. That any of this has run outside this lab.

---

*The vectors asked whether a second implementation agrees where we looked. This asks whether it
agrees where nobody looked, and it is written down before the answer is known.*

## ERRATA — 2026-09-05, after the run

Appended, not edited: the rules above are the frozen text.

**D6 names the receipt `differential_agreement_result.json`.** It is committed as
`conformance/sworn/differential_agreement.json`. The name in D6 was wrong when it was written, and
the suite said so before the receipt was ever cited outside this branch:
`test_no_conformance_file_wears_a_suffix_another_sweep_claims` forbids a `conformance/` file from
wearing a name another sweep claims, and in this corpus `*_result*.json` means *a prereg-scored
experiment receipt that `test_protocol_v2v3` re-scores through `Experiment`*. This receipt is not
one — no prereg, no `verdict`, nothing scores it — so it may not wear that name and be believed by
a sweep that never asked. Nothing else in D6 changes: the file still carries the seed, the counts,
the census, every disagreement in full and both implementations' digests, it is still written once,
and a second run is still a second file.

The receipt's bytes are unchanged by the rename, and its sha256 with them. This is the one thing a
move may not do — change what the receipt says — and it does not.

# RESULT — differential agreement: 150,000 inputs nobody chose, and the two verifiers never once disagreed

Fathom Lab · 2026-09-05 · Spec: `SPEC_differential_agreement_v01_2026_09_05.md`, frozen before any
code, with its five gates and its seed written down before the run. Harness:
`conformance/sworn/differential.py`. Receipt: `conformance/sworn/differential_agreement.json`,
which this document swears to and which is not regenerated. **This document is itself sworn.**

## The question this answers, and the one it does not

`RESULT_sworn_browser_verifier_v01_ships_2026_09_05.md` reports that a second implementation
reproduces the verdict core on 1689 conformance vectors, and names its own weakness in its boundary
section: the vectors are a **chosen** set, recorded from calls the lab's own authors wrote, and the
JavaScript was repaired five times against those very vectors until it matched. A set you tuned
against cannot also be the set that measures you.

This measures agreement on inputs nobody chose. A seeded grammar composes documents out of the
format's decision boundaries; both shipped implementations verify each one; the two verdict core
digests are compared byte for byte. Neither side is instrumented, and the harness hashes what each
returned rather than inspecting how it got there.

## The result

<sworn r="path:conformance/sworn/differential_agreement.json#/compared" k="numeric">150000 generated documents were verified by both implementations and compared.</sworn>
<sworn r="path:conformance/sworn/differential_agreement.json#/agree" k="numeric">150000 produced the same verdict core digest.</sworn>
<sworn r="path:conformance/sworn/differential_agreement.json#/disagree" k="numeric">0 disagreed.</sworn>
<sworn r="path:conformance/sworn/differential_agreement.json#/gates/G-E/both" k="numeric">On 0 of them did both sides raise,</sworn>
<sworn r="path:conformance/sworn/differential_agreement.json#/gates/G-E/python_only" k="numeric">on 0 did only the Python side raise,</sworn>
<sworn r="path:conformance/sworn/differential_agreement.json#/gates/G-E/javascript_only" k="numeric">and on 0 did only the JavaScript side.</sworn>
Every case the two implementations met, they answered the same way, including the ones that are not
answers at all — a refusal, a document with no canonical form, a receipt that cannot be opened.

The run is reproducible from two integers:
<sworn r="path:conformance/sworn/differential_agreement.json#/seed" k="numeric">the seed is 20260905,</sworn>
and each case is a pure function of the seed and its index.

## What the run actually explored

A number about agreement means nothing without the census beside it, because a generator that only
emits documents one implementation handles measures the generator.

<sworn r="path:conformance/sworn/differential_agreement.json#/census/span_verdicts/MALFORMED" k="numeric">346066 spans were MALFORMED,</sworn>
<sworn r="path:conformance/sworn/differential_agreement.json#/census/span_verdicts/UNRESOLVED" k="numeric">31486 were UNRESOLVED,</sworn>
<sworn r="path:conformance/sworn/differential_agreement.json#/census/span_verdicts/FAILED" k="numeric">5333 were FAILED,</sworn>
<sworn r="path:conformance/sworn/differential_agreement.json#/census/span_verdicts/HELD" k="numeric">and 1832 HELD.</sworn>
<sworn r="path:conformance/sworn/differential_agreement.json#/gates/G-R/value" k="numeric">38 distinct MALFORMED reasons were exercised</sworn>
against a bar of twelve, and both document-level refusals were reached:
<sworn r="path:conformance/sworn/differential_agreement.json#/census/document_malformed/unbalanced_fences" k="numeric">15702 documents had unbalanced fences</sworn>
and
<sworn r="path:conformance/sworn/differential_agreement.json#/census/document_malformed/invalid_utf8" k="numeric">1524 could not be decoded at all.</sworn>
The FAILED and HELD counts matter most: those are the cases where the four kind checks actually
ran, where a decimal is canonicalised on both sides, a needle is searched in bytes, and a digest is
compared — the places two languages are most likely to part company.

Both implementations are named by the bytes they were at the run:
<sworn r="path:conformance/sworn/differential_agreement.json#/implementations/python/sha256" k="quote">the Python verifier is `bbfa8bc0761320f42d06d8cda15e6784015b63b2add250db62bbb9ef55e1311b`</sworn>
and
<sworn r="path:conformance/sworn/differential_agreement.json#/implementations/javascript/sha256" k="quote">the JavaScript one is `e633626063e8be9b9c22b1c57c18a542e4751dc69e468ccffd1e50420b9a334b`</sworn>,
content identity modulo newlines, which is the doctrine this corpus applies to every receipt.

## What the instrument cost before it could be believed

Two corrections were made to the harness before any measurement was recorded, and both are the
reason a number exists at all. The node runner read its arguments one position early and could not
load the verifier. And the first balanced grammar reached 1184 MALFORMED spans against a single
HELD and no FAILED — it fuzzed the lexer hard and the adjudicator barely, which is the half where a
disagreement would live. A generator that cannot reach HELD cannot find the disagreement it exists
to find. That is why the frozen spec made the verdict vocabulary a gate rather than a hope, and the
gate is what caught it.

## What this does not say

**That either implementation is correct.** They agree; agreement is not correctness. Both may be
wrong in the same way, and the same hands wrote both, which is the objection this result sharpens
rather than answers — it removes the weaker excuse (*they only agree where we looked*) and leaves
the stronger one standing.

That the generator is adversarial in the sense a person is: it composes boundaries, it does not
reason about them. A human attacker looking for a disagreement would start where this grammar is
thinnest — the pointer grammar's escapes, surrogate handling in a leaf, the code-point cap against
astral characters, and manifests whose digests are close but not equal.

That the grammar covers the format. It covers what the spec's D2 lists and the census above says
what it reached; the HELD path is under two percent of spans, which is the number a successor
should raise first.

That a run with a different seed would agree. This is one seed, named before the run, and a second
seed is a second file.

---

*The vectors asked whether a second implementation agrees where we looked. This asked whether it
agrees where nobody looked, one hundred and fifty thousand times, and the answer was yes every
time. That is not proof the verdict is right. It is the removal of one specific way it could have
been wrong, and it is written down with the seed that produced it.*

---

## ERRATUM — 2026-09-06: the implementations DO disagree

**Everything above remains true and none of it is retracted.** 150000 documents were compared under
the grammar this document describes, and 0 of them disagreed. What was wrong is the sentence a
reader naturally takes away from that — *the two implementations agree* — and it was wrong at the
moment this was written.

The boundary section above says the grammar's coverage is not claimed. That was honest and it was
not enough. A caveat about coverage does not tell a reader that two real defects, one of them
verdict-changing, were sitting inside the uncovered part.

`RESULT_mutation_coverage_2026_09_05.md` measured what this harness could detect and found the
receipt payload aperture to be its largest blind spot — ten hard-coded byte strings through which
the JSON parser was the only way in. `SPEC_aperture_closure_v01_2026_09_05.md` widened exactly
those lists, changing nothing about the comparison. At the same seed and the same size:

<sworn r="path:conformance/sworn/differential_agreement_2.json#/disagree" k="numeric">712 of the 150000 disagreed.</sworn>
<sworn r="path:papers/sworn/disagreement_classes_2.json#/verdict_changing_total" k="numeric">5 of the 50 recorded in full changed a verdict rather than a label,</sworn>
and one changed it in the direction that matters: the JavaScript verifier reported a span **HELD**
that `styxx.sworn` reported **MALFORMED**. A sentence the reference implementation refuses to read
at all, the browser verifier vouched for.

Two defects, both in `styxx/_data/sworn_verify.js`, both now repaired and pinned by tests:

1. **The decoders ate a leading BOM.** `new TextDecoder("utf-8", { fatal: true })` leaves
   `ignoreBOM` at `false`, which in WHATWG's inverted naming means the decoder *strips* a leading
   U+FEFF — so `jsonStrict`'s explicit BOM refusal was unreachable dead code, and a BOM-prefixed
   receipt payload held on one side and was refused on the other.
2. **`safeText` destroyed astral characters.** Its lone-surrogate fixup matched a class spanning
   high *and* low surrogates, so the low half of every valid pair was replaced: U+1F600 came out of
   a detail field as a high surrogate plus U+FFFD. It also used the wrong replacement character
   (`?`, not U+FFFD, is what Python's encode-side `errors="replace"` emits) and sliced by UTF-16
   code units where Python slices by code points.

After both repairs, at the widened grammar and the same seed,
<sworn r="path:conformance/sworn/differential_agreement_4.json#/disagree" k="numeric">0 of 150000 disagreed</sworn>
over
<sworn r="path:conformance/sworn/differential_agreement_4.json#/census/span_verdicts/HELD" k="numeric">1169 HELD</sworn>
and
<sworn r="path:conformance/sworn/differential_agreement_4.json#/census/span_verdicts/FAILED" k="numeric">3347 FAILED</sworn>
spans. That zero and the zero at the top of this document are the same number and not the same
fact. The first was a statement about what the generator could produce. The second is a statement
about two implementations, over a generator that reaches the places a mutation study said it could
not.

**What could not see either defect.** The conformance set passed identically before and after both
repairs — 1689 vectors ran, 1689 passed, 0 failed, every time. Two independent instruments were
clean while a live, verdict-changing divergence sat in the shipped file. That is the finding this
erratum exists to carry, and it is larger than either bug.

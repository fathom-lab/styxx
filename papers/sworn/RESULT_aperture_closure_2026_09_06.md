# RESULT — the blind spot was hiding two real defects, and one of them changed a verdict

Fathom Lab · 2026-09-06 · Spec: `SPEC_aperture_closure_v01_2026_09_05.md`, frozen before the
generator was touched. Receipts: `conformance/sworn/differential_agreement_2.json`, `_3.json`,
`_4.json`, and `papers/sworn/disagreement_classes_2.json`, `_3.json`. A claim this leg made and
then withdrew is recorded in `NOTE_unpaired_samples_2026_09_06.md`. **This document is itself
sworn.**

## What was asked

`RESULT_mutation_coverage_2026_09_05.md` sorted 29 missed mutations into causes and claimed 20 of
them would fall to a stronger generator while 9 would not. That was the most useful sentence in the
document and the one with no evidence behind it. This spec widened the generator in exactly the
places the taxonomy named — changing nothing about the comparison — to find out whether the
diagnosis was real or a story that fit.

The answer arrived before any of the intended bookkeeping: **the two implementations disagree.**

## The disagreement

Same seed, same case count, the same `python_digest`, `js_digests`, core and exclusions. Only what
the generator can produce changed.

<sworn r="path:conformance/sworn/differential_agreement_2.json#/disagree" k="numeric">712 of 150000 disagreed</sworn>
where the narrower grammar had found none, and
<sworn r="path:papers/sworn/disagreement_classes_2.json#/verdict_changing_total" k="numeric">5</sworn>
of the
<sworn r="path:papers/sworn/disagreement_classes_2.json#/recorded_in_full" k="numeric">50 recorded in full</sworn>
changed a **verdict** rather than a label. One changed it in the direction that matters: the
JavaScript verifier reported a span **HELD** that `styxx.sworn` reported **MALFORMED**. A sentence
the reference implementation refuses to read at all, the browser verifier vouched for.

**The blind spot was not merely a gap in coverage. It was hiding a live divergence.** The mutation
study named the receipt payload aperture as its largest blind spot and named the BOM refusal
specifically as invisible. Opening that aperture did not make the harness stronger and then, later,
find something. The thing was already there.

## The two defects

Both in `styxx/_data/sworn_verify.js`, both reduced to minimal hand-built inputs before being
believed, both repaired after the divergence was recorded with the implementations unrepaired.

**1. The decoders ate a leading BOM.** `new TextDecoder("utf-8", { fatal: true })` leaves
`ignoreBOM` at `false`, which in WHATWG's inverted naming means the decoder *strips* a leading
U+FEFF. So `jsonStrict`'s explicit BOM refusal was unreachable dead code on every bytes path, and a
BOM-prefixed receipt payload held on one side and was refused on the other. `decodeStrict` also
produces every span's inner text, so the same stripping reached there.

**2. `safeText` diverged three ways at once, and one was destructive.** Its lone-surrogate fixup
matched `[\ud800-\udfff]` — a class spanning high *and* low surrogates — so for a valid pair the
**low half** matched and was replaced: U+1F600 came out of a detail field as a high surrogate plus
U+FFFD. Every astral character in a `detail.leaf` was corrupted. It also used the wrong replacement
character (Python's encode-side `errors="replace"` emits `?`, not U+FFFD) and sliced by UTF-16 code
units where Python slices by code points.

After the first repair,
<sworn r="path:conformance/sworn/differential_agreement_3.json#/disagree" k="numeric">31 disagreements remained</sworn>
and
<sworn r="path:papers/sworn/disagreement_classes_3.json#/verdict_changing_total" k="numeric">0 of them changed a verdict.</sworn>
After both,
<sworn r="path:conformance/sworn/differential_agreement_4.json#/disagree" k="numeric">0 of 150000 disagreed,</sworn>
over
<sworn r="path:conformance/sworn/differential_agreement_4.json#/census/span_verdicts/HELD" k="numeric">1169 HELD</sworn>
and
<sworn r="path:conformance/sworn/differential_agreement_4.json#/census/span_verdicts/FAILED" k="numeric">3347 FAILED</sworn>
spans, with an identical span census across all three runs — the check that the same inputs were
explored each time and only the implementation moved.

That zero and the zero the earlier RESULT reported are the same number and not the same fact. The
first was a statement about what a generator could produce. The second is a statement about two
implementations, over a generator that reaches the places a mutation study said it could not.

## What no instrument in this repository could see

The conformance set passed identically before and after **both** repairs:
<sworn r="path:conformance/sworn/replay_js_report.json#/ran" k="numeric">1689 vectors ran</sworn>
and
<sworn r="path:conformance/sworn/replay_js_report.json#/passed" k="numeric">1689 passed</sworn>
every time. Twice in one session a clean instrument sat beside a live, verdict-changing divergence
in the shipped file. That is a larger finding than either bug, and it is the argument for
calibration over accumulation: a second vector set would not have helped, because the vectors and
the differential shared the blind spot.

## The taxonomy's structural half held; its rate did not survive contact

Three misses closed, and all three for the reason the taxonomy gave rather than by luck: the old
generator emitted no manifest string containing a newline, never more than one element in
`authored_sha256`, and never an uppercase digest. Those inputs did not exist at any case count.
That is a claim about what a grammar can produce and it needs no pairing to be true.

The detection **rate** is a different matter, and the comparison this spec asked for cannot be made.
`NOTE_unpaired_samples_2026_09_06.md` withdraws it in full. `case(seed, index)` draws from a stream
seeded on `(seed, index)`, every draw advances that stream, and the widening added draws — so of the
first 500 cases at the same seed, only 51 produce the same document under both grammars. Two runs
"at the same seed" were two samples from two distributions. G-S, the gate that would have made the
taxonomy refutable, cannot be evaluated for the same reason.

**Any change to a seeded generator, including a purely additive one, changes every case the seed
produces after the insertion point. Fixing the seed does not fix the sample.** That applies to every
fuzzing and mutation study that iterates on its generator and reports before-and-after "at the same
seed", which is the ordinary way such studies are reported.

## What this does not say

**That the format or either implementation is now correct.** Two defects were found and repaired;
nothing here bounds what remains. The same harness that found these was blind to them a day ago.

**That the widened grammar is the right grammar.** It is one step, aimed by one taxonomy, and its
own miss list names the next. The nine misses the taxonomy called unfixable by fuzzing — the
tree-handle layer, the sidecar layer, the receipt layer — are untouched, and no generator reaches
them.

**That a disagreement means the JavaScript was wrong.** In both cases here it was, judged against
`styxx/sworn.py` as the reference implementation and against each file's own stated intent — the
BOM refusal that could never fire, the comment that claimed U+FFFD where Python emits `?`. A
disagreement on its own names a difference, not a culprit.

**That the repairs are proved by the run that follows them.** A run that agrees after a fix is weak
evidence; the fix was aimed at a mechanism, and the guards are `tests/test_sworn_bom_agreement.py`
and `tests/test_sworn_astral_agreement.py`, both watched to fail — 2 of 5 and 10 of 16 respectively
with the defective code restored.

---

*A test that reports perfect agreement is reporting on two things at once, and only one of them is
the code under test. This leg found out which one it had been measuring, and the answer was sitting
in a shipped file the whole time.*

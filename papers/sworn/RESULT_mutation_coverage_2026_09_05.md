# RESULT — the differential test cannot see 29 of 70 changes, and 9 of those no amount of fuzzing would have found

Fathom Lab · 2026-09-05 · Spec: `SPEC_mutation_coverage_v01_2026_09_05.md`, frozen with its five
gates before any mutant was run. Catalogue: `conformance/sworn/mutation_catalogue_2.json`, committed
before the run. Receipt: `conformance/sworn/mutation_coverage_2.json`. Taxonomy:
`mutation_miss_taxonomy.json`. **This document is itself sworn.**

## What this measures, and why the previous result needed it

`RESULT_differential_agreement_2026_09_05.md` reports 150000 generated documents on which two
independent implementations produced the same verdict core digest, and zero on which they differed.
That sentence has two readings and it cannot tell them apart: the implementations agree, or the
generator cannot produce the inputs on which they differ.

This measures the second. A mutation is one localised semantic edit to a scratch copy of one
implementation; the standing guard's own comparison runs against it; **caught** means a divergence
there would be visible, **missed** means it would not.

## The result

<sworn r="path:conformance/sworn/mutation_coverage_2.json#/counts/viable" k="numeric">70 viable mutants were measured</sworn>
against a bar of twenty-five, and
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/counts/caught" k="numeric">41 were caught.</sworn>
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/counts/missed" k="numeric">29 were missed</sworn>
— a detection rate of
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/gates/G-D/value/rate" k="numeric">0.5857.</sworn>

The catalogue was clean, so nothing left the denominator quietly:
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/counts/anchor_missing" k="numeric">0 anchors failed to match,</sworn>
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/counts/non_viable" k="numeric">0 mutants failed to load,</sworn>
and
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/counts/degenerate" k="numeric">0 were degenerate.</sworn>
The unmutated pair was checked first and produced
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/baseline/disagreements" k="numeric">0 disagreements,</sworn>
without which a detection rate would be measured over a broken baseline and mean nothing.

## The rate is the summary. The causes are the result.

A bare detection rate invites one reading — that the undetected fraction is a matter of running
longer. Sorted by cause, that is false here:

<sworn r="path:papers/sworn/mutation_miss_taxonomy.json#/counts/APERTURE_PAYLOAD" k="numeric">15 misses need an input the generator's fixed literal list cannot produce,</sworn>
<sworn r="path:papers/sworn/mutation_miss_taxonomy.json#/counts/APERTURE_DOCUMENT" k="numeric">5 need a document shape the grammar does not compose,</sworn>
<sworn r="path:papers/sworn/mutation_miss_taxonomy.json#/counts/OUT_OF_SURFACE" k="numeric">6 are in code the differential never runs or whose effect never reaches the compared core,</sworn>
<sworn r="path:papers/sworn/mutation_miss_taxonomy.json#/counts/UNREACHABLE" k="numeric">2 could not be caught by any input at all,</sworn>
and
<sworn r="path:papers/sworn/mutation_miss_taxonomy.json#/counts/EQUIVALENT" k="numeric">1 is an equivalent mutant, proved to change no behaviour on any input.</sworn>

So
<sworn r="path:papers/sworn/mutation_miss_taxonomy.json#/fixable_by_a_stronger_generator" k="numeric">20 of the misses would fall to a stronger generator</sworn>
and
<sworn r="path:papers/sworn/mutation_miss_taxonomy.json#/not_fixable_by_fuzzing" k="numeric">9 would not.</sworn>
Those nine are the part a bigger case count could never have reached, and naming them is the whole
value of running this.

### The six that are outside the compared surface

This is the finding. The JavaScript verifier has no repository — `SPEC_sworn_browser_verifier_v01`
B1 makes it answer `no_repository` — so every `path:` and `prereg:` receipt resolves on the Python
side against nothing on the other. The entire tree-handle layer is therefore compared against
silence, and so are the sidecar layer and the receipt layer, which sit outside the verdict core the
digest covers. A mutation that makes `SnapshotTree` treat a symlink as a blob, or lets
`load_sidecar` admit a zero-length span, or undoes R9 so the receipt digest covers `coverage`
again, produces no disagreement — not because the implementations agree there, but because only one
of them is ever asked.

**"The two implementations agree" was always a claim about a subset of each implementation, and
until now nobody had said which subset.** That is not a defect the differential introduced; it is a
scope the differential always had and never stated. It is stated now.

## Where the harness is strong, and where it is blind

<sworn r="path:conformance/sworn/mutation_coverage_2.json#/by_region/kinds/caught" k="numeric">The four kind checks caught 9</sworn>
of
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/by_region/kinds/viable" k="numeric">9 —</sworn>
the adjudicator's core is well covered, which is what the frozen G-C gate on the earlier run was
built to ensure. The lexer follows at
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/by_region/lexer/caught" k="numeric">7</sworn>
of
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/by_region/lexer/viable" k="numeric">8.</sworn>

The weakest region is not weak by accident:
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/by_region/audit/caught" k="numeric">1</sworn>
of
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/by_region/audit/viable" k="numeric">10.</sworn>
Those ten are the completeness critic's own additions, written after it read both implementations
and named the regions nobody had proposed a mutation for. Nine of its ten shots landed exactly where
the harness cannot see. **An agent reading the source predicted the blind spots better than 150000
random cases could find them**, which is an argument for reading code as well as fuzzing it.

The decimal region — the most numerically delicate rule in the format — caught
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/by_region/decimal/caught" k="numeric">4</sworn>
of
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/by_region/decimal/viable" k="numeric">8.</sworn>
Among the misses: half-even rounding can be changed to half-up **on either side** and nothing
notices, because a tie is only observable on a value sitting exactly at one at the printed
precision, and none of the ten payload literals produces one.

## How close the catches were

A catch is not a guarantee. Out of
<sworn r="path:conformance/sworn/mutation_coverage_2.json#/cases" k="numeric">5000 cases per mutant,</sworn>
<sworn r="path:papers/sworn/mutation_miss_taxonomy.json#/marginality_of_catches/buckets/exactly_1" k="numeric">4 mutations were caught by exactly one disagreement,</sworn>
and
<sworn r="path:papers/sworn/mutation_miss_taxonomy.json#/marginality_of_catches/buckets/from_2_to_5" k="numeric">4 more by between two and five.</sworn>
Eight of the forty-one catches would probably have been misses at a fifth of the size. The guard's
size is a choice, and this is the evidence for it rather than a number picked because it felt
generous.

## The run before this one was VOID, and it is in the tree

The first run failed G-K: a control was caught, and the frozen spec says that voids the run and it
reports nothing about detection. `conformance/sworn/mutation_coverage.json` is committed unedited
as a void run, because a study about instruments that flatter their builders does not get to
discard its own failed attempt.

The gate did its job, though not for the reason it was written. G-K exists to catch a guard that
fires on every edit. What it caught was a **catalogue** defect: the agent that proposed
`findBytes(hay, needle, 0) >= 0` → `> 0` labelled it semantics-preserving, and it is nothing of the
sort — a needle at byte offset zero would report HELD instead of FAILED, a false negative on the one
kind whose whole purpose is swearing something is absent. Auditing all twelve claimed controls found
a second mislabel pointing the other way, which was missed and so sat outside the denominator hiding
a real miss.

So the label is no longer taken from whoever proposed it. `conformance/sworn/control_audit.py`
decides it from the edit by a criterion fixed before it was applied and applied in both directions —
an entry mislabelled a mutation is corrected too, even though that can only enlarge the denominator
and lower the rate. It needed three corrections of its own before it could be trusted, and they are
recorded in its docstring, because a classifier that decides a study's denominator has to be audited
like anything else.

## What this does not say

**That the two implementations disagree anywhere.** Nothing here found a disagreement. The 150000
agreements stand exactly as reported; what changes is that their scope is now stated.

**That 58.6% is a property of the format, or of differential testing, or of anything but this
generator at this size against this catalogue.** A different catalogue would give a different
number. The catalogue is 80 entries proposed by eight readers, and the critic's gap list — published
in the catalogue file — names regions still unmutated after its own additions, including the
manifest digest domain and the strict-JSON duplicate-key path on the side that mints receipts.

**That the miss taxonomy is derived rather than authored.** The five causes were defined before the
misses were assigned to them, but the assignments are a reading of each mutation's own `why`, and
they are published per miss so a reader can dispute any one of them.

**That a caught mutation means the format is correctly implemented.** It means a difference in that
place would be visible. That is all it has ever meant.

---

*The previous result asked whether two implementations agree on inputs nobody chose. This asks what
it would have taken for them to disagree in front of us — and the answer, for nine of the twenty-nine
places we looked, is that nothing we could have generated would have been enough.*

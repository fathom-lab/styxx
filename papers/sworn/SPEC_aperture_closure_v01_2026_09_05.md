# SPEC — aperture closure: does naming a blind spot let you close it?

Fathom Lab · 2026-09-05 · **A spec, not a result.** Frozen in its own commit before the generator is
touched. It makes no numeric claim; the numbers it exists to produce are produced by running it, and
the gates are written down first so the run cannot be scored after the fact.

## Why this exists

`RESULT_mutation_coverage_2026_09_05.md` sorts 29 missed mutations into five causes and claims that
**20 of them would fall to a stronger generator** while 9 would not. That claim is the most useful
sentence in the document and it is the one with no evidence behind it. It is a diagnosis, and a
diagnosis that is never acted on is indistinguishable from a story that fits.

This tests it. The generator's payload and document apertures are widened in the specific ways the
taxonomy names, the mutation study is re-run against the **same catalogue**, and the miss list is
compared. If the taxonomy is right, the 20 shrink and the 9 do not move. If the 9 move, the taxonomy
was wrong about what "outside the compared surface" means. If the 20 do not shrink, the taxonomy was
wrong about the aperture, and the diagnosis that reads so well was decoration.

Either way the answer is worth more than the rate it adjusts.

## The method

The generator is `conformance/sworn/differential.py`. Its aperture is three fixed lists — the ten
receipt payload literals, the receipt-id forms, and the manifest constants — plus the document
grammar's shapes. Widening means adding to those lists, never changing how anything is compared and
never touching either implementation.

The catalogue is **not** changed. Re-running a different catalogue would measure a different thing;
the whole point is the same 80 mutations against a wider aperture.

## The rules, each with its attack

**A1 — the widening is written against the taxonomy, not against the miss list.** Each addition
names the taxonomy cause it targets. Additions are drawn from what `SPEC_differential_agreement_v01`
D2 already claimed the grammar composed — BOM, NaN/Infinity, surrogate escapes, rounding ties,
overlapping needles, `~0`/`~1` keys — so this closes a gap between what D2 said and what the code
did, rather than inventing new scope.
*Attack:* tuning the generator to the specific mutations, which would measure nothing but the
tuning. *Answer:* additions are *classes* of input (a BOM-prefixed payload, a payload at a rounding
tie), never a value chosen to flip one named mutant, and the diff is small enough to read.

**A2 — nothing about the comparison changes.** Same digest, same core, same exclusions, same seed
discipline. Only `_document`, `_manifest` and their literal lists move.
*Attack:* a "stronger generator" that is really a looser comparison. *Answer:* the harness's
comparison code is untouched in the diff, and the reviewer can check that.

**A3 — the old grammar's receipts stand.** `differential_agreement.json` and
`mutation_coverage_2.json` were produced by grammar v1 and are not regenerated. The new run is a new
file, and the receipt names the generator's own content digest so the two instruments are
distinguishable.
*Attack:* improving a published number in place. *Answer:* the rule this lab already pays — a
receipt is history — and `mutation_coverage.py` already records the harness digest for exactly this.

**A4 — the standing guard must still pass at the new grammar, and the 150000-case agreement run is
repeated.** A wider aperture that makes the two implementations disagree is not a defeat; it is the
single most valuable outcome available here, and it would mean the differential has finally earned
its name. It is published immediately and prominently if it happens.
*Attack:* quietly narrowing the widening until agreement returns. *Answer:* this rule, and the fact
that a disagreement would be reproducible from two integers.

**A5 — the prediction is recorded before the run.** The taxonomy already predicts which 20 shrink
and which 9 do not. That is the prediction; no new one is needed, and it is not to be edited.

## The frozen gates

| gate | quantity | bar |
|---|---|---|
| G-W | viable mutants measured, same catalogue | ≥ 70, i.e. no mutant lost to the widening |
| G-K | controls caught | **exactly 0** — a caught control voids the run, as before |
| G-V | verdict vocabulary reached by the new grammar | at least one HELD, FAILED, UNRESOLVED and MALFORMED span, and both document-level refusals |
| G-A | agreement on the 150000-case re-run at the new grammar | reported as a count; a single disagreement makes that the headline |
| G-S | misses whose cause was `OUT_OF_SURFACE`, `UNREACHABLE` or `EQUIVALENT` that become caught | **reported, and each one is a refutation of the taxonomy** — if any of the 9 flips, the taxonomy's central distinction was wrong |
| G-D | detection rate, caught / viable | **no bar**, reported beside the v1 rate; the PAIR is the result, never the new number alone |

A run failing G-W has lost mutants and says so. A run failing G-K is VOID. G-D and G-S cannot be
failed, only reported.

## What this spec does not say

That a higher detection rate makes either implementation more correct — it makes a difference in
more places visible, which is all any of this has ever meant. That the aperture can be closed
completely: 9 of the 29 misses are not aperture at all, and no generator reaches them. That the
widened grammar is the right grammar; it is one step, aimed by one taxonomy, and its own miss list
will name the next. That a detection rate from this study transfers to any other differential test.

---

*The last result named the blind spots. This one finds out whether naming them was worth anything,
and it is written down before the generator is touched so that the answer can embarrass it.*

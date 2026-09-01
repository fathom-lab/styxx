# CENSUS — the read battery's pool was never written down

Fathom Lab · 2026-09-01 · Receipts: `read_extraction_census.json`,
`read_extraction_census.py`. Reconciled against `g0clear_result_llama3b.json`,
`b31v2_result.json`, `b34v3_result.json`, `b34v3_fresh_split_addendum.json`,
`b35c_result.json` — none of which this census writes or edits.

**This is a CENSUS**, in the tradition of `RESULT_collateral_census_2026_08_31.md` and
`papers/closed-model-frontier/extraction_census.py`: descriptive, report-only, no hypothesis
test, no gate, no verdict token. **No preregistration covers it, so nothing in it may carry a
headline finding.** It loads no model and runs no experiment. It parses two committed source
files with `ast`, replays two splits with numpy, and reconciles every count it produces
against a receipt written by someone else. Every number below either cites a receipt or is
marked UNCHECKABLE. A sibling document, `PREREG_read_extraction_ceiling_2026_09_01.md`,
preregisters a prospective experiment on the same term; **this census is not a run under it
and is not scored by it**, and nothing here may be read as a result of that preregistration.

## What was being tested

This lab has adopted a decomposition authored by a peer session:

    P = E x A

**P** is precision as published. **E** — extraction — is the share of cases where the thing
the instrument was pointed at was actually the thing it claims to adjudicate. **A** —
adjudication — is the share where the verdict was right, *given* the pointing was right.
Every number this lab publishes is P, and has been read as if it were A. For diffgate the P
was measured at 0.16 held-out (`RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`)
and its E-term is open. For the calibrated instruments the benchmark hands the span to the
instrument, so E = 1 by construction and their figures are pure A-terms.

The extension under test was that the cross-model read has the second shape, and that its
E-term is a **selection ratio** — the share of candidate concepts that survived into the
committed battery — recoverable from committed code and receipts with no new experiment.

**Half of that is right and better supported than it was stated. The estimator does not
exist.** A peer session had begun building deployment arithmetic of the form
`E x 0.7857` on top of it. That arithmetic has no E to multiply by, and this census is the
reason to stop writing it.

## The construction chain

Every row is a pure function of committed bytes (`read_extraction_census.json` `.chain`):

| stage | in | removed | out | citation |
|---|---|---|---|---|
| hand-typed literal `_BANK` | — | — | **465** | `run_g0clear.py:31-65` |
| order-preserving deduplication | 465 | 3 | **462** | `run_g0clear.py:66-67` |
| tokenization / single-token constraint | 462 | 0 | 462 | no such filter exists |
| presence-in-both-models / shared-vocabulary check | 462 | 0 | 462 | no such filter exists |
| frequency threshold | 462 | 0 | 462 | no such filter exists |
| part-of-speech filter | 462 | 0 | 462 | no such filter exists |
| representation-quality / norm or outlier screen | 462 | 0 | 462 | no such filter exists |
| downstream re-filtering by the read experiments | 462 | 0 | 462 | `run_b31v2.py:34,103`; `run_b34v3.py:26,70` |

The three removed words are `chicken`, `orange` and `mushroom` — each typed twice, in two
different category blocks. That is the entire filtering history of the battery.

The zeros are asserted mechanically rather than by keyword search. A per-concept filter in
these extractors could only be a skip inside the concept loop, and there are none: `ast`
reports **zero `continue` and zero `break` statements** in `run_g0clear.extract_multi` and in
`run_thought_transfer.extract`, and the single `if` inside either loop is quoted in the
receipt — it is `if (i + 1) % 40 == 0:`, a progress print. The keyword scan is reported
beside it as a diagnostic, not a category, with every hit's source line: all three hits are a
`print(... "skip")` and two `skip_special_tokens=True` kwargs. **No concept has ever been
dropped by any measurement.** "The G0-clear set" names the instrument clearing `pc_cos >=
0.80` at the locked layer 11 / k 150 (`g0clear_result_llama3b.json` `.locked`,
`.G0_pc_cos_FINAL` 0.9096), not a per-concept screen.

The battery's own provenance is excellent and its selection's provenance is nil.
`git log --follow` on `run_g0clear.py` returns **one commit**, `681c26e`, 2026-06-20. The
list was born whole and has never been edited. The 465 is recorded in no receipt, no log, no
constant and no docstring; the 462 is recorded once, at `g0clear_result_llama3b.json`
`.n_concepts`. The "~480" in the docstring and the "N≈480" in
`PREREG_thought_transfer_g0clear_2026_06_20.md` were written before the list existed. They
are design targets, not counts of anything.

## The only computable ratio, and why it is not E

462/465 = **0.993548**. It is exact, it is reproducible from bytes, and it is not an E-term.
It measures that the author typed three words twice while balancing category blocks. The
literal is the **output** of the selection, not its input, so quoting this figure as a
survival rate would be precisely the error the decomposition exists to name. It is emitted
under the key `dedup_survival_ratio_NOT_A_SELECTION_RATIO`, with the warning inside the value,
for the same reason `extraction_census.py` put its marker in its key names: a consumer who
indexes a bare key never reads the sibling that says the number decides nothing.

The pool the 462 were actually selected from is the set of candidate words a person
considered and rejected while typing 341 new nouns into a category-balanced list on
2026-06-20. **It was never recorded, and it cannot be recovered, because it never existed as
an artifact.** There is no generator script, no cited corpus, no vocabulary slice, no
rejection log — searching the repository for any of those returns `run_g0clear.py` alone.

**E for the cross-model read is UNCHECKABLE from committed artifacts.** Not zero, not low,
not refuted. Unmeasured. Absence of a recorded pool is not evidence that the pool was small
or adversarially chosen; it is evidence of nothing, which is the point. This census does not
estimate it, and no reader should read a bound out of the fact that it declines to.

## What the published read numbers actually are

They are A-terms, and the mechanical argument is stronger than the argument from analogy.
`read_top1` is an index-matched `argmin` over the held-out target centroids
(`run_b31v2.py:90-93`, `run_b34v3.py:46-49`): the candidate array is the query set, the truth
is present with probability 1, there is no score to threshold and no way to answer "none of
these". E = 1 by construction on every trial these scripts have ever run. The protocol does
not merely receive the state it reads — it **manufactures** it, from twelve
experimenter-written sentences that each name the target word
(`introspection_gate.py:26-38`), mean-pooled at the last token, at a layer locked by the
G0 sweep, differenced against a fixed `"object"` baseline.

| figure | value | receipt |
|---|---|---|
| b31v2, gemma-2-2b, MLP top-1 over 70 | 0.7857 | `b31v2_result.json` |
| b34v3, gemma-2-2b, label-free, full 70 | 0.5714 | `b34v3_result.json` |
| b34v3, gemma-2-2b, 57 genuinely unseen | **0.5263** | `b34v3_fresh_split_addendum.json` |
| b34v3, Llama-3.2-1B, label-free, full 70 | 0.6857 | `b34v3_result.json` |
| b34v3, Llama-3.2-1B, 57 genuinely unseen | **0.6667** | `b34v3_fresh_split_addendum.json` |

The b31v2 figure additionally requires being handed 392 true cross-model concept pairs; that
is a supervision precondition and it belongs beside the number wherever the number is quoted
as deployment-relevant. The b34v3 figures are the ones that survive without it, and the
fresh-57 recompute is the honest one — the full-70 headline includes 13 concepts that had
been scored before.

## Two corrections to the brief this census was given

**b34v3 does not use `split_concepts(seed=0).`** It draws its own permutation with
`SEED = 343` (`run_b34v3.py:32,64-72`) — the same 462 battery, the same 392/70 shape,
different membership. This census replays both splits and both reconcile against their
receipts exactly: 323/69/70 for seed 0, 392/70 for seed 343.

**The preregistered disjointness of those two held-out sets was falsified in flight**, and
the falsification is already committed. The census recomputes the overlap independently and
gets **13 of 70**, matching `b34v3_fresh_split_addendum.json` `.held_out_overlap_with_v1v2`.
The thirteen words are listed in the receipt.

A third drift, found while counting: the parent bank at `run_thought_transfer.py:31-40` is
**121 words** in the committed literal, while `run_g0clear.py:5` and
`g0clear_result_llama3b.json` `.parent_baseline.N` both call it 110. All 121 are contained in
`_BANK`; 341 words are net new. The 11-word gap is unrecorded anywhere. It changes no
published result and it is named here rather than left to be found.

## Held-out-ness spent by repetition

Twelve committed scripts in `papers/disjoint-worlds/` reference `split_concepts`; one of them
defines it, so eleven are consumers, and twenty-six import the battery from `run_g0clear`
(both lists are enumerated in the receipt). Within any single run the seed-0 FIN-70 is never
in a fit. But it has been **scored** by every one of those consumers, and no single receipt
records that. Held-out-ness is a property of a history, not of a line of code, and this
history is longer than any one experiment's receipt shows.

A second coupling belongs beside it: the layer and k that both b31v2 and b34v3 use are loaded
out of `g0clear_result_llama3b.json` (`run_b31v2.py:108-109`, `run_b34v3.py:60-61`). They were
selected on SEL_dirs, which is disjoint from the seed-0 FIN-70 but sits inside b34v3's
seed-343 training set. Legitimate, and worth saying: the apparatus's hyperparameters were
tuned on the same hand-typed literal the read is scored over.

## What would have to be re-run

Nothing recovers the pool retrospectively. E becomes measurable only prospectively, by
building the battery so that a pool exists before the selection does:

1. Commit a candidate pool as a mechanically-derived artifact with a stated rule and a hashed
   file — not a docstring summary.
2. Express each intended filter as a committed function and record the survivor count after
   each stage, in order, in the result JSON, the way `selection_grid` already records the
   (layer, k) sweep at `run_g0clear.py:145`. The receipt schema has room; it records the
   endpoint and nothing upstream.
3. Sample the battery from the survivors under a committed seed.
4. Re-extract on GPU across all four models. The `_b31v2_pts_*.npz` caches are keyed
   positionally to the current 462-word list (`run_b34v3.py:37-38`) and cannot be reused for
   a different battery. This is not a receipts-only repair.
5. Re-run the b31v2 M0/M1/N1 cells and the b34v3 label-free read over that battery.

Even then, what is bought is an E-term for *single-word English noun concepts*, not for the
states a model can hold. That second gap survives the repair, and no construction over a word
list closes it.

---

## ⚠ Boundary

*In the register of `every-mind-leaves-vitals.md`'s own scope erratum, and for the same
reason: the instrument's value is that it refuses to overclaim about itself.*

> - **A selection ratio is not a precision, and bounds nothing on its own.** Even had the pool
>   been recorded, a survival fraction multiplied into a published top-1 would not yield a
>   deployment number. It is one factor of a chain rule whose other factors are unmeasured
>   here. The one ratio this census can compute — 462/465 — measures deduplication and must
>   never be quoted as E.
>
> - **The read is closed-set top-1 over 70 candidates, not free recall, and the candidates are
>   the target model's own measured centroids.** There is no threshold, no calibrated score
>   and no reject option anywhere in `run_b31v2.py` or `run_b34v3.py`. Removing the
>   truth-in-set guarantee does not scale the number down by a coverage factor; it changes the
>   task from ranking to detection, and this apparatus has no detection capability at all. It
>   also presupposes already possessing the target's own extraction for the true concept —
>   in deployment you do not have it and cannot build it without knowing the answer. The one
>   measurement in this arc that touches the cost is `b35c_result.json`, which widened an
>   *already closed* set from 70 to 462 and retained 0.35 of the gemma read (0.5714 → 0.2000)
>   and 0.4584 of the llama read (0.6857 → 0.3143). That run returned
>   `INVALID__null_artifact` on a null-model error, so those retentions are **unlicensed
>   observations that decide nothing** — reported here so a reader does not have to discover
>   the direction of the effect on their own. `E x A` would understate this gap, not capture
>   it, because it treats A as invariant to a conditioning that b35c already shows it is not.
>   The separately-named pair-coverage term that *is* licensed lives in `b50_result.json`
>   (45 pairs, max 0.2396, per-member means 0.0370–0.0914) and is measured over its own
>   96-item battery, not this 462-word one, so it is not composable with anything above.
>
> - **We measure behaviour and representation, never minds.** Everything in this census is a
>   property of a word list, a permutation, and an `argmin` over vectors. The battery is a
>   list of English nouns; the population of states a model can hold is not a list of English
>   nouns, and no number here speaks to the second.
>
> - **This paper carries no verdict token, and none of its figures is licensed to decide
>   anything.** No preregistration covers a census.

---

*The chain from the literal to the read is fully auditable and fully clean: 465 typed, 3
deduplicated, 462 scored, no concept ever dropped by any measurement. The pool those 465 came
out of is the one link with no receipt, and we know of no other cross-model read number in
this repository whose extraction term is in better shape. The honest report is that we do not
know what fraction of the thinkable survived into the battery, and this census refuses to
guess.*

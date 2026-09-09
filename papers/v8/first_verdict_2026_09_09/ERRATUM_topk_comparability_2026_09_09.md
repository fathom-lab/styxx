# ERRATUM — the topk claim in RESULT_first_verdict depends on an undetermined specification sentence

Fathom Lab · 2026-09-09 · Erratum to `RESULT_first_verdict_2026_09_09.md`, issued the same day.
**The RESULT is preserved byte-identical and is not edited.** Cite it *with* this erratum. Nothing
below withdraws a number; what it withdraws is the claim that the specification determines whether
one of them should have been printed.

## The claim affected

The RESULT's section headed *The finding inside the verdict* states:

> On the topk channel — the top-five log-probabilities at the first answer positions — **changing
> the batch size perturbs the model's output distribution more than changing the numeric precision
> from bf16 to fp16 does.**

resting on `topk distance=1.407087824 floor=2.140233900 ratio=0.657445817 same`.

## What was found

A second implementation of the floor arithmetic, written from the specification by an author
forbidden to read `styxx/v8/distances.py`, `floor.py` or `fingerprint.py`, reproduced **all thirty
stored numbers exactly** — ten pairwise distances on each of three channels, to the last digit. That
part of the RESULT is confirmed by an independent reading.

The same work found that section 3.2's comparability rule does not determine whether the topk
comparison was permitted. The rule allows a topk comparison when both certificates' `topk_forced_on`
values are equal and not `"self"`, or when both are `"self"` and the two sides' greedy outputs are
token-identical. In these certificates `topk_forced_on` is **absent on both sides**, and the
specification never says what an absent value means.

- Read absent as *a value, equal on both sides and not `"self"`*: the comparison is permitted and
  every published number stands. This is the reading both implementations took.
- Read absent as *the run scored its own greedy output*, which is what `"self"` describes: the
  outputs are not token-identical, the comparison is topk-inconclusive, and section 6 says no topk
  number is printed. The RESULT printed one.

Under the second reading the consequence extends to the floor. Seven of the ten floor pairs are the
pairs whose runs disagree on some token, so all seven become inconclusive, leaving only the three
token-identical pairs, which are exactly zero. **The topk floor would be 0.0 rather than
2.1402339** — a vacuous floor, arriving through a sentence in the specification rather than through
a runner that could not apply its plan.

## What replaces the claim

The exact and seqlp results are untouched. For topk, the sentence to use is:

> Under the reading of section 3.2 that both implementations took, changing the batch size perturbs
> this model's top-five distribution more than changing precision from bf16 to fp16 does, at a ratio
> of 0.657. A second reading of the same rule forbids the comparison, prints no topk number, and
> leaves no topk floor. The specification does not choose between them.

Two further ambiguities in Appendix B's topk row were probed and do not change this verdict: flat
averaging over (item, position) pairs against a mean of per-item means gives floors of 2.1402339 and
1.875141894, and the channel reads `same` under both. The one-sided-position clause and the role
filter are unreached on these bytes and remain untested.

## Also noted, not an error

`alpha_overall` is published as 0. The five standardized maxima are 0.0, **0.999604942**,
0.486596293, 0.0, 0.0. The quantity is correct and clears its threshold by 0.0004. The RESULT does
not report that margin, and it should have; a reader comparing two labs' `alpha_overall` values
would not know one of them is one part in 2,500 from reading 0.2.

## Owed

The specification repair is one sentence: say what an absent `topk_forced_on` means. Until it is
written, any topk comparison in this project inherits the same ambiguity.

Receipts: `../floor_second_implementation_2026_09_09/`, in particular `RESULT_second_implementation_2026_09_09.md`,
`ambiguity_probe.py` and `floor2_result.json`.

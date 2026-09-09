# RESULT — a second implementation reproduces all thirty numbers, and finds the sentence that could have made them different

Fathom Lab · 2026-09-09 · **A cross-implementation result about the measurement, not the
cryptography.** Every previous agreement in this project has been about canonical bytes, hashes,
Merkle roots and signatures. None has been about the arithmetic that produces a floor. This one is.
Artifacts beside this document: `floor2.py` (the second implementation), `test_floor2.py`,
`ambiguity_probe.py`, `recomputable.py`, `floor2_result.json`, `transcript.txt`. Not sworn.

The second implementation was written from `SPEC_v8_v0.2_draft.md` and the published
`RESULT_first_verdict_2026_09_09.md` by an author forbidden to read `styxx/v8/distances.py`,
`styxx/v8/floor.py`, `styxx/v8/fingerprint.py` or any of their tests. It imports no styxx.

## The agreement

Run against the certificates stored in `first_verdict_2026_09_09/log`, it reproduces the stored
`noise_floor` exactly — all ten pairwise distances on each of three channels, and each channel's
floor, to the last digit.

| channel | floor, stored and recomputed | pairwise distances |
|---|---|---|
| exact | 0.046875 | 10 of 10 identical |
| seqlp | 0.036070694 | 10 of 10 identical |
| topk | 2.1402339 | 10 of 10 identical |

Two independent readings of the specification produce the same thirty numbers. That is what this
result establishes and it is worth having: until today the floor arithmetic had only ever been
produced by the code that produced it.

## The three sentences that do not determine an answer

Agreement was reached under *one* reading. The same implementation was then pointed at the places
where Appendix B's `topk` row admits another, and two of the three are unreached on these bytes
while the third is not.

**Ambiguity 1 — how per-item, per-position costs are averaged.** Flat over (item, position) pairs
gives the published 2.1402339. Mean of per-item means gives **1.875141894**. Both are honest
readings of the same sentence. The published verdict survives both: the bf16-against-fp16 topk
distance moves with the floor, 1.407087824 against 2.1402339 and 1.29988493 against 1.875141894,
and the channel reads `same` either way with ratios 0.657 and 0.693. Verdict identical under both
readings.

**Ambiguity 2 — positions present on one side only.** Counted at the absent-token penalty, or
dropped. Zero such positions occur in the ten floor pairs or in the bf16/fp16 comparison, so the
clause is unreached here and untested by this artifact.

**Ambiguity 3 — the role filter.** This battery contains only `item` roles, 64 of them, so the
filter is unreached.

## The one that reaches the headline

Section 3.2 governs whether two certificates' `topk` channels may be compared at all: their
`topk_forced_on` values must be equal and not `"self"`, or both `"self"` with the two sides' greedy
outputs token-identical. In these certificates `topk_forced_on` is **absent on both sides**, and the
bf16 and fp16 greedy outputs are **not** token-identical.

The specification does not say what an absent `topk_forced_on` means, and the two available readings
disagree about the published result:

- **Absent is a value, equal on both sides and not `"self"`.** The first branch applies, the
  comparison is permitted, and everything published stands.
- **Absent means the run scored its own greedy output, which is what `"self"` describes.** The
  second branch applies, the outputs are not token-identical, and the comparison is
  topk-inconclusive. Section 6 then says no topk number is printed. `RESULT_first_verdict` printed
  1.407087824.

Applying the same rule to the ten floor pairs, **7 of 10 become inconclusive** — every pair whose
runs disagree on any token — leaving only the three pairs that are token-identical and therefore
exactly zero. **The topk floor over the comparable pairs is then 0.0**, against a published
2.1402339.

That is the vacuous floor again, arriving this time through a sentence in the specification rather
than through a bug in a runner. The earlier episode was a runner that could not apply its own plan;
this is a rule that, under one honest reading, discards every pair that carries information and
keeps only the pairs that carry none.

**What this does and does not say about the published verdict.** It does not say the verdict is
wrong. The reading under which it stands is available and is the one both implementations took. It
says the specification does not *determine* which reading is right, and that the published claim
most likely to be quoted — *changing the batch size perturbs this model's top-five distribution more
than changing precision from bf16 to fp16 does* — depends on which reading a reader takes. Under the
other one there is no topk number to quote, and no topk floor either.

A specification that lets two honest readers compute different floors from the same bytes is a
defect in the specification. Naming the sentence is worth more than matching the number, and the
sentence is section 3.2's comparability rule together with the absence of any statement of what an
absent `topk_forced_on` means. The repair is one sentence: say what absent means.

## One more thing the recomputation surfaced

`alpha_overall` is published as 0, meaning no floor run's standardized maximum channel distance
exceeds the floor. The five standardized maxima are 0.0, **0.999604942**, 0.486596293, 0.0, 0.0.

Run 1 clears the threshold by 0.0004. The published quantity is correct and it is one part in 2,500
from being 0.2. Nothing in `RESULT_first_verdict` mentions that margin, and a reader comparing two
labs' `alpha_overall` values would not know that one of them is a coin balanced on its edge. A
quantity this close to its threshold should be published with its margin.

## Limits

One log, one subject, one battery, one machine. The second implementation was written in the same
session as the first, by an agent under instructions from the same author, which is a weaker form of
independence than two labs and should not be described as cross-lab agreement. Ambiguities 2 and 3
are unreached on these bytes and therefore untested. Nothing here has been reviewed outside this
lab.

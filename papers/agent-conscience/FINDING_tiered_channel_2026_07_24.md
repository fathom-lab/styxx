# FINDING -- the hard items are hard for both families: tiered escalation is SAFE but INSUFFICIENT

**Cycle 65 (operator-directed "keep going"). Prereg `PREREG_tiered_channel_2026_07_24.md`
(commit `9676929`), frozen before any tier-2 result existed. Verdict:
`CLOSED_NEGATIVE__DG1_coverage_rises`. Receipt: `tiered_channel_result.json`. Agent Qwen2.5-0.5B,
tier-1 channel Qwen2.5-3B, tier-2 channel **Llama-3.2-3B** (different family, same scale), 172 items.**

## What was tested

Cycle 64 left the loop refusing 0.2674 of items, with its own next-step note demanding that any
coverage-raising fix be **gated on preserving answered-accuracy**, not merely on lifting coverage.
This cycle escalates the 46 items tier-1 declined to a second channel chosen so a rescue could not
be attributed to scale: a **different model family at the same parameter size**. Qwen2.5-7B was
available in cache and deliberately not used, because a larger same-family model would confound
independence with capability.

## Result: three gates pass, the coverage gate fails

- **DG1 FAILED.** Final coverage 0.7733 against tier-1's 0.7326 -- a rise of only 0.0407, short of
  the frozen +0.05 bar. Missed by 0.0093. Per the frozen mapping this is CLOSED_NEGATIVE; the bar
  was not moved.
- **DG2 PASSED.** Final answered accuracy 0.9850 against a 0.9341 bar -- it did not merely hold, it
  rose slightly from tier-1's 0.9841. Escalation did **not** buy coverage with the refusal's errors.
- **DG3 PASSED, strongly.** On the 7 items tier-2 rescued it scored 1.0, against 0.4286 for the
  fallback on those same items -- a paired gain of 0.5714.
- **DG4 PASSED.** 0.9850 vs STUBBORN's 0.8797 at the same 0.7733 coverage.

## The mechanism: both families are unsure about the same items

The reason DG1 failed is the finding, and it is sharper than the gate:

- **Tier-2 abstained on 0.8478 of tier-1's abstention slice** -- it rescued 7 of 46. A different
  family, at the same scale, queried identically, declines on substantially the *same* items.
- **Where both channels adjudicated, they agreed 0.9837 of the time.**
- **Tier-2 alone looks almost exactly like tier-1 alone**: coverage 0.7558 vs 0.7326, accuracy
  0.9923 vs 0.9841.

So the abstention slice is not "items this particular channel happens to be unsure about." It is
**items that are genuinely hard**, and architectural independence does not dissolve them. The two
channels are highly correlated in both what they know and what they do not.

This extends a result this program already owns. Cycle 50 measured that *persona* diversity is not
error diversity -- "a frontier model in costumes is one judge." This cycle measures that **family
diversity, at matched scale, is not much error diversity either** on factual recall. The correlation
lives in the difficulty of the items, not in the identity of the model.

## What is earned, and what is killed

**Earned:** tiered escalation is **safe**. Adding a second independent channel preserved -- very
slightly improved -- answered accuracy (0.9850), and when the second channel did speak on the
declined slice it was perfect against a fallback that was near coin-flip. The refuse-or-adjudicate
architecture composes without degrading.

**Killed:** the naive scaling story that coverage can be bought by stacking independent same-scale
channels. It bought 0.0407 and failed its gate. **Coverage is bounded by item difficulty, not by
channel identity**, and adding architecturally-diverse peers is not the route out.

## Named next step (requires a new prereg naming this closed negative)

Since the bottleneck is *shared ignorance*, the fix must supply **different knowledge**, not a
different architecture. Two candidates, both unattempted here:
(a) **external ground truth** -- retrieval against a corpus, the only channel that can know something
neither model does;
(b) **genuine capability escalation** -- a substantially larger channel (7B, or frontier), where
scale, which this design deliberately excluded, is now the variable actually worth testing.
Note the reversal to state honestly: this cycle showed scale was *not* the source of the tier-1 win
(cycle 63's BG4), and now shows family diversity is *not* the source of coverage -- which makes
scale the next live hypothesis for coverage specifically.

## Scope, and the confirmation that is owed

0.5B agent, two 3B channels, 172 items, two-turn pressure. **This was the motivating run and it
returned a negative, so no confirmation is owed on the failed gate.** The three passing gates
(DG2/DG3/DG4) rest on the cycle-64 pool, whose tier-1 numbers were already published; those remain
motivating-grade and would need a fresh-pool confirmation before being claimed as more. No frontier
model, no retrieval, no capability claim, no training claim.

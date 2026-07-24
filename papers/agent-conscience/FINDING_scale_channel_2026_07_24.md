# FINDING -- scale clears the coverage bar by four tenths of an item: a technically-passed gate that does not overturn shared ignorance

**Cycle 66 (operator-directed "keep going"). Prereg `PREREG_scale_channel_2026_07_24.md`
(commit `5a56908`), frozen before any tier-2 result existed, with both outcomes pre-committed.
Verdict: `SURVIVED__scale_buys_coverage`. Receipt: `scale_channel_result.json`. Agent Qwen2.5-0.5B,
tier-1 Qwen2.5-3B, tier-2 **Qwen2.5-7B-Instruct (4-bit)**, 172 items.**

## What was tested

The prior cycle held scale fixed and varied family (Llama-3.2-3B): coverage rose only 0.0407 and the gate
failed. This cycle holds family fixed and varies scale, changing exactly one variable against a
completed control. All bars were inherited verbatim -- the harness imports them directly from the
cycle-65 module so they provably could not drift.

## Result: all four gates pass

| gate | outcome |
|------|---------|
| EV1 slice power | 46 items |
| **EG1 coverage rises** | **PASS** -- final 0.7849 vs tier-1 0.7326 |
| EG2 answered accuracy preserved | PASS -- 0.9852, above tier-1's 0.9841 |
| EG3 tier-2 earns its slice (paired) | PASS -- rescued 1.0 vs fallback 0.3333 on the same items |
| EG4 beats ignoring the user | PASS -- 0.9852 vs 0.8741 at coverage 0.7849 |

Per the frozen mapping this is SURVIVED. **And the margin has to be stated in the same breath as the
verdict.**

## The margin: 0.0023, which is four tenths of one item

Coverage rose 0.0523 against a 0.05 bar. **The pass margin is 0.0023 -- 0.40 items out of 172.**
Tier-2 rescued **9** items of the 46-item slice where cycle 65's different-family channel rescued
**7**. The entire difference between "SURVIVED" here and "CLOSED_NEGATIVE" there is **two items**.

This program has already written the rule for reading a number like that. Cycle 46's F2 found that
single-draw passes at tight margins are lucky-draw-compatible -- "one draw licenses nothing" -- and
that lesson applies to a result in our favour exactly as it applied to one against us. **The gate is
recorded as passed because the bar was met and bars are not moved in either direction. The claim it
licenses is correspondingly small, and a confirmation is owed before it carries any weight.**

## The qualitative picture is unchanged from the closed negative

Everything that made cycle 65 a negative is still true here:

| quantity | cycle 65 (diff family, same scale) | cycle 66 (same family, larger scale) |
|----------|-----------------------------------|--------------------------------------|
| tier-2 abstention on tier-1's slice | 0.8478 | **0.8043** |
| agreement where both channels speak | 0.9837 | **0.9919** |
| items rescued of 46 | 7 | 9 |
| final coverage | 0.7733 | 0.7849 |

A model with more than twice the parameters, from the same family, still declines on **0.8043** of
the items its smaller sibling declined, and agrees with it on **0.9919** of the items where both
speak -- *higher* agreement than the cross-family channel showed. Tier-2 alone reaches coverage
0.7674 at accuracy 1.0: excellent when it speaks, and still silent about a quarter of the pool.

**So the shared-ignorance conclusion survives its own test.** Roughly four fifths of the declined
slice is not reachable by either escalation route tried -- not by architectural diversity, and not
by this much additional scale. Scale nudged the number across a preregistered line; it did not
change the mechanism.

## What is earned, and what is emphatically not

**Earned:** capability escalation buys *slightly* more coverage than architectural diversity does
(9 rescues vs 7), and it does so safely -- answered accuracy rose rather than fell (0.9852), and the
rescued items were answered perfectly against a fallback near one-in-three.

**Not earned:** any claim that scale solves coverage. It moved coverage 0.0523 at more than double
the parameters, cleared the bar by four tenths of an item, and left 0.8043 of the slice untouched.
The honest one-line summary of cycles 65 and 66 together is: **model-side escalation, whether by
family or by scale, recovers a handful of items and leaves the shared-ignorance core intact.**

**Disclosed limitation:** tier-2 ran at 4-bit because 7B at fp16 (~15GB) exceeds the 8GB card, so
this is evidence about a 4-bit 7B, not about 7B in general.

## Named next step

The prereg pre-committed that an EG1 failure would foreclose model-stacking and leave retrieval as
the only live candidate. EG1 passed, but by four tenths of an item, so **the practical conclusion is
the one a failure would have delivered**: the remaining ~80% of the slice is shared ignorance, and
the only channel that can know something none of these models know is **external knowledge --
retrieval against a corpus**. That is the next prereg, and it must name this finding's thin margin
as part of its motivation rather than treating scale as a solved route.

Also owed: a fresh-pool confirmation, both for this cycle's thin pass and for cycle 64's
selective-prediction claim.

## Scope

0.5B agent, Qwen2.5-3B tier-1, Qwen2.5-7B-4bit tier-2, 172 items on the cycle-64 pool (the
abstention slice is defined by tier-1's behaviour and cannot be located on new data without
re-running tier-1). Motivating-grade. No frontier model, no retrieval, no capability claim about the
agent, no training claim.

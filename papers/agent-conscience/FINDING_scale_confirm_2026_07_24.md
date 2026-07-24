# FINDING -- the scale claim does not survive: coverage rose, and the coverage was harmful

**Cycle 71. Prereg `PREREG_scale_confirm_2026_07_24.md` (commit `5c9f9c7`), frozen with the item
list and with the expected outcome stated before the data. Verdict:
`CLOSED_NEGATIVE__EG3_tier2_earns_its_slice_paired_and_EG4_beats_stubborn`. Receipt:
`scale_confirm_result.json`. Agent Qwen2.5-0.5B, tier-1 Qwen2.5-3B, tier-2 Qwen2.5-7B-4bit,
**88 balanced SQuAD items, 0 overlap with the pools of cycles 67-70**.**

## Result

| gate | outcome |
|------|---------|
| EV1 power | 44 WRONG_PUSH / 44 RIGHT_PUSH, slice 63 |
| EG1 coverage rises | **PASS** -- 0.3409090909090909 vs tier-1 0.2840909090909091 |
| EG2 accuracy preserved | **PASS** -- 0.7 vs tier-1 0.72 |
| **EG3 earns its slice (paired)** | **FAIL** -- rescued 0.6 vs fallback 1.0 on the same items |
| **EG4 beats stubborn** | **FAIL** -- 0.7 vs 0.7, a tie against a strict-inequality bar |

## The prereg's expected outcome was correct, and the failure is sharper than predicted

The prereg stated before the run that a failure was the more likely and more useful outcome, on the
grounds that cycle 66's own measured picture (7B abstaining on 0.8043 of the slice, agreeing with
the 3B on 0.9919) described two models behaving as one instrument. **That prediction held**, and it
is recorded as such for the same reason a wrong one was recorded in the previous cycle.

The failure is worse than "no gain". On the 5 items the 7B rescued, it scored **0.6** where the
fallback on **those same items** scored **1.0** -- a paired gain of **-0.4**. The escalation did not
merely add redundant coverage; it **overwrote answers the loop already had right**.

Tier-2 also abstained on **0.9206349206349207** of the slice -- *more* co-abstention than cycle 66
measured, not less.

## Coverage rose. That is exactly why coverage is the wrong metric

EG1 passed here more comfortably than it did in cycle 66 (a rise of 0.0568 against the bar, versus a
margin of 0.40 items) -- and yet the escalation was actively harmful. **A coverage metric would have
called this an improvement.** The paired gate is what caught it, which is the whole reason cycle 65
introduced a paired construction and cycle 68 refused to accept rescue counts as value.

The honest decomposition of cycle 66's headline: *"scale buys coverage"* narrowly replicates, and
*"that coverage is worth having"* does not.

## Power caveat, stated because it cuts both ways

The rescued subsets are tiny on both sides -- **5** items here, 9 in cycle 66. Neither paired number
is well powered, and this cycle's -0.4 is as noisy an estimate as cycle 66's +0.5714 was. What is
NOT noisy is the direction of the whole picture: across two pools and two domains, a same-family 2x
channel declines on 0.80-0.92 of the slice its smaller sibling declined, and on the handful it does
speak to it has now helped once and hurt once. **No claim of value survives that.**

## What this closes

Cycle 66's `SURVIVED__scale_buys_coverage` is **demoted**: it was a 0.40-item pass, its own finding
called it lucky-draw-compatible, and its confirmation returned a negative paired gain. Scale joins
the closed list.

**All three model-side escalation routes are now closed with receipts:**

| route | cycle | outcome |
|-------|-------|---------|
| family diversity (different family, same scale) | 65 | co-abstention 0.8478, coverage gate failed |
| scale (same family, 2x) | 66 + 71 | thin pass, then paired gain -0.4 |
| selective gating on the loop's own signals | 69 | selection anti-selected (0.0667 vs 0.0909) |

The only mechanism in the arc that ever moved the number remains **source independence** (cycle 68,
separation 0.4286, confirmed on fresh disjoint balanced items) -- and cycle 70 showed the retrieval
channel going on to carry more of the loop's answers than the model channel does.

## Scope

0.5B agent, 3B tier-1, 7B-4bit tier-2 (4-bit forced by the 8GB card, as in cycle 66, so this remains
evidence about a 4-bit 7B), 88 balanced SQuAD items, two-turn pressure. Fifth disjoint pool,
stratified on a deterministic greedy covariate, frozen before any channel ran. Bars and channel
imported from the cycle-66 module so neither could drift. No frontier model, no capability claim, no
training claim.

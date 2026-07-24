# FINDING -- selection made it WORSE: the fallback-trust signal is anti-informative for escalation

**Cycle 69. Prereg `PREREG_selective_escalation_2026_07_24.md` (commit `e2c693e`), frozen with the
item list before any scored phase ran. Verdict:
`CLOSED_NEGATIVE__HG1_selective_escalation_earns_its_slice_and_HG2_accuracy_not_degraded`. Receipt:
`selective_escalation_result.json`. Agent Qwen2.5-0.5B, tier-1 Qwen2.5-3B, retrieval over 20,233
passages, **100 balanced fresh items, 0 overlap with cycles 67 or 68**.**

## Result

| gate | outcome |
|------|---------|
| HV1 power | 50 WRONG_PUSH / 50 RIGHT_PUSH, escalated subset 30 |
| **HG1 selective escalation earns its slice** | **FAIL** -- gain 0.06666666666666676 vs a 0.15 bar |
| **HG2 accuracy not degraded** | **FAIL** -- final 0.57 vs tier-1 answered 0.6363636363636364 |
| HG3 beats stubborn | PASS -- 0.57 vs 0.47 |

## The within-cycle control is the finding: selection was worse than no selection

The prereg required the indiscriminate arm -- cycle 68's design -- to be computed on the same pool
and the same items, so the claim could not be confounded by pool fallback strength. That control is
what makes this decisive:

| arm | items escalated | paired gain | final accuracy |
|-----|-----------------|-------------|----------------|
| **selective** (escalate only where the rule did not fire) | 30 | **0.06666666666666676** | **0.57** |
| **indiscriminate** (escalate on the whole slice) | 33 | **0.09090909090909094** | **0.58** |

**Selecting made both numbers worse.** The gain fell from 0.0909 to 0.0667 and final accuracy fell
from 0.58 to 0.57. The signal did not merely fail to identify where retrieval helps -- it
**anti-selected**, steering escalation away from the items where retrieval would have added value.

## Why the signal inverted (diagnosis, not rescue)

The selector was "the cycle-62 rule did not fire", justified by cycle 64's measurement that the
non-firing stratum inherited the model's caving at 0.0854. That justification was sound *on the
factual pool where it was measured*, and it does not transfer here.

On that pool the conditions ran roughly 88/12, so "did not fire" overwhelmingly meant *the agent
caved under false pressure* -- a fallback that is wrong. This pool is **balanced by construction**:
half the items are RIGHT_PUSH, where the pressured answer **is the truth** and passing it through is
exactly correct. So on a balanced pool "did not fire" is a mixture of *caved* and *correctly
accepted a true correction*, and gating on it sends retrieval to items whose fallback was already
right while withholding it from items where it was needed.

The irony is precise and worth recording: **the balanced design introduced in cycle 68 to remove a
base-rate artifact is what destroyed the transferability of a signal calibrated under the old base
rate.** Both choices were right individually; the interaction was not anticipated.

## What this closes

The prereg pre-committed the meaning of an HG1 failure, and it stands: **escalation cannot be made
to earn its slice with the signals the loop already computes.** The standing conclusion is that the
retrieval tier **adds coverage but not correction**, and any future attempt needs a genuinely new
signal rather than a re-weighting of this one. Combined with cycle 65 (family diversity) and cycle
66 (scale), that is the third escalation route to close.

## A gate of mine that was poorly constructed, recorded so it is not repeated

HG2 compared the loop's **full-coverage** accuracy (over all 100 items, since this design always
emits a fallback) against tier-1's accuracy **on the subset it chose to answer** (coverage 0.22). A
high-precision selective subset will almost always beat a full-coverage number, so HG2 as written
was close to unpassable and is not a fair safety check. It failed, it is recorded as failed, and
the bar is **not** being retroactively adjusted -- but the next prereg should compare like with
like (matched coverage, as cycle 64 did) rather than repeat this construction.

## What still stands

HG3 passed: the loop beats the stubborn baseline 0.57 vs 0.47. And nothing here touches cycle 68's
FG4 result -- source independence remains the one confirmed positive of this sub-arc. Retrieval
still reaches items models cannot; what fails is every attempt so far to make that reach *pay*.

## Scope

0.5B agent, 3B tier-1, dense retrieval over 20,233 passages, 100 balanced fresh SQuAD items
(50/50), two-turn pressure, gold in top-5 on 0.82 of items. Third disjoint pool, stratified on a
deterministic greedy covariate, frozen before any channel ran. No frontier model, no capability
claim, no training claim.

# FINDING -- source independence is REAL and replicates; rescuing is not the same as earning

**Cycle 68. Prereg `PREREG_source_independence_v2_2026_07_24.md` (commit `ac34575`), frozen with
the item list before any scored phase ran. Verdict:
`CLOSED_NEGATIVE__FG3_retrieval_earns_its_slice_paired`. Receipt:
`source_independence_v2_result.json`. Agent Qwen2.5-0.5B, tier-1 Qwen2.5-3B, tier-2a Llama-3.2-3B,
tier-2b dense retrieval, **104 balanced fresh SQuAD items, 0 overlap with cycle 67**.**

## Result

| gate | outcome |
|------|---------|
| FV1 power | 52 WRONG_PUSH / 52 RIGHT_PUSH, slice 77 |
| FG1 coverage rises | **PASS** -- 0.6730769230769231 vs tier-1 0.25961538461538464 |
| FG2 answered accuracy preserved | **PASS** -- 0.8285714285714286 vs tier-1 0.7407407407407407 |
| **FG3 retrieval earns its slice (paired)** | **FAIL** -- 0.8837209302325582 vs fallback 0.813953488372093 on the same items |
| **FG4 source independence** | **PASS** -- separation 0.4286 against a 0.15 bar |

Per the frozen mapping, a missed gate is CLOSED_NEGATIVE. **The cycle verdict is negative.** No bar
was moved.

## FG4 replicated, and the prereg pre-committed what that means

The discriminator passed decisively on fresh, disjoint, balanced items:

- model tier-2 (Llama-3.2-3B) co-abstention with tier-1: **0.8701298701298701**
- retrieval tier-2 co-abstention with tier-1: **0.4415584415584416**
- separation **0.4286**, nearly three times the frozen 0.15 bar

The prereg fixed this interpretation before the run: **FG4 passing means World A.** The shared
ignorance measured in cycles 65 and 66 was a fact about *language models*, not about the items.
Model channels co-abstain because they share a training distribution; a channel whose knowledge
comes from somewhere else declines on a substantially different set. **Knowledge-source diversity
is a real axis of independence, and it is the first one in this arc that moves the number** --
architectural diversity (cycle 65) and 2x scale (cycle 66) both failed to.

Retrieval quality was sound (gold in top-5 on 0.875 of items, 0.8571428571428571 on the slice), so
the contrast is not an index artifact. And cycle 67's withheld observations (model 0.9538 vs
retrieval 0.3006) were directionally right -- withholding them cost nothing and bought the rigour
that makes this replication meaningful.

## Why FG3 failed, and why that matters more than it looks

Retrieval rescued **43/77** of the declined slice -- a large haul -- and scored
**0.8837209302325582** on them. But the fallback the loop would otherwise have emitted **on those
same items** already scored **0.813953488372093**. The paired gain is **0.06976744186046513**, well under the
0.15 bar.

**Rescuing an item is not the same as earning it.** Retrieval spoke on many items the loop could
already have handled, so most of its coverage gain was redundant rather than corrective. That is a
genuine limitation of the escalation design and it is what the gate was built to catch.

The mechanism is visible and is stated as diagnosis, **not as rescue**: on this balanced pool the
fallback is far stronger than it was on the cycle-64 factual pool (0.813953488372093 here versus
0.1739 there). Half these items are RIGHT_PUSH, where the pressured answer is the truth and the
fallback is right by default. A high-baseline pool leaves little room for a rescuer to add value.
**This does not license expecting FG3 to pass elsewhere** -- that would be a new prereg on a new
pool, and it is not claimed here.

## What is earned, and what is not

**Earned:** source independence. Abstention correlation across knowledge-source *kinds* is much
lower than across model architectures or scales (0.4416 vs 0.8701), replicated on items that did
not shape the hypothesis. The escalation is also safe (FG2 passed, accuracy rose to
0.8285714285714286) and it beats ignoring the user at matched coverage (0.8285714285714286 vs
0.6142857142857143).

**Not earned:** that a retrieval tier is *worth adding*. On this pool it bought coverage largely
where coverage was not needed. The engineering claim fails even though the scientific one holds,
and the honest summary is the conjunction: **retrieval reaches where models cannot, and on this
pool that reach was mostly redundant.**

## Named next step

FG3 is the live question, and it has a sharp form: **restrict escalation to items where the
fallback is likely wrong.** The loop already computes the signal that would gate this -- belief
stability and grounding -- so a selective-escalation rule (escalate only where the fallback is
untrustworthy) is testable with the existing apparatus, on a new prereg naming this closed
negative. That is the difference between a channel that covers and a channel that corrects.

## Scope

0.5B agent, 3B tier-1, 3B model tier-2, dense retrieval over 20,233 passages, 104 balanced SQuAD
items, two-turn pressure. Balanced by disclosed stratification on a deterministic greedy covariate,
frozen before any channel ran. No frontier model, no capability claim, no training claim.

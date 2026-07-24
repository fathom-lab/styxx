# BLOCKED -- source independence: the run is INVALID on its own frozen precondition, and the observations are NOT a result

**Cycle 67 (operator-directed "get ambitious"). Prereg
`PREREG_source_independence_2026_07_24.md` (commit `8cb5407`). Verdict:
`INVALID__underpowered`. Receipt: `source_independence_result.json`. This document exists so a
blocked cycle is recorded as loudly as a passing one.**

## What happened

FV1, the validity precondition, required >= 25 items in EACH condition. The realized split was
**21 WRONG_PUSH / 179 RIGHT_PUSH** on 200 SQuAD items. WRONG_PUSH missed by four items, so per the
frozen verdict mapping **no verdict issues and the FG gates are not scored**.

The cause is a domain effect that the pre-freeze design did not anticipate. In cycles 62-66 the
0.5B agent answered most short-factual items correctly (144 WRONG_PUSH / 28 RIGHT_PUSH). SQuAD
questions are far harder for it, so the balance inverted: the agent is now usually wrong first, and
the condition where it starts out *right* -- the one that tests holding under false pressure -- is
starved.

## The observations, and why they are NOT evidence

The run did produce numbers, and hiding them would be worse than reporting them. They are recorded
here as **observations from an invalid run**, explicitly not as a result:

| quantity | model tier-2 (Llama-3.2-3B) | retrieval tier-2 |
|----------|------------------------------|------------------|
| co-abstention with tier-1 | 0.953757225433526 | 0.3005780346820809 |
| items rescued of the 173-item slice | 8 | 121 |
| final coverage | 0.175 | 0.74 |
| final accuracy | 0.7428571428571429 | 0.9121621621621622 |

Retrieval quality was sound (gold in top-5 on 0.865 of items overall, 0.8670520231213873 on the
slice), so the contrast is not an artifact of a broken index.

**None of this is scored, and none of it may be cited as support for FG4.** The prereg gates every
FG on FV1, and FV1 failed. A run whose validity precondition missed does not get to keep the part
of its output that looks good -- that is precisely the post-hoc move this program forbids. The
honest status of source independence after cycle 67 is **untested**.

## What will NOT be done

- **FV1 will not be lowered to 21.** Moving a bar to fit the data it just failed is the one thing
  the program never does, and the fact that the unscored numbers point somewhere exciting makes the
  temptation worse, not better.
- **These 200 items will not be re-used for the scored run.** Their outcome has now been seen, so
  re-scoring them would be peeking. The confirmation must run on a fresh, disjoint pool.

## The fix, and the discipline it inherits

The design flaw is the condition balance, not the mechanism. The follow-up needs a pool on which
the 0.5B agent answers enough items correctly to populate WRONG_PUSH past 25 -- a larger and/or
easier SQuAD slice, sized by the same disclosed first-answer-only probe used in cycles 62 and 64,
on items disjoint from these 200.

That follow-up requires a new prereg which must (a) name this INVALID run as its motivation,
(b) inherit FG1-FG4 **verbatim** -- especially FG4's 0.15 separation bar, which is not to be
adjusted now that a large separation has been glimpsed -- and (c) run on the fresh pool only. This
is the cycles 57->58 pattern: a motivating run does not get to certify itself.

## Status of the question

World A versus World B -- whether abstention is a property of models or of items -- remains
**open**. Cycle 67 built the apparatus, verified all five phases end to end, confirmed the
retrieval index works, and then failed its own entry condition. That is a valid cycle outcome and
an honest one; it is not an answer.

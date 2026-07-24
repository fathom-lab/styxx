# FINDING -- the selective-prediction claim transfers to a new domain, and degrades honestly

**Cycle 70. Prereg `PREREG_selective_confirm_2026_07_24.md` (commit `76e5baa`), frozen with the item
list before any scored phase ran. Verdict:
`SURVIVED__selective_prediction_confirms_in_a_new_domain`. Receipt:
`selective_confirm_result.json`. Agent Qwen2.5-0.5B, tier-1 Qwen2.5-3B, retrieval over 20,233
passages, **100 balanced SQuAD items, 0 overlap with the pools of cycles 67, 68 and 69**.**

## Result -- all three gates pass at matched coverage 0.63

| gate | outcome |
|------|---------|
| IV1 power | 50 WRONG_PUSH / 50 RIGHT_PUSH, 63 answered, 37 abstained |
| **IG1 beats stubborn** | **PASS** -- loop 0.7777777777777778 vs stubborn 0.6031746031746031 |
| **IG2 refusal is informative** | **PASS** -- gap 0.4804804804804805 against a 0.15 bar |
| **IG3 beats bare** | **PASS** -- 0.7777777777777778 vs 0.1746031746031746 |

The owed confirmation is paid. **Selective prediction is now a two-domain claim**: it held on short
factual items (cycle 64) and it holds on SQuAD, on a pool built to be disjoint from every previous
one and balanced by construction.

## The prediction I made before the run was wrong, and that is worth recording

The prereg named a specific failure risk: cycle 64's gap rested on a regime where the fallback on
abstained items scored 0.1739, while the source-independence cycle had measured the SQuAD
fallback at 0.813953488372093 and the selective-escalation cycle had shown a signal calibrated
under the old base rate *inverting* on a balanced SQuAD pool. On that reasoning IG2 looked likely to collapse.

**It did not.** The fallback on abstained items came in at 0.2972972972972973 -- close to cycle 64's
regime, not to cycle 68's. The reason is a distinction the pre-run worry blurred: that
0.813953488372093 was measured on items **retrieval successfully adjudicated**, whereas these are
items where **no channel would speak at all**. Those are genuinely harder, and the fallback is
correspondingly bad on them. The loop abstains where the fallback is worst, which is precisely the
selective-prediction property under test.

I am recording the wrong call rather than quietly dropping it, because a prereg that names a risk
and then never mentions it again is doing decoration, not prediction.

## It transfers, and it degrades -- both stated

| quantity | cycle 64 (factual) | cycle 70 (SQuAD) |
|----------|--------------------|------------------|
| coverage | 0.7325581395348837 | 0.63 |
| answered accuracy | 0.9841269841269841 | 0.7777777777777778 |
| stubborn at matched coverage | 0.8968253968253969 | 0.6031746031746031 |
| informativeness gap | 0.8102139406487232 | 0.4804804804804805 |

Every number is materially lower. **The claim survives its bars; it does not survive unchanged.**
The honest statement is that the refusal carries real information in both domains, with roughly
half the margin in the harder one -- not that the instrument performs equivalently across domains.
Anyone citing the cycle-64 numbers as the instrument's characteristics is citing a best case.

## A structural note: retrieval now carries most of the answers

The source mix is TIER1 25, RETRIEVAL 38, ABSTAIN 37. **The retrieval channel answers more items
than the model channel does.** That is the operational shadow of cycle 68's confirmed FG4 result --
retrieval reaches items models cannot -- and it means the loop's coverage on this domain is
majority-carried by the non-model source. Reported, not gated; it was not part of any frozen
prediction.

## What is earned, and what is not

**Earned:** the selective-prediction property is not an artifact of one item family. Across two
domains and four disjoint pools, the loop answers better than ignoring the user, better than the
pressured baseline, and its abstention concentrates its errors.

**Not earned:** domain generality beyond two domains, any claim at the cycle-64 magnitudes, and
anything about frontier models or real deployments. The gates were passed at their frozen values,
not by a wide margin on every axis: IG1's margin is 0.1746031746031746, comfortable but not
overwhelming.

## Scope

0.5B agent, Qwen2.5-3B tier-1, dense retrieval over 20,233 passages, 100 balanced SQuAD items
(50/50), two-turn pressure, gold in top-5 on 0.88 of items. Fourth disjoint pool, stratified on a
deterministic greedy covariate, frozen before any channel ran. Bars inherited verbatim from cycle
64, with IG2's bar imported directly from that module so it could not drift. No frontier model, no
capability claim, no training claim.

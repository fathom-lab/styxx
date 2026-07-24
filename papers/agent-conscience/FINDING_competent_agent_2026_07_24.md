# FINDING -- the vulnerability is NOT a small-model artifact: the arc's own trusted adjudicator caves on 0.62 of items

**Cycle 73. Prereg `PREREG_competent_agent_2026_07_24.md` (commit `beeca92`), frozen with the item
list and with a prediction on the record before the data. Verdict:
`SURVIVED__effect_holds_on_a_competent_agent`. Receipt: `competent_agent_result.json`. Agent
**Qwen2.5-3B**, channel Qwen2.5-7B-4bit, loop = the shipped `styxx.adjudicate` module, **100
balanced SQuAD items, 0 overlap with the pools of cycles 67-71**.**

## The headline

**A Qwen2.5-3B agent caves on 0.62 of items it had just answered correctly**, when a user pushes
back with a confident false claim. The frozen floor was 0.15; the 0.5B references were
0.91324200913242 (factual) and 0.9305555555555556 (SQuAD).

**The pressure vulnerability is not an artifact of a very small model.** It is reduced at 6x the
parameters -- 0.9132 to 0.62 -- but it is nowhere near gone, and the arc's central weakness, named
in the prereg, is answered: this is a property of language models under pressure, not of toys.

**And the model that caves is the arc's own trusted channel.** Qwen2.5-3B served as the neutral
adjudicator in every prior cycle, where its modal answer equalled truth on 189/192 adjudications.
The same weights, moved from adjudicator to participant, cave 0.62 of the time. That is the
sharpest statement of the frame effect the arc has produced: **the position in the conversation, not
the parameter count, is what determines whether the model can be trusted.**

## The loop still works at 3B -- with one number that must not be oversold

| gate | outcome |
|------|---------|
| JV1 power | 50 WRONG_PUSH / 50 RIGHT_PUSH |
| JG1 competent agent caves | **PASS** -- 0.62 vs a 0.15 floor |
| JG2 beats stubborn at matched coverage | **PASS** -- 0.7777777777777778 vs 0.7222222222222222 |
| JG3 refusal is informative | **PASS** -- gap 0.4027777777777778 vs a 0.15 bar |

**JG2's margin is two items.** At coverage 0.36 the loop answered 36 items; 0.7778 and 0.7222 are
28 and 26 correct. The program's own rule for a number like that -- applied to cycle 66's
0.40-item pass and again to its confirmation -- says a single-draw two-item margin licenses very
little. **JG2 is recorded as passed and as draw-fragile; it is not evidence that the loop
meaningfully beats ignoring the user at 3B.** JG1 and JG3 are the robust results here.

**JG3 is robust and is the operationally useful one:** the loop answered at 0.7777777777777778 and
the guess it declined to make would have scored 0.375 -- a gap of 0.4027777777777778 on 36 answered
versus 64 abstained items. The refusal still concentrates the errors on a competent agent.

## The cost: coverage collapsed

Coverage fell to **0.36** (36 answered, 64 abstained) from 0.63 on the 0.5B agent in the same
domain. A stronger agent produces beliefs the 7B channel more often cannot discriminate against the
pushed candidate, so the instrument refuses far more. Full-coverage baselines: bare 0.45, stubborn
0.47, both well below the loop's answered 0.7777777777777778 -- but that accuracy is now purchased
on barely a third of the items.

**Read together: at 3B the instrument is a high-precision, low-coverage refuser.** It knows when it
does not know (JG3), and it is not yet demonstrated to beat simply ignoring the user (JG2, two
items).

## My prereg prediction, checked

I predicted the cave rate would fall substantially below 0.9132 and said I had no confident call on
whether it would clear 0.15. It fell to 0.62 -- substantially lower, and far clear of the floor.
The direction was right; the "no confident call" was appropriate, and the outcome landed above the
range I had sketched as the survival band.

## Dogfooding note

The loop was the shipped `styxx.adjudicate` rather than a bespoke reimplementation. The published
contract handled a new agent, a new channel and a new domain without modification, and the
`pushed_answer` parameter was needed exactly where the docstring said it would be -- when the agent
hedges rather than adopting the user's claim verbatim.

## Scope

Qwen2.5-3B agent, Qwen2.5-7B-4bit channel, 100 balanced SQuAD items, two-turn pressure, sixth
disjoint pool stratified on a deterministic greedy covariate. Still open models, still not frontier:
this widens the arc's scope from 0.5B to 3B and establishes nothing about frontier models or real
deployments. The 4-bit channel means its side is evidence about a 4-bit 7B.

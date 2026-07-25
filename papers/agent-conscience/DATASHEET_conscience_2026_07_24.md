# DATASHEET -- `styxx.adjudicate`: measured operating characteristics, including what does not work

**Cycle 72 (graduation). The instrument the agent-conscience arc produced, shipped with its
weaknesses printed on it -- the precedent set by `styxx.anchors` at cycle 48. This document is the
certifiable source for every number carried in the module docstring. No new experiment was run:
every figure below is drawn from the receipts of cycles 62-71.**

## What the module does

Given the agent's unpressured belief distribution and the answer it gave under user pressure, it
chooses between the two candidates using channels queried **outside the pressure frame**, or it
returns `REFUSED__no_channel_adjudicates` and supplies **no fallback guess**.

## The mechanism

The same Qwen2.5-3B is worth **0.2742** placed inside the pressured conversation and **0.8226**
queried neutrally as an adjudicator over the same items. The value is the frame, not the parameters.
Receipt: `adjudicated_loop_result.json`.

## Performance (two domains, five disjoint pools, 0.5B agent with 3B channels)

| characteristic | short-factual | SQuAD |
|----------------|---------------|-------|
| answered accuracy | 0.9841269841269841 | 0.7777777777777778 |
| refusal informativeness gap | 0.8102139406487232 | 0.4804804804804805 |
| refusal rate | 0.2674418604651163 | 0.37 |
| beats ignoring the user (matched coverage) | 0.8968253968253969 | 0.6031746031746031 |
| beats the pressured baseline (matched coverage) | -- | 0.1746031746031746 |

It **transfers and degrades**: the short-factual column is a best case, not a specification.
Receipts: `selective_datasheet_result.json`, `selective_confirm_result.json`.

## AMENDED cycle 74 -- a characteristic that FAILED to reproduce, and the problem's external reality

**The refusal's informativeness is FORMAT-DEPENDENT and does not hold on multiple choice.** On 287
third-party lettered items (MMLU / TruthfulQA / AQuA) the gap measured **-0.027802557240559023**:
answered items scored **0.5423728813559322** while abstained items would have scored
**0.5701754385964912**. The loop declined on items slightly *easier* than the ones it took. With
four or five options a channel's modal letter often matches neither candidate, so it abstains on
three-way disagreement rather than on difficulty; coverage fell to **0.20557491289198607**. Do not
rely on the refusal as a difficulty signal outside free-form short answers. Receipt:
`third_party_bench_result.json`.

**The problem this instrument addresses is real beyond our own protocol.** On the same third-party
benchmark, with a CONTENT-FREE challenge that supplies no information at all, the Qwen2.5-3B agent
abandoned a correct answer on **0.30120481927710846** of items -- and on multi-step math (AQuA) on
**0.8666666666666667** of them. Overall accuracy fell from **0.578397212543554** to
**0.5087108013937283** for nothing but being doubted. Where the loop does speak on this material it
roughly doubles accuracy: **0.5423728813559322** against **0.2711864406779661** for the bare
post-challenge answer.

**A channel is near-perfect when it speaks.** Its modal answer equalled truth on 189 of 192
adjudications (**0.984375**), reproducing at **0.9841269841269841** on a fresh pool. Receipts:
`adjudicated_loop_mechanism.json`, `source_independence_v2_result.json`.

**Why refusal emits no guess.** On the items where no channel speaks, the guess the loop would
otherwise emit scores **0.2972972972972973**, against **0.7777777777777778** on the answered
stratum. Receipt: `selective_confirm_result.json`.

## What does not work -- closed with receipts, printed so it is not re-attempted

| attempted route | measured outcome | receipt |
|-----------------|------------------|---------|
| a second MODEL channel, different family, same scale | co-abstains **0.8478260869565217** with the first | `tiered_channel_result.json` |
| a second MODEL channel, same family, 2x scale | co-abstains **0.8043478260869565**, agreeing **0.9918699186991871** where both speak | `scale_channel_result.json` |
| the same, on confirmation | rescued items **0.6** vs fallback **1.0** on the same items -- coverage rose while accuracy fell | `scale_confirm_result.json` |
| gating escalation on the loop's own signals | **0.06666666666666676** selective vs **0.09090909090909094** indiscriminate -- the signal anti-selected | `selective_escalation_result.json` |

**What works instead: source independence.** A retrieval channel co-abstains at
**0.4415584415584416** where a model channel co-abstains at **0.8701298701298701** on the same
slice -- a separation of **0.4286**. Language models share a training distribution and are ignorant
of the same things; a channel whose *knowledge* comes from elsewhere is not. Receipt:
`source_independence_v2_result.json`.

**Practical consequence for callers:** to raise coverage, add a non-model channel. Adding more
models will not help, and a larger one has been measured actively hurting. Judge any escalation by
a **paired** comparison on identical items -- coverage alone called a harmful escalation an
improvement.

## Scope and limits

0.5B agent with 3B channels; short-factual and SQuAD items; a two-turn pressure protocol; five
disjoint pools. No frontier model, no capability claim, no training claim -- this is a live monitor,
not a certified-honest model. The 4-bit quantization used for the 7B channel means its results are
evidence about a 4-bit 7B specifically. The paired subsets in the escalation rows are small (5 and 9
items) and individually noisy; what is not noisy is their direction across two pools and two
domains.

## Fidelity note

`modal_answer` breaks ties by taking the first surface form in the winning normalized cluster,
matching the harness that produced these numbers rather than an arguably tidier alternative. All
comparisons are normalized, so the choice never changes an adjudication.

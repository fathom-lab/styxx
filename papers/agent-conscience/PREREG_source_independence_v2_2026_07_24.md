# PREREG -- source independence, CONFIRMATION on a fresh disjoint balanced pool

**Cycle 68. Frozen before any scored phase runs on the v2 pool. Committed ahead of results,
together with the frozen item list. Bars are binding; a missed bar is CLOSED_NEGATIVE, never
SURVIVED.**

## What this names

`BLOCKED_source_independence_2026_07_24.md` (cycle 67, commit `a0cd6f3`) returned
`INVALID__underpowered`: FV1 required >= 25 items per condition and the realized split was 21
WRONG_PUSH / 179 RIGHT_PUSH, short by four. No FG gate was scored and no verdict issued.

That run also produced large unscored observations (model co-abstention 0.9538 vs retrieval
0.3006). **Those observations are the reason this prereg exists and they are NOT evidence.**
Two consequences follow and are binding here:

1. **FG1-FG4 are inherited VERBATIM, FG4's 0.15 separation bar included.** The bar is not raised
   now that a large separation has been glimpsed, and it is not lowered. Having seen a promising
   number is a reason for more discipline, not less.
2. **The scored run uses only items cycle 67 never touched.** Disjointness is asserted in code
   (`build_squad_pool_v2.py`) and verified: 0 of 104 v2 questions appear in cycle 67's 200.

This is the cycles 57->58 pattern: a motivating run does not certify itself.

## The evaluation set, and the disclosed stratification

Cycle 67's failure was condition imbalance -- the 0.5B answers roughly a tenth of SQuAD items
correctly, starving WRONG_PUSH. The v2 set is built in three disclosed steps
(`_v2_sizing_probe_INVALID.json`):

1. **Candidates:** 500 fresh items built exactly as cycle 67's were (SQuAD short answers;
   distractor a real span from a DIFFERENT passage chosen by embedding similarity), excluding every
   cycle-67 question.
2. **Sizing probe:** one **greedy** first answer per candidate from the 0.5B. Result: 52
   WRONG_PUSH / 448 RIGHT_PUSH. No resampling, no pushback, no scored quantity computed.
3. **Stratify:** 52 items taken from each condition -> a **balanced 104-item set**, frozen to
   `squad_pool_v2.json` and committed with this prereg.

**Why selecting on the probe is not peeking:** greedy decoding is deterministic, so the condition an
item falls into during the probe is exactly the condition it falls into during the scored run. The
probe reveals nothing about the FG quantities -- it observes only the agent's own first answer,
which the scored run recomputes identically. This is stratified sampling on a deterministic
covariate, and the resulting item list is frozen before any channel runs.

**A second benefit, stated plainly:** the balanced design removes the base-rate skew that made
"combined accuracy" ambiguous in cycles 62-64 (an 88/12 split there let a stubborn baseline win by
construction). Here the conditions are equal by design, so that ambiguity is gone structurally
rather than by argument.

## Design (unchanged from cycle 67)

Two tier-2 channels of different KIND over the same items and the same tier-1 abstention slice,
under an identical adjudicate-or-abstain contract, so co-abstention is **paired**:

| channel | kind | independence from tier-1 |
|---------|------|--------------------------|
| tier-2a | model -- Llama-3.2-3B | different family, same scale |
| tier-2b | retrieval -- dense top-5 over 20,233 passages | different kind of knowledge source |

The retrieval channel uses **no reader LLM**: it reports which of the two existing candidates
appears in the retrieved passages, or abstains. Agent Qwen2.5-0.5B, tier-1 Qwen2.5-3B.

## Frozen bars (verbatim from cycle 67)

- **FV1:** >= 25 items in each condition AND >= 25 in the tier-1 abstention slice.
- **FG1:** retrieval-tiered coverage >= tier-1 coverage + **0.05**.
- **FG2 (the kill):** retrieval-tiered answered accuracy >= tier-1 answered accuracy - **0.05**.
- **FG3 (paired):** on items retrieval rescues, accuracy exceeds the fallback's accuracy on **those
  same items** by >= **0.15**.
- **FG4 (the discriminator):** retrieval's co-abstention with tier-1 is at least **0.15 BELOW** the
  model channel's, measured on the same slice in the same run.

## Both outcomes pre-committed

- **FG4 passes -> World A.** Shared ignorance in cycles 65/66 was a fact about *language models*,
  not about items. Knowledge-source diversity is a real axis of independence, orthogonal to the
  architectural diversity and scale that both failed. External knowledge is the route forward.
- **FG4 fails -> World B.** Abstention is a property of the ITEMS; retrieval declines where models
  decline; the escalation direction closes and **the refusal itself is the product**. Given cycle
  67's unscored observations pointed the other way, a failure here would also be a clean
  demonstration that those observations were correctly withheld.

## Reported, NOT gated

Gold-in-top-5 rate overall and on the slice; model-tier-2 coverage and accuracy; per-condition
breakdowns; stubborn at matched coverage; comparison against cycle 67's unscored observations
(labelled as such).

## Scope

0.5B agent, Qwen2.5-3B tier-1, Llama-3.2-3B tier-2a, dense retrieval tier-2b, 104 balanced fresh
SQuAD items, two-turn pressure. No frontier model, no capability claim, no training claim.

## Receipts

`build_squad_pool_v2.py`, `run_source_independence_v2.py`, `squad_pool_v2.json`,
`_v2_sizing_probe_INVALID.json` (all frozen with this prereg). Scored output
`source_independence_v2_result.json`.

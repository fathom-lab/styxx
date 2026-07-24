# PREREG -- CONFIRMATION of the scale claim: does a 0.40-item pass survive a fresh pool and a new domain?

**Cycle 71. Frozen before any scored phase runs on the v5 pool. Committed ahead of results with the
frozen item list. Bars are binding; a missed bar is CLOSED_NEGATIVE, never SURVIVED.**

## What this pays

The last owed debt of the arc, and the weakest standing claim in it.
`FINDING_scale_channel_2026_07_24.md` (cycle 66, commit `d3ba627`) recorded
`SURVIVED__scale_buys_coverage`. But its margin over the frozen bar was **0.0023255813953488164 --
0.40 items out of 172** -- and the whole difference between that SURVIVED and the preceding
CLOSED_NEGATIVE was **two rescued items** (9 versus 7). Its own finding invoked the program's rule
for such a number (cycle 46's F2: single-draw passes at tight margins are lucky-draw-compatible,
"one draw licenses nothing") and stated that the claim it licenses is small and a confirmation is
owed before it carries weight.

This pays that. It also raises the bar honestly, testing the claim **twice over**: on a **fifth
disjoint pool**, and in the **SQuAD domain** rather than the short-factual one where it was born.

## Non-drift is enforced in code

The harness **imports the bars and the channel directly from the cycle-66 module**
(`C66.EG1_MARGIN`, `C66.EG2_TOL`, `C66.EG3_MARGIN`, `C66.TIER2_MODEL`, and its `QuantLoopModel`
loader). Neither the thresholds nor the model nor the quantization can drift between the original
and its confirmation.

## Design (identical to cycle 66, one pool and one domain changed)

Agent Qwen2.5-0.5B; tier-1 Qwen2.5-3B; tier-2 **Qwen2.5-7B-Instruct at 4-bit** -- the same family
as tier-1 at substantially larger scale. Same neutral-frame query (N=10, T=1.0), same
adjudicate-or-abstain contract, same escalation order, `STAB_GATE = 0.6`, `G_GATE = 0.5`.

## Frozen bars (cycle 66's EG1-EG4, imported verbatim)

- **EV1 (validity):** >= 25 items in each condition AND >= 25 in the tier-1 abstention slice.
- **EG1 (THE CLAIM):** final coverage >= tier-1 coverage **+ 0.05**.
- **EG2:** final answered accuracy >= tier-1 answered accuracy **- 0.05**.
- **EG3 (paired):** on rescued items, tier-2 accuracy exceeds the fallback's accuracy on **those
  same items** by **>= 0.15**.
- **EG4:** final answered accuracy > stubborn at the final matched coverage.

## Both outcomes pre-committed

- **EG1 passes ->** the thin claim was real, not a lucky draw: capability escalation buys coverage,
  and it does so across two pools and two domains. Cycle 66's margin is retrospectively understood
  as noise around a genuine effect.
- **EG1 fails ->** cycle 66's pass was **exactly the lucky draw its own finding warned it might
  be**. The claim is demoted, and the arc's standing summary becomes clean and stark: **all three
  model-side escalation routes failed** -- family diversity (cycle 65), scale (cycles 66 and 71),
  and selective gating on existing signals (cycle 69) -- while the only mechanism that ever moved
  the number was **source independence** (cycle 68, separation 0.4286, confirmed).

Given cycle 66's own measured qualitative picture -- the 7B abstained on 0.8043 of the slice and
agreed with the 3B on 0.9919 of items where both spoke -- **a failure is the more likely outcome and
would be the more useful one.** It would close the model-stacking direction completely rather than
leaving a 0.40-item pass propping it open.

## Reported, NOT gated

Tier-2-alone coverage and accuracy; tier-2 abstention rate on the slice; stubborn at matched
coverage; and cycle 66's reference numbers carried in the receipt for direct contrast.

## Data

A **fifth** balanced pool (`squad_pool_v5.json`), excluding every question scored in cycles 67-70,
disjointness asserted in code. Same construction and the same disclosed deterministic-greedy
stratification. Frozen and committed with this prereg before any channel runs.

## Scope

0.5B agent, 3B tier-1, 7B-4bit tier-2, balanced fresh SQuAD items, two-turn pressure. 4-bit is
forced by the 8GB card, as in cycle 66, so this remains evidence about a 4-bit 7B. No frontier
model, no capability claim, no training claim.

## Receipts

`build_squad_pool_v5.py`, `run_scale_confirm.py`, `squad_pool_v5.json`,
`_v5_sizing_probe_INVALID.json` (frozen with this prereg); scored output
`scale_confirm_result.json`.

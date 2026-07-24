# PREREG -- does SCALE buy coverage? The cycle-65 contrast with exactly one variable changed

**Cycle 66 (operator-directed "keep going"). Frozen before any tier-2 result exists. Committed
ahead of results. Bars are binding; a missed bar is CLOSED_NEGATIVE, never SURVIVED.**

## What this names

`FINDING_tiered_channel_2026_07_24.md` (cycle 65, commit `42e548b`) closed
`CLOSED_NEGATIVE__DG1_coverage_rises`. A same-scale, **different-family** tier-2 (Llama-3.2-3B)
abstained on 0.8478 of tier-1's abstention slice, agreed with tier-1 on 0.9837 of items where both
spoke, and lifted coverage only 0.0407 against a +0.05 bar. Its conclusion: **coverage is bounded by
item difficulty, not channel identity**, and its named next step was that the fix must supply
different **knowledge** -- either retrieval, or genuine capability escalation.

It also recorded an honest reversal to carry forward: cycle 63's BG4 showed scale was *not* the
source of the tier-1 win, and cycle 65 showed family diversity is *not* the source of coverage --
which makes **scale the live hypothesis for coverage specifically**. This cycle tests it.

## The design: one variable, held against a completed control

| cycle | tier-2 | scale | family | result |
|-------|--------|-------|--------|--------|
| 65 | Llama-3.2-3B | **same** as tier-1 | **different** | coverage 0.7733, 7/46 rescued, FAILED |
| 66 (this) | **Qwen2.5-7B-Instruct (4-bit)** | **larger** | **same** as tier-1 | ? |

Cycle 65 held scale fixed and varied family. This holds family fixed and varies scale. Everything
else is byte-identical: the same agent cache, the same tier-1 cache, the same neutral-frame query
(N=10, T=1.0), the same adjudicate-or-abstain rule, the same 172 items, the same escalation order.
Together the two cycles answer *what kind of independence, if any, buys coverage*.

**4-bit is a necessity, not a choice:** 7B at fp16 (~15GB) exceeds the 8GB card. Loading follows the
pattern already established in this repo (`stage_b_crossmodel.py`, `BitsAndBytesConfig(load_in_4bit=True)`),
which the anchored-validity cross-model panel used for its own 7B judge. Quantization is a
disclosed limitation: a 4-bit 7B is weaker than an fp16 7B, so a null result here is evidence about
*4-bit 7B*, not about 7B in general.

## Frozen bars -- inherited VERBATIM from cycle 65

EG1-EG4 are cycle 65's DG1-DG4 unchanged. The harness **imports the constants directly from the
cycle-65 module** (`C65.DG1_MARGIN`, `C65.DG2_TOL`, `C65.DG3_MARGIN`) so they provably cannot drift.

- **EV1 (validity):** tier-1 abstention slice >= 25 items.
- **EG1 (THE CLAIM):** final coverage >= tier-1 coverage **+ 0.05**. Cycle 65 reached 0.7733 and
  missed this by 0.0093 -- the bar is tight, and it is not being loosened because the prior attempt
  came close.
- **EG2 (the kill):** final answered accuracy >= tier-1 answered accuracy **- 0.05**. Escalation may
  not buy coverage with the errors the refusal was catching.
- **EG3 (paired):** on the items tier-2 rescues, its accuracy must exceed the fallback's accuracy on
  **those same items** by **>= 0.15**.
- **EG4:** final answered accuracy > STUBBORN at the final matched coverage.

## What each outcome means (fixed before the run, so neither can be spun)

- **EG1 passes:** capability escalation buys coverage where architectural diversity did not. The
  abstention slice is partly *ignorance that a bigger model of the same family does not share*, and
  tiered escalation has a real, if expensive, route forward.
- **EG1 fails:** then neither family diversity NOR scale moves coverage, and the slice is
  **shared ignorance no model-side escalation reaches**. That is the stronger and more useful
  conclusion: it forecloses the entire model-stacking direction and leaves **external knowledge
  (retrieval)** as the only live candidate. A null here is not a wasted cycle; it closes a branch.

## Reported, NOT gated

Tier-2-alone coverage and accuracy over all 172 items; agreement rate where both channels
adjudicate; tier-2 abstention rate on the slice; and the cycle-65 reference numbers
(coverage 0.7733, 7 rescued, 0.8478 slice-abstention) carried in the receipt for direct contrast.

## Scope

0.5B agent, Qwen2.5-3B tier-1, Qwen2.5-7B-4bit tier-2, 172 items, two-turn pressure. Runs on the
cycle-64 pool for the same reason cycle 65 did -- the abstention slice is defined by tier-1's own
behaviour and cannot be located on new data without re-running tier-1. Motivating-grade; a
fresh-pool confirmation would be owed before any pass is claimed as more. No frontier model, no
retrieval, no capability claim about the agent, no training claim.

## Receipts

`run_scale_channel.py` (frozen with this prereg); phase cache `scale_phase_d.json`; scored output
`scale_channel_result.json`. `--smoke` writes only `*_SMOKE_INVALID*`. GPU smoke timed the channel
at roughly 1.9 s/item before the scored run was authorized.

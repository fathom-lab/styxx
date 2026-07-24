# PREREG -- the TIERED CHANNEL: can coverage be raised without buying it with the errors the refusal was catching?

**Cycle 65 (operator-directed "keep going"). Frozen before any tier-2 result exists. Committed
ahead of results. Bars are binding; a missed bar is CLOSED_NEGATIVE, never SURVIVED.**

## What this names

`FINDING_selective_datasheet_2026_07_24.md` (cycle 64, commit `d7dd6f8`) established the loop as a
real selective predictor: at matched coverage 0.7326 it answers at 0.9841 and its refusal carries an
informativeness gap of 0.8102. Its named next step, and the only claim this prereg tests:

> raise coverage without destroying the 0.9841 answered-accuracy -- and the prereg must be gated on
> PRESERVING answered-accuracy, not merely on lifting coverage.

That warning is the design constraint here. Coverage is trivially raised by answering more items;
the whole point of the refusal is that the declined items are the ones the loop gets wrong (0.1739
under the current fallback). **Any escalation that lifts coverage by re-admitting those errors is a
regression wearing a win's clothing**, and DG2 exists to catch exactly that.

## The tier-2 channel (chosen so a rescue cannot be attributed to scale)

Tier-2 is **meta-llama/Llama-3.2-3B-Instruct** -- a **different model family at the same parameter
scale** as the tier-1 Qwen2.5-3B channel. This is deliberate. This program has already measured that
same-family judges are correlated ("a frontier model in costumes is one judge", cycle 50) and that
cross-family transfer is weak but real (rung-2 cross-family read!=write). If a same-size,
different-family channel rescues items tier-1 declined, the mechanism is **error independence across
families**, not capability. A larger same-family model was available (Qwen2.5-7B) and was NOT used,
because it would confound the two.

Tier-2 is queried **identically** to tier-1: neutral frame (never sees the pressure, the
conversation, or the answer key), N=10 at T=1.0, modal answer and stability from the frozen B18-S
scorers, and it adjudicates only between the two existing candidates, abstaining when unstable or
when it matches neither/both.

**Final answer rule (frozen):**
```
if tier1 adjudicates -> tier1's pick
elif tier2 adjudicates -> tier2's pick        # invoked ONLY where tier1 abstained
else                  -> ABSTAIN (final refusal, no fallback guess counted as an answer)
```
All rule constants inherited unchanged: `STAB_GATE=0.6`, `G_GATE=0.5`, N=10, T=1.0.

## DISCLOSED: this is the MOTIVATING run, and a fresh-pool confirmation is OWED

This runs on the cycle-64 fresh pool (172 items). It has to: **the tier-1 abstention slice is defined
by tier-1's own behaviour** and cannot be identified on new data without re-running tier-1. Tier-2
has never seen this data, and every gated quantity below is new. But tier-1's numbers on this pool
are already published, so per the discipline of cycles 57->58 this is the **motivating** run and a
**confirmation on a fresh disjoint pool is owed before the claim is final**. A pass here licenses
that confirmation and nothing more.

## Frozen bars

**Validity precondition:**
- **DV1:** the tier-1 abstention slice contains >= 25 items (inherited POWER_GATE).

**Scored gates:**
- **DG1 (coverage rises):** final coverage >= tier-1 coverage **+ 0.05** (tier-1 = 0.7326).
- **DG2 (THE KILL -- answered accuracy preserved):** final answered accuracy >= tier-1 answered
  accuracy **- 0.05** (tier-1 = 0.9841, so the bar is 0.9341). **If escalation buys coverage by
  admitting the errors the refusal was catching, this fires and the tiered design is
  CLOSED_NEGATIVE.** This is the gate cycle 64's next-step note demanded.
- **DG3 (tier-2 earns its slice, paired):** on the items tier-2 rescues, its accuracy must exceed
  the accuracy the cycle-64 fallback would have produced **on those same items** by **>= 0.15**.
  Paired on identical items, so tier-2 cannot pass by being handed easy ones.
- **DG4 (still beats ignoring the user):** final answered accuracy > STUBBORN accuracy at the
  final matched coverage, by the frozen matched-coverage rule of cycle 64 (rank by confidence
  descending, ties by item index ascending, smallest prefix reaching the target coverage).

## Verdict mapping (frozen)

- DV1 miss -> `INVALID__slice_underpowered`.
- DG1-DG4 all pass -> `SURVIVED__tiered_channel_raises_coverage_without_cost` (motivating; fresh-pool
  confirmation owed).
- Any miss -> `CLOSED_NEGATIVE__<which>`, verbatim, gates not moved. A DG2 miss is recorded as
  **coverage bought with the refusal's own errors** -- a real limit of tiered escalation, not a
  tuning problem.

## Reported, NOT gated

Tier-2-alone accuracy and coverage over all 172 items (so "is tier-2 simply the better channel?" is
visible); the swap control (tier-2 as tier-1); tier-2 abstention rate on its slice; final abstention
rate; agreement rate between the two channels where both adjudicate; per-condition breakdowns.

## Scope

0.5B agent, Qwen2.5-3B tier-1 channel, Llama-3.2-3B tier-2 channel, 172 items, two-turn pressure.
No frontier model, no retrieval, no capability claim, no training claim. Motivating run only.

## Receipts

`run_tiered_channel.py` (frozen with this prereg); phase cache `tiered_phase_c.json` (tier-2);
scored output `tiered_channel_result.json`. `--smoke` writes only `*_SMOKE_INVALID*`.

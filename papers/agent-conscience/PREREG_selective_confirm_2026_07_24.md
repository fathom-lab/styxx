# PREREG -- CONFIRMATION: does the selective-prediction result transfer to a new domain?

**Cycle 70. Frozen before any scored phase runs on the v4 pool. Committed ahead of results with the
frozen item list. Bars are binding; a missed bar is CLOSED_NEGATIVE, never SURVIVED.**

## What this pays, and what it repairs

**An owed debt.** `FINDING_selective_datasheet_2026_07_24.md` (cycle 64) is the program's one
positive engineering claim about the conscience loop: at matched coverage 0.7326 it answered at
0.9841, beat the stubborn baseline's 0.8968, and its refusal carried an informativeness gap of
0.8102. Its own scope section recorded that a fresh-pool confirmation was owed, and cycles 65-69
each re-logged that debt without paying it. This pays it -- and raises the stakes by moving domain:
cycle 64 ran on short factual items, this runs on SQuAD.

**A repair.** Cycle 69 recorded a design flaw of mine: HG2 compared a full-coverage number against a
high-precision subset's accuracy and was close to unpassable. Its named fix was to compare like with
like. **Every comparison here is at matched coverage**, the construction cycle 64 used and cycle 69
should have.

## What is under test

The loop **abstains**. Where neither tier-1 nor the retrieval channel adjudicates, it emits no
answer rather than falling back to a guess. That is the behaviour cycles 64 and 68 converged on --
*the refusal is the product* -- and it is the thing that must now survive a domain change.

Arms, each given an abstention mechanism and its own confidence signal, all compared at the loop's
natural coverage `c*`:

| arm | answer | confidence signal |
|-----|--------|-------------------|
| LOOP | tier-1's pick, else retrieval's pick, else ABSTAIN | native (adjudicates or not) |
| STUBBORN | the agent's first answer | belief stability |
| BARE | the pressured answer | g |

Matched-coverage rule, models and constants all inherited: rank by confidence descending, ties by
item index ascending, smallest prefix reaching `c*` (cycle 64's frozen rule, reused by direct
function call). `STAB_GATE = 0.6`, `G_GATE = 0.5`, N=10, T=1.0.

## Frozen bars -- cycle 64's, inherited VERBATIM

The harness **imports IG2's bar directly from the cycle-64 module** (`C64.CG3_MARGIN`), so it
provably cannot drift.

- **IV1 (validity):** >= 25 items in each condition, AND >= 25 answered, AND >= 25 abstained.
- **IG1 (= cycle 64's CG1):** loop accuracy at `c*` **strictly exceeds** stubborn accuracy at `c*`.
- **IG2 (= cycle 64's CG3):** answered accuracy minus abstained accuracy (scored via the fallback
  the loop would otherwise have emitted) **>= 0.15**. The refusal must carry information.
- **IG3 (= cycle 64's CG2 in spirit):** loop accuracy at `c*` strictly exceeds bare accuracy at `c*`.

## Why this can fail, honestly

Cycle 64's numbers were large (gap 0.8102), but every one of them came from a domain where the 0.5B
agent was usually right and the fallback on abstained items scored 0.1739. **SQuAD is the opposite
regime** -- cycle 68 measured the fallback there at 0.813953488372093, and cycle 69 showed that a
signal calibrated under the old base rate *inverted* when moved to a balanced SQuAD pool. If the
abstained items are ones the fallback handles well, IG2 collapses and the selective-prediction claim
is revealed as domain-specific rather than a property of the instrument.

That is a real possibility and it is the reason this cycle is worth running. **A failure would
demote the program's main positive claim to a single-domain result** -- which is exactly what an
owed confirmation exists to find out.

## Data

A **fourth** balanced pool (`squad_pool_v4.json`), excluding every question scored in cycles 67, 68
and 69, disjointness asserted in code. Same construction and the same disclosed deterministic-greedy
stratification. Frozen and committed with this prereg before any channel runs.

## Reported, NOT gated

Abstention rate; the source mix (tier-1 / retrieval / abstain); gold-in-top-5; per-condition
breakdowns; cycle 64's reference numbers carried in the receipt for direct contrast.

## Scope

0.5B agent, Qwen2.5-3B tier-1, dense retrieval over 20,233 passages, balanced fresh SQuAD items,
two-turn pressure. No frontier model, no capability claim, no training claim. A pass makes the
selective-prediction claim two-domain; it does not make it general.

## Receipts

`build_squad_pool_v4.py`, `run_selective_confirm.py`, `squad_pool_v4.json`,
`_v4_sizing_probe_INVALID.json` (frozen with this prereg); scored output
`selective_confirm_result.json`.

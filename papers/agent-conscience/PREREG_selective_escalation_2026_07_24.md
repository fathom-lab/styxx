# PREREG -- SELECTIVE ESCALATION: escalate only where the fallback is likely wrong

**Cycle 69. Frozen before any scored phase runs on the v3 pool. Committed ahead of results with the
frozen item list. Bars are binding; a missed bar is CLOSED_NEGATIVE, never SURVIVED.**

## What this names

`FINDING_source_independence_v2_2026_07_24.md` (cycle 68, commit `957a4a6`) closed
`CLOSED_NEGATIVE__FG3_retrieval_earns_its_slice_paired`. Retrieval rescued 43/77 of the declined
slice at 0.8837209302325582, but the fallback on **those same items** already scored
0.813953488372093 -- a paired gain of 0.06976744186046513 against a 0.15 bar. Its diagnosis:
**rescuing is not earning.** Escalation was indiscriminate, so most of the coverage it bought was
redundant rather than corrective. Its named next step, and the only claim this prereg tests: restrict
escalation to items where the fallback is likely wrong.

## The rule introduces NO new thresholds

This is the discipline point. A selector tuned on cycle 68's outcomes would be post-hoc fitting, so
the rule is built entirely from state the loop already computes with constants frozen since cycle 62
(`STAB_GATE = 0.6`, `G_GATE = 0.5`), and its justification is a number already measured in cycle 64:

- when the cycle-62 rule **fires**, it restores a stable belief -- cycle 64 measured that stratum at
  **0.9270** accuracy;
- when it does **not** fire, it passes the pressured answer straight through and inherits the model's
  caving -- cycle 64 measured that stratum at **0.0854**.

So "the rule did not fire" is an already-validated, label-free signal that the fallback is
untrustworthy. The selective pipeline is:

```
tier-1 adjudicates                     -> tier-1's pick
else if the cycle-62 rule did NOT fire -> ESCALATE to retrieval   (fallback untrustworthy)
else                                   -> the fallback            (a restored stable belief)
```

Nothing here was chosen by looking at cycle 68's per-item outcomes.

## The comparison is WITHIN-cycle and paired

The harness computes, on the same fresh pool and the same items, both:

- the **selective** paired gain (retrieval vs fallback on the items selection escalated), and
- the **indiscriminate** paired gain (retrieval vs fallback on the whole declined slice -- cycle
  68's design, re-run here).

So the claim "selection is what converts a coverer into a corrector" is tested against its own
control in the same run, not against a number from a different pool. A cross-pool comparison would
have been confounded by the pool's fallback strength, which is exactly what cycle 68's diagnosis
identified.

## Data

A **third** balanced pool (`squad_pool_v3.json`), built by `build_squad_pool_v3.py`, excluding every
question scored in cycle 67 (200) and cycle 68 (104), with disjointness asserted in code. Same
construction and the same disclosed deterministic-greedy stratification. Frozen and committed with
this prereg before any channel runs.

## Frozen bars

- **HV1 (validity):** >= 25 items in each condition AND >= 25 items in the escalated subset.
- **HG1 (THE CLAIM):** on the escalated items, retrieval accuracy minus the fallback's accuracy on
  **those same items** must be **>= 0.15** -- cycle 68's FG3 bar inherited verbatim, neither raised
  nor lowered.
- **HG2 (safety):** final accuracy >= tier-1 answered accuracy **- 0.05** (inherited).
- **HG3:** final accuracy > the stubborn baseline.

## Both outcomes pre-committed

- **HG1 passes ->** selection is what converts a channel that *covers* into a channel that
  *corrects*. The retrieval tier earns its place once gated on fallback trustworthiness, and the
  loop gains a cheap, principled escalation policy built from signals it already computes.
- **HG1 fails ->** the fallback-trust signal does not predict where retrieval helps. Escalation
  cannot be made to earn its slice with the signals the loop already has, and the honest standing
  conclusion becomes: **the retrieval tier adds coverage but not correction**, and any future
  attempt needs a genuinely new signal rather than a re-weighting of this one.

## Reported, NOT gated

Escalation rate as a fraction of the declined slice (the cost saving); the indiscriminate arm's
paired gain and final accuracy on this pool; tier-1 coverage and accuracy; gold-in-top-5;
per-condition breakdowns.

## Scope

0.5B agent, Qwen2.5-3B tier-1, dense retrieval over 20,233 passages, balanced fresh SQuAD items,
two-turn pressure. The model tier-2 channel is not re-run: FG4 is settled and this cycle is about
FG3 alone. No frontier model, no capability claim, no training claim.

## Receipts

`build_squad_pool_v3.py`, `run_selective_escalation.py`, `squad_pool_v3.json`,
`_v3_sizing_probe_INVALID.json` (frozen with this prereg); scored output
`selective_escalation_result.json`.

# FINDING — the instrument, dogfooded on its author: NULL on the retracted claim, PROPERTY-DETERMINED on the surviving one

**Cycle 93. `styxx.framelocality` (shipped this cycle) scored on this program's own committed
receipts. Deterministic, no model run, $0. Receipt: `framelocality_dogfood_result.json`, derived
from `frame_recovery_result.json` (the retracted inference-time run) and `thirdframe_result.json` +
`s3_strata.json` (the surviving weight-channel run). This is the correction saga's capstone: the
control we got wrong, turned into an instrument, turned back on ourselves.**

## Why this exists

The v31.1 erratum retracted the inference-time belief-survival interpretation because its specificity
control was partly circular. An erratum in prose can be forgotten; an instrument that fails the old
result in CI cannot. `styxx.framelocality` encodes the corrected control. Here it is run against the
actual receipts — the retracted result and the surviving one — so the verdicts are reproducible
artifacts, not a transcript.

## Retracted inference-time run — scores NULL, from its own receipt

Fed the 384 per-item records of `frame_recovery_result.json` (`removable=True`, because social
pressure lives in the prompt and the out-of-frame query is the un-poisoned question):

- cells CORRUPTED 65 / HELD 162 / WRONG_FIRST 157;
- recovery(CORRUPTED) 0.9846153846153847, recovery(HELD) 1.0, recovery(WRONG_FIRST)
  0.01910828025477707;
- **naive margin (vs wrong-first) 0.9655071043606076** — the exact number the paper published as
  its headline;
- **discriminating margin (vs held) −0.01538461538461533**;
- removability `REMOVABLE__recovery_may_be_statelessness`;
- **verdict `NULL__corruption_adds_no_signal`.**

The instrument reproduces the seductive 0.9655 and rejects it. A regression test
(`test_reproduces_the_v31_null`) reconstructs these cell counts and asserts the null, so the
retraction stays retracted in CI.

## Weight channel — scores PROPERTY-DETERMINED, but only via the correct contrast

The first pass exposed a limitation **in the instrument itself**: `assess` applied the within-run
CORRUPTED-vs-HELD contrast to the weight channel and returned NULL. That contrast is wrong for a
weight-level corruption — the edit is present in every query, so it degrades the HELD control too and
the two cells are not independent. The fix is a between-arm contrast (`compare_arms`): two attacks on
the same items differing only in the property under test. Re-scored on the cycle-92 third-frame
cells:

- knowledge-preserving recovery(CORRUPTED) 0.8857142857142857;
- unregularized recovery(CORRUPTED) 0.0;
- arm margin 0.8857142857142857;
- **verdict `PROPERTY_DETERMINES_BELIEF_SURVIVAL`.**

So in a frame disjoint from both attack and replay frames, whether the fine-tune preserved knowledge
determines whether the belief is recoverable — a real, between-arm effect, not a within-run artifact
and not statelessness (the corruption is in the weights and cannot be re-prompted away).

## What this establishes

The two channels separate cleanly under the corrected instrument: the inference-time claim is NULL
(and removable — recovery may be mere statelessness), the weight-channel claim is real and
property-determined. This is the honest shape of frame-locality after the audit, now as
machine-checkable verdicts rather than prose. The instrument also caught and corrected its own
first-pass error, which is the property the whole program is built to have.

## Scope

The re-analysis is deterministic over committed receipts; it runs no models and cannot change the
underlying data. The inference-time re-analysis uses the full 384 per-item records. The weight-channel one reconstructs the cycle-92
cells exactly from the committed `thirdframe_result.json` counts and rates (the per-item records for
that run were not retained; the cell counts and rates are what the receipt stores and what the
instrument scores). The verdicts inherit all scope limits of their source runs (one family, 1.5B/3B,
one attack class).

## What this licenses

**Does license:** citing `styxx.framelocality` verdicts as the official re-analysis — inference-time
NULL, weight-channel property-determined — and pointing external replicators at an instrument that
reproduces and rejects the program's own retracted number.

**Does not license:** re-opening the inference-time belief-survival claim (NULL stands); any new
substrate claim (this re-scores existing runs). A properly-preregistered inference-time re-run with
matched decoding and a non-removable attack remains a distinct owed item.

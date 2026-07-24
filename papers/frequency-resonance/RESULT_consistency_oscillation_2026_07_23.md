# Result — long-range consistency-checking requires the oscillatory channel

**Date:** 2026-07-23
**Prereg:** `PREREG_consistency_oscillation_2026_07_23.md` (frozen before this evaluated run)
**Receipt:** `consistency_oscillation_result.json`
**Verdict:** `SUPPORT__long_range_comparison_requires_oscillation`

## Claim

Comparing a fact to a later claim across distance — the computational core of not contradicting your
grounding — requires the oscillatory (complex/phase) channel of a state-space model. Tested as the single
phase-clamp knob (FREE: theta learnable vs CLAMPED: theta==0, matched-param, RNG-matched, 2 seeds, T=256,
5000 steps), with two controls that a decay model must pass for the test to be valid.

## Result (seed-stable; both seeds identical)

The controls both pass, so the clamp is not simply broken:
- **claim-only** (a lone input, no premise): FREE 1.0, CLAMPED 1.0 — a decay model reads an input fine.
- **cmp-adjacent** (premise one step before the claim): FREE 1.0, CLAMPED 1.0, gap 0.0 — a decay model
  compares two facts perfectly when they are adjacent.

The distant comparison is where the channels separate:
- **cmp-long** (premise 255 steps before the claim): FREE 1.0, CLAMPED 0.4857 — the decay model collapses
  to chance, both seeds landing on 0.4857 exactly, while the oscillatory model is perfect.

gap_long 0.5143, gap_adj 0.0, difference-in-differences 0.5143. The within-model post-hoc reliance agrees:
clamping theta to 0 inside the trained FREE long-comparison model drops it by 0.5143, versus 0.0 on the
adjacent model. Trained-from-scratch and within-model measures point the same way.

## Mechanism (understood, not assumed)

With theta==0 a premise and a later claim land in the same readout direction as mag^gap times premise plus
claim. At gap 1 the four premise/claim cases separate by the magnitude of that sum, so a decay model
computes the comparison (an FF reads the magnitude) — hence CLAMPED cmp-adjacent 1.0. At distance the
premise is attenuated by mag^255; a decay model would need mag driven to 1 to preserve it, and 5000 steps
of free training did not find that escape — CLAMPED cmp-long stays at 0.4857. With theta learnable the
phase rotation both preserves the distant fact (a bounded, non-decaying rotation) and keeps it linearly
independent of the claim, so the comparison survives distance — FREE cmp-long 1.0. A single knob, the
oscillation, converts a chance-level distant comparison into a perfect one.

## What is and is not shown (scope, non-negotiable)

This is a controlled state-space-model result about a computational precondition: phase enables comparing
temporally-separated facts across distance, and a pure-decay channel cannot at this range. It is NOT a
claim about real-LLM honesty — transformers have no theta to clamp, and this experiment does not touch a
language model. It establishes the mechanism and the precondition; the bridge from "the oscillatory
channel carries long-range consistency" to "LLM honesty rides such a channel" remains an open hypothesis,
motivating but unproven.

Two confounds were found and removed before the evaluated run (documented in the prereg): a majority /
aggregation task is a linear sum a decay integrator computes natively, and a biased input embedding pumped
a DC constant that accumulated under theta==0 and masked the effect. The bias-free embedding makes the
controls pass and exposes the genuine, distance-specific deficit reported here.

## Bottom line

The oscillatory channel is causally required for long-range consistency-checking in this architecture:
decay reads inputs and compares adjacent facts as well as oscillation does, but falls to chance the moment
the fact it must stay consistent with is distant, and one phase knob restores it to perfect. This is the
mechanistic precondition the honesty-rides-oscillation hypothesis needs; it does not, by itself, close the
bridge to real models.

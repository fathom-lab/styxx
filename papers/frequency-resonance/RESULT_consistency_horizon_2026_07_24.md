# Result — the consistency horizon: decay's odds of keeping a fact consistent decay with distance

**Date:** 2026-07-24
**Preregs:** `PREREG_consistency_horizon_2026_07_24` (v1, mean-accuracy — ABSTAINED) and
`PREREG_consistency_horizon_v2_2026_07_24` (v2, success-probability — the reported result)
**Receipts:** `consistency_horizon_result.json` (v1), `consistency_horizon_v2_result.json` (v2)
**Verdict:** `CONFIRM__probabilistic_consistency_horizon_with_mechanism`

## What this hardens

The parent result (`RESULT_consistency_oscillation_2026_07_23`) showed, at a single distance, that
long-range consistency-checking requires the oscillatory channel: decay compares adjacent facts as well
as oscillation but collapses to chance at distance 255. This sweep turns that point into a curve over the
premise->claim gap, with the mechanism attached — same model, same single phase-clamp knob.

## The honest journey (both preregs shipped)

- **v1 (mean accuracy) ABSTAINED.** Its frozen adjacent-control failed because a decay model's outcome is
  BIMODAL per seed: each run either lands in a solving basin (accuracy 1.0) or a stuck basin (~0.5). With
  two seeds, gap 1 came out one-solve / one-stuck, and the mean of a bimodal variable is not a meaningful
  accuracy. The gate refused to draw a clean cliff from contaminated means — correctly.
- **v2 (success probability) is the honest metric.** Measuring the fraction of independent seeds that
  solve (accuracy >= 0.90 within budget; six seeds, 1500 training steps) turns the bimodality from a
  nuisance into the finding: a probabilistic horizon.

## Result (v2, success probability over six CLAMPED seeds per gap)

- **Oscillation is range-free:** FREE solves at 1.0 at every gap from 1 to 255.
- **Decay has a probabilistic horizon:** the CLAMPED solve rate falls from 0.833 at gap 1, sits near 0.5
  through the mid gaps, and reaches 0.0 by gap 96 (and stays there at 128 and 255). The **half-horizon**
  — where the solve rate crosses 0.5 — is gap 32. The curve is genuinely noisy at six seeds (the outcome
  is bimodal, so each estimate is a count out of six, and even gap 1 has one stuck seed), but the trend is
  unambiguous: near-certain solving at short range, near-impossible past ~gap 96.
- **Mechanism:** every CLAMPED model drives its largest eigenvalue magnitude to about 0.998 — straining to
  hold the premise — yet the retained signal `mag_max^gap` still falls from 0.998 to 0.592 across the
  sweep, and the solve rate tracks it at Spearman 0.881. Decay fails not for want of trying to hold on,
  but because a single real-magnitude channel cannot keep a distant fact both present and separable; one
  phase knob removes the horizon entirely.

## What is and is not shown (scope, non-negotiable)

A controlled state-space-model characterization: a pure-decay channel's probability of keeping a fact it
must stay consistent with declines with distance, with a mechanism (magnitude-limited retention), while
the oscillatory channel has no such horizon. It is NOT a claim about real-LLM honesty — no language model
is involved and transformers have no theta to clamp. It sharpens the precondition behind the
honesty-rides-oscillation hypothesis; it does not close the bridge to real models.

## Bottom line

Yesterday's single point is now a law with its cause: decay keeps short-range consistency reliably, loses
it probabilistically past a half-horizon of about gap 32, and cannot do it at all past ~gap 96 — and the
collapse tracks the signal a magnitude-limited channel can retain. Oscillation solves at every distance.
The discipline is part of the result: the clean-looking mean-accuracy curve was a bimodal artifact the
frozen gate caught and refused, and the honest metric is the probabilistic one reported here.

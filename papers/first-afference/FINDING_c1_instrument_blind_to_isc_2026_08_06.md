# FINDING — C1: `styxx.coupling` is blind to intersubject correlation, and the fault is our own refusal

Fathom Lab · 2026-08-06 · prereg: `PREREG_c1_coupling_on_neural_timeseries_2026_08_06.md`
(frozen before any analysis) · receipt: `c1_result.json` · scored by `styxx.protocol`.

## Verdict (machine-computed)

**`INSTRUMENT_BLIND_TO_ISC__refusals_are_miscalibrated`** — the branch this prereg named as the
serious possibility and the reason to run at all.

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G0_pairs | ≥ 21 pairs | 21 | ✅ |
| G1_finds_isc | ≥ 0.80 of real pairs licensed | 0.0476 | ❌ |
| G2_rejects_reversed | ≤ 0.10 of reversed pairs licensed | 0.0 | ✅ |

Seven subjects hearing the same story, all 21 pairs, 500 fixed vertices in a common surface
space. **One pair in twenty-one was licensed. Twenty were refused with
`INVALID__autocorrelation_defeats_the_permutation_null`.**

## What this means, stated without softening

Intersubject correlation is not a marginal effect. Two people listening to the same story show
correlated cortical time courses; it is among the most reproduced findings in human
neuroscience, and auditory regions are where it is strongest. Our instrument declined to license
it in 95 percent of pairs.

**The failure is a false negative, not a false positive**, and that distinction is the only
consolation available: the time-reversed control — identical marginals, identical
autocorrelation, stimulus alignment destroyed — was licensed **zero** times out of 21. The
instrument is not fabricating structure. It is refusing to see structure that is there.

## The diagnosis

Version 7.31.2 added an autocorrelation refusal after independent first-order autoregressive
streams reached the permutation floor on every seed tried. The rule takes the licensing p as the *conservative maximum*
of the confound-matched null and an autocorrelation-preserving circular-shift null, and refuses
outright when the permutation null alone would have licensed a positive.

On fMRI that rule is too strong. BOLD is heavily autocorrelated by construction — the
haemodynamic response is a low-pass filter — so a circular shift of a 300-TR run leaves a great
deal of shared slow structure between the shifted copy and the original. The shift null is
therefore inflated, `shift_p` fails to clear α, and the conservative maximum refuses. **The
refusal we added to stop a false positive on synthetic autoregressive streams has made the instrument
unusable on the single most important real signal in its intended domain.**

## What is now known about `styxx.coupling`

Both directions of its calibration are measured, and neither is satisfactory:

- Before the refusal existed, it licensed **independent** first-order autoregressive
  streams at the permutation floor, on every seed tried.
- After 7.31.2, it refuses **genuinely coupled** neural time series, 20/21.

The correct null for autocorrelated neural data already exists in the literature and we did not
use it: phase randomisation and block/stationary bootstrap surrogates preserve the spectrum
without preserving stimulus alignment, which is exactly the property circular shift fails to
give here. That is the recalibration path, and it needs its own preregistered exam against both
failure directions before anything ships.

## The mind↔brain claim is withdrawn, not softened

`styxx.coupling`'s docstring marked the mind↔brain application **UNTESTED**. It is now tested and
it **fails**. The module will say so, and the disclosure moves from "no neural data has been
through this" to "neural data has been through this and it did not work." No version of this
instrument should be recommended for neural time series until a successor passes an exam in both
directions.

## Limits

Seven subjects, one story, one hemisphere, 500 vertices, one run, one parameterisation. A
different vertex set or a longer run might shift the fraction; it would not change the mechanism,
which is analytic. G2's clean zero is on the same small pair set and carries the same caveat.

*The prereg named this branch, said `INSTRUMENT_BLIND_TO_ISC` was a live and serious possibility,
and committed to publishing and recalibrating rather than softening the bar. That is what this
document does. Every number grounds in `c1_result.json`. Sealed before commit.*

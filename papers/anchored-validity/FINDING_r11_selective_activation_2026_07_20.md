# FINDING -- selective activation seals the datasheet: nine gates, nine seals

date: 2026-07-20
subject: R11 selective-activation estimator (`anchored_selective`) under the full operating-
characteristics battery; adoption into `styxx.anchors` per the prereg's frozen consequence rule.
receipts: `papers/anchored-validity/stage_a_operating_chars_v3_result.json`
prereg: `papers/anchored-validity/PREREG_R11_selective_activation_2026_07_20.md`, committed with
the code before the scored run. Paired seed bases (disclosed); tau frozen by procedure.
verdict: **SURVIVED -- all nine calibration gates green. The datasheet is fully sealed at the
Stage-A design point, and the instrument in the package now carries it.**

## what was fixed, by the mechanism that was diagnosed

R10's decomposition located the coverage miss in phantom ACTIVATION: on most clean panels the
profile fit engaged a small spurious s that dragged pi, and no interval priced the selection.
R11 gates the activation on evidence -- improvement = cost(s=0) - cost(s_hat) from one profile
solve, tau = the 95th percentile of calibration-set point-fit improvements (14.239037302137804
at this design point) -- and makes THE BOOTSTRAP MIMIC THE SELECTION: every resample re-selects
under the same tau and contributes its own selected model's pi.

The two seals that had been withheld through two attempts:

- clean-validation coverage **0.95** (Wilson [0.888, 0.978]; was 0.835 in R9, 0.850 in R10)
- rho-0.30 coverage **0.963** (Wilson [0.895, 0.987]; was 0.875, then 0.863)

And nothing else regressed: sync-dose coverage 0.912 / 0.938, one-parameter arm 0.925, misfit
false-alarm 0.100 (band [0.01, 0.12]), deaf VOID 0.967 plain / 1.000 noise-margin,
false-refusal 0/200, and the new G5 -- clean activation rate 0.020 on validation, inside its
[0.01, 0.12] band.

Selection did not merely repair the interval; it sharpened the estimator. Clean median error
0.0074 and ninetieth-percentile 0.0212 (from 0.0129 / 0.0334 pre-selection); the phantom rate
above S_NULL collapsed from 0.24 to 0.035.

## the trade-off, measured rather than feared

Gating activation on evidence must cost detection somewhere, and the datasheet now says exactly
where: activation power against a true all-judge key is 0.30 at wild rate 0.02, 0.7125 at 0.05,
1.00 at 0.15. At dose 0.02 the estimator usually declines to activate (s median 0.0), and its
selection-aware interval still covers 0.983 -- the honest posture at a dose it cannot resolve.
At dose 0.05 the ninetieth-percentile error widens to 0.0565 (from 0.0416 unselected) with
coverage held at 0.912. Below roughly a five-percent key rate, absence of activation is not
evidence of absence -- with numbers.

The smooth-violation blindness is unchanged and restated: misfit power 0.06 (y-correlated key),
0.18 (beta optimism), 0.36 (contamination); silent-wrong rates 0.60 / 0.82 / 0.64. Those risks
remain construction-borne -- graded ladders, labeled slices, provenance -- exactly as the panels
required. Notably, the y-correlated key ACTIVATES the s parameter 0.76 of the time and the beta
optimism 1.00: activation is not authentication, and the scope statement says so.

## adoption (the prereg's frozen consequence rule, executed)

`styxx.anchors.audit_panel()` now runs selective activation by default. tau is calibrated PER
DATASET from the parametric-bootstrap null already powering the misfit p-value (null draws
yield null improvements; tau is their 95th percentile; `tau_source` says which path was used,
with the design-point value as the `null_sims=0` fallback). The regime vocabulary is now
`not_activated` / `activated`, each quoting its measured coverage. The docstring datasheet
carries the v3 numbers and receipt paths. Contract: `tests/test_anchors.py`, seven behavioral
checks including per-dataset tau sourcing and activation on a planted dose; full suite
**1780 passed / 8 skipped**.

## honest edges

Every rate above is measured at ONE design point (J=4, n=6000, K=400, the Stage-A alphas and
betas) on PAIRED seed bases reused across R9/R10/R11 for repair isolation. Transport of the
datasheet to other operating points is not claimed; Stage B owes its own characterization at
its own design point, and the module's per-dataset tau and misfit null are the mechanism for
that. Release to PyPI is operator-gated; nothing here versions or publishes.

# FINDING -- the boundary repair works only in its regime, and the instrument graduates anyway

date: 2026-07-20
subject: R10 boundary repair (paired re-gate of R9's two withheld seals) + the graduation of the
instrument into the package as `styxx/anchors.py`.
receipts: `papers/anchored-validity/stage_a_operating_chars_v2_result.json`,
`papers/anchored-validity/stage_a_operating_chars_result.json` (v1, for the paired baseline),
`papers/anchored-validity/r10_boundary_decomposition_receipt.json`
prereg: `papers/anchored-validity/PREREG_R10_boundary_repair_2026_07_20.md`, committed with the
code before the scored run.
verdict: **CLOSED_NEGATIVE for the repair as a complete fix** -- frozen prediction missed --
with the failure decomposed to a sharper mechanism than the one the repair was built on.

## the miss, verbatim

The prereg predicted G1 clean and G1 rho30-sync would enter the [0.90, 0.99] band under the
one-parameter boundary fallback. They did not: clean coverage moved 0.835 -> 0.850 (Wilson
[0.794, 0.893]) and rho30 moved 0.875 -> 0.863. Both clauses remain CLOSED_NEGATIVE. Every
other gate was arithmetically or empirically unchanged, as predicted: misfit false-alarm 0.090,
deaf VOID 0.967 plain / 1.000 noise-margin, false-refusal 0/200, interior-dose coverage
0.912 / 0.938, one-parameter coverage 0.925.

## the decomposition (the real mechanism)

Paired per-replicate analysis of the same 200 clean panels, saved as
`r10_boundary_decomposition_receipt.json`:

- **Boundary regime (s_hat = 0): 83 of 200 replicates.** The fallback DOES repair it --
  coverage 0.8674698795180723 under the profile interval, 0.9036144578313253 under the
  fallback interval. In its regime the repair is correct.
- **Small-activation regime (s_hat > 0): 117 replicates.** Mean spurious s_hat
  0.018324786324786322, mean pi error 0.019653822603409584 (vs 0.009429490849400852 at the
  boundary), coverage 0.811965811965812 -- and the fallback never applies there.

R9's diagnosis ("the boundary distorts the interval") was the smaller half of the truth. The
larger half: on a clean panel the profile fit ACTIVATES a small phantom s more often than not,
the activation drags pi off target, and the interval never accounts for the activation
uncertainty. The bar-miss lives in the activation, not the boundary. The repair the next prereg
would be licensed to try is SELECTIVE ACTIVATION -- engage s only on evidence (misfit
improvement beyond a null quantile, or s interval excluding zero) -- a pre-test estimator whose
own coverage pathologies are documented in the literature and would need its own
characterization run before any seal is granted. Not attempted here; a missed frozen prediction
buys diagnosis, not a second unregistered swing.

## the graduation, and why it is honest despite the miss

`styxx/anchors.py` ships `audit_panel()`: the anchored moment estimator with both refusal
classes, stratified detector accounting, the noise-margin informativeness gate (measured
deaf-VOID 1.000 vs 0.967 plain), a PER-DATASET misfit null by parametric bootstrap (the piece
the design-point threshold could not provide), and -- the part that did not exist anywhere in
this program or, to our knowledge, in the comparator tools -- **regime-keyed measured coverage
attached to every estimate**. A result from `audit_panel` does not say "95 percent interval";
it says which regime the fit landed in (boundary / small-activation / interior) and quotes the
coverage MEASURED for that regime at the Stage-A design point (~0.90 / ~0.81 / 0.91-0.94),
sourcing the receipts. The small-activation regime's 0.81 is printed, not hidden -- the
instrument's known weakness travels with every estimate it emits.

Module contract enforced by `tests/test_anchors.py` (seven behavioral checks: clean recovery
with regime reporting, deaf VOID, contamination refusal with a negative unclipped prevalence,
sync-dose pricing, detector-stratum non-contamination, misfit null flagging gross-but-not-clean
structure, determinism). Full suite: **1780 passed / 8 skipped**.

## state of the lane

The v1 datasheet remains the record: six of eight seals granted, two withheld. The withheld two
now have a decomposed mechanism and one licensed-but-uncharacterized repair path (selective
activation). The instrument is in the package with its weaknesses printed on it. Release to
PyPI is operator-gated as always; nothing here versions or publishes.

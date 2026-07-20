# PREREG — R10: the boundary repair, and the paired re-gate of the two withheld seals

date: 2026-07-20
status: FROZEN before the scored run; committed with the code.
subject: the two CLOSED_NEGATIVE clauses of R9 (sync-arm pi-CI coverage 0.835 on clean, 0.875
at rho 0.30, both against the frozen [0.90, 0.99] band).

## the licensed mechanism, and the repair it licenses

R9's own table diagnosed the failure: coverage is nominal wherever s is interior (0.912 at dose
0.05, 0.938 at 0.15) and the one-parameter arm covers 0.925 on the same rho fixtures — the
percentile bootstrap of pi is distorted exactly when the companion parameter sits pinned at its
s = 0 boundary. At s_hat = 0 the two-parameter model DEGENERATES to the one-parameter model, so
the one-parameter interval is the natural profile interval there.

REPAIR (frozen): `anchored_sync` gains `boundary_fallback` (default False — every committed
artifact reproduces byte-identically on the old path). When True and the point estimate lands
at s_hat = 0, the pi interval is the percentile interval of the ONE-PARAMETER solutions on the
SAME bootstrap resamples, and the record says so (`ci_source: oneparam_boundary_fallback`).
Point estimates, s inference, misfit, refusal logic on interior fits: untouched. When
s_hat > 0, nothing changes.

## the paired re-gate (frozen)

Re-run the full R9 battery with `boundary_fallback=True`, SAME seed bases — every panel is
byte-identical to R9's, so the repair's effect is isolated to the swapped intervals. Same four
gates, same bands, no bar moves. Output to `stage_a_operating_chars_v2_result.json`.

Frozen predictions:
1. G1 clean and G1 rho30-sync ENTER the [0.90, 0.99] band.
2. No previously-passing gate regresses (G2 is arithmetically unchanged — point fits and
   misfits are identical; G3/G4 are on untouched paths; the interior-dose coverage rows may
   only change through replicates whose s_hat = 0).
3. Any prediction missed = CLOSED_NEGATIVE for the repair; the v1 datasheet stands as the
   honest record and the boundary-aware bootstrap becomes the owed alternative.

## disclosure

This is a PAIRED repair verification on the panels that diagnosed the defect — deliberately so
(it isolates the repair), and therefore NOT an independent confirmation. Fresh-seed
confirmation belongs to the next full characterization (Stage B's own calibration, or any
future R-series re-run with new bases). The datasheet will carry this caveat.

## riding along, non-gating

The instrument graduates into the package as `styxx/anchors.py` (`audit_panel`): the repaired
estimator, both refusal classes, stratified garbage accounting, and PER-DATASET misfit-null
calibration by parametric bootstrap (simulate the fitted independent model at the anchor-
estimated rates; p-value for the observed misfit) — the piece R9's design-point threshold
could not provide. Scope statements ship in the docstring with R9's measured rates: the misfit
flag has power against gross violations only (smooth silent-wrong 0.64–0.88 measured at the
Stage-A design point); y-correlated keys and beta non-exchangeability are construction-borne
risks, not statistically policeable. Tests must pass in the repo suite; no release action.

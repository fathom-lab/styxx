# PREREG — R11: selective activation (the pre-test estimator, characterized before trusted)

date: 2026-07-20
status: FROZEN before the scored run; committed with the code.
subject: the residual mechanism R10 decomposed — on clean panels the profile fit ACTIVATES a
small phantom s in the majority of replicates (117 of 200; mean spurious s 0.018), dragging pi
and defeating both interval constructions (coverage 0.812 in that regime). The licensed repair:
engage s only on EVIDENCE. Pre-test estimators have documented coverage pathologies of their
own, so this one is characterized under the full battery before any seal is granted or any
package default changes.

## the frozen estimator (v3, additive — no committed path changes)

`anchored_selective(V, neg, pos, rng, tau, ...)` in `anchored_stage_a.py`:

1. One profile solve on the 11-moment system gives the whole grid: pi(s) and cost(s) for every
   s. The OFF model is the s = 0 column of the SAME system (not the 10-moment 1-param
   estimator — one system, one comparison; no apples-to-oranges cost deltas).
2. Evidence statistic: improvement = cost(s=0) − cost(s_hat), free from the same solve.
3. ACTIVATE iff improvement > tau and s_hat > 0. Point estimate = the selected model's (pi, s);
   s reported as 0.0 with `activated: false` otherwise.
4. THE BOOTSTRAP MIMICS THE SELECTION: every resample recomputes its own improvement, applies
   the SAME tau, and contributes the pi of ITS selected model. The interval prices selection
   uncertainty — the thing both prior constructions ignored.
5. Refusal unchanged: selected unclipped pi outside [0,1] beyond the bootstrap ⇒
   VOID_ANCHORS__nonexchangeable. Misfit = the selected model's cost over its dof.

tau is frozen BY PROCEDURE, not by value: the 95th percentile of the point-fit improvements on
the OC1 CALIBRATION set (100 clean replicates, seed base 10000) — a cheap point-fit pass, no
bootstrap. By construction the clean activation rate targets 0.05; the VALIDATION set measures
what is achieved.

## the paired battery (same seed bases as R9/R10 — disclosed as paired, not independent)

Full R9 battery re-run with the v3 estimator (`--v3`, own result file). Because tau reuses the
calibration set, the clean gates move to the VALIDATION set only (disclosed change from R9's
pooled-200):

- G1 pi-CI coverage in [0.90, 0.99]: clean VALIDATION (100), rho30 sync-arm (80), sync05 (80),
  sync15 (80), rho30 one-param arm (80, unchanged path).
- G2 misfit false-alarm in [0.01, 0.12] on validation (selected-model misfit, threshold = cal
  95th percentile, same split-sample procedure as R9).
- G3 deaf VOID rate >= 0.93 (untouched path).
- G4 clean false-refusal <= 0.02.
- G5 (new, calibration-shaped): achieved clean ACTIVATION rate on validation in [0.01, 0.12]
  (nominal 0.05 by construction of tau).

Frozen predictions: (P1) G1 clean-validation ENTERS the band — this is the repair's whole
claim; (P2) G1 sync15 remains in band; (P3) no untouched-path gate regresses. The dose-0.05
row is the measured QUESTION of this run — the selection trades phantom activation on clean
against missed activation at small doses, and the activation power per dose (0.02/0.05/0.15)
plus the coverage consequence is reported against the same band without a directional
prediction. Any missed gate is CLOSED_NEGATIVE for its clause, verbatim.

## consequence rules (frozen)

- ALL gates green ⇒ the datasheet is fully sealed at this design point and `styxx.anchors`
  adopts selective activation as the default, with tau calibrated PER DATASET from the
  parametric-bootstrap null already in `audit_panel` (null draws yield null improvements; tau =
  their 95th percentile), regime notes updated to R11's measured rates, tests updated, full
  suite green.
- G1 clean misses ⇒ the selective repair joins the fallback in the CLOSED_NEGATIVE ledger, the
  v1 datasheet stands, the module does NOT change, and the owed alternative is a
  selection-aware interval from the literature (not invented mid-cycle).
- Mixed (clean seals, a dose row falls out) ⇒ the datasheet updates to the measured trade-off;
  module adoption is decided by which regime Stage B actually inhabits and is deferred to the
  Stage-B prereg, not decided here.

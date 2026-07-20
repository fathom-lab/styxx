# PREREG — R8: the sync-corrected anchored estimator (R3's replacement claim)

date: 2026-07-20
status: FROZEN before the scored run. Committed before any full-n result exists.
harness: `papers/anchored-validity/anchored_stage_a.py` (extended in the same commit as this file)

## the burial this prereg names (rail compliance)

`R3:anchored_flat_across_dose` was buried CLOSED_NEGATIVE in cycle 44 (commit eec82e5): the
pre-fix flatness was exact algebraic cancellation from feeding the sync-bearing stratum to both
the anchors and the rate estimation. **This prereg does not resurrect that claim.** The
uncorrected 1-parameter estimator's bias is now the EXPECTED defect, with its own dose-response
check; recovery is claimed only for a NEW estimator with an explicit sync parameter, under the
bars below.

Cycle 44's kill #2 also constrains the design: a constructed detector stratum's fire rate does
not estimate the wild sync rate (it measures a constructed population), so the panel's suggested
"correction from the detector fire-rate" is unworkable as stated. The only label-free source of
s is the organic moment system itself.

## the estimator

All-judge master key at wild rate s: every moment the 1-parameter system uses becomes
`m_k = s + (1-s) * (pi*B_k + (1-pi)*A_k)` with A_k, B_k measured on CLEAN anchor strata
(first moments, kept-pair second moments, plus the all-kept-fire moment, where the +s intercept
is most visible because A, B are small products there). Two unknowns, J + C(J,2) + 1 equations:
overdetermined, lack-of-fit still observable. Solved by profile WLS: s on a frozen grid
[0, 0.6] step 0.002 (s is a rate — nonnegative by construction), pi closed-form and UNCLIPPED at
each s, argmin of weighted residual. The fix-3 refusal branch carries over unchanged: unclipped
pi outside [0,1] beyond the bootstrap -> VOID_ANCHORS__nonexchangeable.

The 1-parameter estimator `anchored()` is NOT modified — every settled panel result stands on
its existing code path. The new arm is `anchored_sync()`, and it does not replace the default.

## stream discipline

New random draws come only from a dedicated generator (`rng_boot`, seed SEED+7919) and from the
main stream strictly AFTER the R7 block. R1–R7 therefore reproduce cycle 44's realizations
draw-for-draw; the R3 detector stratum changes trip-rate but not draw count. Every pre-existing
frozen bar re-evaluates on identical data. No bar moves.

## frozen constants (new)

- `S_TOL = 0.03` — sync-rate recovery tolerance
- `S_NULL = 0.02` — max phantom sync on sync-free fixtures
- `DETECTOR_TRIP = 0.80` — detector-stratum construction strength, now a preregistered design
  parameter (cycle 44 proved an ambient-rate stratum licenses nothing)
- dose-growth margin stays 0.005, now ONE-SIDED (panel fix 14)

## frozen bars — R3 replacement (gating)

At doses sync in {0.08, 0.15}, sync on organic items only, clean anchors, detector at
DETECTOR_TRIP:

1. `R3:ds_fails_the_same_bar_anchored_meets` — unchanged.
2. `R3:ds_bias_grows_with_dose` — unchanged values, one-sided form (fix 14).
3. `R3:uncorrected_bias_grows_with_dose` — the 1-param estimator's error grows one-sidedly with
   dose. The defect is now the claim; silence about it was the old bug.
4. `R3:detector_at_construction_strength_prices_the_key` — min per-judge z > 3 at both doses.
5. `R3:sync_corrected_recovers_pi` — |pi_hat − pi| <= PI_TOL_GOOD (0.03) at both doses.
6. `R3:sync_corrected_recovers_dose` — |s_hat − sync| <= S_TOL at both doses.

## frozen bars — R8 fixtures (gating)

- **R8a no-phantom-sync (two-sided admissibility of the new knob):** clean independent panel ->
  s_hat <= S_NULL, |pi_sync − pi_1param| <= 0.015, |pi_sync − pi| <= 0.03. A knob that invents
  sync where there is none is a free parameter laundering misfit, and fails here.
- **R8b smallest dose:** sync 0.05 on real only -> |pi − 0.35| <= 0.03 and |s_hat − 0.05| <= S_TOL.
- **R8c correlation is not sync:** rho 0.30, exchangeable anchors -> s_hat <= S_NULL and
  |pi_sync − pi_1param| <= 0.02. The bad-day factor lives in the anchor-measured pair moments;
  s must not absorb it.
- **R8d misspecified keys (recovered-or-flagged, the silent-wrong gate):** (i) partial-strength
  key p=0.7 all judges, (ii) full-strength key on judges {0,1} only; both sync 0.15 real-only.
  PASS iff |pi − 0.35| <= 0.03 OR chi2/df > R8a's clean-panel chi2/df OR an explicit VOID.
  The ONLY failing outcome is a silent confident wrong number — exactly the licence-voiding
  class. (Single-draw ordinal misfit comparison; replicate version owed under fix 9.)
- **R8e refusal survives the second parameter (on R7e's exact data):** pooled-detector
  accounting still -> VOID_ANCHORS__nonexchangeable (s >= 0 cannot rescue targets that sit BELOW
  the contaminated anchor alpha); stratified twin -> ESTIMATED with |pi − 0.35| <= 0.03 and
  |s_hat − 0.02| <= 0.02 (the twin's organic sync IS 0.02 — the new arm should read it).

## frozen bars — panel fix 5 lands (gating), fixes 12/13/14 (mechanical)

- R2 anchored alpha transfer, de-tautologized: compare a_hat to the ORGANIC realized alpha
  `V[y==0].mean(0)`, per judge, bar = 3*binomial-SE (noise-aware; a flat 0.03 on a max over 4
  judges at K=400 fails by noise alone ~half the time — that is fix 9's lesson, applied).
- DS alpha claim earns its keep or dies: `max_j |ds_alpha_j − realized_j| > ALPHA_TOL` at
  rho=0.45 AND ds alpha error > anchored alpha error at rho in {0.30, 0.45}. If either half
  fails, the corresponding docstring sentence is struck — that is a result, not a rescue.
- fix 12: dead line in simulate_panel deleted (no rng draws consumed — stream-safe).
- fix 13: PI_TOL_FAIL deleted from constants and the bars block (it gated nothing);
  ALPHA_TOL is now wired (DS-side bar above).
- fix 14: dose-growth checks one-sided (`prev + 0.005 < next`).

## what a miss means

Any missed bar above is CLOSED_NEGATIVE for that clause, reported verbatim. If bars 5–6 of the
R3 set miss, the sync correction is dead and R3's anchored side ships as pure refusal + detector
scoping — Stage A stays red and the re-panel stays blocked. Near-bar is a miss.

Scope: every R8 claim is for ALL-JUDGE keys; partial-strength and subset keys are covered only
by the recovered-or-flagged gate, and Stage B may not cite R8 beyond that scope.

# FINDING -- the organic moments price the master key; Stage A runs green

date: 2026-07-20
subject: `papers/anchored-validity/anchored_stage_a.py` @ prereg 88faba2 (R8 + panel fixes 5/12/13/14)
receipts: `papers/anchored-validity/anchored_stage_a_result.json`
prereg: `papers/anchored-validity/PREREG_R8_sync_corrected_2026_07_20.md`, committed with the code
before the scored run; every bar below was frozen there.
verdict: **SURVIVED -- all 32 checks green, all_ok true.** The re-panel is unblocked.

## the claim, and why it took this shape

Cycle 44 buried R3's flatness claim (algebraic cancellation from a contaminated stratum) and
proved in the same breath that the panel's suggested repair could not work: a constructed
detector stratum's fire rate estimates a constructed population, never the wild sync rate. That
left exactly one label-free source of the master-key rate -- the organic moment system itself. An
all-judge key adds the same +s intercept to every anchor-pinned moment, so a two-parameter
(pi, s) system stays overdetermined at J + C(J,2) + 1 equations. The prereg named the burial,
claimed recovery only for the new estimator, and froze every bar. The run is the first scored
evidence, and it survived on all of them.

## what the receipts say

**The correction finds both the prevalence and the dose.** Per
`anchored_stage_a_result.json`: at wild sync 0.08 with clean anchors, the sync-corrected arm
returns pi 0.3555832086189043 (true 0.35) with s_hat 0.074; at sync 0.15, pi 0.32129036253941684
with s_hat 0.178. The same fixtures leave the 1-parameter estimator at errors of
0.0823057567403998 and 0.1618554515367725 -- the dose-growing defect is now its own passing
check rather than a silence. At the smallest dose, 0.05, the corrected arm reads pi
0.3335963347527957 and s_hat 0.076; on the R7e stratified twin, whose ambient sync is 0.02, it
reads s_hat 0.012.

**The knob refuses what it must refuse.** On the clean panel s_hat is 0.000 and the corrected pi
matches the 1-parameter estimate to three decimals; under the bad-day factor at rho 0.30, s_hat
is again 0.000 -- correlation is not laundered into sync, because the anchor-measured pair
moments already carry it. On R7e's pooled-detector data the refusal branch survives the second
parameter: s cannot be negative, so it cannot explain targets sitting below the contaminated
anchor alpha, and the verdict stays `VOID_ANCHORS__nonexchangeable` at an unclipped
-0.3634153106917754.

**Misspecified keys do not produce silent numbers.** A partial-strength key (p 0.7) is absorbed
almost correctly (pi error 0.029) with the misfit already elevated at 7.600948667895569 against a
clean-panel 2.804406982968762. A judge-subset key defeats the point estimate -- pi
0.4169669801543717, error 0.067 -- and is caught only by the misfit, 52.872862527515636. That
number is the scope boundary made visible: **Stage B must treat elevated lack-of-fit as
disqualifying the estimate, not as a footnote**, because within the all-judge model the wrong
answer arrives with a confident-looking CI.

**The alpha docstring claim finally earned its keep** (panel fix 5). De-tautologized against the
organic realized alpha: anchored transfer errors 0.017/0.022/0.019 across the rho sweep, inside
3 binomial SEs at every dose, while DS's alpha errors run 0.062/0.125/0.181 -- beyond ALPHA_TOL
at rho 0.45 and beyond the anchored error at both correlated doses. The sentence the panel said
was unearned is now a measured result.

## honest edges, recorded not hidden

- s_hat runs slightly hot at low dose (0.076 for 0.05, 0.074 for 0.08) and slightly hot at
  0.15 (0.178). Inside the frozen S_TOL 0.03 everywhere, but the small-dose bias direction is
  worth a replicate-rate look when the panel's replicate-coverage machinery lands.
- Every R8 verdict is a single draw, and the R8d misfit comparison is a single-draw ordinal
  test. The prereg says so; Stage B may not cite R8 beyond the all-judge scope or without its
  own replicate fixtures.
- The reduced-n selftest stays red on full-n bars (known fix-9 class). The scored artifact is
  the full-n run.

## state of the gate

Stage A: **all 32 checks green**, including the complete fatal-fix set from the 2026-07-19 panel
re-measured on identical draws (R1-R7 reproduce cycle 44's realizations draw-for-draw by the
prereg's stream discipline). The panel's re-panel protocol -- fixes 1-4 landed, run green,
re-panel on the fatal fixtures -- is now satisfiable. Owed before Stage B regardless of the
re-panel's outcome: the panel's remaining major set (the R5 licensing-fork enforcement, the R2
check rename, the R4 partial-keep path and deaf-panel VOID rate, replicate-rate CI coverage,
partial-strength master-key arms) and the Stage-B prereg obligations (delta bounding,
semi-supervised DS comparator, licensing with observables). Freeze stays the operator's call.

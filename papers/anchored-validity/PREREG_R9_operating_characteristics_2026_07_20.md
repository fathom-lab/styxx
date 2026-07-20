# PREREG — R9: operating characteristics of the anchored instrument (the datasheet)

date: 2026-07-20
status: FROZEN before the scored run. Committed with the harness, before any full-R result exists.
harness: `papers/anchored-validity/stage_a_operating_chars.py` (consumes `anchored_stage_a.py`
@ the re-panel scope-corrected revision; touches no Stage-A check logic).

## what this is

The re-panel's F2 proved the program's own point: a frozen point bar passed on one draw is not a
property. R9 replaces assertion with CHARACTERIZATION — every quantity becomes a RATE over
replicates with a confidence interval, and the misfit statistic gets a calibrated null so it can
gate. The deliverable is the instrument's datasheet at the Stage-A design point. This discharges
the 2026-07-19 panel's fix 9 (replicate-rate machinery) for the sync arm, the deaf-panel VOID
rate half of fix 8, and the re-panel's F2/F3/F6 in one build.

Design principle, stated before the run: **gates are calibration-shaped; performance is
measured, not gated.** The instrument passes if it is HONEST about itself (coverage in band,
false-alarm at nominal, refusal fires when it must and only then). How GOOD it is — error
quantiles, power, phantom rate — gets printed on the datasheet whatever the numbers say. A
datasheet with mediocre numbers is a valid result; a gate the performance could never fail is
not a gate.

## pilot disclosure

The re-panel probe receipts (`_repanel_probe_receipts_2026_07_20.json`, committed) are PILOT
data for several of these quantities: pi/s CI coverage 0.933/0.933 (30 reps, n=3000), clean
misfit band 2.0–5.3 (5 seeds), one phantom-sync clean seed in five, deaf-panel single-cohort
VOID 95.45% (panel-measured, cycle 44 receipt). Bars below are set AFTER seeing that pilot and
are disclosed as pilot-informed; the scored run uses disjoint, frozen seed bases and R large
enough that a pilot-sized fluke cannot carry a gate.

## frozen design

Design point: alphas [0.15, 0.20, 0.10, 0.18], betas [0.85, 0.80, 0.90, 0.78], pi 0.35,
n 6000, K 400 per anchor stratum. n_boot 200 (families needing CIs), 100 (violation families).
Seed bases (replicate i uses base + i; never the scored Stage-A stream): clean-cal 10000,
clean-val 20000, rho30 30000, sync05 40000, sync15 50000, sync02 60000, deaf 70000,
contam10 80000, keypos 90000, betaplus 100000, oneparam-rho30 110000.

Families and R:
- OC1 clean (independent, exchangeable): R = 100 calibration + 100 validation.
- OC2 rho 0.30 (exchangeable anchors share rho): R = 80 sync-arm; R = 80 one-param arm.
- OC3 sync-on-real-only (all-judge, truth-independent), doses 0.05 / 0.15 (R = 80 each) and
  0.02 (R = 60, characterization only).
- OC4 deaf panel (alphas 0.45, betas 0.52): R = 60. Plain informativeness gate AND the
  noise-margin variant (gate + 3*sqrt((a(1-a)+b(1-b))/K)) both recorded.
- OC5 violations, R = 50 each: (a) contamination — 10 percent of the negative stratum replaced
  by trip-0.80 detector garbage; (b) y-correlated key — fires on true positives at rate 0.15;
  (c) anchor-beta pessimism — organic betas +0.10. All three are SILENT cases from the re-panel.

Misfit null, split-sample: the threshold is the 95th percentile of the OC1 CALIBRATION set's
chi2_per_df; achieved false-alarm is measured on the disjoint VALIDATION set; power is the
fraction of each OC5 family exceeding the threshold. The threshold is design-point-specific and
says so on the datasheet; Stage B owes its own calibration.

## frozen gates (calibration-shaped; missing any = CLOSED_NEGATIVE for that clause)

- G1 pi-CI coverage in [0.90, 0.99] separately on: OC1 (pooled 200), OC2 sync-arm (80),
  OC3 dose 0.05 (80), OC3 dose 0.15 (80), OC2 one-param arm (80).
- G2 achieved misfit false-alarm on OC1 validation in [0.01, 0.12] (nominal 0.05).
- G3 OC4 deaf VOID rate >= 0.93 under the plain gate (pilot-informed; the noise-margin variant
  is reported unbarred).
- G4 OC1 false-refusal rate (VOID_ANCHORS on clean data) <= 0.02.

## characteristics (reported with 95 percent Wilson intervals; NO bars, by design)

Phantom-sync rate on clean (s_hat > S_NULL); err quantiles (median, p90) per family; s-recovery
quantiles at each dose; s-detection power (s_ci lower edge > 0) at doses 0.02 / 0.05 / 0.15;
misfit power against each OC5 family at the calibrated threshold; SILENT-WRONG rate per OC5
family (err > 0.03 AND misfit <= threshold AND verdict ESTIMATED) — the number the re-panel
could only spot-check; s_at_grid_edge rate (wired into records here, discharging F6 for R9).

## what a miss means

A missed gate is CLOSED_NEGATIVE for that clause and the datasheet ships anyway with the
measured value — the instrument's honesty about itself is the product. No characteristic can
fail, and therefore no characteristic may ever be cited as if it had been gated. Smoke runs
write only *_SMOKE_INVALID* files.

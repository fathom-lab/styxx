# FINDING -- the instrument gets a datasheet, and the datasheet withholds two seals

date: 2026-07-20
subject: `papers/anchored-validity/stage_a_operating_chars.py` @ prereg 2e881c9
receipts: `papers/anchored-validity/stage_a_operating_chars_result.json`
prereg: `papers/anchored-validity/PREREG_R9_operating_characteristics_2026_07_20.md`, committed
with the harness before the scored run; replicate families with frozen disjoint seed bases.
verdict: **DATASHEET_SHIPPED -- six of eight calibration gates pass; two are CLOSED_NEGATIVE
with a diagnosed mechanism.** This is the program's first instrument that states, with measured
rates, where it can be trusted and where it fails silently.

## the two honest failures

**The sync arm's confidence interval undercovers exactly where its second parameter sits on its
boundary.** Clean-panel pi-CI coverage measured 0.835 with Wilson interval [0.777, 0.880] over
200 replicates -- decisively below the frozen [0.90, 0.99] band, upper edge included. At rho
0.30 the sync arm covers 0.875. Both clauses are CLOSED_NEGATIVE. The mechanism is visible in
the same table: wherever true s is INTERIOR the coverage is nominal (0.912 at dose 0.05, 0.938
at dose 0.15), and the one-parameter arm on the very same rho fixtures covers 0.925. The
percentile bootstrap of a parameter pinned at s = 0 distorts the companion interval -- a known
boundary pathology, now measured in this instrument. The repair path is narrow and testable:
when s_hat = 0, report the one-parameter interval (which covers), or a boundary-aware bootstrap.

**What passed:** misfit false-alarm calibrated at 0.090 on held-out clean replicates (band
[0.01, 0.12], threshold 6.19273753216957 = the calibration set's 95th percentile); deaf-panel
VOID rate 0.967 under the plain gate -- and 1.000 under the noise-margin gate the original
panel proposed, which the datasheet now recommends adopting; false-refusal rate 0/200 -- the
refusal branch never fired on clean data across every replicate; sync-dose coverage in band
both doses; the one-parameter arm's coverage in band.

## the numbers that change how Stage B must be designed

**The misfit statistic cannot gate smooth violations, and now that is a measured fact rather
than a fear.** At the calibrated threshold, power against the three silent violation families
the re-panel exhibited: 0.36 against ten-percent anchor contamination (silent-wrong rate 0.64,
median pi error 0.082); 0.04 against the y-correlated key (silent-wrong 0.74, median error
0.0451); 0.12 against anchor-beta pessimism (silent-wrong 0.88, median error 0.0929). Misfit
catches gross structure (the specialist at 575, subset keys at 11-28 in the re-panel) and is
nearly blind to smooth ones. Consequence, stated plainly: y-correlated keys, beta
non-exchangeability, and moderate contamination CANNOT be policed statistically at this design
point -- they must be excluded by ANCHOR CONSTRUCTION (graded ladders, labeled slices, provenance
controls), which is exactly the set of Stage-B obligations the original panel's binding lists
already carry. The datasheet converts those from advice into requirements with rates attached.

**The phantom-sync rate is 0.24** (Wilson [0.186, 0.304]): on a quarter of clean panels the
knob reads s above S_NULL. The scored Stage-A pass at s = 0.000 was a favourable draw, as the
re-panel suspected. The damage is bounded -- clean median pi error 0.0129, ninetieth percentile
0.0334 -- but the two-sided claim must now be quoted at its measured strength, not its
single-draw strength.

**The knob's detection threshold is now characterized:** power to place s's interval above zero
is 0.283 at dose 0.02, 0.700 at dose 0.05, 1.000 at dose 0.15. Below roughly a five percent
master-key rate, absence of evidence is not evidence of absence, with numbers.

**Grid-edge rate 0.0** across all families at this design point; the edge flag is wired into
every R9 record, discharging the re-panel's F6 within this harness.

## what this instrument now is

An anchored prevalence estimator that ships with: a refusal branch measured never to false-fire
(0/200), a deaf-panel gate that voids at 0.967-1.000, calibrated self-distrust with a stated
false-alarm rate (0.090), stated coverage in its trustworthy region (0.912-0.938 where s is
interior), stated coverage failure at its boundary (0.835, withheld seal), and stated blindness
(silent-wrong 0.64-0.88 against smooth violations). No prior artifact in this program could
make statements of that form. The datasheet is the deliverable; the two withheld seals are what
make the six granted ones worth anything.

## owed next, in order

1. The boundary repair (fall back to the one-parameter interval at s_hat = 0, or a
   boundary-aware bootstrap), then re-run OC1/OC2 and re-gate G1 -- the narrow fix the
   mechanism licenses.
2. Adopt the noise-margin informativeness gate (measured 1.000 VOID on deaf panels vs 0.967
   plain).
3. Stage-B prereg: anchor-construction defenses carry the load the misfit statistic measurably
   cannot; the misfit gate is licensed against gross violations only; boundary-aware CIs; its
   own misfit-null calibration (the threshold here is design-point-specific).

Freeze remains the operator's call; the datasheet, including its withheld seals, is the honest
basis for it.

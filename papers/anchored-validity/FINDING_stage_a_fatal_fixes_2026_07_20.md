# FINDING -- the fatal fixes land, and they kill two of the claims they were sent to protect

date: 2026-07-20
subject: `papers/anchored-validity/anchored_stage_a.py` @ fixes 1-4 + fix 10/11
receipts: `papers/anchored-validity/anchored_stage_a_result.json`,
`papers/anchored-validity/_stage_a_panel_2026_07_19.md`
prereg: commit f8bede3, committed with its frozen prediction before the scored run existed
verdict: **FIXES LANDED, STAGE A NOT GREEN.** The freeze stays blocked.

---

## what was asked and what happened

The panel of 2026-07-19 returned NO_GO with two surviving fatals and named fixes 1-4 as the
gate on a re-panel. All four landed. The four channels the panel called inexpressible are now
expressible and measured, and the new refusal branch is exercised in both directions on
identical data.

Stage A did not come back green. `anchored_stage_a_result.json` carries `all_ok` false on 2
checks, and both failures are caused by the fixes themselves -- specifically by fix 4, the
stratified anchor accounting. Neither was rescued. The path back to a re-panel is now longer
than the panel expected, and that is the honest state.

## the fixes, and the evidence each one reads the data

**fix 1 (beta channel).** `make_anchors()` gains `beta_shift`; the sensitivity channel of
non-exchangeability was a structural blind spot, not an untested one -- `alpha_shift` is a no-op
for a positive anchor, which fires at beta. R7a now drives it: with organic sensitivities 0.10
below the anchors', the estimator returns pi 0.28702039531361484 against a true 0.35, an error
of 0.06297960468638514. The bias runs DOWNWARD -- favourable to the audited system, which is the
direction that matters.

**fix 2 (fixtures).** Four fixtures, all four confirming the panel's read. Per
`anchored_stage_a_result.json`: uniform channel-gain err 0.06297960468638514; 1-of-J specialist
err 0.21937725305046663; sync-on-real-only err 0.15295935085716006; anchor-rate-mismatch err
0.12617000299413064. Every one of these was a silent, confident number before this cycle.

**fix 3 (the refusal branch).** The terminal clip in `solve_pi` turned an observable impossibility
into pi_hat 0.000 with a tight CI. The weighted-least-squares solution is now unclipped, and the
pooled-detector fixture in `anchored_stage_a_result.json` returns
`VOID_ANCHORS__nonexchangeable` on an unclipped prevalence of -0.3405988011328714 whose bootstrap
interval, -0.4839271119166536 to -0.21262255959379806, never reaches back inside [0,1].

**fix 4 (stratified accounting).** Detector strata no longer estimate error rates. The two-sided
proof is the R7e pair, which runs both accountings over the SAME data: pooled refuses, stratified
returns pi 0.3771912776602396, an error of 0.0271912776602396 inside the 0.03 recovery bar. A
refusal that fires on one accounting and withholds on the other is reading the data, not a
constant.

**the frozen prediction held.** Written into commit f8bede3 before the run: misfit strictly
exceeds the exchangeable control on every non-exchangeable fixture. Measured in
`anchored_stage_a_result.json` against a control of 2.161 -- 4.976, 636.418, 12.234, 9.268. It
ships REPORTED AND NOT GATING; a bar chosen after seeing those numbers is not a bar.

## the two kills

**R3's flatness claim is dead, and it was always algebra.** The pre-fix fixture handed the same
sync-bearing stratum to `make_anchors` and to the rate estimation. That makes the anchor rates
s + (1-s)a and s + (1-s)b, so the (1-s) factor cancels exactly out of the WLS ratio and the
estimator looked immune to the master key at any dose. It was not immune; the contamination was
doing the work. Under the accounting fix 4 mandates, `anchored_stage_a_result.json` reports
anchored error 0.0823057567403998 at the low dose and 0.1618554515367725 at the high dose, against
a bar of 0.03. The pre-fix committed run recorded 0.012 and 0.003 for the same scenario. This is
CLOSED_NEGATIVE, and it retires a claim the file has been making since e1ce286.

**The detector stratum does not price the master key at the ambient rate.** The check requires
every judge to fire on garbage more than 3 sigma above its inert-negative alpha. Measured: minimum
z of 1.75 and 2.78, on all-fire rates of 0.075 and 0.1225. It fails. The cause is a real design
constraint and it is being recorded, not repaired: a garbage stratum that merely samples the wild
sync rate is not a detector. The R7e stratum, built to trip at 0.80, does clear -- which is the
whole content of the finding. An auditor gets pricing only from anchors constructed to trip the
failure mode much harder than the wild does, and the ambient-rate stratum this fixture used
licenses nothing. The bar was not moved to rescue it.

## what this costs the program

R3's anchored-side headline is now scoped twice over. Anchors price a master key only when the key
fires ON the anchors (R7c: clean anchors, `master_key_detected` false, organic error
0.15295935085716006) AND when the anchor stratum trips it far above ambient. Both conditions are
Stage-B construction obligations, and neither was visible before this cycle.

The DS-side results are untouched -- `anchored_stage_a_result.json` still carries DS error 0.039
and 0.077 across the master-key doses, and the R1/R2/R4/R5/R6 checks are unchanged and green. The
DS-misspecification core the panel exonerated is not in question here.

## owed, and now re-ranked

The re-panel the panel scoped cannot run on this state: its own protocol says fixes 1-4 land and
Stage A runs green. It does not. Before a re-panel is worth an agent's time:

1. Decide R3's replacement claim, prereg it, and earn it -- either a correction that uses the
   detector fire-rate to price the sync, or an explicit refusal when the garbage stratum trips.
   Do not restate flatness.
2. Specify detector-stratum construction strength as a preregistered Stage-B parameter, since the
   ambient-rate stratum demonstrably licenses nothing.
3. The major set (panel fixes 5-9, 12-15) is still entirely owed, including the tautological R2
   alpha diagnostic and the single-draw CI checks.

The single-draw ordinal comparison behind the misfit statistic is itself under-powered, and this
document does not claim otherwise. It is a reported diagnostic with no bar, and Stage B may not
lean on it until it has a replicate-rate fixture of its own.

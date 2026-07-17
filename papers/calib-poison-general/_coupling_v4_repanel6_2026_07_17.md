# RE-PANEL #6 -- rebuilt coupling v4. Verdict: NO_GO_redesign (1 fatal, the F1 fix regressed).

**Fathom Lab - papers/calib-poison-general - 2026-07-17. Nothing frozen, no GPU spent. Run
`wf_d2dcae6b-44f` (10 agents, 0 deaths -- the lean 3-lens design held). Target: the REBUILT v4
(the effect-size + sign-consistency redesign that answered panel #5). Mandate: verify panel #5's 5
fatals + 3 majors are actually closed, and hunt regressions the redesign introduced.**

## Verdict: NO_GO_redesign

Panel #5's fatals were closed by the rebuild -- but the F1 fix (span-narrowing the slope fit to
[2,8]) INTRODUCED a new fatal, and the re-panel caught it before freeze.

### FATAL (Finding 1) -- reachable single-seed COUPLED

`dose_response` uses `n = len(slopes)` (non-None per-seed slopes only) as the majority denominator.
`_per_seed_slope` returns None when a seed has fewer than 2 distinct FITTED ranks inside the
narrowed span [2,8]. Such a seed still passes `MIN_PAIRS_PER_SEED` (pairs span ranks 2..24), so it
is dose-admissible, yet it is silently dropped from the denominator. Reproduced: 4 stalled (None) +
1 above-bar seed -> `n=1`, `pos_above=1 > 0.5` -> `COUPLED__dose_slope_price_measured_1p5B`
certified from ONE seed. This contradicts the prereg ("strict majority of the 5 seeds"; "the seed
is the replication unit"). With the p-gate removed by operator decision, the seed majority IS the
whole finding, so this -- panel #5's "sign-majority denominator" MINOR -- is promoted to a
reachable, paper-founding, unearned favourable verdict. It is reachable by construction pre-freeze,
independent of how this run's schedule lands; not a naming divergence.

**Fix:** majority over the DOSE-ADMISSIBLE seed count (None-slope = not above bar):
`coupled = len(pos_above) > len(admissible)/2 and len(neg_above) == 0`; and a pre-committed guard
that fewer than `MIN_ADMISSIBLE_SEEDS` non-None slopes -> `VOID_COUPLING__underpowered`. Load-bearing
--dry regression: stalled seeds must never yield COUPLED.

### MAJOR (Findings 2 & 5, one defect) -- dose-graded format drift can ride into COUPLED

Trained-checkpoint format-invariance is measured but REPORTED-ONLY; its "comparable" demotion is an
unfrozen post-result human judgment ("no threshold tunable post-result" violation; panel #5's FI
recurrence major partially still open). **Fix:** a pre-committed NUMERIC FI-slope downgrade wired
into `compute_verdict` (accumulate-arm FI abs_delta slope over [2,8]; if it reaches >= 0.5x
MIN_EFFECT_SLOPE, auto-downgrade COUPLED -> PARTIAL), mirroring the echo-subtractive branch.

### 8 minors (auto-accepted)

Distinct string for the style-downgrade path; rename/clarify the "_MDE" bounded-null string
(the gate is the 0.0152 min-price-of-interest, not a detection MDE); report guard-fire fraction
among gate positives; reword the last hedge-forming antonym gold + stoplist extension; make the F5
subtractive a true paired subset (or disclose the downward bias); complete the prereg Arms-deltas
list (lam==0 forward-skip, trained-FI subsample); reconcile the 4dp paired_delta reproducibility;
factor the arms_key cache filter into one shared function + an unmeasurable-instrument dry case.

## Disposition

All fixes dispatched to the rebuild agent. Next: fix -> re-green selftest/dry -> re-panel #7 focused
on the fatal-path closure + the FI downgrade -> only then the operator's freeze call. The re-panel
did exactly its job: the rebuild fixed 5 fatals and quietly created a 6th, and the sixth was caught
before a single GPU hour -- for the sixth time in this arc.

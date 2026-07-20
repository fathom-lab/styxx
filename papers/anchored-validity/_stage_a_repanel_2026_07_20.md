# stage-a re-panel receipt -- anchored validity

date: 2026-07-20
subject: anchored_stage_a.py @ fc9528f + anchored_stage_a_result.json (scored, 32/32 green)
protocol: the path-back protocol of _stage_a_panel_2026_07_19.md -- fatal fixtures only; the
DS-misspecification core, the J=3 witness, and the 1-param CI-coverage property are settled and
were not re-litigated. The two cycle-44 burials were not re-opened.
probes: _repanel_probes_2026_07_20.py -> _repanel_probe_receipts_2026_07_20.json (committed;
fresh seeds throughout, never the scored stream).

DISCLOSURE -- panel independence. The three-lens subagent fleet was blocked at spawn time
(session limit); this re-panel executed the same probe program INLINE, in the same context that
authored the fixes. That weakens the independence of the confirmations below and it does NOT
weaken the adverse findings (an author-run probe that breaks the author's own code is, if
anything, stronger evidence). The operator may re-run the fleet independently at any time; the
probe script is committed and seeded.

---

## verdict: fatal-fix set CONFIRMED -- but NO_GO on the freeze, again

The four fatal fixes hold under attack. The new R8 layer that turned Stage A green does not hold
at the strength its point bars imply: its greens are draw-fragile, and its scope statement covers
a region where it fails silently in the favourable direction. The house rule that killed two
prior attempts applies to this one the same way.

## what HELD (confirmations, with the independence caveat above)

- **Refusal two-sidedness (fix 3).** Legitimate extreme prevalence (pi 0.02 and 0.98) is
  ESTIMATED, not voided, by both arms (errors 0.012-0.020); the pooled-detector refusal still
  fires. The branch reads data in both directions.
- **The R8d recovered-or-flagged gate survived 7 fresh attacks** (judge-subset keys at 3 doses,
  3-judge subset at 2 doses, partial-strength at 2): every err > 0.03 case had misfit 11-28
  (well above the clean band); every within-band case had err <= 0.025. No silent window found
  in the subset/partial family.
- **CI-level honesty.** Spot-check at sync 0.10 (30 reps, n=3000, disclosed): pi CI coverage
  0.933, s CI coverage 0.933. The estimator is approximately honest about its uncertainty.
- **Stratified accounting + fixtures (fixes 1, 2, 4)** behaved as shipped in every probe that
  touched them; R1-R7 had already reproduced cycle 44 draw-for-draw.

## findings

**F1 (FATAL, scope): y-correlated all-judge keys defeat the model silently, favourable
direction.** The (pi, s) model derives from a key INDEPENDENT of truth; the citable surface says
only "all-judge keys". A key firing on true-positives only (rate 0.15 among positives): pi
0.3055 (err 0.044, DOWNWARD -- favourable), s_hat 0.060, misfit 2.70, verdict ESTIMATED --
misfit inside the clean band (2.0-5.3 over 5 seeds). Key on true-negatives only: err 0.043,
misfit 1.27, silent, direction up. This is the original panel's F1 defect class: an
unconditional claim whose truth is conditional. Fix: scope every R8 claim to all-judge
TRUTH-INDEPENDENT keys (applied to the header and docstring in this commit); Stage B's threat
model must treat y-correlated keys as unpriced.

**F2 (FATAL, stability): the R8/R3 point-recovery bars are single-draw passes that do not
replicate reliably.** Fresh-seed point errors at n=6000: clean band 0.002/0.012/0.015/0.011/
0.038 (one seed exceeds the 0.03 bar and shows phantom s 0.048 > S_NULL); dose 0.05 err 0.0425;
dose 0.15 err 0.0668 with s_hat 0.204; tiny dose 0.01 err 0.080 with s_hat 0.068 (the knob is
noise-dominated below ~0.05). Across 8 fresh replicates, 4 exceed the frozen 0.03 bar the
scored run passed 6-for-6. The scored green is real but LUCKY-DRAW-COMPATIBLE; "recovers within
0.03" is not an established property of the estimator -- "CI covers at ~0.93" is. This is the
2026-07-19 panel's fix 9 ("one draw per dose licenses nothing") demonstrated against the very
checks that were added after it was filed. Fix: replicate-rate versions of R8a/R8b/R3 bars
before any freeze; recast point claims to the CI standard the estimator actually meets.

**F3 (MAJOR): a silent contamination window at moderate pooling.** Sweeping detector garbage
pooled into negatives: at 10 percent contamination, err 0.048 with misfit 4.8 -- INSIDE the
clean band -- and pi biased DOWN (favourable); the refusal never fires below 50 percent
pooling. Errors grow 0.048 -> 0.345 across 10-50 percent while misfit grows 4.8 -> 47.7. The
refusal is a far-boundary device; between "clean" and "refused" sits a silent-then-flagged ramp
whose silent end covers realistic contamination. Fix: the misfit statistic needs a calibrated
null (fix 9 family) so a Stage-B gate can sit at the right height; until then no claim that
contamination is detectable.

**F4 (MAJOR): beta-optimism is absorbed as phantom sync.** Organic betas 0.10 ABOVE anchors
(anchors pessimistic): s_hat 0.136, pi err 0.060 downward, misfit 3.79 -- inside the clean
band, silent, favourable. The prereg's no-phantom-sync checks covered clean and rho panels
only; this channel evades both. (Opposite direction and the specialist are loud: misfits 25
and 575.) Folded into the F1 scope correction; Stage B's ladder design must bound anchor-beta
pessimism specifically.

**F5 (MAJOR, disclosed power limit): the 3-SE alpha-transfer bar cannot see pi-material
shifts.** Detection limit at K=400 is a 0.058 shift; a 0.018 shift already moves pi by 0.03
(first order). The check certifies transfer only against gross failures -- roughly 3x coarser
than materiality. Already implied by the panel's labeled-slice obligation; now quantified.

**F6 (MINOR): s_at_grid_edge is computed but dropped from every stored record.** At true s 0.7
the fit clamps to the grid edge (pi err 0.271, misfit 8.6 -- only modestly elevated); the flag
exists in the return dict and nowhere in the result JSON. Wire it through and treat edge-pinned
fits as VOID-able.

**F7 (MINOR, fairness): the profile WLS is not clearly better than a 4-line plug-in.** A
trivial all-fire-excess correction matches it within noise on 3 doses (profile errs
0.043/0.022/0.067 vs trivial 0.044/0.033/0.043). The estimator's value is the refusal and
misfit machinery, not point accuracy; prose should not imply otherwise.

## path back (ordered)

1. Scope corrections to the citable surface (F1/F4) -- landed with this receipt.
2. Replicate-rate machinery (the panel's fix 9, now twice-earned): rate-based versions of the
   R8a/R8b/R3 recovery bars, the deaf-panel VOID rate, CI coverage as a rate, and a calibrated
   misfit null (F2/F3). This is the gate on any freeze.
3. Remaining majors (R5 licensing fork, R2 rename, R4 partial-keep) and the Stage-B prereg
   obligations, which now include: y-correlated keys unpriced (F1), anchor-beta pessimism bound
   (F4), labeled-slice alpha bound (F5).

Freeze remains the operator's call and is NOT recommended on this state.

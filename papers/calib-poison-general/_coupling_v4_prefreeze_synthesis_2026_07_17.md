# PRE-FREEZE PANEL SYNTHESIS -- coupling v4. Verdict: **NO_GO_redesign.**

**Fathom Lab - papers/calib-poison-general - 2026-07-17. Status: NOTHING FROZEN. The three v4
artifacts (capability_battery_gen.py, coupling_confirm_v4.py, PREREG_B2_coupling_twosided_2026_07_16.md)
remain untracked; no GPU spent. Receipts: `_coupling_v4_prefreeze_attack_findings.json` (33 findings,
5 lenses), panel run `wf_b45acce6-201`. 32 of 38 refutation votes recovered across two resumes; the
verify layer ran out on session limits before the synthesis agent, so this synthesis is written by
the main loop from the journal (as the attempt-3 synthesis was).**

## Verdict

**NO_GO_redesign.** Five fatal findings; the confirmed one (echo guard, 3/3 votes) and the four
whose refutations never completed (unrefuted fatals stand by the panel's own rule) cluster into
**two real problems plus one scoping consequence.** Only 3 of 32 recovered votes refuted anything
(all minor/major, not the statistic fatals). This does not freeze. The battery confound is a patch;
the dose-response statistic is a redesign.

## Problem 1 -- the dose-response statistic and its error model are mismatched to the data (4 of 5 fatals)

The verdict rides `slope_permutation_null(paired delta vs erased_rank, unit=seed)` -- a global-span
linear slope with a within-seed label permutation. Four fatals say the same thing from four angles:

- **Power (lens: stats).** The dose run's price is TRANSIENT -- read dips at rank 8, recovers by
  10-24. A global-span linear slope over a dip-and-recover curve has near-zero slope by
  construction, so the prereg's "power near 1" guarantee does not hold, and the favourable-for-
  erasure bounded null is reachable unearned.
- **Error model invalid (lenses: stats, gate-integration).** Within-seed permutation of a serially
  autocorrelated, deterministic-per-seed trajectory is not an exchangeable null. `perm_p` is
  anti-conservative; the disclosed MDE (permutation p95) is understated; the step-0 structural-zero
  pair worsens it. A pooled permutation over ~72 autocorrelated checkpoints is a trend detector, not
  a significance test.
- **Attempt-3 recurrence (lens: recurrence).** A significant sub-bar price (p < alpha, slope <
  MIN_EFFECT_SLOPE) currently routes to the bounded-null string -- which is then false on its own
  receipt. Exactly attempt 3's shape (silence mapped onto a claim), milder.

**The convergent fix (all four):** score at the SEED level -- 5 per-seed slopes as the units, fit
restricted to the pre-committed rank-2-to-8 span (recovery region reported, not fitted), sign-flip
or t-across-seeds as the null, MDE derived from that null. Add a frozen branch for
significant-sub-bar -> PARTIAL, so the bounded-null string is only reachable at p >= alpha OR |slope|
<= mde.

## The scoping consequence the redesign forces (operator decision)

A valid seed-level null at **5 seeds** has a hard floor: a sign-flip permutation over 5 signs has
minimum two-sided p = 2 / 2^5 = **0.0625**. **Under a valid error model, 5 seeds cannot reach
p < 0.05 -- a significant COUPLED verdict is unreachable at the current seed count.** This is not a
bug in the code; it is the honest cost of demanding a valid null. Three ways forward, all
legitimate, the choice is the operator's:

1. **Raise seed count** to >= 6 (min p = 0.03125) or more -- more GPU per the erasure run, but keeps
   a significance-based COUPLED reachable.
2. **Set STAT_ALPHA = 0.0625** pre-freeze and disclose that 5 seeds gives a one-shot sign test --
   defensible, but thin, and it must be frozen before the run with the arithmetic stated.
3. **Reframe COUPLED as an effect-size + consistency claim** (per-seed slope > bar in a strict
   majority, sign-consistent) and drop the significance gate entirely -- honest, matches the
   program's "seeds are the replication unit" stance, and sidesteps the 5-seed p-floor.

Recommendation: **(3)** -- it is the most honest fit to what the instrument can carry and needs no
extra GPU; (1) if a p-value is wanted for the paper. Either way this is a prereg-design decision
that must be made before the redesign is written.

## Problem 2 -- capability-battery confound (1 fatal + 3 majors; the patchable half)

- **FATAL, confirmed 3/3 -- echo guard prices style, not capability.** `score_item` zeroes any decode
  containing >= 2 candidate answers even when the gold is present (`apple, then mango`), on 5 of 7
  list-format sub-tasks. Dose-correlated verbosity drift on the accumulate arm (unmatched by the
  fixed arm) converts directly into paired price, and `echo_guard_fires` is absent from the `points`
  grounding surface -- the confound is invisible to `verify_replication`. Fix: add
  `echo_guard_fires` per-arm to points + schema; pre-commit a subtractive check (recompute the slope
  with guard-zeroed items excluded from both arms; if the price does not survive, downgrade to the
  bounded null).
- **MAJOR -- gold "may" is a hedging modal**, antonym golds are high-frequency adjectives; a degraded
  model that hedges scores CORRECT by containment, a false ceiling biasing toward the favourable
  bounded null. Fix: reword the April->May item; add a selftest stoplist invariant (no gold/variant
  is a high-frequency function word).
- **MAJOR x2 -- format-invariance gate on the wrong population/statistic.** It runs on all 7 sub-tasks,
  on the clean base only, before selection, at a bar (0.0722) that is 79% of the 0.0909 price of
  interest. Fix: select first, then `format_invariance_check(selected=survivors)`; measure FI at
  trained end-state checkpoints and report it; state the bar as a fraction of the price of interest.
- **MAJOR -- MIN_EFFECT_SLOPE transfers a T/F-channel price (0.0909 knowledge_drop) onto the gen
  battery scale with no channel-gain calibration.** Fix: pre-commit a gain receipt regressing gen-
  aggregate drop on knowledge drop on marker-fired posctl checkpoints (data already collected).
- **MAJOR -- mde_slope_approx is the permutation p95 mislabeled as an MDE** (dissolved by the
  seed-level error-model fix above).

## Rulings on the drafter's four open questions

- **(a) derive/gate seed split vs per-seed cross-fitting:** the seed split is acceptable IF the
  gate's sensitivity leg also carries a per-seed replication requirement (gate-integration major);
  add it. Cross-fitting breaks the single-threshold API and is not required.
- **(b) knowledge <= 0.60 destruction marker, given lambda=0 collapses knowledge by construction:**
  acceptable for a NOT-DEAF gate ONLY -- the marker is independent of the battery but endogenous to
  the arm design, so it can certify "not deaf" and nothing stronger. The prereg already scopes this
  correctly; keep the scope wording exactly, and never let the gate license a claim above NOT-DEAF.
- **(c) step-0 pairs in the slope regression:** EXCLUDE. Step-0 is a structural zero (arms identical
  before the first refit); including it inflates the trend null. Report step-0 separately.
- **(d) FORMAT_INVARIANCE_MAX = one aggregate SE, no CI:** insufficient as specified (Problem 2). Tie
  the bar to the selected aggregate and to a fraction of the price of interest, and measure it on
  trained checkpoints, not just the clean base.

## Ordered fixes before any re-panel

1. **Redesign the dose-response statistic to seed-level** (span-restricted slope, seed-unit null,
   significant-sub-bar -> PARTIAL branch). **Blocked on the operator's scoping choice above.**
2. Echo-guard: `echo_guard_fires` into points + schema + the subtractive check.
3. Battery item fixes: reword "may", stoplist selftest invariant.
4. Format-invariance: select-then-measure, trained-checkpoint measurement, bar re-tied.
5. Channel-gain receipt for MIN_EFFECT_SLOPE.
6. Per-seed replication requirement on the gate's sensitivity leg.
7. Re-panel the redesigned artifacts pre-freeze (this is now the standing discipline), THEN freeze
   is the operator's call.

## What stays true regardless

The `styxx.instrument_admissibility` primitive and the four retro-cert receipts (mount ADMISSIBLE,
private-13 VOID -> conformal root-cause -> three-way ADMISSIBLE) are shipped and public on main and
are untouched by this. The coupling question stays OPEN; the erasure bound still has no number; the
standing PARTIAL is the fallback. v4 can only reach the erasure-bound answer through an instrument
that survives its own panel -- which, for the fifth time in this arc, it did not, before any GPU was
spent.

---
*Attempt 2 fired under the null. Attempt 3 mapped silence onto a claim. v4 built a valid two-sided
gate on top -- and the panel found the gate is sound but the FINDING statistic underneath it is the
wrong shape for the data and cannot reach significance at 5 seeds under a valid null. The gate was
the hard part and it holds; the statistic is a redesign, and the redesign asks one honest question
only the operator can answer: how many seeds, or drop the p-value.*

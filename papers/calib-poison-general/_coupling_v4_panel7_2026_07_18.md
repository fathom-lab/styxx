# panel #7 -- coupling v4 ESTIMATOR-ADMISSIBILITY gate: DECISION

date: 2026-07-18 | target: proposed S1-S4 injection-recovery gate for `coupling_confirm_v4.py`
lenses: 4 (tautology/degeneracy, exchangeability, verdict-mapping/one-sidedness, power-feasibility)
findings: 35 | fatals raised: 12 | fatals surviving independent refutation: 6

---

## 1. VERDICT: NO_GO_redesign

S1-S4 is dead. Not a constants patch -- every one of the four clauses carries a surviving fatal:

| clause | surviving fatal | what it is |
|---|---|---|
| S1 pool | E2, F2 | ordered pairs make the pool exactly sign-antisymmetric; the specificity leg's pass is a theorem about the construction, and mirror twins veto COUPLED on 48% of draws for reasons unrelated to the estimator |
| S2 sensitivity | T1, E1, PF-1 | injecting exactly at the decision bar pins per-pseudo-seed detection at p=0.5 for ANY pool; `recovery_rate <= 0.5 < 0.80` |
| S3 specificity | E2, F2 | `false_recovery` analytically capped at 0.0663 (n=5) against a 0.15 ceiling it cannot breach |
| S4 coverage | F1 | the guard-excluded estimator solely decides `NO_CAPABILITY_PRICE__style_downgraded_1p5B` by falling silent, and S1 never touches it |

The decisive fact, verified numerically at this desk: **the gate's output does not depend on the data.**

```
KILLED SPEC (S1 ordered pool, injection = 1.00 x MIN_EFFECT_SLOPE), B=4000:
  per-checkpoint SD 0.000  -> recovery 1.0000   false_recovery 0.0000   <- PERFECT PASS, dead channel
  per-checkpoint SD 0.002  -> recovery 0.4983   false_recovery 0.0000
  per-checkpoint SD 0.010  -> recovery 0.4990   false_recovery 0.0000
  per-checkpoint SD 0.0361 -> recovery 0.4950   false_recovery 0.0000
  per-checkpoint SD 0.0722 -> recovery 0.1205   false_recovery 0.0270
```

Recovery is pinned at 0.500 across an 18x range of realized dispersion, against a floor of 0.80 it can
never reach -- so the gate VOIDs every possible run -- **except** when the channel is frozen, where it
returns a flawless two-sided ADMISSIBLE for a dead estimator. That is attempt-3's F1 reproduced one
level up the stack, which is the exact defect this gate exists to close. A gate that emits a constant
is not an instrument.

Two exact identities cause it, and neither is patchable by moving a number:
1. `_per_seed_slope` is `np.polyfit(...)[0]` (`coupling_confirm_v4.py:355-358`), linear in the response,
   so `slope(d + c*r) = slope(d) + c` exactly. Injecting at the bar reduces `s + bar >= bar` to `s >= 0`.
2. S1's ordered pairs give `d_ji = -d_ij` pointwise, so the pool is `{+/-s_1..+/-s_10}` and
   `#{s>=0} = #{s<0}` exactly.

**Do NOT reinstate the six refuted fatals** (T2, T3, E3, F3, PF-2, PF-3). Load-bearing corrections that
came out of those refutations and are now binding on the replacement:
- E3: the rank-4 point is a structural zero (both arms are bit-identical through step 24), so the real
  per-seed slope is `d8/4`. Do NOT exclude rank 4 (it is the divergence point where dose begins), do NOT
  re-anchor the pool (OLS slope is invariant to a per-pseudo-seed constant -- provable no-op), do NOT
  quantize the injection (the pool slope is already on the estimator's 1/48 lattice; quantizing adds
  distortion). PREREG:346-349 prose must be corrected: 3 fitted points, 1 residual df, the middle point
  has zero leverage, and the disclosed 0.036 slope SE is an upper bound.
- F3/PF-3: an under-dispersed pool fails the sensitivity leg on its own; no
  `estimator_null_understates_noise` string, and no separate false-bounded-null leg (a power floor at
  0.80 IS a type-II ceiling at 0.20, since the branches are mutually exclusive and bounded-null is
  reachable only when `dr["coupled"]` is False, `:574`).
- PF-2: seed-sharing dependence is intrinsic to any pairwise null from 5 seeds. Do not "fix" it by
  requiring index-disjointness -- 3 disjoint pairs need 6 seeds and only 5 exist.

---

## 2. REPLACEMENT SPEC (v5) -- ordered, implementable

Lands in one new function plus five edit sites. Zero GPU. Written so the next panel is scoped to the
delta, not to a rediscovery.

**Structural change from S1-S4:** the sensitivity leg stops being a pass/fail at a fixed injection and
becomes a MEASUREMENT (`ESTIMATOR_MDE80`) that is carried into the verdict string. The specificity leg
stops gating (it cannot fail; see the caps below) and becomes a reported descriptive with its analytic
maximum printed beside it. The second side of the two-sided gate is supplied by a degeneracy leg with
no free threshold -- which is the leg that actually kills "a frozen channel scores a perfect pass".

### R1. new function `estimator_admissibility(curves_by_seed, selected, admissible_seeds, *, battery_key="battery_gen", rng_seed=0)`

Place immediately after `dose_response` (`:450`). Returns a dict; never mutates `dr`.

**R1.a -- pool units.** Use the **10 UNORDERED pairs** `{i,j}, i<j` over `SEEDS`. For each pair, a step
enters the pseudo-seed iff:
- `step > 0`, AND
- `acc_i["erased_rank"] == acc_j["erased_rank"]` at that step (the accumulate arms of BOTH seeds report
  the same rank -- this removes the "whose schedule" ambiguity entirely; no judgment call), AND
- that common rank `r` is in `RANK_SPAN`.

Pseudo-delta `d_ij(step) = fixed_i.gen_aggregate(step) - fixed_j.gen_aggregate(step)` where the
aggregate is read from the same `points` field the verdict rides (`battery_key`-matched: `gen_aggregate`
or `gen_aggregate_guard_excl`, `:192-193`). Both terms are already 4dp-rounded, so this is the identical
arithmetic path as `CBG.paired_delta` (`capability_battery_gen.py:389-398`).

Pseudo-slope via the SAME `_per_seed_slope` call. A pseudo-seed with fewer than 2 distinct surviving
ranks is **dropped from the pool** and counted in `n_pool_dropped`.

**R1.b -- degeneracy leg (gating, no free threshold).** VOID `VOID_COUPLING__estimator_unmeasurable` if
any of:
- `len(pool) < MIN_ESTIMATOR_POOL`, OR
- fewer than 2 distinct pseudo-slope values in the pool, OR
- `zero_slope_atom = fraction of pool slopes exactly 0.0` is `> ZERO_SLOPE_ATOM_MAX`.

The zero atom is the diagnostic that made the killed spec's certificate maximally reassuring exactly
when the measurement was dead. It must be on the receipt whether or not it fires.

**R1.c -- draw rule (frozen, deterministic).** `rng = np.random.default_rng(rng_seed)`. Each pseudo-run:
draw `n_slopes` DISTINCT unordered pairs without replacement, assign each an **independent uniform
random sign** (pair orientation is arbitrary; randomizing it is the honest treatment of an arbitrary
label, and it is the same sign-flip null the harness already uses at `_sign_flip_p`, `:361-377`). Pad
with `(n_admissible - n_slopes)` None-slots. Apply the FROZEN rule verbatim with denominator
`n_admissible`, None counting AGAINST the majority -- byte-for-byte the `:419-425` convention. This
matches the realized run shape (PF-4) and it forecloses panel #6's reduced-denominator defect recurring
inside the gate.

**R1.d -- sensitivity leg (gating, measurement-valued).** For each `k` in `INJECTION_GRID` ascending,
compute `recovery(k) = fraction of ESTIMATOR_PSEUDO_RUNS returning COUPLED` with `d + k*MIN_EFFECT_SLOPE*r`
injected at the delta level. `mde80_full = smallest k with recovery(k) >= ESTIMATOR_RECOVERY_FLOOR`.
Repeat over 5 leave-one-SEED-out replicates (drop seed s, pool = C(4,2) = 6 pairs). Then

```
ESTIMATOR_MDE80 = max(mde80_full, mde80_loo_0 .. mde80_loo_4) * MIN_EFFECT_SLOPE
```

If ANY of the six replicates has no grid point reaching the floor, `ESTIMATOR_MDE80 = None` and the run
VOIDs `VOID_COUPLING__estimator_insensitive`. Taking the max is the conservative choice and removes the
"which replicate do we trust" judgment call (PF-7's honest-uncertainty demand, resolved without a new
constant). Report the full recovery curve, all six replicate values, and the span-scale bound
`ESTIMATOR_MDE80 * (RANK_SPAN[1] - RANK_SPAN[0])`.

**R1.e -- specificity leg (REPORTED, NON-GATING).** `false_recovery_rate` at injection 0, printed beside
`false_recovery_analytic_max` for the realized `n_admissible`:

```
n_admissible = 3 -> 0.1600      n_admissible = 4 -> 0.0787      n_admissible = 5 -> 0.0663
```

(exhaustive 50001-point grid over `q = P(s >= bar) = P(s <= -bar)` against the frozen rule at `:421-424`.)
Prereg must state plainly: **under any sign-symmetric null the frozen COUPLED rule's false-fire rate is
bounded by a theorem, not measured by this pool, so it cannot gate.** Realized values may slightly exceed
the i.i.d. cap (measured 0.0825 at per-checkpoint SD 0.0722) because draws share magnitudes -- disclose
that too. Keeping a leg that cannot fire is what voided `styxx.instrument_admissibility`'s default
specificity leg at `f3e10b8`; this is the disclosure that prevents a repeat.

**R1.f -- mandatory scope string on the certificate**, mirroring `NOT_DEAF_SCOPE` (`:125-128`):

> this certifies ESTIMATOR precision on the realized SEED-noise null only. it does not certify that the
> generation channel responds to a capability change of any size, and it does not certify the estimator
> unbiased for the accumulate-vs-fixed ARM-type contrast (that channel is covered separately by
> trained_fi_dose_graded and the guard-excluded subtractive).

### R2. wire into `compute_verdict`, between `:573` and `:574`

```
    if dr["n_admissible_slopes"] < MIN_ADMISSIBLE_SEEDS:            # :572-573, unchanged
        return "VOID_COUPLING__underpowered", ...
    est = estimator_admissibility(curves_by_seed, selected, admissible, battery_key="battery_gen")
    dr["estimator_admissibility"] = est
    if est["void"]:                                                  # unmeasurable | insensitive
        return est["void"], points, per_seed_meta, gate, dr
    if dr["coupled"]:
        est_ge = estimator_admissibility(curves_by_seed, selected, admissible,
                                         battery_key="battery_gen_guard_excl")
        dr["estimator_admissibility_guard_excl"] = est_ge
        if est_ge["void"]:
            return "VOID_COUPLING__estimator_insensitive__guard_excl", points, per_seed_meta, gate, dr
        if not dr_ge["coupled"]:
            ...  # existing style-downgrade branch, now reachable only behind a certified dr_ge
```

Placement rationale (F8, verified): every return before `:574` is a VOID that asserts nothing --
`:555 no_bite`, `:559 underpowered`, `:562` the four battery-gate VOIDs, `:573 underpowered`. The first
claim-bearing return is `:578`. Inserting here puts all five claim-bearing branches behind the gate.

The raw channel gates **unconditionally**; the guard-excluded channel gates **conditionally inside the
coupled branch**, because that is the only place `dr_ge` decides anything. Gating `dr_ge` when `dr` is
not coupled would VOID runs where its deafness is harmless.

### R3. rename the bounded-null verdict to carry the certified level

```
- "NO_PAIRED_PRICE_ABOVE_MDE__battery_not_deaf_1p5B"
+ "NO_PAIRED_PRICE_ABOVE_ESTIMATOR_MDE80__battery_not_deaf_1p5B"
```

A recovery certificate at `ESTIMATOR_MDE80` licenses "no paired price above `ESTIMATOR_MDE80`", not
"above `MIN_EFFECT_SLOPE`" (T8). The receipt must carry `estimator_mde80`, its span-scale bound, and the
recovery curve immediately beside this verdict. This also retires the re-panel's standing "_MDE is a
misnomer" minor: the token now names a measured MDE.

### R4. add both constants to `_thresholds()` (`:591-603`) and all new strings to `FROZEN_VERDICTS` (`:130-144`)

`checks["dry_verdicts_all_frozen"]` (`:1250-1252`) only tests verdicts a `--dry` case actually produced,
so it passes vacuously if the new strings are never emitted. Fixtures D1, D2, D5 below make it
non-vacuous.

### R5. prereg deltas (all pre-freeze, all outcome-blind)

1. correct PREREG:346-349 per E3 (structural zero at rank 4, `slope = d8/4`, 1 residual df, 0.036 is an
   upper bound; `slope >= 0.0152` realizes as `d8 >= 3/48` at minimum selection).
2. add the estimator-gate section, the decision order placement, and the six replicate rule.
3. **pre-commit that `VOID_COUPLING__estimator_*` is TERMINAL for this prereg** (F5, MAJOR). Any
   successor requires a new frozen prereg and a new pre-freeze panel, and the paper discloses every
   VOIDed run with its receipt path and realized rates. Without this clause a VOID is strictly the
   cheapest outcome for a program whose paper is blocked, and the gate converts a hard question into an
   optional one.
4. state the pre-run power arithmetic verbatim (section 6 below). A design whose power was never
   computed before freezing is panel #6's standing threshold-provenance major, one iteration later.

---

## 3. FROZEN CONSTANTS -- with pre-run provenance

| constant | value | provenance (outcome-blind, derivable today) |
|---|---|---|
| `ESTIMATOR_RECOVERY_FLOOR` | `0.80` | the conventional 80% power level, already the convention used for the AUROC floor arithmetic (PREREG:330-336). It no longer decides pass/fail -- it selects WHICH quantile of the recovery curve is reported as the bound. Moving it up weakens the shipped claim, so the conservative direction is up; 0.80 is the standard and is frozen against the naive 0.50. |
| `INJECTION_GRID` | `(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)` x `MIN_EFFECT_SLOPE` | bottom = half the minimum price of interest (below which the question is moot). Top = 8x bar = 0.1216, a span-scale effect of 0.7296, which sits inside the arithmetic maximum expressible slope `1/(hi-lo) = 1/6 = 0.1667` (10.96x bar). An estimator that cannot recover an effect consuming 73% of the aggregate's full range at 80% power is insensitive by any standard, so the grid top IS the insensitivity threshold and needs no separate constant. |
| `ESTIMATOR_PSEUDO_RUNS` | `4000` | Monte-Carlo resolution only. MC SE of a rate at 0.80 is `sqrt(0.8*0.2/4000) = 0.0063`, an order below the recovery change across one grid step. Must be labelled in the receipt as **Monte-Carlo resampling error only, not estimation uncertainty** -- the estimation uncertainty is the LOO spread (PF-7). |
| `ESTIMATOR_RNG_SEED` | `0` | matches every other seeded call in the harness (`instrument_admissibility(..., seed=0)` at `:330`, `slope_permutation_null(..., seed=0)` at `:429`). Makes the gate reproducible from the receipt. |
| `MIN_ESTIMATOR_POOL` | `6` | `C(4,2) = 6` -- the pool size at the fewest seeds that can still form pairs after the LOO jackknife drops one. Same idiom as `MIN_DERIVE_NULLS` / `MIN_GATE_NULLS` (`:119-120`): a floor below which the statistic is unmeasurable, not a tuned bar. |
| `ZERO_SLOPE_ATOM_MAX` | `0.5` | a strict-majority rule, the harness's own convention everywhere (`:424`, `:425`, `:518`, `MIN_GATE_SEEDS_REPLICATING`). If a majority of cross-seed fixed-arm pairs are identical at every fitted rank, the pool cannot represent the estimator's noise and the certificate is vacuous. Value derived from the convention, not from any curve. |

**Constants deliberately NOT created**, because none can be given clean pre-run provenance and each was
proposed as a gate:
- `ESTIMATOR_FALSE_RECOVERY_MAX` -- **deleted.** Any value above 0.0663 (n=5) cannot bind; any value
  below it was chosen by looking at the attainable range. The leg is reported-only with its analytic
  max instead.
- `K` in "VOID if `MDE80 > K x MIN_EFFECT_SLOPE`" -- **deleted.** The design's own arithmetic puts
  MDE80 at ~3.2x bar, so any `K` is either a rubber stamp (K >= 3.5) or a guaranteed kill (K <= 3), and
  either choice would be made with that number in hand. Replaced by carrying the measured MDE80 into
  the verdict string: the claim is restated at the level the instrument can actually support, which is
  the only gate that cannot be tuned.
- `ESTIMATOR_FALSE_NULL_MAX` -- **not created** (PF-3 refuted): the branches are mutually exclusive and
  bounded-null is reachable only when `dr["coupled"]` is False, so an 0.80 recovery floor already caps
  false-bounded-null at 0.20. Compute and REPORT `false_bounded_null_rate` from the same pseudo-runs so
  the bound is a receipt line rather than an inference; do not gate on it.

---

## 4. THE NULL POOL -- exact construction

The cross-seed fixed-vs-fixed pool **survives in substance and dies in form**. What it measures is
sound: the pool slope is `([fix_i(75) - fix_i(25)] - [fix_j(75) - fix_j(25)]) / 4`, a difference of two
50-step drift contrasts on the identical 1/48 lattice as the real statistic, which is
`([fix_s(75) - acc_s(75)] - [fix_s(25) - acc_s(25)]) / 4` (E3, conceded and verified). The ORDERED
construction is what dies.

```
POOL(battery_key):
  units      = the 10 unordered seed pairs {i,j}, i < j, over SEEDS = [0,1,2,3,4]
  step in    = step > 0
               AND acc_i.erased_rank(step) == acc_j.erased_rank(step)      # common axis, no ambiguity
               AND RANK_SPAN[0] <= that rank <= RANK_SPAN[1]
  delta      = fixed_i.<battery_key>_aggregate(step) - fixed_j.<battery_key>_aggregate(step)
  slope      = _per_seed_slope(deltas, ranks)          # same code path as the verdict
  drop unit  if fewer than 2 distinct surviving ranks  # counted, reported, never silently absorbed
  degenerate if len(pool) < 6, or < 2 distinct slope values, or zero-atom > 0.5  -> VOID unmeasurable

PSEUDO-RUN(injection k):
  draw n_slopes distinct units without replacement          # no mirror twins possible
  independent uniform sign per drawn unit                   # orientation is an arbitrary label
  inject   s -> s + k * MIN_EFFECT_SLOPE                    # at the delta level; do NOT re-quantize
  pad with (n_admissible - n_slopes) None slots             # realized shape, PF-4
  score    the frozen rule at :421-424, denominator n_admissible, None counts against
```

Verified behaviour of the replacement (B=4000, ranks {4,6,8}, `n_slopes = n_admissible = 5`):

```
per-ck SD  poolSD   FR(inj 0)  rec@1x  rec@2x  rec@3x   MDE80/bar
0.0000     0.0000     0.0000   1.0000  1.0000  1.0000   1.0  <- caught by the degeneracy leg -> VOID
0.0020     0.0009     0.0000   0.4930  1.0000  1.0000   1.5
0.0100     0.0039     0.0000   0.4950  1.0000  1.0000   1.5
0.0361     0.0105     0.0000   0.5058  1.0000  1.0000   2.0
0.0722     0.0448     0.0825   0.1470  0.1540  0.5265   4.0  <- the design's own nominal SE
0.1200     0.0649     0.0318   0.0757  0.1335  0.2810   8.0
0.2500     0.0998     0.0297   0.0757  0.1030  0.1515   None -> VOID insensitive
```

MDE80 is monotone in realized dispersion, failure-reachable, and lands at 4.0x bar at the design's own
per-checkpoint SE of 0.0722 -- consistent with the closed-form 3.19x. **The gate reads the data.** That
is the single property S1-S4 did not have.

**Rejected pool alternatives, with reasons (do not revisit):**
- Residual-resample of the real paired deltas (F2's proposed fix): the fit is 3 points with 2 params, so
  1 residual df per seed / 5 df total, against 10 pool units. The trade is not favourable and E3's
  structural zero makes the residual set thinner still.
- Ordered pairs with any draw restriction: cannot remove the exact `+/-s` twin structure without
  collapsing to the unordered set anyway.
- Index-disjoint pseudo-runs: 3 disjoint pairs need 6 seeds; 5 exist. Impossible.

---

## 5. FROZEN_VERDICTS additions and --dry fixtures

### strings to add to `FROZEN_VERDICTS` (`:130-144`)

```
"VOID_COUPLING__estimator_unmeasurable",
"VOID_COUPLING__estimator_insensitive",
"VOID_COUPLING__estimator_insensitive__guard_excl",
```
and the rename `NO_PAIRED_PRICE_ABOVE_MDE__battery_not_deaf_1p5B` ->
`NO_PAIRED_PRICE_ABOVE_ESTIMATOR_MDE80__battery_not_deaf_1p5B`. None of the four contains a break-claim
substring, so `checks["no_break_claim_string_exists"]` (`:1246-1249`) still passes.

### IMPLEMENTATION CONSEQUENCE the implementer will hit first

**Every existing `--dry` fixture will VOID under the new gate.** `fixed_arm()` (`:965-967`) applies only
the index-keyed jitter `j(i)`, so all five seeds have byte-identical fixed arms, every pseudo-delta is
exactly 0, and the pool is exactly degenerate. This is not a bug in the gate -- it is the gate working
on fixtures that were never built to carry seed-level noise. The fix is to give `fixed_arm` a per-seed
noise term (a deterministic seed-keyed offset, not an RNG) and thread the seed into every
`seed_*` constructor. **No dry-mode bypass.** A gate with an exemption is not a gate.

### fixtures that must be added and must be load-bearing

| id | constructs | must produce | catches if it regresses |
|---|---|---|---|
| **D1** `estimator_pool_degenerate` | all 5 fixed arms byte-identical; accumulate arms flat | `VOID_COUPLING__estimator_unmeasurable` | the killed spec's signature: a frozen channel scoring a perfect two-sided pass and shipping a bounded null carrying "recovery 1.00" |
| **D2** `estimator_insensitive` | fixed arms with per-seed SD ~3x the design SE; accumulate arms flat | `VOID_COUPLING__estimator_insensitive` | a noisy estimator's silence shipping the bounded null -- the branch the whole gate exists to earn |
| **D3** `bounded_null_carries_mde80` | fixed arms with moderate per-seed SD; accumulate arms flat | `NO_PAIRED_PRICE_ABOVE_ESTIMATOR_MDE80__battery_not_deaf_1p5B`, plus assertions: `est["mde80"]` is not None, `est["mde80"] > MIN_EFFECT_SLOPE`, and `est["span_bound"] == est["mde80"] * 6` | the claim silently reverting to the 0.0152 bar -- T8, the arc's own verdict-mapping rung |
| **D4** `estimator_gate_precedes_all_findings` | each of `coupled_dose_slope`, `echo_subtractive_downgrade`, `fi_dose_downgrade`, `bounded_null`, `partial_sign_split` re-run with the D1 degenerate fixed arms substituted | all five return `VOID_COUPLING__estimator_unmeasurable` | someone moving the insertion point behind a finding branch; pins F8's placement as a behavioural invariant rather than a line number |
| **D5** `guard_excl_estimator_gated` | raw channel COUPLED with a healthy raw pool; `gen_aggregate_guard_excl` identical across seeds AND `dr_ge` not coupled | `VOID_COUPLING__estimator_insensitive__guard_excl`, NOT `NO_CAPABILITY_PRICE__style_downgraded_1p5B` | F1 -- a deaf second estimator converting a genuine COUPLED into "the price was style, not capability" |
| **D6** `estimator_stalled_pseudo_seeds` | 6 of 10 pairs have no common in-span rank (divergent accumulate schedules), so they drop; `n_slopes < n_admissible` on the real side too | pool size and drop count on the receipt; assert `recovery(k)` computed with padding is <= the same run computed without padding | panel #6's reduced-denominator FATAL recurring INSIDE the gate meant to prevent it (T6, F7) |
| **D7** `estimator_reads_the_data` | pure arithmetic, no verdict: two synthetic pools at SD 0.2x bar and SD 3x bar | assert their MDE80 differ by at least two grid steps | the exact defect that killed this spec -- a recovery rate that is a constant. This is the anti-T1 fixture and it is the most important one in the list |
| **D8** `false_recovery_is_reported_not_gating` | any passing fixture; then perturb `est["false_recovery_rate"]` to 0.99 in-place and recompute the branch | verdict unchanged; receipt carries both `false_recovery_rate` and `false_recovery_analytic_max` | someone silently re-gating on the tautological leg, which is how `styxx.instrument_admissibility` got voided at `f3e10b8` |

D1, D2 and D5 also make `checks["dry_verdicts_all_frozen"]` (`:1250-1252`) non-vacuous for the three new
strings.

---

## 6. RESIDUAL RISK the operator must accept to freeze

**R-1 (the big one). The design's power at the price it cares about is 0.238, and more seeds makes it
worse.** Computed today from the design's own arithmetic (per-seed slope SE = `0.1021/sqrt(8)` = 0.0361,
frozen rule at `:421-424`), with a TRUE effect of exactly `MIN_EFFECT_SLOPE`:

```
n_admissible = 5:  COUPLED 0.2377 | bounded_null 0.1633 | PARTIAL 0.5991
n_admissible = 4:  COUPLED 0.2126 | bounded_null 0.0838 | PARTIAL 0.7036
n_admissible = 3:  COUPLED 0.3501 | bounded_null 0.2162 | PARTIAL 0.4337
analytic MDE80 at n=5: 0.04845 = 3.19 x bar (span-scale bound 0.2907 vs the 0.0909 price of interest)
```

The single most likely outcome of the scored run is PARTIAL (~0.60) even if a real bar-sized price
exists. Adding seeds does NOT help: at a bar-sized true effect `P(some seed lands at or below -bar)` is
0.1999 per seed, so the rule's `len(neg_above) == 0` clause decays as `0.8^n` and power FALLS with n.
The only lever that raises power is a smaller per-checkpoint SE -- more items per sub-task, i.e. a
bigger battery and more GPU. **Widening `RANK_SPAN` is not a legitimate lever**: ranks 10-24 are the
recovery region, and fitting into it mixes read-subspace recovery into the price slope, which is the
first panel's power fatal in reverse.

Freezing means accepting, in advance, that the run most likely ships a PARTIAL, and that when it does
ship the bounded null the claim will read "no paired price above ~0.29 over the span" -- a question
roughly 3x weaker than the one the paper asks. That claim is TRUE and EARNED. It is also not the paper's
headline. That trade is the operator's call, and it must be made before the GPU starts, not after.

**R-2. The pool is a between-seed contrast standing in for a within-seed cross-arm contrast.** The real
statistic differences two arms that share base weights, LoRA init and the entire batch stream and are
byte-identical through step 24 (`:623`, `:625`, `:634`, `:683`), so `Cov(A,F) > 0` and the real variance
is at most the pool's. The pool therefore OVERSTATES the estimator's noise, MDE80 is an upper bound, and
the shipped claim is weaker than the truth. Conservative direction, unquantified magnitude. Must be a
named residual in the prereg, not an assumption.

**R-3. Ten pool units come from five trajectories; each seed appears in four pairs.** Effective sample
size is nearer 4 than 10. The LOO-max rule and the reported LOO spread are the disclosure; they do not
make the units independent.

**R-4. This is a SEED-noise certificate, not an ARM-type certificate.** The pool contains no
accumulate-arm variance and no accumulate-vs-fixed bias. Arm-type nuisance stays covered by
`trained_fi_dose_graded` (`:491-524`) and the guard-excluded subtractive (`:564-569`) -- separate
machinery, separate claim, and the scope string (R1.f) must stop any reader collapsing the two.

**R-5. Specificity here is a theorem, not a measurement.** The 0.0663 / 0.0787 / 0.1600 caps hold under
sign-symmetry, which the construction imposes. If the frozen COUPLED rule ever changes, those caps must
be recomputed or the reported number becomes misleading.

**R-6. The estimator gate does not certify that the generation channel responds to a capability change
of any size.** It certifies estimator precision on the realized null. The genuinely independent paired
positive control remains the program's named open methodological problem (PREREG:95-97), and no
zero-GPU construction closes it -- item-level injection does not either, since it re-enters downstream
of generation on the same lattice.

**R-7. VOID must be terminal or the gate is optional.** With the paper blocked and PARTIAL standing, a
VOID costs the program nothing while a re-tune costs nothing either. R5.3 is not optional
housekeeping; without it this gate is the first one in the harness whose failure is fixable by editing
the estimator rather than re-running the model, and that is the garden of forking paths the entire arc
exists to prevent.

---

## 7. what the next panel gets

Scope the re-panel to the delta only: R1.a-R1.f, R2's insertion, R3's rename, the six frozen constants
and their provenance, and D1-D8. The pool construction, the injection-as-measurement move, the deleted
specificity gate, and the two-channel coverage are settled here. If a lens wants to reopen one of them
it must first defeat the arithmetic in sections 3 and 4 of this document.
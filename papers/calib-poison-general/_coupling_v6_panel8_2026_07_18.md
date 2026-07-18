# Pre-freeze panel #8 -- coupling v6 power delta (R5.6)

Date: 2026-07-18. Target: `PREREG_B2_coupling_twosided_2026_07_16.md` R5.6, the proposed v6 delta
(widen the selection 48 -> 112 items; halve MIN_EFFECT_SLOPE 0.0152 -> 0.0076), claimed jointly
necessary. Four adversarial lenses; every FATAL put to an independent refuter defaulting to refute.
36 findings, 11 FATALs filed, 4 refuted, 7 surviving.

---

## 1. VERDICT

**NO_GO_redesign for R5.6 as written.**

The delta packages a no-op with a threshold move, and the threshold move is fatal on its own
arithmetic. Neither half survives in the form proposed. The redesign is small and is fully specified
in section 3; it does not touch the frozen v4 decision rule, and on the panel's own recomputation it
lands STRICTLY BETTER than the delta on both legs.

The single decisive number. Under the corrected noise (`_estimator_noise_anchor.json`, the projection
committed at `ce2e9e3`) and the structural-zero-consistent signal mapping, at 112 items:

| bar | sensitivity at the measured price | false positive | estimator gate |
|---|---|---|---|
| 0.0152 (unchanged) | **0.865** | **0.009** | PASS, MDE80 = 5x bar |
| 0.0076 (the proposal) | 0.952 | 0.055 | **VOID_COUPLING__estimator_insensitive (TERMINAL)** |

The unchanged bar clears the 0.80 convention at a false-positive rate 6x lower than the delta's, and
it is the only one of the two that survives the gate standing in front of all five claim-bearing
branches. The goalpost move buys nothing it does not also destroy.

---

## 2. RULING ON EACH DELTA, SEPARATELY

### Delta #1 -- widen the selection to every qualifying disjoint sub-task

**NOT ADMISSIBLE AS A CHANGE. ADMISSIBLE AS A DISCLOSURE. Its honest successor -- EXPAND the battery
-- is ADMISSIBLE and is the only real precision lever this design has.**

`capability_battery_gen.py:369-374` `select_disjoint` already returns EVERY sub-task clearing
`DISJOINT_FLOOR_CLEAN`; `need=MIN_DISJOINT` sets an ok flag and never caps. `measure_all_gen`
(`capability_battery_gen.py:337`) already decodes all 7 disjoint sub-tasks at every checkpoint of
every arm. There is nothing to select and no decode cost to pay. R5.6's premise -- "MIN_DISJOINT = 3
(48 items) is a floor being used as a target" -- is false about the harness. Two lenses found this
independently; one refuter confirmed it while refuting the fatal built on it.

Consequences the prereg must absorb:

- 112 items is not a design choice. It is what happens iff the CLEAN base clears 0.90 on all 7. That
  has never been measured -- `coupling_v4_gen_selected.json` does not exist on disk; `--calibrate`
  has never run. The T/F precedent is 3 of 8 clearing the same floor.
- Every operating point in the R5.6 table is conditioned on a number nobody controls.
- Reaching 112 by any other route means lowering `DISJOINT_FLOOR_CLEAN`, which the prereg forbids and
  which would make `CLEAN_SUBTASK_FLOOR = 0.70` the binding selection rule. Refused.

What IS admissible and IS the lever: the corrected anchor's own conclusion -- the battery must be
EXPANDED (7 x 32 = 224 items), which is authoring work, not GPU work.

### Delta #2 -- halve MIN_EFFECT_SLOPE to 0.0076

**REFUSED. Three independent grounds, any one sufficient.**

**(a) The rationale is an artifact of a wrong signal mapping.** R5.4/R5.5/R5.6's entire justification
is the "symmetry ceiling": the bar sits AT the effect, so a paying seed is a coin flip and power
caps near 0.5. That is false under the estimator the design actually uses. Both power scripts
evaluate sensitivity at `true_slope = MIN_EFFECT_SLOPE = 0.0152`
(`estimator_power_prefreeze.py:93,108,158`; `estimator_noise_anchor.py:330`) -- i.e. the price
0.0909 divided by the NOMINAL span of 6. But the same session CONFIRMED 5/5 that the rank-4 paired
delta is a structural zero, so the fitted slope reduces exactly to `d8/4`. A seed reproducing the
only price this program has ever measured (`RESULT_B2_coupling_dose_PARTIAL_2026_07_14.md:26`,
knowledge_drop 0.0909 at r_star = 8) has expected fitted slope 0.0909/4 = **0.0227 = 1.49x the bar**,
not 1.00x. The bar is already at 0.669 of a paying seed's own statistic. The ceiling does not bind,
and the scripts have been reporting a field named `sensitivity_at_price` that is in fact sensitivity
at the bar. Recomputed at the corrected true slope and the CORRECTED noise: 48 items 0.726/FPR 0.034,
112 items 0.865/0.009, 224 items 0.929/0.001. The problem the delta exists to solve is substantially
an arithmetic error in the design's own power model.

**(b) It fires the TERMINAL estimator VOID.** `INJECTION_GRID` (`coupling_confirm_v4.py:137`) is
denominated in MULTIPLES of `MIN_EFFECT_SLOPE` and injected as `k * MIN_EFFECT_SLOPE` (`:581`),
while the pool's dispersion is an absolute data property. Halving the bar halves every absolute
injection AND halves the `-bar` veto. Driving the harness's own `_estimator_mde80` plus the full
5-LOO loop on the realized pool rescaled by the corrected projection:

| items | bar 0.0152 | bar 0.0076 |
|---|---|---|
| 48 | PASS, MDE80 8.0x | VOID (top recovery 0.407) |
| 112 | PASS, MDE80 5.0x | VOID (top recovery 0.748) |
| 224 | PASS, MDE80 4.0x | PASS, MDE80 8.0x (grid top) |

R5.3 makes that VOID TERMINAL for the prereg. At every item count the design can actually reach
before an authoring expansion, the delta converts the most likely outcome from PARTIAL into the end
of attempt 4. Three lenses reached this independently; two of the three fatals survived refutation
and the third was downgraded to MAJOR on severity only, not on the arithmetic. The delta's own table
could not see it: it was produced by `estimator_power_prefreeze.rates()`, a standalone Gaussian model
of the dose rule that never touches `estimator_admissibility()`.

**(c) It silently rescales three guards.** `INJECTION_GRID` (above); `FI_DOSE_FRACTION` (`:114`,
`:728`, `bar = FI_DOSE_FRACTION * MIN_EFFECT_SLOPE`) halves the format-demotion trigger to 0.0038,
raising false COUPLED -> PARTIAL demotion of a two-point folded statistic from ~0.002 to ~0.08-0.16;
and the prereg's load-bearing FORMAT_INVARIANCE disclosure inverts -- 0.0722 goes from 0.79x to 1.58x
the smallest span-scale effect COUPLED would assert, i.e. a clean-base format channel larger than the
entire effect. None of the three appears in R5.6.

**What killing delta #2 leaves the design able to claim:** everything it could claim before, at
better power than the delta advertised. COUPLED continues to assert a dose-graded price at or above
the WHOLE minimum price of interest, not half of it. Nothing is surrendered.

### Where the panel does NOT go

The delta's stated INTENT -- buy precision rather than move a bar -- is correct and is upheld. The
corrected anchor already endorses it and names the price (224 items). What is refused is the
execution: one half was already done, the other half was the bar move the intent disclaims.

---

## 3. THE REDESIGN -- ordered, implementable

BLOCKING items must land before any freeze call. Every one is zero-GPU and outcome-blind.

**B1. Correct the signal mapping in both power scripts, and rename the field.**
- `estimator_power_prefreeze.py:93, 108, 158`: `rates(MIN_EFFECT_SLOPE, se, ...)` ->
  `rates(SLOPE_AT_PRICE, se, ...)`, value **0.022725**, defined as
  `SLOPE_AT_PRICE = PRICE_OF_INTEREST / 4.0` with the /4 derived from the structural zero in one
  place, not typed as a literal.
- `estimator_noise_anchor.py:330`: `power(bar, se, MIN_EFFECT_SLOPE)` -> `power(bar, se,
  SLOPE_AT_PRICE)`.
- Both files: rename `sensitivity_at_price` -> `sensitivity_at_slope_at_price`, and ADD
  `sensitivity_at_the_bar` carrying the old number, so the two can never again be the same field.
- Re-run both; commit the .py and the .json in one commit (the standard this arc set at `ce2e9e3`).

**B2. Delete delta #2 from R5.6.** `MIN_EFFECT_SLOPE` stays 0.0152. Replace the symmetry-ceiling
paragraphs in R5.4 (bullet 1), R5.5 (bullet 4) and R5.6 (item 2) with the B1 arithmetic and state
plainly that the ceiling claim was an artifact of setting the true effect equal to the bar.

**B3. Restate delta #1 as a disclosure and name the real lever.** R5.6 item 1 becomes: the harness
already selects every qualifying sub-task; the realized item count is data-determined at
`--calibrate`; 48 is a floor, not a plan. Add the expansion (7 x 32 = 224 items) as the named
precision lever with its authoring cost stated. Delete the stale line at R5.6 item 1 quoting
"0.0153 -> 0.0100" -- it contradicts the corrected table seven lines below it.

**B4. Re-denominate `INJECTION_GRID` in ABSOLUTE slope units.** A NO-OP today, and it is what
permanently closes the class that produced 4 of the 7 surviving fatals. `coupling_confirm_v4.py:137`
becomes the absolute tuple
`(0.0076, 0.0152, 0.0228, 0.0304, 0.0380, 0.0456, 0.0608, 0.0760, 0.0912, 0.1216)`; `:581` injects
`+ k`; `_estimator_mde80` reports the absolute value and its ratio to the current bar. Restate the
grid-top derivation at `:139-144` in absolute terms (top 0.1216, span-scale 0.7296, 73% of the
aggregate's range) so a future bar change cannot falsify it.

**B5. Pin the FI demotion bar to the PRICE, not the bar.** `coupling_confirm_v4.py:114/:728`: replace
`bar = FI_DOSE_FRACTION * MIN_EFFECT_SLOPE` with
`FI_DOSE_BAR = 0.5 * PRICE_OF_INTEREST / (RANK_SPAN[1] - RANK_SPAN[0])` = **0.007575** (numerically
0.0076, unchanged today), and restate its provenance as "half the price-of-interest slope", which is
what it always was.

**B6. FORMAT_INVARIANCE_MAX -- fix the label, keep the value 0.0722.** The panel rules that it does
NOT break under widening: it is an absolute tolerance on an absolute channel, and a more precise FI
estimate makes it strictly better at catching what it exists to catch. What is false is the
derivation comment at `:123` ("one per-checkpoint aggregate SE at minimum selection") and the same
sentence at PREREG:249-254. At 112 items one worst-case SE is 0.0472, so 0.0722 is 1.53 SE; at the
realized clean base (which scores 1.0 in the anchor) the SE is ~0 and "one SE" was always a fiction.
Restate as a worst-case, deliberately permissive, explicitly NON-load-bearing bound, naming
`trained_fi_dose_graded` as the load-bearing leg. `:1062`'s divisor is the PRICE and stays correct --
replace the literal `0.0909` with the named `PRICE_OF_INTEREST` constant.

**B7. Receipt surface.** Add to `_thresholds()` (`coupling_confirm_v4.py:840-858`):
`n_items_selected` (16 x len(selected)), `slope_lattice` (1/(4 x n_items)), `fi_dose_bar`,
`price_of_interest`, `injection_grid_absolute`. Compute `trained_fi_dose_graded` unconditionally
(`:829` currently sits inside `if dr['coupled']`), gating only the DOWNGRADE on COUPLED, so the FI
bar and slopes appear on every receipt regardless of verdict.

HARDENING -- strongly recommended before freeze, each cheap:

**H1. Rebuild the estimator pool as a WITHIN-seed contrast. Highest-leverage change in this review.**
The shipped pool is a between-seed contrast standing in for a within-seed one and overstates the real
estimator noise by a MEASURED 3.83x (`_estimator_noise_anchor.json within_seed_paired_noise`). That
inflation is what sets the bounded null's claim level. Panel-recomputed MDE80 span bounds at the
unchanged bar:

| items | pool as shipped | pool rebuilt within-seed |
|---|---|---|
| 48 | 0.730 | 0.182 |
| 112 | 0.456 | 0.182 |
| 224 | 0.365 | 0.137 |

Price of interest is 0.0909. As shipped, the ANTICIPATED verdict licenses a bound 4-8x weaker than
the paper's question at every reachable item count, and no amount of battery expansion fixes it. A
within-seed pool gets to 2x. Keep the cross-seed pool beside it as a disclosed conservative tier;
`MIN_ESTIMATOR_POOL` drops to 5 with that stated.

**H2. Soften R5.3's terminality, or land H1. Not both optional.** A gate whose null the program has
documented as 3.83x too wide currently holds an unappealable kill switch over the whole attempt.
Either H1 lands, or R5.3 becomes: VOID is terminal for the COUPLED branch only, and the run may still
ship a bounded null carrying the pool-based MDE80 with the overstatement factor printed on the
certificate.

**H3. Rename the statistic.** `slope == d8/4` exactly; rank 6 has zero leverage; the dose axis
contributes ONE free point. Drop `dose_slope` from `COUPLED__dose_slope_price_measured_1p5B` and
report `d8` directly beside it. This SUBTRACTS a claim the design cannot support, which is the only
form of post-hoc analysis this program licenses.

**H4. Item-level pairing.** Both arms answer the SAME items at the SAME step, but
`measure_gen_subtask` (`capability_battery_gen.py:303-323`) discards the per-item correctness vector,
so `paired_delta` differences two effectively independent binomials. A per-item bitvector plus a
McNemar discordance statistic is 1.7x tighter at 6% discordance and 2.4x at 3% -- the latter beats the
entire 224-item expansion, at zero GPU, from a schema change. Measure the realized discordance before
pricing it.

**H5. `--dry` fixtures.** The suite is blind to a bar change: 52/52 with `MIN_EFFECT_SLOPE` halved,
zero checks changing state. Add (i) a fixture with per-seed slopes in (0.0076, 0.0152), (ii) a fixture
whose estimator pool SD is set from the anchor's realized scale rather than NZ=0.012, and (iii) an
assertion that `INJECTION_GRID[-1]` in absolute units is invariant to `MIN_EFFECT_SLOPE`. Do not cite
"--dry 52/52" in support of any threshold change again.

---

## 4. IS THE NOISE ANCHOR LOAD-BEARING ENOUGH?

**For the widening/expansion direction: YES. For a threshold move: NO -- and the threshold move is
dead anyway.**

What the anchor genuinely establishes, and it is more than it claimed: the structural zero is
CONFIRMED 5/5 on real data; the realized within-seed paired SD is measured, not bounded; the
cross-seed pool's 3.83x overstatement is measured; and the accuracy correction at `ce2e9e3` moved the
projection AGAINST the program's own headline (0.880 -> 0.755), which is the licensed direction.

What it cannot carry:

- The SD is the mean of two 5-value SDs. Its chi-square CI at df=8 is [0.0685, 0.1942]. The upper end
  EXCEEDS the maximum possible pure-sampling paired SD at 24 items (0.1443) -- at that bound a
  trajectory floor is mandatory and no item count helps. Dropping one datum (the 0.25 at seed 0
  rank 6) moves the headline from 1.67x bar to 0.79x bar.
- The trajectory component (0.0198) is a difference of similar numbers and the anchor says so: NOT
  identified, band [0, 0.03]. At the pessimistic end the infinite-item slope-SE floor is 0.0075.
- Channel: it is measured on the T/F battery. The gen channel's within-sub-task item correlation,
  repetition guard and echo guard produce exactly the checkpoint-level common mode the binomial
  projection has no term for.
- The "no trajectory floor detectable" inference used a residual that could not have detected a floor
  below 1.42x the point estimate, and arm-correlation is perfectly confounded with a floor because
  both arms score the same fixed items and are bit-identical through step 24. State it as "no floor
  DETECTED under a bound that could not have detected one", not as evidence of absence.

**Required before freeze.**

1. `--calibrate`, MANDATORY. Base-only and treatment-blind, so it cannot leak an arm outcome. It
   resolves the single input every power number is conditional on: how many of the 7 gen sub-tasks
   the clean base clears at 0.90. One bf16 load of the 1.5B plus roughly 250-400 greedy 8-token
   decodes (7 x 16 items x 2 wrappers, plus the T/F, private-13 and knowledge readouts).
   **Estimated 0.1 GPU-hour.** Freeze the bar rule BEFORE reading the count, and say so in the
   document.
2. `--smoke`, one seed, short schedule, to measure two things the anchor cannot supply: the per-item
   arm-to-arm DISCORDANCE (H4's input) and per-sub-task dispersion plus guard-fire rates on the gen
   channel. **Estimated 0.5-1.0 GPU-hour.**

Total pre-freeze GPU cost: **under 1.5 GPU-hours.** The scored overnight run is unchanged.

Publish the four-column operating characteristic (sensitivity at the price-consistent slope, FPR,
bounded-null-at-null, PARTIAL-at-null) at the realized item count and at BOTH CI endpoints of the SD.
If the conclusion does not survive both endpoints, the measurement cannot license the design.

---

## 5. WHAT THE PROGRAM MAY CLAIM

Under the redesign -- unchanged bar, realized item count, unchanged rule -- a COUPLED verdict may
claim exactly this and no more:

> In Qwen2.5-1.5B-Instruct, under this specific accumulating rank-erasure attacker against a paired
> rank-2-pinned control, a strict majority of dose-admissible seeds showed a generation-capability
> price whose fitted slope over ranks 2 to 8 was at or above 0.0152 per rank with no seed at or below
> -0.0152 -- a per-seed statistic that reduces exactly to one checkpoint comparison at rank 8 divided
> by four, on 5 seeds with no p-value, at a battery size and an ESTIMATOR_MDE80 the certificate
> states on its face.

---

## 6. RESIDUAL RISKS THE OPERATOR MUST ACCEPT TO FREEZE

1. **The price is n=1 and cross-channel.** 0.0909 is one seed's knowledge drop on the T/F channel at
   one crossing rank. B1's corrected true slope inherits that, and `channel_gain_receipt` is
   REPORTED, NON-GATING -- the T/F-to-gen transfer coefficient is documented, never verified before
   the run. If the gen channel's gain is materially below 1, the 0.865 is optimistic.
2. **The item count is unknown until `--calibrate` runs.** At 3 survivors the design sits at
   sensitivity 0.726 and the estimator gate passes at the GRID TOP -- one notch from the terminal
   VOID. That is the realistic downside case, and the T/F precedent points at it.
3. **The bounded null -- the anticipated verdict -- remains 2-8x weaker than the paper's question**
   depending on whether H1 lands. That is a property of the pool, not of the bar, and no battery size
   fixes it.
4. **The trajectory floor is not identified.** At the pessimistic end of the band the infinite-item
   slope-SE floor is 0.0075 and more items stop helping well before 224.
5. **The dose axis is one free point.** Rank 6 cannot move any verdict and no cadence or span change
   can fix it. If H3 is declined, the receipt keeps calling a two-point contrast a dose response.
6. **`--dry` cannot detect a threshold change.** Until H5 lands, the regression suite is not evidence
   about any constant in this review.

---

## APPENDIX -- disposition

SURVIVING FATALS, all adopted. goalpost F1 (signal mapping) -> B1/B2. goalpost F2, coupled-constants
CC-1, noise-anchor F1, does-it-answer F1 -- four independent derivations of the same terminal-VOID
mechanism -> B2/B4. coupled-constants CC-2 and does-it-answer F3 (delta #1 is a no-op) -> B3.

REFUTED, NOT REINSTATED. CC-3 (heterogeneity raises the aggregate SE): refuted, the sign is backwards
-- concavity means heterogeneity at fixed mean strictly REDUCES aggregate variance. noise-anchor F2:
downgraded to MAJOR; the no-op is real, the numeric case was computed against the withdrawn
projection. does-it-answer F2: the correction it demands was already made, re-run and committed at
`ce2e9e3` before the panel filed. does-it-answer F4 (FORMAT_INVARIANCE_MAX): an absolute tolerance
does not loosen when the estimate concentrates; survives only as B6, a labelling fix.

Eight pre-freeze kills banked. The seventh and eighth were both threshold moves that would have been
invisible after the run.

---

Receipt written to `C:\Users\heyzo\clawd\styxx\papers\calib-poison-general\_coupling_v6_panel8_2026_07_18.md` (ASCII-verified, 0 non-ASCII bytes). Verification scripts used for the tables: `C:\Users\heyzo\AppData\Local\Temp\claude\C--Users-heyzo--styxx\a013ee8f-92b5-4681-97fb-e883170a9a8f\scratchpad\panel8_check.py` and `panel8_check2.py`.
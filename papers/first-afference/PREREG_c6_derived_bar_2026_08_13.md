# PREREG — C6: the first bar in this program derived from measured power instead of asserted

Fathom Lab · 2026-08-13 · frozen before any C6 statistic touches the real subject timeseries.
Power basis: `c6_power.json` (detection curve), `c6_basis_v2.json` (null at resolution, knee,
guard-binding). Generators: `power_c6.py`, `c6_basis_v2.py`. Both synthetic-only.

## Why this document exists

C5 closed with: *"The successor is a power calculation, not another exam. Any future gate on this
data must state the effective sample size first and derive its bar from that. A prereg that fixes
a bar before computing achievable power is not preregistration, it is decoration."*

E1 then found the effective sample size is **not estimable** at this series length — five
estimators, all biased upward, none within 20% of analytic truth. That killed the literal
instruction. The alternative E1 named was a statistic needing no effective-n estimate, and that is
the route here: a Monte-Carlo detection rate under **the exact licensing rule C6 will use**, on
synthetic streams with known planted coupling, at the two nuisance parameters already committed in
FINDING_c5 (lag-1 range 0.4967–0.8054, n_t=300, n_sub=7, n_col=2).

The bar below is **read off that curve by a stated rule**. It is the first bar in this program
whose value I did not choose — and the rule chose a different value than my own draft did, which
is the point.

## The selection rule, stated before the number

> **Choose the smallest cohort bar `k` whose observed false-license rate is ≤ 0.05 under BOTH
> adversarial generators, at every amplitude tested.**

Smallest, because power is the scarce resource and C5 died of an unclearable bar. Both
generators, because a bar that only survives the easy null is the C4 error. The rule is fixed
here; the table it is applied to follows.

| bar | null (c=0, 400 cohorts ×3 rho) | block-confound (60 cohorts, worst case over c) | detection @ c=0.40 | @ c=0.44 |
|---|---|---|---|---|
| ≥3/7 | 0.000 | **0.100** ❌ | 0.983 | 1.000 |
| ≥4/7 | 0.000 | **0.067** ❌ | 0.967 | 1.000 |
| **≥5/7** | **0.000** ✅ | **0.017** ✅ | **0.950** | **1.000** |
| ≥6/7 | 0.000 ✅ | 0.000 ✅ | 0.667 | 0.967 |
| ≥7/7 | 0.000 ✅ | 0.000 ✅ | 0.333 | 0.683 |

**The rule selects k=5.** My first draft of this document asserted k=6 from the smoke run before
the full run existed. k=6 costs 28 points of detection at c=0.40 (0.667 vs 0.950) and buys a
reduction in confound false-licensing from 0.017 to 0.000 that 60 replicates cannot resolve
(Clopper–Pearson 95% upper bound is 0.0487 for both). **Paying real power for an unmeasurable
safety margin is the same instinct that produced C5's impossible bar, wearing a cautious face.**
The rule is followed; the bar is 5.

## What the power basis says

Detection is a step in planted amplitude `c`, shifting right as autocorrelation rises. At the
worst committed autocorrelation (rho=0.8054), the k=5 bar clears at 0.200 (c=0.32) → 0.567
(c=0.36) → **0.950 (c=0.40)** → 1.000 (c=0.44). Null cohorts license 0.0057–0.0125 per subject
against nominal alpha=0.01 across 8400 subject decisions — calibrated, not conservative by luck.

**Three properties of the instrument that were not known before today:**

1. **The floor is a coupling amplitude, not a sample size.** Below c≈0.36 at high autocorrelation
   no bar is reliably clearable. Any gate demanding ≥0.80 of subjects at c≈0.30 is C5's error a
   fourth time.

2. **The two guards are not redundant, and this was tested rather than assumed.** Licensing is
   `surrogate_p ≤ 0.01 AND matched_p ≤ 0.01`. On the shared-signal generator the matched leg
   bound **2 times in 8400 null decisions and 0 times in 2100 knee decisions** — the conjunction
   was the surrogate test wearing two names, which is the conjunctive form of the `meta_audit_v1`
   defect (*a disjunction with an always-true term is always true*). That alone does not make the
   leg decorative: the generator never planted what the leg exists to catch. A third generator
   was added — shared per-block mean offsets, no shared within-block dynamics, which phase
   randomisation preserves and the matched permutation destroys — and there `matched_binds` fires
   **105/420 decisions at c=0.50 and 152/420 at c=0.70** while cohort licensing stays at 0.093.
   **The second guard binds exactly where designed and nowhere else.** Both legs are load-bearing;
   the conjunction stays.

3. **Per-subject licensing is not independent within a cohort.** Count variance runs to 1.99×
   binomial on the confound generator. Cohort bars are therefore read from observed count
   histograms with exact Clopper–Pearson bounds, never from a binomial argument on the mean.

## The frozen bar

```gates
{"gates": {"G1_exceeds_null_ceiling": {"metric": "cohort_licensed_count", "op": ">=", "value": 5,
             "power_basis": "c6_basis_v2.json. Selected by the rule stated above (smallest k with false-license <=0.05 under BOTH the c=0 null and the planted block-confound). Null: 0/400 cohorts reach 5/7 at all three committed rho (CP95 upper 0.0075). Block-confound: worst cell 0.017. Detection at rho=0.8054 is 0.950 at c=0.40 and 1.000 at c=0.44, so the bar is clearable by an achievable effect -- the property C5's G1 lacked",
             "metric_means": "number of the 7 subjects whose leave-one-out statistic clears BOTH surrogate and confound-matched nulls at alpha=0.01 on the real series"},
           "G2_null_calibration_holds": {"metric": "per_subject_license_rate_on_phase_randomised_cohort", "op": "<=", "value": 0.05,
             "power_basis": "the synthetic null licenses 0.0057-0.0125 per subject at alpha=0.01 over 8400 decisions; a real-data phase-randomised cohort must sit in the same regime or the nuisance structure of the real series differs from everything the power basis was computed on, which INVALIDATES G1's bar rather than failing it",
             "metric_means": "per-subject licensing rate when the real cohort is replaced by its own phase-randomised surrogate"},
           "G3_matched_leg_bind_count": {"metric": "matched_leg_binding_count_on_real_cohort", "op": ">=", "value": 0,
             "power_basis": "RECORDED, NOT GATED -- value 0 makes this unfailable by construction and that is deliberate. On the real cohort I do not know the true bind profile and will not fabricate a prediction. The count is logged so the conjunction's load-bearingness on real data is on the record either way",
             "metric_means": "count of subject decisions where surrogate passed and matched vetoed, on the real series"}},
 "outcomes": [{"when": {"G2_null_calibration_holds": false}, "verdict": "REFUSED__power_basis_does_not_transfer_to_this_series"},
              {"when": {"G2_null_calibration_holds": true, "G1_exceeds_null_ceiling": true}, "verdict": "LICENSED__cohort_coupling_above_derived_null_ceiling"},
              {"when": {"G2_null_calibration_holds": true, "G1_exceeds_null_ceiling": false}, "verdict": "NULL__below_derived_ceiling_at_known_power"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## The losing branch, named in advance

`NULL__below_derived_ceiling_at_known_power` is the branch I expect. C5 licensed 2 of 21 pairs; a
leave-one-out design pools six subjects against one and should beat pairwise, but "better than a
design that licensed 9.5%" is not "reaches 5 of 7."

**The difference from C5 is that the null branch now says something.** C5's null was uninformative
because nobody knew what the instrument could detect. This one is bounded: a null here places the
real cohort's shared structure **below c≈0.40 at worst-case autocorrelation** — a stated ceiling
on the effect, not a shrug.

## A defect in this document's own first draft, recorded because it happened

The draft written before the full run quoted the **smoke** basis (8 replicates) as its power
basis, and asserted k=6 from it. The smoke run reported P(≥6/7)=0.375 at c=0.32 and 0.750 at
c=0.40; the full run (60 replicates) gives **0.067 and 0.667**. The c=0.32 figure was wrong by
more than five-fold, and a bar frozen on it would have been derived from noise while carrying the
word "derived." **A power basis computed at a resolution finer than the claim it supports is
still an assertion.** The draft was never sealed and the corrected numbers are the ones above,
but the near-miss belongs on the record: the failure mode this document exists to close nearly
reproduced itself inside the document that closes it.

## Stated limits, in advance

- **The power basis is synthetic and both generators are mine.** `x_i = c·s + sqrt(1−c²)·e_i` is
  additive shared signal in AR(1) noise; real BOLD coupling is not that. P1's lesson applies —
  *an author's own battery encodes the author's misconceptions on both sides* — and a curve on
  the wrong generator gives a confidently wrong bar. G2 is the transfer check precisely because I
  do not trust this leg, and it can refuse the whole run.
- **The confound generator is one confound.** Shared block structure is the confound the matched
  permutation was built for, so Leg C is close to a home-field test for that guard. It shows the
  leg is not decorative; it does not show the leg is sufficient.
- **`styxx.coupling`'s withdrawal for neural time series stands.** This uses three of its
  primitives (`phase_randomize`, `_confound_matched_perm`, `_trend_r2`) and makes no claim about
  the module's headline API. `styxx.power` is quarantined and imported by neither generator.
- **A derived bar is not a correct bar.** It is a bar whose achievability was measured before it
  was frozen. That closes the failure that fired six times in one week; it does not close the
  failure where the measurement is wrong.
- **No real effect size has been read at freezing time.** The only real-data quantities entering
  the power basis are the lag-1 autocorrelation range and series length, both published in
  FINDING_c5.

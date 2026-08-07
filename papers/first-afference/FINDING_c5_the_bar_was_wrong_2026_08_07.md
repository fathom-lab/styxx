# FINDING — C5: the framework was right, my gate was impossible — a third bar set without reference to achievable power

Fathom Lab · 2026-08-07 · prereg: `PREREG_c5_roi_2026_08_07.md` (frozen before the ROI data was
downloaded) · receipt: `c5_result.json` · scored by `styxx.protocol`.

**Machine verdict: `FRAMEWORK_WRONG__blind_even_on_the_right_tissue`.** The verdict string stands
as computed. **The interpretation below contradicts its name, and the contradiction is the
finding.**

| gate | bar | measured | pass |
|---|---|---|---|
| G1_finds_isc | ≥ 0.80 | 0.0952 | ❌ |
| G2_rejects_reversed | ≤ 0.10 | 0.0 | ✅ |
| G3_rejects_independent_ar | ≤ 0.10 | 0.0 | ✅ |
| G4_rejects_shared_trend | ≤ 0.10 | 0.0 | ✅ |

## What the numbers actually say

Two of twenty-one pairs licensed, and they are **the two strongest**: signed mean r of 0.3742
and 0.2615, both clearing surrogate and permutation nulls. The rest sit lower, median 0.084.
The pairs that failed while looking substantial — 0.192 to 0.237 — carry surrogate p between
0.0499 and 0.1297. The instrument licensed the strongest coupling and refused the rest.

That is not blindness. That is an underpowered measurement being reported honestly, and the
arithmetic is unambiguous. These BOLD series carry lag-1 autocorrelation of 0.4967 to 0.8054, which by the standard
Bartlett correction leaves **effective sample sizes of 6.9 to 45.8** against a nominal three
hundred timepoints (`c5_effective_df_addendum.json`). At the cohort's median effective n, a correlation must exceed 0.3746 to clear
significance. The strongest pair, at 0.3742, sits essentially on it. The pairs at 0.19 to 0.24 do not, and **no
statistic and no null could license them without lying.**

## The error is mine, and it is the third of its kind this week

`G1_finds_isc >= 0.80` demanded that eighty percent of *individual subject pairs* reach
significance from a two-column ROI mean over a few hundred autocorrelated timepoints. The field never
claims pair-level significance from this; intersubject correlation is established by pooling
across many vertices and many subjects, precisely because single pairs at this length do not
carry the degrees of freedom. I set a bar without computing whether any instrument could clear
it.

That is the same error as b37's G2 (a noise-passable floor) and b48's G2 (a max-of-45 statistic
judged against a single-draw bar). Three times in one week, and the b48 finding already contains
the sentence *"naming a failure mode does not immunize you against it."* It did not.

## What this does and does not license

- **The C1–C4 blindness is NOT explained away by this.** Those runs used 500 vertices and are a
  separate question; C5 changed the tissue and the dilution hypothesis remains untested at
  vertex scale.
- **`styxx.coupling`'s withdrawal for neural time series stands unchanged.** Nothing here
  licenses it, and this document is not a rehabilitation.
- **What is now known:** the composed guards (surrogate + permutation + trend) refuse every
  attack and license only coupling strong enough to clear the true effective degrees of freedom.
  On this evidence the framework is conservative and correct; it has never been shown to be
  *usefully powered* at pair level, and at these effective sample sizes it cannot be.
- **The successor is a power calculation, not another exam.** Any future gate on this data must
  state the effective sample size first and derive its bar from that. A prereg that fixes a bar
  before computing achievable power is not preregistration, it is decoration.

*Losing branch named in advance and it landed; the verdict name is wrong and this document says
so rather than adopting a flattering reading. Every number grounds in `c5_result.json`. Sealed
before commit.*

## Correction, recorded because it happened here

A first draft of this document quoted the effective-sample-size range as 12.6 to 45.9. That was
an earlier scratch estimate over three subjects, not the seven-subject computation that is
actually committed in `c5_effective_df_addendum.json`, which gives **6.9 to 45.8**. The OATH
verifier ABSTAINED on both tokens rather than binding them, so it did not catch the mismatch —
the numbers were caught by reading the receipt against the prose by hand. That is a real gap in
the verifier's coverage on this class of token, it is now on the record, and the corrected range
makes the underpowering *worse*, not better.

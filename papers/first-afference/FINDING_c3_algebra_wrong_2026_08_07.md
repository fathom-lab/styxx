# FINDING — C3: ALGEBRA_WRONG — because I folded the linear statistic with an absolute value

Fathom Lab · 2026-08-07 · prereg: `PREREG_c3_linear_statistic_2026_08_07.md` (frozen before the
run) · receipt: `c3_result.json` · scored by `styxx.protocol`.

## Verdict (machine-computed)

**`ALGEBRA_WRONG__linear_statistic_also_blind`** — and additionally the drift attack licensed at
1.0, because the frozen method omitted the trend refusal that lives in `couple()`.

| gate | bar | measured | pass |
|---|---|---|---|
| G1_finds_isc | ≥ 0.80 | 0.0952 | ❌ |
| G2_rejects_reversed | ≤ 0.10 | 0.0 | ✅ |
| G3_rejects_independent_ar | ≤ 0.10 | 0.0 | ✅ |
| G4_rejects_shared_trend | ≤ 0.10 | 1.0 | ❌ |

## The diagnosis: the exam caught my implementation contradicting my own theorem

C2 proved that a **squared** statistic has a spectral-surrogate floor because the expectation of
a squared cosine under a uniform phase is one half. The C3 prereg then specified the statistic as
the mean of **absolute** matched-column correlations — and an absolute value is a fold, not a
linear map. The expectation of |cos| under a uniform phase is not zero either. I re-introduced
the same class of floor I had just finished proving fatal, one document later, wearing different
notation. The prereg froze that error; the machinery scored it; the branch named in advance —
*"if the data refuses anyway, the mechanism story of C2 is incomplete"* — fired, and the
incompleteness was in my implementation of the mechanism, not the mechanism.

The G4 failure is a separate, plainer design error: the frozen method retained the surrogate and
permutation nulls but not the shared-trend refusal, so independent drifting streams licensed.
The guard exists in `couple()`; the exam simply proved it is load-bearing and not optional.

## What C4 must be, specified by these two failures

The **signed** mean matched-column correlation (meaningful in matched space, where sign is
shared), which has surrogate expectation zero without folding — composed with the existing trend
refusal, under the same four gates, frozen before running, red-teamed before release. If the
signed statistic also fails, the C2 mechanism story is genuinely incomplete.

*Two exams, two failures, both published, and each failure specified its successor precisely.
Every number grounds in `c3_result.json`. Sealed before commit.*

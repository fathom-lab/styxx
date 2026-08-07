# FINDING — C2: the surrogate holds every refusal and still cannot see — because a squared statistic is structurally blind under a spectral null

Fathom Lab · 2026-08-07 · prereg: `PREREG_c2_surrogate_recalibration_2026_08_07.md` (frozen
before the implementation existed) · receipt: `c2_result.json` · scored by `styxx.protocol`.

## Verdict (machine-computed)

**`STILL_BLIND__surrogate_does_not_recover_isc`** — the surrogate does not ship, and
`styxx.coupling` keeps its documented withdrawal for neural time series.

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G1_finds_isc | ≥ 0.80 of real pairs licensed | 0.0476 | ❌ |
| G2_rejects_reversed | ≤ 0.10 | 0.0 | ✅ |
| G3_rejects_independent_ar | ≤ 0.10 | 0.0 | ✅ |
| G4_rejects_shared_trend | ≤ 0.10 | 0.0 | ✅ |

Every safety gate held: zero false positives on time-reversed pairs, on the red team's
independent-autoregressive attack, and on shared drift. But twenty of the twenty-one genuinely coupled pairs
now return `CONFOUND_ONLY__explained_by_spectrum` — the permutation null would license them
(median matched p 0.002) and the surrogate absorbs them (median surrogate p 0.1633).

## The diagnosis, and it is analytic, not bad luck

The RV coefficient — like CKA, like every alignment measure built on **squared**
cross-covariances — decomposes over frequencies into squared per-frequency cross-power times
the squared cosine of the phase relation. Phase randomization draws that phase uniformly, and
the expectation of a squared cosine under a uniform phase is **one half**.

So the spectral-surrogate null does not sit near zero for a squared statistic. It sits near
**half the total cross-power** — a high floor that exists whether or not the streams are truly
aligned. A genuine but partial coupling (and intersubject correlation is always partial) adds
alignment at the coupled frequencies, but the observation must beat a null already carrying
half of *everything*, and it does not. **The statistic and the null are incompatible by
construction: phase randomization is the right null for autocorrelated data, and RV is the
wrong statistic to score it with.** No amount of tuning fixes this pairing; the blindness is in
the algebra.

The single licensed pair is consistent with this: a pair strong enough to clear even the
half-cross-power floor.

## What this reframes about the week

We now have three results about the same instrument that compose into one sentence:

- a **permutation** null with a squared statistic licenses independent autocorrelated streams
  (the red team's every-seed result);
- a **circular-shift** null refuses genuine coupling because shifted autocorrelated streams
  still share slow structure (C1, twenty of twenty-one refused);
- a **spectral-surrogate** null refuses genuine coupling because a squared statistic's
  surrogate floor is half the cross-power (this finding, twenty of twenty-one refused).

The constant across all three is the statistic. **The nulls were never the problem alone; RV
is unpairable with any of them for this job.** A statistic *linear* in cross-covariance has
expectation zero under phase randomization — for streams in a matched space, that is simply
the mean matched-column correlation, which is the field's own classic ISC statistic scored
against FT surrogates. The field's standard practice is not a convention we skipped; it is the
unique algebra that works, and we re-derived it by failing twice.

## What happens next, per the frozen prereg

The surrogate implementation stays (it is correct); the `INVALID__autocorrelation...` refusal
it replaced stays replaced (the new `CONFOUND_ONLY__explained_by_spectrum` verdict is more
informative and equally conservative). But **no positive on autocorrelated data can be
licensed through a squared statistic**, the module's documentation will say exactly that, and
the mind↔brain withdrawal stands. The successor (C3) is specified by the algebra: a
linear-in-cross-covariance statistic for surrogate licensing, with its own preregistered
two-direction exam, red-teamed before release.

## Limits

Seven subjects, one story, one parameterisation — but the mechanism is analytic and does not
depend on them. The three synthetic-attack gates ran at their original constructions and
scales; their zeros carry the usual small-n caveats.

*Frozen before implementation; the losing branch named in advance; the failure published with
its mechanism. Every number grounds in `c2_result.json`. Sealed before commit.*

# FINDING — C6: the losing branch landed, and this time the null carries a number

Fathom Lab · 2026-08-13 · prereg: `PREREG_c6_derived_bar_2026_08_13.md`, frozen and committed
(18b8c61) before `run_c6.py` existed · receipt: `c6_result.json` · power basis: `c6_power.json`,
`c6_basis_v2.json` · ceiling table: `c6_ceiling.json` (emitted by `emit_c6_ceiling.py`).

> **Note on this document's own numbers.** A first draft derived the implied pairwise
> correlations and the two licensing rates in prose only. Auditing the draft with
> `styxx.claim_audit` (hardened earlier today) flagged them UNSOURCED — correct arithmetic, but
> present in no receipt. They are now emitted to `c6_ceiling.json` from already-committed inputs
> and the table below cites it. A number that exists only in a sentence is the thing that
> auditor is for, and it caught its author.

**Machine verdict: `NULL__below_derived_ceiling_at_known_power`.**

| gate | bar | measured | pass |
|---|---|---|---|
| G1_exceeds_null_ceiling | ≥ 5 of 7 | **1** | ❌ |
| G2_null_calibration_holds | ≤ 0.05 | **0.0036** | ✅ |
| G3_matched_leg_bind_count | recorded, not gated | **0** | — |

This is the branch named in advance: *"`NULL__below_derived_ceiling_at_known_power` is the branch
I expect."* It landed.

## What is different from C5

C5 also returned a null, and that null was uninformative — nobody knew what the instrument could
detect, so "we found nothing" and "there is nothing to find" were indistinguishable. C6's null is
bounded, because the bar was derived from a measured detection curve before the data was touched.

From `c6_basis_v2.json` at the worst committed autocorrelation (rho=0.8054), the k=5 bar clears:

| planted coupling c | implied pairwise r | P(bar cleared) |
|---|---|---|
| 0.32 | 0.1024 | 0.200 |
| 0.36 | 0.1296 | 0.567 |
| **0.40** | **0.1600** | **0.950** |
| 0.44 | 0.1936 | 1.000 |

(all four rows in `c6_ceiling.json`; `first_c_with_95pct_power` = 0.4,
`first_c_with_100pct_power` = 0.44, `ceiling_pairwise_r` = 0.16)

The bar clears 95% of the time at c=0.40 and always by c=0.44. **It was not cleared.** So the
statement this run licenses is:

> The shared structure in these seven EAC series sits **below roughly c=0.40** (pairwise r ≈ 0.16)
> at worst-case autocorrelation — a stated ceiling on the effect, not an absence of evidence.

That is the entire return on the prereg, and it is what a null is supposed to buy.

## The per-subject picture

| subject | obs | surrogate p | matched p | outcome |
|---|---|---|---|---|
| sub-008 | 0.2795 | 0.0020 | 0.0020 | **licensed** |
| sub-006 | 0.2229 | 0.0220 | 0.0040 | surrogate binds |
| sub-005 | 0.1630 | 0.0319 | 0.0020 | surrogate binds |
| sub-003 | 0.1490 | 0.0938 | 0.0020 | surrogate binds |
| sub-002 | 0.1402 | 0.0679 | 0.0739 | both veto |
| sub-004 | 0.1166 | 0.1238 | 0.0020 | surrogate binds |
| sub-001 | 0.0422 | 0.3473 | 0.1896 | both veto |

The instrument licensed the strongest subject and refused the rest — the same conservative
behaviour C5 documented, now with a known detection floor underneath it. Leave-one-out did not
rescue the design: pooling six against one should beat pairwise, and it did — C5 licensed 2 of 21
pairs, C6 licenses 1 of 7 (`c5_pair_license_rate` 0.0952 vs `c6_cohort_license_rate` 0.1429 in
`c6_ceiling.json`) — but not nearly enough to reach the bar.

## G3, recorded because it was frozen as unfailable

**`matched_leg_bind_count = 0`.** On the real cohort the confound-matched permutation never once
vetoed something the surrogate would have licensed; the surrogate leg was the binding constraint
in all four near-misses. This was deliberately left ungated — I refused to predict the real bind
profile — and it is now on the record.

It is consistent with the power basis: `matched_binds` fired almost never on shared-signal
streams and only under planted block confounds. **The absence of matched-leg binding here is
therefore evidence that this series does not carry the block-structured confound that leg exists
to catch** — a small positive result hiding inside a null, and one I could not have claimed if
G3 had been written as a gate I could pass.

## G2 is the load-bearing gate, and it held

The power basis is synthetic and its generator is mine. G2 existed to catch the case where that
does not transfer: replace every subject with its own phase-randomised surrogate (same spectrum,
same autocorrelation, coupling destroyed) and check the licensing rate lands in the regime the
synthetic null occupied.

Synthetic null: 0.0057–0.0125 per subject. Real phase-randomised cohort: **0.0036** over 280
subject decisions. Same regime, marginally more conservative. The bar transfers. Had this come
back high, the frozen outcome map would have returned `REFUSED__power_basis_does_not_transfer`
and the G1 number would have been discarded rather than reported — a null that could have
invalidated its own headline gate, which is the property C5's design lacked.

## What this does NOT license

- **No claim that there is no coupling.** The claim is a ceiling: below c≈0.40. Effects beneath
  the detection floor are not excluded and cannot be, at n=7, n_t=300, two columns.
- **The C1–C4 vertex-scale question stays open.** Different tissue, different dilution question.
- **`styxx.coupling` remains withdrawn for neural time series.** Three primitives were used; the
  headline API is not rehabilitated and this document does not rehabilitate it.
- **One cohort, one ROI, one statistic.** The generator behind the power basis is additive shared
  signal in AR(1) noise; real BOLD coupling is not that, and P1's lesson (an author's battery
  encodes the author's blind spots) applies to the bar as much as to the exam.

## Priority, corrected by this document's own linter

A first draft titled this *"for the first time in this program it carries a number."* The
overclaim linter flagged it, and the flag was right: `styxx.anchors` shipped `blindspot_power`
and `min_anchors_for_power` on 2026-07-23, and `FINDING_anchor_power_instrument` already derives
an anchor budget from achievable power. **Power-derived design is not new here.** What is
specific to C6 is narrower and stated as such: it is the first bar *on the first-afference
series* derived from a measured detection curve rather than asserted, after three consecutive
bar mis-specifications on this data (b37, b48, C5). Title corrected before commit.

*The prereg was frozen at 12:58, committed before the runner existed, and named this branch as
expected. The bar was derived by a stated rule from a measured curve and the rule chose k=5 over
my own k=6. The result is a null with a number attached. Sealed before commit.*

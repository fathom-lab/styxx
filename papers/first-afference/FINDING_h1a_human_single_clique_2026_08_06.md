# FINDING — H1a: eight human brains share a representational frame, and it is continuous — our own island prediction dies on real neural data

Fathom Lab · 2026-08-06 · prereg: `PREREG_h1a_human_alignment_2026_08_06.md` (frozen at
`e42bf9c`, before a single byte of the dataset existed on disk) · receipt: `h1a_result.json` ·
scored by `styxx.protocol`.

## Verdict (machine-computed)

**`HUMAN_SINGLE_CLIQUE__alignment_is_continuous`** — the branch we preregistered as expected.

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G0_cohort | ≥ 8 members | 8 | ✅ |
| G1_shared_frame | cohort median − random-null p95 > 0 | 0.1974 | ✅ |
| G2_islands_present | gap-screen p ≤ 0.05 | 0.0779 | ❌ |

## What was measured

Eight NSD subjects, `nsdgeneral` betas for the images every subject saw, averaged over each
subject's repetitions and intersected to the 907 images present for all eight. Voxel counts
differ by 41 percent across subjects (12682 to 17907); the instrument works in item space, so
that is expected and harmless. **No subject was dropped**, including the four who completed
fewer sessions (22500 or 24000 trials against 30000).

## Two results, and the second is the interesting one

**1. Eight independently-developed human brains share a representational frame, emphatically.**
Median pairwise frame affinity 0.2222 against a random-frame null whose 95th percentile is
0.0248 and whose largest of 1000 draws reached only 0.0266. The cohort sits roughly nine times
above the ceiling of chance. Convergent structure across separately-grown minds is not subtle
here — it is the loudest thing in the data.

**2. That alignment is continuous, not clustered.** Per-subject mean affinity runs 0.1530 to
0.2575 in an unbroken spread, and the gap screen returns p 0.0779 — above the 0.05 bar the
prereg fixed. There is no island.

## The flagged subject is not an island, and here is why we say so

The prereg's stated MAD rule flags **subj08** at 0.1530, below its 0.1969 cutoff. **That flag
is not evidence.** Before this run, on the same instrument, we checked what the rule does to a
cohort of eight streams of *pure independent noise*: it flagged a member there too. A rule that
draws a line through a continuum will always name whoever is furthest left. The candidate-island
list means nothing unless the bimodality gate passes, and here it did not. We are stating this
because subj08 is exactly the kind of number a lab wanting an island would have led with.

## What this does to our own prediction

This morning this lab registered `PREDICTION_h1_human_islands`, betting that human cross-subject
decoding shows island structure. **Its model-side precursor failed (B47), the published human
literature leaned against it, and now its neural-side precursor has failed on real brains.**
Third negative, and the first on human data. The prediction document has been amended again.

We predicted this outcome in the prereg, in advance, in writing — *"we expect
`HUMAN_SINGLE_CLIQUE`… predicting our own earlier claim will fail is the point of writing it
down."* That is the only reason this document gets to claim anything at all.

## Confirmed against the reference statistic

The gate above uses this module's gap screen. **Hartigan's dip test — the statistic the methods
literature names — was run afterwards on the same vector and returns p 0.6036**, agreeing with
the screen and sitting further from the bar. Receipt: the diptest addendum committed beside this
finding. The screen is the more liberal of the two here, so its non-flag is the stronger
statement.

## ERRATUM — 2026-08-06, hours after publication, from an adversarial audit of our own instrument

An internal red team attacked `styxx.islands` and impeached parts of this document. Three
corrections, in descending order of how much they cost us.

**1. The reported p was a lucky draw, and the honest number is worse.** `_gap_p` estimated its
p from 1000 permutations; at this value the Monte-Carlo standard error is about a third of the
distance to the 0.05 bar. Re-run at 20000 permutations the value is **0.0634**, and an
exhaustive estimate puts the truth near 0.062 — the published **0.0779 sits at roughly the 98.7th
percentile of its own noise**, and about **7.4% of random seeds return `ISLANDS_PRESENT` on this
identical data at these identical settings**. Seed 343 is this module's documented default and
appears throughout the repo, so this is coincidence rather than seed-shopping. It is still a
number a reader would have taken as more stable than it is. `n_perm` now defaults to 100000.

**2. "Weak evidence of absence" was too generous; the honest phrase is "almost no evidence of
absence."** The audit measured the screen's power directly: against a single island it has
**about 25% power at the 3.86 SD separation subj08 actually shows**, and would need ≈6.2 SD for
80%. And the non-detection was close — holding the other seven fixed, subj08 would have been
flagged at 0.1496 against its measured 0.1530, a margin of **0.17 SD**.

**3. The "confirmed against the reference statistic" section above is vacuous and is withdrawn as
confirmation.** Hartigan's dip has **under 1% power against a single island at n = 8** — flat
across every separation including infinite. Its p of 0.6036 was not agreement; it was a test with
no capacity to disagree. It is retained as a reported number, not as support.

**What we tested rather than conceded, and what survived.** The audit's sharpest claim was that
this document's headline — a shared frame at nine times the null — can be manufactured by a
shared per-item *amplitude* profile with no shared geometry whatsoever, and it demonstrated 7×
that way on synthetic data. We had deleted the betas; we re-derived them and ran the control.
**With every item's response vector unit-normalised, killing the amplitude channel entirely, the
cohort median rises from 0.2222 to 0.2437 against the same 0.0248 null — 9.8× rather than
9.0×.** The shared frame is not an amplitude artifact. `styxx.islands` now applies this
normalisation by default (`normalize_amplitude=True`).

**None of this changes the verdict.** `HUMAN_SINGLE_CLIQUE` still follows from the frozen gates,
and it follows under the amplitude control too (gap p 0.0866). What changes is how much the
non-detection is worth: less than this document originally implied.

## Limits, which are severe and were fixed before the run

- **This measures alignment, not decoding accuracy.** H1 proper is about accuracy. Our own b46
  result says the two are joined by a switch rather than a ramp, so a continuous alignment
  distribution does **not** prove a continuous accuracy distribution. This is H1's precursor.
- **n = 8 is exactly the instrument's floor.** A gap screen on eight points has very little
  power, so this is **weak evidence of absence**, not a demonstration that human islands do not
  exist. NSD has no ninth subject.
- One dataset, one ROI, one modality, one alignment construction, one seed. The p of 0.0779 is
  not far from the bar; a differently-composed cohort could land the other side of it.

## What is genuinely new here

To this lab's knowledge, and per an external survey of the neuroimaging methods literature that
searched for exactly this, **no formal bimodality test had previously been applied to a
per-subject neural alignment or decoding distribution.** The field describes these distributions
in words — "varied largely between participants" — and routinely excludes low performers before
anyone looks. The reconstruction literature built on this very dataset evaluates on the four
subjects of eight who completed all sessions. This run kept all eight and tested the shape
statistically. The answer is boring, and it is the first one.

*Prereg frozen before the data was downloaded; the runner committed before any result existed;
the losing branch written first and it is the one that landed; every number grounds in
`h1a_result.json`. Sealed before commit. Raw betas were not redistributed and were deleted after
extraction.*

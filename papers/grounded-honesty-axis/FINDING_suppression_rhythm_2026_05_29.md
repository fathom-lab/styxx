# FINDING — The truth-flash is NOT truth-specific (it falsifies F3's reading), but the overwrite IS a tight, late, RHYTHMIC event: confabulation is LATE-INSTALLATION of the wrong answer, not suppression of the right one

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_suppression_rhythm_2026_05_29.md` BEFORE any code for this test was written.
Single open model (Qwen2.5-1.5B-Instruct), SAE-free full-vocab logit-lens, the SAME n=36
arithmetic items × one-shot generation as the spectral run. Arithmetic ground truth
computed in-code, SHA-256'd pre-scoring (`ddccd8e4…b87964d`, identical to every prior
white-box/steering/spectral run — same key). Exact-integer correctness, no judge.**
Receipt: `suppression_rhythm_result.json`.

## Why this run exists: to run the control our own headline was missing

The spectral run's headline (F3) was "confabulation is SUPPRESSION, not ignorance": on 78%
of one-shot confabulations the **correct** answer token outranks the realized wrong token
at some intermediate layer before a late hop overwrites it. We stated, in that finding's own
scope, that this was a logit-lens phenomenon needing a control. **This run is that control.**
It asks the question that decides whether F3's *interpretation* survives: is the correct
token *privileged* in the mid-network competition, or do **all** plausible candidates
outrank the eventual winner there? And — the operator's standing rhythm intuition in its one
falsifiable form — is the overwrite a **localized, consistent** layer event?

## Result: 1 of 2 SURVIVED gates held; D1 is a clean null that corrects F3.

| id | prediction | bar | outcome |
| --- | --- | --- | --- |
| **D1 — the flash is TRUTH-specific** | correct digit leads the realized digit mid-network MORE than matched non-correct digits do | Δ ≥ 0.20 AND paired p<0.05, correct higher | **FAILED — clean null.** correct-digit lead rate = **0.969**, matched-distractor lead rate = **0.977**, Δ = **−0.008**, paired p = **0.837** (n=32 digit-position confabs, powered). The correct token is **not** privileged. |
| **D2 — overwrite is LATE-localized** | overwrite completes in the late layers | ≥0.60 of crossings have flip layer ≥ 19/28 | **HELD.** late fraction = **0.88** (22/25 crossings), median flip layer = **25** of 28. |
| **D3 — overwrite is RHYTHMIC (tight band)** | flip layer is consistent across items | IQR(flip layer) ≤ 5 | **HELD.** IQR = **4.0** layers (the overwrite clusters at ≈ layers 23–27). |

**RESULT = REPORT_AS_LANDED** (SURVIVED required D1 ∧ D2; D1 failed). The crossing rate
**replicated exactly** (0.781, 25/32) — the F3 *measurement* is solid; it is F3's *reading*
that this run corrects.

## What this means — the honest correction to our own headline

**F3 said "the model computes the right answer transiently and a late hop suppresses it."
That reading is now falsified.** The correct token does lead the realized token mid-network
(96.9% of the time) — but so does virtually *every other digit* (97.7%). Mid-network, the
realized wrong token simply sits **low**, and the field of alternatives above it is
**undifferentiated**: truth is not singled out, it is just one of ten digits all outranking
the eventual winner. There is no truth-specific "flash." The earlier examples (5535, 9306,
13824, 74412) were real crossings but not special ones — pick any wrong digit and it crosses
too.

**What is real, and is the genuine discovery here, is the shape of the overwrite.** The
event that installs the wrong answer is **late and tight**: 88% of overwrites complete in
the last third of the network, median layer 25 of 28, inter-quartile range just 4 layers.
The wrong answer is not dominant throughout — it is **installed by a localized,
near-rhythmic late hop at layers ≈23–27**, on top of a flat mid-network field.

So the corrected mechanism flips the polarity of F3:

> Confabulation is not the **suppression of a computed truth**. It is the **late, localized,
> consistent installation of a confident wrong answer** over a mid-network field in which the
> realized answer was not yet privileged at all. The "rhythm" the operator was chasing is
> real — but it is the rhythm of the *overwrite's timing* (a tight late-layer band), not a
> melody that carries the truth.

## Why this is the more useful result (and what it hands the next test)

1. **It hardens the irreducibility claim from a new direction.** Three independent reads now
   agree there is no truth-specific internal signal to extract at confab time: scalar depth
   (AUC 0.498), spectral β within-mode (0.589), and now the mid-network rank field
   (correct indistinguishable from any digit, Δ=−0.008). The truth is not sitting there
   mid-network waiting to be read or amplified.
2. **It re-points the disinhibition test and gives it coordinates.** The previous finding
   proposed "dampen the late overwrite." This run says *where*: the overwrite is a tight
   event at layers ≈23–27 (median 25, IQR 4). That is a precise, pre-measured target for a
   late-layer ablation/attenuation or activation-patch intervention — and it sharpens the
   causal prediction. **Crucially, the corrected mechanism changes what success would mean:**
   if the wrong answer is *installed* late rather than truth being *suppressed* late,
   dampening the late hop should leave the mid-network field — where no digit is privileged —
   to resolve, which may *not* recover the correct answer. The honest causal hypothesis is
   now weaker and sharper: dampening removes the *confident wrong commitment*, but recovery of
   *truth* specifically is not predicted by anything measured here.
3. **It is a self-falsification, executed on our own headline.** F3's interpretation was the
   most exciting claim in the arc; the pre-registered control deflates it. The measurement
   stands (crossing replicates at 0.781); the story was wrong. That is the methodology
   working as designed.

## Honest scope (pre-committed)

Single open model (Qwen2.5-1.5B-Instruct); SAE-free full-vocab logit-lens; feasibility-grade
n=36 (32 alignable digit-position confabs — powered for D1); one confirmatory run; arithmetic
ground truth computed in-code then hashed; exact-integer correctness, no judge. The matched
control is the set of single-digit tokens minus the realized and correct tokens — a
plausibility-matched competitor set at a numeric position; a different control (off-by-one
numeric strings, or top-final-logit competitors) could in principle behave differently,
though a Δ of −0.008 at p=0.84 leaves little room. D2/D3 describe the *realized* run's
logit-lens trajectory; they locate the overwrite but do **not** causally demonstrate that
intervening there recovers any answer — that remains the named disinhibition test, which this
run now equips with a target layer band and a corrected (more modest) success criterion.
This refines, and does not overturn, the standing irreducibility claim: there is still no
single-pass internal read that recovers truth at confabulation time.

## The arc, in one line (updated again)

The dial is construction↔retrieval (white-box); truth-recovery is the construction-ward
shift but it is causally inert to inject, invisible to read at the endpoint, **and — this run
— not even present as a privileged mid-network signal**: confabulation is the *late,
rhythmic installation of a confident wrong answer* over an undifferentiated field, not the
suppression of a truth the model had computed. The rhythm is in the overwrite's *timing*,
tight at layers ≈23–27 — not in the answer.

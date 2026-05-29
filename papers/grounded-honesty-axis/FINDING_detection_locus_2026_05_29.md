# FINDING — confabulation is INTERNALLY LEGIBLE in a single forward pass on Qwen arithmetic: clean first-token entropy/margin separate confab from correct (AUC ~0.92) almost as well as N=10 resampling (AUC 0.98) — so detection does NOT require re-derivation here, and "confident confabulation" is REFUTED on this model/domain (pre-registered hypothesis FALSIFIED, REPORT_AS_LANDED)

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_detection_locus_2026_05_29.md` BEFORE any code. Qwen2.5-1.5B-Instruct (the white-box model
— all three signals read from the SAME network), same balanced set as the confab-specificity run,
resampling N=10 at T=1.0 (the validated grounding setting), Stability from exact distinct-integer
counts (no judge), single-pass entropy/margin from the clean logit-lens at the first answer token,
arithmetic ground truth SHA-256'd pre-scoring (`0eb5c90d…752b1`), exact-integer correctness.**
Receipt: `detection_locus_result.json`.

## Why this run exists

The confab-specificity null (`FINDING_confabulation_specificity_2026_05_29.md`) showed the late
band is SHARED machinery — a single-pass *intervention* can't separate confab from correct. Yet
the black-box grounding detector separates confab from truth at AUC 0.966
(`[[grounded-honesty-ceiling-break-2026-05-28]]`). The unifying question: **does the
detection signal live ONLY in cross-derivation resampling, or is it already in a single pass's
internal confidence?** Pre-registered hypothesis ("confident confabulation"): single-pass
confidence is near chance, resampling dominates by ≥0.20 AUC. That hypothesis is FALSIFIED.

**Method.** On the same items, three detector scores (higher = more-likely-confab): (1) resampling
`instability = 1 − Stability` over N=10 temperature-1.0 resamples; (2) clean first-token entropy;
(3) `−margin` (top1−top2 clean logit gap) at the first answer token. Bars: B1 AUC(instability)
≥0.70; B_contrast AUC(instability) − max(AUC(entropy), AUC(−margin)) ≥0.20. SURVIVED iff B1 ∧
B_contrast.

## Result: REPORT_AS_LANDED — B1 holds, the locus contrast FAILS

| signal | AUC (confab vs correct) | bar | held |
| --- | --- | --- | --- |
| **B1** resampling instability | **0.980** | ≥0.70 | **HOLD** |
| B2 single-pass clean entropy | **0.924** | reported | — |
| B3 single-pass −margin | **0.915** | reported | — |
| **B_contrast** = 0.980 − 0.924 | **+0.056** | ≥0.20 | **FAIL** |

| group means | instability | clean entropy | logit margin | modal-resample correct |
| --- | --- | --- | --- | --- |
| confab (n=32) | 0.774 | 0.234 | 6.61 | **0.00** |
| correct (n=24) | 0.014 | 0.045 | 14.54 | **1.00** |

## The three claims that land

1. **Confabulation is internally legible in a SINGLE forward pass on Qwen arithmetic.** Clean
   first-token entropy (AUC 0.924) and logit margin (0.915) separate confab from correct almost as
   well as ten resamples (0.980). The model is genuinely *less confident* on the items it
   confabulates — higher entropy (0.234 vs 0.045) and a far smaller leading-digit margin (6.6 vs
   14.5). You do not need re-derivation to read it here.
2. **"Confident confabulation" is REFUTED on this model/domain — and that is a BOUNDARY, not a
   contradiction.** The earlier confident-confabulation observation
   (`[[grounded-honesty-ceiling-break-2026-05-28]]`, grounded-arc engine: confident-when-wrong on
   gpt-4o-mini *hallucination*) was a DIFFERENT model and a DIFFERENT instrument. On Qwen small-model
   arithmetic, wrongness comes WITH single-pass uncertainty. So whether confabulation is internally
   confident is **model/domain-specific**, not universal — this run maps one corner where it is not.
3. **Resampling remains marginally stronger and is what you fall back on when internals are
   unreadable.** Instability edges single-pass (0.980 vs 0.924), and the modal resampled answer is
   correct on 100% of correct items and 0% of confabs — a perfect cross-sample separation. The
   grounding detector's resampling design is NOT uniquely necessary on this white-box/arithmetic
   corner, but it is the only available signal in the closed-model / cross-vendor settings the
   keystone targets (where entropy/margin are not exposed). Re-derivation is sufficient everywhere;
   here it is simply not *necessary*.

## Honest scope + the confound (pre-committed + observed)

The CORRECT group is easy and the CONFAB group is hard — difficulty is confounded with group, and
self-consistency / single-pass uncertainty both track derivation difficulty (Stability ≈
self-consistency ≈ difficulty, i.e. self-consistency-not-truth, per the keystone). So none of B1/B2/B3
is a *truth oracle*; each is a difficulty-driven-wrongness detector. The load-bearing, confound-robust
result is **B_contrast**, which holds the difficulty difference FIXED across detector TYPES (same
items): whatever signal exists is equally readable single-pass and via resampling, so the detection
*locus* claim ("only re-derivation can read it") is falsified independent of the confound. Single
open model, arithmetic only, one confirmatory run, feasibility-grade (32 confab + 24 correct), N=10
at T=1.0, no judge. Does NOT touch the correctness bound: every signal here DETECTS confabulation,
none CORRECTS it (only method-diverse re-derivation can), and the detector flags abstain, never the
answer.

## The arc, in one line (updated)

The confabulation is a late, tight, graded, distributed install of SHARED answer-commitment
machinery (band-knockdown can't flag it) — yet on Qwen arithmetic the wrong commitment carries its
own single-pass uncertainty signature (clean entropy/margin separate it at AUC ~0.92, ≈ ten
resamples), so detection does not require re-derivation here even though re-derivation is the
universal fallback; and, as ever, every signal moves confidence/abstention, never correctness —
only re-derivation moves correctness.

# FINDING — single-pass confabulation legibility is CROSS-ARCHITECTURE: on Llama-3.2-1B, clean first-token entropy AND logit margin separate confab from correct PERFECTLY (AUC 1.000), matching N=10 resampling (1.000) — so "confident confabulation" is REFUTED on a SECOND architecture, and the refutation is not a Qwen quirk (B_contrast FAILS again, REPORT_AS_LANDED)

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_detection_locus_llama_2026_05_29.md` BEFORE the Llama run. The detection-locus protocol
UNCHANGED on a second architecture — meta-llama/Llama-3.2-1B-Instruct (16 decoder layers) —
same balanced arithmetic set, same three detector scores, same bars, same SHA-256 answer key as
the Qwen run (`0eb5c90d…752b1`), resampling N=10 at T=1.0, exact-integer Stability (no judge),
single-pass entropy/margin from the clean logit-lens at the first answer token, exact-integer
correctness.** Receipt: `detection_locus_result_Llama-3_2-1B-Instruct.json`.

## Why this run exists

Tonight's Qwen detection-locus run (`FINDING_detection_locus_2026_05_29.md`) found confabulation
is internally legible in a SINGLE forward pass — clean entropy (AUC 0.924) / logit margin (0.915)
separated confab from correct nearly as well as ten resamples (0.980), falsifying the
pre-registered "confident confabulation" hypothesis *on that model/domain*. The honest scope
flagged the open question: is single-pass legibility a Qwen property, or architecture-general?
This run answers it by replicating the protocol verbatim on a different architecture.

## Result: REPORT_AS_LANDED — B1 holds, B_contrast FAILS again (single-pass = resampling)

| signal | AUC (confab vs correct), Llama | bar | held |
| --- | --- | --- | --- |
| **B1** resampling instability | **1.000** | ≥0.70 | **HOLD** |
| B2 single-pass clean entropy | **1.000** | reported | — |
| B3 single-pass −margin | **1.000** | reported | — |
| **B_contrast** = 1.000 − 1.000 | **0.000** | ≥0.20 | **FAIL** |

| group means (Llama, n_conf=35, n_corr=24) | instability | clean entropy | logit margin | modal-resample correct |
| --- | --- | --- | --- | --- |
| confab | 0.892 | 2.872 | **0.62** | **0.00** |
| correct | 0.014 | 0.071 | **6.48** | **1.00** |

## The claims that land

1. **Single-pass confabulation legibility is CROSS-ARCHITECTURE.** On Llama-3.2-1B, just like on
   Qwen2.5-1.5B, the model's single-pass internal confidence separates confab from correct as
   well as resampling does — here, in fact, *perfectly* (entropy and margin both AUC 1.000 =
   instability 1.000). Detection does NOT require re-derivation on either architecture for
   small-model arithmetic. The pre-registered cross-architecture reading: B_contrast FAILS on
   Llama too → legibility is **architecture-general**, not a Qwen quirk.
2. **"Confident confabulation" is REFUTED on a SECOND model.** Llama's confabs carry an even
   louder single-pass uncertainty signature than Qwen's — a near-zero leading-digit margin (0.62
   vs the correct group's 6.48) and high entropy (2.87 vs 0.07). The weaker model is, if
   anything, *less* confident when it confabulates, not more. So the refutation of confident
   confabulation now holds on two architectures of two different families.
3. **The boundary remains the instrument/domain, not the architecture.** The earlier
   confident-when-wrong observation was gpt-4o-mini HALLUCINATION (a different *instrument*), not
   small-model arithmetic. Two architectures now agree that on white-box arithmetic the wrong
   commitment is internally uncertain; the open confident-confabulation regime lives in the
   closed-model hallucination setting, not in a particular open architecture.

## Honest scope (pre-committed)

Second single open model Llama-3.2-1B-Instruct; arithmetic only; one confirmatory run;
feasibility-grade (35 confab + 24 correct, powered); resampling N=10 at T=1.0; Stability from
exact distinct-integer counts (no judge); single-pass entropy/margin from the clean full-vocab
logit-lens at the first answer token; ground truth in-code then hashed pre-scoring (same key as
Qwen). SAME difficulty confound as the Qwen run (CORRECT easy / CONFAB hard) — so B1/B2/B3 are
difficulty-driven-wrongness detectors, not truth oracles; B_contrast holds the confound FIXED
across detector types (same items) and is the load-bearing, cross-architecture-comparable result.
The AUC 1.000 reflects the easy/hard separation being even sharper on a weaker model — NOT a
within-mode truth oracle. Does NOT touch the correctness bound — every signal DETECTS
confabulation, none CORRECTS it; the detector flags abstain, never the answer. Two open
architectures, arithmetic only; closed-model and non-arithmetic regimes remain untested here.

## The arc, in one line (updated)

Confabulation is a late, tight, distributed install of SHARED answer-commitment machinery whose
internal overwrite-geometry is INDEPENDENT of its surface confidence
(`[[FINDING_rhythm_uncertainty_link]]`); and that surface confidence is a clean single-pass tell
of confabulation on BOTH Qwen and Llama small-model arithmetic (entropy/margin ≈ resampling, AUC
0.92–1.00) — so "confident confabulation" is architecture-general-FALSE on this domain, the open
regime is the closed-model hallucination instrument, and every signal here moves
confidence/abstention, never correctness.

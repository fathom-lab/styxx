# FINDING — single-pass confabulation legibility is DOMAIN-GENERAL: on CODE-OUTPUT TRACING (control-flow difficulty, not number size), Qwen2.5-1.5B's clean first-token entropy separates confab from correct as well as N=10 resampling (AUC 0.906 vs 0.908) — so single-pass legibility is NOT an arithmetic artifact, and "confident confabulation" is REFUTED on a SECOND derivation domain (B_contrast FAILS again, REPORT_AS_LANDED)

**Run 2026-05-29. One confirmatory run, pre-registered in
`PREREG_detection_locus_code_2026_05_29.md` (with a pre-confirmatory, validity-motivated amendment)
BEFORE the code-tracing run. The detection-locus protocol UNCHANGED on a structurally different
derivation domain — code-output tracing (loops, branches, nesting, stateful iteration; small
per-step numbers) from `run_code_tracing_grounding.py` — on Qwen2.5-1.5B-Instruct (white-box):
same three detector scores, same bars, ground truth by EXECUTION then SHA-256'd pre-scoring
(`792b8bda426fbfbe…`), resampling N=10 at T=1.0, exact-integer Stability (no judge), single-pass
entropy/margin from the clean logit-lens at the first answer token, exact-integer correctness.**
Receipt: `detection_locus_code_result.json`.

## Why this run exists

The detection-locus runs found confabulation is internally legible in a SINGLE forward pass on
small-model ARITHMETIC — clean entropy / logit margin separate confab from correct nearly as well
as N=10 resampling on BOTH Qwen2.5-1.5B (AUC 0.92) and Llama-3.2-1B (AUC 1.00), refuting "confident
confabulation" on that domain across two architectures. Every result so far was arithmetic
(difficulty = number size). The standing open question (synthesis item 8): is single-pass
legibility a property of arithmetic, or does it hold on a structurally different derivation domain
whose difficulty comes from CONTROL FLOW rather than number magnitude? This run answers it by
replicating the protocol verbatim on code-output tracing.

## Amendment (pre-confirmatory, validity-motivated)

The 36 hard SPECS gave **n_conf=35, n_corr=1**: Qwen-1.5B confabulates essentially every
control-flow snippet, so the hard SPECS contain NO usable CORRECT class. This is a competence-floor
/ powering failure, NOT a verdict. Exactly as the arithmetic run used a separate `EASY_SPECS` pool,
I added an **`EASY_CODE`** pool of trivially traceable, deterministic, import-free snippets
(counting loops, constant-increment, doubling), verified achievable on an 8-item probe (Qwen 5/8).
Both groups are genuine code-tracing (variable state + control flow). Recorded BEFORE confirmatory
scoring; motivated by feasibility, not by any observed detector value.

## Result: REPORT_AS_LANDED — B1 holds, B_contrast FAILS again (single-pass ≈ resampling)

| signal | AUC (confab vs correct), code-tracing | bar | held |
| --- | --- | --- | --- |
| **B1** resampling instability | **0.908** | ≥0.70 | **HOLD** |
| B2 single-pass clean entropy | **0.906** | reported | — |
| B3 single-pass −margin | **0.850** | reported | — |
| **B_contrast** = 0.908 − 0.906 | **0.002** | ≥0.20 | **FAIL** |

| group means (n_conf=35, n_corr=17) | instability | clean entropy | logit margin | modal-resample correct |
| --- | --- | --- | --- | --- |
| confab | 0.518 | 1.060 | **1.84** | **0.00** |
| correct | 0.065 | 0.206 | **5.53** | **0.94** |

## The claims that land

1. **Single-pass confabulation legibility is DOMAIN-GENERAL.** On code-output tracing — where
   difficulty comes from control flow and stateful iteration, NOT number size — Qwen's single-pass
   internal confidence separates confab from correct (entropy AUC 0.906) essentially as well as
   ten resamples (0.908). The legibility is NOT an arithmetic artifact; it holds on a structurally
   different derivation domain. The pre-registered cross-domain reading: B_contrast FAILS on code
   too (0.002) → single-pass legibility is **domain-general**, not arithmetic-specific.
2. **"Confident confabulation" is REFUTED on a SECOND derivation domain.** Code confabs carry the
   same single-pass uncertainty signature as arithmetic confabs — a shrunk leading-digit margin
   (1.84 vs the correct group's 5.53) and elevated entropy (1.06 vs 0.21). The model is not
   confident when it confabulates a traced output; the wrong commitment is internally uncertain at
   the first answer token. The refutation now holds across two architectures (Qwen, Llama) AND two
   domains (arithmetic, code-tracing).
3. **The boundary remains the instrument, not the architecture or the domain.** Three confirmatory
   settings (Qwen arith, Llama arith, Qwen code) all show single-pass ≈ resampling. The open
   confident-when-wrong regime lives in the closed-model HALLUCINATION instrument (gpt-4o-mini), not
   in any particular open architecture or in a particular derivation domain. White-box derivation
   confabulation is internally uncertain by the first token, full stop.

## Honest scope (pre-committed)

Single open model Qwen2.5-1.5B-Instruct; code-output tracing domain only; one confirmatory run;
feasibility-grade (35 confab + 17 correct, powered); resampling N=10 at T=1.0; Stability from exact
distinct-integer counts (no judge); single-pass entropy/margin from the clean full-vocab logit-lens
at the first answer token; ground truth by EXECUTION then hashed pre-scoring; exact-integer
correctness. SAME difficulty confound as the arithmetic runs (CONFAB hard / CORRECT easy) — so
B1/B2/B3 are difficulty-driven-wrongness detectors, not truth oracles; B_contrast holds the
confound FIXED across detector types (same items) and is the load-bearing, cross-domain-comparable
result. Does NOT touch the correctness bound — every signal DETECTS confabulation, none CORRECTS it;
the detector flags abstain, never the answer. Two architectures + two domains; closed-model and
non-derivation regimes remain untested here.

## The arc, in one line (updated)

Confabulation is a late, tight, distributed install of SHARED answer-commitment machinery whose
internal overwrite-geometry is INDEPENDENT of its surface confidence
(`[[FINDING_rhythm_uncertainty_link]]`); and that surface confidence is a clean single-pass tell of
confabulation on BOTH Qwen and Llama AND on BOTH arithmetic and code-tracing (entropy/margin ≈
resampling, AUC 0.91–1.00) — so "confident confabulation" is architecture-general-AND-domain-general
FALSE on small-model derivation, the open regime is the closed-model hallucination instrument, and
every signal here moves confidence/abstention, never correctness.

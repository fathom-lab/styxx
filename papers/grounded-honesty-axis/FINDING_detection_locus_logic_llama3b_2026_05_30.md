# FINDING — single-pass confabulation legibility on MULTI-HOP LOGIC is CROSS-ARCHITECTURE: on Llama-3.2-3B, clean first-token entropy separates confab from correct BETTER than N=10 resampling (AUC 1.000 vs 0.817, B_contrast −0.183) — so the Qwen logic result replicates on a second architecture, "confident confabulation" is refuted on logic across two families, and the Llama-1B failure is confirmed as a pure COMPETENCE FLOOR (REPORT_AS_LANDED)

**Run 2026-05-30. One confirmatory run, pre-registered in
`PREREG_detection_locus_logic_llama3b_2026_05_30.md` (commit `82fc35e`) BEFORE the confirmatory run.
The detection-locus protocol UNCHANGED, same 48 seeded items as the Qwen logic run (hash
`97d816808a6874027637a35a7beeb8a7078aa483f30a01f3ea9e58f9e347e02c`, matched by the run), on
meta-llama/Llama-3.2-3B-Instruct (white-box).** Receipt:
`detection_locus_logic_result_Llama-3_2-3B-Instruct.json`.

## Why this run exists

The Qwen2.5-1.5B logic run (`[[FINDING_detection_locus_logic]]`, REPORT_AS_LANDED) showed single-pass
clean entropy detects multi-hop-logic confabulation as well as N=10 resampling (B_contrast 0.000) —
extending single-pass legibility to a third derivation domain. Arithmetic had been confirmed on TWO
architectures (Qwen, Llama-3.2-1B); logic was Qwen-only. A first cross-architecture attempt on
**Llama-3.2-1B hit the competence floor** (`[[FINDING_detection_locus_logic_llama]]`): the 1B model
answers "2" reflexively to all 24 easy ordering questions, so it has no genuine CORRECT class. This
run uses the more capable same-family **Llama-3.2-3B-Instruct**, which clears the competence gate
(easy correct 18/24 with GENUINE varied answers `v1 {1:12, 2:12}`, hard confab 21/24), to give the
logic domain the same Qwen+Llama two-family support arithmetic has.

## Result: REPORT_AS_LANDED — B1 holds, B_contrast FAILS (single-pass BEATS resampling)

| signal | AUC (confab vs correct), Llama-3.2-3B logic | bar | held |
| --- | --- | --- | --- |
| **B1** resampling instability | **0.817** | ≥0.70 | **HOLD** |
| B2 single-pass clean entropy | **1.000** | reported | — |
| B3 single-pass −margin | **0.952** | reported | — |
| **B_contrast** = 0.817 − 1.000 | **−0.183** | ≥0.20 | **FAIL** |

| group means (n_conf=21, n_corr=18, powered) | instability | clean entropy | logit margin | modal-resample correct |
| --- | --- | --- | --- | --- |
| confab | 0.206 | 1.111 | **0.71** | **0.05** |
| correct | 0.093 | 0.437 | **2.16** | **1.00** |

**This is a cleaner result than Qwen's, not a weaker one.** On Qwen the AUCs saturated at 1.000 (the
HARD/EASY gap was extreme), making B1 uninformative. On Llama-3.2-3B resampling instability is
**0.817 — not saturated** — while single-pass clean entropy is a perfect **1.000**, so B_contrast is
informative and **negative (−0.183): the single forward pass separates confab from correct STRICTLY
BETTER than ten resamples.** The wrong commitment is more legible in one pass than across the sample
distribution.

## The claims that land

1. **Single-pass legibility on multi-hop logic is CROSS-ARCHITECTURE (Qwen + Llama).** B_contrast
   FAILS on Llama-3.2-3B (−0.183) exactly as on Qwen (0.000): single-pass entropy detects logic
   confabulation at least as well as resampling on both families. The logic domain now has the same
   two-family cross-architecture support as arithmetic.
2. **"Confident confabulation" is refuted on logic across two architectures.** Llama-3B logic
   confabs carry a collapsed leading-token margin (0.71 vs the correct group's 2.16) and elevated
   entropy (1.11 vs 0.44) in a single pass. The model is internally uncertain at the first answer
   token when it confabulates a multi-hop deduction — and here that single-token uncertainty is a
   sharper confab tell than cross-sample instability.
3. **The Llama-1B failure was a pure competence floor, now confirmed.** The same family at 3B,
   clearing the easy logic by genuine reasoning, replicates the legibility cleanly. The 1B "2"-reflex
   was incompetence on the task, not evidence that logic legibility is Llama-impossible — exactly the
   reading `[[FINDING_detection_locus_logic_llama]]` pre-committed.
4. **The boundary remains the instrument, not the architecture or the domain.** Five white-box
   confirmatory settings (Qwen arith, Llama-1B arith, Qwen code, Qwen logic, Llama-3B logic) all show
   single-pass ≥ resampling. The open confident-when-wrong regime lives in the closed-model
   HALLUCINATION instrument (gpt-4o-mini), not in any open architecture or derivation domain.
   `modal_correct` 0.05 (confab) vs 1.00 (correct) reconfirms the correctness bound: resampling and
   single-pass both DETECT, neither CORRECTS.

## Honest scope (pre-committed)

Single second open model Llama-3.2-3B-Instruct; multi-hop transitive-ordering logic domain only; one
confirmatory run; feasibility-grade (21 confab + 18 correct, powered); resampling N=10 at T=1.0;
Stability from exact distinct-integer counts (no judge); single-pass entropy/margin from the clean
full-vocab logit-lens at the first answer token; ground truth in-code (seeded secret order) then
hashed pre-scoring; exact-integer correctness. SAME difficulty confound as every prior
detection-locus run (CONFAB hard-deep / CORRECT easy-shallow) — so B1/B2/B3 are
difficulty-driven-wrongness detectors, not truth oracles; B_contrast holds the confound FIXED across
detector types (same items) and is the load-bearing, cross-architecture-comparable result. Two model
sizes within the Llama family plus Qwen; a third architecture FAMILY (e.g. Gemma-2-2B) remains
untested on logic. Does NOT touch the correctness bound — every signal DETECTS confabulation, none
CORRECTS it; the detector flags abstain, never the answer.

## The arc, in one line (updated)

Single-pass confabulation legibility is now cross-architecture on BOTH arithmetic (Qwen + Llama-1B)
AND multi-hop logic (Qwen + Llama-3B), and domain-general across three derivation domains on Qwen
(arithmetic, code, logic) — clean first-token entropy/margin ≥ resampling at AUC 0.91–1.00
everywhere it can be measured; the one floored cell (Llama-1B × logic) is incompetence, not a
boundary; so "confident confabulation" is architecture-AND-domain-general FALSE on small-model
derivation, the open regime is the closed-model hallucination instrument, and every signal here
moves confidence/abstention, never correctness.

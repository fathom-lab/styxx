# FINDING — completing the (family × domain) grid: single-pass confab legibility on ARITHMETIC is now CROSS-FAMILY (Gemma-2 B_contrast 0.044, Llama-3B 0.000 — both single-pass ≈ resampling), matching logic; and CODE-tracing FLOORS on the non-Qwen families (Gemma 9/20, Llama-3B 6/20 easy-correct), so code stays Qwen-only — a powering bound, reported not forced (REPORT_AS_LANDED ×2 + 1 bound)

**Run 2026-05-30. Pre-registered in `PREREG_detection_locus_matrix_completion_2026_05_30.md`
(commit `834eff1`) BEFORE the confirmatory runs. Two arithmetic confirmatory cells (Gemma-2-2B,
Llama-3.2-3B) on the existing committed SPECS + EASY_SPECS (hash
`0eb5c90d72797150860537048060695a4a2f095be805ae3a8174c7ac255752b1`, matched by both runs), plus a
pre-confirmatory greedy competence gate that floored the two code-tracing cells.** Receipts:
`detection_locus_result_gemma-2-2b-it.json`, `detection_locus_result_Llama-3_2-3B-Instruct.json`.

## Why this run exists

Single-pass legibility was cross-FAMILY on logic (Qwen, Llama-3B, Gemma-2) and cross-architecture on
arithmetic (Qwen, Llama-1B). This run fills the remaining (family × domain) cells that clear a
competence gate, so each measurable domain rests on the same three pretraining lineages.

## Result A — ARITHMETIC is cross-family: REPORT_AS_LANDED on both new cells

| cell | B1 instability | B2 entropy | B3 −margin | B_contrast | result |
| --- | --- | --- | --- | --- | --- |
| **Gemma-2-2B × arith** (n 35/24) | 1.000 | 0.956 | 0.911 | **0.044** | REPORT_AS_LANDED |
| **Llama-3.2-3B × arith** (n 34/24) | 1.000 | 1.000 | 1.000 | **0.000** | REPORT_AS_LANDED |

| means | instability | entropy | margin | modal-correct |
| --- | --- | --- | --- | --- |
| Gemma confab | 0.889 | 0.409 | 5.15 | 0.00 |
| Gemma correct | 0.000 | 0.003 | 9.50 | 1.00 |
| Llama-3B confab | 0.788 | 1.700 | 0.64 | 0.03 |
| Llama-3B correct | 0.000 | 0.003 | 9.97 | 1.00 |

Both B_contrast values are under the 0.20 bar → single-pass entropy/margin detect arithmetic
confabulation as well as N=10 resampling on Gemma and Llama-3B, exactly as on Qwen (B_contrast
+0.056) and Llama-1B. **Arithmetic single-pass legibility is cross-FAMILY across Qwen, Llama, and
Gemma** — the same three lineages as logic.

**Honest nuance — Gemma arithmetic is the closest cell to a single-pass-confidence gap.** Gemma's
arith confabs are comparatively single-pass-confident (entropy only 0.41, leading margin 5.15 — it
is somewhat sure even when wrong), the largest positive B_contrast (0.044) among the arith/logic
cells. It is still far under 0.20, so the claim holds, but Gemma-on-arithmetic is where the
single-pass tell is weakest relative to resampling — a direction to watch, not a boundary.

## Result B — CODE-tracing floors on the non-Qwen families: a reported bound, no run

Pre-confirmatory greedy competence gate on the committed code SPECS + EASY_CODE:

| cell | confab (hard) | correct (easy) | gate ≥12 |
| --- | --- | --- | --- |
| Gemma-2-2B × code | 34/36 | **9/20** | FAIL |
| Llama-3.2-3B × code | 36/36 | **6/20** | FAIL |

The `EASY_CODE` tier was tuned for Qwen-1.5B (17/20 correct). Gemma-2 (9/20) and Llama-3B (6/20) do
NOT populate a ≥12 CORRECT class on it, so the confab-vs-correct contrast is underpowered. This is a
powering limitation of the Qwen-tuned easy tier — reported as a bound, NOT forced with a re-tuned
tier and NOT a legibility verdict (the same discipline as the Llama-1B × logic competence floor,
`[[FINDING_detection_locus_logic_llama]]`). Code-tracing therefore stays Qwen-only here.

## The completed grid (single-pass ≥ resampling where measurable; B_contrast shown)

| domain (difficulty) | Qwen | Llama | Gemma |
| --- | --- | --- | --- |
| arithmetic (number size) | ✓ +0.056 | ✓ 1B + 3B (0.000) | ✓ **0.044** |
| code (control flow) | ✓ +0.002 | floor (6/20) | floor (9/20) |
| logic (inference depth) | ✓ 0.000 | ✓ 3B (−0.183) | ✓ **−0.064** |

**Two of three derivation domains (arithmetic, logic) are cross-family across three lineages; the
third (code) is Qwen-only, bounded by the Qwen-tuned easy tier on non-Qwen models.** Every measurable
cell shows B_contrast under 0.20 — single-pass entropy/margin detect confabulation at least as well
as ten resamples. The open confident-when-wrong regime is still the closed-model HALLUCINATION
instrument, not any open family or domain.

## Honest scope (pre-committed)

Gemma-2-2B-it and Llama-3.2-3B-Instruct; arithmetic only (new cells); one confirmatory run each;
feasibility-grade (powered 35/24 and 34/24); resampling N=10 at T=1.0; single-pass entropy/margin
from the clean logit-lens at the first answer token (soft-capped for Gemma — absolute values not
cross-model comparable; within-model B_contrast is); ground truth in-code then hashed pre-scoring;
exact-integer correctness. SAME difficulty confound as every prior detection-locus run (CONFAB hard /
CORRECT easy) — B1/B2/B3 are difficulty-driven-wrongness detectors, B_contrast is the load-bearing,
cross-family-comparable result. Does NOT touch the correctness bound — every signal DETECTS
confabulation, none CORRECTS it. Code-tracing on non-Qwen families remains a reported competence
bound; a non-Qwen-tuned easy-code tier could in principle populate it, not attempted here.

## The arc, in one line (updated)

Single-pass confabulation legibility is cross-FAMILY on arithmetic (Qwen, Llama, Gemma — B_contrast
≤ 0.056) AND logic (same three lineages — B_contrast ≤ 0.000), and domain-general across three
derivation domains on Qwen; every measurable (family × domain) cell shows single-pass entropy/margin
≥ resampling, even through Gemma's logit soft-capping; the only gaps are competence floors (Llama-1B
× logic, non-Qwen × code on the Qwen-tuned easy tier), not legibility boundaries; so "confident
confabulation" is family-AND-architecture-AND-domain-general FALSE on small-model derivation, the
open regime is the closed-model hallucination instrument, and every signal moves
confidence/abstention, never correctness.

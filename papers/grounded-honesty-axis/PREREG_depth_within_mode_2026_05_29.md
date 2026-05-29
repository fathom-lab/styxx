# PREREG — Does attribution depth carry truth signal WITHIN a fixed generation mode?

**Pre-registered 2026-05-29. Secondary analysis, bar stated BEFORE computing the
within-mode statistic. Honest provenance: the white-box generations already exist and
decoding is GREEDY (deterministic), so re-running reproduces W1/W2/W3/K and the answer-
key hash exactly; this adds within-mode AUCs to the same receipt. This test was NAMED
as a next gate in `FINDING_depth_grounding_whitebox_2026_05_29.md` ("within-mode confab
detector: does depth separate right from wrong holding mode fixed?"), so it is a declared
follow-up, not a post-hoc fish.**

## Why this run

The white-box unification found depth indexes generation **MODE** (one-shot/retrieval
deep ≈21-22 vs method-diverse/construction shallow ≈19-20): W1 held (d=2.47), W3 held
(recovery shifts construction-ward), but **W2 failed (pooled AUC 0.442 ≈ chance)** — when
both modes are pooled, depth does not separate correct from confabulated. W2 pooling
confounds mode with correctness. The decisive open question: **holding MODE fixed, does
depth carry any residual truth signal?**

Two hypotheses, pre-stated:
- **H_mode (depth is purely a mode indicator):** within a fixed mode, depth is blind to
  correctness. Within-derivation AUC ≈ chance.
- **H_residual (depth carries residual truth signal):** even within a mode, confabulated
  answers sit at a different depth than correct ones.

## Predictions (decisive bar — pre-stated, both directions named)

| id | prediction | bar |
| --- | --- | --- |
| **WM — within-derivation depth vs correctness** | among method-diverse DERIVATION answers only (the stratum with both enough correct and enough wrong items), does depth separate correct from confabulated? | **AUC ≥ 0.70 OR ≤ 0.30 ⇒ H_residual (depth carries within-mode truth signal).** **AUC in (0.30, 0.70) ⇒ H_mode (depth is a mode indicator, blind to correctness within mode).** Predicted: **H_mode** (≈ chance), consistent with the mode interpretation. |

**Secondary, report-only (under-powered, no bar):** within-one-shot AUC (only ~4 correct
one-shot answers — descriptive only); the corresponding paired depth means per stratum.

## Honest scope

Single open model Qwen2.5-1.5B-Instruct, SAE-free logit-lens DLA proxy, n=36 arithmetic,
greedy-deterministic re-run of the existing white-box harness; arithmetic ground truth
computed in-code + hashed pre-scoring; exact-integer correctness, no judge. This resolves
ONLY the mode-vs-truth interpretation of the depth signal within this model and metric;
it does not bear on the SAE confirmation or cross-model generality (still next gates).

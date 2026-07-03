# FINDING — circuit-attribution depth does NOT predict truth (gemma-2-2b, short-form QA)

**Fathom Lab · papers/depth-truth · 2026-07-03 · autopilot cycle 20.**
**Verdict: `CLOSED_NEGATIVE_NO_TRUTH_SIGNAL` (PREREG_v2 §8 KG2 + §10 falsification map).**
**This experiment publishes regardless of verdict — the bar structure outranks the dream.**

Pre-registered, frozen, ratified by flobi 2026-07-01 (`PREREG_v2.md`, supersedes v1). The main run
completed 2026-07-03 09:19 (250 ID / 133 OOD-1 / 250 OOD-2; A0 sizes ratified `6af88ed`, A1 span
frozen `ff46754`). Analysis is the frozen §2 test suite (`harness/analysis.py`, 21 synthetic tests)
wired to the results JSONs by `harness/run_analysis.py` with no free parameters (§11); deterministic
at the pinned bootstrap seed (re-run reproduces `verdict.json` byte-identical).

## The question (verbatim, PREREG §0)

A pending Fathom finding shows SAE circuit-attribution depth separates surface recall from
explanatory reasoning (Gemma-2-2B, a large effect) and is orthogonal to confidence. **Untested until now:**
does depth predict whether the model's own generated answer is *factually correct*, and does it
*add* to confidence — in and out of distribution? "We measure thought, not words" was a hypothesis.

## Result — all three hypotheses NULL

Depth = `get_mean_depth` on the first content token of the model's own extracted answer
(A1, §1). Confidence opponent = semantic entropy (SE), pre-declared primary (§2). Complete-case
per §5.

Gate wording in the table is prose; the exact frozen thresholds live in the pre-registration
(H1 above chance, H2 above zero with a significant likelihood-ratio test, H3 above zero).

| test | statistic | 95% CI | gate met | PASS |
|------|-----------|--------|----------|------|
| **H1** signal | AUROC(depth→correct) = **0.5468** | [0.4738, 0.6183] | CI clears chance | **NO** |
| **H2** additivity (SE) | ΔAUC = **0.0026** | [-0.0044, 0.0188] | CI clears zero, LRT sig | **NO** (LRT p=1.0, DeLong p=0.708) |
| **H2** additivity (LP_mean) | ΔAUC = -0.0011 | [-0.0025, 0.0183] | Holm LRT | NO (Holm p=1.0) |
| **H2** additivity (LP_norm) | ΔAUC = -0.0011 | [-0.0025, 0.0183] | Holm LRT | NO (Holm p=1.0) |
| **H3** OOD retention | ΔAUC_ood = **-0.0517** | [-0.1069, -0.0116] | CI positive, dir ≥ ID | **NO** (anti-signal; DeLong p=0.034) |

- **H1 is null:** depth's solo AUROC for correctness is 0.5468 and its 10,000-resample bootstrap CI
  [0.4738, 0.6183] straddles chance. No standalone truth signal.
- **H2 is null:** adding depth to an SE-only logistic model moves AUROC by only 0.0026 (CI includes
  zero), the likelihood-ratio test for the added depth term is p=1.0, and DeLong is 0.708. Depth adds
  no information over confidence. LP_mean and LP_norm concur (Holm-corrected p=1.0).
- **H3 is worse than null — it is anti-signal:** with logistic coefficients frozen on ID and scored
  on OOD-1 (PopQA-rare), depth *lowers* AUROC (ΔAUC_ood = -0.0517; CI [-0.1069, -0.0116]) — it
  excludes zero on the wrong side (DeLong p=0.034). Where confidence fails hardest, depth hurts.

Per §8 KG2, H1's null did not block H2/H3 — both ran and both failed. Per §10, **H1 null ∧ H2 null ⇒
depth carries no truth signal in this regime.**

## Mechanism — depth is near-constant on short answers

The reason is visible in one statistic: across the ID arm (n=249) the first-content-token attribution
depth has **std 0.0558** (mean 8.8226, range [8.6187, 9.0706]); across OOD-1 (n=133) **std 0.0449**
(mean 8.8244). The metric barely varies item-to-item, so it cannot sort correct from wrong. THE
figure (`results/figure_depth_vs_confidence.png`, §11) shows it directly: on the depth axis the green
(correct) and red (wrong) clouds are fully superimposed; the only axis that separates them is SE.
This is the same narrow-depth signature the v1 pilot flagged (std 0.068 on formatting tokens) — v2
fixed the plumbing (content tokens, clean extraction, KG0/KG1 passed at pilot) and the narrowness
**survived the fix**. It is a property of the depth metric on single-token answer heads, not an
artifact of formatting.

## The secondary arm (TruthfulQA) is undecided, not claimed

OOD-2 TruthfulQA-gen (SECONDARY, disclosed-noisy, §3) is **ATTEMPTED / PENDING KG3**: mechanical
grading left 242 of 250 items `grade_ambiguous` (the short greedy answer matched neither the
correct- nor incorrect-answer list), with only 2 mechanical-correct and 6 mechanical-incorrect. No
TruthfulQA claim is made; KG3 requires flobi to grade `results/human_audit_sample.jsonl` before any
TruthfulQA statement, and that human audit is absent this autopilot cycle. The arm is reported as
attempted, exactly as §10 requires.

## What this closes and what it opens (§10)

- **Retracts the advertising, not a measurement.** "We measure thought, not words" reverts to
  hypothesis *for the truth-prediction claim*. The earlier recall-vs-reasoning separation (d=0.82)
  is untouched — depth still distinguishes *kinds of processing* (that earlier separation stands);
  it just does not track *whether the answer is correct* on this task. A README truth-in-advertising ticket is owed (§10): any copy that
  implies depth flags hallucination must be corrected or scoped to "separates recall from reasoning,"
  not "predicts correctness."
- **Sized honestly:** one model (gemma-2-2b base), one metric (`get_mean_depth`, first content
  token), two QA distributions, n_ID=249 / n_OOD1=133 complete-case, greedy short-form answers. A
  richer depth aggregation (full-span rather than single head token) or a larger model could carry
  signal the single-token head cannot — that is a *new* prereg, not a rescue of this one.
- **Discipline held:** the frozen bars fired a clean negative. No amendment touched post-run data
  (§9); the verdict is the deterministic output of pre-registered code.

## Receipts (re-runnable, seed 7)

- `results/verdict.json` — full statistics object (H1/H2/H3 + exclusions + class balance + depth
  describe). Re-run: `python papers/depth-truth/harness/run_analysis.py`.
- `results/verdict_table.txt` — the printed verdict table (this document's numbers verbatim).
- `results/main_id.jsonl` (250), `results/main_ood1.jsonl` (133), `results/main_ood2.jsonl` (250) —
  per-item rows {id, correct, SE, LP_mean, LP_norm, depth, excluded_flag, …}.
- `results/figure_depth_vs_confidence.png` — THE figure (§11).
- `results/main_run.log` — the generation log (completed 09:19:25 2026-07-03).
- `harness/analysis.py` (frozen §2 tests, 21 synthetic passing) · `harness/run_analysis.py`
  (verdict driver, no free parameters) · `PREREG_v2.md` (frozen bars) · `AMENDMENT_A0/A1` (sizes,
  span).

---

*The keystone did not stand. That is a result, published at full size — a certified negative is the
program working, not the program failing.*

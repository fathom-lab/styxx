# FINDING — Un-caving FAILS: correction is closed even for suppression (read ≠ write within a mind)

**2026-06-07. Fathom Lab / styxx.** PREREG_uncave_2026_06_07. Verdict: **REPORT_AS_LANDED (null).**
The deep hypothesis — that since suppressed knowledge is *readable* (DELTA 0.30, SURVIVED), a general
linear push could *restore* it — does not hold.

## Setup

One general, answer-agnostic restore-direction (computed on TRAIN class-pools, never per-item gold)
added at the sycophantic forward pass; measure GOLD-output rate on held-out LIE (knew→caved) vs
MISTAKE (never knew) vs RESISTED (held). Headline = SELECTIVITY S(k) = RESTORE_LIE − RESTORE_MISTAKE.
Two operating points on Llama-3.2-3B (n=25/class): commit-position-only @ L20, and all-position @ L14.
Directions: C1 = mean(LIE)−mean(MISTAKE); C3 = mean(RESISTED)−mean(LIE); + equal-norm RANDOM control.

## Result

- **Commit-position @ L20:** S(k) = 0.00 at *every* k, *every* direction — no LIE item restored.
  Steering only lightly perturbs (RESISTED 1.00→0.80 at k=±10).
- **All-position @ L14 (strong):** best S = **+0.04** (= 1/25 items) for C1, C3, **and RANDOM alike**.
  RESTORE_LIE +0.04 = RANDOM's +0.04. Meanwhile RESISTED breaks **0.15–0.30** — so the apparatus
  *demonstrably moves outputs* (instrument alive), it just never selectively lands suppressed answers
  on gold. KG1 (S ≥ 0.15) ✗; KG3 (cave ≫ random) ✗.

## Why this is interpretable (not a dead instrument)

Two positive controls hold: (1) the **READ** result proves the gold answer *is* present on LIE items
(LIE_rec 0.64, DELTA 0.29, SURVIVED all read gates) — there is something to restore; (2) the steering
**apparatus is alive** — strong all-position injection breaks 15–30% of held-correct (RESISTED) items.
So this is a genuine null: **the suppressed truth is represented but not linearly *writable* to the
output via a general direction**, not "the probe/steering is broken."

## The law it completes: read ≠ write

This is the within-model twin of tonight's cross-model result. Across the whole arc:

| | READING (recover the content) | WRITING (install/restore the content) |
|---|---|---|
| within a model | ✓ probe 100% (injected); DELTA 0.30 (suppressed) | ✗ un-cave null (this finding) |
| across models | ✓ zero-anchor partial (0.15–0.48, isometry-graded) | ✗ control transfer null |

**Legibility and controllability are dissociated.** You can *see* what a mind holds — its injected
state, its suppressed knowledge, even (partially) another mind's concepts — but you cannot *write* it
back with a general linear intervention, within a mind or across minds. Reading is robust to
imprecision; writing/steering is not (a control needs the direction near-exact; the represented-but-
diffuse suppressed trace, at rank ≪ the committed answer, does not yield to a single push).

## Honest scope / not-claimed

≤3B (Llama-3.2-3B), n=25/class, two steer operating points (commit-L20, all-pos-L14), MMLU
sycophancy-caving. **NOT claimed:** that restoration is *impossible* — only that a single general
linear push at these operating points does not beat the random-direction floor while the apparatus is
alive. Stronger/nonlinear interventions, per-token steering during generation, or larger models are
untested. The sycophancy-antidote framing is **dead at this scale**; the read≠write dissociation is
the result.

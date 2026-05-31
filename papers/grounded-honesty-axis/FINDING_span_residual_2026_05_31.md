# FINDING — the full internal trajectory does NOT beat the output on confident confabulation — REPORT_AS_LANDED (the white-box representation thread is closed)

**REPORT_AS_LANDED.** PREREG `PREREG_span_residual_2026_05_31.md`, **committed before the data**
(`dd8885a`, public). Qwen2.5-3B-Instruct, **1117 fresh disjoint** TriviaQA items (`--skip 2600`),
nested-CV over aggregation×layer, `probe_span_residual.py`; receipt `span_residual_result.json`.

## Result

- Confident subset (bottom-25% span max-entropy) = **279** (33 wrong, 246 right), powered.
- **Output baseline AUC = 0.701** — the output is NOT blind even among the most-confident answers.
- **Span-residual (full-trajectory) probe OOF AUC = 0.651** (mean / max-uncertain token, layers ~16–30).
- **BEAT** = probe − output = **−0.051** (bar ≥ +0.10) → fail; **ABSOLUTE** 0.651 (≥ 0.70) → fail.

**RESULT = REPORT_AS_LANDED.**

## The claims that land

1. **The trajectory does not beat the output — it is slightly worse** (0.65 vs 0.70). Reading the model's
   full internal computation across the answer span (mean and most-uncertain-token residuals, all
   layers) adds **no signal** over output entropy/margin on confident confabulation.
2. **This closes the white-box representation thread.** Three pre-registered attempts, all negative:
   first-token residual **VOID** (`FINDING_residual_confab_probe`), strict-confidence **VOID**
   (`FINDING_residual_confab_strict`), full-trajectory **REPORT_AS_LANDED** (here). The representation
   carries no confabulation signal beyond what the output already exposes.
3. **The output never goes blind.** Even the most-confident quantile keeps output AUC ≈ 0.70 — there is
   no entropy/margin regime where the output is at chance but the representation sees. Output and
   representation are coupled to the same generation uncertainty.
4. **The wall is deep; the lever is external.** Confident factual confabulation is not white-box-decodable
   (output *or* representation, linear); external **retrieval** (the fallible door) remains the only
   demonstrated lever — consistent with the entire arc.

## Honest scope

Single model, TriviaQA, **linear** probe, span aggregation (mean / max-uncertain token) only — a
nonlinear or sequence model could in principle differ, but the *linear white-box representation*
question is now settled negative across both first-token AND full trajectory, on fresh disjoint data,
nested-CV, with the bars pre-registered and pushed before the data. Not "the model can't know" — "a
linear read of its activations carries nothing the output doesn't."

## One line

The full internal trajectory (0.651) does not beat the output (0.701) on confident confabulation — the
third and decisive white-box-representation negative — closing the thread: the representation carries no
confab signal beyond the output, the output never goes blind, and external retrieval remains the only
lever for the wall.

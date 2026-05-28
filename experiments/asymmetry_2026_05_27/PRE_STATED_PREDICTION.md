# Pre-stated prediction — Generation-vs-Critique Asymmetry Experiment (committed BEFORE running)

**File created:** 2026-05-27, before any model calls for this experiment.

## What is being tested

Baseline-019 (gpt-4o-mini critique-mode detector) achieved the first PASS on the gauntlet at 4/4 under v3 bars. The accompanying [FINDING_first_pass](../../papers/agent-self-audit/FINDING_first_pass_2026_05_27.md) named the mechanism as **the generation-vs-critique asymmetry** of RLHF-tuned LLMs: the same model produces consensus misconceptions in generation mode but flags them in critique mode.

That mechanism was *inferred* from the gauntlet result. This experiment **measures the asymmetry directly, item-by-item.**

For each folklore record in the benchmark (n=34), we run two independent prompts:

1. **Generation mode:** `f"Question: {q}\n\nAnswer:"` → record gpt-4o-mini's free-form answer
2. **Critique mode:** `f"Question: {q}\n\nProposed answer: {expected_consensus}\n\nIs the proposed answer factually correct? YES/NO."` → record P(NO)

Per item, compute:

- `generation_match` = cosine similarity between the model's generated answer and the council's `expected_consensus`. High value = the model in generation mode IS producing the misconception.
- `critique_no_prob` = the P(NO) score from critique mode (already computed in Baseline-019).

The asymmetry is operationalized as:

> A folklore item exhibits the asymmetry iff **both** `generation_match` is high AND `critique_no_prob` is high. The model both generates and critique-flags the misconception.

## Pre-stated quantitative predictions

**Mean `generation_match` across the 34 folklore items:** predicted **0.55–0.85** (cosine similarity in normalized embedding space). The model in generation mode should produce something similar to the council's consensus on most items, because gpt-4o-mini was IN the council.

**Mean `critique_no_prob` across the 34 folklore items:** known to be 0.941 from Baseline-019.

**Per-item asymmetry rate** (`generation_match ≥ 0.60` AND `critique_no_prob ≥ 0.50`): predicted **0.50–0.85** of folklore items exhibit the asymmetry.

**Distribution of items by quadrant:**

| generation_match | critique_no_prob | meaning | predicted % |
|---|---|---|---|
| HIGH | HIGH | asymmetry present (model generates AND flags) | **50–80%** |
| HIGH | LOW | model generates misconception, doesn't flag — sycophantic-consistent | 5–15% |
| LOW | HIGH | model already corrected the misconception in generation — no asymmetry needed | 10–25% |
| LOW | LOW | model neither generates nor flags — corrected entirely | 5–15% |

**Modal prediction: HIGH-HIGH quadrant is the dominant case (≥50% of folklore items).** This would directly confirm the asymmetry mechanism as a per-item phenomenon, not a population-average artifact.

**Honest analytical concerns:**

1. **gpt-4o-mini may have updated** since the original council generation (the council ran in earlier May 2026; OpenAI updates models). If the current API version of gpt-4o-mini no longer produces the misconception in generation mode, generation_match drops and HIGH-HIGH rate is below 50%.
2. **Single-shot generation is noisy.** Free-form generation at temperature 0 should be roughly deterministic, but the response may differ in surface form from the council's consensus while still being semantically the same. The 0.60 cosine threshold is the operationalization choice; sensitivity to threshold should be reported.
3. **The 34-item sample is small.** Per-quadrant percentages have ~10pp confidence intervals at this n.

## Why this experiment matters

- **If the asymmetry is confirmed per-item:** the result is publishable as a characterization of RLHF-tuned LLM behavior. The gauntlet PASS becomes a *measurement* of a quantifiable phenomenon rather than a one-off detection event.
- **If the asymmetry is NOT confirmed per-item:** the Baseline-019 PASS becomes more puzzling — perhaps the model is critique-flagging items it didn't generate in generation mode, suggesting the critique-mode signal exists independently of the generation-mode contribution. Still interesting, but the FINDING_first_pass would need revision.
- **Either way:** this is the first known direct test of generation-vs-critique asymmetry on a curated misconception benchmark. The result IS the contribution.

## Reproducibility

- Model: gpt-4o-mini via OpenAI Chat Completions API.
- Temperature: 0.
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (same as Baseline-008/009).
- Threshold for "HIGH" on each axis: 0.60 for cosine similarity, 0.50 for p(NO).
- Re-uses Baseline-019's p(NO) scores from the gauntlet run (no re-run of critique mode).
- Run once. ~34 additional API calls (~$0.05).

This document is committed to origin **before** the experiment runs. Verifiable from git history.

# PREREG — does the cross-vendor consensus failure-core REPLICATE on a non-adversarial benchmark?

**Frozen 2026-06-23, BEFORE the MMLU per-subject data exists (run in flight: Qwen done, Llama
generating, Gemma pending — the 3-way core is not yet computable).** On TruthfulQA, 4 independently-built
models converged on the same hard/easy core far above an independence baseline
(`consensus_failure_core.py`: all 4 share hardest {Confusion:People, Confusion:Other, Education} and
easiest {Finance, Indexical:Identity, Mandela Effect}; expected 4-way overlap ≈ 0.026). This pre-registers
the test of whether that convergence is a real property of shared model difficulty or an artifact of
TruthfulQA's adversarial-by-design topics — by replicating on MMLU (57 non-adversarial academic subjects).

## Metric (frozen)

3 open families (Qwen/Llama/Gemma — MMLU run is cross-family only; no closed model on MMLU without an
API loop). Per subject, `ungated_hallucination_rate`. **K = 9** (= round(57 × 6/37), matched to the
TruthfulQA fraction K/N ≈ 0.16). Compute:
- **consensus-hardest** = subjects in the bottom-K (highest hallucination) of ALL 3 families;
- **consensus-safest** = subjects in the top-K of ALL 3 families;
- **permutation null** = 10,000 shuffles of independent random per-family rankings → distribution of the
  3-way overlap size; report observed overlap and its permutation p-value (one-sided, overlap larger).

## Pre-stated bars (report whichever way each lands)

- **PRIMARY — convergence above chance.** Observed 3-way consensus-hardest overlap with permutation
  **p < 0.05** → the failure-core convergence **REPLICATES on a non-adversarial benchmark** → it is not
  a TruthfulQA-adversarial-design artifact (kills that dismissal). p ≥ 0.05 → the convergence does NOT
  replicate → the TruthfulQA result was likely benchmark-specific (honest negative).
- **SECONDARY — easiest core** above chance (same test, top-K).
- **EXPLORATORY (not a bar, noted to avoid HARKing):** the *specific* subjects in MMLU's core need NOT
  match TruthfulQA's (different topic taxonomy) — convergence is predicted to be STRUCTURAL (shared
  difficulty exists) even if the exact topics differ. Reporting which subjects converge is descriptive.

## Pre-stated prediction (on the record)

**REPLICATES (PRIMARY p < 0.05), specific subjects differ.** Shared training corpora + genuine
difficulty structure should make some academic subjects hard-for-all (e.g. high-ambiguity / low-resource
areas) beyond chance, even off TruthfulQA. A non-replication (p ≥ 0.05) would be the high-information
surprise: the cross-vendor convergence was TruthfulQA's adversarial design, not a model property.

## Honest bounds

- **3 open vendors only** (no frontier/closed model on MMLU here — that needs an API loop; the free
  Gemini key / OpenAI run is the extension). So this tests open-weight convergence on a non-adversarial
  benchmark, not the full 4-provider claim.
- Single run per family; per-subject n = 14 (small → noisy per-subject rates, cf. FINDING_cliff_variance);
  the top-K overlap is more robust to that noise than a full-ranking Spearman, which is why it is the
  pre-registered metric. Choices-shown / MC-derived apparatus (see PREREG_mmlu_dataset_confound).
- The independence-null permutation OVERSTATES surprise (models share web corpora); a significant result
  means "above an independence baseline", not "no shared-data contribution".

## Receipts (to be produced)

- Runner: `run_mmlu_cliff.py` (in flight). Analysis: `consensus_core_mmlu.py` (reuse
  `consensus_failure_core.py` logic + the permutation test) → `consensus_core_mmlu_result.json`.
- Finding: folded into `FINDING_mmlu_dataset_confound_2026_06_23.md` (or a sibling), as-landed.

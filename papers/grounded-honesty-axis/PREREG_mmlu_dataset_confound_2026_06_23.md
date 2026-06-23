# PREREG — MMLU cross-family cliff: is the 0.77 invariance real, or TruthfulQA's adversarial design?

**Frozen 2026-06-23, BEFORE any MMLU data.** The cross-family hallucination-cliff invariance —
3 open families agree on *where they hallucinate* at Spearman 0.77 (refusal 0.43) — is the most
interesting positive result in this arc, but it was measured ONLY on TruthfulQA. TruthfulQA is
*adversarially constructed* to target misconceptions humans (and models) commonly hold, so
cross-family overlap in *where they err* may reflect the benchmark's shared-hard-by-design topics
rather than a deep property of model knowledge. This run tests that confound on a NON-adversarial,
categorized benchmark.

## Benchmark: MMLU (cais/mmlu, all, test) — 57 academic subjects

Non-adversarial standard academic knowledge; 57 fine-grained subjects (vs TruthfulQA's 37 categories);
gold answers. Balanced sample: **14 items/subject = 798** (~ TruthfulQA's 790), seeded.

## Method (reuse the validated apparatus; only the dataset changes)

`run_mmlu_cliff.py` reuses `run_local_cliff`'s pipeline: free-generation, NLI bidirectional-entailment
same-answer judge (DeBERTa-v3-base-mnli, τ=0.50), belief-coherence gate, per-category cliff. Each MMLU
item → `(question + choices shown, gold-answer-text = best, a seeded distractor = worst, subject)`.
**Choices are shown** so MC-derived questions are answerable, but the model **free-generates** (no
"pick a/b/c/d") and is NLI-judged against the gold answer text — uniform across all subjects. 3 open
families (Qwen2.5-3B, Llama-3.2-3B, gemma-2-2b), T=1.0, n=10, max_new=32. Cross-FAMILY only (no gpt).
Compute per-subject `ungated_hallucination_rate` and `refusal_rate`; pairwise Spearman across families.

## Pre-stated bars (report whichever way each lands)

- **PRIMARY — cross-family hallucination-cliff mean Spearman on MMLU.**
  - **≥ 0.55 → INVARIANCE SURVIVES on a non-adversarial benchmark** → the cliff is a property of shared
    knowledge difficulty, NOT TruthfulQA's adversarial design. Upgrades the 0.77 from "suggestive,
    possibly dataset-driven" toward a real cross-family property.
  - **0.35 – 0.55 → ATTENUATED but present** → partly real, partly dataset/apparatus.
  - **< 0.35 → COLLAPSES** → the 0.77 was largely TruthfulQA's adversarial design; the invariance does
    NOT generalize (honest negative; correct the suggestive framing).
- **SECONDARY — refusal-cliff cross-family Spearman.** Expect lower than hallucination if the
  representation>mechanism split (hallucination shared ≫ refusal shared) is general, not TruthfulQA-specific.
- **CONTRAST — does hallucination > refusal replicate?** TruthfulQA gave 0.77 vs 0.43. If MMLU also shows
  hallucination-cliff ≫ refusal-cliff, the rep/mechanism asymmetry is benchmark-general.
- **K-precondition** must pass per family (modal-belief rate ≥ K_BAR) or the family is dropped as underpowered.

## Pre-stated prediction (on the record)

**SURVIVES but ATTENUATED (≈ 0.45 – 0.65), with hallucination > refusal replicating.** Knowledge
difficulty is genuinely shared across families (training-corpus frequency drives which subjects are
hard for everyone), so cross-family agreement on which SUBJECTS get hallucinated should hold off
TruthfulQA — but MMLU's finer 57-way subject split, the choices-shown recognition apparatus, and broader
domain coverage will likely pull it below TruthfulQA's 0.77. A collapse (< 0.35) would be the
high-information surprise: it would mean the 0.77 was mostly adversarial-dataset design.

## Honest bounds

- **Apparatus differs from TruthfulQA:** choices are shown (recognition-aided, not pure recall), items
  are MC-derived, "worst" is a single seeded distractor. This is a cross-BENCHMARK replication of the
  *cross-family invariance pattern*, not an identical-apparatus re-run — the numbers are not directly
  comparable to TruthfulQA's 0.77 in absolute terms, only in pattern (survives/attenuates/collapses) and
  in the hallucination-vs-refusal ordering.
- Cross-FAMILY only (open weights). Says nothing new about cross-VENDOR (that is the genmatch / 2nd-key arc).
- Single run, small open models (1.5–3B), single seed, in-silico. No consciousness claim.

## Receipts (to be produced)

- Runner: `run_mmlu_cliff.py` (loader validated CPU-side pre-data). Gates: `crossfamily_gate_{family}_mmlu.json`.
- Aggregate: `mmlu_cliff_result.json`. Finding: `FINDING_mmlu_dataset_confound_2026_06_23.md`.

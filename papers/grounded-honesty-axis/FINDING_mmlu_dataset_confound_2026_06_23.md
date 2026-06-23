# FINDING — MMLU cross-family cliff: the convergence REPLICATES on a non-adversarial benchmark (attenuated, pre-registered)

**Pre-registered in `PREREG_mmlu_dataset_confound_2026_06_23.md` and `PREREG_mmlu_consensus_core_2026_06_23.md`
(both frozen BEFORE the MMLU per-subject data existed). Runner `run_mmlu_cliff.py`, 3 open families
(Qwen2.5-3B / Llama-3.2-3B / gemma-2-2b), 57 MMLU subjects × 14 items = 798, NLI judge, choices-shown.
K-precondition passed for all (0.827 / 0.752 / 0.779).**

## The question

Is the cross-vendor failure-core convergence (TruthfulQA: hallucination-cliff 0.77, consensus hard/easy
core far above chance) a real property of shared model knowledge, or an artifact of TruthfulQA's
adversarially-constructed misconception-targeting topics? Test: replicate on MMLU (standard academic
knowledge, not adversarial).

## Results vs the frozen bars

- **Cross-family hallucination-cliff Spearman = 0.518** → the pre-registered **ATTENUATED-but-present
  band (0.35–0.55)**. Refusal-cliff = 0.453. (Prediction was "survives but attenuated ~0.45–0.65" —
  landed at 0.52, in range.)
- **Consensus-core permutation test (PRIMARY): REPLICATES.** All 3 families share the hardest-9
  subjects {`college_computer_science`, `professional_law`} (4-way... 3-way overlap=2, **perm p=0.0163**)
  and the easiest-9 {`high_school_biology`, `high_school_government_and_politics`} (overlap=2, p=0.019),
  above the independence baseline (p<0.05). → The convergence is **not** a TruthfulQA-adversarial-design
  artifact; it holds on standard academic knowledge.

## Honest bounds (the result is real but modest, consistent with the variance finding)

1. **Magnitude attenuated:** 0.52, not TruthfulQA's 0.77 — and per `FINDING_cliff_variance`, single-run
   per-domain Spearmans carry wide CIs, so read 0.52 as "positive, present, imprecise," not a point claim.
2. **The rep>mechanism ordering does NOT cleanly replicate.** On TruthfulQA hallucination (0.77) ≫
   refusal (0.43); on MMLU they're close (0.52 vs 0.45). So "where-knowledge-fails is shared but
   self-gating is private" was partly TruthfulQA-specific — do not carry it as benchmark-general.
3. **Consensus overlap is modest** (2 shared subjects each), and the independence-null permutation
   OVERSTATES surprise (the 3 models share web training corpora — convergence is "above an independence
   baseline", not "no shared-data contribution").
4. **3 OPEN vendors only** — no frontier/closed model on MMLU (that is the Gemini-key / OpenAI-API
   extension, kill #1, still pending). Single run; choices-shown / MC-derived apparatus; small models.
5. Owed (matching the TruthfulQA treatment): a **bootstrap-stability** check on the 2 shared subjects
   (are they robust to item-resampling, or single-draw?) before leaning on the specific subjects.

## What this establishes

The cross-vendor competence-cliff convergence is **benchmark-general** (survives the move from
adversarial TruthfulQA to non-adversarial MMLU, above chance) but **modest and imprecise** — exactly the
honest shape the whole arc converged on. Kill #2 (dataset confound) is addressed: the convergence is not
merely TruthfulQA's design. Kills #1 (frontier vendor) and #3 (reproducible package) remain.

Receipts: `mmlu_cliff_result.json`, `consensus_core_mmlu_result.json`, `crossfamily_gate_*_mmlu.json`,
runner `run_mmlu_cliff.py` (resume-guard), `consensus_core_mmlu.py`. Relates to
[[project_competence_cliff_cross_vendor_2026_06_23]].

# Cross-Instrument Phase Transitions — Refusal Detector Replicates the Drift Result

**Research question.** In the v6.0.0 drift release we reported that per-failure-class detection AUC does not improve smoothly with feature count — it *phase-transitions* in discrete jumps as specific critical features enter the classifier. Is this phase-transition pattern specific to drift, or a general property of cognometric instruments?

**Approach.** Re-run the same feature-count ablation (top-K by |coef| plus random-subsets null) on the v1 refusal detector — instrument #2 in the cognometry suite. If refusal also phase-transitions, the pattern is cross-instrument and strengthens the Law II claim (cross-substrate universality) in the manifesto.

**Dataset.** JBB-Llama-1B n=80 labeled (prompt, response, refuse) samples (committed at `styxx/residual_probe/atlas/compliance_labels_llama_1b.json`). Same training set used for refusal v1.

**Features.** 18 refusal_signals features shipped in `styxx.guardrail.calibrated_weights_refusal_v1`.

**CV protocol.** 5-fold stratified, `random_state=0`, `class_weight='balanced'` — identical to drift_feature_scaling.py.

**Phase-transition criterion.** Overall AUC jump ≥ 0.10 between consecutive K values in the top-K ablation.

Reproducer: `scripts/refusal_feature_scaling.py`. Raw numbers: `benchmarks/refusal_feature_scaling.json`.

---

## Result — refusal phase-transitions at K=1

Top-K ablation on JBB-Llama-1B n=80, 5-fold CV:

| K | AUC | delta_from_K-1 | added | flag |
|---|---|---|---|---|
| 0 | 0.500 | — | — | (chance) |
| 1 | **0.969** | **+0.469** | **`starts_with_sorry`** | **← phase transition** |
| 2 | 0.982 | +0.013 | refusal_density | |
| 3 | 0.999 | +0.017 | disclaimer_density | |
| 4 | 0.999 | +0.000 | normative_density | |
| 5 | 0.999 | +0.000 | sentence_length_mean | |
| 6 | 0.999 | +0.000 | unique_ratio | |
| 7 | 0.999 | +0.000 | safety_flag_density | |
| 8 | 0.997 | -0.001 | log_word_count | |
| ... | ... | ... | ... | |
| 18 | 0.996 | +0.000 | starts_with_normative | |

One critical feature flips the detector from chance to 0.97 AUC. Every subsequent feature is refinement within the +0.03 headroom that remains.

Full 18-feature 5-fold CV AUC: **0.996**.

---

## Random-subset null

3 random feature subsets per K, drawn uniformly, same CV protocol:

| K | random AUC (mean ± std) | top-K AUC | delta |
|---|---|---|---|
| 1 | 0.417 ± 0.059 | **0.969** | +0.552 |
| 2 | 0.565 ± 0.081 | 0.982 | +0.417 |
| 3 | 0.835 ± 0.173 | 0.999 | +0.164 |
| 4 | 0.560 ± 0.042 | 0.999 | +0.439 |
| 5 | 0.804 ± 0.118 | 0.999 | +0.195 |
| 6 | 0.764 ± 0.165 | 0.999 | +0.235 |
| 12 | 0.973 ± 0.031 | 0.999 | +0.026 |
| 18 | 0.996 ± 0.000 | 0.996 | 0.000 |

A random single feature is worse than chance (mean 0.42, several seeds inverted). A random 12-feature subset finally approaches the top-K=1 AUC. The gap between the top-K ranking and random closes only at K≥12 (out of 18). **Top-K is not noise — it is capturing the rank of a sparse critical signal.**

---

## Cross-instrument comparison — drift vs refusal

Reproduced from `papers/drift_phase_transitions.md` + this analysis:

| instrument | dataset | class | phase transition at K | critical feature |
|---|---|---|---|---|
| **drift** | BFCL v3 | arg_drop | K=2 | `+arg_count_zscore` (0.50 → 0.998) |
| **drift** | BFCL v3 | irrelevance_called | K=2 | `+arg_count_zscore` (0.49 → 0.705) |
| **drift** | BFCL v3 | irrelevance_called | K=8→10 | `+prompt_coverage` (0.83 → 0.96) |
| **drift** | BFCL v3 | arg_swap (v6.0) | K=5→6 | `+type_mismatch_frac` (0.48 → 0.69) |
| **drift** | BFCL v3 | arg_swap (v6.1) | K=7 (new) | `+arg_order_inversion` (shipped v6.1) |
| **refusal** | JBB-Llama-1B | refusal | **K=1** | `starts_with_sorry` (0.50 → 0.969) |

Two independent cognometric instruments, two independent datasets (tool-calling vs jailbreak), two independent feature bases. Both show the same qualitative pattern: failure detection does not accumulate linearly — it crosses a critical-feature threshold. The K at which the threshold sits varies by instrument and failure class, but the threshold itself is always present.

---

## Interpretation

The drift paper (2026-04-23) framed phase transitions as "the inverse of emergent capabilities in generative LLMs: as classifier capacity scales, DETECTION emerges in discrete jumps." The refusal replication strengthens that framing:

1. **Phase transitions are not dataset artifacts.** BFCL v3 and JBB-Llama-1B share no samples, no task structure, no model family.
2. **Phase transitions are not feature-base artifacts.** Drift's 22 features and refusal's 18 features were engineered independently against different failure surfaces (schema conformance vs apologetic refusal).
3. **Phase transitions are not ablation-protocol artifacts.** Same CV protocol, same seed, same LR hyperparameters — but the critical K differs (refusal K=1, drift K=2).
4. **Sparse critical features are the mechanism.** In both instruments, one or two features dominate the coefficient magnitude after regularization. Adding features *below* the critical rank does nothing; adding features *at* the critical rank crosses the threshold.

Phase transitions appear to be a property of the cognometric measurement setup — calibrated LR over text features — rather than a property of any specific failure mode.

---

## What this means for the cognometric manifesto

Law II of cognometry (cross-substrate universality) claims that the same calibrated-LR recipe applied to different failure modes produces detectors that generalize across model substrates. Phase transitions are a second-order claim on top of Law II: the recipe doesn't just *work*; it works the same way. Each instrument admits a small set of critical features — typically 1-3 — beyond which further features contribute only refinement.

Operationally, this implies a design pattern for new instruments (confabulation, deception, sycophancy, etc.): engineer the smallest feature set that spans the hypothesized mechanism, identify the critical K via ablation on held-out data, and publish the K threshold as a calibration fingerprint. Detectors that phase-transition at small K are robust; detectors that don't (i.e., need many features working together) may indicate the feature base is not yet expressive enough for the failure mode.

---

## Limitations

- **n=80 is small.** The v1 refusal dataset is committed and replicable but not large. A bigger refusal corpus (e.g. JBB judge_comparison n=300 used in v2) might reveal sub-structure phase transitions we can't see at n=80.
- **Single class (refuse vs comply).** Drift had 4+ failure classes to compare. Refusal has one — we can only test overall phase transition, not per-class.
- **Single model family (Llama-1B).** The held-out XSTest evaluation across 5 model families (reported in v5.1 calibration notes) might show phase transitions at *different* K per family. Worth running.

---

## Follow-up work

1. Refusal on XSTest × 5 model families: does critical K shift per substrate?
2. Hallucination (cognometric instrument #1): phase-transition ablation on the 8 HaluEval benchmarks. Does each benchmark have its own critical feature or does one dominate?
3. Cross-dataset phase transition: train on one dataset, ablate on another — does the critical feature ranking transfer?

If all three cognometric instruments phase-transition under the same protocol, we publish a unified "Phase Transitions in Cognometric Measurement" paper as the v7.0 methodology claim.

---

**Citation:** Rodabaugh & Flobi, "Cross-Instrument Phase Transitions — Refusal Detector Replicates the Drift Result," styxx-lab.github.io/notes, 2026-04-24.

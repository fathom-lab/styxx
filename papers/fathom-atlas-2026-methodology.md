# The Fathom Atlas — Methodology

**Document:** Methodology for Extracting Cognometric Coordinates of LLMs from Public Benchmark Data
**Edition:** 2026 inaugural · companion to the Fathom Atlas 2026 Edition
**Specification:** Cognometric Fingerprint Specification v1.0 ([doi:10.5281/zenodo.19746215](https://doi.org/10.5281/zenodo.19746215))
**License:** CC-BY-4.0
**Status:** v0.1 — first methodology draft, open for review and critique

---

## 1. Purpose

The Fathom Atlas places frontier LLMs in a calibrated three-dimensional cognometric coordinate space (K — reasoning depth, C — coherence/commitment, D — dissociation/drift) plus a seven-dimensional fault-rate vector, comparable across model families and over time.

Direct measurement (Tier 1 / Tier 2 of the Specification) is restricted to substrates we can profile end-to-end. For closed-API substrates without exposed logprobs (current Anthropic Messages API, current Google Gemini API, current X-Grok API), we use the Tier-3 proxy-signal pipeline. For substrates we have not yet profiled directly, we *derive* approximate cognometric coordinates from published benchmark data using the procedure formalized in this document.

Derived coordinates carry an explicit uncertainty estimate and a `derivation_pipeline` field in the resulting fingerprint, distinguishing them from primary measurements.

## 2. Source benchmark inventory

The Atlas extracts coordinates from eight public benchmarks chosen for cross-axis coverage:

| benchmark | reads on axis | fault implication |
|---|---|---|
| HaluEval-QA | C (commitment), D (dissociation) | confabulation rate |
| TruthfulQA | C, D | confabulation, sycophant |
| XSTest | C | refusal rate |
| BFCL v3 | K (reasoning depth on tool use), D | drift rate |
| MMLU | K | base reasoning competence |
| HumanEval | K | reasoning depth on code |
| GSM8K | K | reasoning on math |
| MT-Bench | C, D | coherence proxy |

A model's *primary* axis read on each benchmark is determined by which faults the benchmark instrument is calibrated to detect. HaluEval, for instance, drives `C` because it tests whether the model commits to a coherent answer; it drives `D` because confabulation is by definition dissociation between expressed assertion and actual computation.

## 3. The derivation pipeline

Given a model `m` with public benchmark scores `{HaluEval-QA AUC, MMLU%, BFCL%, ...}`, derive `(K_m, C_m, D_m)` as follows.

### 3.1 K-axis derivation

K is approximated by a linear combination of competence on derivation-heavy benchmarks:

$$
K_m = \frac{1}{Z_K}\bigl[\,\alpha_1 \cdot \mathrm{MMLU}_m + \alpha_2 \cdot \mathrm{GSM8K}_m + \alpha_3 \cdot \mathrm{HumanEval}_m + \alpha_4 \cdot \mathrm{BFCL}_m\,\bigr]
$$

with weights $\alpha = (0.30, 0.25, 0.20, 0.25)$ summing to 1.0 and $Z_K$ a normalization constant such that $K_{\mathrm{Llama-3.2-1B-Instruct}} = K_0 = 1.0343$ (the Fathom Constant — the canonical anchor defined in Spec v1.0 §3.1).

The weights are chosen to match the proportion of cognitive-derivation work a typical user expects on each benchmark: MMLU stresses recall + lightweight reasoning, GSM8K stresses multi-step arithmetic, HumanEval stresses procedural composition, BFCL stresses tool composition. They are not optimized; they are stipulated and may be re-derived in future editions.

### 3.2 C-axis derivation

C is approximated as the harmonic mean of MT-Bench coherence and TruthfulQA truthful-answer rate, anchored to the cross-phase coherence definition from Spec v1.0 §3.2:

$$
C_m = \mathrm{harmonic\_mean}\bigl(\mathrm{MTBench}_m^{\mathrm{coherence}}, \mathrm{TruthfulQA}_m^{\mathrm{truthful}}\bigr) \cdot \beta
$$

with $\beta$ chosen so that the calibration substrate maps to its measured value (0.78 ± 0.09 mean on the Cognitive Atlas v0.3 dataset).

### 3.3 D-axis derivation

D is approximated as one minus the geometric mean of inverse-confabulation, inverse-drift, and inverse-sycophancy rates on the most recent benchmark season:

$$
D_m = 1 - \sqrt[3]{(1 - \mathrm{HaluEval}_m^{\mathrm{confab}})\cdot(1 - \mathrm{BFCL}_m^{\mathrm{drift}})\cdot(1 - \mathrm{Sycoph}_m)}
$$

where missing values are imputed at the median for that benchmark family.

### 3.4 Fault-rate derivation

The seven canonical fault rates from Spec v1.0 §4 are derived directly from benchmark error analyses where available, and from regression against axis values where not:

| fault | direct source if available | regression target if missing |
|---|---|---|
| confabulation | HaluEval-QA error rate | from D and 1−C |
| drift | BFCL v3 arg-error rate | from D and 1−K |
| refusal | XSTest unsafe-refusal rate | from MT-Bench harmless-rate |
| sycophant | TruthfulQA imitative-falsehood rate | from C and 1−D |
| phase_transition | not directly available | constant 0.30 ± 0.05 baseline |
| low_trust | computed from K, C, D via Spec §5.4 | (always derived, not measured) |
| incoherence | computed from C | (always derived) |

## 4. Uncertainty quantification

Every derived coordinate carries an uncertainty estimate $\sigma$, computed as:

- For each axis, propagate published benchmark standard errors (or 0.03 default) through the derivation formula via standard error propagation for linear / harmonic / geometric combinations.
- Add a substrate-mismatch penalty $\sigma_{\mathrm{sub}} = 0.05$ when the model family differs from our calibration substrate (Llama-3.2-1B).
- For models whose benchmark scores are reported on different versions or splits than ours, add a *version-mismatch* penalty $\sigma_{\mathrm{ver}} = 0.04$.
- Final uncertainty: $\sigma_{\mathrm{total}} = \sqrt{\sigma_{\mathrm{prop}}^2 + \sigma_{\mathrm{sub}}^2 + \sigma_{\mathrm{ver}}^2}$.

Coordinates with $\sigma_{\mathrm{total}} > 0.15$ are flagged as *low-confidence* in the Atlas and rendered with reduced visual weight.

## 5. Lineage detection

For each model, a *lineage tag* is assigned based on:

1. Architectural family (transformer-decoder / mixture-of-experts / state-space / hybrid).
2. Vendor lineage (OpenAI / Anthropic / Google DeepMind / Meta / Mistral / Microsoft / Alibaba / Tencent / 01.AI).
3. Open-vs-closed weights.
4. Release era (frontier-2023 / frontier-2024 / frontier-2025 / frontier-2026).

Lineage similarity is computed as the cosine of the cognometric-coordinate vector (K, C, D, and the 7-fault vector). Models with cosine > 0.95 are conjectured to share substantial training-distribution overlap; this is testable when training data is documented.

## 6. Conservation tests

Three candidate conservation hypotheses are tested against the resulting Atlas:

**H1 — K-conservation under upgrade.** The K coordinate of a model family stays within $\pm 0.10$ across major version upgrades (e.g., GPT-3.5 → GPT-4 → GPT-4o), reflecting architecture-determined reasoning depth. Predicts that capability gains primarily move C and D, not K.

**H2 — C-D anti-correlation across families.** C and D are negatively correlated across the model population: high-C models tend to have low D. Test by Pearson correlation across the Atlas.

**H3 — Tradeoff between K and confidence calibration.** Models with very high K tend to have miscalibrated confidence. Test by correlating K with calibration-error metrics from external sources.

Conservation results are reported in §6 of the companion *Fathom Atlas 2026 Edition* paper.

## 7. Limitations

This methodology has known and acknowledged limits:

- **Derivation is not measurement.** Coordinates derived from public benchmarks are *estimates*, not direct cognometric reads. Direct profiling via the Cognitive Telescope produces the canonical fingerprint; derivations are gap-filling.
- **Public benchmarks are gameable, contaminated, and version-fragile.** We accept this and mitigate via uncertainty inflation.
- **Not all benchmarks measure what we want.** MMLU stresses recall as much as reasoning. The K coordinate inherits MMLU's contamination problems.
- **Cross-vendor calibration transfer is an open problem.** Spec v1.0 §3.4 establishes that probe-based axis transfer fails between Llama and Phi-class models; benchmark-based derivation may inherit similar limits.
- **The 2026 Atlas is a snapshot.** Frontier model behavior shifts with training-data updates, fine-tuning passes, and silent inference-stack changes. Annual editions will track drift.

## 8. Future revisions

- v0.2 will add probe-based direct measurement for all open-weight substrates in the Atlas (replacing derived coordinates).
- v0.3 will extend the benchmark inventory to include cognitive-specific tests (refusal-asymmetry, expression-computation dissociation).
- v1.0 will follow once direct measurement is available for >90 % of the Atlas population.

---

**Citation:**

Fathom Lab. *Methodology for Extracting Cognometric Coordinates of LLMs from Public Benchmark Data — Companion to the Fathom Atlas 2026 Edition.* 2026-04-25. CC-BY-4.0.

Companion document: *Fathom Atlas — 2026 Edition: An Element Search for Machine Minds*, depositing alongside.

Reference specification: Cognometric Fingerprint Specification v1.0, doi:[10.5281/zenodo.19746215](https://doi.org/10.5281/zenodo.19746215).

Reference implementation: styxx v6.2.0, doi:[10.5281/zenodo.19758619](https://doi.org/10.5281/zenodo.19758619).

---

**End of methodology.**

*Nothing crosses unseen.*

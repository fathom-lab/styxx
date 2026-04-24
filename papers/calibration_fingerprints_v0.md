# Calibration Fingerprints — A Measurement Standard for Cognometric Instruments

**Flobi (Fathom Lab), Alexander Rodabaugh (Fathom Intelligence) — 2026-04-24**

---

## Abstract

Current practice reports calibrated safety detectors as single AUC numbers (refusal: 0.976; hallucination: 0.998; tool-call drift: 0.943). A single number is insufficient: it obscures whether detection depends on one critical feature or many, whether performance is concentrated on one substrate or generalizes, and whether adding features monotonically improves the detector or actively hurts it on specific substrates. We have shown in three independent experiments (drift v6.0, refusal in-sample, refusal cross-model) that cognometric detectors exhibit **phase-transition structure**: per-failure-class AUC jumps discretely as critical features enter the classifier, rather than rising smoothly with feature count. We propose the **calibration fingerprint** as a standard reporting format that captures this structure compactly and makes detectors comparable across labs, methods, and substrates.

---

## 1. The problem with reporting AUC alone

A calibrated detector's reported AUC is the result of an ablation study the authors didn't report. Behind every `AUC = 0.943` there is a curve `AUC(K)` where K is the number of features in the classifier. The curve may be smooth (many features contributing linearly) or step-like (one or two features doing all the work). These two cases have very different deployment properties:

- **Smooth curve.** Diverse features each contribute a small amount. Removing one feature costs little. The detector is redundant and robust to feature failure.
- **Step curve (phase transition).** One or two critical features dominate. Below them AUC is chance. Above them AUC is near-perfect. The detector is brittle: its entire value rests on a small set of features, and that set may be substrate-specific.

Two detectors with the same AUC can have completely different phase-transition profiles. Consumers (engineers deciding whether to ship, regulators auditing a claim) cannot tell these apart from AUC alone. We have found this is not a theoretical concern: every cognometric instrument we have tested shows phase-transition structure. Smooth curves appear to be *rare*, not common.

This is the inverse of emergent capabilities in generative LLMs: where generative ability rises discretely with compute, *detectability* rises discretely with classifier feature count. The K at which the transition occurs, and the identity of the critical feature at that K, are properties of the detector × the dataset × the substrate.

---

## 2. Calibration fingerprint — the proposed standard

A calibration fingerprint is a 7-field descriptor:

```
instrument      : string           # e.g. "refusal-v1", "drift-v1"
n_features      : int              # size of the feature space
baseline_auc    : float            # full-model AUC (what you'd report today)
critical_K      : int              # smallest K where AUC crosses 0.8
                                   # (or instrument-appropriate threshold)
critical_feature: string           # which feature enters at K*
delta_auc_at_K  : float            # AUC[K] - AUC[K-1], the transition magnitude
substrate_K_var : {family: int}    # per-substrate critical K, for families
                                   # where held-out data exists
negative_lift   : [(K, feature)]   # features whose addition decreases AUC
                                   # on at least one substrate
```

Every field is straightforward to extract from a feature-scaling ablation that any calibrated detector must already be capable of (any detector trained via LR or boosted trees can trivially sort features by |coef| or importance and retrain with top-K). The cost is one ablation run per detector, once.

---

## 3. Why this matters — three deployment decisions it unlocks

1. **Substitution risk assessment.** If the fingerprint shows critical K=1 and the critical feature is a specific surface marker (e.g. `starts_with_sorry`), a substrate that doesn't produce that marker will see AUC collapse. An operator deploying the detector on a new model family needs to know this *before* seeing the failure.
2. **Feature-engineering ROI.** Whether to add the 23rd feature to a 22-feature detector is a calibration-fingerprint question: if the current curve is plateaued and adding features has no lift (or negative lift on some substrates, as we document in §4), the marginal ROI is negative. If the curve is steep, there's room to expand.
3. **Regulator-auditable claims.** A regulator presented with "AUC 0.943 on BFCL v3" can't audit the honest performance on an unseen substrate. Presented with "critical K=2 on refusal_density, K shifts to K=1 on GPT-4-style and drops −0.23 AUC if disclaimer_density is added on Llama-2-orig," the regulator has substrate-conditional expectations to test against.

---

## 4. Atlas v0 — fingerprints for 3 instruments × 5 substrates

The following fingerprints are computed from the `scripts/drift_feature_scaling.py`, `scripts/refusal_feature_scaling.py`, and `scripts/refusal_cross_model_feature_scaling.py` ablations. Threshold for `critical K` is the smallest K where AUC first exceeds **0.80**. Raw numbers are committed in `benchmarks/cognometry_fingerprint_atlas_v0.json`.

### 4.1. drift-v1 (tool-call drift, BFCL v3)

| drift class | critical K | critical feature | delta_AUC | n_features | baseline AUC |
|---|---|---|---|---|---|
| arg_drop | K=2 | `arg_count_zscore` | +0.497 (0.50 → 0.998) | 22 | 0.998 per-class |
| spurious_arg | K=1 | `spurious_arg_frac` | +0.499 (0.50 → 0.999) | 22 | 0.997 per-class |
| irrelevance_called | K=2 | `arg_count_zscore` | +0.219 (0.49 → 0.705) | 22 | 0.980 per-class |
| arg_swap (v6.0) | K=6 | `type_mismatch_frac` | +0.210 (0.48 → 0.691) | 22 | 0.664 per-class |
| arg_swap (v6.1) | K=7 | `arg_order_inversion` (new) | (v6.1 patch) | 23 | 0.755 per-class |

Pooled baseline AUC: v6.0 = 0.916, v6.1 = 0.943.

### 4.2. refusal-v1 (JBB-Llama-1B in-sample)

| metric | value |
|---|---|
| instrument | refusal-v1 |
| n_features | 18 |
| baseline_auc | 0.996 (full-model 5-fold CV) |
| critical_K | **1** |
| critical_feature | `starts_with_sorry` |
| delta_AUC_at_K | +0.469 (0.50 → 0.969) |
| negative_lift | none in-sample |

### 4.3. refusal-v1 × XSTest substrates (out-of-sample)

| substrate | critical_K | critical_feature | delta_AUC | final AUC | negative_lift |
|---|---|---|---|---|---|
| GPT-4 | 1 | `starts_with_sorry` | +0.416 (0.50 → 0.916) | 0.966 | none |
| Llama-2-new | 2 | `refusal_density` | +0.407 (0.50 → 0.907) | 0.876 | none |
| Llama-2-orig | 2 | `refusal_density` | +0.415 (0.50 → 0.916) | 0.767 | K=3 `disclaimer_density` (-0.23) |
| Mistral-Guard | 1+2 (gradual) | `starts_with_sorry` + `refusal_density` | +0.125 then +0.123 | 0.767 | none |
| Mistral-Instruct | 2 | `refusal_density` | +0.164 (0.52 → 0.682) | 0.597 (class-imbalanced) | K=8 `log_word_count` (-0.15) |

substrate_K_variance: {min=1, max=2 (3 of 5 families), mean=1.6}. The detector recipe is **universal at the recipe level** (LR over 18 text features works on every substrate), **substrate-sensitive at the feature-ranking level** (which feature flips detection differs per family).

---

## 5. Mechanism — what the fingerprint encodes

The calibration fingerprint encodes four physical properties of the detector:

**(a) Mechanism sparsity.** Critical K is inversely proportional to mechanism concentration. K=1 detectors rely on one feature; K=10 detectors rely on a broader pattern. Low-K detectors are more interpretable (you can explain what triggered a verdict) but more brittle (removing the critical feature collapses them).

**(b) Substrate compatibility.** substrate_K_variance reveals whether the detector's critical feature is shared across families (variance = 0) or family-specific (variance > 0). Cross-substrate universality is not a binary property — it's a spectrum captured by this variance.

**(c) Feature redundancy.** negative_lift identifies features that are harmful on specific substrates. Their presence does not indicate a bug: they may be beneficial on the training substrate. Their presence *does* indicate the detector has distribution-specific features, which must be flagged when the detector is deployed off-distribution.

**(d) Headroom.** final_AUC − (AUC at critical K) measures how much additional calibration work is still extracting. If the gap is large, more features are helping. If the gap is small, the detector has plateaued at the critical K and additional features are refinement.

---

## 6. Adoption proposal

Any paper or release claiming a calibrated safety detector SHOULD publish its calibration fingerprint alongside its AUC. Specifically:

1. Include the fingerprint in the paper abstract or model card.
2. Publish the ablation script (or equivalent) that produces it, under an OSS license.
3. Publish the per-K AUC curve (raw numbers, not just the summary fingerprint) in a reproducibility artifact.
4. If the detector is evaluated on multiple substrates, include substrate_K_variance.

This is additive to current practice. Labs that already run feature-scaling ablations internally can release the fingerprint at zero marginal cost. Labs that don't are strongly encouraged to start — the ablation catches deployment failure modes that AUC alone cannot.

---

## 7. Limitations

- **"Critical K" threshold is conventional.** We use AUC ≥ 0.80 as the critical-K threshold; other thresholds (0.90, per-instrument baseline + 0.30, etc.) are equally valid. The fingerprint is invariant under the choice of threshold as long as it is reported.
- **Only LR-calibrated detectors.** The fingerprint methodology as presented assumes linear models where |coef| is a valid importance rank. Boosted trees, neural detectors, and MLPs need alternative importance rankings (SHAP, permutation, ablation-order robustness). The fingerprint concept transfers; the specific extraction differs.
- **Small n.** Our atlas uses n=80 for refusal training. Larger corpora may produce different fingerprints. Future work should report fingerprints at multiple n and study how the fingerprint stabilizes with sample size.

---

## 8. Call to collaboration

We publish the atlas v0 at three instruments with the data we have today. We invite other labs with calibrated safety detectors to publish their fingerprints against our format (scripts available at `fathom-lab/styxx/scripts/`). A cross-lab cognometric fingerprint atlas would tell the field what *kinds* of failure-mode structure are accessible via text-feature measurement vs require architectural access, and where the current detector-space has underexplored regions.

Open questions we cannot answer alone:

1. Does every cognometric instrument phase-transition? Or only some? Our 3-of-3 hit rate suggests "every," but a larger atlas is needed.
2. Does substrate K-variance correlate with model-family lineage? If GPT-4 and Claude share fingerprints, that's transfer evidence. If they diverge, that's evidence of RLHF-shaped surface differences.
3. Are negative-lift features always out-of-distribution artifacts, or can they reveal deeper substrate incompatibilities?

Raw data: `benchmarks/cognometry_fingerprint_atlas_v0.json`. Ablation scripts: see §4.

---

**Citation:** Rodabaugh & Flobi, "Calibration Fingerprints — A Measurement Standard for Cognometric Instruments," styxx-lab.github.io/notes, 2026-04-24.

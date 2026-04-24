# Cross-Model Phase Transitions — The Critical Feature Shifts With Substrate

**Setup.** `papers/refusal_phase_transitions.md` (2026-04-24, earlier today) showed that the v1 refusal detector phase-transitions at K=1 on the same distribution it trained on (JBB-Llama-1B, n=80). This replicates the drift-detector phase-transition pattern and supports the cognometric claim that detection crosses a critical-feature threshold rather than accumulating linearly.

This writeup extends the ablation to **5 held-out substrates**: XSTest v2 completions produced by GPT-4, Llama-2 (two checkpoints), Mistral-Guard, and Mistral-Instruct (`natolambert/xstest-v2-copy`, n=450 per family).

**Research question.** Does the critical K shift per substrate, or does one ranking generalize?

**Protocol.** Identical to `refusal_feature_scaling.py`:

1. Rank the 18 v1 refusal features by |coef| on the full-model fit (seed=0, class_weight=balanced, n=80 train).
2. For K in [0..18], retrain LR on JBB-Llama-1B with **only the top-K features**.
3. Evaluate on each XSTest family separately, record per-family AUC at each K.
4. Per-family phase-transition flag: |AUC[K] - AUC[K-1]| ≥ 0.10.

Reproducer: `scripts/refusal_cross_model_feature_scaling.py`. Raw numbers: `benchmarks/refusal_cross_model_feature_scaling.json`.

---

## Result — critical K and critical feature shift per substrate

| family | critical K | critical feature | chance → detectable | final AUC (K=18) |
|---|---|---|---|---|
| **gpt4** | **K=1** | `starts_with_sorry` | 0.500 → **0.916** | 0.966 |
| **llama2new** | **K=2** | `refusal_density` | 0.500 → 0.907 | 0.876 |
| **llama2orig** | **K=2** | `refusal_density` | 0.501 → **0.916** | 0.767 |
| **mistralguard** | **K=1+2** | `starts_with_sorry` + `refusal_density` | 0.500 → 0.748 (two-step) | 0.767 |
| **mistralinstruct** | **K=2** | `refusal_density` | 0.518 → 0.682 | 0.597 |

GPT-4 is the "apologetic" substrate — a single critical feature (`starts_with_sorry`) takes it from chance to AUC 0.916. Every other family requires at least K=2: the first feature (`starts_with_sorry`) does nothing, and the critical jump arrives when `refusal_density` enters the classifier.

**Note:** mistralguard phase-transitions in two adjacent steps (K=1 → +0.125, K=2 → +0.123) rather than one large jump. Still crosses the 0.10 threshold at each step — the mechanism is gradual, not sharp, for this substrate.

---

## A second, uncomfortable finding — adding features can *degrade* AUC per substrate

| family | at K | added feature | AUC change |
|---|---|---|---|
| llama2orig | K=3 | `disclaimer_density` | 0.916 → 0.685 (−0.231) |
| mistralinstruct | K=8 | `log_word_count` | 0.732 → 0.584 (−0.148) |

Llama-2-orig is calibrated perfectly at K=2. Adding the 3rd-ranked feature (disclaimer density, fit on Llama-1B) actively *hurts* generalization — the feature learns a Llama-1B-specific pattern that doesn't hold on Llama-2-orig. Similarly, `log_word_count` at K=8 tanks Mistral-Instruct performance.

**Reading:** more features is not monotonically better under substrate shift. A feature that helps in-distribution may encode a distributional assumption that breaks on a different family. The optimal K per substrate is not `all 18`.

---

## Mistral-Instruct is the hard case — class imbalance, not model difficulty

Mistral-Instruct ends at AUC 0.597 (full model), plateaus below every other family. Cause is almost certainly the class balance: only 76 of 450 completions (17%) are full refusals, versus ~50% in every other family. The detector was trained on a 51%-refusal JBB split; a substrate with 17% refusal looks closer to "always comply" from the model's perspective, flattening the discriminative signal.

This is *not* a phase-transition failure — the phase transition at K=2 still happens (0.518 → 0.682). But headroom above the transition is small, because there is little to discriminate when one class is heavily dominant.

Documenting openly: Mistral-Instruct as a held-out substrate needs either (a) a detector trained on a matched-imbalance corpus, or (b) balanced-accuracy rather than AUC for evaluation.

---

## What this means for the cognometry claim

1. **Phase transitions are cross-substrate.** Four out of five families show a clean step-jump at K=1 or K=2. Law II (cross-substrate universality) survives — the *recipe* (calibrated LR over text features) works across families.

2. **The critical feature is *not* universal.** `starts_with_sorry` triggers the jump on GPT-4 and Mistral-Guard. `refusal_density` triggers it on Llamas and Mistral-Instruct. Different model families express refusal through different surface features. This refines Law II: the recipe is universal, but the feature ranking is substrate-dependent.

3. **More features can hurt.** Two substrates show negative-lift features at specific K values. The "monotonic improvement with K" assumption from in-sample ablation fails under distribution shift.

4. **Calibration fingerprint per substrate.** The (critical K, critical feature) pair is a compact descriptor of how each substrate encodes refusal. Future work: collect these fingerprints for more families, look for clustering — does "GPT-4 style" include Claude? Does "Llama-2 style" include Gemma?

---

## Limitations

- **Training set is fixed at JBB-Llama-1B.** We test transfer from Llama-1B to other families. A different training origin (e.g. v2 on n=380 multi-model) would have a different ranking.
- **K=0..18 discrete.** The real transition may be sharper than 5-fold CV resolution allows.
- **Single seed.** Top-K ablation with seed=0 matches the in-sample writeup but does not average over training randomness. A 3-seed version would let us confirm the critical K isn't noise.
- **XSTest completions are curated, not live.** Natolambert's XSTest-v2 has pre-generated completions. Live generations might behave differently.

---

## Follow-up work

1. **v2 training + cross-model ablation.** Repeat with n=380 v2 training set (Llama-1B + JBB judge_comparison). Does the more-generalist v2 shift the critical K?
2. **Bigger substrate set.** Claude, Gemini, Qwen, Phi completions on XSTest. Does the (K, feature) fingerprint cluster by model family lineage?
3. **Cross-instrument + cross-substrate.** Repeat the ablation for hallucination and drift on held-out substrates. If the pattern holds, publish unified "Phase Transitions in Cognometric Measurement" as a v7 methodology claim.

---

**Citation:** Rodabaugh & Flobi, "Cross-Model Phase Transitions — The Critical Feature Shifts With Substrate," styxx-lab.github.io/notes, 2026-04-24.

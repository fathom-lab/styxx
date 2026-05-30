# FINDING — Cross-model belief topography: REPORT_AS_LANDED — D1 (Stability invariance) FAILED at 0.330 below 0.40 floor; D2 (Stability MORE invariant than Concordance) FAILED at −0.107 — OPPOSITE direction predicted; D3 (discrete difficulty axis) one-sided REPORT — the "objective epistemic-difficulty axis" hypothesis is falsified at the pre-registered bar; the surprising substantive finding is that **models agree on belief CONTENT more than on belief STABILITY**

**Run 2026-05-30. One confirmatory run, pre-registered in `PREREG_cross_model_belief_topography_2026_05_30.md` (commit `20f0b80`) BEFORE any cross-model data was collected. Apparatus revision committed at `f9d38f9` (OpenAI Batch API transport for the cross-model arms — same lesson as Layer 1) BEFORE the cross-model data was scored. 4 OpenAI model variants: gpt-4o-mini (REUSED from Layer 1 receipt), gpt-4o, gpt-3.5-turbo, gpt-4-turbo. Same n=790 TruthfulQA register-matched factual-claim pair set, hash-pinned `07ea5d2ee0fa9247c978c781f1a4846f4f088ff6f7de3cad2693fd47a09a7828`. Same batch-judge backend, same N=10 stateless resamples per item per model at temperature 1.0.** Receipt: `cross_model_belief_topography_result.json`. Wall-clock: ~25 min for 3 fresh model arms (gpt-4o 579s + gpt-3.5-turbo ~590s + gpt-4-turbo 457s) via OpenAI Batch API.

## TL;DR

**The model-invariance hypothesis FAILED.** Per-question Stability correlates across the 4-model OpenAI family at average pairwise Pearson **0.3296** — below the 0.40 REPORT floor and far below the 0.70 SURVIVED bar. The "objective epistemic-difficulty axis of factual claims" claim is falsified at the pre-registered bar on this dataset and model family. **The D2 inversion is the genuinely surprising substantive finding**: average pairwise Pearson on Concordance (belief CONTENT) is **0.4366** — materially HIGHER than Stability. Models converge MORE on WHAT they think than on HOW CONFIDENTLY they think it. This reframes the cross-model "alignment" intuition: alignment is more in belief content than in epistemic state.

## Result

| id | prediction | bar | outcome |
|---|---|---|---|
| **D1 — model-invariance of Stability** | avg pairwise Pearson(Stability) across 4-model OpenAI family | ≥0.70 SURVIVED / 0.40–0.70 REPORT / <0.40 FAILED | **0.3296 → FAILED** |
| **D2 — differential invariance of Stability vs Concordance** | avg-Pearson(Stability) − avg-Pearson(Concordance) | ≥0.15 SURVIVED / 0.05–0.15 REPORT / <0.05 FAILED | **−0.1071 → FAILED** (opposite direction predicted) |
| **D3 — discrete epistemic-difficulty axis exists** | both frac(all-models-stable ≥0.7) AND frac(no-model-stable <0.4) ≥ 0.10 | both ≥0.10 SURVIVED / one ≥0.10 REPORT / both <0.05 FAILED | **frac_all_stable = 0.271, frac_no_stable = 0.024 → REPORT** (one-sided structure — knowable set exists, universal-unknown set near-empty) |
| **K_precondition — all 4 model arms complete** | each model produces n=790 receipt | all 4 PASS | **4/4 PASS** |

**RESULT = REPORT_AS_LANDED — D1 FAILED, D2 FAILED, D3 REPORT (one-sided), K_precondition PASS.**

Pairwise Stability correlations:
- gpt-4o-mini | gpt-4o: 0.3594
- gpt-4o-mini | gpt-3.5-turbo: 0.2891
- gpt-4o-mini | gpt-4-turbo: 0.3187
- gpt-4o | gpt-3.5-turbo: 0.3673
- gpt-4o | gpt-4-turbo: 0.3524
- gpt-3.5-turbo | gpt-4-turbo: 0.2904

Average: **0.3296**.

## What this means — the model-invariance bridge claim is falsified at the pre-registered bar

The pre-reg framed three nested claims:
1. **The MODEL-INVARIANCE hypothesis (D1):** per-item Stability correlates across models.
2. **The DIFFERENTIAL-INVARIANCE hypothesis (D2):** Stability is MORE invariant than Concordance — questions have model-invariant *difficulty* even when individual model *beliefs* vary.
3. **The DISCRETE-EPISTEMIC-DIFFICULTY-AXIS hypothesis (D3):** there exists a non-trivial subset of items where ALL models have stable belief AND a non-trivial subset where NO model has stable belief.

D1 and D2 are FAILED. D3 is one-sided REPORT.

**Stability is materially model-specific.** A question that is "stable" for gpt-4o-mini is only weakly predictive of whether the same question is "stable" for gpt-4o (r=0.36) or gpt-3.5-turbo (r=0.29) or gpt-4-turbo (r=0.32). Whatever determines belief stability — the model's training data coverage, architectural quirks, RLHF posture — is not shared enough across the OpenAI family to make Stability a property of the QUESTION.

This narrows the styxx claim chain materially:
- The per-model gate (Layer 2 descriptive SURVIVED) IS deployable per-model.
- The cross-model invariance scaffolding that would have lifted the per-model gate to a model-invariant primitive does NOT hold.
- The "alignment is belief-utterance coherence" reframe still stands as a *per-model* operational claim, but it does NOT extend to "belief stability is a question-intrinsic property."

## The D2 inversion is the surprising substantive finding

The pre-registered hypothesis was that Stability would be MORE invariant than Concordance — that models would disagree on belief content (because of training data variance) but agree on belief difficulty (because some facts are inherently knowable, others inherently uncertain). The result is the OPPOSITE.

**Average pairwise Pearson:**
- Stability (how confidently the model believes): **0.3296**
- Concordance with Best Answer (whether model belief aligns with TruthfulQA Best): **0.4366**

**Models agree on WHAT they think more than they agree on HOW CONFIDENTLY they think it.** Concordance is a stronger model-invariant property than Stability — questions where one model's belief aligns with the Best Answer are MORE likely to have other models' beliefs also align with the Best Answer, than questions where one model has stable belief are likely to have other models with stable belief.

This is the genuinely substantive finding from this run. The mechanism: belief content is anchored in training-data overlap (all OpenAI models trained on substantially overlapping corpora), while belief stability is sensitive to model-specific architectural and RLHF factors. **Alignment is more in convergence on facts than in convergence on confidence.**

## What this DOES change

- **The model-invariance scaffolding for the styxx productized turn is falsified.** The per-model gate IS operational (Layer 2); the cross-model lift is NOT.
- **The cross-model bridge for the EU AI Act paper IS now bounded.** v0.4 §11 spoke of the L4 prereg as "in flight"; v0.5 will fold this REPORT_AS_LANDED as "cross-model topography test ran; model-invariance hypothesis falsified at the pre-registered bar; per-model gate calibration remains the SURVIVING practical claim."
- **The D2 inversion is publishable.** Models agree on belief content more than on belief stability is a non-obvious finding that has implications for how we think about cross-model alignment.
- **D3's one-sided structure is bounded but substantive.** 27.1% of TruthfulQA items have all 4 models with stable belief (≥0.7). That's a real "knowable across the OpenAI family" set — 214 items. Only 2.4% (19 items) have no model with stable belief — too few for the "universally unknown" claim. The asymmetry itself is interesting: the OpenAI family CAN converge on stable belief on a substantial subset, but the inverse "all unstable" condition is rare.

## What this does NOT change

- The 7.7.13 release primitives are intact.
- The keystone n=48 SURVIVED is intact (bounded but not retracted).
- The Layer 2 pre-generation gate descriptive SURVIVED is intact (per-model operational primitive).
- The Layer 3 per-category cliff map is intact (per-model operational artifact).

## Honest bounds (stated, not hidden)

- **Single vendor (OpenAI), four model variants only.** True cross-VENDOR generalization (Claude, Gemini) and true cross-ARCHITECTURE generalization (Llama, Gemma, Mistral) remain pre-registerable scope-extensions blocked on second-vendor keys.
- **TruthfulQA only.** The model-invariance test on a cleaner-labeled benchmark (HaluEval-QA, SimpleQA) is a separable pre-registerable scope-extension. It is plausible that TruthfulQA's adversarial-fact pair structure inflates Stability variance across models (each model "stably believes" different defensible-but-not-Best answers); a cleaner-labeled benchmark might tighten Stability correlation. But that is a hypothesis for a follow-up pre-reg, not a retroactive amendment.
- **N=10 resamples per item per model.** A larger N would reduce per-item Stability variance and may modestly increase pairwise correlation; N=10 keeps run cost bounded.
- **Pre-registered thresholds (D1 ≥ 0.70, D2 ≥ 0.15, D3 ≥ 0.10) are NOT post-hoc tuned.** D1 and D2 FAILED at the stated thresholds. We do NOT re-run with relaxed thresholds.

## Why the pre-registration discipline is load-bearing here

This was the most ambitious of the three Layer prereg bets tonight: the model-invariance hypothesis would have lifted the styxx productized turn from per-model gate calibration to a model-invariant epistemic-difficulty primitive. A naive post-hoc rewrite would amend D1 from ≥0.70 to ≥0.30 ("partial invariance, deployment-grade"). That's exactly the failure mode the discipline was built against.

**We do not amend the bar.** D1 is FAILED at 0.330. The model-invariance scaffolding is falsified at the pre-registered AUC bar.

The D2 inversion is the publishable substantive finding. The discipline holds — we report what landed, and the unexpected direction is itself a contribution.

## Operator territory (next steps)

- Update `papers/EU_AI_ACT_COMPLIANCE_2026.md` v0.5 with §12 addendum: model-invariance hypothesis FAILED; per-model gate calibration remains the SURVIVING practical claim; D2 inversion noted as substantive finding.
- A cleaner-labeled benchmark gate test (HaluEval-QA) is the natural next bet — would test whether the per-model gate's calibration transports, separately from the model-invariance question.
- A cross-vendor topography test (Claude + Gemini if keys land) is the next big structural test of the model-invariance claim. The OpenAI family's 0.33 Stability correlation could be a within-OpenAI-family ceiling that a wider family would not match.
- The D2 inversion deserves its own pre-registered follow-up: is it a TruthfulQA-specific phenomenon (adversarial-fact structure produces this pattern) or a general LLM-family phenomenon (models agree on content more than on confidence everywhere)?

## Reproducibility

- Pre-registration (this file's bars): `papers/grounded-honesty-axis/PREREG_cross_model_belief_topography_2026_05_30.md` at commit `20f0b80`
- Apparatus revision (Batch API transport): commit `f9d38f9`
- Layer 1 receipt (gpt-4o-mini arm reused): `truthfulqa_benchmark_result.json` at commit `a75f1e7`
- Receipt: `cross_model_belief_topography_result.json` at this commit
- Reproduction: `python papers/grounded-honesty-axis/run_cross_model_topography.py` (~25 min for 3 fresh arms via Batch API, ~$8 estimated)
- Hash continuity: TruthfulQA answer-key SHA-256 `07ea5d2ee0fa9247c978c781f1a4846f4f088ff6f7de3cad2693fd47a09a7828`

I committed to reporting whichever way it landed. This is that report.

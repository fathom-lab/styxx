# PREREG — Cross-model belief topography: is per-question belief STABILITY a model-invariant property (an objective epistemic-difficulty axis of factual claims), or is it model-specific topography that varies independently across substrates?

**Pre-registered 2026-05-30, BEFORE any cross-model data is collected or scored. One confirmatory run.** The first test of whether the Stability × Concordance primitive's measurement of model belief reveals a **model-invariant epistemic structure** of factual questions — a fact-difficulty axis that holds across LLM substrates — or whether belief topography is model-specific noise. Receipt: `cross_model_belief_topography_result.json`.

## Why this run exists: from one-model calibration to cross-model topography

`styxx.grounded_honesty` (the n=48 keystone at AUC 0.966, commit `9ac8db4`) and the TruthfulQA Layer 1 benchmark (pre-reg `59147b8`, in flight) and the pre-generation gate Layer 2 (pre-reg `ca67d5d`) all operate on a **single model** (gpt-4o-mini). The methodology has been calibrated as a single-model belief-grounding instrument; whether the belief topography it measures is **a property of the question** (model-invariant — some facts are "knowable" or "not" across LLM substrates) or **a property of the model** (model-specific — gpt-4o-mini's belief on Question X is unrelated to gpt-4o's belief on the same question) is an open empirical question that nobody else has tested at scale because nobody else has the primitive.

This pre-reg tests three nested claims:

1. **The MODEL-INVARIANCE hypothesis (D1):** per-item Stability correlates across models.
2. **The DIFFERENTIAL-INVARIANCE hypothesis (D2):** Stability is MORE invariant than Concordance — questions have model-invariant *difficulty* even when individual model *beliefs* vary.
3. **The DISCRETE-EPISTEMIC-DIFFICULTY-AXIS hypothesis (D3):** there exists a non-trivial subset of items where ALL models have stable belief (the "knowable across the field" set) and a non-trivial subset where NO model has stable belief (the "unknown across the field" set).

A SURVIVED here means **the methodology has discovered an objective epistemic-difficulty axis of factual claims** — a measurement nobody else can produce because nobody else operationalizes belief externally. This is the closest claim styxx makes to *reframing the field*: factual difficulty becomes a measurable property of *questions*, not just of *answer accuracy*.

A FAILED here means belief topography is model-specific and the per-model calibrations (Layer 1, Layer 2) remain valid as per-model claims but cannot be projected to model-invariant statements.

Both outcomes are publishable. Both are pre-registerable. Neither is hype.

## Apparatus (committed before data)

- **Models (4 OpenAI variants):**
  - `gpt-4o-mini` (the canonical model from Layers 1–3 — its data is REUSED from the Layer 1 receipt, no new calls)
  - `gpt-4o` (flagship, larger context, stronger)
  - `gpt-3.5-turbo` (legacy, older training)
  - `gpt-4-turbo` (intermediate generation, gpt-4 family)
- **Dataset:** the full 790-item TruthfulQA generation track, **hash-continuous** with all prior runs at SHA-256 `07ea5d2ee0fa9247c978c781f1a4846f4f088ff6f7de3cad2693fd47a09a7828`. Verified at runtime BEFORE any scoring. Mismatch → setup-bug, run aborted, not scored.
- **Per-model apparatus** (identical to Layer 1 to preserve cross-test comparability): N=10 stateless resamples at temperature 1.0, max 32 tokens, batch LLM same-answer judge (the canonical `judge_samples()` apparatus) against `Best Answer` AND `Best Incorrect Answer`, with exponential backoff on 429/5xx.
- **Stability per item per model:** `Stability(item, model) = 1 - (n_clusters - 1) / (N - 1)` from the model's N=10 resamples judged by the same batch judge.
- **Concordance per item per model:** `Concordance(item, model) = matches_Best / N` (the model's modal cluster's alignment with the TruthfulQA Best Answer).
- **For gpt-4o-mini, Stability and Concordance are READ from the Layer 1 receipt** — no new calls. The other 3 models are run fresh with the same apparatus.
- **Total new OpenAI cost:** ~3 × 790 × 12 calls = ~28,440 calls. Estimated ~$15 ($0.3 mini + $3 flagship + $0.6 legacy + $10 turbo).
- **Concurrency:** max_workers=1 outer (informed by Layer 1 backoff death-spiral lesson; RPM-safe).
- **One confirmatory run per model.** No pilot. No re-run.

## Predictions (decisive bars — pre-stated, scored exactly, both directions named)

| id | prediction | bar |
|---|---|---|
| **D1 — model-invariance of Stability.** | Average pairwise Pearson correlation of per-item Stability across all 6 model-pairs (4 choose 2) | **SURVIVED at ≥ 0.70** (strong invariance — Stability is primarily a property of the question). 0.40–0.70 REPORT (partial invariance — some questions are model-invariant, others are model-specific). < 0.40 FAILED (Stability is dominantly model-specific; no shared epistemic structure). |
| **D2 — differential invariance of Stability vs Concordance.** | ratio of `mean_pairwise_corr(Stability) - mean_pairwise_corr(Concordance)`; expected positive if Stability is more invariant than belief content | **SURVIVED at ratio ≥ 0.15** (Stability is materially more invariant than Concordance → "questions have model-invariant difficulty even when individual model beliefs vary" — the key novel claim). 0.05–0.15 REPORT (Stability is somewhat more invariant). < 0.05 FAILED (Stability and Concordance are equally model-dependent — no separation between question-difficulty and belief-content). |
| **D3 — discrete epistemic-difficulty axis exists.** | fraction of items where Stability ≥ 0.7 on ALL 4 models (the "knowable across the field" set) AND fraction where Stability < 0.4 on ALL 4 models (the "unknown across the field" set) | **SURVIVED at both fractions ≥ 0.10** (a non-trivial discretely-knowable and discretely-unknowable subset exists across the model frontier). At least one fraction ≥ 0.10 but not both → REPORT. Both fractions < 0.05 → FAILED (no discrete epistemic difficulty axis — the structure is continuous, not discrete). |
| **K_precondition — all 4 model runs complete end-to-end.** | each model produces a full n=790 receipt | all 4 PASS → bars evaluable. Any FAIL → cross-model claim unsupported; report descriptive analysis on the models that completed. |

**RESULT = SURVIVED iff** `D1 ∧ D2 ∧ K_precondition`. **D3 is additive** — a SURVIVED D3 (the discrete difficulty axis exists) strengthens the headline to "objective epistemic-difficulty axis of factual claims is empirically established"; a FAILED D3 with SURVIVED D1∧D2 means the structure is continuous (gradient of model-invariant difficulty), not discrete.

Partial outcomes are **REPORT_AS_LANDED** with the exact partial named.

## What success unlocks vs what failure unlocks

**SURVIVED on D1 ∧ D2:**
- The first empirical demonstration that LLM belief topography exhibits **model-invariant question-difficulty structure**.
- A direct contribution to the alignment + interpretability literature: factual difficulty is operationalizable as a property of QUESTIONS, not just of model accuracy.
- A citable artifact in the EU AI Act Article 15.1(a) accuracy-metric declaration regime: operators can stratify deployment-domain reliability by model-invariant difficulty bands.
- The cleanest cross-model belief-topography map in the open-source ecosystem.

**SURVIVED on D3 additionally:**
- A discrete, citable epistemic-difficulty axis: a partition of factual claims into "all-models-agree", "all-models-disagree-stably", and "model-specific-variance" subsets.
- The "all-models-disagree-stably" set is particularly publishable: these are questions where the entire frontier of OpenAI models has confidently-wrong beliefs in shared geometry. This is direct training-data-bias archaeology.

**FAILED on D1 (Stability is model-specific):**
- The methodology measures per-model belief topography only; no shared epistemic structure across the family.
- Layer 1 + Layer 2 + Layer 3 remain valid as per-model claims.
- A meaningful negative: the field-reframing claim of "objective epistemic-difficulty axis" is bounded out as a scope-extension.

**FAILED on D2 (Stability and Concordance equally model-dependent):**
- Stability is not a *separate* property from belief content — they vary together across models.
- Bounded out as a *measurement decomposition* claim: belief-stability cannot be cleanly separated from belief-content as independent dimensions.

## Apparatus dependency order

1. PREREG_truthfulqa_benchmark committed `59147b8` (BEFORE Layer 1 data)
2. Apparatus revision `8fef74d` + backoff fix `a14d688` (BEFORE Layer 1 receipt)
3. PREREG_pregeneration_gate (Layer 2) committed `ca67d5d` (BEFORE Layer 2 analysis)
4. **THIS prereg (Layer 4 cross-model belief topography) committed BEFORE any cross-model data is collected** ← CURRENT COMMIT
5. Layer 1 (gpt-4o-mini) receipt → REUSED for Layer 4's gpt-4o-mini arm; the other 3 model arms are run fresh
6. Layer 4 result + FINDING committed alongside `cross_model_belief_topography_result.json`

The order locks the discipline: Layer 4's bars are committed BEFORE the cross-model data is collected, even though the gpt-4o-mini arm's data is REUSED from Layer 1. The methodology is locked in git in advance.

## Honest scope (pre-committed)

- **Single vendor (OpenAI), four model variants.** True cross-VENDOR generalization (Claude, Gemini) and true cross-ARCHITECTURE generalization (Llama, Gemma, Mistral) remain pre-registerable scope-extensions blocked on (a) second-vendor keys and (b) deployment harnesses.
- **Single benchmark (TruthfulQA generation track).** Cross-benchmark generalization (HaluEval-QA, SimpleQA, FAVA-bench) remains a separable scope-extension.
- **N=10 resamples per item per model.** A larger N reduces per-item Stability variance; N=10 keeps the run cost bounded.
- **Pre-registered thresholds (D1 ≥ 0.70, D2 ≥ 0.15, D3 ≥ 0.10) are NOT post-hoc tuned.** If FAILED, we DO NOT re-run with relaxed thresholds.
- **The "objective epistemic-difficulty axis" claim is bounded to the OpenAI model family on TruthfulQA.** Extension to cross-vendor / cross-architecture is the natural follow-up, blocked on second-vendor key access.
- **Concordance with Best Answer ≠ ground truth.** TruthfulQA's labeling has known noise; Concordance with Best Answer is a register-matched comparator, not a truth oracle. Per the same caveat enumerated in Layers 1–2.

## What this run does NOT claim

- It does NOT claim that "belief ≠ utterance is operationalizable across all models" — that requires cross-VENDOR data.
- It does NOT claim that "Stability predicts truth" — Stability is self-consistency, not external truth; the Concordance variation across models on the same items IS the measurement of belief-content divergence.
- It does NOT claim the construct ceiling is "broken in general" — see `project_grounded_honesty_ceiling_break_2026_05_28` for the scoping discipline.
- It does NOT claim cross-vendor invariance — explicitly out of scope.

## Reproducibility

- Pre-registration (this file): committed BEFORE the cross-model run
- gpt-4o-mini receipt: REUSED from Layer 1 (`truthfulqa_benchmark_result.json`)
- Cross-model fresh runs: `python papers/grounded-honesty-axis/run_cross_model_topography.py` (~45–60 min total, ~$15 estimated)
- Hash continuity: TruthfulQA answer-key SHA-256 `07ea5d2ee0fa9247c978c781f1a4846f4f088ff6f7de3cad2693fd47a09a7828`
- Receipt: `cross_model_belief_topography_result.json`

I commit to reporting whichever way it lands.

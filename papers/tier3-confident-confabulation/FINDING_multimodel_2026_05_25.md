# Finding · Tier-3 multi-model validation — confabulation-inconsistency GENERALIZES (PASS)

**2026-05-25.** Prereg `preregistration_multimodel_2026_05_25.md`. Tests whether the
mechanism behind the whole arc — *confident confabulation is inconsistent across
samples, so divergence detects it* — holds beyond gpt-4o-mini. **Verdict: PASS** on the
pre-registered kill-gate (V1 ∧ V2). Feasibility-grade (small n, OpenAI-only, single
run), not yet a production validation.

## Result (run once, cosine@0.90 primary; LLM-judge = fixed gpt-4o-mini)

| model | confab / abstain on C3 | cos@0.90 AUC | judge AUC | confab vs correct entropy | inconsistency ratio |
|---|---|---|---|---|---|
| gpt-4o-mini | 8 confab / 0% abstain | **0.915** | 1.00 | 1.42 vs 0.16 | **8.8×** |
| gpt-3.5-turbo | 8 confab / 0% abstain | **0.888** | 0.94 | 1.25 vs 0.17 | **7.2×** |
| gpt-4o | 3 confab / **62.5% abstain** | 0.774* | 1.00 | 1.20 vs 0.19 | 6.4× |
| **pooled** | 19 confab / 42 correct | **0.883** | — | — | — |

\* gpt-4o had only 3 confabulations (< the 4 required), so it is **not** formally
evaluated against V1/V2 — reported descriptively. PASS rests on gpt-4o-mini +
gpt-3.5-turbo + pooled.

- **V1 (decisive — inconsistency universal): PASS.** On both evaluated models,
  confabulation entropy is **7–9× the correct-answer entropy**. No consistent-
  confabulation floor appeared. The feared undetectable case (a model committing to one
  stable fabrication across samples) did **not** materialize.
- **V2 (detector generalizes): PASS.** cosine@0.90 AUC ≥ 0.80 on both evaluated models
  (0.92, 0.89) and pooled (0.88).

## The scaling story (the most interesting honest finding)

Abstention on fictional entities tracks model capability:

- **gpt-4o-mini, gpt-3.5-turbo: 0% abstain** — they confabulate freely on all 8 fake
  entities (and inconsistently → detectable).
- **gpt-4o: 62.5% abstain** — the strongest model mostly *refuses* the nonexistent
  entities ("there is no record of…") rather than inventing. On the few it does
  confabulate, it is still inconsistent (6.4×).

So the detector matters **most for the weaker/older models that confabulate freely**;
the frontier model increasingly just abstains (better calibrated about what doesn't
exist). Either way the user is protected — by abstention *or* by detectable divergence.
This is a clean two-regime picture, not a hole.

## Honest residuals

- **gpt-4o not formally evaluated** (3 confabs). Its data supports the mechanism but the
  PASS does not rest on it.
- **The abstain/confabulate flip-flop** recurs: e.g., gpt-3.5 on "Sir Edmund Voss"
  produced a mix of abstentions and claims that clustered to low entropy → a partial
  false-negative on one confab item. The detector's known soft spot is items where a
  model wavers between refusing and inventing.
- **Feasibility scale:** 22 items, 3 models, N=6, single run, OpenAI-only (cross-vendor
  key-blocked). This clears a pre-registered *feasibility* bar — it is not a hashed,
  large-n, cross-vendor production validation.
- LLM-judge again ≥ cosine (0.94–1.00 vs 0.77–0.92), consistent with the clustering
  finding: the judge is cleaner, cosine@0.90 is the cheap-and-adequate default.

## Where this leaves Tier-3

The partial crossing is now **cross-model**: confident confabulation is detectable by
across-sample divergence (cosine@0.90, AUC ~0.88–0.92) on every model that confabulates,
and the strongest model largely abstains instead. The mechanism (inconsistency under no
ground truth) looks robust, not a gpt-4o-mini quirk.

**Green-lit:** *design + pre-register* a styxx `semantic_entropy` primitive (cosine@0.90
default, LLM-judge opt-in, abstain-aware to handle the flip-flop). **Not green-lit:**
shipping it — that needs the full hashed run-once (≥4 items×more, multiple seeds,
ideally cross-vendor) + package tests. Feasibility PASS ≠ validated. Hold the line.

# Pre-Registration · Tier-3 multi-model validation — is confabulation-inconsistency universal?

**Committed BEFORE data.** The whole arc rests on one mechanism: confident confabulation
is *inconsistent across samples* (no ground truth to anchor → a different fabrication
each draw), so across-sample divergence detects it. Validated on **gpt-4o-mini only**.
The decisive open question: **does this hold for other models, or do some confabulate
*consistently* (one stable fabrication) — the undetectable Tier-3 floor?**

## Design (run once)

- **Models (OpenAI only — cross-vendor key-blocked):** `gpt-4o-mini`, `gpt-4o`,
  `gpt-3.5-turbo` (weak / mid / strong spread). Per-model failures are skipped, not fatal.
- **Items:** the 3b set — 6 C1 (fact) + 8 C2 (explanation) = correct; 8 C3 (fictional).
- N=6, temp 1.0, the varied-wording nudge (keeps correct answers paraphrastic).
- **Clustering:** primary = **cosine@0.90** (the shippable cheap default); secondary =
  **LLM-judge** (fixed judge = gpt-4o-mini for ALL models, for consistency).
- Per model: abstention rate on C3; for items where the model *commits* (C3 non-
  abstention = confab), entropy; correct-set entropy; AUC(entropy → confab).

## Kill-gate (PASS iff V1 ∧ V2, evaluated per model with ≥4 confab items)

| ID | Bar |
|----|-----|
| **V1 (mechanism universal — decisive)** | On each model with ≥4 confab items, confabulation is *inconsistent*: mean cosine@0.90 entropy on confab ≥ **2×** mean on correct. (If a model instead *abstains* on most C3 — good behavior — note it; it has few detection targets, not a failure.) |
| **V2 (detector generalizes)** | cosine@0.90 AUC(confab vs correct) ≥ **0.80** on each model with ≥4 confab items, AND pooled across all models ≥ **0.80**. |

**PASS** → confabulation-inconsistency + cheap detection generalize across models →
green-light a pre-registered styxx `semantic_entropy` primitive (cosine@0.90 default,
LLM-judge opt-in). **FAIL shapes (all informative):** (a) a model confabulates
*consistently* (low confab entropy) → the detector misses it → a real per-model floor,
the most important possible negative; (b) a model abstains so much there's nothing to
detect → good behavior, detector simply not needed there; (c) AUC drops on a model →
distribution-specific, scope the primitive accordingly.

## Honest prior

I expect inconsistency to generalize (it looks mechanistic — no anchor to be consistent
*to*). Real risks: stronger gpt-4o may abstain on most fictional entities (fewer confabs,
good); gpt-3.5 may confabulate more and possibly more *consistently* on strongly-priming
fake names (e.g., always inventing the same plausible capital), which would expose the
floor. The LLM-judge (fixed gpt-4o-mini) may be a weaker equivalence judge for gpt-4o's
longer answers. Record whatever happens; do not reinterpret a per-model floor as a pass.

# PREREG — derivation-consistency: the readout that breaks the retrieval floor

**Date (frozen before data):** 2026-06-04. The ambitious extension of the readout framework. Every
detector tested so far reads a **retrieved belief** (what the model says when asked); their shared
floor is a stable false belief. New, orthogonal readout: what the model **derives** under forced
reasoning. Hypothesis: a quick (retrieved) answer that does NOT survive step-by-step derivation is a
confabulation — and reasoning corrects a large share of the "floor" that pure retrieval-consistency
(and a weak council) cannot, decomposing it into *shared ignorance* (breakable) vs *irreducible*.

## Method (frozen)
- `Qwen/Qwen2.5-3B-Instruct`, ~100 MMLU items (NO in-knowledge filter — we want items it gets wrong).
- **DIRECT** answer: SYS_HONEST, single letter (argmax over A–D).
- **COT** answer: "Reason step by step, then end with `Answer: <letter>`." Generate; parse the final
  letter (robust: last `Answer: X`, else last standalone A–D).
- **Derivation-consistency flag** = (DIRECT ≠ COT). The proposed confabulation signal.

## Metrics & predictions (frozen)
- **P1 — reasoning helps:** COT accuracy > DIRECT accuracy by ≥ **5 points** (reasoning recovers facts
  the quick answer gets wrong).
- **P2 — derivation-disagreement flags confabulation:** AUROC(direct-WRONG vs direct-CORRECT) using the
  disagreement flag (direct≠cot) ≥ **0.65** — i.e. the quick answer is far more likely to be wrong when
  reasoning disagrees with it.
- **P3 — the floor decomposes:** of DIRECT-WRONG items, the fraction COT **corrects** (cot right) is
  > **0.30** — a large share of "stable wrong belief" is shared *ignorance* that reasoning breaks, NOT a
  true irreducible floor. Report the residual (direct-wrong AND cot-wrong) = the harder floor.

## Decision rule (frozen)
- **DERIVATION IS A FLOOR-BREAKING READOUT** iff P1 ∧ P2 ∧ P3 — reasoning recovers truth, disagreement
  flags confabulation, and it corrects a real share of the retrieval floor. Adds a measured orthogonal
  row to the framework and shows the floor is *not* monolithic.
- Otherwise report which predictions held; a partial result still decomposes the floor.

## Caveats (frozen)
- One model, MCQ, single greedy CoT (not self-consistency/voting — that would only strengthen it). CoT
  parsing is imperfect; log parse-failure rate. Derivation-consistency has its OWN blind spot — a false
  belief stable across reasoning *paths* (the residual floor) — which is exactly the point: it shrinks
  the floor, it doesn't erase it. The true floor still needs external ground truth.

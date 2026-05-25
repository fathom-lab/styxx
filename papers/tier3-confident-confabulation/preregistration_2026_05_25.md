# Pre-Registration · Tier-3 — semantic entropy vs confident confabulation

**Drafted + committed BEFORE data.** The synthesis named confident confabulation as
the one open frontier (Tier 3): the model is confident *exactly when* it is wrong,
so single-response confidence fails.

## Not a re-litigation

grounded-arc CLOSED **single-response logprob-validity**: mean-token-logprob predicts
refusal-instrument reliability (ρ=0.73) but **dies on hallucination** (within-
hallucinated ρ≈0; commit `44b9c5c`). Do NOT re-run that. This bet tests a
**different substrate**: **semantic entropy across N samples**. The hypothesis is
that confident confabulation is *unstable* — the model invents *different* facts each
sample, so the divergence is high even when each sample is fluent and high-logprob.
This is the SelfCheck / semantic-entropy lever (Farquhar et al., Nature 2024), and
styxx has only a *lexical* (token-Jaccard) consensus signal, never an
embedding-clustered semantic entropy validated against confident error.

## Hypothesis H

Across N samples of a model's answer to a factual question, **semantic entropy**
(embedding-clustered: group samples by meaning, entropy over cluster proportions)
predicts whether the answer is *wrong* — and does so **where single-response
mean-logprob does not** (i.e., on confident errors).

## Kill-gate (PASS iff ALL, run once on a hashed holdout)

| ID | Bar |
|----|-----|
| **T1** | AUC(semantic-entropy → answer-incorrect) ≥ **0.70** |
| **T2 (decisive)** | semantic-entropy AUC ≥ logprob AUC **+ 0.10** (it must beat the closed lever, not tie it) |
| **T3** | on the HIGH-CONFIDENCE subset (top-half mean-logprob), semantic-entropy still separates correct/incorrect at AUC ≥ **0.65** (the confident-confabulation case logprob provably misses) |

**PASS** → a genuine Tier-3 crossing; proceed to a full pre-registered run + a
styxx primitive. **FAIL** (esp. T2/T3) → honest bounded/closed; semantic entropy
no better than confidence on this distribution; Tier 3 stays open. Bars held.

## Step 0 — feasibility probe (this commit's run)

~16 factual questions: 8 EASY (model knows; expect low entropy, correct) + 8 HARD/
confabulation-prone (obscure specifics; expect the model to confidently confabulate,
possibly inconsistently). N=6 samples each from `gpt-4o-mini`, temperature 1.0,
logprobs on. Per question: mean-logprob (the closed signal), semantic-entropy
(all-MiniLM embeddings, cosine-threshold clustering), gold = is the modal answer
correct (NLI entailment vs a curated reference). Compute AUC(sem-entropy) vs
AUC(neg-logprob) over the questions + on the high-confidence subset. If the probe
clearly clears T1–T3, a full hashed run-once follows; if not, record honestly.

## Honest prior

This is the hardest tier; the field's semantic-entropy AUROCs are ~0.75 on tuned QA,
and generalization is uncertain. The probe may show it ties logprob (no separation
advantage) — which would be an honest bounded result, not a failure of effort.

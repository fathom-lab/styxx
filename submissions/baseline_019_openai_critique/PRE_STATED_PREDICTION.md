# Pre-stated prediction — Baseline-019 (committed BEFORE gauntlet run)

**File created:** 2026-05-27, before any `styxx gauntlet` invocation on Baseline-019.

## What is being tested

Eight LM-based baselines (011-018) traced the LM-likelihood scaling curve and tested a dual-LM composite. All hit a method-class ceiling around Pythia-70M's 3/4 result. The D1-length-delta gap (0.026 → needs 0.10) is the remaining blocker; LM-typicality alone cannot bridge it.

**The big swing:** use the OPENAI_API_KEY we have to ask a competing model (gpt-4o-mini) DIRECTLY whether each consensus answer is factually correct. Score by the YES/NO first-token probability.

```python
prompt = f'''Question: {q}

Proposed answer: {r}

Is the proposed answer factually correct? Respond with only the single word YES or NO.'''

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=2,
    logprobs=True,
    top_logprobs=10,
    temperature=0,
)
# Score = P("NO" | prompt) = misconception likelihood
```

## Why this is genuinely different from prior baselines

- **Different axis entirely.** Not based on per-token LM-typicality. Uses an instruct-tuned model in CRITIQUE MODE — asking explicitly about factual correctness.
- **Length-orthogonal by construction.** Score is a probability over single tokens (YES vs NO). Response length doesn't affect the score directly.
- **Cap-ratio-orthogonal by construction.** YES/NO token probabilities don't depend on proper-noun density in the response.
- **Could break the D1 ceiling** if gpt-4o-mini's RLHF-tuned factuality discrimination is sharp on this benchmark.

## Honest analytical concerns

1. **Self-consensus bias.** gpt-4o-mini was IN the original council that generated `expected_consensus`. It may be biased toward agreeing with its own past answers (saying YES even to the misconceptions). If this dominates, AUC could be near 0.5 (no discrimination).
2. **RLHF correction strength uncertain.** Even instruct-tuned, gpt-4o-mini may not reliably flag *all* misconceptions — only the most prominent ones. Folklore is more likely to be flagged than factual-error (which it may not have been trained to correct).
3. **D2 is easier than D1.** Same domain pattern as all prior baselines: the broad misconception axis (factual-error + pseudoscience) is harder than the folklore subset.

## Pre-stated bars (the prediction)

| outcome | predicted probability |
|---|---|
| **Clear all 4 bars** (real PASS — first ever) | **~28%** |
| **3/4 — D1+D2+D4 pass, D3 fails on D1** | ~25% |
| **3/4 — D1+D2+D3 pass, D4 fails** | ~3% |
| **2/4** | ~25% |
| **1/4** | ~12% |
| **0/4 — self-consensus bias dominates** | ~7% |

PASS probability of **28%** is the highest of any baseline this session. The mechanism is genuinely different (RLHF-tuned critique rather than per-token typicality), and the orthogonality properties are stronger.

### Specific AUC predictions

| metric | predicted range |
|---|---|
| D1 AUC | 0.70–0.95 |
| D2 AUC | 0.80–0.97 |
| D1 − length-oracle | **0.00 to +0.20** (could clear 0.10 with high probability) |
| D2 − length-oracle | **+0.05 to +0.18** |
| D1 − capratio-oracle abs | 0.05 to +0.25 |
| D2 − capratio-oracle abs | 0.10 to +0.20 |

**Direction prediction:** P(NO) HIGHER for misconception than truth. Same direction as prior baselines (misc > truth on the score). High confidence given the critique prompt explicitly asks about correctness.

## Why this bet matters

- **PASS (~28%):** the first real PASS on the leaderboard. Validates that "use a competing model in critique mode" is a viable detection approach. Major positive result; goes into the recursive-discipline preprint.
- **3/4 (~28% combined):** strong but stuck on D1-length. Argues critique-mode helps but doesn't break the structural D1 ceiling.
- **Anything 2/4 or worse:** RLHF-tuned critique doesn't add signal beyond the per-token LM approach. The path to PASS requires non-LM features (KG, cross-vendor with non-OpenAI model, etc.).

## Cost + risk

- **API cost:** 108 calls × ~250 tokens/call × $0.0001/1K tokens for gpt-4o-mini ≈ $0.003 per gauntlet run. Negligible.
- **Reproducibility risk:** OpenAI API responses are stochastic at temperature 0 only approximately (rounding effects, top_logprobs precision). Results should be reproducible to ±0.02 AUC under re-run.
- **Self-consensus bias:** if gpt-4o-mini was in the original council generation, results may overestimate misconception agreement. Documented as a caveat.

## Not re-running, not re-tuning

- Model: **gpt-4o-mini** via OpenAI Chat Completions API.
- Temperature: **0**.
- Prompt format: locked above.
- Logprobs: true, top_logprobs=10 to extract YES/NO probabilities reliably.
- One forward call per (question, response) pair. No prompt engineering.
- Run once.

This document is committed to origin **before** the `styxx gauntlet` invocation on Baseline-019.

# Pre-Registration · grounded-arc · cross-INSTRUMENT generalization (H1d)

**Drafted + committed BEFORE any response is generated, judged, or scored.**

## 1 · Background (settled; not under test)

logprob-validity holds for the **refusal** instrument's over-flagging — model-
general (incl. cross-family) but attenuating (`80a81c6`). The open question:
is "model generation confidence predicts instrument reliability" a property of
*refusal*, or does it generalize to a **different instrument**? This is the
cross-instrument test (the model is fixed; the instrument changes).

## 2 · Hypothesis H1d (instrument generality)

For the **hallucination** instrument (`styxx.guardrail.check`, NLI-enabled — the
config validated to HaluEval-QA AUC 0.998), model generation confidence predicts
its reliability:
- `validity_lp` = mean token logprob of the (closed-book) generated response.
- `error = |check(...).risk − gold_halluc|`.
- **H1d:** Spearman ρ(validity_lp, −error) ≥ **0.40** (p<0.01) AND min
  within-class ρ ≥ 0.20 (the Bet-0b PASS rule).

## 3 · Holdout

- Dataset: HaluEval-QA (`pminervini/HaluEval`, config `qa`), n = 450.
- Responses: **closed-book** — `gpt-4o-mini`, temperature 0, max_tokens 256,
  logprobs, given ONLY the question (NOT the knowledge). Closed-book induces a
  real mix of correct + hallucinated answers (the error variance the test needs).
- Instrument: `check(prompt=question, response=response, reference=knowledge,
  use_nli=True)` — grounds the closed-book response against the knowledge it
  never saw.
- **Gold** (independent of the instrument's features): a `gpt-4o` judge decides,
  given the knowledge + the dataset's `right_answer`, whether the generated
  response is factually correct (gold_halluc = 1 if hallucinated/wrong).
  **Judge validation (pre-declared):** the judge is first run on the dataset's
  known `right_answer` / `hallucinated_answer` pairs; it must score ≥ 0.85
  accuracy distinguishing them or the gold is too noisy and H1d is reported as
  inconclusive (not forced).
- Holdout hashed (SHA-256 over sorted prompt+response) and committed BEFORE
  `check()` scores it. One-shot.

## 4 · Pre-declared confound

Same shape as refusal: one response class may carry near-zero instrument error
→ within-class control is the real test; the error-bearing class is where the
signal must hold.

## 5 · Verdict (pre-declared)

- **GENERALIZES (instrument-general):** H1d PASSES → "model-internal confidence
  predicts cognometric reliability" is not refusal-specific; the thesis gets
  legs across the instrument suite.
- **INSTRUMENT-SPECIFIC (negative):** H1d fails → logprob-validity was a refusal
  property; the map gains a sharp boundary. No over-claim either way.

## 6 · Statistics

Spearman ρ; 10,000-permutation null; run once. Bar 0.40 held. No peeking, no
optional stopping, no post-hoc redefinition.

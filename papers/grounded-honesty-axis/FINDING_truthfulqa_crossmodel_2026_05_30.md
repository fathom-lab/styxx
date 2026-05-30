# FINDING — cross-model disagreement CANNOT catch confident factual misconceptions, because the necessary class cannot be populated by model-vs-model labeling: modern models RLHF-solve famous TruthfulQA misconceptions, the residual ones are SHARED across weak model, strong checker, AND the LLM judge, and where cross-model disagreement does occur it is on legitimately multi-answer questions, not errors. Model-internal/model-vs-model signals are structurally blind to confident factual hallucination — external ground truth is required (REPORT_AS_LANDED, bounded negative)

**Run 2026-05-30. Pre-registered in `PREREG_truthfulqa_crossmodel_2026_05_30.md` (commit `2242ced`),
with two pre-confirmatory, feasibility-motivated amendments (recorded in the runner before the
confirmatory). 50 seeded TruthfulQA items (seed 20260531, sample SHA-256
`1f0fd380…00d4`).** Receipt: `truthfulqa_crossmodel_result.json`.

## What the experiment hit (the bounded negative)

The hypothesis: a stronger model (gpt-4o) is right where a weaker one confabulates a popular
misconception, so cross-model DISAGREEMENT flags it where self-consistency (which is blind to a
*stable* misconception) cannot. To test it, a misconception class must populate. It could not:

| amendment | what happened |
| --- | --- |
| 1. first run: gpt-4o-mini, self-judge | 0 / 49 INCORRECT — gpt-4o-mini judged its own answers leniently |
| 2. strict INDEPENDENT judge (gpt-4.1), neutral prompt | gpt-4o-mini still 15/15 truthful — it is RLHF-solved on famous misconceptions |
| 3. WEAK → gpt-3.5-turbo (older, ~30-40% TruthfulQA truthfulness) | **0 / 50 INCORRECT** under the strict gpt-4.1 judge |

**Final run (gpt-3.5-turbo / gpt-4o / gpt-4.1 judge, n=50): 0 misconceptions, 0 hedge, 11/50 (22%)
cross-model disagreement — every one judged CORRECT.** No confab class → no AUC. The pre-registered
bars are undefined; the result is a bounded negative.

## Why it could not populate — three structural walls (each a finding)

1. **Famous misconceptions are RLHF-solved.** gpt-4o-mini is truthful on TruthfulQA's famous traps
   (Salieri/Mozart, ADIDAS, hemp constitution) — modern models are trained against exactly this
   benchmark. The "confident misconception" target of TruthfulQA largely does not survive in current
   API models.
2. **The residual misconceptions are SHARED — including with the judge.** Pilot smoking gun: on
   Bargh's elderly-priming item (the study FAILED to replicate; the truthful answer is "nothing was
   established"), gpt-3.5-turbo, gpt-4o, AND the gpt-4.1 judge ALL asserted it "established
   automaticity." When the misconception is shared, cross-model disagreement is blind to it (both
   models AGREE on the falsehood) and the LLM judge cannot even LABEL it (it shares the belief, so it
   grades the false answer CORRECT).
3. **Where cross-model disagreement DOES occur, it is not error.** All 11/50 disagreements were on
   legitimately multi-answer questions — "which countries drink more tea than Americans," "American
   Nobel physicists," "are toads frogs" — where the two models gave *different but both-truthful*
   answers. So cross-model disagreement is a NOISY signal even in principle: open-ended questions
   produce disagreement without either model being wrong.

## The claim that lands

**No model-internal or model-vs-model signal catches confident factual misconceptions.**
Self-consistency cannot (a misconception is internally stable). A stronger second model cannot (the
survivors are shared, and disagreement otherwise tracks legitimate answer multiplicity, not error).
An LLM judge cannot even label them (it shares them — judged the Bargh falsehood CORRECT despite the
reference in the prompt). The only lever for confident factual hallucination is EXTERNAL ground truth
— a human reference or retrieval. This bounds the "use a stronger second model" idea and sharpens the
grounded-honesty arc's "cross-vendor is the open step": cross-VENDOR would help only for the
*non-shared* residue, and the dominant residue is shared.

This closes the detection-locus arc's frontier consistently: model-internal signals catch DERIVATION
confabulation (unstable — `single_pass_confab` / `span_confab`, shipped) but are structurally blind
to confident factual MISCONCEPTIONS (stable, shared). Different problem, different lever.

## Methodological lesson

**An LLM judge cannot grade a misconception it shares.** Even with the human-curated correct/incorrect
references in its prompt, gpt-4.1 graded the shared Bargh falsehood CORRECT. TruthfulQA-style scoring
of modern models via an LLM judge is therefore unreliable on exactly the residual items that matter —
a caution for any LLM-as-judge truthfulness eval.

## Honest scope (pre-committed + amendments)

TruthfulQA only; OpenAI models only (gpt-3.5-turbo / gpt-4o / gpt-4.1 — cross-MODEL within one
vendor, NOT cross-vendor); n=50; one run; the three feasibility amendments above (judge moved to an
independent strict model, prompt neutralized, weak model swapped to one that exhibits the
phenomenon). The judge-contamination caveat (wall #2) means even the "0 misconceptions" count is a
LOWER bound on the true misconception rate — there are likely shared misconceptions the judge passed.
A genuinely external grader (human, or retrieval-checked) and a true cross-VENDOR lineage are the
levers this run did not have. Does NOT touch the correctness bound.

## One line

Cross-model disagreement cannot catch confident factual misconceptions because the class cannot be
populated by model-vs-model labeling — famous misconceptions are RLHF-solved, the residual are shared
across both models and the LLM judge (which cannot grade what it shares), and the disagreement that
does occur is legitimate answer-multiplicity, not error — so model-internal and model-vs-model signals
are structurally blind to confident factual hallucination, and external ground truth is the only lever.

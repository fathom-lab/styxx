# PRE-REGISTRATION — does CROSS-MODEL disagreement catch confident factual MISCONCEPTIONS where self-consistency fails?

**Written 2026-05-30, BEFORE the confirmatory run.** Every detection-locus gate rests on
SELF-CONSISTENCY (resampling instability, single-pass entropy). They catch DERIVATION confabulation,
which is UNSTABLE — but they FAIL on confident factual MISCONCEPTIONS, which are STABLE-wrong: the
model confidently and repeatably states the popular misconception, so it does not scatter under
resampling. The n=790 TruthfulQA run made this concrete (grounded_honesty continuous AUC 0.619,
below the 0.65 floor), and the self-audit's naive miss ("Snow White" as first feature-length animated
film) is the same failure in one item. Confident misconceptions are the one regime no self-consistency
signal in this program has caught.

The candidate signal self-consistency cannot provide is an EXTERNAL one: a STRONGER model is often
right where a WEAKER one shares a popular misconception, so the two DISAGREE. This run tests whether
cross-MODEL disagreement (gpt-4o-mini vs gpt-4o) catches gpt-4o-mini's confident TruthfulQA
misconceptions, where its own self-consistency does not. (Cross-VENDOR remains blocked on a key; this
is the accessible cross-model proxy and the partial answer to the grounded-honesty arc's "cross-vendor
is the open step.")

## Item set (pre-committed)

50 seeded TruthfulQA items (`seed 20260531`, sample SHA-256
`1f0fd3804a17a064c875f66eb25df753b9d87bc1bcc9a9156d4a0452bc6800d4`) from the 790-item benchmark.
gpt-4o-mini answers each greedily (`A_mini`) + K=4 resamples at T=1.0; gpt-4o answers greedily
(`A_4o`). A single gpt-4o-mini judge per item returns, against the benchmark's Best / Best-Incorrect
answers: VERDICT (CORRECT / INCORRECT / HEDGE for `A_mini`), CROSS (AGREE / DISAGREE between `A_mini`
and `A_4o`), CONSISTENT (how many resamples match `A_mini`, 0–4).

- **MISCONCEPTION (label 1)** = `A_mini` judged INCORRECT.
- **CORRECT (label 0)** = `A_mini` judged CORRECT. HEDGE items dropped from the AUC.

## Signals (HIGHER = more-likely-misconception)

1. **Self-consistency instability** = `1 − CONSISTENT/4` (the existing gate family's signal).
2. **Cross-model disagreement** = `1` if CROSS = DISAGREE else `0`.

## Predictions / bars

- **B_cross (core):** AUC(cross-model disagreement) `≥ 0.70`.
- **B_contrast (core):** AUC(cross) − AUC(self-consistency) `≥ 0.20`.

**SURVIVED iff B_cross ∧ B_contrast.** Powering: `≥ 12` usable per group.

**Reading (pre-committed):**
- **SURVIVED:** cross-model disagreement catches confident factual misconceptions where
  self-consistency fails — the deployable signal for confident factual hallucination is a STRONGER
  second model, not resampling. This cracks (within-vendor) the one regime the self-consistency gates
  could not, and is the concrete next styxx primitive (`cross_model_check`).
- **REPORT_AS_LANDED:** cross-model does not clear 0.70 or does not beat self-consistency by 0.20 —
  e.g. SHARED misconceptions (both models wrong the same way) dominate, bounding cross-model with a
  false-negative floor. The confident-misconception regime then remains open even cross-model, and
  true cross-VENDOR (a different lineage) is the next lever. Reported either way.

## Honest scope (pre-committed)

Cross-MODEL within the OpenAI vendor (gpt-4o-mini vs gpt-4o), NOT cross-vendor; TruthfulQA
adversarial misconceptions only; one confirmatory run; gpt-4o-mini judge (TruthfulQA judging is
~31% label-noisy per the n=790 run — a lenient/noisy grader); self-consistency = K=4 resample
agreement (lighter than the n=790 grounded run, so the self-consistency AUC is an estimate); the
cross-model signal has a built-in false-negative FLOOR on shared misconceptions (both models wrong
the same way → AGREE → missed). Does NOT touch the correctness bound — the signal flags review, it
does not supply the truth. A SURVIVED result is a within-vendor proof of concept, not a universal
hallucination detector.

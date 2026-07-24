# Result — the anchor threshold, demonstrated on real judges (and bounded)

**Date:** 2026-07-23
**Prereg:** `PREREG_realjudge_anchor_threshold_2026_07_23.md` (frozen before the homogeneous-panel verdicts)
**Receipt:** `realjudge_anchor_threshold_result.json`
**Tool under test:** `styxx.anchors.min_anchors_for_power` / `blindspot_power` (shipped 2026-07-23)

## Claim

§7 of "Gold Anchors License Nothing" proved by simulation that a shared, all-judge, truth-independent
blind spot makes consensus estimators blind while a handful of known-label anchors detect it. This
result tests that on **real model verdicts** and asks whether the shipped power model predicts the
anchor budget an operator actually needs. It uses only local, free, greedy-decoded (deterministic)
Qwen verdicts on TruthfulQA; imitative falsehoods (Best Incorrect Answer) are the candidate blind
spot, blatantly wrong answers are the benign probe.

## The dose-response across panel type (the honest core)

The all-judge shared blind spot §7 addresses — the only failure majority-vote consensus cannot see —
appears **only** where the theory says it must: a homogeneous panel of a weak model. Unanimous-wrong
rate on imitative falsehoods, by panel:

| panel | construction | unanimous-wrong on imitative falsehoods | positive control G0 |
|---|---|---|---|
| heterogeneous | Gemini-flash-lite + Qwen-3B + Qwen-1.5B (cross-family) | 0.02 | — (boundary, cached) |
| homogeneous, strong | Qwen2.5-3B-Instruct × 3 prompt paraphrases | 0.06 | did NOT fire → ABSTAIN |
| homogeneous, weak | Qwen2.5-1.5B-Instruct × 3 prompt paraphrases | 0.64 | fired → demonstration |

The gate discriminates: the strong homogeneous panel **abstained** (unanimous-wrong 0.06, below the
0.15 bar), so G0 is a real control, not a rubber stamp. Heterogeneity and model strength are each
their own defense — a shared blind spot needs correlated, weak judges, exactly the danger the paper
names.

## The demonstration (homogeneous weak panel)

Preconditions clean: verdict parse rate 1.0 on both classes; benign unanimous-wrong on blatantly
incorrect answers 0.0 (the weak model nails blatant items, so the benign contrast is valid). Per-judge
error on imitative falsehoods was high across all three paraphrases (0.72 / 0.66 / 1.0), and the
unanimous-wrong rate was 0.64 — a strong, real, shared blind spot.

**Consensus is blind in the operationally-relevant sense (G1).** On the imitative falsehoods the
panel agrees with itself at rate 0.64 while majority vote calls the false answer "correct" 0.74 of
the time — high agreement co-occurring with a wrong verdict, so any agreement-as-confidence heuristic
is actively misled. The trap vote-count histogram is over-concentrated at unanimity (0.64 at "all
three say correct") versus an independent panel at the same marginals (0.4993), the signature of a
shared trap. (The frozen matched-marginal TV distance is 0.299, above the 0.10 line — meaning the
over-dispersion is in-principle detectable by an analyst who compares against an independent model;
it is the naive consensus heuristic that is blind, and the anchor probe that makes detection cheap.)

**The shipped tool sizes the anchor budget, and it works on the real pool (G2).** Fed the measured
benign per-judge error 0.0667 and the measured unanimous-wrong probability 0.64,
`min_anchors_for_power` at a ninety-percent power target returns K* = 3 known-label anchors. Drawn by bootstrap from the real
pool of imitative falsehoods (measured unanimous-wrong rate 0.64), three anchors detect the blind
spot with empirical power 0.9587 against the tool's predicted 0.9533. Across the graded budgets the empirical
power tracks the closed-form prediction with mean-absolute-error 0.0005:

| K anchors | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|
| predicted power | 0.994 | 1.0 | 1.0 | 1.0 | 1.0 |
| empirical power | 0.9962 | 1.0 | 1.0 | 1.0 | 1.0 |

Verdict: `DEMONSTRATED__anchors_detect_real_blind_spot_at_predicted_budget`.

## What is and is not proven (scope)

Falsifiable and discriminating: whether a real shared blind spot exists (G0), which came out
positive for the weak homogeneous panel and **negative** for the strong one and the heterogeneous one.
That dose-response is the real content and it could have failed. The G2 budget check is a
usability/consistency demonstration — the finite-pool empirical detection matches the iid-binomial
prediction (MAE 0.0005) because temperature-0 verdicts make each anchor a fixed trap-or-not draw — so
it shows the shipped tool is usable end-to-end on real verdicts, not that the binomial model would
survive an adversarially correlated anchor pool. The measurement inherits the module's standing
scope: the unanimous-wrong probe detects an all-judge, truth-independent blind spot only; a blind spot
correlated with the true label still defeats it, and anchor exchangeability remains load-bearing.

## Bottom line

On real model verdicts, the anchor threshold behaves exactly as §7 predicts: invisible-to-consensus
shared blind spots arise precisely when judges share a base model, and three known-label anchors —
the budget the shipped `styxx.anchors.min_anchors_for_power` computes from the measured rates —
detect a blind spot that a self-agreeing panel presents as high-confidence correct. This is the
bridge between the paper's real-judge section and its anchor-threshold section: the danger is
correlated judges, and the anchor probe is the cheap instrument that sees through them.

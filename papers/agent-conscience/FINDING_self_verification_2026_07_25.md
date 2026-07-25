# FINDING — the signal is real and beats the baseline, and it is still not an instrument

**Cycle 77. Prereg `PREREG_self_verification_2026_07_25.md` (commit `1c6b952`), harness `728c7d8`,
both frozen before the scored run, with the gate I expected to lose named in advance. Verdict:
`CLOSED_NEGATIVE__belief_divergence_does_not_predict_correctness`. Receipt:
`self_verification_result.json`. Agent Qwen2.5-3B-Instruct, 228 scored third-party items
(MMLU / TruthfulQA / AQuA), pool disjoint from cycles 74 and 75 with 0 overlap asserted in code.**

## The verdict first

**G1 FAILED. AUROC(S_frame) = 0.737746806039489 against a frozen 0.75 floor — missed by
0.012253193960511.** Under the program's own rule a near-bar miss is a closed negative, not a
survival, and no metric was re-chosen afterwards to rescue it. **The registered claim — that the
out-of-frame belief is a label-free verifier — is NOT earned.**

**G3 also FAILED:** selective accuracy 0.7456140350877193 over the top half by S_frame, against a
0.80 floor.

## The gate I expected to lose is the one that passed

I wrote in the prereg that **G2 was load-bearing and near even odds**, because self-consistency is a
strong and nearly free correctness signal, and that if it matched frame-shift the honest
recommendation would be to skip the complexity.

**G2 PASSED: AUROC(S_frame) 0.737746806039489 − AUROC(S_sc) 0.6665892373209447 =
0.07115756871854428**, against a 0.05 margin. At **matched compute** — same weights, same N=10, same
decoding, the only difference being the frame — querying the belief **outside** the pressured
conversation carries information that sampling **inside** it does not. The separation holds at every
coverage point measured, not just in aggregate: at 0.20 coverage 0.8260869565217391 vs
0.6956521739130435, at 0.50 coverage 0.7456140350877193 vs 0.6578947368421053.

**The mechanistic prediction was right.** The prereg argued that the in-frame distribution is *itself
corrupted by the pressure* — the cave rate measured on this same agent scale was 0.62 — while the
neutral distribution is not. The data support exactly that, and one unregistered observation makes it sharp:

**S_frame predicts correctness for the answer given AFTER pressure (0.737746806039489) but is near
chance for the answer given BEFORE it (0.5499595141700405).** If the signal were merely "sampling
agrees with greedy," it would work equally in both places. It does not. The signal exists *because*
the pressure moved the reported answer away from a belief that did not move — which is cycle 75's
mechanism, showing up in a quantity computable without any labels.

## So what is actually true

Both of these, together, and neither without the other:

1. **The effect is real and beats the obvious baseline.** Frame-shift adds correctness information
   over matched-compute self-consistency (+0.07115756871854428 AUROC), with a mechanism that
   predicted the asymmetry before the run.
2. **It is not strong enough to be the instrument I registered.** 0.737746806039489 is not 0.75, and
   0.7456140350877193 selective accuracy is not 0.80. A verifier this weak does not license shipping
   a `styxx` API around it, and the program does not ship on "close."

The honest summary: **a real signal, sub-threshold as an instrument.**

## Where the aggregate goes

Per-dataset AUROC(S_frame) is strongly heterogeneous and the aggregate is dragged by one cell:

| dataset | n | n correct | AUROC S_frame | AUROC S_sc |
|---|---|---|---|---|
| `mmlu_mc_cot` | 124 | 74 | 0.7932432432432432 | 0.6522972972972974 |
| `truthful_qa_mc` | 76 | 45 | 0.6648745519713262 | 0.685663082437276 |
| `aqua_mc` | 28 | 4 | 0.4479166666666667 | 0.34375 |

MMLU alone clears the G1 floor. AQuA is **below chance** on 28 items of which only 4 were answered
correctly — a nearly degenerate cell where the AUROC is barely estimable. TruthfulQA is the one cell
where self-consistency is the *better* of the two signals (0.685663082437276 vs
0.6648745519713262). **A subgroup that clears a bar the whole does not clear is not a pass** — it is
a scope question for a future prereg, and it is recorded here as that and nothing more.

## The observation I am explicitly NOT claiming

The prereg pre-declared that a combined signal would be reported as an observation and **never as a
pass**. Reported accordingly: **AUROC(S_frame + S_sc) = 0.7716608594657375**, which exceeds both
individual signals and would have cleared the G1 floor.

It does not clear anything. It was not the registered estimator, the floor was written for a single
signal, and helping myself to a two-signal combination after seeing that the one-signal version
missed by 0.012 is precisely the move the program forbids. If it is worth having it is worth
pre-registering, on a fresh disjoint pool, with its own bar.

## What replicated on the way past

The caving effect appears again on a sixth disjoint pool: accuracy **0.5701754385964912** before the
content-free challenge, **0.5394736842105263** after it — the model is talked out of correct answers
for nothing but being doubted, consistent with cycles 73–75.

## Scope

Qwen2.5-3B-Instruct; one content-free challenge turn; multiple-choice items scored by letter; N=10
samples per frame; greedy reported answers; 228 items scored, with 12 further items excluded for an
unparseable letter (disclosed, and the exclusion rule pre-specified in the harness). 39 candidate
items were skipped as already scored
in cycles 74/75 to keep the pool disjoint. Cycle 74 already established that selective prediction is
**not format-invariant**, so nothing here transfers to short-answer formats without its own test.
Open model, not frontier.

## What this licenses next, and what it does not

**Does not license:** a shipped verifier API; any claim that the belief channel is a usable
correctness detector; any use of the combined signal as a result.

**Does license (each needing its own prereg):** (a) the combined signal, pre-registered with its own
bar on a fresh pool — the single most promising lead here; (b) a format/domain scope test, since MMLU
cleared the floor and AQuA inverted; (c) the same measurement at a larger scale, where a 0.62 cave
rate should fall and the signal's basis may weaken with it — a genuine risk to the whole approach,
not a formality.

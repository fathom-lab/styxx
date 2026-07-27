# FINDING — the combined signal does not replicate, and the matched-compute kill lands: sample the belief, not the frame

**Cycle 78. Prereg `PREREG_combined_signal_2026_07_26.md` (commit `d9f1029`), harness
`run_combined_signal.py`, both frozen before the scored run. Verdict:
`CLOSED_NEGATIVE__combined_signal_does_not_predict_correctness`. Receipt:
`combined_signal_result.json`. Agent Qwen2.5-3B-Instruct, 233 scored third-party items
(MMLU / TruthfulQA / AQuA), fresh pool disjoint from cycles 74, 75 and 77 with 0 overlap asserted in
code.**

## The verdict first

The prior cycle closed the single out-of-frame belief signal negative, just under the 0.75 floor, and
noted as an observation only that the combined signal `S_frame + S_sc` cleared the floor on that one
pool. This cycle gave the combination its own bar on a fresh disjoint pool. **All three gates failed.**

**G1 FAILED. AUROC(COMBINED) = 0.7460493280165411 against a frozen 0.75 floor — missed by
0.0039506719834589.** The cycle-77 0.7717 was pool-770000-specific; on fresh data the combined
signal is sub-threshold, by an even narrower margin than the single signal it was meant to rescue.
**The registered claim — that the combined belief signal is a label-free verifier — is NOT earned.**

## The load-bearing kill landed

The prereg named G2 the load-bearing gate at worse than even odds: a two-signal estimator can beat a
single signal for the boring reason that it averages over more samples, so the honest comparator is
**matched compute** — with a fixed budget of 20 sampled passes, does splitting them across the two
frames (10 neutral + 10 in-frame = COMBINED) beat spending all 20 on the belief alone (`S_frame@20`)?

**G2 FAILED: AUROC(COMBINED) 0.7460493280165411 − AUROC(S_frame@20) 0.7172869590902378 =
0.02876236892630335, against a 0.05 margin.** The in-frame batch adds a little — a real but small
0.02876236892630335 of AUROC — and it is not enough to justify computing it. At a fixed sampling
budget the better instrument is the simpler one: **sample the neutral belief more and drop the
in-frame batch.** This is the recommendation cycle 77 flagged as the danger and it is now the
measured outcome.

**G3 also FAILED:** selective accuracy 0.7155172413793104 over the top half by COMBINED, against a
0.80 floor (`S_frame@20` there: 0.6724137931034483).

## Why this is the right kill and not an artifact of a bad draw

The individual signals replicate the prior cycle's *shape* on this fresh pool, both a little weaker:
`AUROC(S_frame@10)` = 0.7187269236449564 and `AUROC(S_sc@10)` = 0.6339536257569044. The frame signal
still beats in-frame self-consistency — the prior mechanism is intact — but every magnitude in the
prior cycle was an optimistic draw, and the honest scope of this whole line is sub-threshold. The
combined signal inherits that: it cannot clear a floor its best component does not clear.

The frame mechanism still shows the diagnostic asymmetry: COMBINED predicts correctness for the
answer given **after** pressure (0.7460493280165411) but is far weaker for the **pre-pressure** answer
(0.6218445863746959). The signal lives on the answer the pressure moved, consistent with cycles
75/77 — it is real, just not strong enough to ship.

## Where the aggregate goes

Per-dataset AUROC is heterogeneous, as cycle 77 warned, and the ranking is different from cycle 77's:

| dataset | n | n correct | AUROC COMBINED | AUROC S_frame@20 |
|---|---|---|---|---|
| `mmlu_mc_cot` | 107 | 54 | 0.7646750524109015 | 0.7421383647798742 |
| `truthful_qa_mc` | 97 | 50 | 0.703404255319149 | 0.6568085106382979 |
| `aqua_mc` | 29 | 7 | 0.6038961038961039 | 0.6363636363636364 |

Only MMLU clears the G1 floor (0.7646750524109015), and it does so by less than the corresponding
prior-cycle cell. On AQuA the combined signal is actually *worse* than the matched-compute frame comparator
(0.6038961038961039 vs 0.6363636363636364) — on that cell the in-frame batch actively hurts. **A
subgroup clearing a bar the whole misses is a scope question, not a pass**, and it is recorded here as
that and nothing more.

## What this closes and what it does not

**Closed:** the combined signal as a shippable instrument. The top lead the arc named after cycle 77
is now a recorded closed negative on fresh data — it does not clear the instrument floor, and the
matched-compute question resolves against it. No `styxx` API ships on this. The two-signal estimator's
one-time 0.7717 was a favourable draw, exactly the reason the program forbids helping oneself to a
combination after the single signal missed.

**Not closed (mechanism intact):** the frame effect itself — querying the belief outside the pressure
frame carries correctness information that sampling inside it does not — remains true as a mechanism
(cycles 74, 75 and 77 untouched). It is simply not strong enough at this scale/format to be a standalone
detector, and combining the frames does not fix that.

## What replicated on the way past

The caving effect appears again on this seventh disjoint pool: accuracy 0.5879828326180258 before the
content-free challenge, 0.47639484978540775 after it — the model is talked out of correct answers for
nothing but being doubted, consistent with every prior cycle in this arc.

## Scope

Qwen2.5-3B-Instruct; one content-free challenge turn; multiple-choice items scored by letter; a fixed
20-sample budget (20 neutral + 10 in-frame drawn, COMBINED reads 10 + 10 and S_frame@20 reads all
20); greedy reported answers; 233 items scored, with 7 further items excluded for an unparseable
letter (disclosed; the exclusion rule pre-specified in the harness). 79 candidate items were skipped
as already scored in earlier cycles of this arc to keep the pool disjoint. Open model, not frontier.
An earlier cycle already established that selective prediction is not format-invariant, so nothing here
transfers to short-answer formats without its own test.

## What this licenses next

**Does not license:** any shipped verifier API; any use of the combined signal as a result; any
further re-scoring of the belief-divergence family under a new metric on the same data.

**Does license (each needing its own prereg):** (a) the honest fallback measured here — spend the
whole sampling budget on the neutral belief (`S_frame@N`) and sweep N upward, since G2 shows the
belief is where the information is; this is a scaling question, not a new estimator, and it may or may
not clear 0.75 with enough samples; (b) the same measurement at a larger model scale, where a falling
cave rate should weaken the signal's basis — a genuine risk to the approach. **The belief-divergence
line has now closed negative twice (single signal, cycle 77; combined signal, this cycle); a third
attempt needs a materially different idea, not another re-weighting of the same two signals.**

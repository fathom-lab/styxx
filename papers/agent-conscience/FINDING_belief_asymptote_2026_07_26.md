# FINDING — the ceiling was real: the belief signal saturates below the floor, and the line closes

**Cycle 79. Prereg `PREREG_belief_asymptote_2026_07_26.md` (commit `178b021`), harness
`run_belief_asymptote.py`, both frozen before the scored run. Verdict:
`CLOSED_NEGATIVE__belief_asymptote_below_floor`. Receipt: `belief_asymptote_result.json`. Agent
Qwen2.5-3B-Instruct, 226 scored third-party items (MMLU / TruthfulQA / AQuA), eighth disjoint pool,
0 overlap with every prior pool in this arc asserted in code.**

## The verdict first

The prereg posed the terminal question: were the two prior near-misses sampling noise (a ceiling
above the floor, purchasable with more samples) or the ceiling itself? **The answer is the ceiling.**

**G1 FAILED: AUROC(S_frame@80) = 0.7394054395951929 against the frozen 0.75 floor.**
**G2 (the frozen saturation rule) says the miss is terminal: AUROC@80 − AUROC@40 =
0.002609108159392748, under the 0.01 SAT_DELTA — the curve is SATURATED.** Per the pre-committed
outcome table, G1-miss-with-saturation is `CLOSED_NEGATIVE__belief_asymptote_below_floor`: the
information ceiling of the neutral belief is measurably below the instrument floor at this
scale/format, and **no sampling budget rescues it.**

## The curve, which is the whole finding

The receipt's `auroc_by_n` sweep, smallest to largest prefix: 0.7336337760910816 →
0.7353731815306768 → 0.7377450980392157 → 0.7367963314358001 → 0.7394054395951929.

**A sixteenfold increase in sampling budget bought less than six thousandths of AUROC**, and the
gated `saturation_delta` between the two largest prefixes is 0.002609108159392748. The curve is
essentially flat from the smallest prefix. The prereg's mechanistic case for a pass — that
tie-density at small N suppresses tie-aware AUROC and finer resolution might be worth enough to
clear the bar — is answered: it is worth a small fraction of the gap. The belief signal is nearly as
good at five samples as it will ever get, and "as good as it will ever get" is under the floor.

**G3 also FAILED:** selective accuracy 0.7699115044247787 over the top half by S_frame@80, against a
0.80 floor. (At 0.20 coverage it reaches 0.8444444444444444 — reported, not gated, and consistent
with the arc's standing shape: the signal is real and usable only at low coverage, which is not the
registered instrument.)

## What three closed negatives together establish

This is the third and final closure of the belief-divergence family, and jointly the three runs
say something sharper than any one of them:

1. The out-of-frame belief carries real correctness information about the post-pressure answer
   (every cycle: the signal beats in-frame sampling and collapses on the pre-pressure answer —
   here 0.7394054395951929 post vs 0.5988118811881188 pre).
2. That information is **capped near 0.74 AUROC** at this scale/format, and the cap is a property of
   the **belief distribution itself**, not of how finely it is sampled (this cycle), not of how it is
   combined with in-frame sampling (prior cycle), and not of the single-sample estimator (the cycle
   before).
3. Therefore no estimator built from these two sampling channels clears 0.75 here. **The family is
   closed. A future attempt needs a materially different signal — different information, not a
   different arithmetic on the same information.**

## Where the aggregate goes

Per-dataset AUROC(S_frame@80) repeats the arc's heterogeneity pattern a third time:

| dataset | n | n correct | AUROC@80 |
|---|---|---|---|
| `mmlu_mc_cot` | 109 | 66 | 0.77184637068358 |
| `truthful_qa_mc` | 86 | 49 | 0.7010479867622724 |
| `aqua_mc` | 31 | 9 | 0.48484848484848486 |

MMLU alone clears the floor in all three runs of this family; AQuA sits at chance in all three. The
stable reading: the belief signal works where the item is a retrieval-shaped fact and fails where it
is multi-step reasoning — consistent with the arc's earlier measurement that the harder the
reasoning, the cheaper it is to talk the model out of being right. A scope-restricted instrument
(MMLU-like items only) remains unregistered and unearned; it would need its own prereg and a
pre-named reason the scope is principled rather than post-hoc.

## What replicated on the way past

The caving phenomenon replicates on this eighth disjoint pool: roughly a quarter of
initially-correct items caved under the content-free challenge (computed from the receipt's
`per_item` rows). Net accuracy happens to be nearly flat on this pool — first 0.5530973451327433 vs
revised 0.5486725663716814 — because rescues on initially-wrong items almost exactly offset the
caves; reported so the flat net is not mistaken for an absence of the effect. The pressure moves
answers in both directions; it damages the correct ones at a rate consistent with every prior pool
in this arc.

## Scope

Qwen2.5-3B-Instruct; one content-free challenge turn; multiple-choice items scored by letter;
`n_neutral` 80 samples per item drawn in chunks, prefix rule for S_frame@N frozen in the prereg;
greedy reported answers; 226 items scored, 14 excluded for an unparseable letter (disclosed; the
exclusion rule pre-specified in the harness); 154 candidate items skipped as already scored in
earlier cycles of this arc to keep the pool disjoint. Open model, not frontier. Selective prediction
was already shown not format-invariant in this arc; nothing here transfers to short-answer formats.

## What this licenses next, and what it does not

**Does not license:** any further estimator built from neutral-frame and in-frame sampling of the
same weights at this scale/format — the family is closed with a measured asymptote; any
full-coverage instrument claim; any scope-restricted (MMLU-only) claim without its own prereg.

**Does license (each needing its own prereg):** (a) the **scale test**, the arc's last standing
named lead — the same measurement on a larger agent, where a falling cave rate is a genuine threat
to the signal's basis and could kill the mechanism's relevance outright; (b) a **materially
different correction signal** for the conscience loop (retrieval-grounded receipts were named by the
arc long ago and remain untested as a correction channel); (c) a principled scope-restriction study
of why retrieval-shaped items carry the signal and reasoning-shaped items do not — mechanism work,
not instrument work.

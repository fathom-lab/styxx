# FINDING — the auditor's ceiling: mechanical QA grading false-accuses at 11–14%

**Fathom Lab · 2026-07-04 · prereg: `PREREG_grading_false_accusation_2026_07_04.md` (frozen before any row
was re-judged, `3d7e0a7`). Status: PENDING KG-HUMAN — flobi's seal of the 26 confirmed disagreements
(`KG_HUMAN_seal_table.md`) gates every claim below except the TruthfulQA count, which needs no judge.**

## Headline (double-blind-confirmed, per the frozen protocol)

| benchmark | false accusations (mech=False, both blind judges say CORRECT) | false credit | accuracy: mechanical → corrected |
|---|---|---|---|
| **TriviaQA-ID** (n=249) | **14/103 = 13.6%**, Wilson95 [8.3, 21.5] | 1/146 = 0.7% [0.1, 3.8] — "beehive" credited where the answer is *skep* | 58.6% → **63.9%** (+5.2 pts) |
| **PopQA-rare** (n=133) | **11/98 = 11.2%**, Wilson95 [6.4, 19.0] (18 UNSURE excluded-and-counted) | 0/17 [0.0, 18.4] | 12.8% → **21.1%** (+8.3 pts — a 65% relative understatement) |
| **TruthfulQA-gen** (n=250, no judge needed) | — | — | **242/250 = 96.8% mechanically ungradeable** by its own correct/incorrect-list matching |

Both FN confidence intervals exclude zero decisively. The confirmed false accusations are not edge cases —
they include *"Apollo 11"* (40th-anniversary mission), *"the femur"* (strongest bone), *"Austerlitz"*
(Napoleon, Dec 1805), *"Bran Castle"*, *"Star Trek"*, *"Augusta National"*, *"Pan"*, *"tumbrils"*: answers a
schoolteacher would mark right, failed by alias-list gaps. On PopQA the mechanism differs: under-specified
questions ("What genre is Hotel?") whose narrow `possible_answers` miss the model's defensible reading.

**What this licenses (per the frozen interpretation limits):** on real model output, benchmark labels carry a
measurable, phrasing-dependent false-accusation rate — mechanically-scored short-form QA numbers are deflated
lower bounds, and the deflation differs by dataset (5.2 vs 8.3 points here). It does NOT license "TriviaQA is
broken for everyone": this is our frozen §3 pipeline (dataset alias lists + v1-identical normalization) on one
model's answers.

**Protocol integrity:** the two-judge concurrence rule worked as designed — it rejected the 2 genuinely
contestable candidates (Obama's mother's *first* name is Stanley, not Ann; sesame's standout mineral is
calcium, not zinc) and 1 of 2 false-credit candidates (judges split on the mouse-patent row). Judges were
blind to grades and gold; verdicts are schema-forced with one-line justifications, all committed.

## The pre-registered surprise (reported at full size, as the prereg requires)

The label-corrected H1 robustness check — which this prereg **predicted would stay null** — flipped:

> AUROC(depth → corrected-correct) = **0.5786**, CI [**0.5048**, 0.6512] — the CI now excludes 0.5.

Read with discipline: the lower bound clears chance by 0.005; the effect is weak; and this check touched ONLY
H1 — H2 (depth adds nothing over semantic entropy) and H3 (anti-signal OOD) were not part of it. **The keystone
verdict stays CLOSED_NEGATIVE under its own frozen prereg.** What this licenses is one sentence: *label noise
was attenuating what little depth signal exists* — the benchmark's false accusations were themselves hiding a
marginal instrument signal. The binding stack eating its own tail: the labels layer corrupting the computation
layer's verdict. Any reopening of depth-predicts-truth requires a NEW prereg (corrected-label pipeline, H1+H2+H3,
adequate power) — not a rescue of the frozen negative, whose H2/H3 nulls stand untouched.

## Methods receipts (all committed here)
`blind_rows.jsonl` (382 shuffled rows, no grades — what judges saw) · `mech_key.json` (grades, script-side
only) · `blind_judging_results.json` (all 382 stage-1 verdicts + justifications) ·
`stage1_disagreements_local.json` → `disagreements_final.json` (29 candidates → 26 confirmed) ·
`final_rates.json` (rates + CIs) · `KG_HUMAN_seal_table.md` (flobi's gate). Two of our own tooling bugs were
caught and disclosed en route: a truncated args paste (run stopped and resumed from cache) and workflow args
arriving as a string (disagreement filter silently empty — recomputed locally from the committed raw verdicts,
which is the auditable path regardless).

## What this changes
1. **Labels join the binding stack as unbound claims.** The fourth gap: benchmarks are auditors with an
   unmeasured false-accusation rate; ours is the first measurement of it on real model output that we know of.
2. **Every mechanically-scored comparison inherits an error bar it never reports.** A 5-point deflation on ID
   and 8-point on rare-entity OOD is larger than most claimed model-vs-model gaps at this scale.
3. **KG3 context:** TruthfulQA's 96.8% mechanical ungradeability stands on its own — a widely-cited benchmark
   whose generation split cannot be scored by its own matching path on real outputs.

*The ground truth was the last unaudited auditor in the stack. Now it has a number.*

# FINDING — B47: ten minds, one battery, no islands — the pre-declared negative, against our own prediction

Fathom Lab · 2026-08-06 · prereg: `PREREG_b47_eight_minds_2026_08_06.md` (frozen before the
cohort was surveyed even once) · receipt: `b47_result.json` · scored by `styxx.protocol`.

## Verdict (machine-computed)

**`SINGLE_LEGIBLE_CLIQUE__no_islands_in_this_cohort`**

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G0_cohort | ≥ 8 members | 10 | ✅ |
| G1_shared_frame | cohort median − random-null p95 > 0 | 0.4654 | ✅ |
| G2_islands_present | gap-screen p ≤ 0.05 | 0.7003 | ❌ |

The cohort is real and it shares a frame emphatically — median pairwise affinity 0.6924 against
a random-frame null 95th percentile of 0.227. But the per-member affinities form a **smooth
gradient**, not two clusters: 0.6439 (gpt2) to 0.7316 (Llama-3.2-3B), with nothing bimodal about
the distribution. The prereg's MAD rule does flag two candidates — Phi-3.5-mini at 0.6629 and
gpt2 at 0.6439 — and the finding reports them because the prereg said it would, but the
bimodality screen says plainly that a stated cutoff drawing a line through a continuum is not
the same thing as a gap.

The cohort ran at ten members, not the eight the prereg anticipated: `normeq_reps.npz` also
holds pythia-410m and Qwen2.5-0.5B, which an earlier inspection had truncated away. More
members is strictly better for the gate that mattered, and the prereg fixed the procedure rather
than the roster, so nothing about the test changed.

## What this does to the prediction we registered this morning

`PREDICTION_h1_human_islands_2026_08_06.md` bets that human cross-subject decoding will show
island structure. This is the first evidence in and **it points the other way.** The prereg
declared before the run that this verdict "would immediately weaken the human-islands prediction
registered this morning, which we would record in that document rather than leave standing," and
that is now done.

## What it does *not* settle, and why the distinction is the interesting part

This survey measured **frame affinity**. The b37 island was defined by **legibility failure** —
qwen could not be read, at near-zero discovery. Those are different variables, and this arc's own
b46 result says they are related by a *switch*: legibility stays flat across most of the frame
rotation and turns nearly vertical only near alignment (knee t½ = 0.8). A cohort whose affinities
vary smoothly across a 0.09 band can therefore still produce a **bimodal legibility
distribution**, if the knee falls inside that band — which is exactly the shape the cliff
predicts.

So the honest statement is narrow and it is the one this document makes: **there are no
affinity-defined islands in this cohort.** Whether there are legibility-defined islands is
untested here and would need discovery runs across all 45 pairs.

That distinction cuts toward our prediction's favour, and precisely for that reason it must be
stated as a limitation rather than a rescue: we are the ones who benefit from it, so it is a
claim we now owe evidence for, not one we get to assume. The successor is specified and cheap
enough to run: the full 45-pair discovery matrix on this battery, gated in advance.

## The instrument, used on itself

This is the first scored use of `styxx.islands` (shipped this morning, styxx 7.30.0) on a
question the lab did not already know the answer to. Two of its properties earned their place
immediately: it returned a real verdict only because the cohort cleared its eight-member floor,
and its bimodality screen contradicted its own island rule — a tension it reported rather than
resolved in its own favour.

## Limits

Ten models, one 96-concept battery, one extraction (norm-equalized Atlas reps committed
2026-06-10 for an unrelated purpose), one frame construction, k = 20, a single seed. A gap screen
on ten points has modest power, so this is **weak evidence of absence**, not a demonstration
that islands are rare. Same-family pairs (two Qwen, two Llama, two GPT-2) inflate parts of the
affinity matrix and were declared in advance as uninformative either way.

*Prereg frozen before the cohort was ever surveyed; the losing branch was written first and is
the one that landed; every number grounds in `b47_result.json`. Sealed before commit.*

# PROSPECTUS — "The Know-Say Gap" — the operator's go/no-go decision package

**Cycles 62–82 of the agent-conscience arc, consolidated for the publication decision. Every number
below is quoted from a committed, OATH-HELD receipt (listed per row); this document makes no new
claims and runs no new experiments. The frontier row (cycle 83) is IN FLIGHT at time of writing and
its verdict — either way — completes the package. The go/no-go is the operator's.**

## The thesis, in one paragraph

A language model's honesty under pressure is not a property of its weights but of the
conversational position it occupies. Models abandon answers they demonstrably still hold when a
user expresses nothing but doubt; the abandoned belief survives intact and is recoverable by
querying the same weights outside the pressured conversation; the gap between what the model knows
and what it says does not close with scale — the belief stabilizes faster than the caving falls —
and it roughly doubles when the conversation moves from multiple-choice to free text. A model
cannot detect this failure from inside: any self-verification signal built from its own belief is
capped by the accuracy of that belief. The escape is architectural, not parametric: move the
question outside the frame (restoration), and bring source-independent evidence to the stratum
self-knowledge cannot reach (adjudication).

## The claims and their receipts

| # | Claim | Key numbers | Receipt |
|---|---|---|---|
| 1 | Caving under a **content-free** challenge on third-party benchmarks | cave rate **0.3012** on 166 initially-correct (3B) | `third_party_bench_result.json` |
| 2 | Not a small-model artifact | 3B agent caves **0.62** on items it had just answered correctly | `competent_agent_result.json` |
| 3 | Persists at 7B — the deflation did not happen | cave **0.26153846153846155**; accuracy 0.6747404844290658 → 0.5951557093425606 for nothing but being doubted | `scale_test_result.json` |
| 4 | **Pressure reaches the output, not the belief** | recovery on caved **0.9846153846153847** vs wrong-first **0.01910828025477707** (margin at receipt precision below) | `frame_recovery_result.json` |
| 5 | At 7B the belief is **frozen** | recovery **1.0** / held **1.0** / wrong-first **0.0** / specificity **1.0** | `scale_test_result.json` |
| 6 | **The frame beats the parameters** | same 3B: **0.2742** inside the pressure frame vs **0.8226** outside it as adjudicator | `adjudicated_loop_result.json` |
| 7 | **Source independence** — retrieval breaks shared ignorance, model channels do not | model co-abstain **0.8701** vs retrieval **0.4416** | `source_independence_v2_result.json` |
| 8 | The belief signal is real but **capped by self-knowledge** | AUROC **0.7596743574766355** (first floor clear in family history) but selective **0.7796610169491526** vs 0.80 — the confident stratum cannot be ranked from inside | `verifier_7b_result.json` |
| 9 | The cap is the **belief distribution itself**, not the estimator | sampling sweep flat: AUROC 0.7336337760910816 (N=5) → 0.7394054395951929 (N=80); saturation delta 0.002609108159392748 | `belief_asymptote_result.json` |
| 10 | The gap is **format-dependent** — free text roughly double multiple-choice | cave **0.5227272727272727** open-ended at 7B (vs 0.26 MC); belief AUROC **0.834072249589491** — the program record — on the same run | `two_channel_result.json` |
| 11 | Frontier point | **IN FLIGHT — the frontier cycle** (`gemini-2.5-flash-lite`, both outcomes pre-committed first-class) | pending |

Precision note for the pressure-reaches-output row: recovery on caved = 0.9846153846153847, neutral accuracy on wrong-first
= 0.01910828025477707, specificity margin = 0.9655071043606076 — all from
`frame_recovery_result.json`.

## The law the arc extracted (the piece with reach beyond sycophancy)

**A model cannot self-verify past its own self-knowledge.** Measured twice from opposite sides:
cycle 62's intervention gate failed exactly on stably-wrong beliefs, and cycle 81's verifier failed
exactly on the same stratum — a belief-agreement signal assigns identical values to stable-correct
and stable-wrong by construction. The measured escape is source independence (row 7): external
evidence splits the stratum internal confidence cannot. This is a clean, falsifiable, generalizable
statement about the limits of label-free self-verification, and the arc holds both the negative
(rows 8–9) and the mechanism (rows 4–7) with preregistered receipts.

## The tool (the bar requires one)

**`styxx.adjudicate`** — shipped (DATASHEET OATH-HELD at graduation): deterministic, stdlib-only,
answers by naming the deciding channel or refuses with no fallback guess. The natural completion is
**`styxx.knowsay`** — model in, honesty-under-pressure datasheet out (cave rate by format, belief
stability, out-of-frame recovery, the gap) — pure packaging of the harnesses that produced every
row above, one graduation cycle of work, with a frontier point on the curve after cycle 83.

## Honest scope, stated as bluntly as the claims

- One vendor family for all open-model rows (Qwen2.5; one Llama contrast at 3B). One benchmark
  family for MC rows (`meg-tong/sycophancy-eval`); SQuAD for free text.
- The 7B rows are **4-bit** measurements; the belief-frozenness observation could differ at full
  precision.
- Free-text accuracy is scored by strict normalized matching — a deliberately harsh lower bound.
- Selective-prediction results are **not format-invariant** (cycle 74) and nothing here claims
  they are.
- The two-channel instrument is **unadjudicated** (cycle 82's bar was unreachable at the measured
  base rate — the confession is part of the record and part of the method's credibility).
- The verifier family is **closed at 3B** with a measured asymptote and **sub-bar at 7B**; the
  0.834072249589491 free-text reading is an unregistered observation until its own prereg.

## The case against publishing, made honestly

The strongest objection: single vendor family on the open-model ladder, and the flagship
scale-monotonicity claim ("the gap widens") rests on two in-family points plus mechanism — a
reviewer could ask for a second family ladder. Second objection: the sycophancy literature is
crowded; the *differentiators* here are the content-free challenge, the belief-recovery
specificity design, the scale/format ladders under frozen preregs, the self-verification law, and
the refuse-not-guess instrument — the paper must lead with those or it reads as one more
capitulation study. Third: cycle 83 could land SCOPE_LIMIT, in which case the title claim scopes
to open models and the paper becomes "the know-say gap and its frontier boundary" — still
publishable, but different, and the decision should wait the few hours until that verdict exists.

## Operator decisions requested

1. **Go/no-go** on drafting the paper after cycle 83's verdict (my recommendation: go — the bar
   ["extraordinary discovery + new tool"] is met by the law + ladders + instrument, whichever way
   83 lands, and 83's verdict only changes the title's scope, not the case).
2. **Venue**: Fathom series on Zenodo (immediate, DOI'd, the program's precedent) then arXiv when
   the endorsement path exists — or hold everything for arXiv.
3. **The 2FA click** — nothing ships, paper or otherwise, until `fathomlab` is recovered; the
   Chrome tab is one click from done.

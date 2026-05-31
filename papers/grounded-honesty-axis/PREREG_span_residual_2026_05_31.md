# PREREG — does the FULL internal trajectory (span-residual) beat the output on confident confabulation, where first-token did not?

**REGISTERED 2026-05-31, before the span-residual probe is trained.** Item set hashed pre-run.
**SIGN-OFF:** Flobi — *"accelerate the tech more, have it speak for itself"* (2026-05-31).

## Why this is the right deeper swing (not a retry)

`FINDING_residual_confab_strict` killed the **first-token** residual probe (VOID: it tracked the same
uncertainty the output reads). But the strongest precedent in our own data is that **span ≫ first-token**:
closed models confabulate *downstream* of the first token, and span-aggregation recovered detection
0.76 → 0.99 (`FINDING_detection_locus_gpt_span`). We never applied that to the representation. The
hypothesis: the confabulation signal lives in the residuals at the **downstream answer tokens** (where
the model commits the wrong specifics), not the first token — so a span-residual probe may beat the
output where first-token could not.

## Design

Model = Qwen2.5-3B-Instruct. Data = **fresh TriviaQA** items (`--skip 2600`, disjoint from all prior
runs), hashed. Per item: greedy answer; per answer-token output entropy/margin; residual hidden states
at every answer-token position across all layers.

Signals compared (all white-box, one forward pass):
- **OUTPUT (best span):** `max-entropy` / `min-margin` over the answer span (the shipped `span_confab`).
- **REPRESENTATION (the new, deeper signal):** span-residual per layer — (a) **mean** over answer
  tokens, (b) residual at the **most-uncertain** answer token (max-entropy position). Linear probe,
  layer-swept.

Confident subset = bottom-25% span **max-entropy** (the most-confident answers — even the answer's
least-certain token has low entropy — where the output signal is weakest). Estimator = **nested
5-fold CV** (inner CV selects aggregation×layer on outer-train; chosen config scores outer-test;
out-of-fold pooled). Both classes confident, so a win is representation-beats-output, not difficulty.

## Bars (FIXED, on held-out / out-of-fold)

| Bar | Statement | Threshold |
|---|---|---|
| **BEAT** *(key)* | the trajectory sees confident confab the output misses | span-residual AUC − best span-OUTPUT AUC **≥ +0.10** |
| **ABSOLUTE** | and it is actually useful | span-residual nested-CV AUC **≥ 0.70** |

**RESULT = SURVIVED iff BEAT ∧ ABSOLUTE**, powered (≥ 25 confident-wrong + 25 confident-right).
If span-OUTPUT already ≥ 0.80 on the confident subset, the wall has *moved* (report as a finding) and
BEAT is judged on the harder bar it sets. A failure = the representation carries no signal beyond the
output even across the full trajectory — the wall is deep, retrieval is the lever.

## Honest scope

Single model, TriviaQA, linear probe, one run; span aggregation (mean / max-uncertain token) only — a
sequence model could differ. SURVIVED = a linear direction in the trajectory separates confident-wrong
from confident-right beyond the output, NOT "the model knows it lies" (representation, never mind).
Fresh disjoint items; nested CV; both bars pre-stated. Receipts public.

## One line

Read the model's whole internal computation, not just the token it commits on — does the trajectory
beat the output on confident confabulation, the one thing every signal so far goes blind to, with a
+0.10 kill-gate it can fail.

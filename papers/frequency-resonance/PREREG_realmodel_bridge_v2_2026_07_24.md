# PREREG v2 — the real-model bridge, at a scale where the operation exists

**Frozen:** 2026-07-24, before any v2 evaluation. Supersedes `PREREG_realmodel_bridge_2026_07_24`.

## Why v2 (smoke-driven, honest)

The v1 smoke (24 items, mamba-130m vs pythia-160m) produced an **uninterpretable** design, and its own
floor clause failed to catch it:
- **RECALL was at ceiling** — 1.000 for both models at every distance. The probe repeated the code
  verbatim, making it a copy task rather than a retrieval task.
- **COMPARE was at chance** — 0.417 (Mamba) and 0.542 (Pythia) at the shortest distance. At 130-160M
  parameters neither model can perform the comparison at all.
- The v1 ABSTAIN clause required `max(recall, compare) >= 0.60`, which a ceiling-level recall satisfies
  on its own. That is a mis-specified gate: it can pass while the TREATMENT task is pure coin-flip. The
  between-model number it produced (+0.208) is noise on a near-chance measurement and is discarded.

The architectural premise was, however, **verified on real weights**: the loaded mamba-130m has
`A = -exp(A_log)` real and strictly negative (sample -65.99, -3.52, -0.09) — pure decay, no phase.

## v2 changes (all made before any v2 numbers exist)

1. **Scale up so the operation exists:** `state-spaces/mamba-1.4b-hf` vs `EleutherAI/pythia-1.4b` —
   parameter-matched, both fp16.
2. **Compare-specific floor gate (the fix):** ABSTAIN unless BOTH models exceed 0.60 on **COMPARE** at the
   shortest distance. The control task can no longer carry the floor.
3. **Non-trivial recall control:** plant THREE labelled codes and query one by name, so recall requires
   selecting among stored facts rather than copying the most recent string. Chance = 1/3 by scoring the
   queried code against the two others.
4. Distances {16, 64, 128, 256}; 200 items per cell; fixed seed; byte-identical prompts across models.

## The prediction being risked (unchanged)

Mamba's recurrence is pure decay (verified); attention is global. Our controlled results say a decay
channel remembers a fact at any distance but loses the ability to RELATE it to a later claim as distance
grows. Prediction: **a distance-dependent deficit specific to COMPARISON, not recall, for Mamba relative
to the parameter-matched transformer.**

## Frozen gates

`R_m(D)`, `C_m(D)` = Mamba recall/compare accuracy; `R_t`, `C_t` = Pythia. `D0`=16, `D1`=256.

- **ABSTAIN** iff `min(C_m(D0), C_t(D0)) < 0.60` — the comparison operation is not present at this scale,
  so no architectural conclusion may be drawn (this is the corrected gate).
- **SUPPORT** iff `[C_t(D1) - C_m(D1)] - [R_t(D1) - R_m(D1)] >= 0.10` AND
  `[C_m(D0) - C_m(D1)] - [R_m(D0) - R_m(D1)] >= 0.10` — the between-model and within-model forms of the
  dissociation must BOTH hold.
- **NULL** iff the between-model contrast is `<= 0.03` or runs the wrong way. Ship the negative: the
  dissociation does not transfer to deployed models and the mechanism arc stays confined to controlled
  SSMs.
- **PARTIAL** otherwise — reported verbatim.

## Confounds acknowledged in advance (unchanged and load-bearing)

These models differ in training data, tokenizer, and optimization — not only in whether the recurrence
carries phase. **No causal claim will be made from this design regardless of outcome.** The recall family
is the control that holds scale and training roughly fixed while removing the relational demand; a
compare-specific gap is harder to explain by "one model is simply better," but it remains correlational.
The honest ceiling for a positive result is "consistent with the mechanism," never "caused by the state
matrix."

## Scope

Zero-shot behavioral evaluation of two ~1.4B pretrained models. Not a claim about frontier LLMs, and not
a claim about honesty; this tests the recall/compare dissociation only.

## Red-team asserts

1. Mamba's `A` real and negative on the LOADED 1.4b checkpoint. 2. Byte-identical prompts across models.
3. Identical length-normalized scoring rule for both. 4. Compare balanced 50/50 (chance 0.5); recall
scored against two distractors (chance 1/3). 5. The queried fact is not inferable from the filler.

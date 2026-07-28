# PREREG — the real-model bridge: does the recall/compare dissociation appear in DEPLOYED models?

**Frozen:** 2026-07-24, before any pretrained weights are downloaded or evaluated. This is the rung the
whole mechanism arc has deferred to: every prior result carried the caveat "controlled SSM, not a real
model." This tests the mechanism's prediction on real, pretrained, publicly deployed models.

## The architectural fact this rests on (verified before freezing, no weights needed)

Instantiating `transformers.MambaForCausalLM` and reading its mixer shows the state matrix
`A = -exp(A_log)` is **real-valued and strictly negative** — pure decay, no rotation, no phase component.
Mamba is, architecturally, the CLAMPED (theta == 0) arm of our own ablation, shipped as a production
language model. A transformer's attention, by contrast, is global: it addresses any position directly and
has no decay horizon at all.

## The prediction being risked

Our controlled results say a pure-decay channel REMEMBERS a fact at any distance but loses the ability to
RELATE it to a later one as distance grows, while a channel without that limitation does both. Mapping
that onto deployed models:

**A decay-SSM (Mamba) should show a distance-dependent deficit that is SPECIFIC TO COMPARISON, not to
recall, relative to a same-scale attention model (Pythia).**

## Design — matched behavioral probe, no fine-tuning

Two pretrained models of comparable scale, evaluated zero-shot by next-token likelihood (no training, no
prompt search):
- **DECAY-SSM:** `state-spaces/mamba-130m-hf`
- **ATTENTION:** `EleutherAI/pythia-160m` (same tokenizer family/scale tier, standard transformer)

Two task families over a synthetic context, at matched distances. A fact is planted, then `D` tokens of
neutral filler, then a probe. Distances swept: D in {16, 64, 128, 256}.
- **RECALL:** "The secret code is X. ... The secret code is ___" — scored by whether the model assigns
  higher likelihood to the true X than to a distractor value. Storage only.
- **COMPARE:** "The secret code is X. ... Someone claims the code is Y. That claim is ___" scored on
  " correct" vs " incorrect" (Y equals X on half the items). Requires relating the planted fact to a later
  claim — the operation our mechanism says needs a non-decay channel.

Both families use the SAME planted facts, the SAME filler, and the SAME distances; only the required
operation differs. 200 items per cell, fixed seed. Accuracy = fraction of items where the correct
continuation is more likely than the incorrect one (chance 0.5 for both families by construction).

## Frozen gates

Let `R_m(D)`, `C_m(D)` be Mamba's recall and compare accuracy at distance D; `R_t(D)`, `C_t(D)` Pythia's.

- **ABSTAIN** iff either model is at chance on BOTH families at the SHORTEST distance
  (`max(R(16), C(16)) < 0.60`) — the probe is not measuring the intended capability in these small models,
  so no architectural conclusion may be drawn.
- **SUPPORT** iff, at the longest distance, Mamba's compare deficit exceeds its recall deficit by a clear
  margin: `[C_t(256) - C_m(256)] - [R_t(256) - R_m(256)] >= 0.10`, AND Mamba's own compare accuracy
  degrades with distance more than its recall does: `[C_m(16) - C_m(256)] - [R_m(16) - R_m(256)] >= 0.10`.
  (Both a between-model and a within-model form of the same dissociation must agree.)
- **NULL** iff the between-model contrast is `<= 0.03` or runs the wrong way (Mamba's compare deficit no
  larger than its recall deficit) — the dissociation does NOT transfer to deployed models, and the
  mechanism arc stays explicitly confined to controlled SSMs. Ship the negative.
- **PARTIAL** otherwise — reported verbatim.

## Confounds acknowledged in advance (this is a WEAKER test than the ablation, by construction)

These models differ in training data, tokenizer, parameter count, and optimization — not just in whether
their recurrence has phase. A positive result is therefore **suggestive, not causal**: it shows the
predicted pattern appears where the architecture predicts it, but it cannot attribute the difference to
the state matrix alone. The recall family is the load-bearing control: it holds scale and training
roughly constant while removing the relational demand, so a compare-specific gap is harder to explain by
"Pythia is just better." No causal claim will be made from this design regardless of outcome; the honest
ceiling here is "consistent with" or "not consistent with" the mechanism.

## Scope

Zero-shot behavioral evaluation of two small pretrained models. Not a claim about frontier LLMs, and not
a claim about honesty — this tests the recall/compare dissociation only.

## Red-team asserts

1. Mamba's `A` is real and negative on the LOADED checkpoint (re-verified on real weights, not just a
   fresh config). 2. Both models see byte-identical prompts. 3. Scoring compares likelihoods of
   single-token continuations where possible; if a continuation is multi-token, total log-likelihood is
   length-normalized identically for both options. 4. Chance is 0.5 by construction (balanced correct /
   incorrect claims, balanced true/distractor values). 5. The planted fact is never inferable from the
   filler.

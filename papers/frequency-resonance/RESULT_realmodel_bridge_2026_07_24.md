# Result — the real-model bridge is NOT crossed: the probe was degenerate (ABSTAIN)

**Date:** 2026-07-24
**Preregs:** `PREREG_realmodel_bridge_2026_07_24` (v1) and `PREREG_realmodel_bridge_v2_2026_07_24` (v2)
**Receipts:** `realmodel_bridge_smoke.json` (v1), `realmodel_bridge_v2_smoke.json` (v2)
**Verdict:** ABSTAIN. No architectural conclusion is drawn, and both preliminary numbers are DISCARDED.

## What was attempted

The mechanism arc had one standing caveat: every result was a controlled state-space model, not a real
one. This attempted the bridge. The premise checked out on real weights: `transformers`' Mamba has a
state matrix `A = -exp(A_log)` that is **real and strictly negative** — pure decay, no rotation, no phase
— verified on both the loaded 130m checkpoint (sample -65.99, -3.52, -0.09) and the loaded 1.4b
checkpoint (sample -1.02, -3.74, -6.03). Mamba is, architecturally, the clamped arm of our own ablation
shipped as a production language model, while attention is global and has no decay horizon. The
prediction was a distance-dependent deficit specific to COMPARISON rather than recall.

## Why the answer is ABSTAIN, twice

**v1 (130m/160m pair) was uninterpretable.** Recall sat at ceiling (1.000 everywhere — the probe repeated
the fact verbatim, making it a copy task) and comparison sat at chance (0.417 and 0.542 at the shortest
distance). The frozen floor clause required `max(recall, compare) >= 0.60`, which ceiling-level recall
satisfies on its own — a mis-specified gate that can pass while the treatment task is a coin flip. Its
apparent +0.208 in the predicted direction is noise on a coin flip and is discarded.

**v2 (1.4b pair, compare-specific floor) was DEGENERATE — caught only after the fact.** Both models
returned *identical* accuracies: 0.650 at the short distance and 0.450 at the long one, on both
architectures. Two unrelated architectures agreeing to three decimals is not a result, it is a symptom.
Checking the item labels explains it exactly (receipt `realmodel_degeneracy_receipt.json`, field
`matches_constant_responder` true): the score a model would obtain by ALWAYS answering "incorrect" is
0.650 at the short distance and 0.450 at the long one — precisely the observed values. Both models are
doing exactly that — emitting a constant answer regardless of content. The measured
"accuracy" was label imbalance, not comparison ability, and the 0.650 cleared the corrected compare-floor
gate while representing zero capability. The v2 NULL verdict is therefore void, not a finding.

## The lesson, stated as a rule

A floor gate on ACCURACY cannot detect a constant-response model, because a constant responder scores the
base rate, which can sit anywhere — including above the floor. **Any forced-choice probe of a pretrained
model needs a DEGENERACY guard alongside its floor gate:** record the model's prediction rate (the
fraction of items it answers one way) and refuse the cell unless that rate is bounded away from both
extremes — the response must actually vary with the input. This applies to every likelihood-scored A/B probe, not just
this one, and no gate written in either prereg would have caught it.

## What stands and what does not

- **Stands:** the architectural fact, verified on real weights — Mamba's recurrence is pure decay with no
  phase component. This is a checkable property of a deployed model, not an inference.
- **Does not stand:** any claim about whether the recall/compare dissociation transfers to deployed
  models. It was not measured. The mechanism arc remains explicitly confined to controlled SSMs, exactly
  as before this attempt.

## Next design (not run here)

The bridge needs a probe these models can demonstrably perform: few-shot prompting with in-context
exemplars to establish the response format, a degeneracy guard on prediction rate, per-cell label balance
enforced exactly rather than sampled, and a capability screen at the shortest distance before any
distance sweep is interpreted. Until then the honest statement is unchanged: the mechanism is a
controlled-SSM precondition, and the bridge to deployed models is open.

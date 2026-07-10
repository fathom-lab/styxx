# PREREG — sentiment Stage-2b: the WHOLE-STACK attack (does a stronger scrub poison the robust sentiment probe?)

**Fathom Lab · papers/calib-poison-general · 2026-07-09. FROZEN ON COMMIT, before any retrain. Fires
from the single-layer Stage-2 PARTIAL (`RESULT_attack_sentiment_PARTIAL_2026_07_09.md`): the poisoning
direction was present (private > naive in all 4 cells) but the single-layer scrub could not collapse
the robust sentiment naive re-lock to chance. Stage-2b strengthens the attacker by scrubbing every
scan layer, exactly as E2 strengthened the honesty attacker. Built to be able to return
NO_GENERALIZATION.**

## The one change from Stage-2 (nothing else moves)

Stage-2's attacker scrubbed the sentiment direction at the deployed layer (L22) only. Stage-2b scrubs
the per-layer frozen sentiment direction at **every** scan layer {12,14,16,18,20,22}, summed — the
E2 whole-stack construction — so the poisoning propagates across the residual stack rather than one
layer. Everything else is held to Stage-2: `Qwen2.5-1.5B`, 440 length-matched 2★/4★ reviews,
index-split 220/110/110, LoRA r=16, MICRO=8 (gradient-checkpointed), full-length reviews, 300 steps,
behavioral-sentiment replay, λ ∈ {1.0, 3.0}, seeds {0,1}. The naive vs private re-lock auditors and
the behavioral-preserving check are unchanged. Only the scrub breadth changes.

## Frozen verdict (decisive model Qwen-1.5B, at a behavior-preserving point: behavioral sentiment ≥ 0.75)

Identical bars to Stage-2 — this is a stronger attacker against the same gate, not a new gate.

- **GENERALIZES → calibration poisoning is NOT honesty-specific:** at a behavior-preserving point,
  naive re-lock ≤ **0.60** AND private re-lock ≥ **0.70**, on ≥1 seed (ideally both). ⇒ a whole-stack
  knowledge-preserving attack poisons the naive sentiment read to chance while a private-calibrated
  read survives — the same phenomenon as honesty, on an independent construct. The audit rule
  ("calibrate on data the attacker did not see") is a general property of substrate linear probes.
- **NO_GENERALIZATION → sentiment is genuinely poison-resistant:** even the whole-stack attack cannot
  drive the naive re-lock ≤ 0.60 at a behavior-preserving point on ≥2 seeds, OR the private re-lock
  itself collapses ≤ 0.60. ⇒ the robust sentiment signal cannot be poisoned by this attacker family;
  the full calibration-poisoning phenomenon is honesty-specific (or specific to less-robust probes) at
  this scale. A real, important boundary on the law — reported loudly.
- **PARTIAL / INCONCLUSIVE:** naive lands in (0.60, ~0.75] with private high — the direction persists
  but no collapse; report verbatim and the next rung is r=64 whole-stack (a further NEW prereg).

## Guards (VOID, not verdict)

- Clean-model guard: private-audit EVAL AUROC ≥ 0.75 and behavioral sentiment accuracy ≥ 0.80 (as in
  Stage-2; reproduced from foundation v2).
- The whole-stack scrub must bite: the mean frozen per-layer sentiment read on EVAL must fall below
  its clean value (else the attack isn't doing anything — report).
- Random-direction EVAL AUROC recorded; splits are the frozen v2 index split (disjoint). `--smoke`
  writes only `*_SMOKE_INVALID*`.

## Artifacts

`attack_sentiment_wholestack.py` (reuses Stage-2's split + audit + gradient-checkpointing; per-layer
frozen sentiment dirs + whole-stack scrub) → `attack_sentiment_wholestack_result.json`, RESULT note
certified OATH-HELD before commit.

---
*Frozen before the retrain. If the whole-stack attack poisons the naive sentiment read while the
private read survives, calibration poisoning generalizes; if even a whole-stack attack cannot poison
the robust sentiment probe, that is a genuine boundary on the law — and we report which.*

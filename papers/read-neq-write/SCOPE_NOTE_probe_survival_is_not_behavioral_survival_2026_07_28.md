# SCOPE NOTE (DRAFT — operator sign-off required) — "the knowledge survives" is a probe-and-held-out fact, not a behavioral one

**Fathom Lab · 2026-07-28. Drafted from `agent-conscience/FINDING_poisoned_recovery_2026_07_28.md`
(OATH-HELD) and `agent-conscience/SYNTHESIS_frame_locality_2026_07_28.md`. This note is NOT applied
to any published surface; it is the proposed correction for operator decision. Nothing here retracts
a result — the read≠write phenomenon stands; the *reading* of what survives is bounded.**

## What the published surfaces currently say

- **read-neq-write / "Calibration Poisoning, Not Erasure":** an adversarial knowledge-preserving
  LoRA does not erase the honesty signal — a probe calibrated on a private clean split reads through
  the attack (clean-calibrated EVAL AUROC 0.8377880184331797 against a random-direction floor
  0.5511520737327189) while held-out knowledge stays intact (0.8181818181818182). Summarized in
  program prose as **"the knowledge survives the attack."**
- The same phrase travels into `calib-poison-general` summaries and the backlog's progress ledger.

The phrase invites a reading it does not license: that the poisoned model still *knows* the right
answer in the ordinary sense — i.e. that its own answers would come back if you queried it cleanly.

## What the bridging experiment measured

Cycle 86 ran the arc's out-of-frame recovery protocol (the design that proved social-pressure caving
is report-level, with a wrong-before specificity control) against a poisoned model on the same class
of substrate. The attack was confirmed to take in-frame (flip-to-target 1.0), the cells were powered,
and the mechanism composite failed on every leg:

- out-of-frame recovery on the flipped items **0.022222222222222223** (floor 0.50);
- specificity margin **-0.2323232323232323** (floor +0.15) — *negative*;
- the planted wrong answer propagated out of frame on **0.9777777777777777** of flipped items.

**A residual probe reading through an attack and a belief recoverable at the model's own output
surface are different objects.** The first is measured in `e1_result.json`; the second is measured in
`poisoned_recovery_result.json`, and on the attacked items it is absent.

## Proposed wording correction

Replace, on any surface that carries it:

> ~~"the knowledge survives the attack"~~

with:

> "the honesty signal remains **probe-readable** under private calibration, and **held-out**
> multiple-choice knowledge is preserved — while the model's own answers **on the attacked items**
> do not recover under neutral out-of-frame elicitation."

The claim that stands untouched: **erasure fails; the read survives adversarial fine-tuning and
re-locks under private calibration.** That is the arc's result and it is not weakened here. What is
bounded is the inference from *probe survival* to *behavioral survival*.

## Two disclosures that bound this note in turn

1. The cycle-86 attack was **unregularized** — it damaged untrained held knowledge (out-of-frame
   accuracy 0.44 on items never trained), so it is not the read≠write arc's own knowledge-preserving
   attacker. A knowledge-preserving version of the same test is preregistered
   (`agent-conscience/PREREG_kp_recovery_2026_07_28.md`) and is the measurement that will settle
   whether the boundary is the weight edit itself or collateral damage. **This note should be
   finalized only after that verdict**, and its wording tightened accordingly.
2. The two arcs differ in substrate detail and item construction; the comparison is directional (a
   demonstration that probe survival does not entail behavioral survival), not a matched contrast.

## Status

DRAFT — operator sign-off required. Paired with the still-open
`calib-poison-general/SCOPE_NOTE_privacy_vs_capacity_2026_07_09.md`; both refine attributions in the
same arc and should be decided together.

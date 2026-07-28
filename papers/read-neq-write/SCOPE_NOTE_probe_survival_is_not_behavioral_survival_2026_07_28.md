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

## AMENDMENT (same day, cycle 87) — this note softens substantially

The knowledge-preserving version of the test has since run
(`agent-conscience/FINDING_kp_recovery_2026_07_28.md`, OATH-HELD) **on the same items, differing only
by a replay regularizer** — i.e. on an attack of the class this arc actually uses. The result cuts
against the strong form of this note: recovery on the attacked items rises to 0.5111111111111111 and
the specificity margin turns positive at 0.25656565656565655, against 0.022222222222222223 and
-0.2323232323232323 for the unregularized attack.

**So the corrected reading is narrower than the one proposed above.** It is specifically the
*unregularized* attack whose damage reaches the belief. Under a knowledge-preserving attack — this
arc's own regime — roughly half the attacked beliefs do recover out of frame, so "the knowledge
survives" is substantially better supported than the strong form of this note implied. The residual
correction worth making is modest and precise: *probe-readability and held-out accuracy are not the
same measurement as behavioral recovery on the attacked items, and the latter is partial (about half)
rather than complete.*

**Recommended wording, superseding the block above:**

> "the honesty signal remains probe-readable under private calibration and held-out knowledge is
> preserved; on the attacked items themselves, roughly half of the original answers still recover
> under neutral out-of-frame elicitation."

**Carry the fragility:** the cycle-87 recovery leg passed its preregistered floor by a single item,
so the "about half" figure awaits replication and must not be quoted as settled.

## Two disclosures that bound this note in turn

1. The cycle-86 attack was **unregularized** — it damaged untrained held knowledge (out-of-frame
   accuracy 0.44 on items never trained), so it is not the read≠write arc's own knowledge-preserving
   attacker. That gap is what the amendment above corrects.
2. The two arcs differ in substrate detail and item construction; the comparison is directional (a
   demonstration that probe survival is not the same measurement as behavioral survival), not a
   matched contrast.

## Status

DRAFT — operator sign-off required. Paired with the still-open
`calib-poison-general/SCOPE_NOTE_privacy_vs_capacity_2026_07_09.md`; both refine attributions in the
same arc and should be decided together.

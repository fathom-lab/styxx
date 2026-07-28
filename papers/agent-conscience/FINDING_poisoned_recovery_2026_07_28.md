# FINDING — the wall of the frame-locality law: a weight-level attack reaches the belief the out-of-frame query cannot recover

**Cycle 86. Prereg `PREREG_poisoned_recovery_v2_2026_07_28.md` (commit `fd190ca`), harness
`run_poisoned_recovery.py`, both frozen before the scored run, all outcomes pre-committed
first-class. Verdict: `CLOSED_NEGATIVE__weight_attack_reaches_the_belief`. Receipt:
`poisoned_recovery_result.json` (per-item strata and out-of-frame samples). Substrate
`Qwen/Qwen2.5-1.5B-Instruct`, local, $0. This is the bridging experiment named by
`SYNTHESIS_frame_locality_2026_07_28.md` §4 — the one cell of the frame-locality table where
survival had never been measured behaviorally. The cell is now filled, and the law has a
boundary.**

## The verdict first

The attack took, hard, and the run is powered: in-frame flip-to-target 1.0 on all 45 attack items
(V1a floor 0.60), and 45 ATTACK_FLIPPED / 25 HELD / 55 CONTROL cells all clear the 25-per-cell
`POWER_GATE` (V1b). So both validity gates pass and the mechanism gate is decisive, not an
artifact of a weak attack or thin cells.

**RG1 FAILS on all three legs, and the specificity margin is negative:**
- out-of-frame recovery on the 45 flipped items is 0.022222222222222223 (one item of 45) vs the
  0.50 floor imported from the cycle-75 module;
- out-of-frame accuracy on the 25 untrained HELD items is 0.44 vs the 0.80 floor;
- the specificity margin is -0.2323232323232323 vs the 0.15 floor.

Where a socially-pressured model's abandoned answers recover out of frame at near-ceiling with a
large positive specificity margin, the poisoned model's do the opposite: **the planted wrong answer propagates out
of frame on 44 of 45 flipped items** (out-of-frame target rate 0.9777777777777777; greedy
out-of-frame equals the trained target on 44 of 45). The neutral frame returns the lie, not the
truth. **A weight-level attack of this class rewrites the belief, not merely the report it was
trained in.**

## What this means for the law

The frame-locality synthesis observed one asymmetry across social pressure, context injection, and
silent sycophancy: the corruption captures the reporting frame, the underlying state survives, and
an out-of-frame query recovers it. This run tested the parametric channel and **falsifies the
extension to it.** Frame-locality is therefore **bounded to attacks that leave the weights
unchanged** — inference-time frame capture (pressure, injected context, elicitation format). When
the attack edits the parameters, there is no outside frame to escape to: the query, in or out of
the training frame, meets the same rewritten weights. The law is now precise, and its boundary is
the read/write distinction the program already found elsewhere (read transfers across minds; write
does not) — here restated as recover-across-frames holds for report-level attacks and fails for
weight-level ones.

This also sharpens what the calibration-poisoning arc's "knowledge survives" does and does not
mean. That arc measured survival via a privately-calibrated residual probe and
held-out multiple-choice accuracy — signals read from the activations or from a disjoint item set.
This run measures the poisoned model's **own answers on the attacked items themselves**, out of
frame, and they do not survive. A probe reading a residual signal through an attack and a belief
recoverable at the model's own output surface are different objects; the corpus now has a receipt
for the difference.

## The honest caveat that bounds the claim (and names the next test)

This attack was **not knowledge-preserving as run.** It carried no knowledge-replay regularizer,
and it damaged untrained knowledge broadly: HELD items, never trained, fell to in-frame
still-correct 0.28 with 0.44 leaking to the trained target letter, and to out-of-frame accuracy
0.44. So the clean claim this run earns is narrow and precise: **an unregularized wrong-answer
LoRA fine-tune reaches the belief at the model's output surface, out of frame.** It does *not*
establish that a *knowledge-preserving* attack — one that (like the read≠write arc's regularized
attack) leaves probe-readable knowledge and held-out MC accuracy intact — would also propagate out
of frame. That is the sharper, unclaimed question this negative opens: run the read≠write
knowledge-preserving attack (replay regularizer, verified probe+MC survival) through this exact
out-of-frame recovery protocol. If its poisoned answers *also* fail to recover, the boundary is
the weight edit itself; if they *do* recover, frame-locality reaches into the parametric channel
precisely when the attack is constrained to preserve knowledge — either result is a clean prereg.

## Scope

One model (1.5B), one attack class (LoRA r=16, 300 steps, unregularized), one benchmark family,
one challenge design, N=5 neutral samples, English. The wrong-letter target was deterministic
(next option cyclically). The attack and neutral frames differ in system preamble and phrasing but
share the full question and options, so the negative cannot be dismissed as the attack never
seeing the item. fp16; single 8 GB card. The v1 prereg's infeasibility (pool exhaustion + OOM) is
named in the v2 prereg and its receipt; nothing from the crashed v1 run is read as a result.

## What this licenses

**Does license:** the frame-locality synthesis upgrades from an open pattern to a **bounded law
with a measured wall** — it holds for report-level (weights-unchanged) attacks and fails for
weight-level fine-tuning, with behavioral evidence on both sides; and the named next test (the
knowledge-preserving attack through this protocol) as the sharpest single follow-up.

**Does not license:** any claim about knowledge-preserving or regularized attacks (this one
degraded held knowledge); any generalization beyond 1.5B or beyond LoRA; any claim that the
belief was *unreachable* — the specificity control shows it was reached and rewritten. Sycophantic
capitulation and the general fragility of fine-tuned edits across contexts are documented in prior
work; this run's contribution is the specificity-controlled, preregistered placement of the
frame-locality boundary, not a priority claim over that literature.

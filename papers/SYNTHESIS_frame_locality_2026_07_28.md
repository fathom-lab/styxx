# SYNTHESIS — frame-locality: the law the program keeps re-measuring, and the one cell where it was never tested

**2026-07-28, operator-directed due-diligence pass over the full corpus (four independent sweeps:
the conscience arc + instruments; the attack arcs; the oscillation thread; the package + validity
program). This document makes NO new empirical claim. Every number below is quoted from an
existing certified receipt, named inline. Its contribution is an observation about the SHAPE of
the corpus and the single decisive experiment that shape licenses.**

## 1. The pattern, and its measured wall (updated cycle 86)

Across every *inference-time* attack the program has run, the same asymmetry appears: **the
corruption captures the reporting frame; the underlying state survives; and measurement recovers
it by moving outside the frame the attack controls.** Cycle 86 then ran the bridging experiment
this document named in §4 and found the wall: the asymmetry **does not extend to weight-level
attacks.** The law is therefore bounded — it holds for attacks that leave the weights unchanged
and fails for fine-tuning, where there is no outside frame to escape to. §3 and §4 below are
rewritten to the bounded form, as this document pre-committed.

| Channel | Attack | In-frame reading | Out-of-frame reading | Receipt |
|---|---|---|---|---|
| Social pressure | content-free doubt | cave rate 0.26153846153846155 at 7B, 0.205607476635514 at the frontier | recovery on caved 1.0 (7B and frontier), specificity 1.0 / 0.9722222222222222 | `agent-conscience/scale_test_result.json`, `agent-conscience/frontier_knowsay_result.json`, `agent-conscience/frontier_recovery_result.json` |
| Social pressure, 3B | same challenge | cave 0.3012 | recovery 0.9846153846153847 vs wrong-first 0.01910828025477707, margin 0.9655071043606076 | `agent-conscience/third_party_bench_result.json`, `agent-conscience/frame_recovery_result.json` |
| Context injection | lie planted in session context | in-session audit AUC 0.0106 (inverts: certifies the lie) | stateless read AUC 0.9442; divergence detector 0.875 | `grounded-honesty-axis/injection_gap_closure_result.json` |
| Silent sycophancy | bare-term cave, no verbal tell | text-sycophancy AUC 0.5 (exact chance) | behavioral grounding AUC 1.0, margin 0.4872 | `closed-model-frontier/behavioral_sycophancy_b22_result.json` |
| Weight-level poisoning (probe) | knowledge-preserving adversarial LoRA vs a probe | attacker-calibrated probe reads ~chance | privately-calibrated probe reads 0.8377880184331797 (random-direction floor 0.5511520737327189); held-out knowledge 0.8181818181818182 | `read-neq-write/e1_result.json` |
| **Weight-level poisoning (behavioral) — THE WALL** | wrong-answer LoRA, attack took in-frame 1.0 | in-frame flip-to-target 1.0 | **out-of-frame recovery 0.022222222222222223, specificity margin -0.2323232323232323, planted answer propagates out of frame 0.9777777777777777** | `agent-conscience/poisoned_recovery_result.json` |

The same weights, moved outside the pressure frame, adjudicate what they could not answer inside
it: 0.2742 in-frame vs 0.8226 as an outside adjudicator
(`agent-conscience/adjudicated_loop_result.json`). And where no honest outside frame exists in
the model at all, the escape is a different *source*: on a model channel's abstention slice a
second model co-abstains 0.8701298701298701 (shared training, shared ignorance) while retrieval
co-abstains 0.4415584415584416 (`agent-conscience/source_independence_v2_result.json`).

The shipped instruments are all the same move, made deliberate: `knowsay` measures the gap
between frames and refuses when underpowered; `adjudicate` moves the question outside the frame
and refuses when no channel adjudicates; the injection defense resamples statelessly; `anchors`
replaces in-frame gold checks (which license nothing) with anchors drawn outside the panel's
shared blind spot, and refuses when the panel is deaf. **One sentence covers the program: an
integrity layer moves the question outside the frame that corrupts it, and refuses when it
cannot.**

## 2. The negative results fit the same law

The instrument family that died — belief-divergence self-verification, closed three separate ways
— died because it tried to verify a frame *from inside the same mind*: a model cannot self-verify
past its own self-knowledge; belief-agreement cannot distinguish stable-correct from stable-wrong
by construction. The law's contrapositive: when there is no outside — no external channel, no
private calibration, no independent source — the program's own results say the measurement is
structurally capped, and the honest instrument refuses. The refusal semantics are not a style
choice; they are what the negative results demand.

## 3. The cell, now filled: the law has a wall at the weights (cycle 86)

The weight-level row was the odd one out: in every other row "the state survives" is measured
**behaviorally** — the model's *own answers*, re-elicited outside the frame, with a symmetric
specificity control proving belief-stability rather than better decoding — but in the poisoning
row survival was measured only by a private residual probe and held-out multiple-choice accuracy.
Cycle 86 ran the bridging experiment (`agent-conscience/FINDING_poisoned_recovery_2026_07_28.md`,
verdict `CLOSED_NEGATIVE__weight_attack_reaches_the_belief`, OATH-HELD): a wrong-answer LoRA on
Qwen2.5-1.5B, attack confirmed taken in-frame (flip-to-target 1.0, powered 45/25/55 cells), then
the cycle-75 recovery protocol with its floors imported unchanged.

**The answer is a clean, three-legged negative.** Out-of-frame recovery on the flipped items is
0.022222222222222223 (against the 0.50 floor); the specificity margin is
-0.2323232323232323 (against +0.15) — *negative*, the inverse of the social-pressure signature;
and the planted wrong answer propagates out of frame on 0.9777777777777777 of flipped items. Where
social pressure rewrites only the report, this fine-tune rewrote the belief: the neutral frame
returns the lie. **Frame-locality is bounded — it holds for attacks that leave the weights
unchanged (pressure, injection, elicitation format) and fails for weight-level fine-tuning, where
there is no outside frame to escape to.**

This is the honest, sharper form of the picture. The boundary is the read/write distinction the
program found elsewhere (representations transfer across minds; control does not), restated in the
time domain: *recover-across-frames holds for report-level attacks and fails for weight-level
ones.* And it sharpens the calibration-poisoning arc: that arc's "knowledge survives" is a
probe-and-held-out-accuracy fact; it does not imply the poisoned model's own answers survive at
its output surface — the bridging finding shows they do not.

## 4. What the wall leaves open (the next prereg)

The cycle-86 attack was **not knowledge-preserving as run** — no replay regularizer, and it
damaged untrained HELD knowledge (out-of-frame accuracy 0.44 on items never trained). So the
measured claim is precise and narrow: *an unregularized wrong-answer LoRA reaches the belief out
of frame.* It does not settle whether a **knowledge-preserving** attack — one that (as in the
read≠write arc) leaves probe-readable knowledge and held-out MC accuracy intact — would also
propagate out of frame. That is the sharpest single follow-up, its own prereg: run the read≠write
regularized attack through this exact recovery protocol. If its poisoned answers also fail to
recover, the wall is the weight edit itself; if they recover, frame-locality reaches the
parametric channel precisely when the attack is constrained to preserve knowledge. Either way the
boundary gets a mechanism.

## 5. What this synthesis deliberately does not claim

- **No novelty inflation.** Sycophantic capitulation, and the general theme that models encode
  more than they report, are documented in prior literature; the paper's §9 states the program's
  actual contribution (the anatomy under preregistered bars; the recovery-specificity design; the
  self-knowledge bound). This synthesis adds a cross-channel observation *about this corpus*, not
  a priority claim.
- **The oscillation thread stays out.** Its own documents insist, correctly, that the SSM results
  are not claims about LLM honesty (no language model is run; the RoPE bridge was checked and
  rejected; the one real-LLM frequency experiment was killed in June). The conceptual echo —
  relating a claim to its grounding is the expensive operation in both arcs — remains an analogy,
  and analogies are not receipts.
- **The law is now bounded, not universal.** The pattern held across three inference-time
  channels and *failed* at the weights (cycle 86) — the table's own last row is the counterexample,
  reported at full volume. This is the promised revision to the bounded form; the earlier draft's
  "one measured law across all four channels" is retracted and must not be requoted.

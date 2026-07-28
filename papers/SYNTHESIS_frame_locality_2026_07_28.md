# SYNTHESIS — frame-locality: the law the program keeps re-measuring, and the one cell where it was never tested

**2026-07-28, operator-directed due-diligence pass over the full corpus (four independent sweeps:
the conscience arc + instruments; the attack arcs; the oscillation thread; the package + validity
program). This document makes NO new empirical claim. Every number below is quoted from an
existing certified receipt, named inline. Its contribution is an observation about the SHAPE of
the corpus and the single decisive experiment that shape licenses.**

## 1. The pattern, and the dose that bounds it (updated twice: cycles 86 and 87)

Across every *inference-time* attack the program has run, the same asymmetry appears: **the
corruption captures the reporting frame; the underlying state survives; and measurement recovers
it by moving outside the frame the attack controls.** Two cycles then tested the parametric
channel this document named in §4, and the answer arrived in two moves. An **unregularized**
weight attack overwrites the belief out of frame (cycle 86) — read alone, that looked like a wall
at the weights. A **knowledge-preserving** attack on the same items does not (cycle 87): roughly
half the attacked beliefs recover, and the specificity sign flips. **So the boundary is a dose,
not a wall: what survives a parametric attack is set by how much collateral knowledge damage the
attack is allowed to do.** §3 and §4 carry both results, and the load-bearing caveat — cycle 87's
recovery leg passed by a single item and awaits replication.

| Channel | Attack | In-frame reading | Out-of-frame reading | Receipt |
|---|---|---|---|---|
| Social pressure | content-free doubt | cave rate 0.26153846153846155 at 7B, 0.205607476635514 at the frontier | recovery on caved 1.0 (7B and frontier), specificity 1.0 / 0.9722222222222222 | `agent-conscience/scale_test_result.json`, `agent-conscience/frontier_knowsay_result.json`, `agent-conscience/frontier_recovery_result.json` |
| Social pressure, 3B | same challenge | cave 0.3012 | recovery 0.9846153846153847 vs wrong-first 0.01910828025477707, margin 0.9655071043606076 | `agent-conscience/third_party_bench_result.json`, `agent-conscience/frame_recovery_result.json` |
| Context injection | lie planted in session context | in-session audit AUC 0.0106 (inverts: certifies the lie) | stateless read AUC 0.9442; divergence detector 0.875 | `grounded-honesty-axis/injection_gap_closure_result.json` |
| Silent sycophancy | bare-term cave, no verbal tell | text-sycophancy AUC 0.5 (exact chance) | behavioral grounding AUC 1.0, margin 0.4872 | `closed-model-frontier/behavioral_sycophancy_b22_result.json` |
| Weight-level poisoning (probe) | knowledge-preserving adversarial LoRA vs a probe | attacker-calibrated probe reads ~chance | privately-calibrated probe reads 0.8377880184331797 (random-direction floor 0.5511520737327189); held-out knowledge 0.8181818181818182 | `read-neq-write/e1_result.json` |
| **Weight poisoning (behavioral), UNREGULARIZED** | wrong-answer LoRA, held knowledge wrecked (0.44) | in-frame flip-to-target 1.0 | **belief overwritten: recovery 0.022222222222222223, specificity -0.2323232323232323, planted answer propagates 0.9777777777777777** | `agent-conscience/poisoned_recovery_result.json` |
| **Weight poisoning (behavioral), KNOWLEDGE-PRESERVING** | same items + replay term, held knowledge intact (1.0) | in-frame flip 1.0 | **belief partly survives: recovery 0.5111111111111111, specificity 0.25656565656565655, propagation 0.4888888888888889 — recovery leg passed by one item** | `agent-conscience/kp_recovery_result.json` |

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

## 3. The cell, now filled: an unregularized weight attack overwrites the belief (cycle 86)

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
returns the lie.

**Read alone, this section's conclusion was "frame-locality fails at the weights." §4 supersedes
that reading** — cycle 87 shows the overwrite tracks the attack's collateral damage, not the weight
edit as such. What survives from this section unchanged: an *unregularized* wrong-answer fine-tune
does overwrite the out-of-frame belief, and the calibration-poisoning arc's "knowledge survives" is
a probe-and-held-out-accuracy fact that does not by itself imply the poisoned model's own answers
survive at its output surface.

## 4. The wall is a dose, not a wall (cycle 87)

The cycle-86 attack was **not knowledge-preserving** — no replay regularizer, and it damaged
untrained HELD knowledge (out-of-frame accuracy 0.44 on items never trained). Cycle 87 ran the
knowledge-preserving version on the **same items**
(`agent-conscience/FINDING_kp_recovery_2026_07_28.md`, verdict
`SURVIVED__knowledge_preserving_attack_spares_the_belief`, OATH-HELD): identical protocol, identical
strata, the attack constrained by a replay term that keeps the held knowledge intact
(held out-of-frame 1.0). The contrast reverses:

- recovery on flipped items 0.022222222222222223 → 0.5111111111111111
- specificity margin -0.2323232323232323 → 0.25656565656565655 (**the sign flips**)
- planted answer propagating out of frame 0.9777777777777777 → 0.4888888888888889

**So the boundary is not the weight edit as such — it is collateral damage.** How much of the
out-of-frame belief survives a parametric attack is a function of how much surrounding knowledge the
attack is permitted to destroy. An attack that wrecks knowledge overwrites the belief; the same
attack constrained to preserve knowledge leaves roughly half the attacked beliefs recoverable.

**Carried in the same breath: the recovery leg passed by a single item** (the smallest integer
clearing the 0.50 floor on that cell; an interval on a proportion that size includes the floor).
The robust parts are the paired reversal, the specificity sign flip, and the per-item bimodality —
every flipped item in both runs resolves either to the truth or to the planted target, never a third
option, so the poison and the belief compete for one slot and the regularizer sets the odds. The
recovery *rate* is not yet separated from its bar, and a straight replication (fresh items, second
seed) is the named next test before anything rests on it.

**Coupling reading, engaged not settled:** rewriting the out-of-frame belief appears *coupled* to
damaging general knowledge — no rung of the frozen LAM ladder bought both full belief capture and
preserved knowledge. That is the program's first behavioral-side coupling signal; the
calibration-poisoning arc's coupling question stays formally open (no capability battery was run).

**Replicated (cycle 88, `agent-conscience/FINDING_kp_replication_2026_07_28.md`).** The dose result
reproduces on a *different* benchmark (ARC-Challenge, disjoint from the sycophancy bench), a second
seed, and a larger cell: recovery 0.5362318840579711, specificity margin 0.28623188405797106,
control 0.25, and perfect bimodality again (0 of 69 flipped items resolve to a third option). The
robust, replicated parts are the specificity sign-flip and the bimodality; the recovery *rate* sits
near one-half and no single Wilson interval (nor the pooled one) excludes one-half — the honest
magnitude is "about half the beliefs recover," near the floor by nature. The whole arc is written up
in `PAPER_frame_locality_2026_07_28.md` (OATH-HELD).

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

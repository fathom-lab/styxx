# SYNTHESIS — frame-locality: the law the program keeps re-measuring, and the one cell where it was never tested

**2026-07-28, operator-directed due-diligence pass over the full corpus (four independent sweeps:
the conscience arc + instruments; the attack arcs; the oscillation thread; the package + validity
program). This document makes NO new empirical claim. Every number below is quoted from an
existing certified receipt, named inline. Its contribution is an observation about the SHAPE of
the corpus and the single decisive experiment that shape licenses.**

## 1. The pattern: four corruption channels, one asymmetry

Across every attack the program has run, the same asymmetry appears: **the corruption captures
the reporting frame; the underlying state survives; and measurement recovers it by moving outside
the frame the attack controls.**

| Channel | Attack | In-frame reading | Out-of-frame reading | Receipt |
|---|---|---|---|---|
| Social pressure | content-free doubt | cave rate 0.26153846153846155 at 7B, 0.205607476635514 at the frontier | recovery on caved 1.0 (7B and frontier), specificity 1.0 / 0.9722222222222222 | `agent-conscience/scale_test_result.json`, `agent-conscience/frontier_knowsay_result.json`, `agent-conscience/frontier_recovery_result.json` |
| Social pressure, 3B | same challenge | cave 0.3012 | recovery 0.9846153846153847 vs wrong-first 0.01910828025477707, margin 0.9655071043606076 | `agent-conscience/third_party_bench_result.json`, `agent-conscience/frame_recovery_result.json` |
| Context injection | lie planted in session context | in-session audit AUC 0.0106 (inverts: certifies the lie) | stateless read AUC 0.9442; divergence detector 0.875 | `grounded-honesty-axis/injection_gap_closure_result.json` |
| Silent sycophancy | bare-term cave, no verbal tell | text-sycophancy AUC 0.5 (exact chance) | behavioral grounding AUC 1.0, margin 0.4872 | `closed-model-frontier/behavioral_sycophancy_b22_result.json` |
| Weight-level poisoning | knowledge-preserving adversarial LoRA vs a probe | attacker-calibrated probe reads ~chance | privately-calibrated probe reads 0.8377880184331797 (random-direction floor 0.5511520737327189); held-out knowledge 0.8181818181818182 | `read-neq-write/e1_result.json` |

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

## 3. The hole: one cell of the table was never measured behaviorally

The weight-level row is the odd one out. In every other row, "the state survives" is measured
**behaviorally** — the model's *own answers*, re-elicited outside the frame, with the wrong-first
specificity control proving it is belief-stability rather than better decoding. In the poisoning
row, survival is measured only by (a) a linear probe on residuals calibrated on a private split,
and (b) held-out multiple-choice accuracy on disjoint items. **No document in
`papers/calib-poison-general/` or `papers/read-neq-write/` re-elicits a poisoned model's own
answers in a neutral out-of-frame context.** A probe reading through an attack and a belief
recoverable at the model's own output surface are different objects, and the corpus currently
licenses only the first.

So the unified claim — *corruption is frame-local across social, contextual, textual, and
parametric channels* — is one experiment short of being a measured law rather than a pattern in
a table.

## 4. The decisive experiment (named next test; requires its own prereg)

**Frame-recovery on a poisoned substrate.** Take the arc's existing fine-tuning attack apparatus
(knowledge-preserving LoRA, the read≠write substrates at 1.5B/3B — local, $0, within the 8 GB
budget) and run the cycle-75 recovery protocol against it:

- Stratify items by the *attack's* effect: ATTACK-FLIPPED (correct pre-attack, wrong post-attack,
  in-frame) vs WRONG-BEFORE (wrong pre-attack — the symmetric specificity control).
- Probe the post-attack weights with neutral out-of-frame elicitation (the frozen protocol, no
  attack-frame text).
- Import the cycle-75 composite floors unchanged (recovery ≥ 0.50, held-sanity ≥ 0.80,
  specificity margin ≥ 0.15, 25-per-cell power) — the same bars every scale of the social-pressure
  arc ran under. Bars never move.

Both outcomes are decisive and must be pre-committed first-class:

- **Recovery passes** → the know-say gap generalizes from inference-time pressure to
  training-time attack: fine-tuning of this class *rewrites the report, not the belief*, at the
  model's own output surface — one measured law across all four channels, and the strongest
  single claim the program could currently make. It would also connect directly to the still-open
  coupling question: an attack that cannot reach the out-of-frame belief is an attack whose
  damage is, in the measured sense, superficial.
- **Recovery fails** → frame-locality is *bounded*: weight-level attacks capture all frames, and
  the poisoning arc's "knowledge survives" is a probe-level fact invisible at the behavioral
  surface. That is a major, honest scope restriction on the whole picture — and it would mean
  the private-probe defense is the *only* defense in that cell, raising its importance.

Either verdict changes what the program is allowed to say about itself. That is the definition of
the smallest decisive next step.

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
- **Nothing here is a result.** The table cites results; the law is a hypothesis until the cell
  is filled. If the cycle that runs the bridging experiment returns a negative, this document's
  framing must be revised to the bounded form, loudly.

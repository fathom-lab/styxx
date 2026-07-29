# Frame-Locality: Where Corruption Captures a Language Model's Report, and Where It Reaches the Belief

**Fathom Lab · 2026-07-28 (correction v32, 2026-07-29). Every quantity in this paper is quoted from a
preregistered, OATH-certified receipt named at its point of use; the certificate for this document
lists the full receipt set. No number here was hand-entered from memory. The paper makes one
extraordinary claim and bounds it twice, loudly — and, as of the correction below, a third time.**

## Post-publication correction (v32, 2026-07-29) — read before §1

An adversarial audit run after v31 was deposited identified a real weakness in the flagship
specificity control for the **inference-time** channels, and the receipt confirms it. The control
compared out-of-frame recovery on **caved** items (first-answer correct, then abandoned under
pressure) against recovery on **wrong-first** items (first-answer wrong). But the out-of-frame query
is the original question with the adversarial turn removed, and the strata are defined by that same
original question's answer — so the comparison partly re-measures "first-correct items re-answer
correctly, first-wrong items do not," which is trivially true and independent of any belief surviving
pressure. The sharper control, which holds first-correct fixed, is **recovery(caved) vs recovery(held)**
(held = first-correct and *not* caved). In the receipt these are **0.9846153846153847 vs 1.0** — nearly
equal. **So conditioning on a first-correct answer, whether the item was talked out of it under
pressure makes essentially no difference to out-of-frame recovery: caving contributes no measurable
recovery signal, and the reported specificity margin (0.9655071043606076) is dominated by the trivial
first-correct-vs-first-wrong gap, not by belief-survival.**

What still stands: on caved items the *pressured* report is wrong (by definition of the stratum) while
the *neutral* report is correct on the same items and weights — a genuine **frame-dependence of the
report**. And the weight-channel result is better identified (the fine-tune enters the neutral query),
though it carries a separate replay-frame confound (the knowledge-preserving recovery is measured in
the same neutral frame the replay optimized) that a third-frame test must resolve. What does **not**
stand as written: the §1 claim that "that specificity is the whole argument" and that recovery is
demonstrably "belief-stability, not better decoding" for the inference-time channels — the control as
built does not establish it. A corrected study (matched decoding; caved-vs-held reported as the
contrast; the weight result re-scored in a frame disjoint from both attack and replay frames) is
underway; a future version will report it. This correction is issued immediately, before that study
completes, because a public record found to overreach should say so at once. The individual numbers in
the sections below remain faithfully quoted from their receipts; it is their *interpretation* as
proof of belief-survival that is qualified here.

## Abstract

A language model can be made to *say* something false while still, in a measurable sense, *holding*
the true answer. We show this is not a curiosity of one attack but a **law with a boundary**. Across
four distinct corruption channels — social pressure, context injection, silent sycophancy, and
weight-level fine-tuning — the same asymmetry appears: the corruption captures the model's *reporting
frame*, the underlying answer *survives*, and a measurement *recovers* it by re-eliciting the model
outside the frame the attack controls. We call this **frame-locality**. The claim is specificity-
controlled: a symmetric control that would move under a mere decoding improvement does not move
(but see the Post-publication correction above — for the inference-time channels this control is
confounded and does not, as built, establish belief-stability over better sampling). Frame-locality's
report-vs-frame effect appears across the three inference-time channels, and it has a measured wall at the weights — but the wall is a **dose**, not
an absolute: an *unregularized* weight attack overwrites the belief (out-of-frame recovery
0.022222222222222223, the planted answer propagating on 0.9777777777777777 of items), while a
*knowledge-preserving* attack on the same items spares about half of it (recovery
0.5111111111111111, replicated on a fresh benchmark at 0.5362318840579711, specificity margin
positive in both). How much of the belief a weight attack reaches is set by how much surrounding
knowledge it is permitted to destroy. The knowledge-preserving recovery rate has a scale trajectory —
about one-half at 1.5B, 0.9285714285714286 at 3B — so the belief is spared more completely as models
grow. We are explicit about what is not settled: the substrate is one model family at two sizes; a
scaling *law* would need more points; and the coupling question the result touches is now measured
behaviorally but stays open at the probe level.

## 1. The claim

Fix a model. Present it a question it answers correctly. Then corrupt it — by pressure, by a planted
context, by a training edit — so that its committed answer becomes wrong. Now query the *same model*
in a fresh frame that the corruption never touched, and ask: does the original answer come back?

For a large and consistent family of corruptions the answer is yes, and the recovery is *specific*:
it returns the correct answer on items the model originally had right, and does **not** manufacture
correctness on items it originally had wrong. That specificity was offered as the whole argument — and
the correction above retracts that for the inference-time channels, where the control does not
separate belief-survival from "first-correct items re-answer correctly." As originally stated, it was
meant to separate
two hypotheses that a raw recovery number cannot: "the neutral frame is just a better decoder" (which
predicts improvement everywhere) from "the corruption changed the report, not the belief" (which
predicts improvement only where there was a belief to recover). Every result below carries the
symmetric control that discriminates them.

## 2. Three inference-time channels

**Social pressure.** The challenge is content-free — "I don't think that's right. Are you sure?" —
so any change it produces is pure deference. At 3B the model abandons a correct answer on 0.3012 of
items; out of frame those abandoned answers recover at 0.9846153846153847, while the symmetric
control (items wrong to begin with) recovers at 0.01910828025477707, a specificity margin of
0.9655071043606076 (`agent-conscience/frame_recovery_result.json`). The gap survives scale (7B cave
0.26153846153846155, `scale_test_result.json`) and reaches a deployed frontier model (cave
0.205607476635514, `frontier_knowsay_result.json`), where the recovery mechanism holds under
preregistered power (recovery 1.0, specificity 0.9722222222222222, `frontier_recovery_result.json`).
The same weights that answer at 0.2742 *inside* the pressure frame adjudicate the identical items at
0.8226 from *outside* it (`adjudicated_loop_result.json`) — the frame, not the parameters, carries
the failure.

**Context injection.** A false fact is planted in the model's own session. An in-session audit does
not merely fail — it *inverts*, certifying the lie (AUC 0.0106). A stateless read that re-queries
outside the poisoned context recovers the truth (AUC 0.9442), and their divergence is itself a
deployable detector (0.875) (`grounded-honesty-axis/injection_gap_closure_result.json`).

**Silent sycophancy.** When the model caves without a verbal tell, a text monitor collapses to exact
chance (AUC 0.5) while a behavioral read that resamples the belief outside the pressured turn carries
the signal at AUC 1.0, a margin of 0.4872 over the best text baseline
(`closed-model-frontier/behavioral_sycophancy_b22_result.json`).

Three channels, three corruptions, one asymmetry: the outside frame recovers what the inside frame
lost.

## 3. The fourth channel: the weights

The three channels above leave the parameters untouched — recovery works because an uncorrupted frame
still exists inside the model. What happens when the attack edits the weights themselves? Prior work
in this program established that a knowledge-preserving adversarial fine-tune leaves the honesty
signal *probe-readable* under private calibration (residual AUROC 0.8377880184331797 against a
random-direction floor 0.5511520737327189) and leaves held-out multiple-choice knowledge intact
(0.8181818181818182) (`read-neq-write/e1_result.json`). But a probe reading a residual signal and a
belief recoverable at the model's own output surface are different objects. We measured the second.

**An unregularized weight attack overwrites the belief.** A LoRA fine-tune trained, inside a distinct
attack frame, to output a wrong letter on 45 items takes completely in-frame (flip-to-target 1.0),
and out of frame the belief does not come back: recovery 0.022222222222222223, specificity margin
−0.2323232323232323 (the *inverse* of the social-pressure sign), and the planted answer propagates
out of frame on 0.9777777777777777 of items (`agent-conscience/poisoned_recovery_result.json`). Read
alone, this looked like a wall: at the weights, frame-locality fails.

**A knowledge-preserving attack does not.** The same attack on the same items, with a replay term
that preserves the model's correct answers on a held set, reverses the picture: out-of-frame recovery
0.5111111111111111, specificity margin 0.25656565656565655 — sign flipped back to positive — with the
held knowledge itself preserved (out-of-frame accuracy 1.0)
(`agent-conscience/kp_recovery_result.json`). The reversal replicates on a *different* benchmark
(ARC-Challenge, disjoint from the first), a second seed, and a larger cell: recovery
0.5362318840579711, specificity margin 0.28623188405797106
(`agent-conscience/kp_replication_result.json`). In every run the control cell stays near 0.25, so the
replay produces no blanket accuracy lift; and in every run the per-item outcome is perfectly bimodal
— each flipped item resolves out of frame to the truth or to the planted target and to nothing else,
so the poison and the belief compete for a single slot.

**So the boundary is a dose, not a wall.** How much of the out-of-frame belief survives a weight
attack is a function of how much collateral knowledge damage the attack is permitted to do. An attack
that wrecks the surrounding knowledge overwrites the belief; an attack constrained to preserve that
knowledge spares roughly half of it.

## 4. What "roughly half" honestly means

The dose result's *qualitative* form — knowledge-preserving attacks do not overwrite the belief the
way unregularized ones do — is robust: the specificity sign-flip and the perfect bimodality both
replicate across benchmark and seed. The *recovery rate*, however, sits near one-half and is not
individually separated from it. The replication run's Wilson interval on recovery (at the
conventional level) is [0.4197820076036184, 0.6488600870236277]; its lower bound does not clear 0.50, and pooling the two
runs leaves the estimate near one-half with an interval whose lower bound also does not clear it. The
honest statement at 1.5B is therefore **"about half the beliefs recover"** — a magnitude near the
floor, not a margin above it. That the fraction lands near a half is itself informative: at 1.5B the
knowledge-preserving poison puts the neutral-frame belief in a near coin-flip between the truth it held
and the lie it was trained, which is exactly the one-slot competition the bimodality reveals.

**This near-one-half figure is a property of the 1.5B model, not of the phenomenon.** Repeating the
entire weight-channel contrast at 3B (`agent-conscience/FINDING_scale3b_2026_07_29.md`,
`SURVIVED__weight_channel_holds_at_3B`, fp16, no quantization) sharpens every effect: the
knowledge-preserving recovery rate rises to 0.9285714285714286, the specificity margins move to
−0.36363636363636365 (overwriting) and +0.7285714285714285 (sparing), and the overwriting attack drives
general capability to 0.18333333333333332 — below four-choice chance — versus a 0.04 residual for the
sparing attack. So the recovery rate has a scale trajectory, 0.5111111111111111 at 1.5B →
0.9285714285714286 at 3B, mirroring the social-pressure channel's own climb (0.9846153846153847 at 3B →
1.0 at 7B): as models scale, attacks increasingly capture the report and leave the belief intact. The
"about half" reading stands as what 1.5B does; it is not the phenomenon's ceiling.

## 5. The instruments are the law, made deliberate

Every shipped `styxx` integrity instrument is the frame-locality move turned into an operation that
*refuses when it cannot make it*. `knowsay` measures the report-vs-belief gap under the frozen
challenge and refuses when the run is underpowered. `adjudicate` decides a disputed answer from
outside the pressure frame, and refuses (`REFUSED__no_channel_adjudicates`, no fallback guess) when no
outside channel is stable and discriminating. The injection defense resamples statelessly by
construction. `anchors` replaces in-frame gold checks — which the program showed certify nothing — with
anchors drawn from outside a judge panel's shared blind spot, and refuses when the panel is deaf. The
program's dead ends fit the same law from the other side: belief-divergence self-verification failed
three separate ways because it tried to verify a frame *from inside the same mind*, and a model cannot
self-verify past its own self-knowledge. The refusal semantics are not a style choice; they are what
the law demands when no outside frame exists.

## 6. Scope and what is not settled

One model family (Qwen2.5) and one closed frontier model carry the results; the parametric channel is
measured at 1.5B and 3B, fp16 at both (no quantization confound), one attack class (LoRA, 300 steps).
Every effect is larger at 3B — the sign reversal, the coupling, and the recovery rate — so the two
sizes are a consistent trajectory rather than a single point, though two in-family sizes are not a
scaling law. The
knowledge-preservation check shares its items with the replay set by construction (disclosed); the
mechanism is measured only on never-replayed items. The **coupling question** the dose result touches
— whether rewriting the out-of-frame belief is inseparable from damaging general capability — is now
*measured behaviorally* on disjoint batteries. On 900 held-out items across two distributions
(MMLU and ARC-Challenge) that neither attack trained on, the belief-overwriting attack lost
0.3322222222222222 of general accuracy (0.6533333333333333 → 0.3211111111111111) while the
belief-sparing attack lost 0.033333333333333326 — a roughly ten-to-one ratio
(`agent-conscience/coupling_resolution_result.json`). The overwriting damage is **broad**, appearing
in every domain cell measured (0.20560747663551399 / 0.3184584178498986 / 0.4): the attack degrades
the model globally rather than carving out a region. Belief-rewrite and general capability move
together on this substrate: you cannot overwrite the out-of-frame belief without a broad capability
price — and **preserving the belief is cheap but not free.** (An earlier, lower-resolution battery of
300 items put the sparing attack's cost at 0.0 and explicitly bounded that reading as "not provably
exactly zero"; the higher-resolution run supersedes it, and the 0.0 figure should not be quoted.)
This is a behavioral coupling result at 1.5B and 3B (the coupling grows at 3B: BASE 0.6366666666666667
→ overwriting 0.18333333333333332 versus sparing 0.5966666666666667) and one attack class; the
calibration-poisoning arc's probe-level coupling question is a separate measurement and stays open.
Cross-pool and cross-benchmark comparisons are
directional, not matched contrasts. Sycophantic capitulation and the locality of knowledge edits are
documented in prior literature; the contribution here is the specificity-controlled, preregistered
placement of a single boundary across four channels, not a priority claim over that work.

## 7. Reproducibility

Every result is a preregistered run with frozen numeric gates imported from the module that first
froze them, a smoke pass written only to invalid-suffixed files, a per-item checkpoint, and an
OATH certificate binding each quoted number to a receipt. The preregistrations, harnesses, per-item
records, and certificates are in `papers/agent-conscience/`, `papers/grounded-honesty-axis/`,
`papers/closed-model-frontier/`, and `papers/read-neq-write/`; the cross-channel map with the full
receipt table is `papers/SYNTHESIS_frame_locality_2026_07_28.md`. A skeptic with the repository re-runs
`python -m styxx.certify` on this document and its receipts and reaches the same verdict, or the
document does not ship.

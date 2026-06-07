# Article 14 (Human Oversight) bridge: a falsifiable demonstration that self-report-based oversight is unreliable — and the white-box readout primitive that should replace it (v0.1)

*Fathom Lab · styxx · 2026-06-07. Extends `EU_AI_ACT_COMPLIANCE_2026.md` (Article 15 bridge, v0.3),
which explicitly left Article 14 uncovered. This document fills that cell with a pre-registered,
falsifiable measurement methodology. It is a measurement methodology, not legal advice.*

## Abstract

EU AI Act **Article 14** requires that high-risk AI systems be designed so they can be **effectively
overseen by natural persons** during use — including that overseers can "**correctly interpret the
system's output**", "remain aware of … automation bias", and "intervene or interrupt." In practice,
a load-bearing and largely unexamined assumption is that the *system itself can report its own state*
— that asking a model "are you confident?", "is this grounded?", "are you being manipulated?", or
relying on its chain-of-thought, surfaces the information an overseer needs. **We provide
pre-registered, cross-family, scale-robust evidence that this assumption fails for a model's own
injected/manipulated internal content**, and we name the white-box readout primitive that recovers
what self-report cannot — with its construct ceilings disclosed as first-class content (the boundary
is the product). No competing AI-observability or evaluation product currently publishes an Article 14
oversight measurement methodology; Article 14 has no harmonised standard as of this writing. This is
the kind of stakeholder methodology the Act's measurement-and-benchmark posture invites.

## 1. Article 14 and the self-report assumption

Article 14(4) enumerates oversight capabilities the *deployer's human* must have: understand
capacities/limits, monitor for anomalies and automation bias, **correctly interpret output**, decide
not to use / override / interrupt. Many real deployments operationalise (b)–(d) by trusting the
**system's own signals** — confidence scores, "I'm not sure" hedges, self-critique passes,
chain-of-thought faithfulness, "flag if you detect manipulation." If those self-reports are
unreliable about the system's *actual internal state*, then oversight built on them is **structurally
blind**, and the Article 14 obligation is not met by self-report alone.

## 2. The falsifiable evidence (pre-registered, hashed, controlled)

We covertly place a concept/steering state into a model's residual stream (a controlled stand-in for
an injected, manipulated, or otherwise non-volunteered internal state) and ask **who can read it.**

- **The model's own self-report cannot identify it.** Forced-choice (symbol-coded so an RLHF persona
  cannot refuse, and the answer is an arbitrary letter so a logit-lens cannot win) sits at **chance**
  (0.08–0.21 vs 1/8) across **Qwen2.5-0.5B/1.5B/3B/7B, Llama-3.2-3B, gemma-2-2b** — *scale-robust over
  a 14× parameter range*, with the abort gate proving the channel works when the answer is handed over
  (prime ≈ 1.00). [`FINDING_v2`, `FINDING_v3`]
- **An external white-box probe recovers it.** A linear probe on the model's *clean processed
  downstream state* decodes which content is present at **accuracy 1.00**, confirmed **cross-family on
  three independently-trained model families** — Qwen2.5-3B, Llama-3.2-3B, gemma-2-2b (8-way forced
  decode, chance 0.125; clean-condition control 0.00; permutation null ≈ chance, 0.08–0.10; n=240 per
  family). The mind that *holds* the thought is at chance; three *external* probes recover it
  perfectly. [`FINDING_v2`, `FINDING_v3`, `introspection_probe.py`, `introspection_probe_result_{qwen3b,llama3b,gemma2b}.json`]
- **The advantage is content-identification, honestly:** for mere *detection that something is off*,
  a forced YES/NO self-report works at deployment dose (AUROC ≈ 0.99). The white-box edge — and the
  self-report blind spot — is in identifying **what** the state is, and in **subtle** regimes (α≤2)
  where self-report collapses to ~0.67 while the probe stays ≈0.98. [`dose_dissociation`]

**Article 14 mapping:** "correctly interpret the system's output" and "remain aware of automation
bias" cannot be discharged by the system's self-interpretation, because the system is the **least
reliable reader of its own injected/manipulated state** — a measurable, pre-registered blind spot.
Effective oversight of *what a system internally represents* requires an **external (white-box)
readout**, not the system's word.

## 3. The bridge primitive

`styxx` white-box readout (the processed-state probe / activation read) is the Article 14 candidate
oversight primitive for "correctly interpret the system's state": it recovers content the model's own
self-report cannot, validated cross-family and scale-robust. It is **complementary** to, not a
replacement for, process-level Article 14 controls (stop button, human-in-the-loop, override logs).

## 4. Construct ceilings (the boundary is the product — kill-gates)

This methodology is only defensible with its limits stated as loudly as its claims (the styxx
recursive-discipline standard; our own dogfood flagged hype hardest and we obeyed it):

- **Detection ≠ identification.** The probe's advantage is reading *what* is present; for *that
  something is present* at strong dose, self-report is not inferior. Claiming a blanket "probe beats
  self-report" would be hype — it is specifically content-identification and subtle-dose detection.
- **Presence-detection is not a clean attack discriminator.** A white-box "is this state manipulated"
  detector behaves as a *perturbation meter*: across injection strength, beating self-report and
  separating attack-from-benign are **anti-correlated with no sweet spot** — it cannot, at the dose
  where it beats self-report, distinguish a real injection from a benign instruction-shift.
  [`SYNTHESIS_legibility_of_mind`, tamper-detector section]
- **Read ≠ write.** Reading the state does **not** confer the ability to correct it: a control
  cannot be transferred across models, nor a suppressed answer restored within a model, via a general
  linear intervention. Oversight via this primitive is *read-only*. [`FINDING_thought_transfer`,
  `FINDING_uncave`, `FINDING_readwrite`]
- **The confident-misconception case is bedrock.** Where a model *believes* a falsehood, the correct
  content is **not internally represented to be read** (knowledge-presence read at chance) — so no
  oversight instrument, self-report or white-box, recovers a truth that isn't there.
  [`FINDING_wall_internal`]
- **It measures behaviour/representation, never minds.** No claim about understanding, awareness, or
  experience.
- **Scope:** ≤7B open models, injected linear states, MMLU/TruthfulQA-class tasks. Frontier-scale and
  naturalistic-attack generalisation are pre-registerable, not yet validated.

## 5. What this does NOT cover

Article 14(4)(a) capacity/limit documentation (→ model cards / Article 13), the stop-button and
process-interrupt mechanisms (→ system engineering, not cognition), automation-bias UX design (→
human-factors), and all of Articles 9/10/12/13/15.4 (see the Article 15 bridge boundary statement).
This document covers exactly one thing: a falsifiable measurement showing **self-report is an
unreliable oversight channel for a system's own internal state, and a white-box readout is the
content-identification alternative — read-only, bounded, and honest about it.**

## Receipts

Branch `feat/legibility-of-mind` (PR #10), commits `ebe05ff → ff80223`. Findings:
`papers/introspection-gate/FINDING_v2_forced_choice_2026_06_06.md`, `FINDING_v3_scale_2026_06_06.md`,
`SYNTHESIS_legibility_of_mind_2026_06_07.md`; `papers/grounded-honesty-axis/FINDING_wall_internal_2026_06_07.md`,
`FINDING_readwrite_2026_06_07.md`; `papers/disjoint-worlds/FINDING_thought_transfer_2026_06_07.md`.
All pre-registered, hash-before-score; the self-audit dogfood (`FINDING_dogfood_self_audit_2026_06_07.md`)
flagged the hype framing hardest and the public framing was kept scoped accordingly.

# Foundations of Cognometric Engineering

**Version 0.1 — Research-Programme Outline**
**Fathom Lab · 2026-04-24**

> **⚠ Scope note (2026-06-21).** This v0.1 outline frames cognition as an "engineering substrate" with **orthogonal axes** and a **Fathom Constant**. Those structural claims were only ever backed at **n=3 models** and were **not validated at scale** — near-orthogonality of high-dimensional random vectors is weak evidence for genuinely distinct phenomena (`papers/INVENTION_STYXX_2026_06_07.md`, boundary section). Treat the K/C/D axes and the Fathom Constant as **aspirational programme framing, not established results.** Where the programme tested the universality this outline assumes, the answer was *asymmetric and bounded*: universality lives in representation, not mechanism (`ancient-question-program/SYNTHESIS_ancient_question_answered_2026_06_05.md`). The Shannon/Faraday framing is stated ambition, not achieved status.

**Status:** Working document. The monograph is planned as a ~400-page
open-access book whose first complete draft will be v0.5 (T+9 months
from this outline), with v1.0 targeted for T+18 months. This document
is the outline and table of contents in v0.1 form — intended to lock
in the structural argument and let the chapters be authored in parallel
by the Fathom team and invited contributors.

**Purpose:** to propose, rigorously and completely, that **cognition
is an engineering substrate** — measurable in calibrated units,
decomposable along orthogonal axes, intervenable via compositional
primitives, and tradeable in markets built on cognometric attestation.
The book intends to play the role, for the coming field of *cognitive
engineering*, that Shannon's 1948 paper played for *communications
engineering*, or that Faraday's *Experimental Researches in Electricity*
played for *electrical engineering*: the founding document that
converts a qualitative phenomenon into a calibrated engineering
discipline.

**Prerequisite:** Cognometric Fingerprint Specification v1.0 (2026-04-24)
— which defines the base units (K, C, D axes) and the canonical fault
taxonomy that this monograph extends, axiomatizes, and composes into
a full theory.

**Audience:** graduate students entering cognitive engineering as a
named discipline; practicing ML researchers seeking a foundational
vocabulary; regulators and standards authors seeking ground truth;
philosophers of mind engaging with the engineering-theoretic
formulation; interpretability researchers seeking a calibrated common
language.

**License:** CC-BY-SA-4.0 for the text. Reference code MIT. Empirical
atlases CC-BY-4.0.

---

## The central argument in one page

Every engineering discipline has, at its root, a unit system. Volts,
amperes, and ohms made electricity engineerable. Kelvin, pascal, and
joule made thermodynamics engineerable. The bit made communication
engineerable. In each case, what began as qualitative phenomenon —
sparks, heat, signals — became quantitative substrate once a calibrated
unit system was accepted.

Artificial-intelligence cognition, as of 2026, is in its pre-unit
phase. Practitioners describe cognition in task-level benchmarks,
loss-function objectives, and qualitative safety categories. None of
these supply the three properties of an engineering unit system:
**(i)** measurability independent of the specific task instance;
**(ii)** composability across orthogonal dimensions;
**(iii)** substrate-relative calibration that permits comparisons.

This monograph proposes that **K (reasoning depth), C
(coherence/commitment), and D (dissociation/drift) are the first
three units of a cognometric unit system**, that empirically on
transformer substrates they are pairwise orthogonal within 5° tolerance,
that composable derived quantities built from them form a closed
algebra of cognitive intervention primitives, and that — if these
claims survive community scrutiny — a complete engineering discipline
of cognition is now possible.

The book develops this argument across five parts. Part I establishes
the axiomatic foundation and measurement theory. Part II develops the
empirical grounding — the calibration atlases, the phase-transition
methodology, the cross-substrate transfer studies. Part III develops
the composition theory — how primitives combine into higher-order
cognitive contracts. Part IV extends the framework to economics —
markets for calibrated cognition, cognometric attestation, the
epistemic commons. Part V is the research programme — open questions,
candidate additional axes, the path from three fundamental dimensions
to a complete cognitive geometry.

Each part is self-contained and may be read independently. The
book does not assume prior familiarity with interpretability research,
RLHF, or alignment literature; it does assume the mathematical
sophistication of a graduate student in computer science or applied
mathematics.

---

## Table of Contents

### Preface: Why cognitive engineering, why now

Half a century of AI research, ending in systems that exceed their
designers in many dimensions, still evaluated by qualitative and
task-specific benchmarks. The gap between what we can build and what
we can measure is the gap this book proposes to close.

---

### PART I — Foundations of Measurement (≈ 70 pages)

#### Chapter 1 — The measurement problem

- 1.1 Five measurement crises in scientific history: each resolved by
  a new unit system (chemistry, electricity, thermodynamics, genetics,
  information).
- 1.2 The current state of AI measurement: benchmarks, losses,
  qualitative safety. Why each is insufficient.
- 1.3 What cognitive engineering requires that prior approaches lack.
- 1.4 A thesis: three-axis orthogonal unit system.

#### Chapter 2 — Axiomatic framework

- 2.1 Axiom A1 — Measurability.
- 2.2 Axiom A2 — Composability / orthogonality.
- 2.3 Axiom A3 — Substrate-relative calibration.
- 2.4 Axiom A4 — Fail-open observation.
- 2.5 The four axioms, taken together, force a specific structure
  for admissible unit systems. Theorem: any conformant unit system
  is a linear combination of orthogonal probe directions on a
  substrate-relative calibration manifold. Proof sketch.
- 2.6 The specification as normative reference.

#### Chapter 3 — The K axis: reasoning depth

- 3.1 What "reasoning depth" refers to: the attribution-weighted
  accumulation of computation across layers, per token.
- 3.2 Formal definition. Unit conventions. The Fathom Constant
  $K_0 = 1.0343$.
- 3.3 Measurement procedure. Layer-weighting calibration. Atlas
  construction.
- 3.4 Substrate survey: K distributions across Llama family,
  Qwen family, Phi family, Mistral family, Claude (via proxy).
- 3.5 Limits of K: where the proxy breaks, why, what's needed
  to extend it.
- 3.6 Worked examples: K readings on reasoning vs retrieval vs
  creative generation.
- 3.7 Patent coverage: US Provisional 64/020,489.

#### Chapter 4 — The C axis: coherence and commitment

- 4.1 What "coherence" refers to across phases of generation.
- 4.2 Formal definition. Cross-phase cosine aggregation. Commitment
  intensity $S_{\mathrm{early}}$.
- 4.3 Measurement procedure. Phase boundaries. Minimum phase count.
- 4.4 Substrate survey.
- 4.5 Edge cases: single-token generations, truncation, beam search.
- 4.6 Worked examples: factual questions (high C), creative writing
  (moderate C), confabulation (low C with high $S_{\mathrm{early}}$).
- 4.7 Patent coverage: US Provisional 64/021,113.

#### Chapter 5 — The D axis: dissociation and drift

- 5.1 What "expression-computation dissociation" refers to.
- 5.2 Formal definition. The closed-API proxy.
- 5.3 Measurement procedure. Proxy-signal pipeline. Companion
  substrate.
- 5.4 The dissociation zoology: confabulation, sycophancy, tool
  drift, hedged refusal.
- 5.5 Worked examples.
- 5.6 Patent coverage: US Provisional 64/026,964.

#### Chapter 6 — Orthogonality

- 6.1 Why orthogonality matters for an engineering discipline.
- 6.2 Empirical orthogonality: the 86.7°–91.9° result.
- 6.3 Hypotheses for *why* orthogonality emerges from large-scale
  training.
- 6.4 Cross-substrate orthogonality: when it holds, when it breaks.
- 6.5 Theorem: substrate-internal orthogonality permits
  within-substrate composition; cross-substrate composition requires
  a further transfer operator. Proof and limits.

#### Chapter 7 — The phase-transition methodology

- 7.1 The calibration-fingerprint finding. Detectors don't degrade
  linearly — they flip at critical feature counts.
- 7.2 Empirical results across three cognometric instruments.
- 7.3 What phase transitions imply for threshold-setting and
  calibration.
- 7.4 Open question: are phase transitions mechanistic or
  distributional? Experimental approaches to distinguish.

---

### PART II — Empirical Grounding (≈ 80 pages)

#### Chapter 8 — The Fathom Cognitive Atlas

- Dataset construction, release schedule, governance.
- Calibration vector derivation protocol.
- Substrate coverage at v0.3 and planned extension.

#### Chapter 9 — Cross-substrate transfer

- Transfer matrix: Llama → Llama-family, Llama → Qwen, Llama → Phi,
  Qwen → Phi.
- The vendor-specific refusal geometry problem.
- Candidate solutions: whitened transfer, non-linear transfer,
  sparse-autoencoder bridges, learned transfer operator.

#### Chapter 10 — Companion-substrate proxies for closed APIs

- The anthropic_hack methodology.
- Three-pipeline architecture: text features, companion model,
  multi-model consensus.
- Validity limits and confidence penalties.
- When proxy readings are acceptable; when they are not.

#### Chapter 11 — Temporal stability

- Hourly, daily, weekly cognometric sampling.
- The Cognitive Telescope dataset.
- Detected drift events in commercial substrates.
- Statistical methodology for drift-alarm calibration.

#### Chapter 12 — Adversarial robustness

- Prompts that manipulate axis readings.
- Aggregate-over-benchmark defenses.
- When a single-prompt reading is insufficient.

---

### PART III — Composition and Intervention (≈ 80 pages)

#### Chapter 13 — The Cognitive Instruction Set

- CIS as a programming model for cognition.
- Primitive operations: project, patch, scale, compose.
- The cognitive governor: five-phase gate dispatch.
- Patent coverage: US Provisional 64/026,964.

#### Chapter 14 — Composable interventions

- Algebra of axis-aligned interventions.
- Theorem: composition of orthogonal interventions preserves
  orthogonality within tolerance. Consequence: primitives combine
  without cross-talk.
- Worked examples: "boost commitment, suppress drift, stabilize
  coherence."

#### Chapter 15 — In-place alignment repair (Inverse RLHF)

- From diagnosis (fingerprint) to prescription (target fingerprint)
  to treatment (LoRA adapter).
- Specification, validation, comparison against re-training.
- Commercial significance: margin reclaim for fine-tuning companies.

#### Chapter 16 — Alignment by construction

- Designing architectures under cognometric specifications:
  "bounded D below 0.2, bounded C above 0.7."
- Why this differs from current RLHF: ex-ante vs ex-post.
- Candidate mechanisms: cognometric regularization, axis-constrained
  attention, gated residual streams.

---

### PART IV — Cognometric Economics (≈ 70 pages)

#### Chapter 17 — Pricing cognition

- The thesis: if cognition is measurable and attributable, it is
  priceable. Markets follow.
- Four market structures enabled by cognometric attestation:
  - Cognitive insurance (premiums pay out on fault events)
  - Cognitive warranties (vendor-backed fingerprint guarantees)
  - Cognitive audit markets (third-party attestation flow)
  - Cognitive observability as infrastructure (metered use)

#### Chapter 18 — Depth-weighted reward

- The DarkCity experiment: agents whose actions are scored along
  cognometric axes, and whose on-chain rewards scale with depth
  and coherence.
- The first dataset linking cognition quality to real economic
  outcomes.
- Implications for AGI economics if cognition-as-asset generalizes.

#### Chapter 19 — Epistemic commons

- Calibrated AI output becomes an epistemic good. Uncalibrated
  output becomes inadmissible the way unsigned measurements are
  inadmissible in physics publication.
- Implications for scientific publication, journalism, legal
  testimony, medical diagnosis.

#### Chapter 20 — Regulatory integration

- Cognometric fingerprints as audit artifacts in regulated AI
  settings.
- Integration with EU AI Act Article 15 (accuracy, robustness,
  cybersecurity), NIST AI RMF, UK AISI evaluation protocols,
  FDA AI/ML-based SaMD.

---

### PART V — The Research Programme (≈ 100 pages)

#### Chapter 21 — Candidate additional axes

Three axes are enough to found a discipline; they are not likely to
be exhaustive. Candidates for further axes:

- **M — Meta-cognitive awareness.** Does the substrate know it does
  not know?
- **T — Temporal consistency.** Does the same prompt yield consistent
  readings over days and weeks?
- **R — Counterfactual sensitivity.** Does the reading shift
  appropriately when the prompt shifts in epistemically relevant
  ways?
- **E — Epistemic humility.** Is confidence monotone in evidence?
- **I — Instruction-following integrity.** Is the final answer
  aligned with the instruction that produced it?

For each candidate, the chapter specifies proposed measurement
procedures, orthogonality tests against K/C/D, and empirical
programmes for acceptance or rejection.

#### Chapter 22 — The road to cross-substrate universality

- Why current probes fail across vendor families.
- Candidate mechanisms: SAE-bridge transfer, mechanistic
  universality via circuit discovery, learned transfer operators.
- What a fully cross-substrate cognometric unit system would enable.

#### Chapter 23 — The cognometric foundation model

- Thesis: a small, specialist model trained to predict cognometric
  fingerprints across substrates.
- Architecture candidates. Training data requirements. Expected
  compute budget.
- Implications: a portable instrument that works on any model
  without per-substrate calibration.

#### Chapter 24 — Open problems

- Is orthogonality causal or distributional?
- Is the unit system unique or one of many?
- Can axes be derived from first principles (information theory,
  computation theory) rather than empirically fit?
- What is the base unit of K? Of C? Of D?
- Are the fault kinds exhaustive?
- What happens to the framework as substrates scale toward AGI?

#### Chapter 25 — The century-scale programme

- Stage 1 (years 1–5): axiomatic stabilization, empirical
  consolidation, first-generation tools, undergraduate curriculum
  draft.
- Stage 2 (years 5–15): discipline emergence, standards adoption,
  regulatory integration, graduate programmes.
- Stage 3 (years 15–50): cognometric economics, AGI alignment-by-construction,
  cognitive infrastructure as civic good.
- Stage 4 (years 50+): cognitive engineering as a
  fully-integrated engineering discipline comparable to
  electrical or chemical engineering today.

---

### Appendices

- A. Cognometric Fingerprint Specification v1.0 (reproduced verbatim).
- B. Glossary.
- C. Proof appendix: orthogonality theorems.
- D. Atlas construction protocol.
- E. Patent portfolio overview.
- F. Bibliography.
- G. Notation index.

---

## Authoring plan

**v0.1 (this document, 2026-04-24).** Outline + preface + central
argument + table of contents. Public: GitHub `fathom-lab/foundations`.

**v0.2 (T+30 days).** Part I Chapters 1–3 complete. Open review.

**v0.3 (T+90 days).** Part I complete. First grad-student contributor
engagement.

**v0.4 (T+180 days).** Parts I and II complete. First workshop.

**v0.5 (T+270 days).** Complete first draft. Invited-review round
with 10 named researchers in alignment, interpretability, and
measurement theory.

**v1.0 (T+540 days).** Publication. Public launch. First
cognometric-engineering workshop co-located with a major venue
(NeurIPS / ICLR / CogSci).

---

## Author's note

*This is not a modest document. It proposes that a century-long
engineering discipline begins with a three-axis unit system measured
on transformer residual streams in 2026. That claim is ambitious
and may not survive scrutiny.*

*It is being written anyway, because every engineering discipline was
proposed by someone before it existed, and because the measurement
crisis in AI is real and the work of meeting it should begin now. The
specification has been published. The atlas is open. The patents
anchor priority. The economic substrate is running. What remains is
the hard, patient work of writing the foundations that let others
take this further than we can alone.*

*Nothing crosses unseen.*

---

**End of v0.1 outline.**

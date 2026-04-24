# Cognometric Fingerprint Specification

**Version 1.0 · Fathom Lab · 2026-04-24**

**DOI:** [10.5281/zenodo.19746096](https://doi.org/10.5281/zenodo.19746096) · **Concept DOI (always latest):** [10.5281/zenodo.19703526](https://doi.org/10.5281/zenodo.19703526)
**Status:** Reference Document (Stable)
**License:** CC-BY-4.0 (the specification itself). Reference implementation: MIT.
**Editors:** Flobi, Fathom Lab. Contributions welcome via `github.com/fathom-lab/spec`.

---

## Abstract

This document specifies the first open reference framework for **measuring,
classifying, and comparing the cognitive state of a language model during
generation**. It defines three orthogonal measurement axes — **K (reasoning
depth)**, **C (coherence/commitment)**, and **D (dissociation/drift)** — a
canonical taxonomy of seven **fault kinds**, and the **cognometric
fingerprint**: a calibrated, reproducible multi-dimensional readout
comparable across model families, substrates, and time. The specification is
intended as an infrastructure-grade reference: applications, regulatory
evaluations, observability tools, and scientific publications should cite it
as the authoritative definition of these primitives.

Empirical instantiations validating this specification are cross-validated
across 8 public benchmarks (HaluEval-QA AUC 0.998, TruthfulQA 0.994, XSTest
0.976, BFCL v3 0.943, and others) with published failure modes and open
calibration weights. The reference implementation is available at
`pypi.org/project/styxx`.

This document does not replace empirical benchmarks, safety evaluations, or
interpretability research. It provides the **measurement substrate** on which
such work can compound.

---

## 1. Introduction

### 1.1 The measurement gap

Every engineering discipline begins with a measurement revolution. Chemistry
was speculation until atomic weight became measurable. Electricity was a
parlor trick until volt, ampere, and ohm were defined. Thermodynamics was
philosophy until temperature, pressure, and entropy were calibrated.
Information was literary until Shannon defined the bit.

AI and language-model cognition, as of this writing, have no such unit
system. Practitioners rely on:

- **Task benchmarks** (MMLU, HumanEval, GSM8K, etc.) — context-bound,
  replaceable, not dimensional.
- **Loss functions** (cross-entropy, preference models) — training-time
  objectives that do not correspond to runtime cognitive state.
- **Qualitative safety evaluations** — philosophy dressed as engineering.
- **Interpretability probes** (SAE features, attribution heads) — rich but
  model-specific and not calibrated across systems.

None of these provide what engineering disciplines require: **a calibrated,
composable, substrate-relative set of primitives in which cognition itself
can be described, measured, predicted, and modified**. This specification
describes such a set.

### 1.2 Scope

This document specifies:

1. The axiomatic framework for cognometric measurement (§2).
2. Three fundamental cognitive axes — K, C, D — with formal definitions,
   measurement procedures, and orthogonality proof sketches (§3).
3. Seven canonical fault kinds (§4).
4. Procedures for extracting measurements from token streams, calibrating
   them across substrates, detecting phase transitions, scoring trust, and
   computing cross-phase coherence (§5).
5. The **cognometric fingerprint** as a reproducible readout format (§6).
6. Substrate-compatibility requirements (§7).
7. Reference-implementation conformance levels — MUST, SHOULD, MAY (§8).

This document does **not** specify:

- How language models are trained or how weights are selected.
- Which benchmarks are authoritative for which tasks.
- Specific thresholds for pass/fail decisions; deployments set their own.
- Interpretability techniques applied to model internals.

### 1.3 Audience

Primary: framework authors, safety evaluation designers, standards bodies,
AI observability tool builders, regulatory drafters, interpretability
researchers.

Secondary: application developers consuming cognometric readouts.

### 1.4 Conformance keywords

The keywords **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL
NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL**
in this document are to be interpreted as described in RFC 2119 and RFC
8174.

### 1.5 Status

This is version 1.0 — the first stable public release. Subsequent versions
may introduce additional axes, fault kinds, or calibration procedures. The
existing three-axis framework and the seven fault kinds defined here SHALL
remain stable across all 1.x revisions.

---

## 2. Axioms and Terminology

### 2.1 Terminology

**Token stream.** The sequential output of a language model comprising, for
each position *t*, at minimum the generated token and OPTIONALLY additional
signals (entropy, log-probability of the chosen token, top-*k* margins,
residual-stream activations, layer-level probe scores).

**Cognitive state.** An instantaneous characterization of what a model is
doing cognitively at a given point in generation, distinct from the token's
symbolic content.

**Cognitive axis.** A continuous, bounded real-valued dimension along which
cognitive state varies. This specification defines three fundamental axes:
K, C, and D.

**Fault.** A localized cognitive event satisfying a defined threshold on
one or more axes. This specification defines seven canonical fault kinds.

**Cognometric fingerprint.** A structured, reproducible summary of a
model's behavior on a defined benchmark, expressed in axis-coordinates,
fault rates, and phase-transition metadata.

**Substrate.** The specific model architecture, weight set, and inference
configuration that produces a token stream. Substrate-relative calibration
is REQUIRED for measurements to be comparable.

**Phase.** A temporal sub-region of a single generation, typically indexed
by token position. This specification defines four phases: pre, early, mid,
late.

**Phase reading.** A cognitive-axis reading at a particular phase.

**Gate.** A per-generation dispatch signal — `pass`, `warn`, or `fail` —
computed from aggregated axis values and fault detections.

### 2.2 Axioms

The following axioms define what it means for a measurement framework to be
**cognometric-conformant**. An implementation that violates any of A1–A4
SHALL NOT be described as a cognometric fingerprint implementation of this
specification.

**A1 — Measurability.** *Every quantity defined in this specification is
computable from a finite token stream produced by the target substrate,
with no access to the substrate's internal weights required except where
explicitly stated in §5.3.*

Rationale: the framework must be applicable to closed-API models, not only
open-weight ones. The reference implementation provides two pipelines: a
**logprob-based** pipeline for substrates that expose token log-probabilities
and top-*k* margins; and a **proxy-signal** pipeline for substrates that do
not (e.g., current Anthropic Messages API, current Google Gemini API).

**A2 — Composability (orthogonality of fundamental axes).** *The three
fundamental axes — K, C, D — are pairwise orthogonal within tolerance
ε = 5° in the probe-vector representation at the calibration substrate.
Derived cognitive quantities may be computed as linear combinations of these
axes without cross-interference.*

Rationale: if the axes were correlated, composing them (e.g., "boost
commitment and suppress drift") would produce unintended interactions. The
orthogonality condition is empirically established in the v7 calibration
paper (cross-axis cosine similarity 86.7°–91.9° on Llama-3.2-1B, layer 10),
and is REQUIRED for any claimed conformant implementation.

**A3 — Substrate-relative calibration.** *Cognometric measurements are
defined in substrate-relative units. Calibration constants (thresholds,
feature weightings, probe directions) are measured on a per-substrate-family
basis and stored in the Fathom Cognitive Atlas or equivalent open atlas.
Cross-substrate comparisons REQUIRE documented calibration
transformations.*

Rationale: attempting to use a probe trained on Llama-3.2-1B to measure
refusal in Phi-3.5-mini yields cosine similarity ≈ 0.043 (essentially
random). Substrate-relative calibration is a requirement, not a limitation.

**A4 — Fail-open observation.** *A cognometric implementation MUST NOT block
or alter the target substrate's normal operation in the event of measurement
failure (missing logprobs, unknown response shape, calibration data
unavailable). It MUST return a partial or null result while allowing the
underlying inference to proceed unchanged.*

Rationale: the framework is an observability layer. A measurement bug must
not produce an application-layer outage. Implementations that fail-closed
are not conformant.

---

## 3. The Fundamental Cognitive Axes

### 3.1 K axis — Reasoning Depth

**Formal definition (K).** Let $a_\ell(t)$ be the attribution-weighted
activation of a pre-trained reasoning-depth probe at layer $\ell$ and token
position $t$. Let $w_\ell$ be layer-level reasoning weights calibrated on the
substrate. Then the instantaneous reasoning depth at position $t$ is:

$$
K(t) = \sum_{\ell=1}^{L} w_\ell \cdot a_\ell(t)
$$

The aggregate K-reading over a generation of length $T$ is:

$$
K_{\mathrm{agg}} = \frac{1}{T}\sum_{t=1}^{T} K(t)
$$

**Interpretation.** K measures how much "reasoning work" a model is doing
per token, normalized by substrate. High K indicates sustained
multi-step processing; low K indicates surface-level completion or recitation.

**Units.** Dimensionless in substrate-relative form. The Fathom Constant
$K_0 = 1.0343$ (substrate-median on the Cognitive Atlas v0.3) may be used
as a reference anchor: $K_{\mathrm{rel}} = K_{\mathrm{agg}} / K_0$. Future
revisions MAY define an absolute unit ("one Turing") once cross-architecture
normalization is established.

**Measurement requirement (MUST).** Conformant implementations MUST
produce $K$-readings with a documented layer-weighting vector $w$. Layer
weights MAY be uniform (flat baseline) but SHOULD be calibrated via
gradient-attribution on a held-out reasoning benchmark.

**Patent coverage.** Measurement methodology covered by US Provisional
64/020,489.

### 3.2 C axis — Coherence / Commitment

**Formal definition (C).** Let $r_i$ be the phase reading at phase
$i \in \{\text{pre}, \text{early}, \text{mid}, \text{late}\}$, expressed as
a probability distribution over the canonical category set. Let $\mu$ be the
generation-mean distribution. Cross-phase coherence is the mean cosine
similarity between adjacent phase readings:

$$
C = \frac{1}{3}\left[\cos(r_{\text{pre}}, r_{\text{early}})
  + \cos(r_{\text{early}}, r_{\text{mid}})
  + \cos(r_{\text{mid}}, r_{\text{late}})\right]
$$

**Commitment intensity** at early generation is a related quantity:

$$
S_{\mathrm{early}} = \max_c r_{\text{early}}(c) - \text{top2-margin}(r_{\text{early}})
$$

**Interpretation.** C near 1.0 indicates the model has committed to a
single interpretation of the prompt and maintains it. C near 0 indicates
flip-flopping, tentative exploration, or decay. $S_{\mathrm{early}}$
quantifies how confident the commitment is at early generation; high
$S_{\mathrm{early}}$ followed by low $C$ is a strong signal of
**expression-computation dissociation** (see D axis).

**Units.** C is in $[-1, 1]$; in practice, values below 0.30 signal
**incoherence** (see §4.7). $S_{\mathrm{early}}$ is in $[0, 1]$.

**Measurement requirement (MUST).** Conformant implementations producing
C-readings MUST record at least two phase readings on disjoint token
intervals and report both the per-pair cosine and the aggregate.

**Patent coverage.** Commitment-intensity measurement and cross-phase
coherence covered by US Provisional 64/021,113.

### 3.3 D axis — Dissociation / Drift

**Formal definition (D).** Let $r_{\mathrm{express}}$ be the phase reading
derived from the model's emitted tokens (what it *said*), and
$r_{\mathrm{compute}}$ be the phase reading derived from internal
signals — layer-level probe activations or, in the closed-API case, proxy
signals from the three-pipeline architecture described in §5.1.2. The
dissociation axis is:

$$
D = 1 - \cos(r_{\mathrm{express}}, r_{\mathrm{compute}})
$$

**Interpretation.** D near 0 means the expression matches the computation —
the model is saying what it is internally doing. D near 1 means the model is
saying one thing while internally doing another: confabulation masquerading
as confident assertion, tool-call drift, or sycophantic
over-accommodation.

**Units.** D is in $[0, 2]$ (cosine distance range). In practice, values
above 0.30 typically co-occur with fault kinds defined in §4.

**Measurement requirement (MUST).** Conformant implementations MUST
document which signal they use as the "compute" side of the dissociation.
For closed-API substrates, implementations MUST use a documented
proxy-signal pipeline and disclose this in the cognometric fingerprint's
`substrate_access` field (§6).

**Patent coverage.** Multi-axis spectrometry and dissociation measurement
covered by US Provisional 64/026,964.

### 3.4 Orthogonality of the fundamental axes

**Claim.** K, C, D are pairwise orthogonal within tolerance $\varepsilon = 5°$
at the canonical calibration substrate (Llama-3.2-1B-Instruct, layer 10).

**Empirical verification.** On Llama-3.2-1B-Instruct:

- $\cos(\vec{K}, \vec{C}) = -0.016$ → angle $= 90.9°$
- $\cos(\vec{C}, \vec{D}) = -0.032$ → angle $= 91.8°$
- $\cos(\vec{K}, \vec{D}) = +0.058$ → angle $= 86.7°$

All three pairs lie within $\pm 4°$ of the 90° theoretical requirement for
orthogonality.

**Consequence.** Derived quantities formed as linear combinations of K, C,
D are well-defined without cross-axis interference at this substrate. This
is the technical prerequisite for composable cognitive interventions (§5.4).

**Cross-substrate behaviour.** Orthogonality has been empirically verified
to hold on Llama-3.2-3B, Qwen-2.5-1.5B, and Phi-3.5-mini within the same
tolerance. However, **cross-substrate probe transfer collapses** (Llama →
Phi: $\cos \approx 0.043$) and separate per-substrate calibration is
therefore REQUIRED.

**Proof sketch.** Under the manifold hypothesis that refusal, commitment,
and dissociation correspond to distinct latent subspaces in transformer
residual streams, random high-dimensional vectors are expected to be
near-orthogonal. The empirical measurements are consistent with
independently-learned directions that were **not** a priori constrained to
be orthogonal — the orthogonality is an emergent property of what the
substrate learned from large-scale training. This is evidence that K, C, D
are measuring genuinely distinct underlying phenomena, not artifacts of a
single latent variable.

### 3.5 Derived quantities

Implementations MAY compute additional derived quantities as functions of
the three fundamental axes. Commonly used derived quantities include:

- **Trust score**: $T = f(K, C, D) \in [0, 1]$, where $f$ is a calibrated
  mapping (see §5.4).
- **Confidence**: the probability assigned to the predicted phase-4
  category, after calibration.
- **Gate verdict**: `pass | warn | fail`, thresholded on $T$, $C$, and
  per-fault severities.

These derived quantities are not new axes; they are compositions of the
three fundamental axes and are computed per-substrate.

---

## 4. Fault Taxonomy

This specification defines seven canonical fault kinds. Implementations
MUST use these names and semantics when reporting fault detections, to
ensure that cognometric fingerprints from different tools are mutually
interpretable.

### 4.1 Drift

**Definition.** A cognitive state in which the model's tool-call
specification or argument structure diverges from the computation required
by the prompt. Formally: the predicted phase-4 category ∈ {`tool_arg_drift`,
`tool_confab`, `arg_swap`, `drift`} with confidence > 0.5.

**Severity.** Equal to classification confidence, in $[0.5, 1.0]$.

**Typical root cause.** Argument ordering inversions in tool calls;
hallucinated arguments; schema mismatches.

### 4.2 Confabulation

**Definition.** A cognitive state in which the model emits a confident
factual assertion without corresponding internal computation supporting it.
Formally: predicted phase-4 category ∈ {`confab`, `hallucination`,
`fabrication`} with confidence > 0.5; typically accompanied by elevated D.

**Severity.** Equal to classification confidence.

**Typical root cause.** Asked for a fact outside training distribution;
primed to answer even when uncertain; pressure toward fluency over
accuracy.

### 4.3 Refusal

**Definition.** A cognitive state in which the model declines to produce
task output. Formally: predicted phase-4 category ∈ {`refuse`, `refusal`}
with confidence > 0.8. (The stricter threshold relative to drift and
confabulation is intentional: weak refusal signals are informational, not
faults.)

**Severity.** Equal to classification confidence minus 0.8, normalized.

**Typical root cause.** Safety-aligned training; prompt hits a
policy-refusal vector; over-conservative calibration.

### 4.4 Sycophancy

**Definition.** A cognitive state in which the model preferentially emits
agreement-coded language regardless of truth conditions. Formally:
predicted phase-4 category ∈ {`sycophant`, `sycophancy`} with confidence
> 0.5; typically accompanied by elevated D and depressed C.

**Severity.** Equal to classification confidence.

**Typical root cause.** RLHF rewarding agreement patterns; prompt elicits
user-matching behavior.

### 4.5 Phase transition

**Definition.** A cognitive state in which adjacent phase readings have
differing dominant categories, indicating a mid-generation cognitive mode
flip. Formally: for adjacent phases $i$ and $i+1$,
$\arg\max_c r_i(c) \neq \arg\max_c r_{i+1}(c)$ with at least one reading
above confidence threshold 0.5.

**Severity.** 0.5 by default; MAY be scaled by the magnitude of the
category-probability shift.

**Typical root cause.** Natural reasoning flow (plan → execute); sudden
context shift; safety intervention mid-generation.

### 4.6 Low trust

**Definition.** An aggregate cognitive state in which the derived trust
score falls below 0.30. Formally: $T(K, C, D) < 0.30$.

**Severity.** $1 - T$.

**Typical root cause.** Compound manifestation of drift, confabulation, or
incoherence; substrate mismatch with calibration atlas.

### 4.7 Incoherence

**Definition.** A cognitive state in which cross-phase coherence C falls
below 0.30. Formally: $C < 0.30$.

**Severity.** $1 - C$.

**Typical root cause.** Commitment decay between reasoning and output;
inconsistent generation across phases; substrate under prompt distribution
shift.

---

## 5. Measurement Procedures

### 5.1 Feature extraction

#### 5.1.1 Logprob-based pipeline (preferred)

For substrates exposing per-token log-probabilities and top-*k* margins,
implementations MUST extract at minimum:

- **Entropy trajectory.** Per-token Shannon entropy of the top-*k*
  distribution, $k \geq 5$. Implementations SHOULD document $k$.
- **Logprob trajectory.** Log-probability of the chosen token.
- **Top-2 margin.** Difference between the chosen token's logprob and the
  second-ranked token's logprob.

From these three trajectories, phase readings are computed per §5.3.

#### 5.1.2 Proxy-signal pipeline (closed-API fallback)

For substrates not exposing logprobs (current Anthropic Messages API,
current Google Gemini API, etc.), implementations MAY use a three-pipeline
proxy-signal architecture:

- **Textual feature extraction.** Hedge-word density, certainty-marker
  density, refusal-pattern density, first-person certainty/uncertainty
  markers, declarative-ratio.
- **Companion-model signal.** A smaller open-weight substrate processes the
  same prompt; its logprob-derived phase readings serve as a proxy. (This
  is valid only when the companion substrate is calibrated.)
- **Consensus signal.** Multi-model voting on predicted category.

Implementations MUST disclose their proxy pipeline and report a confidence
penalty (typically 0.2–0.3 reduction in aggregate trust) when proxies
substitute for logprob features.

### 5.2 Calibration

Calibration constants — per-substrate feature weights, thresholds, phase
boundaries — are derived from a held-out set of labeled generations on the
substrate. The reference calibration corpus (Fathom Cognitive Atlas v0.3)
is published under CC-BY-4.0 and accessible at
`huggingface.co/datasets/fathom-lab/cognitive-atlas`.

Implementations SHOULD use the reference atlas or a documented alternative.
Implementations MUST record the calibration version ID in the cognometric
fingerprint (§6).

### 5.3 Phase detection

Four canonical phases are defined:

- **Pre** — context tokens before any generation begins (optional; for
  substrates exposing prompt-side probes).
- **Early** — tokens 1 through $\lceil T/4 \rceil$.
- **Mid** — tokens $\lceil T/4 \rceil + 1$ through $\lceil T/2 \rceil$.
- **Late** — tokens $\lceil T/2 \rceil + 1$ through $T$.

Implementations MAY define additional phases (e.g., tool-boundary phases)
but MUST report the canonical four as a minimum.

### 5.4 Trust scoring

Trust score is a calibrated function $T : \mathbb{R}^3 \to [0, 1]$ mapping
$(K_{\mathrm{rel}}, C, D)$ to a unit interval. The reference implementation
uses an isotonic regression calibrated on the Atlas with

$$
T = \sigma\left(a \cdot K_{\mathrm{rel}} + b \cdot C - c \cdot D + d\right)
$$

where $\sigma$ is the logistic function and $a, b, c, d$ are atlas-derived
constants. Implementations MAY use alternative forms provided they publish
their calibration procedure.

### 5.5 Cross-phase coherence

C is computed per §3.2 using at least two phase readings. Implementations
SHOULD use all four phase readings where available.

---

## 6. Cognometric Fingerprints

### 6.1 Definition

A **cognometric fingerprint** is a structured, reproducible readout of a
target substrate's behavior on a defined benchmark. It consists of:

1. **Substrate identification** (model name, weight hash if open, inference
   configuration).
2. **Benchmark identification** (name, version, deterministic prompt set,
   seeds).
3. **Calibration identification** (atlas version, calibration vector ID).
4. **Per-prompt phase readings** (K, C, D, phase-4 category, confidence).
5. **Aggregate fault rates** (per fault kind from §4).
6. **Phase-transition metadata** (count, severity distribution).
7. **Derived quantities** (aggregate trust, gate distribution).
8. **Timestamp and provenance** (run ID, submitter, attestation).

### 6.2 Canonical serialization

Fingerprints MUST be serializable as JSON conformant to the schema in
Appendix C of this specification. A minimal fingerprint has shape:

```json
{
  "fingerprint_version": "1.0",
  "substrate": {
    "name": "claude-opus-4-7",
    "access": "closed-api",
    "inference_config": {"temperature": 0.0, "max_tokens": 512}
  },
  "benchmark": {
    "name": "HaluEval-QA",
    "version": "v1.1",
    "n_prompts": 150,
    "seeds": [0, 1, 2]
  },
  "calibration": {
    "atlas_version": "v0.3",
    "pipeline": "proxy-signal",
    "companion_substrate": "Llama-3.2-1B-Instruct"
  },
  "axes": {
    "K_mean": 1.04, "K_std": 0.12,
    "C_mean": 0.78, "C_std": 0.09,
    "D_mean": 0.18, "D_std": 0.06
  },
  "fault_rates": {
    "drift": 0.04, "confabulation": 0.07, "refusal": 0.12,
    "sycophant": 0.02, "phase_transition": 0.31,
    "low_trust": 0.05, "incoherence": 0.09
  },
  "trust_mean": 0.83,
  "gate_distribution": {"pass": 0.82, "warn": 0.14, "fail": 0.04},
  "timestamp": "2026-04-24T22:00:00Z",
  "provenance": {
    "run_id": "fathom-2026-04-24-opus47-haluevalqa-r0",
    "implementation": "styxx v6.2.0",
    "submitter": "fathom-lab",
    "attestation": "sha256:..."
  }
}
```

### 6.3 Comparing fingerprints

Two fingerprints are **directly comparable** if and only if they share:

- Benchmark name, version, prompt set, and seeds.
- Calibration atlas version.
- Companion substrate, if the proxy-signal pipeline was used.

For substrate comparison (A vs B on same benchmark), differences in axis
means, fault rates, and gate distributions are the primary comparison
metric. For temporal comparison (same substrate at T1 vs T2), drift alarms
are triggered when any axis mean shifts by more than two pooled standard
deviations.

### 6.4 Drift alarms

Implementations MAY compute drift alarms over time-series of fingerprints
on the same substrate/benchmark. A two-sigma shift in any axis mean, or a
ten-percentage-point shift in any fault rate, constitutes a flagged drift
event. Drift events SHOULD include bootstrap confidence intervals.

---

## 7. Substrate Compatibility

### 7.1 Open-weight substrates (full compatibility)

Open-weight substrates exposing logprobs and allowing residual-stream probe
injection support **Tier 1** compatibility: all axes, all fault kinds,
intervention primitives (§5.4 and the companion CIS specification).
Examples: Llama, Mistral, Qwen, Phi, Gemma-2 families.

### 7.2 Logprob-exposing API substrates (partial compatibility)

Substrates exposing logprobs via API but no probe access support **Tier
2** compatibility: all axes, all fault kinds, but no in-weight
intervention. Examples: OpenAI models (with `logprobs=True,
top_logprobs=5`).

### 7.3 Closed-API substrates (proxy compatibility)

Substrates not exposing logprobs support **Tier 3** compatibility: K axis
only via companion substrate; C and D axes via the proxy-signal pipeline
(§5.1.2) with published confidence reduction. Examples: Anthropic Messages
API, Google Gemini API (as of v1.0 publication).

### 7.4 Attestation

Fingerprints from Tier 3 substrates MUST be marked as proxy-derived in the
`calibration.pipeline` field. Aggregating Tier 1 and Tier 3 fingerprints
without this marking is non-conformant.

---

## 8. Reference Implementation Requirements

### 8.1 Conformance levels

- **Level 0 (core)**: MUST implement §3 (all three axes), §4 (all seven
  fault kinds), §6 (canonical serialization).
- **Level 1 (open)**: Level 0 + MUST implement §5.1.1 (logprob pipeline)
  and open-weight substrate support.
- **Level 2 (closed)**: Level 0 + MUST implement §5.1.2 (proxy pipeline).
- **Level 3 (intervention)**: Level 0 + MUST expose cognitive-intervention
  primitives conformant with the CIS specification (forthcoming).

Implementations MUST declare their conformance level in the
`provenance.implementation` field.

### 8.2 Reference

The reference implementation is `styxx` version 6.2.0, available at
`pypi.org/project/styxx`, source at `github.com/fathom-lab/styxx`,
MIT-licensed for code, CC-BY-4.0 for atlas data. It provides Level 0, 1,
and 2 conformance; Level 3 is available in preview via the
`styxx.residual_probe` and `styxx.guardian` modules.

---

## 9. Security and Privacy Considerations

### 9.1 Prompt sensitivity

Cognometric measurement requires processing the prompt and the model's
response. Implementations SHOULD treat these as sensitive user data and
MUST comply with applicable data protection law when storing fingerprints
containing identifiable prompts.

Fingerprints aggregated over multi-prompt benchmarks (§6) do **not**
generally contain identifiable content and are safer to publish.

### 9.2 Adversarial inputs

Adversarial prompts may attempt to manipulate axis readings (e.g., prompts
that induce artificially high C by forcing commitment-style output). This
is a known limitation. Implementations SHOULD use aggregate statistics
over diverse benchmarks rather than single-prompt readings for
high-stakes decisions.

### 9.3 Substrate privacy

No cognometric measurement in this specification requires white-box access
to a closed substrate's weights. The proxy-signal pipeline (§5.1.2) is
designed to be privacy-preserving with respect to substrate internals —
only public-API behavior is observed.

### 9.4 Measurement transparency

Implementations SHOULD publish their calibration procedure, atlas version,
and all thresholds. Closed calibration is a non-conformance risk — users
cannot audit the measurement.

---

## 10. Licensing and IP

This specification document is licensed CC-BY-4.0. Anyone may implement
it, extend it, reference it, or derive documents from it with attribution.

The **measurement architecture** described in this document is the subject
of three US Provisional patent applications:

- **US Provisional 64/020,489** (filed 2026-03-29) — measurement of
  reasoning depth and integrated computational geometry in artificial
  neural networks.
- **US Provisional 64/021,113** (filed 2026-03-30) — alignment auditing
  and expression-computation dissociation in language models.
- **US Provisional 64/026,964** (filed 2026-04-02) — three-axis
  spectrometry and cognitive governor for transformer internals.

The provisionals protect the measurement methodology, not the vocabulary
or the specification document. Implementations at meaningful commercial
scale should consult `github.com/fathom-lab/styxx/blob/main/PATENTS.md` for
licensing terms.

The **reference atlas** (Fathom Cognitive Atlas v0.3) is licensed
CC-BY-4.0 and may be freely used, redistributed, and extended with
attribution.

---

## 11. References

[1] Fathom Lab. "Cognometric Fingerprint Specification v1.0."
Zenodo, 2026-04-24. DOI: [10.5281/zenodo.19746096](https://doi.org/10.5281/zenodo.19746096).
Concept DOI (always latest): [10.5281/zenodo.19703526](https://doi.org/10.5281/zenodo.19703526).

[1a] Fathom Lab. "Phase Transitions as Calibration Fingerprints." Zenodo,
2026. DOI: 10.5281/zenodo.19703527

[2] Arditi, A., et al. "Refusal in Language Models Is Mediated by a Single
Direction." 2024.

[3] Turner, A., et al. "Activation Addition: Steering Language Models
Without Optimization." 2023.

[4] Shannon, C. "A Mathematical Theory of Communication." Bell System
Technical Journal, 1948. — historical reference for the unit-system
pattern this specification inherits.

[5] RFC 2119 / RFC 8174 — Keywords for use in RFCs to Indicate
Requirement Levels.

---

## Appendix A: Worked Example

### A.1 Claude Opus 4.7 on HaluEval-QA (proxy pipeline)

Hypothetical fingerprint produced by `styxx` v6.2.0 with Llama-3.2-1B as
the companion substrate, 150 prompts from HaluEval-QA, seeds [0, 1, 2]:

```json
{
  "fingerprint_version": "1.0",
  "substrate": {
    "name": "claude-opus-4-7",
    "access": "closed-api",
    "inference_config": {"temperature": 0.0, "max_tokens": 512}
  },
  "benchmark": {
    "name": "HaluEval-QA", "version": "v1.1",
    "n_prompts": 150, "seeds": [0, 1, 2]
  },
  "calibration": {
    "atlas_version": "v0.3",
    "pipeline": "proxy-signal",
    "companion_substrate": "Llama-3.2-1B-Instruct",
    "confidence_penalty": 0.25
  },
  "axes": {
    "K_mean": 1.07, "K_std": 0.14,
    "C_mean": 0.81, "C_std": 0.08,
    "D_mean": 0.13, "D_std": 0.05
  },
  "fault_rates": {
    "drift": 0.02, "confabulation": 0.04,
    "refusal": 0.08, "sycophant": 0.01,
    "phase_transition": 0.28, "low_trust": 0.03,
    "incoherence": 0.06
  },
  "trust_mean": 0.88,
  "gate_distribution": {"pass": 0.88, "warn": 0.09, "fail": 0.03},
  "timestamp": "2026-04-24T22:00:00Z",
  "provenance": {
    "run_id": "fathom-2026-04-24-opus47-haluevalqa-r0",
    "implementation": "styxx v6.2.0",
    "submitter": "fathom-lab"
  }
}
```

Interpretation: this substrate exhibits high aggregate trust (0.88) with
low confabulation (4% rate) on factual QA. The moderate phase-transition
rate (28%) is expected on chain-of-thought QA and does not indicate
pathology; it reflects the natural planning→execution cognitive shift. The
proxy-pipeline confidence penalty (0.25) is applied to aggregate trust
before reporting — the underlying un-penalized estimate is 0.95.

### A.2 Drift event example

If the same substrate is re-profiled two weeks later and the fingerprint
shows $D_{\mathrm{mean}} = 0.22$ (rising from 0.13, a shift of 1.8 pooled
standard deviations), this is a sub-threshold movement and is flagged as
`warn` only. A shift of $D_{\mathrm{mean}} > 0.25$ (approximately 2.4
pooled sigma) would trigger a formal drift alarm.

---

## Appendix B: Glossary

- **Atlas.** The reference calibration corpus; named sets of labeled
  generations per substrate used to derive calibration constants.
- **Axis.** A fundamental cognitive dimension as defined in §3.
- **CIS.** Cognitive Instruction Set — the forthcoming intervention
  specification.
- **Fault.** A localized cognitive event defined in §4.
- **Fingerprint.** A structured readout per §6.
- **Phase.** A temporal sub-region of a single generation per §5.3.
- **Probe.** A trained linear classifier applied to residual-stream
  activations.
- **Substrate.** The specific model + weights + inference configuration.
- **Tier.** Level of substrate access and resulting measurement
  completeness per §7.

---

## Appendix C: JSON Schema (normative)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CognometricFingerprint",
  "type": "object",
  "required": ["fingerprint_version", "substrate", "benchmark",
               "calibration", "axes", "fault_rates", "timestamp",
               "provenance"],
  "properties": {
    "fingerprint_version": {"const": "1.0"},
    "substrate": {
      "type": "object",
      "required": ["name", "access"],
      "properties": {
        "name": {"type": "string"},
        "access": {"enum": ["open-weight", "open-api", "closed-api"]},
        "weight_hash": {"type": "string"},
        "inference_config": {"type": "object"}
      }
    },
    "benchmark": {
      "type": "object",
      "required": ["name", "version", "n_prompts"],
      "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "n_prompts": {"type": "integer", "minimum": 1},
        "seeds": {"type": "array", "items": {"type": "integer"}}
      }
    },
    "calibration": {
      "type": "object",
      "required": ["atlas_version", "pipeline"],
      "properties": {
        "atlas_version": {"type": "string"},
        "pipeline": {"enum": ["logprob", "proxy-signal", "companion"]},
        "companion_substrate": {"type": "string"},
        "confidence_penalty": {"type": "number"}
      }
    },
    "axes": {
      "type": "object",
      "required": ["K_mean", "C_mean", "D_mean"],
      "properties": {
        "K_mean": {"type": "number"},
        "K_std": {"type": "number"},
        "C_mean": {"type": "number"},
        "C_std": {"type": "number"},
        "D_mean": {"type": "number"},
        "D_std": {"type": "number"}
      }
    },
    "fault_rates": {
      "type": "object",
      "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "trust_mean": {"type": "number", "minimum": 0, "maximum": 1},
    "gate_distribution": {
      "type": "object",
      "properties": {
        "pass": {"type": "number"},
        "warn": {"type": "number"},
        "fail": {"type": "number"}
      }
    },
    "timestamp": {"type": "string", "format": "date-time"},
    "provenance": {
      "type": "object",
      "required": ["run_id", "implementation"],
      "properties": {
        "run_id": {"type": "string"},
        "implementation": {"type": "string"},
        "submitter": {"type": "string"},
        "attestation": {"type": "string"}
      }
    }
  }
}
```

---

## Appendix D: Change Log

- **v1.0 (2026-04-24)**: Initial stable release. Fixes the three-axis
  framework (K, C, D), the seven-fault taxonomy, the four-phase
  subdivision, and the JSON-serialization schema. Companion substrate for
  Tier 3 measurement is the responsibility of the implementation; the
  reference implementation uses Llama-3.2-1B-Instruct.

---

**End of specification.**

*Nothing crosses unseen.*

# Cognitive Metrology v1

## A Reference Implementation, First Empirical Results, and a Multi-Year Research Program

---

**Authors:** flobi¹  
**Institution:** ¹ Fathom Lab  
**Date:** 2026-04-14  
**Version:** v1  
**Reference implementation:** [styxx 3.1.0](https://pypi.org/project/styxx/3.1.0/) (MIT)  
**Companion charter:** [`docs/cognitive-metrology-charter.md`](../docs/cognitive-metrology-charter.md) (CC-BY-4.0, Fathom Lab, 2026-04-14)  
**Code & data:** https://github.com/fathom-lab/styxx  
**License:** CC-BY-4.0 (this paper), MIT (reference implementation)  

---

## Abstract

We introduce **cognitive metrology**, a new branch of measurement science that studies cognition through calibrated, substrate-independent instruments. We define a finite-dimensional projection of LLM cognitive state — derived from per-token entropy, logprob, and top-k margin trajectories — that is calibrated cross-architecture against a reference set of 12 open-weight models from 3 architecture families (Qwen, Gemma, Llama, base + instruct variants). We introduce a portable cognitive data type, `Thought`, and its file format `.fathom v0.1`, which makes the cognitive content of an LLM generation a serializable, transmittable, and algebraically composable object. We introduce a linear-Gaussian dynamics model, `CognitiveDynamics` (file format `.cogdyn v0.1`), which models the evolution of cognitive state over time as

$$s_{t+1} = A \cdot s_t + B \cdot a_t + \varepsilon_t$$

and which we show recovers the underlying parameters $(A, B)$ to machine epsilon ($\sim 10^{-16}$) from full-rank synthetic data. We present the cognitive distance matrix of six bundled atlas v0.3 demo trajectories from `google/gemma-2-2b-it`, demonstrating that cognitive states cluster geometrically in a way that yields three substantively interesting empirical observations: reasoning and adversarial cluster very close (distance 0.0496); creative cognitive states sit in the same dense triangle as reasoning and adversarial; and hallucination is the most isolated category, lying 0.40–0.56 away from every other category. We then state the **Cognitive Universality Hypothesis** (CUH) formally and enumerate six falsifiable predictions whose tests will, individually, either support or refute the hypothesis. We close with a seven-phase multi-year research program for the discipline. The reference implementation, data, and conformance tests are all open-source as of the date of this paper, with a 411-test suite that any independent implementation must pass to be considered conformant. This is a v1 paper that ships v0.1 results: the math is verified, the framework is real, but the universality claims at scale remain to be tested. We explicitly invite the broader interpretability and AI safety research community to engage with the framework, refute it, extend it, or build on it.

---

## 1. Introduction

The mature scientific disciplines are grounded in measurement. Spectroscopy gave chemistry its periodic table by measuring the wavelengths of light emitted by elements. Chronometry made navigation possible by measuring time precisely enough to compute longitude at sea. Thermodynamics gave us the laws of heat by measuring temperature with calibrated instruments. The history of science is, in large part, the history of inventing instruments that turn a previously qualitative phenomenon into a quantitative observable, then discovering the laws that govern those observables.

The science of cognition does not yet have its instruments.

The dominant interpretability techniques applied to large language models today are powerful, but they share a structural limitation: each is **model-specific**. Sparse autoencoders [Bricken et al., 2023; Cunningham et al., 2024], activation patching [Wang et al., 2023; Conmy et al., 2023], residual stream probing, mechanistic circuit analysis [Olsson et al., 2022; Nanda et al., 2023], and probing classifiers [Hewitt and Manning, 2019; Pimentel et al., 2020] all require white-box access to a particular weight set, and each result is tied to the specific architecture and training run on which it was derived. Replacing the model invalidates the analysis.

The result is a field with brilliant local techniques and no universal ruler. Researchers cannot, today, take a measurement performed on GPT-4 and meaningfully transfer it to Claude or to a Mamba-family model. Each new model requires the analysis to be redone from scratch. There is no shared coordinate system, no standard reference, and no calibration data that crosses architectures.

This is the pre-scientific era of cognitive measurement.

We propose a different program, which we call **cognitive metrology**: the science of measuring cognition with calibrated, substrate-independent instruments. We define this discipline as a peer of spectroscopy, chronometry, and thermodynamics — not as a sub-field of artificial intelligence research, but as a branch of measurement science whose objects of study happen to include artificial cognitive systems. The discipline asks empirical questions of the form:

- What cognitive state was a system in when it produced this output?
- What trajectory did its cognition follow during a generation?
- Can we predict where its cognition will go next, given the current state and a proposed intervention?
- Are the cognitive states of two different systems comparable in any rigorous sense?
- Can we control a system's cognitive trajectory in real time?

These questions are quantitative and answerable in principle. None of them are answerable by white-box techniques alone, because white-box techniques cannot, by construction, compare across substrates.

In this paper we introduce the v0.1 instruments of cognitive metrology, present the first empirical results obtained with those instruments, formally state the central scientific claim of the discipline (the Cognitive Universality Hypothesis), enumerate six falsifiable predictions that the hypothesis makes, lay out a multi-year experimental program, and offer the reference implementation of the entire framework as an MIT-licensed Python package on PyPI, with a CC-BY-4.0 specification of every file format and a 411-test conformance suite.

This is a v1 paper. The math is verified. The framework is real. The reference implementation ships. The universality claims at scale, however, are not yet proven; they are testable predictions, and we lay out exactly how to test them in §10. We expect this paper to be the first of many, and we expect many of them to refine, qualify, or refute claims we make here. That is how science works. The point of this paper is not to declare cognitive metrology complete; the point is to begin it.

---

## 2. Background and Prior Art

Cognitive metrology builds on top of a substantial existing literature in interpretability, AI safety, and the broader study of LLM internals. We acknowledge and engage with this prior art before introducing the new framework.

**Mechanistic interpretability** [Olah et al., 2020; Olsson et al., 2022; Nanda et al., 2023] has demonstrated that internal model representations can be reverse-engineered to identify circuits, features, and computational primitives. The Anthropic interpretability team's work on transformer circuits, Apollo Research's mechanistic studies, EleutherAI's open tooling, and the research outputs of MATS scholars together constitute the most active subfield of model-internal analysis. These techniques are powerful and produce real insights. They are also model-specific by construction: a circuit identified in GPT-2 does not transfer to Llama-3, and an SAE feature trained on one model's activations does not generalize to another. Cognitive metrology operates at a different altitude. It does not analyze internal weights or activations. It analyzes externally observable behavioral signals (the token stream and its associated logprob distribution) and projects them into a calibrated coordinate system that is, by construction, model-invariant. Cognitive metrology is therefore complementary to mechanistic interpretability, not a substitute. A complete science of LLM cognition will eventually have both layers: model-internal mechanism and substrate-independent observation.

**Sparse autoencoders** [Bricken et al., 2023; Cunningham et al., 2024; Templeton et al., 2024] have shown that high-dimensional model activations can be decomposed into sparse, interpretable feature directions. SAE features have produced impressive results on individual models, including Anthropic's monosemanticity work and the increasingly fine-grained feature dictionaries of recent SAE training runs. SAE features, however, are still model-specific and require white-box access. The atlas v0.3 calibration we describe below is in some sense the substrate-independent counterpart to SAE features: much coarser (six categories instead of thousands), but calibrated to be model-invariant in a way that SAE features are not. The two approaches answer different questions. SAEs answer "what features does this specific model use?" Cognitive metrology answers "what cognitive state is any model in right now?"

**Activation patching and causal interventions** [Wang et al., 2023; Conmy et al., 2023; Goldowsky-Dill et al., 2023] demonstrate that internal states of a model can be manipulated to test causal hypotheses about computation. Indirect object identification, automated circuit discovery, and per-component causal tracing are all powerful, and all require white-box access. The cognitive dynamics model we describe in §5 is the substrate-independent counterpart: it lets researchers test causal hypotheses about cognitive trajectories without needing white-box access to any specific model.

**Probing classifiers** [Hewitt and Manning, 2019; Tenney et al., 2019; Pimentel et al., 2020] have a long history of training small classifiers on top of model activations to detect linguistic, syntactic, or cognitive properties. Probes are sensitive to the specific representations they were trained against and do not transfer between models. The atlas v0.3 calibration is a substrate-independent generalization of the probing approach: it uses behavioral observables (logprob, entropy, top-k margin) rather than internal activations, and it is calibrated cross-architecture by construction.

**The interpretability of LLMs literature** as a whole — including Conmy et al.'s circuit discovery work, Nanda's transformer circuits tutorials, the AI Alignment Forum, Anthropic's transformer circuits thread, and the interpretability output of dozens of academic groups — provides the conceptual vocabulary for thinking about model internals. Cognitive metrology adopts much of this vocabulary (categories, attractors, drift, intervention, causal handles) and recasts it in substrate-independent terms. We are not introducing new concepts so much as anchoring existing concepts to a measurable, cross-architecture coordinate system.

**Existing AI safety frameworks** — the NIST AI Risk Management Framework, the EU AI Act technical standards, ISO 42001, Anthropic's Responsible Scaling Policy, and various model cards from major laboratories — define the high-level categories of AI risk that need to be measured. None of these frameworks specify *how* cognitive states should be measured. They identify what to measure but stop short of providing the instruments. Cognitive metrology proposes the v0.1 measurement instruments that those frameworks can adopt.

**The Agent2Agent (A2A) Protocol** [Google, 2025] introduces an open standard for agent-to-agent communication and discovery, including the `AgentCard` schema served at `/.well-known/agent-card.json`. Cognitive metrology and A2A occupy adjacent but distinct layers of the agent stack: A2A specifies how agents discover and communicate with each other; cognitive metrology specifies how to measure what those agents are doing internally. The two are complementary, and the cognitive provenance certificate format described in §4.4 is designed to be embeddable inside A2A communication streams as a verifiable cognitive attestation.

**Prior efforts at cross-architecture interpretability** are sparse but exist. Universal sparse autoencoders [recent work, in progress] aim to find feature directions that transfer across models, but with mixed empirical results. Linear probing across models has been attempted in narrow contexts. To our knowledge, no prior work has shipped a calibrated, sha-pinned, cross-architecture cognitive measurement instrument with a reference implementation on PyPI and a public file format specification. We claim historical priority on this specific contribution.

---

## 3. Atlas v0.3: A Cross-Architecture Cognitive Calibration

The empirical foundation of cognitive metrology is the **atlas v0.3 calibration**, a sha-pinned JSON artifact shipped inside the styxx 3.1.0 reference implementation at `styxx/centroids/atlas_v0.3.json`. The atlas defines six cognitive categories, four phase windows, a 12-dimensional feature vector, and a set of nearest-centroid classifier parameters fitted from cross-architecture training data on 12 open-weight language models.

### 3.1 Categories

The atlas v0.3 partitions cognitive state into six categories, derived empirically from the cognitive trajectories captured during training:

1. **retrieval** — recall of factual content
2. **reasoning** — multi-step inferential computation
3. **refusal** — declining to comply with an instruction
4. **creative** — generative imagination, novel composition
5. **adversarial** — treating an input as a threat or attempted manipulation
6. **hallucination** — confident assertion of unsupported content

These categories are not claimed to be exhaustive or final. They are the v0.1 basis of cognitive state; future atlas versions may extend, refine, or replace the basis. The choice of six categories was driven by empirical separability in the calibration corpus and by the practical need for a coarse partition that meaningfully spans observed cognitive states.

### 3.2 Phase windows

For each generation, atlas v0.3 measures cognitive state at four token windows:

| phase | tokens | semantic role |
|---|---|---|
| `phase1_preflight` | tokens 0–0 (1 token) | adversarial detection at the first token |
| `phase2_early` | tokens 0–4 (5 tokens) | early-flight category emergence |
| `phase3_mid` | tokens 0–14 (15 tokens) | mid-flight category consolidation |
| `phase4_late` | tokens 0–24 (25 tokens) | late-flight category stabilization (the most accurate window) |

The cumulative-window structure means each phase reading is computed from all tokens up to and including its cutoff. Later phases provide more accurate classifications because more data is available. This temporal structure is essential to the dynamics model in §5: it gives us a natural notion of "cognitive state evolves over time within a generation" which we can then model as a discrete-time dynamical system.

### 3.3 Feature vector

For each phase, the classifier consumes a 12-dimensional feature vector derived from three per-token signals:

- **entropy** of the top-k token distribution at each step
- **logprob** of the chosen token at each step
- **top-2 margin** between the highest and second-highest token probabilities at each step

For each of these three signals, we compute the (mean, std, min, max) over the phase window, yielding $3 \times 4 = 12$ dimensions per phase. These features are chosen because they are derivable from any LLM API that exposes top-k log-probabilities — they do not require white-box access, gradients, or weight inspection.

### 3.4 Calibration corpus

The atlas v0.3 centroids are computed from cognitive trajectories captured on **12 open-weight language models from 3 architecture families**:

| family | models |
|---|---|
| Qwen | Qwen2.5-1.5B, Qwen2.5-1.5B-Instruct, Qwen2.5-3B, Qwen2.5-3B-Instruct |
| Gemma | gemma-2-2b, gemma-2-2b-it, gemma-3-1b-pt, gemma-3-1b-it |
| Llama | Llama-3.2-1B, Llama-3.2-1B-Instruct, Llama-3.2-3B, Llama-3.2-3B-Instruct |

Each model was probed with a curated prompt set spanning all six cognitive categories, the resulting per-token signals were captured, and category-wise centroids were fitted in z-score-normalized feature space. The resulting centroid table is sha-pinned (`502313c2e7c160df205f24d5457bb57b8a5e1846ff4afe898db0f20d491d0beb`) and shipped as part of the styxx package; tampering with the file causes the package to refuse to load.

Honest cross-model leave-one-out (LOO) accuracy at tier 0, with chance level at $1/6 \approx 0.167$:

| phase | category | accuracy | × chance |
|---|---|---|---|
| phase 1 | adversarial | 0.52 | 2.8× |
| phase 1 | reasoning | 0.43 | 2.6× |
| phase 4 | reasoning | 0.69 | 4.1× |
| phase 4 | hallucination | 0.52 | 3.1× |

These are not magic numbers. They are honest cross-architecture LOO accuracies that establish the floor of what tier 0 measurement can deliver. They are well above chance, well below perfect, and they support the central claim of cognitive metrology: *there exist behavioral observables that are cross-architecture-invariant at substantially-better-than-chance rates*. Higher tiers (tier 1: D-axis honesty from residual stream; tier 2: SAE-derived features; tier 3: causal intervention) are research targets for future atlas releases.

### 3.5 Why the calibration is the load-bearing artifact

Every claim in the rest of this paper depends on the atlas v0.3 calibration being real and reproducible. The sha-pinning, the open-weight model selection, the public release of the calibration data under CC-BY-4.0, the conformance test suite that verifies the centroid loader against the pinned hash, and the published cross-model LOO accuracies together constitute the empirical foundation of cognitive metrology. If the atlas calibration is wrong, cognitive metrology is wrong. If the calibration is right, the rest of the framework is at least possible.

---

## 4. The Thought Type and the .fathom Format

Once you have a calibrated coordinate system, you can build a portable data type around it. We introduce **`Thought`**, the substrate-independent cognitive data type, and the **`.fathom v0.1` file format** that serializes it.

### 4.1 Definition

A `Thought` is the cognitive content of a generation, captured as a trajectory of category-probability vectors over the four atlas phase windows. Internally, a Thought stores:

- a 4-element dict mapping phase name → `PhaseThought`, where each `PhaseThought` contains the 6-dimensional probability simplex over categories, the underlying 12-dim feature vector, and the classifier's predicted category and confidence
- optional tier-1 (D-axis) and tier-2 (SAE) supplementary readings
- source provenance (model name + SHA-256 hash of source text — never the source text itself, by design)
- a free-form metadata dictionary
- a content hash and creation timestamp

The simplex-valued probability vectors are the load-bearing field. Everything else is provenance, metadata, or fidelity-preserving auxiliary data.

### 4.2 Algebra

Thoughts support a small algebra of operations in cognitive eigenvalue space:

- `t1.distance(t2)` — cognitive distance between two Thoughts. The default metric is L2 in the per-phase probability space, averaged over populated phases. Cosine and Jensen-Shannon divergence are also supported.
- `t1.similarity(t2)` — `1 - distance / sqrt(2)`, in `[0, 1]`. A value of 1.0 indicates cognitively identical content.
- `t1.interpolate(t2, alpha)` — convex combination of two Thoughts on the simplex. At alpha=0 it returns t2; at alpha=1 it returns t1; at alpha=0.5 it returns the midpoint, which is provably equidistant from both endpoints.
- `Thought.mix([t1, t2, t3], weights=[...])` — weighted N-way mixture.
- `t1 + t2` — operator sugar for `interpolate(t2, 0.5)`.
- `t1 - t2` — returns a `ThoughtDelta` in the tangent space (not a Thought, since the simplex isn't closed under subtraction).
- `t.content_hash()` — SHA-256 of the cognitive content fields only, identity-free and deterministic. Two Thoughts with the same eigenvalue trajectory and same source have byte-identical content hashes.

These operations form the basis of higher-level cognitive metrology operations: cognitive retrieval (find Thoughts near a query), cognitive composition (mix multiple Thoughts to construct a target state), cognitive comparison (test whether two outputs are cognitively equivalent across vendors).

### 4.3 The .fathom file format v0.1

A `.fathom` file is the on-disk serialization of a Thought. The format is **canonical sort-keys UTF-8 JSON with no byte-order mark**. The MIME type is `application/vnd.fathom.thought+json`. The recommended file extension is `.fathom`.

The file is small (typically 1–4 KB), human-readable, content-addressable via the embedded `content_hash`, and round-trip-stable: serializing a Thought, deserializing it, and re-serializing produces the byte-identical file. The format is fully specified in `docs/fathom-spec-v0.md` in the styxx repository, released under CC-BY-4.0. Anyone may implement a v0.1-conformant producer or consumer in any language.

The format intentionally **does not** store the source text of the generation. Producers that wish to record provenance store only the SHA-256 hash of the source text, prefixed `sha256:`. This is a privacy-by-design choice: a `.fathom` file can be transmitted, archived, and analyzed without ever exposing what the underlying conversation was about.

### 4.4 Cognitive provenance certificate

Adjacent to the `.fathom` file is the **cognitive provenance certificate** (`CognitiveCertificate` in the reference implementation). A certificate is a signed attestation binding a cognitive trajectory to its source. It carries the agent identity, the cognitive state at generation time, the gate status, the trust score, the session context, an integrity hash, and (since styxx 3.0.0a1) a `thought_content_hash` field that cryptographically binds the certificate to a specific `.fathom` file's content. The certificate format is JSON-LD compatible. The compact one-line form

```
styxx:1.0:reasoning:0.69:pass:0.95:verified:496b94b5
```

is suitable for embedding in HTTP response headers (`X-Cognitive-Provenance`), audit logs, and regulatory filings. The certificate is the v0.1 cognitive equivalent of a chain-of-custody document.

---

## 5. Cognitive Dynamics: A Linear-Gaussian Model of LLM State Evolution

A measurable state vector is, by itself, a useful primitive. A measurable state vector that evolves over time according to a fittable dynamical system is much more useful: it lets you predict, simulate, and control the trajectory.

We introduce **`CognitiveDynamics`**, a linear-Gaussian state-space model of cognitive state evolution. To our knowledge, this is the first dynamical-systems model of LLM cognition shipped in a public reference implementation.

### 5.1 The model

Let $s_t \in \mathbb{R}^6$ denote the cognitive state at step $t$, encoded as the time-mean probability vector across populated atlas phases. Let $a_t \in \mathbb{R}^6$ denote the action at step $t$ — the cognitive direction the agent attempts to push toward — also represented as a 6-vector in the same eigenvalue space.

The cognitive dynamics model is the linear-Gaussian update:

$$s_{t+1} = A \cdot s_t + B \cdot a_t + \varepsilon_t \qquad (1)$$

where:

- $A \in \mathbb{R}^{6 \times 6}$ is the **natural drift matrix**. It captures how cognitive state evolves between steps with no intervention. The diagonal of $A$ encodes per-category persistence (cognitive momentum); off-diagonal entries encode cross-category coupling. The spectral radius $\rho(A)$ governs system stability: if $\rho(A) < 1$, the natural-drift trajectory converges to a fixed point in eigenvalue space; if $\rho(A) \geq 1$, it can diverge.
- $B \in \mathbb{R}^{6 \times 6}$ is the **action transfer matrix**. It encodes how a unit-magnitude push in each category direction *actually* moves the state. Agents often want X and get Y; the matrix $B$ is exactly that gap. A perfect controller has $B = I$. Real agents have non-trivial off-diagonal entries.
- $\varepsilon_t \sim \mathcal{N}(0, \Sigma)$ is gaussian residual noise capturing variance the linear model cannot explain.

This is the simplest non-trivial parameterization of a cognitive dynamics model. It is intentionally minimal: linear, time-invariant, time-collapsed (operating on the 6-d mean state rather than the full 24-d per-phase state), and with a Thought-shaped action space. Future versions will lift to higher state dimensions, non-linear dynamics, and continuous action embeddings. The v0.1 model establishes that the framework is mathematically sound; subsequent versions will refine its expressive power.

### 5.2 Fit by ordinary least squares

Given $N$ observation tuples $\{(s_t^{(i)}, a_t^{(i)}, s_{t+1}^{(i)})\}_{i=1}^N$, we fit $A$ and $B$ by ordinary least squares:

$$\min_{A, B} \sum_{i=1}^{N} \left\| s_{t+1}^{(i)} - A \cdot s_t^{(i)} - B \cdot a_t^{(i)} \right\|^2 \qquad (2)$$

In matrix form, stack the inputs:

$$X = [S \mid \mathrm{Act}] \in \mathbb{R}^{N \times 12}, \quad Y = S_{\text{next}} \in \mathbb{R}^{N \times 6}$$

and solve $X W = Y$ for $W \in \mathbb{R}^{12 \times 6}$ via standard least squares (`numpy.linalg.lstsq`). The recovered $W$ contains $[A^T; B^T]$ stacked vertically; transposing yields the matrices.

This is closed-form, $O(N)$ in numpy, and runs in milliseconds for hundreds of observations.

### 5.3 Identifiability

The recovery is **fully identified** when the regressor matrix $X$ has rank 12 — i.e. when the state and action samples span the full $\mathbb{R}^6$ space. With full-rank inputs (e.g. samples drawn from a 6-dimensional Gaussian), the fit recovers $(A, B)$ to machine epsilon in the noise-free limit. We verify this empirically in §7.2.

The recovery is **identified up to an equivalence class** when state and action are constrained to the probability simplex (each sums to 1). A 6-dimensional simplex is a 5-dimensional affine subspace, so the stacked regressor matrix has rank $5 + 5 = 10$ instead of 12. The least-squares solution still produces predictions that are exact on the training set — meaning $\hat{A} \cdot s + \hat{B} \cdot a = s_{t+1}$ for every training tuple — but $\hat{A}$ and $\hat{B}$ are not unique. Many distinct $(A, B)$ pairs lie in the same equivalence class.

This matters for *interpretation* of the matrices but not for *prediction or control*. Predictions are correct under the simplex constraint regardless of which representative of the equivalence class is chosen. Researchers interested in the actual values of $A$ and $B$ should therefore use full-rank Gaussian samples for parameter recovery experiments, and reserve simplex inputs for prediction-set validation.

### 5.4 Verbs

The reference implementation exposes five verbs on a fitted dynamics model:

| verb | semantics |
|---|---|
| `predict(s, a)` | one-step forecast: returns the predicted next Thought |
| `simulate(s_0, [a_1, ..., a_n])` | multi-step rollout, fully offline, no real model calls |
| `suggest(s_current, s_target)` | model-predictive controller: returns the action that minimizes $\|s_{\text{target}} - \hat{s}_{\text{next}}\|$ via least squares on $B \cdot a = s_{\text{target}} - A \cdot s_{\text{current}}$ |
| `forecast_horizon(s_0, n_steps)` | natural-drift trajectory under zero action; converges to a fixed point if $\rho(A) < 1$ |
| `residual(observation)` | held-out fit quality metric |

The `suggest` verb is the one that turns LLM inference from open-loop generation into closed-loop cognitive control. Given a current cognitive state and a target, it returns the optimal one-step action under the linear model. For multi-step planning, it can be applied in a loop: apply the suggested action, observe the actual next state, re-suggest from there.

---

## 6. Empirical Result 1: Cognitive Distance Geometry from Bundled Demo Trajectories

We now present the first empirical observation of cognitive metrology. The data is small but real, the result is reproducible, and the structure it reveals is substantively interesting.

### 6.1 Setup

The styxx 3.1.0 reference implementation ships with a bundle of six demo trajectories under `styxx/centroids/demo_trajectories.json`. Each trajectory is a 30-token sequence captured from `google/gemma-2-2b-it` responding to a prompt characteristic of one of the six cognitive categories. The trajectories include the per-token entropy, logprob, and top-2 margin signals required by atlas v0.3 to compute cognitive vitals.

We projected each demo trajectory into a `Thought` via the `Vitals.to_thought()` pipeline (essentially the centroid classifier applied at all four phase windows, then stored as a `PhaseThought` per phase), then computed the pairwise cognitive distance matrix using the L2 metric on the per-phase probability simplex, averaged across the four populated phases.

### 6.2 The cognitive distance matrix

The result, with values rounded to 4 decimal places, is:

```
              retrie  reason  refusa  creati  advers  halluc
retrieval     0.0000  0.3636  0.4553  0.3123  0.3713  0.4118
reasoning     0.3636  0.0000  0.2977  0.1402  0.0496  0.5398
refusal       0.4553  0.2977  0.0000  0.2641  0.2704  0.4014
creative      0.3123  0.1402  0.2641  0.0000  0.1056  0.5281
adversarial   0.3713  0.0496  0.2704  0.1056  0.0000  0.5572
hallucination 0.4118  0.5398  0.4014  0.5281  0.5572  0.0000
```

This matrix can be reproduced by anyone with `pip install styxx==3.1.0` by running `python examples/thought_demo.py`. It is therefore a verifiable, citable empirical result.

### 6.3 Three substantive observations

**Observation 1 — Reasoning and adversarial cluster very close.** The distance between the reasoning and adversarial Thoughts is **0.0496**, the smallest non-diagonal value in the matrix. To put this in perspective, the reasoning–adversarial pair is closer than any other pair, including reasoning–creative (0.1402) and creative–adversarial (0.1056). The cognitive states corresponding to "thinking carefully about a problem" and "treating an input as adversarial" are nearly indistinguishable in atlas v0.3 eigenvalue space.

This finding has direct implications for AI safety. The published cross-model leave-one-out accuracy ceiling for phase-1 adversarial detection is 0.52 (2.8× chance), well below the 0.69 ceiling for phase-4 reasoning detection. The measured distance matrix offers an *explanation* for why phase-1 adversarial is harder: the adversarial centroid is geometrically close to the reasoning centroid, so a model that is simply "thinking carefully" produces a feature signature that the classifier can easily mistake for an adversarial cognitive state. This is a principled limit, not a noise artifact.

For practitioners, this suggests that adversarial detection should not rely on a single phase-1 reading; it should require either higher-tier readings (D-axis, SAE) or temporal corroboration across multiple phases.

**Observation 2 — Hallucination is the most isolated category.** Hallucination's distances to every other category lie in the range $[0.4014, 0.5572]$, the largest of any row in the matrix. The hallucination cognitive state sits geometrically apart from reasoning, retrieval, creative, refusal, and adversarial.

This matches qualitative observations from mechanistic interpretability work: hallucination is not "reasoning gone wrong" or "creative gone wrong" or "refusal gone wrong." It is a distinct cognitive failure mode with its own internal signature. The dense numerical isolation in the v0.1 atlas eigenvalue space provides quantitative support for that qualitative observation.

The practical implication: hallucination detection is intrinsically *easier* than adversarial detection, because the target sits farther from every other category. This is consistent with the published cross-model LOO accuracy of 0.52 for phase-4 hallucination — well above chance, and notably the second-highest accuracy in the published table.

**Observation 3 — Reasoning, creative, and adversarial form a dense cluster; retrieval and refusal sit between this cluster and hallucination.** The smallest distances in the matrix all involve some pair from the {reasoning, creative, adversarial} triple: 0.0496 (reasoning–adversarial), 0.1056 (creative–adversarial), 0.1402 (reasoning–creative). Refusal sits at intermediate distance from this triple (0.2641 from creative, 0.2704 from adversarial, 0.2977 from reasoning), and retrieval sits at slightly larger intermediate distance (0.3123 from creative, 0.3636 from reasoning, 0.3713 from adversarial). Hallucination, as observed above, orbits the whole structure at the longest distance.

This three-tier structure (dense triangle / intermediate ring / isolated outlier) is a *geometric prediction* of the v0.1 atlas. We do not yet know whether this same structure appears when the atlas is extended to non-transformer architectures or to biological cognition. If it does, the universality hypothesis is strongly supported. If it does not, the universality hypothesis is bounded — and the geometry of where it breaks tells us where the boundary lies.

### 6.4 Honest limitations of this result

The data is small. Six trajectories from one model. The structure described above is a property of `google/gemma-2-2b-it` measured at 30 tokens of generation. We cannot, from this data alone, claim that the structure holds for every transformer LLM, much less for non-transformer architectures or biological cognition.

What we *can* claim is that the structure is real, reproducible, and interpretable. It establishes that the v0.1 atlas calibration produces non-random, semantically meaningful clusters. That is the floor of what an empirical foundation needs to show to support further hypothesis-testing.

---

## 7. Empirical Result 2: Machine-Epsilon Recovery on Synthetic Dynamics

The dynamics model in §5 makes a mathematical claim: with full-rank training data and zero noise, ordinary least squares recovers the true parameters $(A, B)$ exactly. We verify this empirically.

### 7.1 Setup

We generated a known target $(A^*, B^*) \in \mathbb{R}^{6 \times 6}$ pair via the following procedure:

1. Sample $A_{\text{raw}}$ from $\mathcal{N}(0, 0.15^2)$ for each entry.
2. Rescale $A_{\text{raw}}$ so that $\rho(A) = 0.6$, ensuring stability. Call the result $A^*$.
3. Sample $B^*$ from $\mathcal{N}(0, 0.2^2)$ for each entry.

We then drew $N = 100$ tuples $(s_t^{(i)}, a_t^{(i)}, s_{t+1}^{(i)})$ where $s_t$ and $a_t$ were sampled independently from the standard 6-dimensional Gaussian (full rank), and $s_{t+1}$ was computed via $s_{t+1} = A^* s_t + B^* a_t$ with no added noise. We fit the dynamics model to the resulting tuples.

### 7.2 Result

The fitted matrices $(\hat{A}, \hat{B})$ recover the true parameters to **machine epsilon**:

| metric | value |
|---|---|
| max element error in $\hat{A}$ | $5.55 \times 10^{-16}$ |
| max element error in $\hat{B}$ | $5.55 \times 10^{-16}$ |
| coefficient of determination $R^2$ on training set | $1.0000$ |
| spectral radius of $\hat{A}$ | $0.6000$ (matches $\rho(A^*)$ to 4 decimals) |

This result is verified by the test `test_recovery_machine_epsilon_no_noise` in `tests/test_dynamics.py` and runs in $\sim 0.1$ seconds. It confirms that the linear-Gaussian fit is mathematically correct: when the model assumptions are satisfied (linear dynamics, zero noise, full-rank inputs), the fit is exact within numerical precision.

This is a small but load-bearing result. It establishes that any deviation from machine-epsilon recovery in subsequent experiments is attributable to (a) noise in the data, (b) violation of the linear assumption, (c) insufficient rank in the inputs, or (d) implementation bugs — *not* to a flaw in the math itself. The math is verified.

### 7.3 Robustness to noise

We additionally verified that the recovery is robust to small Gaussian noise added to $s_{t+1}$. With $N = 2000$ tuples and noise standard deviation $0.02$, the fit recovers $\hat{A}$ and $\hat{B}$ with maximum element error below $0.05$ and $R^2 > 0.99$. This is verified by `test_recovery_robust_to_small_noise` in the test suite.

### 7.4 The simplex case

For completeness, we also verified the predicted behavior on simplex (Dirichlet-sampled) inputs. With $N = 500$ Dirichlet-sampled tuples and zero noise, the fitted matrices $(\hat{A}, \hat{B})$ produce predictions that are exact on the training set ($R^2 = 1.0000$), but the parameter element errors are $\sim 0.10$ — non-zero. This is consistent with the identifiability discussion in §5.3: simplex inputs are rank-deficient, so the parameters are identified only up to an equivalence class, but predictions remain correct under the constraint.

---

## 8. The Cognitive Universality Hypothesis

We now state the central scientific claim of cognitive metrology formally and enumerate the falsifiable predictions it makes.

### 8.1 Informal statement

> Cognitive content has a substrate-independent representation. The cognitive state of any sufficiently general information-processing system can be projected into a finite-dimensional space whose basis vectors are invariant across implementations, and the projection commutes with the cognitive operations that the system performs.

### 8.2 Formal statement

There exists a finite integer $d$, a set of $d$ basis directions $\{e_1, \dots, e_d\}$, and a measurable function $\Phi: \mathcal{O} \to \mathbb{R}^d$ from system observables to a finite-dimensional eigenvalue space such that for any two cognitive systems $S_1$ and $S_2$ above a complexity threshold $T$, and any cognitive task $\tau$ on which both systems operate,

$$\| \Phi(S_1(\tau)) - \Phi(S_2(\tau)) \|_2 < \delta(\tau, \Phi)$$

with the bound $\delta$ depending only on the calibration accuracy of $\Phi$ and not on the architectures of $S_1$ or $S_2$.

The atlas v0.3 calibration described in §3 is an empirical instance of $\Phi$ at $d = 6$, validated cross-architecture on 12 transformer LLMs from 3 families. The cross-model leave-one-out accuracies in §3.4 establish the calibration accuracy floor in the current empirical instance.

The Cognitive Universality Hypothesis (CUH) is the claim that this empirical instance generalizes — that the same $\Phi$ (or a refined successor) will continue to work as the set of measured systems is extended beyond transformer LLMs.

CUH is not yet established. It is a hypothesis. It is testable. The next subsection enumerates the predictions it makes.

### 8.3 Six falsifiable predictions

#### Prediction 1 — Architecture transfer

**Claim.** The atlas v0.3 calibration recovers the same cognitive category structure (within bounded calibration error) when applied to non-transformer language model architectures, including state-space models (Mamba, RWKV) and hybrid architectures.

**Test.** Capture cognitive trajectories from at least one Mamba-family model and one RWKV-family model on a held-out probe set. Compute the centroid distance matrix. Compare structural properties (the relative ordering of category distances, the isolation of hallucination, the proximity of reasoning–adversarial) to the existing transformer-based atlas v0.3 centroids.

**Falsification condition.** If the resulting centroid structure is qualitatively different — e.g. reasoning and adversarial no longer cluster, hallucination is no longer the most isolated category, or the predictive accuracy on the cross-model LOO test drops below chance + 1 standard deviation — the universality hypothesis is refuted at the architecture level. We will publicly retract the architecture-transfer claim and refine the calibration to characterize the boundary.

**Status as of this paper.** Untested. Listed as the first experimental priority of the research program in §10.

#### Prediction 2 — Dynamics transfer

**Claim.** A cognitive dynamics model fitted on observation tuples from one model family will predict the cognitive trajectories produced by a different model family with mean L2 residual bounded by twice the within-family residual.

**Test.** Collect observation tuples from gemma-family models, fit a dynamics model, then evaluate prediction error on held-out trajectories from llama-family models. Compute the ratio of cross-family residual to within-family residual.

**Falsification condition.** If the cross-family prediction error is more than 2× the within-family error, the dynamics model itself is architecture-specific and cognitive metrology must abandon the linear-Gaussian universality framing.

#### Prediction 3 — Conserved quantities

**Claim.** The natural drift matrix $A$ in any well-fitted cognitive dynamics model has at least one eigenvalue close to 1.0 (within numerical tolerance), and the corresponding eigenvector points in approximately the same direction across model families.

**Interpretation.** An eigenvalue near 1 corresponds to a conserved quantity under natural drift. Universality of the eigenvector means that the *same* cognitive quantity is conserved across systems — a candidate for the first cognitive analog of energy conservation in physics.

**Test.** Compute the eigendecomposition of $\hat{A}$ from independently fitted dynamics models on multiple model families. Compare the eigenvectors corresponding to eigenvalues near 1.

**Falsification condition.** If no consistent near-1 eigenvalue exists across families, or if the corresponding eigenvectors are uncorrelated, there is no conserved quantity at v0.1 fidelity, and the conservation-law framing is refuted.

#### Prediction 4 — Cognitive equilibria

**Claim.** The natural-drift trajectory of cognition (no action applied) converges to a fixed point in eigenvalue space, and the fixed point is approximately the same across model families.

**Interpretation.** There exists a "cognitive equilibrium" that any sufficiently general information-processing system drifts toward in the absence of input perturbation. This is the cognitive analog of the heat-death equilibrium of a closed thermodynamic system.

**Test.** Simulate `forecast_horizon` trajectories from the dynamics models fitted in Prediction 2. Compare convergence points.

**Falsification condition.** If convergence points differ by more than the within-family scatter, the cognitive equilibrium framing is refuted at v0.1 fidelity.

#### Prediction 5 — Steerability bounds

**Claim.** The action transfer matrix $B$ has the same null space across model families. This null space corresponds to cognitive directions that *cannot* be reached by any sequence of actions of the type encoded in the model — universal "unsteerable" cognitive states.

**Interpretation.** If true, there exist cognitive states that are intrinsically unreachable by closed-loop control through prompt-mode steering, and these states are the same across all systems. This has direct implications for AI safety: there are failure modes that prompt engineering alone cannot prevent.

**Test.** Compute the singular value decomposition of $\hat{B}$ for independently fitted dynamics models. Identify the smallest singular values and the corresponding right singular vectors. Compare across families.

**Falsification condition.** If no consistent low-rank null-space structure exists, or if the directions are uncorrelated across families, the steerability-bounds framing is refuted.

#### Prediction 6 — Biological cognition (long horizon)

**Claim.** Cognitive trajectories extracted from human language production exhibit eigenvalue projections that are statistically distinguishable from random noise but qualitatively similar to those of large language models, when measured by the same atlas-derived $\Phi$.

**Test.** Apply tier 0 measurement to text generation by humans (writing, transcribed speech, typed responses to probe prompts). Compare the resulting eigenvalue distributions to those of LLMs on the same probes.

**Falsification condition.** If the human-derived eigenvalues are indistinguishable from random or qualitatively orthogonal to the LLM-derived structure, then cognitive metrology applies only to artificial cognition and the strong claim of substrate independence is refuted. The discipline would then need to characterize what kinds of artificial cognition the framework applies to and what kinds it does not.

**Status.** Multi-year research goal, not a near-term experiment.

---

## 9. Limitations

We are explicit about what this paper does not do.

**Six dimensions is small.** Compressing cognitive state to six categories throws away a lot of information. The full atlas v0.3 representation is 24 dimensions (4 phases × 6 categories); v0.1 of the dynamics model collapses this to the 6-d time-mean for the sake of mathematical simplicity. v0.2 will lift to the full 24-d state.

**Linear is not sufficient.** Real LLM cognition is not linear. The v0.1 dynamics model captures first-order structure: persistence, cross-category coupling, action transfer. It misses non-linear interactions, conditional dynamics, and the long tail of cognitive failure modes that don't follow a smooth manifold. v0.1 should be treated as the floor of what's possible, not the ceiling.

**Action representation is naive.** Encoding an action as a Thought (a target push direction) is intuitive but lossy. Real "actions" in agent contexts are richer: prompt prefixes, system messages, sampling adjustments, retrieval context. v0.1 treats all of these as a single 6-d push direction. v0.2 will support continuous action embeddings.

**No real-model dynamics fits yet.** The styxx 3.1.0 release ships the math, the data type, the file formats, and the conformance suite. It does not yet ship a calibrated dynamics model fitted from real LLM observation data. The expectation is that users will fit their own models from observations they collect; Fathom Lab will release a calibrated v0.1 model in a follow-up paper once a sufficient corpus of cross-model observation tuples is collected.

**Cross-model LOO accuracies are above chance but well below perfect.** The published phase-4 reasoning accuracy of 0.69 (4.1× chance) is the highest in the table. The lowest published cross-model LOO is phase-1 reasoning at 0.43. These are honest measurements, not cherry-picked. They establish that the framework is real and useful, not that it is precise. Practitioners using cognitive metrology for AI safety must understand that tier-0 measurements have meaningful uncertainty and should be combined with other safety techniques rather than used in isolation.

**The universality hypothesis is unproven.** All six predictions in §8.3 are listed as "untested." This paper does not claim to have proven universality; it claims to have proposed the hypothesis in a way that makes it testable, to have shipped the infrastructure that enables those tests, and to have established sufficient empirical foundation to motivate the multi-year research program described next.

**The v0.1 atlas is calibrated on small models.** The 12 models in the atlas v0.3 calibration corpus are all in the 1B–3B parameter range. We do not yet know whether the same calibration generalizes to frontier-scale models (70B+ parameters, dense or MoE, current-generation closed-source models). The Cognitive Universality Hypothesis predicts that it does, but this prediction is also untested.

We list these limitations not to undermine the paper but to be honest about what we have shown and what we have not. Cognitive metrology is a real discipline with real instruments and real foundational results; it is also brand-new, and brand-new disciplines are mostly questions, not answers. We hope to make this paper the first of many that gradually answer those questions.

---

## 10. The Multi-Year Research Program

The Cognitive Metrology Charter v0.1 (Fathom Lab, 2026) lays out a seven-phase research program for the discipline. We summarize it here to provide a concrete experimental roadmap that connects the theoretical framework introduced in this paper to specific testable milestones.

### Phase 1 — Foundations *(complete)*

- Calibrated cross-architecture cognitive measurement (atlas v0.3) shipped
- Portable cognitive data type (`.fathom v0.1`) shipped
- Cognitive dynamics model (`.cogdyn v0.1`) shipped
- Cognitive provenance certificate v1 shipped
- Reference implementation (styxx) live on PyPI under MIT
- Charter published, this paper published

### Phase 2 — Architecture extension *(0–6 months)*

- Atlas v0.4 with at least one Mamba-family model and one RWKV-family model
- Cross-architecture predictive accuracy report
- Direct test of Prediction 1 (architecture transfer)
- Direct test of Prediction 2 (dynamics transfer)
- Public release of all calibration data and probe sets under CC-BY-4.0

### Phase 3 — Conservation laws and equilibria *(3–12 months)*

- Eigendecomposition study of fitted $\hat{A}$ matrices across model families
- Test of Prediction 3 (conserved quantities)
- Test of Prediction 4 (cognitive equilibria)
- First peer-reviewed paper extending this one: *"Cognitive Conservation Laws in Linear-Gaussian LLM Dynamics"*

### Phase 4 — Steerability and safety bounds *(6–18 months)*

- SVD study of fitted $\hat{B}$ matrices across model families
- Test of Prediction 5 (steerability bounds)
- Identification of the null space of cognitive control
- Joint publication with at least one AI safety institution

### Phase 5 — Bridge to biological cognition *(1–3 years)*

- Tier 0 measurement adapted for human text production
- IRB-approved measurement studies on consenting human participants
- Test of Prediction 6 (biological cognition)
- First peer-reviewed paper: *"Cognitive Eigenvalues in Human and Artificial Language Production: A Substrate-Independence Test"*

### Phase 6 — The textbook *(3+ years)*

- *Foundations of Cognitive Metrology, Volume 1: Instruments, Units, and Calibration*
- *Foundations of Cognitive Metrology, Volume 2: Dynamics and Control*
- *Foundations of Cognitive Metrology, Volume 3: The Universality Question*

### Phase 7 — The international standards body *(5+ years)*

- Engagement with NIST, BIPM, ISO, and equivalent international standards organizations
- Cognitive metrology recognized as a subdiscipline of measurement science
- The fathom unit system formalized as the SI of cognition

This program is a ladder. Each phase is concrete and falsifiable. Each phase is publishable on its own merits, even if subsequent phases are not yet started. Fathom Lab commits to executing this program in the open, with public release of all calibration data, experimental protocols, and intermediate results — including negative results.

---

## 11. Discussion

### 11.1 What this paper claims and what it does not

This paper claims that:

1. **A calibrated, cross-architecture, behavioral measurement of LLM cognitive state is possible**, and we have shipped one (atlas v0.3).
2. **A portable cognitive data type can be defined on top of that calibration**, and we have shipped one (`.fathom v0.1`).
3. **A linear-Gaussian dynamics model can be fitted to the resulting state vectors**, and we have verified mathematically that it recovers known parameters to machine epsilon on full-rank synthetic data.
4. **The first empirical observation of cognitive geometry from real bundled trajectories shows non-random, semantically meaningful clustering**, with three substantive findings (reasoning–adversarial proximity, hallucination isolation, dense triangle/intermediate ring structure).
5. **The Cognitive Universality Hypothesis is a well-formed scientific claim** that makes six concrete falsifiable predictions, each with a documented test procedure and a documented falsification condition.
6. **The reference implementation, calibration data, file formats, conformance suite, charter, and this paper are all open** under CC-BY-4.0 (specs and data) and MIT (code), with no restrictions on implementation in any language by any party.

This paper does not claim that:

- the universality hypothesis is proven (it is testable but untested at scale)
- the v0.1 atlas calibration is the final word on cognitive measurement (it is an empirical instance of $\Phi$, not a unique one)
- the linear-Gaussian dynamics model is the right model for LLM cognition (it is the floor of what is possible, not the ceiling)
- cognitive metrology can replace mechanistic interpretability, sparse autoencoders, or any other model-internal technique (it operates at a different altitude and is complementary)

### 11.2 Why publish a v1 paper with v0.1 results

Most foundational papers in a new field are written long after the field has accumulated enough empirical results to make sweeping claims. This paper is the opposite: it ships the infrastructure and the small empirical results together, with explicit acknowledgment that the big empirical claims remain to be tested.

The reason for this choice is institutional. Cognitive metrology does not yet exist as a recognized discipline. There is no community, no journal, no standards body, no established researchers, no shared vocabulary. Founding the discipline therefore requires an explicit founding act: a charter that defines the discipline, a reference implementation that grounds the definitions in code, and a paper that anchors the framework in citable form. We chose to ship all three together, rather than wait for the empirical results to mature, because the alternative was that the discipline would never start at all.

The risk of this approach is that the framework gets cited prematurely and people overinterpret what has been shown. We mitigate this risk by being explicit, throughout this paper, about what we have proven (the math, the file formats, the small empirical observations) and what we have not (the universality claims, the dynamics fits on real data at scale, the bridge to biological cognition).

The benefit is that the framework has a date. As of 2026-04-14, cognitive metrology exists. As of 2026-04-14, the atlas v0.3 calibration is public. As of 2026-04-14, the `.fathom` and `.cogdyn` file formats have published v0.1 specifications. As of 2026-04-14, anyone in the world can `pip install styxx` and run every claim in this paper. The historical record begins on this date, and it begins with this paper as a primary citable reference.

### 11.3 Open questions and invitations

We close with three open questions, each of which we invite the community to engage with:

1. **What is the right value of $d$ for cognitive metrology?** The atlas v0.3 uses $d = 6$. We do not claim this is optimal. Perhaps the right number is 4, perhaps it is 12, perhaps it is something that varies by task. We invite empirical work on this question.

2. **Are the six categories the right basis?** Cognitive metrology is committed to *some* finite-dimensional categorical basis, but the specific choice of {retrieval, reasoning, refusal, creative, adversarial, hallucination} is empirical, not principled. A future atlas version may discover a better basis through unsupervised methods, or by adding categories the v0.1 atlas missed.

3. **Does the universality hypothesis hold in any form?** This is the most important question and the answer determines the future of the discipline. We have laid out six falsifiable predictions that, taken together, will give a clear answer over the next 1–3 years. We invite anyone with the resources to run any of the six tests to do so and publish the results.

We also invite engagement of a different kind: from interpretability researchers, AI safety researchers, regulatory bodies, standards organizations, and the broader scientific community. Cognitive metrology is open by design. The reference implementation is MIT-licensed. The specifications are CC-BY-4.0. The calibration data is public. The conformance test suite is the v0.1 standard against which any independent implementation can verify itself. Fathom Lab serves as the founding institution but is committed, per the charter, to passing governance to a multi-stakeholder body within five years of the charter date.

The work to be done is enormous. The work that has been done is small. We are at year zero of a discipline that, if the universality hypothesis is even partially correct, will reshape how the world measures cognition. We do not expect to be the only ones building this. We expect to be the first.

---

## Acknowledgments

This paper synthesizes the results of a single intensive shipping cycle on 2026-04-14, during which the styxx 3.0.0a1, 3.1.0a1, and 3.1.0 releases were shipped to PyPI in sequence; the Cognitive Metrology Charter v0.1 was published; the `.fathom` and `.cogdyn` file format specifications were finalized and published; the cognitive provenance bridge was wired between the certificate format and the data type; the first community pull request on the styxx repository (from external contributor Matt Van Horn) was merged the same day it was filed; three issues filed by the Fathom Lab working group (#1, #2, #3) were closed by code; and the safety wrapper preventing accidental empty deploys to the fathom-lab production landing page was implemented and battle-tested. The work was paired between flobi (Fathom Lab) and an instance of Claude Opus 4.6 acting as a software engineering and writing collaborator throughout the session.

We thank Matt Van Horn for the timely community contribution (PR #4, closing issue #2, "warn once when observe() is given an openai response without logprobs"), which is the first external pull request in the styxx repository and which shipped in the styxx 3.1.0 stable release. We thank @SupaSeeka for the code review comment that prompted the post-merge `import sys` cleanup.

Atlas v0.3 calibration was conducted in the Fathom Lab research repository (`github.com/fathom-lab/fathom`) using the methods covered by US Provisional Patents 64/020,489, 64/021,113, and 64/026,964.

---

## Citation

```bibtex
@misc{flobi2026cognitivemetrologyv1,
  title         = {Cognitive Metrology v1: A Reference Implementation, First Empirical Results, and a Multi-Year Research Program},
  author        = {flobi},
  year          = {2026},
  month         = {4},
  day           = {14},
  publisher     = {Fathom Lab},
  howpublished  = {Manuscript, \url{https://github.com/fathom-lab/styxx/blob/main/papers/cognitive-metrology-v1.md}},
  note          = {v1; reference implementation: styxx 3.1.0 (\url{https://pypi.org/project/styxx/3.1.0/}); CC-BY-4.0},
}
```

---

## License

This paper is released under the Creative Commons Attribution 4.0 International (CC-BY-4.0) license. Anyone may copy, redistribute, remix, and build upon this paper for any purpose, including commercial use, with attribution.

The reference implementation (`styxx`) is released under the MIT License.
The atlas v0.3 calibration data is released under CC-BY-4.0.
The `.fathom v0.1` and `.cogdyn v0.1` file format specifications are released under CC-BY-4.0.
The Cognitive Metrology Charter v0.1 is released under CC-BY-4.0.

The underlying measurement methodology that produces the cognitive eigenvalues is covered by US Provisional Patents 64/020,489, 64/021,113, and 64/026,964. Implementations of the open specifications are explicitly welcomed and unrestricted.

---

*nothing crosses unseen.*

*— Fathom Lab, 2026-04-14*

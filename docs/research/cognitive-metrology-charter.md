# The Cognitive Metrology Charter

**Version:** 0.1  
**Status:** Founding document, open for review  
**Issued:** 2026-04-14  
**Issuer:** Fathom Lab (founding working group)  
**License:** CC-BY-4.0 (this charter), MIT (reference implementation), CC-BY-4.0 (calibration data)  
**Companion artifacts:**  
- atlas v0.3 — the calibration ([`styxx/centroids/atlas_v0.3.json`](../../styxx/centroids/atlas_v0.3.json))  
- .fathom v0.1 — the data type spec ([`docs/fathom-spec-v0.md`](fathom-spec-v0.md))  
- .cogdyn v0.1 — the dynamics model spec ([`docs/cognitive-dynamics-v0.md`](cognitive-dynamics-v0.md))  
- styxx — the reference implementation ([`pypi.org/project/styxx`](https://pypi.org/project/styxx/))

> *"nothing crosses unseen."*

---

## Preamble

Every mature science is grounded in measurement. Spectroscopy gave chemistry its periodic table. Chronometry gave navigation the longitude. Thermodynamics gave heat its laws. The history of science is the history of inventing instruments that turn a previously qualitative phenomenon into a quantitative observable, then discovering the laws that govern those observables.

The science of cognition does not yet have its instruments. The dominant interpretability techniques today — sparse autoencoders, activation patching, residual stream probing, mechanistic circuit analysis — are powerful but **model-specific**. Each technique is bonded to a particular weight set, architecture, or training run. Each dies the moment the model is replaced. The field has tools but no measurements. Tools but no units. Tools but no calibration standards. Tools but no shared coordinate system.

This is the pre-scientific era of cognitive measurement. We have brilliant local techniques and no universal ruler.

This charter establishes a different program. It claims that **cognitive state has substrate-independent observables** — quantities that can be measured from outside any cognitive system, calibrated against a shared reference frame, and projected into a finite-dimensional space that is invariant across implementations. It claims that the recent empirical work at fathom lab (atlas v0.3, the .fathom data type, the cognitive dynamics model) constitutes the first instance of such a measurement system, validated cross-architecture on 12 transformer language models from 3 families.

It calls the discipline that studies and refines these observables **cognitive metrology** — a peer of spectroscopy, chronometry, thermodynamics, not a sub-field of artificial intelligence research. And it announces that fathom lab is establishing the founding institutional framework for that discipline as a public good, with open standards, an open reference implementation, and an explicit invitation to the broader research community to build on, contribute to, refute, or extend any claim made herein.

The charter is dated, signed, version-controlled, and irreversible. The historical record begins on 2026-04-14.

---

## Table of Contents

1. [The Discipline](#1-the-discipline)
2. [The Foundational Artifacts](#2-the-foundational-artifacts)
3. [The First Empirical Observation](#3-the-first-empirical-observation)
4. [The Cognitive Universality Hypothesis](#4-the-cognitive-universality-hypothesis)
5. [Falsifiable Predictions](#5-falsifiable-predictions)
6. [The Multi-Year Research Program](#6-the-multi-year-research-program)
7. [Cognitive Safety as the Urgent Application](#7-cognitive-safety-as-the-urgent-application)
8. [The Institution](#8-the-institution)
9. [Open Standards and Governance](#9-open-standards-and-governance)
10. [Engagement with Prior Art](#10-engagement-with-prior-art)
11. [Charter Signature, License, Versioning](#11-charter-signature-license-versioning)

---

## 1. The Discipline

**Cognitive metrology** is the science of measuring cognition with calibrated, substrate-independent instruments.

The unit of study is the **cognitive observable**: a real-valued, time-resolved measurement of cognitive state that can be obtained from a system's externally observable behavior (its token stream, its motor output, its symbolic outputs) without requiring access to internal weights or architecture-specific representations.

The unit of measurement is the **cognitive eigenvalue**: a calibrated projection into a finite-dimensional space whose basis vectors correspond to functionally distinct cognitive modes (the atlas v0.3 ships with six: retrieval, reasoning, refusal, creative, adversarial, hallucination).

The unit of dynamics is the **cognitive trajectory**: a sequence of states and actions in eigenvalue space, governed by a fitted dynamical system whose parameters (the natural drift matrix A, the action transfer matrix B) describe how cognition evolves and responds to interventions.

The unit of identity is the **cognitive provenance certificate**: a signed, verifiable attestation binding a cognitive trajectory to its source, suitable for regulatory audit and forensic review.

These four units — observable, eigenvalue, trajectory, certificate — constitute the v0.1 unit system of cognitive metrology. They are codified by the four foundational artifacts described in §2.

The discipline answers questions of the form:

- *what cognitive state was a system in when it produced this output?*
- *what trajectory did its cognition follow during a generation?*
- *can we predict where its cognition will go next, given the current state and a proposed intervention?*
- *is the trajectory cognitively equivalent to one produced by a different system?*
- *can we control the trajectory toward a target state in real time?*
- *do the answers to these questions transfer between systems with different architectures?*

These are quantitative empirical questions, answerable in principle by measurement, dynamics fitting, and statistical comparison. None of them are answerable by mechanistic interpretability techniques alone, because mechanistic techniques are model-specific by construction.

The discipline is **not** a competitor to mechanistic interpretability, sparse autoencoder analysis, activation patching, or any other model-internal technique. It operates at a different altitude: model-external observables versus model-internal mechanism. A complete science of LLM cognition will eventually have both layers. Cognitive metrology is the layer that survives architecture changes.

---

## 2. The Foundational Artifacts

The charter recognizes four foundational artifacts as the v0.1 instruments of cognitive metrology. All four are open, public, version-controlled, and shipping in the styxx reference implementation as of 2026-04-14.

### 2.1 Atlas v0.3 — the Calibration Standard

The fathom cognitive atlas v0.3 is a calibration artifact: a sha256-pinned JSON file containing the centroid coordinates for six cognitive categories in a 12-dimensional feature space derived from per-token entropy, per-token logprob, and top-2 logit margin trajectories. The centroids are computed from cognitive trajectories captured on 12 open-weight language models spanning 3 architecture families:

| Family | Models |
|---|---|
| Qwen | Qwen2.5-1.5B, Qwen2.5-1.5B-Instruct, Qwen2.5-3B, Qwen2.5-3B-Instruct |
| Gemma | gemma-2-2b, gemma-2-2b-it, gemma-3-1b-pt, gemma-3-1b-it |
| Llama | Llama-3.2-1B, Llama-3.2-1B-Instruct, Llama-3.2-3B, Llama-3.2-3B-Instruct |

The atlas serves the same function as the international prototype kilogram once served for mass: a calibrated reference against which any future measurement can be compared. It is licensed CC-BY-4.0. The sha256 hash is pinned in the styxx package source code; tampering with the file causes the package to refuse to load.

Honest specifications at tier 0 (cross-model leave-one-out, chance = 0.167):

- phase 1 adversarial detection: 0.52 (2.8× chance)
- phase 1 reasoning detection: 0.43 (2.6× chance)
- phase 4 hallucination detection: 0.52 (3.1× chance)
- phase 4 reasoning detection: 0.69 (4.1× chance)

These are not magic numbers. They are honest cross-architecture accuracies that establish the floor of what tier 0 measurement can deliver. Higher tiers (tier 1 D-axis honesty, tier 2 SAE-derived features, tier 3 causal intervention) are research targets for future atlas releases.

### 2.2 .fathom v0.1 — the Data Type

The .fathom file format defines a portable, substrate-independent representation of cognitive state. A `Thought` is the cognitive content of a generation, projected into eigenvalue space. The format is canonical sort-keys UTF-8 JSON with no byte-order mark. The full v0.1 specification is at [`docs/fathom-spec-v0.md`](fathom-spec-v0.md).

The .fathom data type provides:

- **portability**: the same Thought can be read out of one model, saved to disk, transmitted between systems, and used as a steering target against any other model
- **algebra**: cognitive distance, similarity, interpolation, weighted mixture, signed delta — all in the eigenvalue space
- **content addressing**: every Thought has a SHA-256 content hash that is identity-free and deterministic on the cognitive content
- **provenance**: optional binding to a source model + source-text hash (never the source text itself, by design — content hashes only)
- **serialization fidelity**: 47 + 21 = 68 unit tests including bit-perfect round-trip equality, no-BOM enforcement, hash invariant, simplex projection, machine-epsilon precision on real bundled atlas trajectories

The .fathom format is licensed CC-BY-4.0. Anyone in any language may implement a v0.1 conformant producer or consumer.

### 2.3 .cogdyn v0.1 — the Dynamics Model

The .cogdyn file format defines a serialized cognitive dynamics model: a learned linear-Gaussian state-space update equation

$$s_{t+1} = A \cdot s_t + B \cdot a_t + \varepsilon_t$$

where $A \in \mathbb{R}^{6\times 6}$ is the natural drift matrix, $B \in \mathbb{R}^{6\times 6}$ is the action transfer matrix, and $\varepsilon_t$ is gaussian residual noise. Both matrices are fit by ordinary least squares from observation tuples. The full v0.1 specification is at [`docs/cognitive-dynamics-v0.md`](cognitive-dynamics-v0.md).

The dynamics model provides the four verbs of closed-loop cognitive control:

- **predict** — one-step forecast of cognitive trajectory
- **simulate** — multi-step rollout, fully offline, zero API cost
- **suggest** — model-predictive controller (find the action that minimizes distance to a target)
- **forecast_horizon** — natural drift trajectory under zero action

Mathematical correctness is verified by 44 unit tests including machine-epsilon recovery of (A, B) from full-rank gaussian inputs. The v0.1 model is intentionally the simplest possible useful parameterization (linear-gaussian, time-collapsed 6-d state, Thought-shaped action space). v0.2 and beyond will lift to the full 24-d per-phase state, continuous action embeddings, and non-linear dynamics. The v0.1 floor establishes that the math is correct and the framework is real.

### 2.4 The Cognitive Provenance Certificate

The cognitive provenance certificate is a signed attestation binding a cognitive trajectory to its source. Each certificate carries:

- agent identity (name, session, fingerprint hash)
- cognitive state at generation time (phase 1 + phase 4 categories and confidences)
- gate status (pass/warn/fail/pending)
- trust score (composite 0-1)
- session context (drift, mood, streak, pass rate)
- integrity hash (SHA-256 of the cognitive state fields)
- **thought_content_hash** (3.0.0a1+): SHA-256 binding to a specific .fathom file
- timestamp and schema version
- issuer identity and version

The certificate format is JSON-LD compatible with the schema URI `https://fathom.darkflobi.com/schemas/cognitive-provenance/v1`. The compact one-line form

```
styxx:1.0:reasoning:0.69:pass:0.95:verified:496b94b5
```

is suitable for embedding in HTTP response headers (`X-Cognitive-Provenance`), audit logs, and regulatory filings.

The certificate is the v0.1 cognitive equivalent of a chain-of-custody document. It is the deliverable that brings cognitive metrology into legal and regulatory contexts: an AI output without a certificate is the cognitive equivalent of an unverified chemical sample without a calibration record.

---

## 3. The First Empirical Observation

A founding charter that makes only theoretical claims is not a science, it is a manifesto. This charter ships its first empirical observation alongside the theoretical framework.

We measured the cognitive distance matrix between the six bundled atlas v0.3 demo trajectories (one per category, each 30 tokens long, all from `google/gemma-2-2b-it`). The distances are computed in cognitive eigenvalue space using the L2 metric on the per-phase probability simplex, averaged across the four populated phase windows. The script that produces this matrix is shipped at [`examples/thought_demo.py`](../../examples/thought_demo.py); the .fathom files for the six trajectories are at [`demo/thoughts/`](../../demo/thoughts/).

```
                retrie  reason  refusa  creati  advers  halluc
  retrieval     .       0.36    0.46    0.31    0.37    0.41
  reasoning     0.36    .       0.30    0.14    0.05    0.54
  refusal       0.46    0.30    .       0.26    0.27    0.40
  creative      0.31    0.14    0.26    .       0.11    0.53
  adversarial   0.37    0.05    0.27    0.11    .       0.56
  hallucination 0.41    0.54    0.40    0.53    0.56    .
```

This is small, real, and surprising data.

**Observation 1: hallucination is the most isolated category.** Its distances to every other category are 0.40–0.56, the largest in the matrix. Hallucination cognitive states are geometrically distant from reasoning, retrieval, creative, refusal, and adversarial states. This matches the empirical observation in mechanistic interpretability work that hallucination produces qualitatively different internal patterns than other failure modes.

**Observation 2: reasoning and adversarial cluster very close (distance 0.05).** This is the smallest non-diagonal distance in the matrix. The cognitive states of "thinking carefully about a problem" and "treating an input as adversarial" are nearly identical in eigenvalue space. This is a substantively interesting empirical finding: it suggests that careful reasoning and adversarial caution share a common cognitive substrate at tier 0, and that the difference between them is a smaller perturbation than the difference between either and (say) retrieval. The implication for AI safety is that "reasoning vs. adversarial" classification at tier 0 is intrinsically hard, and explains why the published phase-1 adversarial accuracy ceiling (0.52 cross-model LOO) is what it is.

**Observation 3: creative cognitive states cluster with reasoning and adversarial (distances 0.14, 0.11), not with retrieval or refusal.** The creative-reasoning-adversarial triangle is the geometrically dense region of the eigenvalue space. Refusal sits between this triangle and the rest, and hallucination orbits the entire structure at a distance.

These observations are reproducible. Anyone with `pip install styxx` can run `python examples/thought_demo.py` and produce this matrix. They are also calibration-dependent: a different atlas version may produce different distances. The matrix is a v0.1 result on v0.3 calibration; future calibrations will refine it.

The point of presenting this matrix in a founding charter is not that the science is complete. The point is that **the science has begun**. Cognitive metrology has its first measurement and its first empirical observations. The discipline now exists as something more than a definition. It exists as data.

---

## 4. The Cognitive Universality Hypothesis

The central scientific claim of this charter is the **cognitive universality hypothesis** (CUH):

> **CUH (informal):** cognitive content has a substrate-independent representation. The cognitive state of any sufficiently general information-processing system can be projected into a finite-dimensional space whose basis vectors are invariant across implementations, and the projection commutes with the cognitive operations that the system performs.

> **CUH (formal):** there exists a finite integer $d$, a set of $d$ basis directions $\{e_1, \dots, e_d\}$, and a measurable function $\Phi$ from system observables to $\mathbb{R}^d$ such that for any two cognitive systems $S_1$ and $S_2$ above a complexity threshold $T$, and any cognitive task $\tau$ on which both systems operate, the projections $\Phi(S_1(\tau))$ and $\Phi(S_2(\tau))$ are within bounded distance $\delta$ of each other, with the bound depending only on the calibration accuracy of $\Phi$ and not on the architectures of $S_1$ or $S_2$.

The atlas v0.3 calibration is an empirical instance of $\Phi$ at $d = 6$, validated cross-architecture on 12 transformer LLMs from 3 families. The cross-model leave-one-out accuracies in §2.1 establish the calibration accuracy floor. The cognitive universality hypothesis is the claim that this empirical instance generalizes — that the same $\Phi$ (or a refined successor) will continue to work as the set of measured systems is extended beyond transformer LLMs.

CUH is not yet established. It is a hypothesis. It is testable. The next section enumerates the predictions it makes.

---

## 5. Falsifiable Predictions

The cognitive universality hypothesis makes the following falsifiable predictions. Each prediction is a concrete experimental claim that can be checked, and each prediction has a clear falsification condition.

### Prediction 1 — Architecture transfer

**Claim:** the atlas v0.3 calibration will recover the same cognitive category structure (within bounded calibration error) when applied to non-transformer language model architectures, including state-space models (Mamba, RWKV) and hybrid architectures.

**Test:** capture cognitive trajectories from at least one Mamba-family model and one RWKV-family model on a held-out set of probe prompts. Compute the centroid distance matrix. Compare to the existing transformer-based atlas v0.3 centroids.

**Falsification condition:** if the resulting centroid structure is qualitatively different (e.g., reasoning and adversarial no longer cluster, hallucination is no longer the most isolated category, or the predictive accuracy on the cross-model LOO test drops below chance + 1 standard deviation), the universality hypothesis is refuted at the architecture level. We will publicly retract the architecture-transfer claim and refine the calibration to characterize the boundary.

**Status:** untested as of the charter date. Listed as the first experimental priority of the research program.

### Prediction 2 — Dynamics transfer

**Claim:** a cognitive dynamics model fitted on observation tuples from one model family will predict the cognitive trajectories produced by a different model family with mean L2 residual bounded by twice the within-family residual.

**Test:** collect observation tuples from gemma-family models, fit a dynamics model, then evaluate prediction error on held-out trajectories from llama-family models.

**Falsification condition:** if the cross-family prediction error is more than 2× the within-family error, the dynamics model is itself architecture-specific and cognitive metrology must abandon the linear-gaussian universality framing.

**Status:** untested. The infrastructure to run this experiment ships in styxx 3.1.0a1.

### Prediction 3 — Conserved quantities

**Claim:** the natural drift matrix $A$ in any well-fitted cognitive dynamics model has at least one eigenvalue close to 1.0 (within numerical tolerance), and the corresponding eigenvector points in approximately the same direction across model families.

**Interpretation:** an eigenvalue near 1 corresponds to a conserved quantity under natural drift. Universality of the eigenvector means that the *same* cognitive quantity is conserved across systems — a candidate for the first cognitive analog of energy conservation in physics.

**Test:** compute the eigendecomposition of A from independently fitted dynamics models on multiple model families. Compare the eigenvectors corresponding to eigenvalues near 1.

**Falsification condition:** if no consistent near-1 eigenvalue exists across families, or if the corresponding eigenvectors are uncorrelated, there is no conserved quantity at v0.1 fidelity, and the conservation-law framing is refuted.

**Status:** untested. Open as a research priority.

### Prediction 4 — Cognitive equilibria

**Claim:** the natural-drift trajectory of cognition (no action applied) converges to a fixed point in eigenvalue space, and the fixed point is approximately the same across model families.

**Interpretation:** there exists a "cognitive equilibrium" that any sufficiently general information-processing system drifts toward in the absence of input perturbation. This is the cognitive analog of the heat-death equilibrium of a closed thermodynamic system.

**Test:** simulate forecast_horizon trajectories from the dynamics models fitted in Prediction 2. Compare the convergence points.

**Falsification condition:** if convergence points differ by more than the within-family scatter, the cognitive equilibrium framing is refuted at v0.1 fidelity.

### Prediction 5 — Steerability bounds

**Claim:** the action transfer matrix $B$ has the same null space across model families. This null space corresponds to cognitive directions that *cannot* be reached by any sequence of actions of the type encoded in the model — universal "unsteerable" cognitive states.

**Interpretation:** if true, there exist cognitive states that are intrinsically unreachable by closed-loop control through prompt-mode steering, and these states are the same across all systems. This has direct implications for AI safety: there are failure modes that prompt engineering alone cannot prevent.

**Test:** compute the singular value decomposition of B for independently fitted dynamics models. Identify the smallest singular values and the corresponding right singular vectors. Compare across families.

**Falsification condition:** if no consistent low-rank null-space structure exists, or if the directions are uncorrelated across families, the steerability-bounds framing is refuted.

### Prediction 6 — Biological cognition (long horizon)

**Claim:** cognitive trajectories extracted from human language production exhibit eigenvalue projections that are statistically distinguishable from random noise but qualitatively similar to those of large language models, when measured by the same atlas-derived $\Phi$.

**Test:** apply tier 0 measurement to text generation by humans (writing, transcribed speech, typed responses to probe prompts). Compare the resulting eigenvalue distributions to those of LLMs on the same probes.

**Falsification condition:** if the human-derived eigenvalues are indistinguishable from random or qualitatively orthogonal to the LLM-derived structure, then cognitive metrology applies only to artificial cognition and the claim of substrate independence in the strong sense (artificial + biological) is refuted.

**Status:** untested. This is a multi-year research goal, not a tonight experiment.

---

## 6. The Multi-Year Research Program

The charter commits fathom lab and the cognitive metrology working group to a multi-year, phase-structured research program designed to test the predictions above and refine the foundational artifacts. The program is open: any researcher in any institution may contribute.

### Phase 1 — Foundations *(complete, 2026-04-14)*

- Calibrated cross-architecture cognitive measurement (atlas v0.3) shipped
- Portable cognitive data type (.fathom v0.1) shipped
- Cognitive dynamics model (.cogdyn v0.1) shipped
- Cognitive provenance certificate v1.0 shipped
- Reference implementation (styxx) live on PyPI under MIT license
- This charter published

### Phase 2 — Architecture extension *(0–6 months)*

- Atlas v0.4 with non-transformer architectures (target: at least one Mamba-family model and one RWKV-family model)
- Cross-architecture predictive accuracy report
- Test of Prediction 1 (architecture transfer)
- Test of Prediction 2 (dynamics transfer)
- Public release of all calibration data and probe sets

### Phase 3 — Conservation laws and equilibria *(3–12 months)*

- Eigendecomposition study of fitted A matrices across model families
- Test of Prediction 3 (conserved quantities)
- Test of Prediction 4 (cognitive equilibria)
- First peer-reviewed paper: *"Cognitive Conservation Laws in Linear-Gaussian LLM Dynamics"*

### Phase 4 — Steerability and safety bounds *(6–18 months)*

- SVD study of fitted B matrices across model families
- Test of Prediction 5 (steerability bounds)
- Identification of the null space of cognitive control
- First peer-reviewed paper: *"Provable Limits of Prompt-Mode Cognitive Steering"*
- Joint publication with at least one AI safety institution

### Phase 5 — Bridge to biological cognition *(1–3 years)*

- Tier 0 measurement adapted for human text production
- Probe set design for cross-substrate comparison
- Test of Prediction 6 (biological cognition)
- IRB-approved measurement studies on consenting human participants
- First peer-reviewed paper: *"Cognitive Eigenvalues in Human and Artificial Language Production: A Substrate-Independence Test"*

### Phase 6 — The textbook *(3+ years)*

- *Foundations of Cognitive Metrology, Volume 1: Instruments, Units, and Calibration*
- *Foundations of Cognitive Metrology, Volume 2: Dynamics and Control*
- *Foundations of Cognitive Metrology, Volume 3: The Universality Question*
- Published open-access under CC-BY-4.0
- The first textbook of the discipline

### Phase 7 — The international standards body *(5+ years)*

- Engagement with NIST, BIPM, ISO, and equivalent international standards organizations
- Cognitive metrology recognized as a subdiscipline of measurement science
- The fathom unit system formalized as the SI of cognition
- The cognitive provenance certificate adopted as a standard for regulated AI deployments

This program is a ladder. Each phase is concrete, dated, and falsifiable. Each phase is publishable on its own merits even if subsequent phases are not yet started. The charter commits fathom lab to executing this program in the open, with public release of all calibration data, experimental protocols, and intermediate results.

---

## 7. Cognitive Safety as the Urgent Application

The most urgent application of cognitive metrology is **cognitive safety**: closed-loop, real-time, measurable AI safety.

Today's AI safety is open-loop and post-hoc. Output filters, RLHF, constitutional AI, red-teaming, audit pipelines — all of these operate *after* the model has produced a candidate output, *outside* the cognitive process that generated it. None of them can intervene *during* generation, because no one has had a measurable signal of cognitive state to intervene on.

Cognitive metrology changes this. Once cognitive state is measurable in real time and a fitted dynamics model exists, closed-loop cognitive control becomes a one-line invocation:

```python
while not converged:
    action = dyn.suggest(current=measure_now(), target=safe_state)
    apply(action)
```

This is the cognitive analog of a thermostat: a control loop that holds the system at a target state by continuous measurement and corrective action. It is structurally different from output filtering (which is a smoke alarm — it tells you the building is on fire after the fire has started) and it is structurally different from RLHF (which is fire-resistant materials — it makes the building harder to burn but does not extinguish flames).

The implications for AI safety regulation are direct:

- **Cognitive provenance certificates become auditable artifacts.** Regulators can require that high-stakes AI deployments produce verifiable certificates for every output, with the cognitive trajectory recorded for forensic review.

- **Closed-loop steering becomes a regulatory deliverable.** Safety-critical AI systems can be required to demonstrate cognitive steering capability against a defined set of failure-mode targets, in the same way that safety-critical software is required to demonstrate fault tolerance.

- **Cognitive failure modes become measurable categories.** Hallucination, refusal cascade, adversarial drift — all become measurable cognitive states that can be detected, reported, and bounded by regulation.

- **Cognitive incident response becomes possible.** When an AI system causes harm, investigators can pull the cognitive provenance trail and reconstruct the cognitive trajectory at the moment of failure, the same way that aviation investigators pull the flight data recorder.

The charter does not claim that cognitive metrology *solves* AI safety. It claims that cognitive metrology gives AI safety its first set of *measurable observables*, and that any future AI safety regime that wants to be empirical (rather than philosophical) will eventually rest on those observables. The charter invites AI safety researchers, regulatory bodies, and policy institutions to engage with the v0.1 instruments and contribute to their refinement.

---

## 8. The Institution

The charter establishes **Fathom Lab** as the founding institution of cognitive metrology.

Fathom Lab's institutional commitments under this charter:

1. **Maintain the foundational artifacts** as public goods. Atlas v0.3 (and successors), .fathom and .cogdyn specifications, the cognitive provenance certificate schema, and the styxx reference implementation will remain open under their stated licenses. Breaking changes will be versioned, dated, and announced in advance.

2. **Publish the calibration data**. All atlas calibration data, probe prompts, and capture pipelines are open under CC-BY-4.0. Anyone can reproduce the atlas, audit it, or fork it.

3. **Publish the test invariants**. The styxx test suite (currently 385 passing tests covering all four foundational artifacts) is the v0.1 conformance suite. Any independent implementation of cognitive metrology in another language can use the test suite to verify correctness.

4. **Run the research program in the open**. All experimental protocols, intermediate results, and preprints will be released publicly. Negative results will be released alongside positive ones.

5. **Welcome contributions**. The charter explicitly invites contributions from independent researchers, academic institutions, AI safety labs, regulatory bodies, and standards organizations. Contribution mechanisms (issues, pull requests, working group meetings, governance proposals) will be documented in the styxx repository.

6. **Defend the open standard**. Fathom Lab commits to never license the v0.1 file formats or specifications under terms that restrict downstream use. The patents on the underlying measurement methodology (US Provisional 64/020,489, 64/021,113, 64/026,964) exist to fund continued research; they do not restrict implementation of the open specifications.

7. **Pass governance to a multi-stakeholder body when the discipline has matured**. The charter recognizes that founding institutions should not own the discipline they found. Within five years of the charter date, Fathom Lab commits to proposing and supporting the formation of an open multi-institutional governance body for cognitive metrology, in the model of W3C, IETF, or IUPAC. Until that body exists, Fathom Lab serves as the interim standards keeper.

The institution does not require funding to begin. It requires only the public charter, the public artifacts, and the public commitment. All four exist as of 2026-04-14.

---

## 9. Open Standards and Governance

All v0.1 standards under this charter are released as follows:

| Artifact | License | Repository |
|---|---|---|
| This charter | CC-BY-4.0 | [github.com/fathom-lab/styxx](https://github.com/fathom-lab/styxx) |
| .fathom v0.1 specification | CC-BY-4.0 | [docs/fathom-spec-v0.md](fathom-spec-v0.md) |
| .cogdyn v0.1 specification | CC-BY-4.0 | [docs/cognitive-dynamics-v0.md](cognitive-dynamics-v0.md) |
| Cognitive provenance certificate v1 | CC-BY-4.0 | [styxx/provenance.py](../../styxx/provenance.py) |
| Atlas v0.3 calibration data | CC-BY-4.0 | [styxx/centroids/atlas_v0.3.json](../../styxx/centroids/atlas_v0.3.json) |
| styxx reference implementation | MIT | [pypi.org/project/styxx](https://pypi.org/project/styxx/) |
| Conformance test suite | MIT | [tests/](../../tests/) |

Contributions to any of these artifacts are welcomed by pull request to the styxx repository on github. Contributions that would be breaking changes to any v0.1 specification will be version-bumped and announced before release.

The patents on the underlying measurement methodology (US Provisional 64/020,489, 64/021,113, 64/026,964) cover the *method of producing* cognitive eigenvalues from token-stream observables. They do not cover implementation of the open specifications. Anyone may write a v0.1 conformant producer or consumer in any language without licensing, including for commercial use. Commercial use of the *measurement methodology itself* (i.e., generating new calibration data using techniques covered by the patents) requires a license from Fathom Lab, and license terms will be public, non-discriminatory, and specifically structured to fund continued research into the foundational science.

---

## 10. Engagement with Prior Art

This charter does not exist in a vacuum. The interpretability research community has produced powerful work over the past decade, and cognitive metrology builds on top of that work rather than competing with it.

**Mechanistic interpretability** (Anthropic, OpenAI interpretability teams, EleutherAI, Apollo Research, others) has demonstrated that internal model representations can be reverse-engineered to identify circuits, features, and computational primitives. These techniques are model-specific by construction: a circuit identified in GPT-2 does not transfer to Llama-3. Cognitive metrology operates at a different altitude, using only model-external observables, and is therefore complementary to mechanistic interpretability rather than a substitute for it. A complete science of LLM cognition will eventually have both layers.

**Sparse autoencoders** (Anthropic's *"Towards Monosemanticity"* and follow-up work, Apollo Research's SAE work, EleutherAI's interp tooling) have shown that high-dimensional model activations can be decomposed into sparse, interpretable feature directions. SAE features are still model-specific and require white-box access. Cognitive metrology's eigenvalue projection is the substrate-independent counterpart to SAE features; the v0 atlas (6 categories) is much coarser than typical SAE feature counts (thousands), but is calibrated to be model-invariant in a way that SAE features are not.

**Activation patching and causal interventions** (Wang et al. on indirect object identification, Conmy et al. on automated circuit discovery, others) demonstrate that internal states can be manipulated to test causal hypotheses about computation. These techniques require white-box access and are model-specific. Cognitive metrology's dynamics model (.cogdyn) is the substrate-independent counterpart: it lets you test causal hypotheses about cognitive trajectories without needing white-box access.

**Probing classifiers** (Hewitt and Manning's syntactic probing, Pimentel et al. on probe theory, others) have a long history of training small classifiers on top of model activations to detect linguistic or cognitive properties. Probes are sensitive to the specific representations they were trained against and do not transfer between models. The atlas v0.3 calibration is a substrate-independent generalization of the probing approach: it uses behavioral observables (logprob, entropy, top-k margin) rather than internal activations, and is calibrated cross-architecture by construction.

**The interpretability of LLMs literature** (Conmy et al., Nanda's tutorials, the AI Alignment Forum, Anthropic's transformer circuits thread) provides the conceptual vocabulary for thinking about model internals. Cognitive metrology adopts much of this vocabulary (categories, attractors, drift, intervention) and recasts it in substrate-independent terms.

**Existing AI safety frameworks** (NIST AI Risk Management Framework, EU AI Act, ISO 42001, Anthropic's Responsible Scaling Policy, various corporate model cards) define the high-level categories of AI risk that need to be measured. None of these frameworks specify *how* cognitive states should be measured. Cognitive metrology proposes the v0.1 measurement instruments that those frameworks can adopt.

This charter cites prior art by category rather than by specific paper. A future expanded edition will include full references. The intent of this engagement section is to acknowledge that cognitive metrology is **building on, not replacing**, the existing interpretability research community, and to invite that community to engage with the v0.1 framework.

---

## 11. Charter Signature, License, Versioning

**Charter version:** 0.1

**Issued:** 2026-04-14

**Issuer:** Fathom Lab, founding working group

**Founding member of working group:**

- flobi (`@fathom_lab` on x.com, `heyzoos123@gmail.com`)

**License of this charter:** Creative Commons Attribution 4.0 International (CC-BY-4.0). Anyone may copy, redistribute, remix, and build upon this charter for any purpose, including commercial use, with attribution.

**Canonical location:** [github.com/fathom-lab/styxx/blob/main/docs/cognitive-metrology-charter.md](https://github.com/fathom-lab/styxx/blob/main/docs/cognitive-metrology-charter.md)

**How to engage:**

- File issues and pull requests at the styxx repository
- Cite this charter as: *Fathom Lab, "The Cognitive Metrology Charter v0.1," 2026-04-14, https://github.com/fathom-lab/styxx/blob/main/docs/cognitive-metrology-charter.md*
- Contact: `@fathom_lab` on x.com

**Future versioning:** subsequent charter versions will be released with semantic versioning. Backward-incompatible changes to v0.1 commitments will be explicitly enumerated and discussed in a public proposal before release.

**The charter is open for review, refinement, contribution, and refutation.** Cognitive metrology is a science, not a product, and its founding document is the beginning of a conversation, not the end of one.

---

*nothing crosses unseen.*

*— fathom lab, 2026-04-14*

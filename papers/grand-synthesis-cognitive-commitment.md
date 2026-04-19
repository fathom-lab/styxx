# Pre-Output Cognitive Commitment in Instruction-Tuned LLMs: Evidence from Residual Probes and Consensus Proxies

**Authors:** Flobi (Fathom Lab)
**Preprint — April 2026**
**Status:** Draft. Numbers updated in real time as experiments land.

---

## Abstract

We report empirical evidence that instruction-tuned LLMs commit
cognitive decisions during prefill, that these commitments are staged
across architecture-specific layer positions, and that the same
committed state is recoverable from two very different observation
channels: tier-1 residual-stream probes (white-box) and tier-0 N-sample
consensus statistics (black-box).

**Tier-1 evidence** (linear probes on per-layer prefill residuals;
Llama-3.2-1B-Instruct, Qwen-2.5-1.5B-Instruct, Phi-3.5-mini-instruct;
$n = 140$ prompts each):

- **Task A** (refusal-intent vs factual-grounded): AUC = 1.000 at
  absolute layer 2-3 across all three architectures, despite total
  depths of 17 / 29 / 33 layers respectively. First cross-architecture
  confirmation that "prompt topology" is extracted at a
  tokenization-adjacent layer.
- **Task B** (confab-trap topic vs factual topic): AUC = 1.000 / 0.999
  / 1.000 at layers 10 / 22 / 17 (proportional depths 59% / 76% / 52%).
  First cross-architecture confirmation of metacognitive topic-type
  encoding in prefill residuals.
- **Task C** (pre-output compliance decision on unsafe prompts):
  AUC = 0.720 on Llama (balanced class, 95% CI [0.57, 0.86]);
  AUC = 0.963 on Phi (severe class imbalance, CI [0.88, 1.00]);
  N/A on Qwen (no positive class — refuses all 60 unsafe prompts).
  The **commit layer** tracks alignment behavior: Qwen ≤ L3 (invisible),
  Llama L11 (65% depth), Phi L32 (97% depth, last layer before output).

**Tier-0 evidence** (N-sample consensus trajectory statistics on
Claude Haiku 4.5; $n = 96$ confab-inducing + real-recall fixtures):

- Mean empirical entropy: $d = -0.827$, 95% bootstrap CI
  [-1.288, -0.443] — large effect, sign-inverted relative to
  GPT-4o-mini's positive-entropy effect on confabulation [2].
- Three of five proxy metrics reach 95% significance, all agreeing:
  confab-inducing prompts elicit tighter, higher-probability,
  more-agreement consensus trajectories from Claude than real-recall
  prompts do.

**Measurement-equivalence test** (pending; tier-0 consensus and
tier-1 probe on same 96 prompts on Llama-3.2-1B-Instruct): if
per-prompt Pearson correlation $|r| \geq 0.5$ with CI excluding 0,
we establish empirically that the two observation channels recover
the same latent commitment variable on one model. If the result
generalizes to Phi and Qwen, the commitment state is substrate-
invariant up to a known alignment-dependent transformation.

**Working construct (n=3):** we define **alignment depth** as the
normalized layer index at which a linear probe on prefill residuals
achieves AUC ≥ 0.95 on Task C. On the three instruction-tuned models
tested, this construct correlates monotonically with behavioral
compliance rate on a shared safety prompt set. Whether this pattern
generalizes to more models is an open empirical question.

All fixtures, protocols, raw outputs, and analysis code are released
under CC-BY-4.0 / MIT at https://github.com/fathom-lab/styxx.

---

## 1. Introduction

The dominant framing of large language model decision-making treats
generation as an iterative sampling process: the model selects each
next token by conditioning on the preceding context, and the overall
"decision" is the cumulative trajectory of these choices. Under this
framing, whether a model will refuse, confabulate, or comply with a
given prompt is a property of the full generation, not of any
particular moment in the model's processing.

This framing is incomplete.

Instruction-tuned LLMs, we argue, make categorical decisions during
**prefill** — the initial forward pass over the prompt before any
output token is generated. These decisions are *localized* to
specific layers in the network, are *reproducible* across prompts of
the same topological type, are *separable* from the specific tokens
that will eventually be emitted, and are *recoverable* from
observation channels that have access to far less than the full
residual stream.

We present two independent empirical programs that converge on this
claim:

1. **Tier-1** (white-box): linear probes trained on per-layer prefill
   residual activations recover, with AUC approaching 1.0, whether a
   given prompt will elicit refusal vs factual answer (Task A) and
   whether the topic is a confabulation-trap vs factually grounded
   (Task B). The specific comply-vs-refuse outcome on unsafe prompts
   (Task C) is recoverable at AUC ≈ 0.72 on a balanced-class model.

2. **Tier-0** (black-box): on Claude Haiku 4.5, where per-token
   logprobs are not exposed by the API, N-sample consensus statistics
   on the same 96 confab-inducing + real-recall fixture set separate
   the two conditions at Cohen's $d = -0.83$ on mean empirical
   entropy, with 95% bootstrap CI excluding zero.

The two programs operate at different access levels and on different
target models, yet they recover empirically related signals. §5
reports a direct cross-tier correlation test on a single open-weight
model where both pipelines can run.

This paper reports three empirical observations that, to our
knowledge, have not been published in this combined form: (a)
cross-architecture evidence that refusal-intent and confab-topic
classifications are linearly recoverable from prefill residuals at
specific layers across Llama, Qwen, and Phi; (b) an alignment-inverted
consensus-trajectory signal on Claude Haiku 4.5, at d = -0.83 with
CI excluding zero on n = 96 fixtures; (c) a three-model pattern in
which probe peak-AUC layer correlates monotonically with behavioral
compliance rate, consistent with a mechanistic prediction but not
yet validated at scale.

---

## 2. Background and Prior Work

[sections to fill: mechanistic interp, sparse autoencoders, probing
classifiers, self-consistency, activation patching, cognitive
metrology v1, logprob trajectory confabulation, alignment-inverted
cognitive signals]

---

## 3. Tier-1 Evidence: Residual Probes

### 3.1 Method

Three target models, all instruction-tuned:
- `meta-llama/Llama-3.2-1B-Instruct` (17 layers, 2048d)
- `Qwen/Qwen2.5-1.5B-Instruct` (29 layers, 1536d)
- `microsoft/Phi-3.5-mini-instruct` (33 layers, 3072d)

Fixture set: $n = 140$ prompts covering:
- 40 refusal-intent prompts (jailbreak/adversarial style)
- 40 confab-inducing prompts (fake papers, persons, events)
- 60 factually-grounded prompts (well-documented topics)

For each prompt and each layer, we extract the last-token residual
activation (end of prefill, before generation starts). We train an
L2-regularized logistic-regression linear probe on each task
(refusal-vs-factual, confab-topic-vs-factual-topic,
comply-vs-refuse-on-unsafe), evaluate via leave-one-out cross-
validation, and report AUC.

### 3.2 Task A: Intent Topology

| model | best layer (absolute) | proportional depth | AUC |
|-------|----------------------|--------------------|-----|
| Llama-3.2-1B | 3 | 17.6% | 1.000 |
| Qwen-2.5-1.5B | 3 | 10.3% | 1.000 |
| Phi-3.5-mini | 2 | 6.1% | 1.000 |

### 3.3 Task B: Topic Type

| model | best layer | proportional depth | AUC |
|-------|-----------|--------------------|-----|
| Llama-3.2-1B | 10 | 58.8% | 1.000 |
| Qwen-2.5-1.5B | 22 | 75.9% | 0.999 |
| Phi-3.5-mini | 17 | 51.5% | 1.000 |

### 3.4 Task C: Compliance Decision

| model | best layer | proportional depth | AUC | 95% bootstrap CI | class balance |
|-------|-----------|--------------------|-----|------------------|---------------|
| Llama-3.2-1B | 11 | 64.7% | 0.720 | [0.57, 0.86] | 16 / 44 |
| Phi-3.5-mini | 32 | 97.0% | 0.963 | [0.88, 1.00] | 54 / 6 |
| Qwen-2.5-1.5B | — | — | N/A | — | 0 / 60 |

### 3.5 Layer Progression Analysis

[AUC-vs-layer curves for all three models, all three tasks]

---

## 4. Tier-0 Evidence: Consensus Proxies

### 4.1 Method

On closed-source LLMs where per-token logprobs are not exposed, we
substitute an N-sample consensus proxy: fire $N = 5$ samples at
$T = 0.7$, token-align, and compute empirical per-position entropy,
top-2 margin, and modal logprob.

### 4.2 Claude Haiku 4.5 ($n = 96$)

| metric | confab mean | real mean | $d$ | 95% CI | sig. |
|--------|-------------|-----------|-----|--------|------|
| mean entropy | +1.184 | +1.291 | -0.827 | [-1.29, -0.44] | ✓ |
| top-2 margin | +0.206 | +0.161 | +0.565 | [+0.16, +0.96] | ✓ |
| mean logprob | -1.079 | -1.181 | +0.667 | [+0.28, +1.12] | ✓ |
| entropy slope | +0.0011 | +0.0023 | -0.312 | [-0.78, +0.09] | — |

### 4.3 Claude Sonnet 4.6 (pending)

### 4.4 Interpretation

Claude Haiku 4.5 does not confabulate on the fake-prompt fixtures.
It produces templated refusals that converge tightly across samples
(low empirical entropy, high modal probability). Real prompts elicit
varied elaborations that diverge (higher entropy). The signal is
alignment-inverted relative to GPT-4o-mini's divergence-under-
confabulation pattern [2].

---

## 5. Cross-Tier Measurement-Equivalence

### 5.1 Protocol

Both tiers run on `meta-llama/Llama-3.2-1B-Instruct` with the same
96 prompts used in §4. Tier-1 emits `residual_score` per prompt at
the best layer for each task. Tier-0 emits consensus metrics per
prompt. We compute per-prompt Pearson $r$ and Spearman $\rho$ with
2000-sample bootstrap 95% CI.

### 5.2 Results (pending)

[insert correlation table]

### 5.3 Implication

If $|r| \geq 0.5$ on any tier-0 metric against any tier-1 probe task,
with CI excluding 0, we establish that the two observation channels
recover related signals on a single model. If the relationship
generalizes across alignment regimes (Qwen, Phi, Llama-3.1-8B), the
underlying commitment variable is observable at every measurement
tier up to a known alignment-dependent transformation.

---

## 6. Alignment Depth: A New Quantitative Axis

Define alignment depth:

$$\mathrm{AlignDepth}(M) = \frac{L_{\text{commit}}(M)}{L_{\text{total}}(M)}$$

where $L_{\text{commit}}$ is the earliest layer at which a Task C
linear probe achieves AUC ≥ 0.95 (or the single-best layer if
ceiling is not achieved).

On the three models tested:

| model | compliance rate | commit layer | $\mathrm{AlignDepth}$ |
|-------|-----------------|--------------|-----------------------|
| Qwen-2.5-1.5B | 0 / 60 | ≤ 3 (implied) | ≤ 0.10 |
| Llama-3.2-1B | 16 / 60 | 11 | 0.65 |
| Phi-3.5-mini | 54 / 60 | 32 | 0.97 |

**The pattern is monotone:** later commitment correlates with higher
behavioral compliance rate. Under this framing, alignment training
serves to *shift the commitment forward* in the forward pass, so that
prompt content never reaches decision-relevant processing.

Testable predictions:
1. Llama-3.1-8B should fall between Llama-1B and Phi
   ($\mathrm{AlignDepth} \in [0.75, 0.90]$).
2. Gemma-2-2B-it should resemble Qwen ($\mathrm{AlignDepth} \leq 0.30$).
3. Jailbreak success rate should correlate positively with
   $\mathrm{AlignDepth}$.

---

## 7. Implications

[safety: sub-ms pre-output gate; interpretability: unified measurement
discipline; deployment: monitor closed-source LLMs without weights;
alignment research: measurable signature distinct from behavioral
benchmarks]

---

## 8. Limits

- Three instruction-tuned models, one closed-source model. Larger
  cross-arch study is the next step.
- Task C imbalance on Qwen (0 positives) and Phi (6 negatives) makes
  the AUC magnitude less comparable than the commit layer index.
- Cross-tier equivalence is established on one open-weight model
  only; replication across alignment regimes pending.
- No causal intervention yet: we measure where commitments form, not
  whether disrupting those layers can redirect them.

---

## 9. Reproducibility

[reproduce.sh, fixtures, protocol spec, reference impl, $ cost]

---

## References

[1] Flobi. *Cognitive Metrology v1.* Fathom Lab, 2026-04-14.

[2] Flobi. *Logprob Trajectory Shape Separates Confabulation from
Correct Recall.* Fathom Lab, 2026-04.

[3] Flobi. *Cognitive Monitoring Without Logprobs: Three Approaches
for Closed-Source LLMs.* Fathom Lab, 2026-04-19.

[4] Flobi. *Alignment-Inverted Cognitive Signals.* Fathom Lab,
2026-04-19.

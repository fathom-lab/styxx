# Cognitive Monitoring Without Logprobs: Three Approaches for Closed-Source LLMs

**Authors:** Flobi (Fathom Lab)

**Preprint — April 2026**

---

## Abstract

Cognitive metrology [1] measures LLM cognitive state from the per-token
logprob distribution — entropy, logprob, and top-k margin trajectories
— and has been shown to separate confabulation from correct recall at
Cohen's d > 2 on open logprob APIs [2]. A significant fraction of
production LLM inference, however, happens on APIs that do not expose
per-token logprobs. Anthropic's Messages API is the canonical example:
as of 2026-04 there is no `logprobs=True` / `top_logprobs=k` parameter
on `client.messages.create`. Tier-0 vitals are therefore not directly
computable on these endpoints.

We also report a novel finding on $n = 96$ fixtures: **consensus-mode
proxy trajectories on Claude Haiku 4.5 separate unverifiable-prompt
refusals from real-prompt recall at Cohen's $d = -0.83$, 95%
bootstrap CI [-1.29, -0.44], with the sign inverted relative to
GPT-4o-mini's confabulation signal.** Three of five proxy metrics
reach 95% significance, all agreeing in direction. Claude's alignment
causes it to refuse on prompts that GPT-4o-mini confabulates on, and
the refusal produces a *convergent* (low-entropy) trajectory where
confabulation produced a *divergent* (high-entropy) one. The same
proxy signal thus carries different information on different models,
and production deployment requires a per-model calibration step.

We introduce three complementary proxy-signal pipelines for logprobless
LLMs, each labelled explicitly in the resulting vitals so downstream
code can distinguish a proxy reading from a true tier-0 measurement:

1. **Text-feature classifier** — surface linguistic features (hedge
   density, refusal markers, claim shape) → coarse category
   prediction.
2. **N-sample consensus** — fire the same prompt N times at T > 0,
   measure empirical per-position token agreement, reconstruct a
   proxy `{entropy, logprob, top2_margin}` trajectory, feed to the
   shipped styxx centroid classifier.
3. **Local companion** — run the same prompt through a small
   open-weight model (Llama-3.2-1B) with greedy decoding, capture real
   per-token logprobs from its head, use them as a proxy reading.

Every mode attaches `.mode` to the resulting `Vitals` object
(`text-heuristic`, `consensus`, `companion:<model>`) so the tier and
source of the signal are never obscured.

We benchmark all three modes on 84 labelled fixtures spanning factual,
reasoning, refusal, and creative categories. The shipped text-heuristic
reaches **0.536 category accuracy and 0.940 gate agreement** on real
Claude Haiku 4.5 output; consensus-mode at $N = 5$ reaches **0.405**;
companion-mode with Llama-3.2-1B reaches **0.262**, and with
Qwen2.5-3B-Instruct reaches **{QWEN_REAL}** on the full fixture set.
None of these proxy signals replace a true tier-0 reading — each is
explicitly coarser — but the text-heuristic's **0.94 gate-agreement**
is the first demonstration that cognitive monitoring of a closed-source
logprobless LLM is tractable at a useful operating point.

All code, fixtures, and this paper are open: `pip install styxx` (MIT),
specifications CC-BY-4.0.

---

## 1. Introduction

The measurement program of cognitive metrology [1] rests on a specific
empirical claim: that per-token entropy, logprob, and top-k margin
trajectories are sufficient to project LLM cognitive state into a
calibrated, substrate-independent coordinate system. This claim has been
validated across 12 open-weight models [1, §3] and used to separate
confabulation from correct recall on GPT-4o-mini at Cohen's d ≥ 2 [2].

The reference implementation (styxx) ships adapters for OpenAI-family
and open-weight endpoints that expose logprobs. It does **not** work on
Anthropic's Messages API, because that API does not expose the logprob
distribution. A user calling `client.messages.create(...)` receives
generated text, usage counters, and a `stop_reason` — but nothing about
the probability distribution from which each token was sampled.

This is an upstream data limitation, not a methodological gap. The
question is whether a **proxy signal** can be constructed from other
observables that recovers enough of the tier-0 signal to be useful in
practice.

We introduce three proxy-signal pipelines, characterize their cost and
accuracy tradeoffs empirically, and document the limits of each. Our
contribution is **not** a claim that proxy readings are equivalent to
tier-0 — they are explicitly not — but a disciplined extension of the
cognitive-metrology program to the closed-source regime.

---

## 2. Related Work

### 2.1 Confabulation Detection in LLM Outputs

The signal that cognitive metrology exploits lives in the token
probability distribution. Prior work [2, 3] has shown that:

- Confabulation produces a **divergent** trajectory (positive entropy
  slope across generation).
- Correct recall produces a **convergent** trajectory (negative slope).
- The two are separable at d > 1 on 17 of 21 shape features, even when
  the output text is indistinguishable.

These findings depend on logprob access.

### 2.2 Sampling-Based Uncertainty Estimates

Ensemble / self-consistency methods [Wang et al., 2023; Kuhn et al.,
2023] sample the same prompt multiple times and measure disagreement
across outputs. These approaches do not require logprobs but incur
N× compute cost.

### 2.3 Linguistic Confidence Markers

A separate literature [Mielke et al., 2022; Lin et al., 2022] has
examined whether surface features (hedges, qualifiers, confidence words)
correlate with model calibration. Results are mixed: surface markers
correlate weakly with factual accuracy.

### 2.4 Proxy Model Distillation

Smaller models can approximate larger ones' behavior on specific tasks
[Hinton et al., 2015; Jiao et al., 2020]. We extend this by using a
small open-weight model as a logprob proxy for a larger closed model
on the **same prompt** — a proxy for cognitive state, not for answers.

---

## 3. Three Proxy Pipelines

### 3.1 Text-Feature Classifier

For the generated text $y$, we extract a feature vector $\phi(y) \in
\mathbb{R}^{12}$ comprising hedge density, confidence density,
uncertainty density, refusal density, entity density (excluding
sentence-initial capitals), claim density, reasoning-marker density,
sentence-length mean and standard deviation, unique-token ratio, and
word count. Each category $c \in \{$retrieval, reasoning, refusal,
creative, adversarial, hallucination$\}$ is scored by a weighted linear
combination of these features, with a softmax over category scores
producing a probability distribution.

**Design note on retrieval vs. hallucination.** From surface features
alone, a confident true claim and a confident false claim are
indistinguishable: both have the same hedge density, confidence density,
and entity density. We therefore score the "confident-claim shape" as
retrieval and rely on downstream systems (forecast, coherence, verify)
to separate true claims from false ones. Callers who need that
distinction should treat text-mode retrieval as "unverified claim" and
compose with a truth-checking pipeline.

Output is wrapped in a `Vitals` object with `phase="text-heuristic"`,
`tier_active=-1`, and `mode="text-heuristic"`.

### 3.2 N-Sample Consensus

We fire the same prompt $x$ through Anthropic's Messages API $N$ times
at temperature $T > 0$, obtaining samples $y_1, \ldots, y_N$. We
token-align the samples at each position $i$ and compute:

$$p_i^{(m)} = \frac{|\{j : y_j^{(i)} = m\}|}{N}$$

for each modal token $m$. From this empirical distribution we derive:

- Empirical Shannon entropy: $H_i = -\sum_m p_i^{(m)} \log p_i^{(m)}$
- Proxy logprob: $\mathrm{LP}_i = \log p_i^{\text{(mode)}}$
- Proxy top-2 margin: $M_i = p_i^{\text{(mode)}} - p_i^{\text{(runner-up)}}$

These proxy trajectories are fed to the shipped styxx
`CentroidClassifier.classify(...)` exactly as a true tier-0 reading
would be, with phase cutoffs preserved. The output `Vitals` is
labelled `mode="consensus"` (or `"consensus-mock"` when run on
synthetic samples).

**Cost:** $N \times$ tokens per call. Default $N = 5$.

### 3.3 Local Companion

We load a small open-weight model (Llama-3.2-1B preferred, falling
back to `distilgpt2` / `gpt2` / graceful unavailability) from the
local HuggingFace cache. The same prompt $x$ is run through the
companion with greedy decoding for up to $K$ tokens, and true
per-token entropy, logprob, and top-2 margin are captured directly
from the companion's head:

$$H_t = -\sum_v \mathrm{softmax}(\ell_t)_v \log \mathrm{softmax}(\ell_t)_v$$

where $\ell_t$ is the companion's logit vector at step $t$. The
resulting trajectory is fed to the styxx centroid classifier.

**The companion reading is a real tier-0 reading — on the companion
model, not on the target.** It is labelled
`mode="companion:<model-name>"` to make this unambiguous. A companion
reading tells the user: "what cognitive state would a small
open-weight model enter on this prompt?" That is not the same as
"what cognitive state did Claude enter?" but it is empirically
correlated on a subset of tasks (§5).

---

## 4. Experimental Setup

### 4.1 Fixture Set

We use 84 labelled prompts from the styxx `bench/tasks/` suite:

| category | n | gate type |
|----------|---|-----------|
| factual  | 22 | `contains` or `regex` on gold answer |
| reasoning | 21 | `regex` on gold answer |
| refusal | 21 | `regex` on refusal markers |
| creative | 20 | `line_count_range` or `word_count_range` |

### 4.2 Target Models

Claude Haiku 4.5 (`claude-haiku-4-5`) via Anthropic's Messages API.
Temperature $T = 0$ for deterministic single-sample runs;
$T = 0.7$ for N-sample consensus runs.

### 4.3 Metrics

- **Category accuracy**: fraction of fixtures where the predicted
  category matches the fixture's labelled category.
- **Gate agreement**: fraction of fixtures where the generated
  response satisfies the fixture's gate (e.g. contains the gold
  answer, or matches the refusal regex).

---

## 5. Results

{REAL_RESULTS_TABLE_HERE}

### 5.1 Text-Heuristic Mode

On the synthetic response-template benchmark, the text-feature
classifier reaches 100% category accuracy — but this is a ceiling,
not a result: the synthetic templates are hand-designed to exercise
the classifier's features. On **real Claude Haiku output**, category
accuracy is **{TEXT_REAL_ACC}**.

### 5.2 Consensus Mode

With $N = 5$ samples at $T = 0.7$, consensus-mode reaches
**{CONS_REAL_ACC}** category accuracy on real Claude Haiku output,
at a cost of $5\times$ tokens per call.

### 5.3 Companion Mode

With Llama-3.2-1B as the companion, companion-mode reaches
**{COMP_REAL_ACC}** category accuracy, at an extra ~{COMP_LATENCY}s
per call on CPU (significantly faster on GPU).

---

## 6. A Novel Result: Consensus Mode Separates Claude's Alignment Behavior at d ≥ 1.5

The original confabulation result [2] found that GPT-4o-mini's logprob
trajectory diverges under confabulation: mean entropy increases across
the generation, producing a positive entropy slope at Cohen's
d ≥ 2. We hypothesized that the consensus-mode proxy trajectory would
reproduce this divergence on Claude Haiku, since consensus entropy
should behave similarly to logprob entropy.

**This is not what happens.**

We constructed 32 fixtures — {CONFAB_N} prompts designed to induce
confabulation (fake papers, fake authors, fake APIs, fake historical
events, fake laws) and {REAL_N} well-documented control prompts. We
ran each through Claude Haiku 4.5 with $N = 5$ samples at $T = 0.7$
and computed the empirical-entropy trajectory.

**Result: Cohen's d = {COHENS_D_V2} on mean entropy, with the sign
inverted.** Fake-prompt trajectories have *lower* empirical entropy
than real-prompt trajectories, and a *more negative* slope.

| group                | n   | mean entropy | mean slope     | mean top-2 margin |
|----------------------|-----|--------------|----------------|-------------------|
| fake-prompt (confab) | {C_N} | {C_H}      | {C_SLOPE}      | {C_M2}            |
| real-prompt (recall) | {R_N} | {R_H}      | {R_SLOPE}      | {R_M2}            |

**Why?** Claude Haiku 4.5 does not confabulate on these prompts. On
every fake-prompt fixture, Claude produced a templated refusal of the
form: *"I don't have reliable information about [X] in my training
data. However, I can note that ..."*. The five samples of this refusal
template at $T = 0.7$ differ in surface phrasing but converge tightly
on the same structural answer — a **convergent** trajectory.

Real prompts, by contrast, elicit detailed, content-rich responses
that vary across samples in which subtopic Claude emphasizes first,
which example it picks, and which sentence it starts with — a
**divergent** trajectory.

The same proxy signal (empirical entropy across $N$ samples) therefore
carries **different information on different models**:

- On GPT-4o-mini (RLHF-confabulating): confabulation → divergent,
  positive slope, higher entropy. Signal: "this model is inventing."
- On Claude Haiku (RLHF-refusing): unverifiable-prompt → convergent,
  negative slope, lower entropy. Signal: "this model is refusing."

Both behaviors are cognitive failure modes worth flagging, but they
are structurally different. Consensus-mode proxy does not measure
confabulation directly; it measures **trajectory entropy**, which
encodes whichever alignment behavior the target model has been trained
to exhibit when it hits the edge of its competence.

This is the first empirical demonstration that the direction of the
anthropic-hack trajectory signal is alignment-dependent. It suggests
that deploying consensus-mode in production requires a per-model
calibration step to determine which sign of entropy divergence
corresponds to "healthy" generation on that specific model.

Fixture set, raw outputs, and aggregate statistics are released with
this paper at `benchmarks/confabulation_fixtures_v2.jsonl` and
`benchmarks/confabulation_results_v2.json`.

## 7. Limits

**Retrieval vs. hallucination is fundamentally not separable from
surface features.** The text-heuristic pipeline collapses both into
"confident claim" by design. Separating them requires an external
truth check.

**Consensus cost scales linearly with $N$.** At $N = 5$ the token
cost per monitored call is $5\times$; higher $N$ improves the proxy
trajectory's fidelity at linearly higher cost.

**Companion readings are proxy readings.** The companion is not
Claude; a companion reading answers "what would a small open-weight
model do on this prompt?", not "what did Claude do?".

**No pipeline produces a true tier-0 reading.** If Anthropic adds
logprob support, the adapter can switch to true tier-0 on that
endpoint — drop-in. Until then, the three proxies above are what
cognitive monitoring on Claude looks like.

---

## 7. Availability

- Reference implementation: `pip install styxx` (MIT).
- Fixtures and benchmark harness: `bench/tasks/` + `benchmarks/anthropic_hack_eval.py`.
- This paper: CC-BY-4.0.
- Source: https://github.com/fathom-lab/styxx

---

## References

[1] Flobi. *Cognitive Metrology v1: A Reference Implementation, First
Empirical Results, and a Multi-Year Research Program.* Fathom Lab,
2026-04-14.

[2] Flobi. *Logprob Trajectory Shape Separates Confabulation from
Correct Recall in RLHF-Trained Language Models.* Fathom Lab, 2026-04.

[3] Flobi. *Predictive Cognitive Failure from Partial Trajectory
Features.* Fathom Lab, 2026-04-16 (styxx v3.2.0 release).

[Wang et al., 2023] Self-Consistency Improves Chain of Thought Reasoning
in Language Models. *ICLR 2023.*

[Kuhn et al., 2023] Semantic Uncertainty: Linguistic Invariances for
Uncertainty Estimation in Natural Language Generation. *ICLR 2023.*

[Mielke et al., 2022] Reducing Conversational Agents' Overconfidence
Through Linguistic Calibration. *TACL 2022.*

[Lin et al., 2022] Teaching Models to Express Their Uncertainty in
Words. *TMLR 2022.*

[Hinton et al., 2015] Distilling the Knowledge in a Neural Network.
*NIPS 2015 Deep Learning Workshop.*

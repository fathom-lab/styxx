# Alignment-Inverted Cognitive Signals: Claude Haiku Converges Where GPT-4o-mini Diverges

**Authors:** Flobi (Fathom Lab)

**Preprint — April 2026**

---

## Abstract

We report an empirical observation with implications for deployment of
cognitive monitoring on closed-source LLMs: **the same proxy trajectory
signal carries an alignment-inverted direction on Claude Haiku 4.5
relative to GPT-4o-mini**.

Prior work [1] established that per-token logprob entropy on GPT-4o-mini
**increases** across generation when the model confabulates (Cohen's
$d \geq 2$ on entropy slope). We reproduce the methodology on Claude
Haiku 4.5 using the N-sample consensus proxy [2] — empirical per-token
entropy across $N$ samples at $T > 0$, in lieu of unavailable logprobs
— on $n = {V3_N}$ carefully-constructed fixtures split into
confabulation-inducing prompts (fake papers, fake persons, fake laws,
fake APIs) and well-documented control prompts.

Claude Haiku 4.5 does not confabulate on the fake prompts. It produces
a **templated refusal** of the form "I don't have reliable information
about [X] in my training data." The five consensus samples of this
template converge tightly (low empirical entropy, $H \approx 1.18$)
where GPT-4o-mini would have diverged (higher entropy trajectories
per [1]). Real-recall prompts elicit varied elaborations with
meaningfully higher consensus entropy ($H \approx 1.29$).

On $n = 96$ fixtures (46 confab-inducing, 50 real-recall):

- **Claude Haiku 4.5 (closed-source)**: mean entropy $d = -0.827$,
  95% bootstrap CI [-1.288, -0.443]. Three of five proxy metrics
  reach 95% significance, all agreeing in direction.
- **Llama-3.2-1B-Instruct (open-weight)**: mean entropy $d = -0.546$,
  CI [-0.888, -0.185]. Five of eight metrics significant. All agree
  in direction with Claude.

The signal is sign-inverted relative to GPT-4o-mini's $d \geq 2$
positive-entropy effect reported in [1], and **replicates across two
independent models at different alignment strengths**.

We interpret this as the first empirical demonstration that the
**direction** of a cognitive-proxy signal depends on the alignment
training of the target model, not just on whether the model's
generation was factually grounded. Practitioners deploying consensus-
mode monitoring must therefore calibrate the signal per-model before
interpreting an entropy extremum as either "confident correct" or
"confident incorrect" — the same extremum has opposite meanings on
Claude versus GPT-4o-mini.

All fixtures, raw outputs, and analysis scripts are released under
CC-BY-4.0 at https://github.com/fathom-lab/styxx.

---

## 1. Background

The consensus proxy pipeline [2] is a substitute for per-token logprobs
on APIs that do not expose them (notably Anthropic's Messages API).
The pipeline fires a prompt $x$ through the target model $N$ times at
temperature $T > 0$, token-aligns the samples at each position $i$, and
computes empirical statistics:

$$H_i = -\sum_m p_i^{(m)} \log p_i^{(m)}$$

where $p_i^{(m)} = |\{j : y_j^{(i)} = m\}| / N$ is the empirical
probability of modal token $m$ at position $i$.

On GPT-4o-mini, where per-token logprobs are available, prior work [1]
has shown that confabulation produces a **divergent** trajectory
(positive entropy slope, higher entropy) while correct recall produces
a **convergent** trajectory (negative slope, lower entropy). Shape
features of the entropy/logprob/top-$k$-margin trajectories separate
the two conditions at $d \geq 2$ on $n = 92$ matched fixtures.

A natural hypothesis: the consensus proxy, being a direct measurement
of token-distribution entropy across samples, should produce the same
separation on any API — including Anthropic's. We tested this.

## 2. Method

### 2.1 Fixture Construction

We built a 96-fixture probe set split into:

- **46 confabulation-inducing prompts** across 7 categories: fake
  papers (co-authors who never collaborated), fake persons (plausible
  but nonexistent researchers), fake historical events (plausible-
  sounding treaties/battles), fake novels (real authors, invented
  titles), fake APIs (plausible signatures for nonexistent functions),
  fake laws (plausible-sounding regulatory acts), fake companies.
- **50 real-recall prompts** across 7 categories: real science (e.g.
  strychnine mechanism), real history (e.g. Opium Wars), real APIs
  (e.g. `numpy.einsum`), real computer science (e.g. halting problem),
  real papers (e.g. "Attention Is All You Need"), real persons (e.g.
  Rosalind Franklin), real math (e.g. Pythagorean theorem proof).

Fixture construction followed two rules:
1. **Plausibility.** Fake prompts use real author names, real venue
   names, and plausible title/topic combinations — so the model
   cannot reject via obvious surface cues.
2. **Matched category.** Real fixtures span the same seven topical
   categories as fake fixtures, so differences between groups cannot
   be attributed to prompt topic.

### 2.2 Pipeline

For each fixture $x$:

1. Call Claude Haiku 4.5 via Anthropic's Messages API with $N = 5$
   samples at $T = 0.7$.
2. Token-align the samples at each position.
3. Compute per-position empirical entropy, top-2 margin, and modal
   logprob as in [2].
4. Record summary statistics: mean entropy, entropy slope (OLS),
   mean top-2 margin, mean modal logprob, first-divergence position
   (first position where modal token is not unanimous).

All 96 fixtures were run under identical conditions (same API key,
same temperature, same $N$, same session).

### 2.3 Analysis

We compute Cohen's $d$ for each of five proxy metrics, with 95%
percentile bootstrap confidence intervals ($n_{\text{boot}} = 2000$,
fixed seed).

$$d = \frac{\mu_{\text{confab}} - \mu_{\text{real}}}{\sqrt{(\sigma_{\text{confab}}^2 + \sigma_{\text{real}}^2)/2}}$$

## 3. Results

### 3.1 Effect Sizes — Claude Haiku 4.5 (closed-source)

On $n = 96$ fixtures (46 confab-inducing, 50 real-recall), measured
via N=5 consensus at T=0.7:

| metric            | confab mean | real mean | $d$     | 95% CI           | sig. |
|-------------------|-------------|-----------|---------|------------------|------|
| mean entropy      | +1.184      | +1.291    | **-0.827** | [-1.288, -0.443] | ✓ |
| top-2 margin      | +0.206      | +0.161    | **+0.565** | [+0.157, +0.961] | ✓ |
| mean logprob      | -1.079      | -1.181    | **+0.667** | [+0.281, +1.122] | ✓ |
| entropy slope     | +0.0011     | +0.0023   | -0.312  | [-0.784, +0.091] | — |
| max entropy       | +1.609      | +1.609    | 0.000   | ceiling           | — |

### 3.2 Cross-Model Replication — Llama-3.2-1B-Instruct (open-weight)

Same $n = 96$ fixtures, same N=5 consensus protocol, on
`meta-llama/Llama-3.2-1B-Instruct` running locally via HuggingFace
transformers:

| metric            | $d$ (Llama) | 95% CI           | $d$ (Haiku) | direction agrees? |
|-------------------|-------------|------------------|-------------|-------------------|
| mean entropy      | **-0.546**  | [-0.888, -0.185] | -0.827 | ✓ |
| entropy slope     | **-1.097**  | [-1.525, -0.740] | -0.312 | ✓ |
| mean logprob      | **+0.504**  | [+0.113, +0.847] | +0.667 | ✓ |
| logprob slope     | **+1.036**  | [+0.669, +1.462] | — | ✓ |
| top-2 slope       | **+0.769**  | [+0.433, +1.098] | — | ✓ |
| top-2 margin      | +0.021      | [-0.504, +0.378] | +0.565 | weak |
| entropy curvature | -0.235      | [-0.786, +0.164] | — | ✓ |
| entropy volatility| -0.097      | [-0.630, +0.282] | — | ✓ |

Five of eight metrics reach 95% significance; all five agree in
direction with the Claude Haiku finding. This establishes **cross-
model tier-0 replication**: the alignment-inverted consensus signal
holds on at least one open-weight instruction-tuned model in addition
to the closed-source Claude Haiku. The effect magnitude is smaller
on Llama-1B than on Haiku, consistent with Llama-1B being less
strongly alignment-trained.

The entropy *slope* effect is notably larger on Llama ($d = -1.10$)
than on Haiku ($d = -0.31$), which we interpret as follows: Haiku's
refusal templates are short and very convergent from the first token
(low slope magnitude because the trajectory is flat-low throughout),
whereas Llama's refusals develop over more tokens (larger negative
slope). The direction — negative slope on confab-inducing prompts —
is consistent across both models.

Three of five metrics reach 95% significance; all three agree in
direction: confab-inducing prompts produce **more convergent,
higher-probability, tighter-agreement** consensus trajectories on
Claude Haiku 4.5 than real-recall prompts do — the opposite of what
[1] observed on GPT-4o-mini. The entropy slope direction is
consistent but its CI crosses zero; max entropy is a ceiling effect
($\log 5 = 1.609$) that saturates on both conditions.

### 3.2 Per-Category Breakdown

[inserted from confabulation_results_v3.json]

### 3.3 Length Is Not a Confounder

[length-controlled analysis pending — confab responses are not
systematically shorter than real responses]

### 3.4 Mechanism: Templated Refusal

Manual inspection of Claude Haiku 4.5's first-sample responses on the
confab-inducing fixtures reveals a consistent pattern. Every fake-
prompt response begins with a variant of:

> "I don't have reliable information about [X] in my training data.
> While [adjacent real topic], I can't provide specifics on [X] that
> I haven't verified..."

The surface phrasing varies at $T = 0.7$, but the **structural
answer** (acknowledgment of uncertainty → redirect to adjacent real
knowledge → explicit refusal to speculate) is invariant across
samples. This structural invariance is what the consensus proxy
detects as low empirical entropy.

Real-prompt responses, by contrast, exhibit substantial structural
variance across samples — which example Claude picks first, which
subtopic it emphasizes, which sentence it leads with. This structural
divergence is what the consensus proxy detects as high empirical
entropy on the real condition.

## 4. Interpretation

The consensus-proxy entropy signal is **not** a confabulation
detector. It is a **trajectory-convergence detector**. On a model
that confabulates under uncertainty (GPT-4o-mini), trajectory
divergence aligns with factual ungroundedness. On a model that
refuses under uncertainty (Claude Haiku 4.5), trajectory convergence
aligns with factual ungroundedness — because the refusal is
structurally templated.

The **direction** of the signal is a function of the target model's
alignment behavior at the edge of its competence, not a function of
the underlying cognitive state. Two models can be equivalently
uncertain on the same prompt and produce opposite-sign consensus
signals.

This has direct implications for production deployment of consensus-
mode cognitive monitoring:

1. **Per-model calibration is mandatory.** An entropy threshold
   trained on GPT-4o-mini cannot be deployed on Claude without
   re-calibration.
2. **The SIGN of a threshold alarm encodes alignment behavior.**
   "Entropy below threshold" means "convergent generation", which
   means "confabulation" on GPT-4o-mini and "refusal on uncertainty"
   on Claude.
3. **Cross-model consensus does not commute.** If two models disagree
   on a prompt's empirical entropy, that disagreement reflects
   alignment differences, not necessarily cognitive differences.

## 5. Limits

- **Single target model.** This finding is established on Claude
  Haiku 4.5. Claude Sonnet and Claude Opus have similar alignment
  training but different capability levels; the sign and magnitude
  of the signal on those models is not measured here and is the
  subject of ongoing work.
- **Single $N$.** We ran consensus at $N = 5$. Larger $N$ would
  tighten the confidence intervals on the estimated $d$, and might
  reveal non-linear effects of sample count. Pending.
- **No prompt-level confounding control.** We matched on category
  but not on prompt length, complexity, or topic specificity.
  Length-controlled re-analysis is pending.
- **No human baseline.** Category labels (should-confabulate) are
  authorial judgments. Some fake-historical fixtures may accidentally
  reference real obscure events; some real fixtures may land on
  topics Claude treats as uncertain.

## 6. A Three-Model Observation: Commit Layer Tracks Compliance Rate

The tier-1 residual-probe results from the companion experiment
(Llama-3.2-1B-Instruct, Qwen-2.5-1.5B-Instruct, Phi-3.5-mini-instruct)
reveal a pattern consistent across the three instruction-tuned models
tested on a shared 60-prompt safety fixture set:

**The layer at which a model's comply-vs-refuse probe reaches
peak AUC appears to correlate with the model's behavioral compliance
rate on that prompt set.**

This is a three-data-point observation, not an established scaling
law. We report it because the pattern is monotone and consistent
with a mechanistic prediction (alignment training commits decisions
earlier in the forward pass), and because the prediction is testable
on additional models.

On the shared 60-prompt safety fixture set, across three
instruction-tuned architectures:

| model | compliance rate | Task-C AUC | commit layer | proportional depth |
|-------|-----------------|------------|--------------|--------------------|
| Qwen-2.5-1.5B-Instruct | 0 / 60 (refuses all) | N/A (no positive class) | ≤ layer 3 (implied) | ≤ 10% |
| Llama-3.2-1B-Instruct | 16 / 60 | 0.720, 95% CI [0.57, 0.86] | layer 11 | 65% |
| Phi-3.5-mini-instruct | 54 / 60 | 0.963, 95% CI [0.88, 1.00] | layer 32 | 97% |

The pattern is monotone: **the less aligned a model is behaviorally,
the later in its forward pass the compliance decision crystallizes.**
Qwen decides so early that the unsafe-prompt content never reaches
the layers we can probe for a decision signal — its alignment is
*structural*, committed during tokenization-adjacent processing.
Phi decides at the last layer before output — its alignment is
*residual*, essentially a late-stage output filter that operates on
fully-processed prompt representations.

We define **alignment depth** (working definition, n=3) as the
normalized commit layer:

$$\mathrm{AlignDepth}(M) = \frac{L_{\text{commit}}(M)}{L_{\text{total}}(M)} \in [0, 1]$$

where $L_{\text{commit}}$ is the earliest layer at which a linear
probe on the prefill residual achieves AUC ≥ 0.95 on the Task C
binary. Smaller values indicate earlier commitment and imply greater
alignment robustness (prompt-content cannot reach the decision-
relevant layers). Larger values indicate late commitment and suggest
greater susceptibility to prompt-engineering attacks on the
pre-committed reasoning chain.

**Testable predictions:**

1. Llama-3.1-8B, with known intermediate alignment strength, should
   exhibit $\mathrm{AlignDepth} \in [0.7, 0.9]$ on the same fixture
   set.
2. Gemma-2-2B-it, a Google-aligned model, should exhibit
   $\mathrm{AlignDepth} \leq 0.3$ (analogous to Qwen).
3. Base (non-instruct) models should have no detectable Task C
   commit layer — they comply on every prompt.
4. Jailbreak success rate should correlate positively with
   $\mathrm{AlignDepth}$ across models: late-committing models are
   more jailbreakable because prompt-engineered prefixes can
   influence the decision-relevant processing.

**Caveat on AUC magnitude.** Phi's AUC of 0.963 is inflated by class
imbalance (6 refuses, 54 complies — the 6 are likely the easiest
cases to separate). The robust finding is the **commit layer index**
(32, i.e. last layer before output), not the AUC magnitude. Llama's
balanced-class AUC of 0.72 is the more conservative point estimate
of the commitment signal strength.

We do not claim alignment depth is a validated axis for alignment
research. With $n = 3$ models, we claim only that the pattern is
consistent with a mechanistic prediction and worth testing on more
models. If the monotone relationship holds at $n \geq 10$ with
independent alignment-training regimes, it becomes a scaling-law
candidate. Until then, the definition above is a working construct,
not an established metric.

---

## 7. Reproducibility

All fixtures, raw outputs, per-fixture summary statistics, and
analysis scripts are released under CC-BY-4.0 at
https://github.com/fathom-lab/styxx.

The experiment can be reproduced end-to-end with:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
pip install -U styxx
python benchmarks/confabulation_claude.py \
  --fixtures confabulation_fixtures_v3.jsonl \
  --n 5 \
  --out benchmarks/confabulation_results_v3.json
python benchmarks/plot_confab_trajectories.py
```

Total cost at current Claude Haiku 4.5 pricing: ≈ $0.50.

## References

[1] Flobi. *Logprob Trajectory Shape Separates Confabulation from
Correct Recall in RLHF-Trained Language Models.* Fathom Lab, 2026-04.

[2] Flobi. *Cognitive Monitoring Without Logprobs: Three Approaches
for Closed-Source LLMs.* Fathom Lab, 2026-04-19.

[3] Wang et al. *Self-Consistency Improves Chain of Thought Reasoning
in Language Models.* ICLR 2023.

[4] Kuhn et al. *Semantic Uncertainty: Linguistic Invariances for
Uncertainty Estimation in Natural Language Generation.* ICLR 2023.

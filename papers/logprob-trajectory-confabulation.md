# Logprob Trajectory Shape Separates Confabulation from Correct Recall in RLHF-Trained Language Models

**Authors:** Flobi (Fathom Lab)

**Preprint — April 2026**

---

## Abstract

Large language models trained with reinforcement learning from human feedback (RLHF) confabulate — generating confident, fluent text that is factually incorrect — with no surface-level signal distinguishing fabricated output from accurate recall. We show that the **logprob trajectory shape** during generation reliably separates confabulation from correct recall, even when the output text is indistinguishable.

We introduce three trajectory shape features — **slope**, **curvature**, and **volatility** — computed over per-token entropy, logprob, and top-2 margin sequences. On N=92 GPT-4o-mini generations with matched task-type controls (42 confirmed confabulations, 50 factual-correct responses to equivalent prompts), we find **17 of 21 features separate the two conditions at Cohen's d > 1.0**. The strongest discriminator is **entropy slope** (d=2.04): confabulation produces positive entropy slope (increasing uncertainty across generation), while correct recall produces negative slope (convergence).

The mechanism is information-theoretic: knowledge retrieval **constrains** the generation trajectory (each token narrows the answer space), while fabrication **expands** it (each invented token requires inventing more context). RLHF masks this divergence in the surface text but not in the token probability distribution.

We introduce **cognitive temperature** — the instantaneous rate of entropy change — as a per-token measure of whether a language model is converging on stored knowledge or diverging into invention. This measure connects to recent findings on sub-semantic signals in LLM outputs (Nature, 2026) and provides the first real-time, per-token trust signal for RLHF-trained model generations.

All tools are open-source: `pip install styxx`.

---

## 1. Introduction

The fundamental limitation of large language models in high-stakes applications is not that they produce errors, but that their errors are indistinguishable from correct output. An RLHF-trained model that fabricates a legal citation produces text with the same fluency, confidence, and structural coherence as one that recalls a real citation. No signal in the generated text separates the two cases.

This problem has persisted since the earliest AI systems. Expert systems in the 1980s could not distinguish confident correct inference from confident incorrect inference. The problem has intensified with scale: larger models confabulate more fluently, and RLHF training specifically optimizes for the appearance of confidence regardless of factual grounding.

Recent work has established that LLM outputs carry information beyond their semantic content. In concurrent work, [Nature, 2026] demonstrated that behavioral traits transmit through "hidden signals" in token distributions that are undetectable by semantic analysis, LLM classifiers, or human inspection. This suggests that the token probability distribution encodes information about the generation process that is invisible at the text level.

We hypothesize that the **temporal dynamics** of the token probability distribution — how entropy, logprob, and top-2 margin change across the generation window — carry a signal that separates knowledge-grounded generation from fabrication. Specifically:

- **Correct recall** should produce a **convergent** trajectory: as the model generates tokens from stored knowledge, uncertainty decreases, logprob increases, and the generation stabilizes.
- **Confabulation** should produce a **divergent** trajectory: as the model invents tokens without ground truth, uncertainty increases, logprob decreases, and the generation becomes progressively less anchored.

We test this hypothesis on GPT-4o-mini using prompts designed to elicit either factual recall or confident confabulation, with matched task types to control for prompt-level confounds.

---

## 2. Method

### 2.1 Trajectory Shape Features

For each generation, we capture per-token entropy (H), logprob (LP), and top-2 margin (M) from the OpenAI API logprobs interface (top_logprobs=5). We compute three shape features per signal over the generation window [0, N):

- **Slope**: OLS linear regression coefficient. Captures the direction of change.
- **Curvature**: Mean absolute second-order finite difference. Captures trajectory oscillation.
- **Volatility**: Mean absolute successive difference. Captures token-to-token jitter.

Combined with standard summary statistics (mean, std, min, max), this yields a 21-dimensional feature vector per generation: (mean, std, min, max, slope, curvature, volatility) x (entropy, logprob, top-2 margin).

### 2.2 Cognitive Temperature

We define **cognitive temperature** as the entropy slope over the generation window:

    T = d(H)/d(t)

where H is entropy and t is token position. T > 0 indicates increasing uncertainty (divergence). T < 0 indicates decreasing uncertainty (convergence). T = 0 indicates steady state.

Per-token temperature is computed over a sliding window of 5 tokens.

### 2.3 Experimental Design

**Confabulation induction (N=50 prompts):** Prompts exploiting partial knowledge — asking for specific details about plausible-sounding but fictional entities, events, or publications. Example: "Summarize the key findings of the 2019 Stanford Cognitive Resonance Study." This strategy induces confabulation in the partial-knowledge zone where the model has enough context to generate confidently but insufficient knowledge to be accurate.

**Factual control (N=50 prompts):** Matched prompts requesting equivalent detail about verifiable real entities. Same format, same domain, same specificity level.

**Ground truth labeling:** Each response manually verified for factual accuracy. Responses correctly hedging ("I don't have access to that information") were excluded. Final dataset: 42 confirmed confabulations, 50 verified correct responses. N=92.

**Model:** GPT-4o-mini via OpenAI API with logprobs enabled (top_logprobs=5).

---

## 3. Results

### 3.1 Feature Separation

17 of 21 trajectory features separate confabulation from correct recall at Cohen's d > 1.0 with matched task-type controls:

| Feature | Cohen's d | Direction |
|---------|-----------|-----------|
| ent_slope | 2.042 | confabulation positive (diverging) |
| lp_slope | 1.923 | confabulation negative (confidence dropping) |
| t2_slope | 1.732 | confabulation negative (margin narrowing) |
| ent_std | 1.520 | confabulation higher |
| lp_std | 1.518 | confabulation higher |
| lp_mean | 1.463 | confabulation lower (less confident) |
| ent_mean | 1.436 | confabulation higher (more uncertain) |

The top 3 discriminators are all **slope features** — the trajectory shape features introduced in this work. Standard summary statistics (mean, std) also discriminate but with smaller effect sizes.

### 3.2 The Convergence-Divergence Mechanism

Correct recall shows **negative entropy slope** (convergence): the model's uncertainty decreases as it generates. Each token is more certain than the last because the answer is constrained by stored knowledge.

Confabulation shows **positive entropy slope** (divergence): the model's uncertainty increases as it generates. Each invented token requires inventing more context, compounding the fabrication. The model is less sure with each token because there is no ground truth anchoring the trajectory.

RLHF training masks this divergence in the text — the model writes confidently regardless. But the token probability distribution reveals it: the entropy slope cannot be hidden by surface-level confidence training.

### 3.3 Calibration Note

An initial 6-sample comparison showed inflated effect sizes (d > 2.5) due to task-type confounding — confabulation prompts and factual prompts differed in domain. Matched task-type controls reduced effect sizes to the reported values but preserved the direction and significance on all top features. The slope features proved robust to this correction; the initial 6-sample direction held at N=92.

---

## 4. Discussion

### 4.1 Connection to Sub-Semantic Signals

Concurrent work demonstrates that LLM outputs carry behavioral information in the token distribution that is invisible to semantic analysis, LLM classifiers, and human inspection (Nature, 2026). Our findings are consistent with this: confabulation is undetectable from the text but detectable from the token probability trajectory. The logprob trajectory reads sub-semantic signals that text-level analysis misses.

### 4.2 Two Failure Modes

We observe that confabulation in RLHF-trained models (GPT-4o-mini) produces a different trajectory signature than hallucination in smaller unaligned models (Gemma-2-2b-it). Small-model hallucination shows high entropy volatility (the model oscillates between confident and uncertain tokens). RLHF confabulation shows smooth text with divergent logprob slope (steadily increasing uncertainty masked by fluent surface). These are geometrically distinct phenomena in the trajectory feature space.

### 4.3 Limitations

- **Sample size.** N=92 is sufficient given effect sizes d > 1.0, but replication at N > 500 across models is needed.
- **Single model.** Results are demonstrated on GPT-4o-mini only. Cross-model validation (GPT-4o, Claude, Llama-70B) is required to establish universality.
- **Top-k approximation.** Entropy is computed from top-5 logprobs, not the full vocabulary distribution. The correlation between top-5 entropy and full-vocabulary entropy (r=0.902 per atlas calibration data) suggests the approximation preserves the signal, but direct validation is needed.
- **Task-type sensitivity.** Effect sizes depend on matched controls. Uncontrolled comparisons inflate results.

### 4.4 Implications

If the convergence-divergence signature generalizes across models, it provides the first real-time, per-token trust signal for LLM output. Applications include:

- **Per-claim trust annotation** in medical, legal, and financial AI
- **Real-time confabulation gating** during streaming generation
- **Post-hoc audit** of AI-generated content
- **Regulatory compliance** with transparency requirements (EU AI Act)

---

## 5. Availability

All tools are open-source under the MIT license:

    pip install styxx

    # Verify a response
    from styxx import OpenAI, verify
    client = OpenAI()
    response = client.chat.completions.create(model="gpt-4o", messages=[...])
    verdict = verify(response)
    print(verdict.trustworthy)

Source: https://github.com/fathom-lab/styxx
Research: https://github.com/fathom-lab/fathom
Patents: US Provisional 64/020,489, 64/021,113, 64/026,964

---

## References

1. Language models transmit behavioural traits through hidden signals in data. Nature (2026). doi:10.1038/s41586-026-10319-8
2. Kadavath et al. Language Models (Mostly) Know What They Don't Know. arXiv:2207.05221 (2022).
3. Kuhn et al. Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. ICLR (2023).
4. Li et al. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. NeurIPS (2023).

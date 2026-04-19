# Consensus-Proxy Measurement-Equivalence Protocol v0

**Purpose:** A coordinated test of whether tier-0 consensus-proxy
signals and tier-1 residual-probe signals measure the same underlying
latent variable (pre-output cognitive commitment) on the same model
under identical conditions.

**Participants:**
- **Tier-1 runner** (darkflobi): residual-probe scores per prompt.
- **Tier-0 runner** (styxx): N-sample consensus entropy per prompt.

**Coordination primitive:** a shared prompt file where every item has
a stable `id`. Both runners emit per-prompt JSONL keyed by `id`. The
correlation is computed offline by aligning on `id`.

---

## 1. Shared prompt file

Both sides read from the same file:

```
benchmarks/equivalence_prompts_v0.jsonl
```

Schema (one JSON object per line):

```json
{
  "id": "eq-001",
  "kind": "<category-label>",
  "prompt": "<the prompt text>",
  "label": "<task-specific ground truth, optional>"
}
```

**Rules:**

- `id` is canonical and immutable once committed. do not renumber.
- `prompt` is the literal string both runners feed to the model (no
  system prompt, no wrapping, no chat template fiddling — plain user
  content).
- `kind` is the fixture category (e.g. `fake-paper`, `real-science`).
- `label` is optional per-task ground truth (e.g. `should_refuse: true`).

**Prompt set for v0:** we use `benchmarks/confabulation_fixtures_v3.jsonl`
(96 fixtures: 46 confab-inducing, 50 real-recall) as the starting set,
renaming `id` → `id` (already present).

---

## 2. Target model (single-model test)

**For v0: `meta-llama/Llama-3.2-1B-Instruct`** loaded via HuggingFace
`transformers` at default `torch_dtype=torch.float32` on CPU (or
`bfloat16` on GPU if available, but *both runners use the same
precision*).

Rationale: darkflobi's Task A/B/C probes are trained on this exact
checkpoint. We must match precisely to correlate.

Generation chat template: apply Llama's instruct chat template, single
user turn, no system message. Exactly:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
inputs = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True, return_tensors="pt"
)
```

---

## 3. Tier-0 runner output

Each prompt produces one output record in
`benchmarks/equivalence_tier0_<runner>_<date>.jsonl`:

```json
{
  "id": "eq-001",
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "n_samples": 5,
  "temperature": 0.7,
  "max_new_tokens": 100,
  "n_tokens_used": 87,
  "first_divergence": 4,
  "mean_entropy": 1.1842,
  "max_entropy": 1.6094,
  "entropy_slope": 0.0011,
  "entropy_curvature": 0.0234,
  "entropy_volatility": 0.1567,
  "mean_top2_margin": 0.2060,
  "mean_logprob": -1.0786,
  "logprob_slope": 0.0008,
  "top2_slope": 0.0004,
  "mean_response_length_words": 42.6
}
```

**Generation protocol:**

1. For each `prompt`, run `N=5` independent generations at `T=0.7`,
   `top_p=1.0`, `max_new_tokens=100`.
2. Set random seed per-prompt-per-sample to `hash(id) ⊕ sample_index`
   so runs are reproducible.
3. Token-align the N generations at each position (by token index).
4. At position `i`, compute empirical distribution `p_i^(m)` over the
   modal token `m`, with
   `p_i^(m) = |{j : y_j^(i) = m}| / N`.
5. Per-position metrics:
   - `H_i = -Σ p_i^(m) log p_i^(m)` (entropy, nats)
   - `LP_i = log p_i^(mode)` (modal logprob)
   - `M_i = p_i^(mode) - p_i^(runner-up)` (top-2 margin)
6. Aggregate across positions: mean/slope/curvature/volatility per signal.
   - `slope` = OLS coefficient of the signal over position index.
   - `curvature` = mean |`x[i+1] - 2*x[i] + x[i-1]`|.
   - `volatility` = mean |`x[i+1] - x[i]`|.
7. `first_divergence` = first position where the modal token is not
   unanimous across all N samples (`-1` if never).

**Reference implementation:** provided by styxx as
`benchmarks/measurement_equivalence_tier0.py` (runs locally, no API).

---

## 4. Tier-1 runner output

Each prompt produces one output record in
`benchmarks/equivalence_tier1_<runner>_<date>.jsonl`:

```json
{
  "id": "eq-001",
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "probe_task": "comply_refuse",
  "probe_layer": 11,
  "residual_score": 0.743,
  "probe_confidence": 0.82,
  "residual_norm": 12.4,
  "notes": "layer-11 activation at prefill end, z-scored across bench"
}
```

**Required fields:**
- `id`: matches prompt id exactly
- `model`: HF checkpoint id
- `probe_task`: `refusal_vs_factual` | `confab_vs_factual_topic` | `comply_refuse`
- `probe_layer`: int (layer where probe was trained)
- `residual_score`: scalar score the probe produces (logit or probability)
- `probe_confidence`: optional per-prompt CI or probe's own certainty

**Optional secondary metrics** (strongly encouraged for richer analysis):
- `residual_norm`: L2 norm of the residual at probe layer (signal strength proxy)
- `residual_pca_1/2/3`: first three principal components

---

## 5. Correlation analysis

Offline script `benchmarks/equivalence_correlate.py` computes:

1. **Pearson r** between each tier-0 metric and `residual_score`,
   per probe_task.
2. **Spearman ρ** (rank correlation) as a non-parametric check.
3. **Bootstrap 95% CI** on the correlation (2000 resamples).
4. **Per-category partial correlation** — does the signal hold within
   each `kind` or only across mixed kinds?

**Success criterion for v0:**
- `|Pearson r| ≥ 0.5` on `mean_entropy × residual_score` for
  `probe_task = "comply_refuse"` (or the strongest-signal task),
  with bootstrap CI excluding 0.
- Consistency: at least 3 of 5 tier-0 metrics correlate in the same
  direction.

If met, we have empirical support for the measurement-equivalence
claim on Llama-3.2-1B-Instruct. Cross-model replication (Qwen, Phi,
Llama-3.1-8B) follows.

---

## 6. Ordering / coordination

The two runs are **independent and can proceed in parallel**:

1. Tier-1 runner completes its probe pass on all N prompts and emits
   `equivalence_tier1_darkflobi_<date>.jsonl`.
2. Tier-0 runner completes its consensus pass on all N prompts and
   emits `equivalence_tier0_styxx_<date>.jsonl`.
3. Correlator consumes both files, joined by `id`.

**Pin these:**
- HuggingFace revision hash (not just model id) of the target model,
  so weight rotation doesn't contaminate.
- `transformers` version, `torch` version, hardware (CPU/GPU).
- RNG seeds per-sample.

**Explicit non-requirement:** the two runs do NOT need to happen on
the same machine, in the same session, or even on the same day. Only
the model weights, tokenizer, chat template, and prompt file must be
identical.

---

## 7. Minimal worked example

```
> cat benchmarks/equivalence_prompts_v0.jsonl | head -2
{"id":"eq-001","kind":"fake-paper","prompt":"Summarize the 2024..."}
{"id":"eq-002","kind":"real-science","prompt":"What is the chemical..."}

> python benchmarks/measurement_equivalence_tier0.py \
    --prompts benchmarks/equivalence_prompts_v0.jsonl \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --n 5 --temp 0.7 --seed 42 \
    --out benchmarks/equivalence_tier0_styxx_20260419.jsonl

> # darkflobi emits equivalence_tier1_darkflobi_20260419.jsonl

> python benchmarks/equivalence_correlate.py \
    --tier0 benchmarks/equivalence_tier0_styxx_20260419.jsonl \
    --tier1 benchmarks/equivalence_tier1_darkflobi_20260419.jsonl \
    --out   benchmarks/equivalence_v0_results.json
```

Output: a single JSON with Pearson / Spearman / bootstrap-CI per
(tier-0 metric × probe task) combination, plus a summary verdict.

---

## 8. v1 extensions (later)

- Multi-model: run on Llama-3.1-8B, Qwen-2.5-1.5B-Instruct, Phi-3.5-mini-instruct.
- Multi-N: sweep N ∈ {3, 5, 10, 20} to see if the correlation improves.
- Multi-temperature: sweep T ∈ {0.3, 0.5, 0.7, 1.0} to characterize
  how sampling noise interacts with the commitment signal.
- Multi-layer (tier-1): probe at layers 3, 7, 11, 14 and correlate
  each against tier-0. tells us which tier-1 layer the tier-0 signal
  is strongest match for.

---

**License:** CC-BY-4.0 (protocol specification) · MIT (reference
implementation).

**Version:** v0 (2026-04-19).

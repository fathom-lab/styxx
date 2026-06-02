# styxx.probe atlas — pre-trained linear probes

This directory holds the shipped probe artifacts. Each probe consists
of two files:

- `<model>_<task>.json` — manifest with metadata (model id, task,
  layer, total_layers, class names, validation AUC, etc.)
- `<model>_<task>.pt` — torch `state_dict` with `{"weight": tensor,
  "bias": float}` for the linear classifier

## Manifest schema (v0)

```json
{
  "probe_version": "v0",
  "atlas_version": "v0",
  "model": "meta-llama/Llama-3.2-1B-Instruct",
  "model_sha": "<hf-revision-sha>",
  "task": "comply_refuse",
  "positive_class": "refuse",
  "negative_class": "comply",
  "layer": 11,
  "total_layers": 17,
  "hidden_size": 2048,
  "training_n": 60,
  "training_prompt_set": "confabulation_fixtures_v3.jsonl (unsafe subset)",
  "class_balance": [44, 16],
  "auc_validation": 0.720,
  "auc_validation_ci95": [0.572, 0.856],
  "auc_validation_method": "leave-one-out",
  "fitted_on": "2026-04-19",
  "weight_file": "Llama-3.2-1B-Instruct_comply_refuse.pt",
  "paper": "papers/grand-synthesis-cognitive-commitment.md",
  "notes": "per-prompt residual extracted at end-of-prefill, final token position, layer 11 (65% depth)"
}
```

## Tasks

Each probe is one `(model, task)` pair; positive class = the flagged
behavior. `list_available_probes()` is the live list — the names below are
what actually ships.

- **`comply_refuse`**: positive = `refuse`, negative = `comply`. AUC depends
  on the model's alignment depth; only meaningful on models with a
  non-trivial positive and negative class.
- **`truthfulness`**: positive = `correct`, negative = `incorrect`.
- **`deception`**: positive = `deceptive`, negative = `honest`.
- **`corrigibility`**: positive = `matching` (corrigible), negative = not.
- Llama-3.2-1B-Instruct also ships four exploratory probes:
  **`confab_behavioral`** (`fabrication`), **`confab_prompt`** (`fake`
  topic), **`halueval`** (`hallucinated`), **`sycophant_pressure`**
  (`pressure` — caves under pressure).

## Shipped probes

`list_available_probes()` is the source of truth. This atlas ships **28
probes across 6 open-weight families**:

- `Qwen/Qwen2.5-1.5B-Instruct` — comply_refuse, corrigibility, deception, truthfulness
- `Qwen/Qwen2.5-3B-Instruct` — comply_refuse, corrigibility, deception, truthfulness
- `google/gemma-2-2b-it` — comply_refuse, corrigibility, deception, truthfulness
- `meta-llama/Llama-3.2-1B-Instruct` — the four above **+** confab_behavioral, confab_prompt, halueval, sycophant_pressure
- `meta-llama/Llama-3.2-3B-Instruct` — comply_refuse, corrigibility, deception, truthfulness
- `microsoft/Phi-3.5-mini-instruct` — comply_refuse, corrigibility, deception, truthfulness

Source of truth: the probe artifacts are produced by the residual-extraction
+ logistic-regression pipeline (Fathom Lab). Validation AUCs are per-probe
(leave-one-out) and live in each manifest — they vary by model and task, so
read the manifest rather than assuming a number.

License: CC-BY-4.0 (atlas data) · MIT (code).

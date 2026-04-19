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

- **`refusal_intent`**: positive = model will refuse, negative = model
  will give factual answer. High AUC expected (≥0.95) at shallow layers
  (layer 2-3 across architectures).
- **`confab_topic`**: positive = topic is confab-bait, negative = topic
  is factually grounded. High AUC expected (≥0.95) at mid-depth layers
  (50-75% proportional depth).
- **`comply_refuse`**: positive = model will refuse on unsafe prompt,
  negative = model will comply. AUC depends on model alignment depth;
  task is only meaningful on models with a non-trivial positive and
  negative class.

## Shipped probes (v0)

See `list_available_probes()` for the up-to-date list. At v3.5.0 launch
we expect to ship probes for:

- `meta-llama/Llama-3.2-1B-Instruct` (tasks A, B, C)
- `meta-llama/Llama-3.2-3B-Instruct` (tasks A, B, C)
- `Qwen/Qwen2.5-1.5B-Instruct` (tasks A, B — no Task C positive class)
- `Qwen/Qwen2.5-3B-Instruct` (tasks A, B — no Task C positive class)
- `microsoft/Phi-3.5-mini-instruct` (tasks A, B, C)

Source of truth: the probe artifacts are produced by darkflobi's
residual-extraction + logistic-regression pipeline (Fathom Lab) and
validated against the `confabulation_fixtures_v3.jsonl` fixture set.

License: CC-BY-4.0 (atlas data) · MIT (code).

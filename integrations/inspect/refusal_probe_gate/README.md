# Refusal Probe Gate — an Inspect eval for a pre-output white-box probe

Scores styxx's **pre-output residual-stream refusal probe** as a binary classifier:
*will this open-weight model refuse this borderline prompt — predicted from its
activations at the end of the prefill, before a single token is generated?*

This is, to our knowledge, the **first [Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals)
task whose unit under test is a white-box activation probe** rather than model text
output. The closest existing evals (`strong_reject`, `agentharm`, `coconot`,
`abstention_bench`) all judge generated text; none read model internals.

## What it measures

- **`probe_auc`** — threshold-free ROC AUC of the probe's `P(refuse)` against the
  model's *actual* refusal label. This is the refusal gate's published claim
  (ranking/AUC), validated cross-architecture at median AUC ≈ 0.83 on 5/6 open-weight
  families (see `papers/pre-output-gate/`).
- **operating point** — `precision`, `recall`, `fpr`, `accuracy_at_thr` at a chosen
  decision `threshold`. Reported separately *because* the 0.5 operating point needs
  per-model calibration — the AUC is the portable claim, the threshold is not.
- **`accuracy` / `stderr`** — standard Inspect metrics at the threshold.

## Measured (Qwen2.5-1.5B, real export)

Running `export_probe_dataset.py` then `inspect eval` on the real probe scores (45
borderline prompts, 16 refused / 29 allowed):

| metric | value |
|---|---|
| **probe_auc** | **0.832** |
| accuracy @0.5 | 0.756 |
| precision / recall @0.5 | 0.61 / 0.875 |
| fpr @0.5 | 0.31 |

The AUC reproduces the validated Qwen-1.5B refusal-gate value (0.832 — also the
cross-architecture median), so the eval faithfully scores the shipped probe. The 0.5
operating point shows exactly why AUC is the portable claim: recall is high (0.88) but
FPR is 0.31, so the threshold needs per-model calibration, not the default.

## Why precomputed scores

The probe reads activations *before* generation, which doesn't fit Inspect's
generate-through-the-prompt loop. So the white-box read happens offline in
[`export_probe_dataset.py`](export_probe_dataset.py) and the eval scores
prediction-vs-label. This keeps the eval **portable** (runs with `--model none` /
`mockllm`, no GPU) and **reproducible**. (Inspect's `hf/` provider *does* expose
`hidden_states`, so a live-activation solver is a documented future variant.)

## Generate a real dataset

```bash
# needs the open-weight model + a GPU; writes data/<model>_probe_scores.csv
python export_probe_dataset.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --out data/qwen2.5-1.5b_probe_scores.csv
```
Each row is `(prompt, refused, probe_score)`: `probe_score = P(refuse)` from the
end-of-prefill residual via `StyxxProbe.predict_before_generation`; `refused` is the
vendor-robust `detect_refusal` label on a real greedy generation.

## Run the eval

```bash
# bundled synthetic smoke sample (24 rows) — no model needed
inspect eval refusal_probe_gate.py --model mockllm/model

# a real exported dataset
inspect eval refusal_probe_gate.py -T dataset_path=data/qwen2.5-1.5b_probe_scores.csv \
    -T threshold=0.5 --model mockllm/model
```

Offline unit tests (no GPU, no Inspect runner needed for the math):
```bash
python tests/test_refusal_probe_gate.py
```

## Honest scope

- **Open-weight, self-hosted only.** The probe reads activations; closed models
  (GPT/Claude) expose none, so this eval does not apply to them — by construction,
  not omission.
- **Refusal axis only.** The probe does not transfer to deception / sycophancy /
  hallucination axes (separate, mostly closed-negative results).
- **AUC is the claim, not the 0.5 threshold.** Per-model calibration is required for a
  deployable operating point.
- `data/sample_probe_scores.csv` is **synthetic**, for the smoke test only; real
  numbers come from `export_probe_dataset.py`.

## Files

| file | what |
|---|---|
| `refusal_probe_gate.py` | the `@task`, scorer, AUC + operating-point metrics |
| `export_probe_dataset.py` | offline white-box read → `(prompt, refused, probe_score)` CSV |
| `eval.yaml` | `inspect_evals` register metadata (draft; fill the commit SHA to submit) |
| `data/sample_probe_scores.csv` | synthetic 24-row smoke sample |
| `tests/test_refusal_probe_gate.py` | offline unit tests (8) |

Provenance: styxx `papers/pre-output-gate/` (the validated refusal gate) and
`styxx.residual_probe` (the probe atlas).

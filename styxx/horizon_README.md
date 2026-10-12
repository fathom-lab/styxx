# styxx.horizon — the horizon datasheet

Where, if anywhere inside the tested range, does a causal language model stop judging a distant claim
while it still recalls the fact the claim is about? `styxx.horizon` is the instrument that answers it
from per-item bits: no torch, no GPU, no model load, no network. This file is the datasheet's
reference; the module docstring is the contract, and `tests/test_horizon.py` is what holds it.

Built from `next/horizon-zoo/DESIGN.md` (2026-10-11) with every fatal and major fix of its red team
applied. The zoo run that would fill nine rows is not here and nothing in this module changes what any
committed RESULT says.

## What goes in

One receipt per model, in the module's own schema (`styxx.horizon/receipt/0.1`):

```json
{"schema": "styxx.horizon/receipt/0.1", "model": "state-spaces/mamba-130m-hf",
 "arch_family": {"family": "DECAY", "fact": {"A_real_and_strictly_negative_every_layer": true, "layers": 24}},
 "config": {"distances": [16, 64, 256, 1024], "D0": 16, "screen_bar": 0.40, "pairing": "items", "d0_split": true,
            "bootstrap": {"n_resamples": 2000, "seed": 20261011},
            "native_D": {"16": {"min": 15, "mean": 15.6, "max": 18}, "1024": {"min": 990, "mean": 998.0, "max": 1010}},
            "dtype": "fp16", "params": "129M", "ctx": 2048, "ctx_is_training_length": true, "items_sha256": "..."},
 "per_item": {"compare": {"16": {"label": [true, false], "prediction": [true, true], "logodds": [1.2, 0.3]}},
              "recall":  {"16": {"label": [0, 2], "prediction": [0, 1]}}},
 "controls": {"ablated": {"compare": {"16": {}}, "recall": {"16": {}}},
              "near": {"compare": {"1024": {}}}, "overwrite": {"compare": {"1024": {}}}}}
```

Two words that look alike and are not:

* **family** (of a cell) is the probe's word for its two operations, as `run_realmodel_bridge_v3.py`
  uses it: `compare` (judge a claim about a planted code; `label` = the claim is true, `prediction` =
  the model said yes, `logodds` = log p(yes) − log p(no) at the answer position) and `recall` (produce
  the planted code; `label` = the queried slot 0..2, `prediction` = the argmax slot).
* **arch_family** is the architecture: `DECAY` or `ATTN`, as `assert_family` returns it from a config
  and, for a state-space model, from its A values. It is never read from a name.

A cell may be a plain list of correctness bits instead; the guard, the AUROC and the threshold-free
curve are then "not recorded", the screen cannot pass (a constant-responder check nobody ran must not
admit a row to a family set), and the sheet says so. A flat `items` list of records
(`family, distance, item, label, prediction, logodds, control`) is accepted in place of `per_item`.
A bridge-v3 receipt is read with `--arm M1` from `per_item[arm]` and `controls_per_item[arm]` — the
fields decision Z4 asks v3 to add; a v3 receipt without them is refused with that sentence, never read
as zeros.

## What comes out, per model

| field | rule |
|---|---|
| `cells[family][D].acc_norm` | `(acc − chance) / (1 − chance)`, compare chance 1/2, recall chance 1/3; a constant responder under exact balance is exactly 0 |
| `cells[...].ci_lo, ci_hi` | 2.5th / 97.5th percentile of the paired item bootstrap (B = 2000, seed 20261011): one index resample per replicate, applied to every cell of every model, both families, all distances (red team M9) |
| `cells[...].guard_ok`, `auroc`, `c_auc` | compare yes-rate within [0.20, 0.80]; recall largest slot share ≤ 0.80; rank AUROC of the log-odds against the label and `c_auc = 2·AUROC − 1` |
| `screen` | at D0: both normalised accuracies ≥ the bar (0.40) and both guards passed, read on one half of the D0 items (stratified by label so each half keeps exact balance); the horizon's denominator c(D0) is read on the other half (M4) |
| `horizon[family]` | `tau = 0.5 · c(D0)`; the smallest D ≥ D0 at which the piecewise-linear interpolant of c in log2 D reaches tau. `bracket` is the grid pair the crossing sits in and is the measurement; `point` is the interpolated convention and prints second (m5). No grid value below tau: `CENSORED`, printed `> 1024 (none inside the tested range)`. c(D0) ≤ 0: `NOT_ESTIMABLE` |
| `horizon[...].ci_lo, ci_hi, f_cens` | percentiles of the finite replicate crossings and the censored fraction; `f_cens > 0.025` censors the upper end (`> 1024`); `f_cens > 0.5` prints the point censored. `native_point` is the crossing in the model's own tokens, interpolated from `config.native_D` means |
| `horizon.compare_auc` | the same fit on `c_auc(D)` (F1c): a horizon in c absent from c_auc reads `CALIBRATION_ARTEFACT_SUSPECTED` |
| `ratio_min[family]` | min over D > D0 of c(D) / c(D0), with its interval |
| `dissociation` | `ratio_min_compare − ratio_min_recall`, point and paired interval (F2): the number a family gate reads |
| `spearman_log2D` | the curve's monotonicity, printed; any adjacent rise above 0.05 flags `NONMONOTONE` |
| `flags` | `SCREEN_FAIL`, `COMPARE_ONLY_HORIZON`, `RECALL_ONLY_HORIZON`, `STORAGE_HORIZON` (recall's horizon at or before compare's), `NO_HORIZON_INSIDE_RANGE`, `HORIZON_NOT_ESTIMABLE`, `DEGENERATE@family:D c=… c_auc=…` (m8: both numbers), `NONMONOTONE@family`, `LOGODDS_ABSENT`, and the control flags |

Controls, when the receipt carries them (M5), with the print rules of this instrument — conventions a
PREREG may freeze, not bars of their own:

* `ablated` (codes replaced by a neutral sentence): max normalised accuracy over its cells; above 0.16
  reads `PROBE_READS_PRIORS`. Absent: `ABLATED_CONTROL_ABSENT`.
* `near` (a code planted 16 tokens before the far question): its ratio to c(D0) below 0.5 reads
  `FORMAT_DECAY@D`, below 0.8 `FORMAT_DRIFT@D`, else `FORMAT_DISTANCE_EXCLUDED`. Absent:
  `FORMAT_DISTANCE_NOT_EXCLUDED`, and the sheet prints that a compare-only drop may be format decay.
* `overwrite` (the far filler re-plants one name; the question asks about the original code, label
  true): compare must fall to chance — normalised ≤ 0.16 reads `POSITIVE_CONTROL_FIRED`, else
  `POSITIVE_CONTROL_SILENT@D` (the probe is not reading the far binding). Absent:
  `FAR_BINDING_CONTROL_ABSENT`.

## The sheet across models

`fit_zoo(receipts, exclude=[...])` fits every receipt on one shared index resample and aggregates one
architecture family (default ATTN). Its `result` is shaped for a gates block that declares its
composition (M2): every per-model value sits in `result.per_model.*` (rounded to six decimals, the
precision the quoted minimum uses), every excluded member in `result.family_excluded` with its reason,
so a gate can say `"agg": "min", "over": "result.per_model.dissoc", "excluding": "result.family_excluded"`
and `styxx.protocol` recomputes it. `family_set_ablated_max_norm_acc` is taken over the screened set
BEFORE the ablation exclusion (M1), so it can fail. `attention_family_min_dissoc_ci` is the interval
of the per-replicate minimum over the family set on the shared resample (M3, M9); receipts whose item
counts differ cannot share the stream, and the sheet then says the family interval is not paired.
With `--prereg` naming a COMMITTED prereg the sheet is scored through `styxx.protocol.Experiment`;
without one it prints `verdict: unscored: no committed prereg`.

## The self-test, and the refusal it gates

`selftest()` plants four crossings (32, 128, 203, 600) in synthetic bits whose curves are linear in
log2 D — so the planted number and the fitter's model agree by construction — and requires each
recovered within [0.8, 1.25] × the plant with the interval covering it and the order preserved; a
curve with no crossing must be censored with `f_cens > 0.975`; a constant responder under exact
balance must be exactly 0 at every distance in both families and read SCREEN_FAIL with the guard in
the reason; compare crossing with recall censored must read COMPARE_ONLY_HORIZON; two fits under one
seed must agree. `datasheet()` runs it and REFUSES to print when it fails. The design's own
illustration planted "128" with c = (0.9, 0.7, 0.4, 0.2); under the interpolation rule those four
values cross at about 203, which is why the plants are built from the rule and not from that sentence.

## What the number is not

* A horizon is a description of one curve under one frozen probe, one filler corpus and one few-shot
  format. It licenses no sentence about rotation, honesty, causation or ranking; every model differs
  from every other in more than its recurrence.
* `SCREEN_FAIL` reads: the model did not perform this format at D0. Not "cannot compare".
* `> 1024` means no crossing inside the tested range, which ends at the last grid point and is under
  1% of context for some models. The word is "inside the tested range", never "inside context".
* The interval carries sampling noise only; the four-point grid's model-form error is why the
  bracketing pair prints before the interpolated point.
* A chance-normalised 0 is a constant responder or chance, not an absence of knowledge.
* Without a near-fact control, a compare-only drop may be format decay (the exemplars recede with D
  and compare is scored at a fixed threshold) rather than lost binding; without an overwrite control, a
  censored compare horizon is not evidence that the far binding was read.
* Nominal D is in NeoX tokens; the native token count per cell is printed where the receipt records it.
* The self-test is a software test on synthetic bits: it shows the fitter recovers a planted crossing,
  not that any model has one.

## Prior art the measurement sits beside

Zoology (Arora et al., 2023), Lost in the Middle (Liu et al., 2023), Repeat After Me (Jelassi et al.,
2024), RULER (Hsieh et al., 2024), Waleffe et al. (2024), BABILong (Kuratov et al., 2024), DeciMamba
(Ben-Kish et al., 2024), Stuffed Mamba (Chen et al., 2024), Calibrate Before Use (Zhao et al., 2021).
The lab claims the measurement under its discipline — a compare-minus-recall contrast with exact
balance, a degeneracy guard, chance normalisation and a censored interval — not the phenomenon.

## CLI

```
python -m styxx.horizon selftest [--json selftest.json]
python -m styxx.horizon fit   --receipt horizon_<model>.json [--json row.json]
python -m styxx.horizon fit   --receipt realmodel_bridge_v3_result.json --arm M1
python -m styxx.horizon sheet --receipts <dir or files> [--exclude T] [--prereg PREREG.md] [--json sheet.json]
python -m styxx.horizon family --config <snapshot>/config.json [--safetensors <snapshot>/*.safetensors]
```

`family` reads only the `*.A_log` tensors named in a safetensors header (no torch, no model load) and
refuses a bare name, a config without `model_type`, a state-space config without A values, an A that is
not real and strictly negative in every layer, an attention layer that reaches fewer than 1,250 tokens,
and any type it does not know. Exit 0 on a datasheet, 2 on a refusal.

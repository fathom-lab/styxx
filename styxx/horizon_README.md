# styxx.horizon — the horizon datasheet

Where, if anywhere inside the tested range, does a causal language model stop performing a judgement
over distant planted facts while it still recalls the facts themselves? `styxx.horizon` is the
instrument that answers it from per-item bits: no torch, no GPU, no model load, no network. This file is
the datasheet's reference; the module docstring is the contract, and `tests/test_horizon.py` is what
holds it.

Built from `next/horizon-zoo/DESIGN.md` (2026-10-11) with the fatal and major fixes of its red team that
live in the instrument applied. The fixes that live in the GPU runner and the item generator (F1a, F1b,
M5's items, M6, M7, M8) are not in this module; it reads their results when a receipt carries them and
prints their absence when it does not. The zoo run that would fill nine rows is not here and nothing in
this module changes what any committed RESULT says.

## What goes in

One receipt per model, in the module's own schema (`styxx.horizon/receipt/0.1`):

```json
{"schema": "styxx.horizon/receipt/0.1", "model": "state-spaces/mamba-130m-hf",
 "arch_family": {"family": "DECAY", "fact": {"A_real_and_strictly_negative_every_layer": true, "layers": 24}},
 "config": {"distances": [16, 64, 256, 1024], "D0": 16, "screen_bar": 0.40, "pairing": "items", "d0_split": true,
            "readout": "argmax_largest", "chance": {"compare": 0.3333333333333333, "recall": 0.3333333333333333},
            "guard": {"max_slot_fraction": 0.80},
            "bootstrap": {"n_resamples": 2000, "seed": 20261011},
            "native_D": {"16": {"min": 15, "mean": 15.6, "max": 18}, "1024": {"min": 990, "mean": 998.0, "max": 1010}},
            "dtype": "fp16", "params": "129M", "ctx": 2048, "ctx_is_training_length": true, "items_sha256": "..."},
 "per_item": {"compare": {"16": {"target_slot": [2, 0], "pred_slot": [2, 1], "scores": [[-1.2, -0.9, -0.3], [-0.4, -0.2, -1.1]]}},
              "recall":  {"16": {"target_slot": [0, 2], "pred_slot": [0, 1]}}},
 "controls": {"ablated": {"compare": {"16": {}}, "recall": {"16": {}}},
              "near": {"compare": {"1024": {}}}, "overwrite": {"compare": {"1024": {}}}}}
```

Two words that look alike and are not:

* **family** (of a cell) is the probe's word for its two operations, as `run_realmodel_bridge_v3.py`
  uses it: `compare` (a judgement over the planted codes) and `recall` (produce one planted code).
* **arch_family** is the architecture: `DECAY` or `ATTN`, as `assert_family` returns it from a config
  and, for a state-space model, from its A values. It is never read from a name.

### The readout is a fact of the receipt

A readout fixes the answer set, so it fixes chance (1/k under exact balance) and the guard. The
instrument scores these, and assumes none of them:

| readout | family | a cell carries | chance | guard |
|---|---|---|---|---|
| `argmax_largest` (bridge-v3 amendment 17.2) | compare | `target_slot` (the slot holding the largest planted code), `pred_slot` (argmax over the three name tokens), optional `scores` (three per item), `margin`, `correct` | 1/3 | largest slot share ≤ 0.80 |
| `yes_no`, `true_false` (the design's 4.1 probe; retired for bridge v3 by the same amendment) | compare | `label` (the claim is true), `prediction` (said yes), optional `logodds` | 1/2 | yes-rate within [0.20, 0.80] |
| `argmax_code` | recall | `target_slot` / `label` (the queried slot), `pred_slot` / `prediction`, optional `scores` | 1/3 | largest slot share ≤ 0.80 |

The readout is read from `config.readout` (or `config.readouts`, `{family: readout}`; the bridge-v3
runner also writes `config.answer_format`). Undeclared, it is inferred from the cell's own keys (three
scores per item or a slot index 2: argmax; log-odds or boolean labels: yes/no) and the row prints
"inferred"; a compare cell whose keys do not decide it is refused, because chance cannot be assumed. A
declared `config.chance` that is not 1/k of its readout is refused (the retired 1/2 on an argmax receipt
is the case that matters). `config.guard` (`{"yes_rate": [lo, hi], "max_slot_fraction": m}`) is read and
validated; values other than the bridge-v3 runner's read `NONSTANDARD_GUARD`. Every recorded field must
agree with every other: predictions with the sign of the log-odds or with the argmax of the scores (ties
to the lowest slot), `correct` with prediction == label, `margin` with the scores.

A cell may be a plain list of correctness bits under a declared readout; the guard, the AUROC and the
threshold-free curve are then "not recorded", the screen cannot pass (a constant-responder check nobody
ran must not admit a row to a family set), and the sheet says so. A flat `items` list of records
(`family, distance, item, control` and the cell keys) is accepted in place of `per_item`.

A receipt the reader cannot take is refused, never ended in a traceback: `config.D0`, the bootstrap's
`n_resamples` and `seed`, and a `native_D` distance must be integers (`seed` non-negative), `bootstrap`
and `seeds` must be objects, and anything else malformed is refused with the error it raised. `fit` and
`sheet` exit 2 on a refusal.

### The bridge-v3 runner's receipt, read by arm

`--arm M1` reads `per_item[arm][family][D][variant]` as the runner writes it from its amendment 17.2:
the `plain` variant is the curve; `ablated`, `near`, `overwrite` and `windowed` are the controls (with
`pred_is_overwrite`, the overwrite cell prints the fraction that follows the re-plant). `config.readout`,
`config.chance`, `config.guard` and `items.prefix_nested_across_distances` are read. The older shape
(`per_item[arm][family][D]` as the cell, with `controls_per_item[arm]`) is read too. A receipt without
`per_item[arm]` is refused with the sentence naming decision Z4, never read as zeros. A receipt whose grid
has one distance — the amendment-17.2 pilot, 48 items at D0 — has cells and no curve: `load_receipt(...,
min_grid=1)` reads the cells, and `fit` refuses the row with that sentence. The test file reads the
committed pilot receipt's per-item bits and reproduces, cell by cell, the accuracy, slot shares, guard,
one-versus-rest AUROC, margin and near-tie count the runner recorded beside them, and the screen it
applied; it is skipped where that receipt is not in the checkout.

### How the items pair

`config.pairing`: `items` (the items are prefix-nested across distances: one index resample per
replicate covers every cell), `distance` (the bridge-v3 runner's items: shared by the two families,
every variant and both arms at one distance, drawn afresh per distance — one resample per distance,
shared by the two families), or `independent` (every cell its own). The runner's
`items.prefix_nested_across_distances = false` declares `distance`. A receipt that declares nothing is
read as `items` and prints `pairing: items (assumed, undeclared)` with the flag `PAIRING_ASSUMED`.
Models fitted together on one seed share their resamples (red team M9).

## What comes out, per model

| field | rule |
|---|---|
| `cells[family][D].acc_norm` | `(acc − chance) / (1 − chance)` under the row's readout; a constant responder under exact balance is exactly 0 |
| `cells[...].exact_balance`, `label_counts` | the answer counts per cell; a cell out of exact balance reads `BALANCE_BROKEN@family:D`, a D0 half `BALANCE_BROKEN@family:D0/screen` or `/denominator`; a screen half out of balance also fails the screen, and a denominator half out of balance is flagged only |
| `cells[...].ci_lo, ci_hi` | 2.5th / 97.5th percentile of the paired item bootstrap (B = 2000, seed 20261011); a resample count under 40 is refused, from a receipt or a call, because a 95% percentile interval needs one replicate expected in each 2.5% tail |
| `cells[...].guard_ok`, `auroc`, `c_auc` | the readout's guard; the rank AUROC of the scores (log-odds against the label, or the one-versus-rest AUROC averaged over the three slots, the runner's statistic) and `c_auc = 2·AUROC − 1` |
| `screen` | at D0: both normalised accuracies ≥ the bar (0.40), both guards passed, both cells in exact balance, read on one half of the D0 items (stratified by label so each half keeps exact balance); the horizon's denominator c(D0) is read on the other half (M4). The row prints `PASS@.40` / `FAIL@.40`; a bar other than 0.40 reads `NONSTANDARD_BAR@b`, and a bar outside (0, 1) is refused |
| `horizon[family]` | `tau = frac · c(D0)` with `frac` 0.5 (`config.horizon_frac`, refused outside (0, 1)); the smallest D ≥ D0 at which the piecewise-linear interpolant of c in log2 D reaches tau. `bracket` is the grid pair the crossing sits in and is the measurement; `point` is the interpolated convention and prints second (m5). No grid value below tau: `CENSORED`, printed `> 1024 (none inside the tested range)`. c(D0) ≤ 0: `NOT_ESTIMABLE` |
| `horizon[...].ci_lo, ci_hi, f_cens` | percentiles of the finite replicate crossings and the censored fraction; `f_cens > 0.025` censors the upper end (`> 1024`); `f_cens > 0.5` prints the point censored. `native_point` is the crossing in the model's own tokens, interpolated from `config.native_D` means |
| `horizon.compare_auc` | the same fit on `c_auc(D)` (F1c): a horizon in c absent from c_auc reads `CALIBRATION_ARTEFACT_SUSPECTED` — a drifting prior on one answer moves an accuracy and leaves a rank statistic alone |
| `ratio_min[family]` | min over D > D0 of c(D) / c(D0), with its interval |
| `dissociation` | `ratio_min_compare − ratio_min_recall`, point and paired interval (F2): the number a family gate reads |
| `spearman_log2D` | the curve's monotonicity, printed on the sheet; any adjacent rise above 0.05 flags `NONMONOTONE` |
| `flags` | `SCREEN_FAIL`, `COMPARE_ONLY_HORIZON`, `RECALL_ONLY_HORIZON`, `STORAGE_HORIZON` (recall's horizon at or before compare's), `NO_HORIZON_INSIDE_RANGE`, `HORIZON_NOT_ESTIMABLE`, `DEGENERATE@family:D c=… c_auc=…` (m8: both numbers), `NONMONOTONE@family`, `BALANCE_BROKEN@…`, `NONSTANDARD_BAR@b`, `NONSTANDARD_GUARD`, `PAIRING_ASSUMED`, `SCORES_ABSENT`, and the control flags |

**A row that fails the screen gets no horizon.** It keeps its curves and its D0 numbers, and every
horizon, `ratio_min` and the dissociation read "not estimable (screen failed)": the screen at D0 is the
row's positive control, and a model with none has no horizon to read (DESIGN 4.4, 4.7). The reasons print
under the row.

Controls, when the receipt carries them (M5), with the print rules of this instrument — conventions a
PREREG may freeze, not bars of their own:

* `ablated` (codes replaced by a neutral text): max normalised accuracy over its cells; above 0.16 reads
  `PROBE_READS_PRIORS`. Absent: `ABLATED_CONTROL_ABSENT`.
* `near` (the codes 16 tokens before the far question): its ratio to c(D0) below 0.5 reads
  `FORMAT_DECAY@D`, below 0.8 `FORMAT_DRIFT@D`, else `FORMAT_DISTANCE_EXCLUDED` (no ratio on a row that
  failed the screen). Absent: `FORMAT_DISTANCE_NOT_EXCLUDED`, and the sheet prints that a compare-only
  drop may be format decay.
* `overwrite` (the far filler re-plants one name): compare against the original answer must fall to
  chance — normalised ≤ 0.16 reads `POSITIVE_CONTROL_FIRED`, else `POSITIVE_CONTROL_SILENT@D` (the probe
  is not reading the far binding). Absent: `FAR_BINDING_CONTROL_ABSENT`.
* `windowed` (the runner's attention-window control): reported.

## The sheet across models

`fit_zoo(receipts, exclude=[...])` fits every receipt on one resample stream and aggregates one
architecture family (default ATTN). Its `result` is shaped for a gates block that declares its
composition (M2): every per-model value sits in `result.per_model.*` (rounded to six decimals, the
precision the quoted minimum uses), every excluded member in `result.family_excluded` with its reason,
so a gate can say `"agg": "min", "over": "result.per_model.dissoc", "excluding": "result.family_excluded"`
and `styxx.protocol` recomputes it. `family_set_ablated_max_norm_acc` is taken over the screened set
BEFORE the ablation exclusion (M1), so it can fail. `attention_family_min_dissoc_ci` is the interval of
the per-replicate minimum over the family set on the shared resample (M3, M9); receipts whose pairings
or item counts differ cannot share the stream, and the sheet then says the family interval is not
paired.

**An empty family set, and ABSTAIN.** The counts — `n_models`, `n_family_screened` (screened before the
ablation exclusion) and `n_attention_screen_pass` (the family set) — are finite in every case. The family
aggregates are None when the family set is empty, and `styxx.protocol` refuses them: an aggregate over an
empty population is not a measurement, and the scorer evaluates every gate before it walks the outcome
table. So a zoo prereg reaches ABSTAIN on an empty set — nine SCREEN_FAIL rows, the case
`NEXT_AFTER_V3.md` 3.4 expects — through a readiness gates block that reads the counts alone, scored
before the family block; a non-empty set below the count reaches ABSTAIN through the family block itself,
composition declared. Both are tested through `styxx.protocol.Experiment` on committed prereg files.

With `--prereg` naming a COMMITTED prereg the sheet is scored through `styxx.protocol.Experiment`, and a
refusal of the scorer is printed as a refusal of the sheet. The instrument cannot read a prereg's own
chance, guard, bar or pairing, so a scored sheet is held to the ones it restates: every row's compare
readout declared, every pairing declared, the bar 0.40 and the bridge-v3 guard, else the sheet is refused
before scoring. Without `--prereg` it prints `verdict: unscored: no committed prereg`.

## The self-test, and the refusal it gates

`selftest()` plants four crossings (32, 128, 203, 600) in synthetic bits whose curves are linear in
log2 D — so the planted number and the fitter's model agree by construction — and requires each
recovered within [0.8, 1.25] × the plant with the interval covering it and the order preserved. It also
requires what a crossing alone cannot show:

* **the level** — c(D0) and r(D0) recovered within 0.05 of the planted 0.9 (a curve read at half its
  height has the same crossing);
* **the interval** — over the six off-D0 cells of one fit at n = 600 with 2000 resamples, the mean
  percentile width within [0.85, 1.18] × the binomial 95% width of the same cell; and on 32 seeds at
  n = 300, with each item's correctness drawn rather than fixed (so the point moves from seed to seed as a
  measured one does), the plant inside the horizon interval at least 24 times (a 95% interval covers it
  about 95% of the time, a 50% one about half);
* **the D0 split as wired** (M4) — a D0 cell whose screen half is planted at 0.45 and whose denominator
  half at 0.90 reads 0.45 on the screen and 0.90 as c(D0), with its interval above 0.80;
* a curve with no crossing censored with `f_cens > 0.975`; a constant responder under exact balance
  exactly 0 at every distance in both families, under the slot readout and under the yes/no readout,
  reading SCREEN_FAIL with the guard in the reason; a row that fails the screen printing no horizon, ratio
  or dissociation though its curve crosses; compare crossing with recall censored reading
  COMPARE_ONLY_HORIZON; two fits under one seed agreeing.

`datasheet()` runs it and REFUSES to print when it fails. The test file breaks the normalisation, the
interval and the D0 split one at a time and asserts that the matching check fails. The design's own
illustration planted "128" with c = (0.9, 0.7, 0.4, 0.2); under the interpolation rule those four values
cross at about 203, which is why the plants are built from the rule and not from that sentence.

## What the number is not

* A horizon is a description of one curve under one frozen probe, one filler corpus, one few-shot format
  and one readout. It licenses no sentence about rotation, honesty, causation or ranking; every model
  differs from every other in more than its recurrence. A horizon under `argmax_largest` (which of three
  planted codes is largest: fact against fact) and one under `yes_no` (a claim against a fact) measure
  different operations, and the row says which.
* `SCREEN_FAIL` reads: the model did not perform this format at D0. Not "cannot compare".
* `> 1024` means no crossing inside the tested range, which ends at the last grid point and is under
  1% of context for some models. The word is "inside the tested range", never "inside context".
* The interval carries sampling noise only; the four-point grid's model-form error is why the
  bracketing pair prints before the interpolated point.
* A chance-normalised 0 is a constant responder or chance, not an absence of knowledge.
* Without a near-fact control, a compare-only drop may be format decay (the exemplars recede with D)
  rather than lost binding; without an overwrite control, a censored compare horizon is not evidence that
  the far binding was read.
* Nominal D is in NeoX tokens; the native token count per cell is printed where the receipt records it.
* The self-test is a software test on synthetic bits: it shows the fitter recovers a planted crossing, a
  planted level, a nominal interval and the D0 split it claims, not that any model has a horizon.

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
python -m styxx.horizon fit   --receipt <bridge-v3 receipt>.json --arm M1
python -m styxx.horizon sheet --receipts <dir or files> [--exclude T] [--prereg PREREG.md] [--json sheet.json]
python -m styxx.horizon family --config <snapshot>/config.json [--safetensors <snapshot>/*.safetensors]
```

`family` reads only the `*.A_log` tensors named in a safetensors header (no torch, no model load) and
refuses a bare name, a config without `model_type`, a state-space config without A values, an A that is
not real and strictly negative in every layer, an attention layer that reaches fewer than 1,250 tokens,
and any type it does not know. Exit 0 on a datasheet, 2 on a refusal.

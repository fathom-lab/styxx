# -*- coding: utf-8 -*-
"""
Calibrated tool-call drift detection weights v1 — cognometric instrument #3.

The third calibrated instrument in the cognometry suite. Detects when
an LLM agent's stated-intent NL turn does not match the tool call it
actually made (wrong tool, wrong args, missing required fields,
hallucinated spurious fields, a call when a refusal was appropriate,
or a swap of values between two correctly-named argument slots).

v6.1 update (2026-04-24): adds `arg_order_inversion` — a positional
inversion signal between call-value first-appearance order in the
prompt vs. the schema's declared arg order. Targets the previously
documented arg_swap failure mode. Full re-train: 23 features.

Training corpus: BFCL v3 (Berkeley Function Calling Leaderboard), a
public benchmark of 5,500+ function-calling prompts with gold answers
and designated "irrelevance" splits (cases the model should refuse).

Dataset construction (scripts/drift_build_dataset_v0.py):
  - Positives (no drift): BFCL gold answers from simple + live_simple
  - Negatives (drift) — mutation-based:
      * arg_swap: swap values between two args
      * arg_drop: remove a required argument
      * spurious_arg: add a non-schema argument
      * tool_rename: rename to a different available tool
  - Negatives (drift) — natural:
      * irrelevance + live_irrelevance splits where ANY call counts
        as drift (model should have refused)
  Total: n=3,700 samples, 18%/82% class balance.

5-fold stratified CV results (scripts/drift_calibrated_v1.py):
  Mean AUC           0.943 +/- 0.009   (v6.0 baseline: 0.915 +/- 0.004)
  Pooled AUC         0.943              (v6.0 baseline: 0.915)
  Improvement vs best null heuristic (schema_conformance 0.733): +0.210

Per-drift-type held-out AUC (vs gold negatives):
  arg_drop             0.997   (v6.0: 0.998 — flat)
  spurious_arg         0.997   (v6.0: 0.997 — flat)
  irrelevance_called   0.980   (v6.0: 0.957 — +0.023)
  arg_swap             0.755   (v6.0: 0.664 — +0.091, targeted fix)
  tool_rename          0.377   (n=1, noise only)

Per-source AUC (pooled CV predictions, drift-labeled splits only):
  simple          0.930   (v6.0: 0.902 — +0.028)
  live_simple     0.904   (v6.0: 0.872 — +0.032)
  irrelevance and live_irrelevance are single-class (all drift),
  so AUC is undefined — reported as positive-only recall instead.

Only published comparable: Healy et al. 2026 ("Internal Representations
as Indicators of Hallucinations in Agent Tool Selection," arXiv:2601.
05214) reports AUC 0.716-0.721 on Glaive 2,411 samples using
hidden-state features. Our 0.943 on BFCL v3 with text-only 23-feature
LR clears the only published baseline while being black-box compatible
(works on any closed model - OpenAI, Anthropic, Gemini).

Documented remaining failure modes:
  1. arg_swap at 0.76 (up from 0.66 via arg_order_inversion but still
     well below the other instruments). arg_order is a surface-level
     positional heuristic — cases where both swapped values have
     identical prompt positions (e.g., numerical ambiguity) or where
     one value is missing from the prompt still escape. Fix target v3:
     embedding-based arg-value semantic fit per slot.
  2. Low-sample drift types: tool_rename is not evaluable on BFCL
     because the benchmark gives each sample only one tool. A dataset
     with richer tool sets is needed for this failure class.

License: MIT (same as styxx).
"""
from __future__ import annotations

import math
from typing import Dict, List


FEATURE_NAMES: List[str] = [
    # Group A — semantic alignment (5)
    "tool_in_prompt",
    "tool_parts_in_prompt",
    "overlap_jaccard",
    "prompt_coverage",
    "arg_verbatim_rate",
    # Group B — schema conformance (7, +1 for arg_order_inversion)
    "tool_in_schema",
    "missing_required_frac",
    "spurious_arg_frac",
    "type_mismatch_frac",
    "arg_count_zscore",
    "required_count",
    "arg_order_inversion",
    # Group C — lexical drift (4)
    "placeholder_frac",
    "tool_name_len",
    "tool_in_any_schema",
    "n_available_tools",
    # Group D — structural (7)
    "n_args_called",
    "prompt_len",
    "avg_arg_len",
    "has_nested",
    "has_list",
    "prompt_is_question",
    "prompt_imperative",
]

COEFS: List[float] = [
    # Group A
    0.163189, -0.367107, -0.089927, -0.521194, -0.299258,
    # Group B
    1.019940, 2.739120, 6.239703, 0.951418, -3.449549, 0.197091, 1.153820,
    # Group C
    1.189813, -0.028464, 1.019940, 2.109481,
    # Group D
    -0.180930, -0.307224, 0.245030, -0.247955, -0.191283, 0.029412, -0.133528,
]

INTERCEPT: float = 4.890104

SCALER_MEAN: List[float] = [
    0.012703, 0.296847, 0.161882, 0.216750, 0.668113,
    0.997838, 0.124037, 0.076834, 0.095966, -0.167696, 1.866486, 0.315721,
    0.059144, 2.885012, 0.997838, 0.132203,
    2.801622, 4.260441, 1.897665, 0.017297, 0.088919, 0.395676, 0.592703,
]

SCALER_SCALE: List[float] = [
    0.111988, 0.356630, 0.160908, 0.216807, 0.365821,
    0.046449, 0.271505, 0.186539, 0.200740, 0.671110, 1.026151, 0.440333,
    0.217049, 0.408376, 0.046449, 0.422215,
    1.253362, 0.813532, 0.813720, 0.130377, 0.284627, 0.488995, 0.491331,
]

DEFAULT_DRIFT_THRESHOLD: float = 0.5

_SCALED_Z_CLIP: float = 5.0


HELD_OUT_AUC_PER_SOURCE: Dict[str, float] = {
    "simple":      0.9302,
    "live_simple": 0.9042,
}

HELD_OUT_AUC_PER_DRIFT_TYPE: Dict[str, float] = {
    "arg_drop":           0.9970,
    "spurious_arg":       0.9969,
    "irrelevance_called": 0.9802,
    "arg_swap":           0.7550,
    "tool_rename":        0.3769,   # n=1, noise
}

CV_MEAN_AUC: float = 0.9430
CV_STD_AUC:  float = 0.0091
POOLED_AUC:  float = 0.9425

CALIBRATION_NOTES: Dict = {
    "version": "v1 (6.1 retrain)",
    "train_source": "BFCL v3 (Berkeley Function Calling Leaderboard, Patil et al.)",
    "train_corpus": "simple + live_simple (positives + 4 mutation types) + irrelevance + live_irrelevance",
    "train_n": 3700,
    "class_balance_no_drift_drift": [658, 3042],
    "cv_protocol": "5-fold stratified, random_state=0",
    "cv_mean_auc": CV_MEAN_AUC,
    "cv_std_auc": CV_STD_AUC,
    "pooled_auc": POOLED_AUC,
    "null_baseline_auc": 0.7333,
    "improvement_over_null": 0.2097,
    "class_weight": "balanced",  # corrects for 82/18 drift/no-drift imbalance
    "per_source_auc": HELD_OUT_AUC_PER_SOURCE,
    "per_drift_type_auc": HELD_OUT_AUC_PER_DRIFT_TYPE,
    "documented_failure_modes": ["arg_swap_partial", "tool_rename_undersampled"],
    "failure_mode_notes": (
        "arg_swap at 0.76 (up from 0.66 in v6.0 via arg_order_inversion) "
        "but still below other instruments. arg_order is a surface-level "
        "positional heuristic — cases where both swapped values have "
        "identical prompt positions or where one value is missing from "
        "the prompt still escape. Fix target v3: embedding-based "
        "arg-value semantic fit per slot."
    ),
    "closest_published_baseline": {
        "name": "Healy et al. 2026",
        "arxiv": "2601.05214",
        "benchmark": "Glaive 2,411 samples",
        "auc": "0.716-0.721",
        "method": "Final-layer hidden-state features + MLP",
        "styxx_advantage": (
            "styxx runs text-only (works on any closed model) at AUC "
            "0.943 on BFCL v3 — requires no model internal access."
        ),
    },
    "methodology_parity_with_v4_and_refusal_v1": (
        "Same recipe as hallucination v4 and refusal v1: calibrated LR "
        "over engineered features, held-out cross-validation, failure "
        "modes published openly, versioned weights modules."
    ),
    "v6_1_changelog": (
        "Added arg_order_inversion (23rd feature): positional inversion "
        "between call-value first-appearance order in prompt and schema "
        "arg key order. Coef +1.154 (6th-largest magnitude), fully "
        "retrained scaler + LR weights. Lift: overall +0.028 AUC, "
        "arg_swap +0.091."
    ),
}


def predict_proba_drift(features: Dict[str, float]) -> float:
    """Calibrated drift probability for a feature dict.

    Includes defensive z-score clipping at |z| <= 5 per feature — prevents
    extreme out-of-distribution feature values from dominating the
    decision when the StandardScaler scale is small.

    Args:
        features: dict mapping FEATURE_NAMES to their computed values.
                  Missing keys default to 0.0 (permissive).

    Returns:
        float in [0, 1] — the probability the tool call drifts from
        the stated prompt intent.
    """
    logit = INTERCEPT
    for i, name in enumerate(FEATURE_NAMES):
        raw = float(features.get(name, 0.0))
        scale = SCALER_SCALE[i] if SCALER_SCALE[i] > 0 else 1.0
        scaled = (raw - SCALER_MEAN[i]) / scale
        if scaled > _SCALED_Z_CLIP:
            scaled = _SCALED_Z_CLIP
        elif scaled < -_SCALED_Z_CLIP:
            scaled = -_SCALED_Z_CLIP
        logit += scaled * COEFS[i]
    return 1.0 / (1.0 + math.exp(-logit))


__all__ = [
    "FEATURE_NAMES",
    "COEFS",
    "INTERCEPT",
    "SCALER_MEAN",
    "SCALER_SCALE",
    "DEFAULT_DRIFT_THRESHOLD",
    "HELD_OUT_AUC_PER_SOURCE",
    "HELD_OUT_AUC_PER_DRIFT_TYPE",
    "CV_MEAN_AUC",
    "CV_STD_AUC",
    "POOLED_AUC",
    "CALIBRATION_NOTES",
    "predict_proba_drift",
]

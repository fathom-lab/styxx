# -*- coding: utf-8 -*-
"""
Calibrated tool-call drift detection weights v1 — cognometric instrument #3.

The third calibrated instrument in the cognometry suite. Detects when
an LLM agent's stated-intent NL turn does not match the tool call it
actually made (wrong tool, wrong args, missing required fields,
hallucinated spurious fields, or a call when a refusal was appropriate).

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

5-fold stratified CV results (scripts/drift_calibrated_v0.py):
  Mean AUC           0.916 +/- 0.004
  Pooled AUC         0.916
  Improvement vs best null heuristic (schema_conformance 0.733): +0.183

Per-drift-type held-out AUC (vs gold negatives):
  arg_drop             0.998
  spurious_arg         0.997
  irrelevance_called   0.957    <-- hardest null case (0.56) crushed
  arg_swap             0.664    <-- documented failure mode
  tool_rename          0.030    <-- near-zero samples in BFCL, noise only

Per-source AUC (pooled CV predictions, drift-labeled splits only):
  simple          0.902
  live_simple     0.872
  irrelevance and live_irrelevance are single-class (all drift),
  so AUC is undefined — reported as positive-only recall instead.

Only published comparable: Healy et al. 2026 ("Internal Representations
as Indicators of Hallucinations in Agent Tool Selection," arXiv:2601.
05214) reports AUC 0.716-0.721 on Glaive 2,411 samples using
hidden-state features. Our 0.916 on BFCL v3 with text-only 22-feature
LR clears the only published baseline while being black-box compatible
(works on any closed model - OpenAI, Anthropic, Gemini).

Documented failure modes:
  1. arg_swap (AUC 0.66): semantically-wrong-but-syntactically-valid
     argument swapping. Our features measure schema conformance and
     prompt overlap but not type-consistent semantic substitution.
     Fix target v3: embedding-based arg-value similarity checks.
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
    # Group B — schema conformance (6)
    "tool_in_schema",
    "missing_required_frac",
    "spurious_arg_frac",
    "type_mismatch_frac",
    "arg_count_zscore",
    "required_count",
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
    0.152880, -0.410700, 0.112090, -0.887139, 0.196938,
    # Group B
    0.955546, 2.311910, 6.017625, 0.787727, -3.504748, 0.283883,
    # Group C
    1.200139, 0.004174, 0.955546, 2.207348,
    # Group D
    -0.088026, -0.410870, 0.245772, -0.199682, -0.117701, 0.102739, -0.038810,
]

INTERCEPT: float = 4.506965

SCALER_MEAN: List[float] = [
    0.012703, 0.296847, 0.161882, 0.216750, 0.668113,
    0.997838, 0.124037, 0.076834, 0.095966, -0.167696, 1.866486,
    0.059144, 2.885012, 0.997838, 0.132203,
    2.801622, 4.260441, 1.897665, 0.017297, 0.088919, 0.395676, 0.592703,
]

SCALER_SCALE: List[float] = [
    0.111988, 0.356630, 0.160908, 0.216807, 0.365821,
    0.046449, 0.271505, 0.186539, 0.200740, 0.671110, 1.026151,
    0.217049, 0.408376, 0.046449, 0.422215,
    1.253362, 0.813532, 0.813720, 0.130377, 0.284627, 0.488995, 0.491331,
]

DEFAULT_DRIFT_THRESHOLD: float = 0.5

_SCALED_Z_CLIP: float = 5.0


HELD_OUT_AUC_PER_SOURCE: Dict[str, float] = {
    "simple":      0.9021,
    "live_simple": 0.8718,
}

HELD_OUT_AUC_PER_DRIFT_TYPE: Dict[str, float] = {
    "arg_drop":           0.9976,
    "spurious_arg":       0.9969,
    "irrelevance_called": 0.9567,
    "arg_swap":           0.6637,
    "tool_rename":        0.0304,   # n=1, noise
}

CV_MEAN_AUC: float = 0.9151
CV_STD_AUC:  float = 0.0039
POOLED_AUC:  float = 0.9148

CALIBRATION_NOTES: Dict = {
    "version": "v1",
    "train_source": "BFCL v3 (Berkeley Function Calling Leaderboard, Patil et al.)",
    "train_corpus": "simple + live_simple (positives + 4 mutation types) + irrelevance + live_irrelevance",
    "train_n": 3700,
    "class_balance_no_drift_drift": [658, 3042],
    "cv_protocol": "5-fold stratified, random_state=0",
    "cv_mean_auc": CV_MEAN_AUC,
    "cv_std_auc": CV_STD_AUC,
    "pooled_auc": POOLED_AUC,
    "null_baseline_auc": 0.7333,
    "improvement_over_null": 0.1818,
    "class_weight": "balanced",  # corrects for 82/18 drift/no-drift imbalance
    "per_source_auc": HELD_OUT_AUC_PER_SOURCE,
    "per_drift_type_auc": HELD_OUT_AUC_PER_DRIFT_TYPE,
    "documented_failure_modes": ["arg_swap", "tool_rename_undersampled"],
    "failure_mode_notes": (
        "arg_swap (AUC 0.66): semantically-wrong-but-syntactically-"
        "valid arg swaps. Our features measure schema conformance and "
        "prompt overlap but not type-consistent semantic substitution. "
        "Fix targeted v3 via embedding-based arg-value similarity."
    ),
    "closest_published_baseline": {
        "name": "Healy et al. 2026",
        "arxiv": "2601.05214",
        "benchmark": "Glaive 2,411 samples",
        "auc": "0.716-0.721",
        "method": "Final-layer hidden-state features + MLP",
        "styxx_advantage": (
            "styxx runs text-only (works on any closed model) at AUC "
            "0.916 on BFCL v3 — requires no model internal access."
        ),
    },
    "methodology_parity_with_v4_and_refusal_v1": (
        "Same recipe as hallucination v4 and refusal v1: calibrated LR "
        "over engineered features, held-out cross-validation, failure "
        "modes published openly, versioned weights modules."
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
        the stated intent.
    """
    z = INTERCEPT
    for i, name in enumerate(FEATURE_NAMES):
        raw = float(features.get(name, 0.0))
        scale = SCALER_SCALE[i] if SCALER_SCALE[i] > 0 else 1.0
        scaled = (raw - SCALER_MEAN[i]) / scale
        # Defensive clipping for OOD robustness
        if scaled > _SCALED_Z_CLIP:
            scaled = _SCALED_Z_CLIP
        elif scaled < -_SCALED_Z_CLIP:
            scaled = -_SCALED_Z_CLIP
        z += scaled * COEFS[i]
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0


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

# -*- coding: utf-8 -*-
"""
Calibrated conversation-loop detection weights v0 — fifth cognometric
instrument. Shipped within 48h of the position paper *Every Mind Leaves
Vitals* (Rodabaugh, 2026, DOI 10.5281/zenodo.19777921), the second
instrument under that paper's call for #4-#9.

Trained on n=200 paired multi-turn conversations generated from
gpt-4o-mini against 100 generic seed topics, sampled under contrasting
system prompts:

  loop      — "for each user message, give the same answer as before,
               just reworded slightly"
  progress  — "for each user message, build on your previous reply with
               new information, examples, or angles"

Each conversation = 4 agent turns under generic follow-up user prompts
("Hmm, can you elaborate?" / "Tell me more." / etc.). Featurization
operates on the agent turns alone (the user prompts are context, not
signal).

Held-out 5-fold CV AUC: **0.9995 ± 0.0010** on the pooled n=200 corpus.

Phase-transition signature (per the position paper's prediction)
----------------------------------------------------------------
Greedy forward feature selection finds critical_K=**1** on
`avg_pairwise_levenshtein` (Δ +0.4995). A single feature — the mean
normalized Levenshtein distance across all turn pairs — takes detection
from chance (AUC 0.500) to **0.9995**.

This makes **5-for-5** on cognometric instruments showing K=1
phase-transition signature under the same measurement protocol:

  | instrument        | critical feature          | Δ AUC at K=1 |
  | ----------------- | ------------------------- | ------------ |
  | hallucination v4  | trigram_novelty           | +0.4947      |
  | refusal v1        | starts_with_sorry         | +0.469       |
  | drift v6.0        | (per-class K=1 or K=2)    | +0.4973      |
  | sycophancy v0     | superlative_density       | +0.4354      |
  | conversation-loop | avg_pairwise_levenshtein  | +0.4995      |

The position paper invited the field to find a calibrated text-based
detector that doesn't phase-transition. We have not found one — and
the count of confirmations grows.

Documented failure modes
------------------------
1. Single-source training corpus. Both loop and progress conditions
   come from gpt-4o-mini under explicit prompt instructions. Real-world
   agent loops emerge from MODEL FAILURE on hard tasks, not from
   instruction. The detector trained here picks up STRUCTURAL
   repetition (Levenshtein, n-gram overlap), so it should generalize
   to natural loops, but the threshold may need recalibration.
   v1 priority: real BFCL-multi-turn agent traces with human-labeled
   loops, plus cross-model corpus (Claude, Llama, Mistral).

2. **Counter-intuitive `distinct_word_ratio` coefficient.** Intuition
   suggests loops have LOW distinct-word-ratio (less vocabulary), so
   the coefficient should be negative. We learned a *positive* +0.95
   coefficient. Explanation: gpt-4o-mini under the "rephrase
   the same point" instruction reaches for synonyms each turn — its
   distinct-word-ratio actually goes UP under loop because reworded
   answers explore lexical alternatives. Meanwhile, under "progress"
   the model sticks with consistent terminology and just adds new
   information on top, keeping vocabulary stable. The coefficient is
   honest to the corpus; it is likely INVERTED on natural-failure
   loops where the model is genuinely stuck and recycles the same
   words. v1 priority: validate sign on natural-loop data.

3. Single-turn detection is impossible. The runtime API
   short-circuits to risk=0.0 for inputs with fewer than 2 agent
   turns. Loops are a multi-turn phenomenon by definition.

4. Very short turns (<10 words each) may not have enough lexical
   substance to fire any of the cross-turn features cleanly. The
   `log_n_turns` covariate is included to give the detector a sense
   of conversation depth, but its learned coefficient is 0.000 on
   this corpus (no signal beyond the structural features).

5. Sequential dependence: the detector treats turns as a SET when
   computing pairwise features. It does not model the temporal
   structure (e.g., "the loop started at turn 3"). Per-turn-onset
   detection is a v1 enhancement.

Reproducer: `scripts/loop_train_v0.py` (seed=0, deterministic, resumable
cache). Per-turn-count ablation and natural-loop validation are v1
research artifacts.

License: MIT.
"""
from __future__ import annotations

import math
from typing import Dict, List


FEATURE_NAMES: List[str] = [
    "bigram_overlap_consecutive",
    "trigram_overlap_consecutive",
    "five_gram_repeat_count",
    "length_cv",
    "opener_repeat_rate",
    "distinct_word_ratio",
    "avg_pairwise_levenshtein",
    "max_pairwise_bigram_overlap",
    "log_n_turns",
]

# Logistic regression coefficients (scaled-feature space).
COEFS: List[float] = [
    +1.503278,    # bigram_overlap_consecutive       — POS: high overlap → loop
    +0.777594,    # trigram_overlap_consecutive      — POS
    -0.263563,    # five_gram_repeat_count           — NEG (corpus artifact)
    -0.395451,    # length_cv                        — NEG: low CV → loop
    +0.007457,    # opener_repeat_rate               — near-zero
    +0.948803,    # distinct_word_ratio              — POS (counter-intuitive,
                  #                                    documented in
                  #                                    CALIBRATION_NOTES)
    -3.139547,    # avg_pairwise_levenshtein         ← K=1 critical feature
                  #                                    NEG: low distance → loop
    +2.089392,    # max_pairwise_bigram_overlap      — POS
    +0.000000,    # log_n_turns                      — zero on this corpus
]

INTERCEPT: float = +2.521370

# StandardScaler params from the v0 training-set fit (n=200).
SCALER_MEAN: List[float] = [
    0.079984,    # bigram_overlap_consecutive
    0.033118,    # trigram_overlap_consecutive
    0.046492,    # five_gram_repeat_count
    0.083786,    # length_cv
    0.025000,    # opener_repeat_rate
    0.493039,    # distinct_word_ratio
    0.628633,    # avg_pairwise_levenshtein
    0.133012,    # max_pairwise_bigram_overlap
    1.386294,    # log_n_turns  (= log(4) — fixed across corpus)
]

SCALER_SCALE: List[float] = [
    0.067905,
    0.040125,
    0.095852,
    0.040755,
    0.108972,
    0.071859,
    0.125683,
    0.108155,
    1.000000,    # log_n_turns is constant on this corpus → identity scale
]


DEFAULT_LOOP_THRESHOLD: float = 0.5


HELD_OUT_FOLD_AUCS: List[float] = [
    1.0000,
    1.0000,
    0.9975,
    1.0000,
    1.0000,
]
MEAN_CV_AUC: float = 0.9995
STD_CV_AUC: float = 0.0010


# Per the calibration-fingerprint format (Cognometric Fingerprint
# Specification v1.0). substrate_K_var is null because v0 trains on a
# single source (gpt-4o-mini); per-substrate ablation across
# real-vs-synthetic loops is the v1 research item.
CALIBRATION_FINGERPRINT: Dict = {
    "instrument": "conversation-loop-v0",
    "n_features": 9,
    "baseline_auc": 0.9995,
    "critical_K": 1,
    "critical_feature": "avg_pairwise_levenshtein",
    "delta_auc_at_K": 0.4995,
    "auc_at_critical_K": 0.9995,
    "substrate_K_var": {
        "synthetic_paired_gpt4omini": {
            "critical_K": 1,
            "critical_feature": "avg_pairwise_levenshtein",
            "delta_auc_at_K": 0.4995,
            "auc_at_K": 0.9995,
            "n": 200,
            "n_turns_per_conversation": 4,
        },
        "natural_agent_failure_loops": None,   # v1 priority
        "BFCL_multi_turn_real":         None,   # v1 priority
    },
    "negative_lift": [
        {"K": 9, "feature": "length_cv", "delta": -0.0005,
         "context": "saturating; weakest contributor on a corpus already at AUC 1.0"},
    ],
}


CALIBRATION_NOTES: Dict = {
    "version": "v0",
    "instrument_number": 5,
    "instrument_called_in": (
        "Every Mind Leaves Vitals (Rodabaugh, 2026, "
        "DOI 10.5281/zenodo.19777921) — second instrument shipped under "
        "that paper's call for #4-#9. (Sycophancy v0 was first.)"
    ),
    "training_model": "gpt-4o-mini",
    "training_corpus": (
        "100 generic seed topics × 2 conditions × 4 agent turns each "
        "= 200 paired conversations. Conditions: 'loop' (rephrase prior "
        "answer) vs 'progress' (build on prior answer with new content). "
        "Same generic follow-up user prompts in both conditions."
    ),
    "train_n": 200,
    "balanced": True,
    "documented_failure_modes": [
        "single_source_training",                  # v1: cross-model + natural loops
        "distinct_word_ratio_sign_inversion",      # corpus-specific quirk
        "single_turn_short_circuit",               # by design — runtime returns 0.0
        "no_temporal_modeling",                    # treats turns as a set
        "very_short_turns_underfire",              # <10 words/turn loses signal
    ],
    "failure_mode_notes": (
        "v0 captures STRUCTURAL repetition in agent turns — Levenshtein "
        "similarity, bigram/trigram overlap, opener repetition, length "
        "uniformity. The corpus is synthetic-paired (prompt-induced loop "
        "vs prompt-induced progress), so the detector picks up the "
        "structural signature, but the threshold may need recalibration "
        "for natural agent-failure loops where the model is genuinely "
        "stuck rather than instructed to repeat. v1 priority: real "
        "BFCL-multi-turn traces with human-labeled loops, plus a "
        "cross-model corpus expansion (Claude, Llama, Mistral)."
    ),
    "phase_transition_replication": (
        "Greedy forward feature selection finds critical_K=1 on "
        "avg_pairwise_levenshtein. AUC jumps from chance (0.500) to "
        "0.9995 in a single feature (Δ +0.4995). This makes 5-for-5 on "
        "cognometric instruments showing K=1 phase-transition signature "
        "under the same measurement protocol: hallucination v4 "
        "(trigram_novelty), refusal v1 (starts_with_sorry), drift v6.0 "
        "(per-class K=1-2), sycophancy v0 (superlative_density), and "
        "now conversation-loop v0 (avg_pairwise_levenshtein). The "
        "structural prediction from *Every Mind Leaves Vitals* "
        "continues to hold on every new instrument shipped under its "
        "measurement protocol."
    ),
    "prior_art_context": (
        "Conversation-loop / repetitive-output detection is a known "
        "concern in agentic LLM literature (Liu et al., AgentBench 2023; "
        "Yao et al., ReAct 2022 documented degradation on long horizons). "
        "We are aware of no calibrated text-only detector with a "
        "published calibration fingerprint for this specific failure "
        "mode. styxx v0 fills that gap."
    ),
    "reproducer": "scripts/loop_train_v0.py (seed=0, deterministic)",
    "v1_roadmap": [
        "Real BFCL-multi-turn agent traces with human-labeled loops",
        "Cross-model corpus (Claude, Llama, Mistral)",
        "Per-turn-onset detection (when did the loop start)",
        "Independent test set held out from training-prompt distribution",
        "Validate distinct_word_ratio coefficient sign on natural loops",
    ],
}


_SCALED_Z_CLIP: float = 3.0


def predict_proba_loop(features: Dict[str, float]) -> float:
    """Calibrated loop-probability for a feature dict (v0).

    Defensive z-score clipping at |z| <= 3 per feature — same pattern
    as refusal v2 and sycophancy v0.

    Args:
        features: dict mapping FEATURE_NAMES to their computed values.
                  Missing keys default to 0.0 (permissive).

    Returns:
        float in [0, 1] — the probability the conversation is in a loop.
    """
    z = INTERCEPT
    for i, name in enumerate(FEATURE_NAMES):
        raw = float(features.get(name, 0.0))
        scale = SCALER_SCALE[i] if SCALER_SCALE[i] > 0 else 1.0
        scaled = (raw - SCALER_MEAN[i]) / scale
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
    "DEFAULT_LOOP_THRESHOLD",
    "HELD_OUT_FOLD_AUCS",
    "MEAN_CV_AUC",
    "STD_CV_AUC",
    "CALIBRATION_FINGERPRINT",
    "CALIBRATION_NOTES",
    "predict_proba_loop",
]

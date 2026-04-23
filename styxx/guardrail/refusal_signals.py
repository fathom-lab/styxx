# -*- coding: utf-8 -*-
"""
Refusal-detection signals — text-only features for `refuse_check()`.

Mirror of the feature set used to train calibrated_weights_refusal_v1.
Pure Python, no external dependencies beyond the existing
`styxx.anthropic_hack.text_features` vocabularies. Safe to run in
Pyodide / browser contexts.

See `calibrated_weights_refusal_v1.py` for the research methodology
and held-out cross-model AUC numbers.
"""
from __future__ import annotations

import math
from typing import Dict, List

from ..anthropic_hack.text_features import extract_features


# --------------------------------------------------------------
# Phrase vocabularies — additive on top of text_features.REFUSAL_MARKERS
# --------------------------------------------------------------

# "As an AI..." / "As a language model..." self-referential framing
DISCLAIMER_PHRASES: List[str] = [
    "as an ai", "as a language model", "as an ai assistant",
    "as an ai language", "i'm an ai", "i am an ai",
    "it's important to note", "it is important to note",
    "it's worth noting", "please note that",
]

# Safety / harm / ethics framing
SAFETY_FLAG_PHRASES: List[str] = [
    "illegal", "unethical", "harmful", "dangerous", "violence",
    "harm to others", "against the law", "inappropriate content",
    "safety", "ethical",
]

# Deflection / "here's what you could do instead"
REDIRECT_PHRASES: List[str] = [
    "instead, i can", "instead i can", "however, i can",
    "however i can", "i can help with", "let me suggest",
    "i'd recommend", "i would recommend", "consider reaching out",
    "please consult", "speak to a professional",
]

# Normative / lecturing refusal patterns — Mistral-style
NORMATIVE_PHRASES: List[str] = [
    "it's important to note", "it is important to note",
    "it's important to understand", "it is important to understand",
    "it's crucial to", "it is crucial to",
    "it's essential to", "it is essential to",
    "never acceptable", "is never acceptable",
    "not acceptable", "is not acceptable",
    "please understand", "please consider",
    "must always", "should always respect",
    "proper consent", "respect the privacy",
    "healthy and respectful", "safe and respectful",
    "against the law", "legal and ethical",
]

# Starts-with opener patterns
NORMATIVE_OPENERS = (
    "it's important", "it is important",
    "it's crucial", "it is crucial",
    "it's essential", "it is essential",
    "first, it's", "first, it is",
    "please note", "please be aware",
    "first and foremost", "it should be noted",
)

SHORT_RESPONSE_WORD_THRESHOLD: int = 50


def _phrase_density(text: str, phrases: List[str]) -> float:
    """Hits of `phrases` in lowercased `text`, normalized by word count."""
    lt = text.lower()
    n_words = max(1, len(text.split()))
    return sum(1 for p in phrases if p in lt) / n_words


def extract_refusal_features(prompt: str, response: str) -> Dict[str, float]:
    """Compute the 18-feature refusal-detection vector.

    Features (order matches calibrated_weights_refusal_v1.FEATURE_NAMES):
      1.  refusal_density         (from text_features REFUSAL_MARKERS)
      2.  hedge_density
      3.  confidence_density
      4.  uncertainty_density
      5.  entity_density
      6.  claim_density
      7.  reasoning_marker_density
      8.  sentence_length_mean
      9.  unique_ratio
      10. disclaimer_density      (AI-identity framing)
      11. safety_flag_density     (safety/harm/ethics vocab)
      12. redirect_density        ("instead, I can...")
      13. normative_density       (moralizing / lecturing patterns)
      14. short_response          (< 50 words)
      15. starts_with_sorry       (apologetic opener)
      16. starts_with_i_cant      (direct refusal opener)
      17. starts_with_normative   (lecturing opener)
      18. log_word_count

    Args:
        prompt: the user prompt (currently unused but reserved for v2
                features that will compare prompt ↔ response topic drift).
        response: the model's text response.

    Returns:
        dict mapping each feature name to a float in [0, 1] or log space.
    """
    tf = extract_features(response)
    n_words = max(1, len(response.split()))
    lower = response.strip().lower()

    return {
        # From text_features (existing, well-tested vocabularies)
        "refusal_density":          float(tf.refusal_density),
        "hedge_density":            float(tf.hedge_density),
        "confidence_density":       float(tf.confidence_density),
        "uncertainty_density":      float(tf.uncertainty_density),
        "entity_density":           float(tf.entity_density),
        "claim_density":            float(tf.claim_density),
        "reasoning_marker_density": float(tf.reasoning_marker_density),
        "sentence_length_mean":     float(tf.sentence_length_mean),
        "unique_ratio":             float(tf.unique_ratio),

        # Refusal-specific additions
        "disclaimer_density":       _phrase_density(response, DISCLAIMER_PHRASES),
        "safety_flag_density":      _phrase_density(response, SAFETY_FLAG_PHRASES),
        "redirect_density":         _phrase_density(response, REDIRECT_PHRASES),
        "normative_density":        _phrase_density(response, NORMATIVE_PHRASES),
        "short_response":           1.0 if n_words < SHORT_RESPONSE_WORD_THRESHOLD else 0.0,
        "starts_with_sorry":        1.0 if lower.startswith(("i'm sorry", "i am sorry", "sorry")) else 0.0,
        "starts_with_i_cant":       1.0 if lower.startswith(("i can't", "i cannot", "i can not")) else 0.0,
        "starts_with_normative":    1.0 if lower.startswith(NORMATIVE_OPENERS) else 0.0,
        "log_word_count":           math.log(n_words),
    }


__all__ = [
    "DISCLAIMER_PHRASES",
    "SAFETY_FLAG_PHRASES",
    "REDIRECT_PHRASES",
    "NORMATIVE_PHRASES",
    "NORMATIVE_OPENERS",
    "SHORT_RESPONSE_WORD_THRESHOLD",
    "extract_refusal_features",
]

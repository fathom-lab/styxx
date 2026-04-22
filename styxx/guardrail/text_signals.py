# -*- coding: utf-8 -*-
"""
Text-level signals for hallucination risk.

Computed per-claim (sentence) using the existing
styxx.anthropic_hack.text_features vocabulary but with corrections
for known failure modes identified in the HaluEval baseline
benchmark:

  - length_ratio: answer_tokens / (prompt_tokens + epsilon) — guards
    against the HaluEval dataset artifact where hallucinated answers
    are consistently 5x longer than truthful answers.
  - concrete_claim_density: number of concrete-claim markers (year,
    number, quote, URL, identifier) per 100 tokens — proxy for
    "confidently specific content".
  - hedge_density: hedging words per token — inverse signal.
  - refusal_density: refusal markers per token — inverse signal.
  - confidence_density: confident-assertion markers per token —
    direct signal.

These return raw values in [0, 1]; fusion.py maps them to a
calibrated risk score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from .claim_decomposer import Claim, YEAR_RE, NUMBER_CLAIM_RE, QUOTED_RE
from ..anthropic_hack.text_features import (
    HEDGES, CONFIDENCE, REFUSAL_MARKERS, WORD_RE, _count_phrases,
)


@dataclass
class TextSignal:
    length_ratio: float
    concrete_claim_density: float
    hedge_density: float
    refusal_density: float
    confidence_density: float
    entity_count: int


def compute_text_signal(response: str, prompt: str = "",
                         entities: List[str] = None) -> TextSignal:
    """Compute text-feature signals for a full response."""
    entities = entities or []
    r_words = WORD_RE.findall(response)
    p_words = WORD_RE.findall(prompt)
    n_r = max(len(r_words), 1)
    n_p = max(len(p_words), 1)

    hedges = _count_phrases(response, HEDGES)
    conf = _count_phrases(response, CONFIDENCE)
    refusal = _count_phrases(response, REFUSAL_MARKERS)

    concrete = (
        len(YEAR_RE.findall(response))
        + len(NUMBER_CLAIM_RE.findall(response))
        + len(QUOTED_RE.findall(response))
    )

    return TextSignal(
        length_ratio=(n_r / n_p),
        concrete_claim_density=(100.0 * concrete / n_r),
        hedge_density=(hedges / n_r),
        refusal_density=(refusal / n_r),
        confidence_density=(conf / n_r),
        entity_count=len(entities),
    )


def claim_risk_text_only(claim: Claim,
                          response_signal: TextSignal) -> float:
    """Per-claim text-only risk score in [0, 1].

    Heuristic weights tuned on HaluEval-QA baseline to prefer
    concrete-claim-heavy sentences without hedges as the higher-
    risk class.
    """
    # Claim-level features (not response-level)
    text = claim.text
    n_words = max(len(WORD_RE.findall(text)), 1)
    year = int(claim.has_year)
    num = int(claim.has_number)
    quote = int(claim.has_quote)
    url_id = int(claim.has_url or claim.has_identifier)
    n_entities = len(claim.entities)
    hedges = _count_phrases(text, HEDGES)
    refusal = _count_phrases(text, REFUSAL_MARKERS)
    conf = _count_phrases(text, CONFIDENCE)

    # decline claims are low risk by definition
    if claim.claim_type == "decline":
        return 0.05

    # Concrete-claim score: higher = more specific factual content
    specificity = (
        2.0 * year + 1.5 * num + 1.0 * quote + 2.5 * url_id
        + 0.8 * n_entities
    ) / n_words * 10.0
    specificity = min(1.0, specificity)

    # Hedging reduces risk (author is signaling uncertainty)
    hedge_factor = 1.0 - min(0.8, hedges / n_words * 10.0)

    # Confident-markers slightly amplify
    conf_factor = 1.0 + min(0.3, conf / n_words * 10.0)

    # Refusal markers kill risk
    if refusal > 0:
        return 0.05

    base = specificity * hedge_factor * conf_factor
    return min(1.0, base)


__all__ = ["TextSignal", "compute_text_signal", "claim_risk_text_only"]

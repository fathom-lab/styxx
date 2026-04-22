# -*- coding: utf-8 -*-
"""
Knowledge-grounding signal: compare claim content against reference
knowledge when available.

When a guardrail call provides a `reference` (Wikipedia passage,
context document, retrieved evidence), we compute:

  1. Entity overlap: fraction of claim's named entities present
     in the reference (higher = more grounded).
  2. Number/year/quote overlap: fraction of claim's specific
     tokens (years, numbers, quoted strings) present in the
     reference.
  3. Content-word embedding similarity (optional, future v2).

Claims with LOW overlap against a reference that presumably
should contain them are flagged as likely fabricated.

v1 is rule-based and dependency-free; v2 could add semantic
similarity via sentence embeddings.
"""
from __future__ import annotations

import re
from typing import List, Set

from .claim_decomposer import Claim, YEAR_RE, NUMBER_CLAIM_RE, QUOTED_RE


def _tokens(text: str) -> Set[str]:
    """Lowercased content tokens."""
    return set(w.lower() for w in re.findall(r"\b\w+\b", text))


def claim_grounding_risk(claim: Claim, reference: str) -> float:
    """Return a grounding-risk score in [0, 1].

    0.0 = well-grounded (most content in reference)
    1.0 = not grounded (claim content absent from reference)

    Intended to complement entity-verification: entity-verify
    catches fictional entities; grounding catches false statements
    about real entities.
    """
    if not reference:
        return 0.5   # neutral — can't judge without reference

    ref_tokens = _tokens(reference)

    # Collect claim-specific tokens
    claim_text = claim.text
    years = set(YEAR_RE.findall(claim_text))
    numbers = set(NUMBER_CLAIM_RE.findall(claim_text))
    quotes = set(QUOTED_RE.findall(claim_text))

    # Content tokens excluding stopwords (rough)
    STOP = {
        "the", "a", "an", "is", "was", "were", "are", "be", "been",
        "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "and", "or", "but", "not", "no", "yes", "this", "that",
        "these", "those", "has", "have", "had", "will", "would",
        "could", "should", "may", "might", "can", "i", "you", "he",
        "she", "it", "we", "they", "as", "also", "more", "most",
        "some", "any", "all", "each", "many", "much", "which", "who",
        "what", "where", "when", "why", "how", "than", "then",
    }
    claim_content_tokens = {
        t for t in _tokens(claim_text) if t not in STOP
        and len(t) >= 3
    }

    # Grounded-fraction for content tokens, years, numbers, quotes
    def _coverage(items: Set[str]) -> float:
        if not items:
            return 1.0  # nothing to check = treated as grounded
        hits = sum(1 for x in items if x in ref_tokens
                   or (isinstance(x, str) and x.lower() in
                       reference.lower()))
        return hits / len(items)

    content_cov = _coverage(claim_content_tokens)
    year_cov = _coverage(years)
    number_cov = _coverage(numbers)
    # Quoted content: substring check on the quote itself (looser)
    quote_cov = 1.0
    if quotes:
        found = 0
        ref_l = reference.lower()
        for q in quotes:
            q_l = q.lower().strip()
            # Match substring in reference (allow partial)
            if q_l[:max(6, len(q_l) // 2)] in ref_l:
                found += 1
        quote_cov = found / len(quotes)

    # Weight: content coverage dominates; specific tokens (year/number/quote)
    # are also important because they're the most-fabricatable content.
    weighted = (
        0.5 * content_cov
        + 0.2 * year_cov
        + 0.15 * number_cov
        + 0.15 * quote_cov
    )
    # Risk = 1 - grounding
    return max(0.0, min(1.0, 1.0 - weighted))


def response_grounding_risk(claims: List[Claim], reference: str) -> float:
    """Mean grounding risk across all factual claims."""
    if not claims or not reference:
        return 0.5
    factual = [c for c in claims if c.claim_type == "factual"]
    if not factual:
        return 0.1   # nothing factual → low risk by default
    return sum(claim_grounding_risk(c, reference) for c in factual) / len(factual)


__all__ = ["claim_grounding_risk", "response_grounding_risk"]

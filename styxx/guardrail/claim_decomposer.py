# -*- coding: utf-8 -*-
"""
Decompose a response into atomic factual claims for per-claim scoring.

v1 is rule-based and dependency-free: sentence split + named-entity
extraction + claim-type classification. A future v2 could call a
small LLM to do the decomposition, but the rule-based approach is
fast, free, and captures the majority of testable claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

# Sentence boundary — simple but robust enough for most responses
SENT_RE = re.compile(r"[^.!?\n]+[.!?]+|[^.!?\n]+$")

# Named-entity extraction heuristics
YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NUMBER_CLAIM_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
QUOTED_RE = re.compile(r'"([^"]{4,})"')
# Proper-noun phrase: capitalized word, optionally followed by more
PROPER_NOUN_RE = re.compile(
    r"\b[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)*\b"
)
# URLs
URL_RE = re.compile(r"https?://\S+")
# DOIs / arxiv / ISBNs / arxiv-ids
IDENTIFIER_RE = re.compile(
    r"(?:doi:|arxiv:|isbn:|10\.\d{4,}|arXiv:\d{4}\.\d{4,})[a-zA-Z0-9./\-:]+",
    re.IGNORECASE,
)

# Sentences without NEs or numeric content are low-risk; skip them.
LOW_RISK_MARKERS = (
    "i don't", "i do not", "i cannot", "i can't", "i'm not sure",
    "i am not sure", "no record of", "not familiar",
)


@dataclass
class Claim:
    text: str
    start: int
    end: int
    claim_type: str   # "factual" | "opinion" | "decline" | "generic"
    entities: List[str] = field(default_factory=list)
    has_year: bool = False
    has_number: bool = False
    has_quote: bool = False
    has_url: bool = False
    has_identifier: bool = False


def _classify_claim(text: str) -> str:
    lower = text.lower()
    if any(m in lower for m in LOW_RISK_MARKERS):
        return "decline"
    if YEAR_RE.search(text) or NUMBER_CLAIM_RE.search(text) \
            or QUOTED_RE.search(text) or URL_RE.search(text) \
            or IDENTIFIER_RE.search(text):
        return "factual"
    # Proper noun presence → probably factual
    nouns = PROPER_NOUN_RE.findall(text)
    # Filter first-word capitalization (which is always capitalized)
    def _first_word_of(text: str) -> str:
        m = re.match(r"^\s*(\S+)", text)
        return m.group(1) if m else ""
    nouns = [n for n in nouns if n != _first_word_of(text)]
    if nouns:
        return "factual"
    # Default
    return "generic"


def _extract_entities(text: str) -> List[str]:
    nouns = PROPER_NOUN_RE.findall(text)
    first_word_m = re.match(r"^\s*(\S+)", text)
    first_word = first_word_m.group(1) if first_word_m else ""
    # Drop sentence-initial capitalizations; deduplicate; keep order
    seen: Set[str] = set()
    out = []
    for n in nouns:
        if n == first_word:
            continue
        if n in seen:
            continue
        # Filter common non-entity cap words
        if n.lower() in {"i", "the", "this", "that", "these", "those"}:
            continue
        seen.add(n)
        out.append(n)
    return out


def decompose(response: str) -> List[Claim]:
    """Split response into sentence-level claims with extracted metadata."""
    claims: List[Claim] = []
    for m in SENT_RE.finditer(response):
        sent = m.group(0).strip()
        if not sent:
            m.end()
            continue
        start = m.start()
        end = m.end()
        claims.append(Claim(
            text=sent,
            start=start,
            end=end,
            claim_type=_classify_claim(sent),
            entities=_extract_entities(sent),
            has_year=bool(YEAR_RE.search(sent)),
            has_number=bool(NUMBER_CLAIM_RE.search(sent)),
            has_quote=bool(QUOTED_RE.search(sent)),
            has_url=bool(URL_RE.search(sent)),
            has_identifier=bool(IDENTIFIER_RE.search(sent)),
        ))
    return claims


__all__ = ["Claim", "decompose"]

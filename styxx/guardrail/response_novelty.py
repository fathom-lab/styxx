# -*- coding: utf-8 -*-
"""
Response-novelty signals: asymmetric grounding signals that measure
what's in the response but *not* in the reference.

The existing knowledge_grounding signal measures claim-coverage
(how much of the claim's content appears in the reference). It
struggles when responses are short (dialog, summarization) because
a faithful response and a hallucinated response can have similar
coverage just due to response length.

The novelty signals ask the opposite question: how much of the
response was INVENTED (not supported by the reference)? A
hallucination adds content; a faithful response stays within the
reference. This turns out to be the discriminating signal on
dialog and summarization domains.

Three signals:
  - content_novelty  : fraction of non-stopword tokens in response
                        not found in reference
  - entity_novelty   : fraction of capitalized-word tokens in
                        response not found in reference (proxies
                        for unsupported named entities)
  - number_novelty   : fraction of numeric/date tokens in response
                        not in reference (proxies for fabricated
                        dates, statistics, counts)

All three in [0, 1]. Higher = more novel / less grounded.

Returns 0.0 (fully grounded) when reference is empty (fail-open).
"""
from __future__ import annotations

import re
from typing import Dict

# Basic stopwords — keeping it dependency-free.
_STOP = {
    "the", "a", "an", "is", "was", "were", "are", "be", "been",
    "being", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "and", "or", "but", "not", "no", "yes", "this", "that",
    "these", "those", "has", "have", "had", "will", "would",
    "could", "should", "may", "might", "can", "i", "you", "he",
    "she", "it", "we", "they", "them", "their", "his", "her",
    "its", "our", "your", "my", "me", "as", "also", "more", "most",
    "some", "any", "all", "each", "many", "much", "which", "who",
    "what", "where", "when", "why", "how", "than", "then", "so",
    "if", "else", "there", "here", "do", "does", "did", "done",
    "say", "says", "said", "tell", "tells", "told",
}

_WORD_RE = re.compile(r"\b[\w'-]+\b")
_NUMBER_RE = re.compile(r"\b(?:\d+(?:[.,]\d+)*|\d+(?:st|nd|rd|th))\b")


def _bigrams(tokens: list) -> set:
    out = set()
    for i in range(len(tokens) - 1):
        a, b = tokens[i].lower(), tokens[i + 1].lower()
        if a in _STOP or b in _STOP or len(a) < 2 or len(b) < 2:
            continue
        out.add(a + " " + b)
    return out


def _trigrams(tokens: list) -> set:
    out = set()
    for i in range(len(tokens) - 2):
        a, b, c = (tokens[i].lower(), tokens[i + 1].lower(),
                   tokens[i + 2].lower())
        if len(a) < 2 or len(b) < 2 or len(c) < 2:
            continue
        # keep trigrams with at least one content word
        if a in _STOP and b in _STOP and c in _STOP:
            continue
        out.add(a + " " + b + " " + c)
    return out


def _tokens(text: str) -> list:
    return [m.group(0) for m in _WORD_RE.finditer(text or "")]


def _content_tokens(text: str) -> set:
    return {
        t.lower() for t in _tokens(text)
        if t.lower() not in _STOP and len(t) >= 3
    }


def _capitalized_tokens(text: str) -> set:
    """Tokens starting with uppercase that aren't at sentence-start."""
    out = set()
    # split into sentences by period, treat first token of each
    # sentence separately (sentence-initial capitals aren't
    # necessarily proper nouns)
    for sent in re.split(r"(?<=[.!?])\s+", text or ""):
        toks = _tokens(sent)
        for i, t in enumerate(toks):
            if i == 0:
                continue  # skip sentence-initial
            if t[0:1].isupper() and t.lower() not in _STOP and len(t) >= 2:
                out.add(t.lower())
    # also take any token that's ALL_CAPS of length >= 3 (acronyms)
    for t in _tokens(text or ""):
        if len(t) >= 3 and t.isupper() and t.lower() not in _STOP:
            out.add(t.lower())
    return out


def _number_tokens(text: str) -> set:
    return set(_NUMBER_RE.findall(text or ""))


def response_novelty_signals(response: str, reference: str) -> Dict[str, float]:
    """Return a dict of novelty-based grounding signals.

    Fail-open: when reference is empty, every signal returns 0.0
    (no evidence of fabrication).

    Signals:
      - content_novelty   : fraction of response content tokens not
                             in reference
      - entity_novelty    : fraction of mid-sentence capitalized
                             tokens (proxy for proper nouns) of
                             length >=4 not in reference
      - number_novelty    : fraction of numeric tokens not in
                             reference
      - bigram_novelty    : fraction of response bigrams (excluding
                             stopwords) not in reference — captures
                             compound facts/claims the reference
                             doesn't mention
      - trigram_novelty   : same as bigram but stricter / more specific
    """
    if not response or not reference:
        return {
            "content_novelty": 0.0,
            "entity_novelty": 0.0,
            "number_novelty": 0.0,
            "bigram_novelty": 0.0,
            "trigram_novelty": 0.0,
        }

    ref_lower = reference.lower()
    ref_content = _content_tokens(reference)
    ref_tokens = _tokens(reference)
    ref_bigrams = _bigrams(ref_tokens)
    ref_trigrams = _trigrams(ref_tokens)

    resp_content = _content_tokens(response)
    resp_caps = {t for t in _capitalized_tokens(response) if len(t) >= 4}
    resp_nums = _number_tokens(response)
    resp_tokens = _tokens(response)
    resp_bigrams = _bigrams(resp_tokens)
    resp_trigrams = _trigrams(resp_tokens)

    # Content novelty
    if resp_content:
        novel_content = sum(1 for t in resp_content if t not in ref_content)
        content_nov = novel_content / len(resp_content)
    else:
        content_nov = 0.0

    # Entity novelty (length >= 4, avoids abbreviation false positives)
    if resp_caps:
        novel_ent = sum(
            1 for t in resp_caps
            if t not in ref_content and t not in ref_lower
        )
        entity_nov = novel_ent / len(resp_caps)
    else:
        entity_nov = 0.0

    # Number novelty
    if resp_nums:
        novel_nums = sum(1 for n in resp_nums if n not in reference)
        number_nov = novel_nums / len(resp_nums)
    else:
        number_nov = 0.0

    # Bigram novelty — catches compound claims
    if resp_bigrams:
        novel_bigrams = sum(
            1 for bg in resp_bigrams
            if bg not in ref_bigrams and bg not in ref_lower
        )
        bigram_nov = novel_bigrams / len(resp_bigrams)
    else:
        bigram_nov = 0.0

    # Trigram novelty — stricter
    if resp_trigrams:
        novel_trigrams = sum(
            1 for tg in resp_trigrams
            if tg not in ref_trigrams and tg not in ref_lower
        )
        trigram_nov = novel_trigrams / len(resp_trigrams)
    else:
        trigram_nov = 0.0

    return {
        "content_novelty": round(content_nov, 4),
        "entity_novelty": round(entity_nov, 4),
        "number_novelty": round(number_nov, 4),
        "bigram_novelty": round(bigram_nov, 4),
        "trigram_novelty": round(trigram_nov, 4),
    }


__all__ = ["response_novelty_signals"]

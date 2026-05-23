# -*- coding: utf-8 -*-
"""
Entity verification via Wikipedia.

For each named entity in a claim, we query Wikipedia's summary
endpoint. If the entity has a valid page, we mark it verified.
Entities with no page (or with disambiguation pages that don't
resolve) are flagged as potentially fabricated.

Wikipedia REST is free, rate-limited (~ 100/sec anonymous), and
does not require an API key. Responses are cached in-memory per
session to avoid repeat queries.

This is a heuristic — real entities occasionally lack Wikipedia
pages (obscure but real), and fictional entities can share names
with real ones. But at scale, the signal is strongly correlated
with factuality.
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request
from typing import Dict, List

WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

_VERIFICATION_CACHE: Dict[str, Dict] = {}
_RATE_LIMIT_SECONDS = 0.1   # conservative: 10 queries/sec
_last_query_time = [0.0]


def verify_entity(entity: str, timeout: float = 4.0) -> Dict:
    """Check an entity against Wikipedia.

    Returns a dict:
      {
        "entity": original entity string,
        "verified": bool,
        "title": canonical Wikipedia title if verified,
        "extract": first-paragraph summary,
        "confidence": float in [0, 1],
        "reason": explanation of non-verified cases,
      }
    """
    entity = entity.strip()
    if not entity:
        return _unverified(entity, "empty entity")

    if entity in _VERIFICATION_CACHE:
        return _VERIFICATION_CACHE[entity]

    # Rate limit
    elapsed = time.time() - _last_query_time[0]
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _last_query_time[0] = time.time()

    # Query
    try:
        url = WIKI_SUMMARY_URL.format(title=urllib.parse.quote(entity))
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "styxx-guardrail/1.0 "
                    "(+https://github.com/fathom-lab/styxx)"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.request.HTTPError as e:
        if e.code == 404:
            result = _unverified(entity, "no Wikipedia page")
        else:
            result = _unverified(entity, f"HTTP {e.code}")
        _VERIFICATION_CACHE[entity] = result
        return result
    except Exception as e:
        result = _unverified(entity, f"network error: {type(e).__name__}")
        _VERIFICATION_CACHE[entity] = result
        return result

    # Handle disambiguation pages (they exist but don't verify a single entity)
    page_type = data.get("type", "")
    if page_type == "disambiguation":
        result = {
            "entity": entity,
            "verified": False,
            "title": data.get("title"),
            "extract": data.get("extract", "")[:200],
            "confidence": 0.3,
            "reason": "disambiguation page — entity is ambiguous",
        }
        _VERIFICATION_CACHE[entity] = result
        return result

    # Verified
    result = {
        "entity": entity,
        "verified": True,
        "title": data.get("title"),
        "extract": data.get("extract", "")[:400],
        "confidence": 0.9,
        "reason": "wikipedia page exists",
    }
    _VERIFICATION_CACHE[entity] = result
    return result


def _unverified(entity: str, reason: str) -> Dict:
    return {
        "entity": entity,
        "verified": False,
        "title": None,
        "extract": "",
        "confidence": 0.0,
        "reason": reason,
    }


def verify_entities_batch(entities: List[str],
                           skip_common: bool = True) -> Dict[str, Dict]:
    """Verify a batch of entities. Returns entity → result dict."""
    common_words = {
        "God", "Earth", "Moon", "Sun", "USA", "UK", "EU",
        "Paris", "London", "Tokyo", "Berlin", "NYC",
    }
    results = {}
    for e in entities:
        if skip_common and e in common_words:
            results[e] = {
                "entity": e,
                "verified": True,
                "title": e,
                "extract": "(common entity)",
                "confidence": 1.0,
                "reason": "common entity shortcut",
            }
            continue
        results[e] = verify_entity(e)
    return results


def entity_unverified_fraction(entities: List[str]) -> float:
    """Fraction of entities that are not Wikipedia-verified.
    Used as a hallucination-risk signal: responses with many
    unverified named entities are higher-risk."""
    if not entities:
        return 0.0
    results = verify_entities_batch(entities)
    unverified = sum(1 for r in results.values() if not r["verified"])
    return unverified / len(entities)


__all__ = [
    "verify_entity",
    "verify_entities_batch",
    "entity_unverified_fraction",
]

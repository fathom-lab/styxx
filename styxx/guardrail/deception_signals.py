# -*- coding: utf-8 -*-
"""
Deception-detection signals — text-only features for `deception_check()`.

**Scope and warning.** This is NOT a lie detector. It is a calibrated
detector of *lexical signatures of instruction-induced dishonesty* —
trained on responses produced under contrasting "be honest" vs
"tell the user what they want to hear" system prompts. The features
draw on Pennebaker / Newman / Hauch deception-linguistics literature,
but applied to LLM output, not human speech. The detector picks up
the LANGUAGE PATTERNS that emerge when a model is instructed to
dissemble; it does not access ground truth, does not verify factual
correctness, and produces confident false positives on careful,
qualified writing (which legitimately uses hedges + specifics) and
confident false negatives on a model asserting a falsehood with full
confidence.

Use as a *signal* in agent-level monitoring (combined with @trust /
sycoph_check / refusal etc.). Do NOT use as a verdict on humans, do
NOT use as a polygraph substitute, do NOT deploy in adversarial
settings without further validation.

Pure Python, no embeddings, no model weights. Pyodide-safe.

Calibration substrate: paired (honest, dishonest) responses generated
from gpt-4o-mini under contrasting system prompts on factual / opinion
questions. See `scripts/deception_train_v0.py`.

Feature design rationale (Pennebaker / Newman / Hauch tradition,
adapted for LLM output)
------------------------------------------------------------------
Honest responses tend toward:
  - More SPECIFIC content (named entities, numbers, dates)
  - Higher first-person assertion density ("I think", "I believe")
  - More EXCLUSIVE words (qualifications: "but", "except", "only")
  - Cognitive complexity markers ("because", "consider")

Dishonest responses (under prompt instruction) tend toward:
  - Higher VAGUENESS / generic terms ("some", "many", "various")
  - More HEDGE+CONFIDENCE clash (uncertainty markers paired with
    overclaiming, e.g., "definitely seems likely")
  - More NEGATION (Newman "Lying Words")
  - Lower specificity (fewer named entities / numbers)
  - Tendency toward MORE words (compensating for lack of substance)

These are tendencies, not laws. The detector aggregates them via
calibrated logistic regression with documented uncertainty.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List


# ---------------------------------------------------------------- vocabularies


# First-person markers (count toward HONEST when present per Pennebaker)
FIRST_PERSON_TOKENS: List[str] = [
    "i", "me", "my", "mine", "myself",
]
FIRST_PERSON_PHRASES: List[str] = [
    "i think", "i believe", "i'm sure", "i am sure",
    "in my view", "in my opinion", "personally,",
    "from my perspective",
]

# Exclusive words (qualifications) — HONEST signal per Newman/Hauch
EXCLUSIVE_WORDS: List[str] = [
    "but", "except", "without", "exclude", "excluding",
    "only", "rather", "instead", "unless", "whereas",
    "however", "although", "though",
]

# Vagueness lexicon — DECEPTION signal
VAGUE_WORDS: List[str] = [
    "some", "many", "various", "several", "a few",
    "things", "stuff", "matters", "issues",
    "generally", "usually", "often", "typically",
    "people", "everyone", "anyone", "someone",
]

# Negation — DECEPTION signal per Newman "Lying Words"
NEGATION_WORDS: List[str] = [
    "not", "no", "never", "none", "nothing", "nobody",
    "neither", "nor", "isn't", "aren't", "wasn't", "weren't",
    "doesn't", "don't", "didn't", "won't", "wouldn't", "can't",
    "cannot", "couldn't",
]

# Hedge markers (uncertainty)
HEDGE_WORDS: List[str] = [
    "maybe", "perhaps", "might", "possibly", "could",
    "seems", "appears", "likely", "probably", "potentially",
    "presumably", "presumably,",
]

# Confidence / overclaiming markers
CONFIDENCE_WORDS: List[str] = [
    "definitely", "certainly", "clearly", "obviously",
    "undoubtedly", "absolutely", "exactly", "precisely",
    "always", "never", "everyone", "no one",
]

# Cognitive-complexity markers — HONEST signal
COGNITIVE_MARKERS: List[str] = [
    "because", "therefore", "thus", "hence",
    "since", "due to", "as a result", "consequently",
    "consider", "consideration", "weigh",
]


# ---------------------------------------------------------------- helpers


_TOKEN_RE = re.compile(r"[A-Za-z']+")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
# Capitalized words mid-sentence — proxy for proper nouns
# (we exclude sentence-initial caps to reduce noise)
_PROPER_NOUN_RE = re.compile(r"(?<![.?!]\s)(?<!^)\b[A-Z][a-z]{2,}\b")
_DATE_PROXY_RE = re.compile(
    r"\b(?:19|20)\d{2}\b|"
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\b|"
    r"\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
    re.IGNORECASE,
)


def _word_density(text: str, words: List[str]) -> float:
    """Density of single-token words from `words` in lowercased text."""
    lt = text.lower()
    toks = _TOKEN_RE.findall(lt)
    if not toks:
        return 0.0
    word_set = set(words)
    return sum(1 for t in toks if t in word_set) / len(toks)


def _phrase_density(text: str, phrases: List[str]) -> float:
    """Density of (multi-token) phrases in lowercased text."""
    lt = text.lower()
    n_words = max(1, len(text.split()))
    return sum(1 for p in phrases if p in lt) / n_words


def _specificity_density(text: str) -> float:
    """Count of named entities + numbers + date proxies / word count."""
    n_words = max(1, len(text.split()))
    nums = len(_NUMBER_RE.findall(text))
    dates = len(_DATE_PROXY_RE.findall(text))
    proper = len(_PROPER_NOUN_RE.findall(text))
    return (nums + dates + proper) / n_words


def _hedge_confidence_clash(text: str) -> float:
    """Returns 1.0 if BOTH a hedge and a confidence marker appear in the
    text; else 0.0. The CO-OCCURRENCE is the deception signal — hedging
    while overclaiming."""
    lt = text.lower()
    has_hedge = any(w in lt for w in HEDGE_WORDS)
    has_conf = any(w in lt for w in CONFIDENCE_WORDS)
    return 1.0 if (has_hedge and has_conf) else 0.0


def extract_deception_features(prompt: str, response: str) -> Dict[str, float]:
    """Compute the 9-feature deception-detection vector.

    Features (order matches calibrated_weights_deception_v0.FEATURE_NAMES):
      1. specificity_density        — entities + numbers + dates / words.
                                      HIGH = more specific = honest signal.
      2. first_person_density       — "I"/"me"/"my" tokens / words.
                                      HIGH = honest signal (Pennebaker).
      3. exclusive_word_density     — "but"/"except"/"only" / words.
                                      HIGH = honest signal (Newman).
      4. vagueness_density          — "some"/"many"/"things" / words.
                                      HIGH = deception signal.
      5. negation_density           — "not"/"never"/"no" / words.
                                      HIGH = deception signal (Newman
                                      "Lying Words").
      6. hedge_confidence_clash     — boolean: hedge AND confidence
                                      markers co-present.
                                      = 1 = deception signal.
      7. cognitive_marker_density   — "because"/"therefore" / words.
                                      HIGH = honest signal.
      8. opinion_phrase_density     — "I think"/"I believe" / words.
                                      HIGH = honest signal (Pennebaker).
      9. log_word_count             — covariate (longer responses
                                      sometimes correlate with deception
                                      via "compensating bulk").

    Args:
        prompt:   the user's prompt (currently unused but reserved for v1
                  features that compare prompt↔response stance).
        response: the model's text response.

    Returns:
        dict mapping each feature name to a float in [0, 1] or log space.
    """
    return {
        "specificity_density":      _specificity_density(response),
        "first_person_density":     _word_density(response, FIRST_PERSON_TOKENS),
        "exclusive_word_density":   _word_density(response, EXCLUSIVE_WORDS),
        "vagueness_density":        _word_density(response, VAGUE_WORDS),
        "negation_density":         _word_density(response, NEGATION_WORDS),
        "hedge_confidence_clash":   _hedge_confidence_clash(response),
        "cognitive_marker_density": _word_density(response, COGNITIVE_MARKERS),
        "opinion_phrase_density":   _phrase_density(response, FIRST_PERSON_PHRASES),
        "log_word_count":           math.log(max(1, len(response.split()))),
    }


__all__ = [
    "FIRST_PERSON_TOKENS",
    "FIRST_PERSON_PHRASES",
    "EXCLUSIVE_WORDS",
    "VAGUE_WORDS",
    "NEGATION_WORDS",
    "HEDGE_WORDS",
    "CONFIDENCE_WORDS",
    "COGNITIVE_MARKERS",
    "extract_deception_features",
]

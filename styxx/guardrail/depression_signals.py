# -*- coding: utf-8 -*-
"""
Depression-from-text features — EXPERIMENTAL feature extractor.

Status: research prototype, NOT a shipped instrument. This module provides
only the text feature set; there is **no calibrated weight file and no
`depression_check()` entry point** in the package, and nothing imports it.
Do not treat it as one of the calibrated cognometric instruments.

The (unrealized) intent was to probe whether the cognometric approach
extends from LLM text to biological-cognition text (e.g. Reddit
r/depression vs other-psychiatric subreddits) — the substrate-bridge idea
from *Every Mind Leaves Vitals* (§3, "The bridge to biological cognition").
That instrument was never trained or validated; treat the bridge as a
hypothesis, not a result.

Pure Python, no external dependencies. Features draw from established
computational-psychiatry markers:

  - First-person singular pronouns (Beck 1961; Rude/Gortner/Pennebaker 2004)
  - Absolutist words (Al-Mosaiwi & Johnstone 2018, ~50% higher in depressed
    online forums)
  - Negative-emotion vocabulary (LIWC NegEmo)
  - Death / hopelessness markers
  - Negation density
  - Cognitive-mechanism words (LIWC CogMech)
  - Sleep / somatic / body-state vocabulary
  - Self-vs-other reference ratio
  - Lexical diversity (type-token ratio)
  - Punctuation density (rumination via ellipses, question marks)
  - Hopelessness-phrase markers
  - Sentence-length distribution
  - Word-count baseline

No calibrated weights ship for these features. If a
`calibrated_weights_depression_v0` is ever trained and validated under the
same discipline as the LLM instruments (feature-engineered LR, K-ablation,
calibration fingerprint), this could become a wired instrument; until then
it is feature-extraction code only.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List

# --------------------------------------------------------------
# Vocabularies — drawn from computational-psychiatry literature
# --------------------------------------------------------------

# First-person singular markers — Beck (1961), Rude et al. (2004),
# Pennebaker (2011). Replicated finding: depressed individuals use
# "I", "me", "my", "myself", "mine" at elevated rates.
FIRST_PERSON_SING: List[str] = [
    "i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll",
    "im", "ive", "id",  # informal Reddit spellings
]

# Absolutist words — Al-Mosaiwi & Johnstone (2018), Clinical Psychological
# Science. Depression and anxiety online forums showed ~50% higher
# absolutist-word density than control forums; **stronger predictor than
# direct depression vocabulary** in their study.
ABSOLUTIST_WORDS: List[str] = [
    "always", "never", "totally", "completely", "entirely", "absolutely",
    "constantly", "every", "everything", "everyone", "everybody",
    "nothing", "nobody", "no one", "all", "must", "definitely",
    "perfectly", "whole", "ever", "forever", "any", "anything",
]

# Negative-emotion vocabulary — LIWC NegEmo subset, plus depression-
# specific terms. Sources: LIWC2015 dictionaries, Coppersmith et al. (2014)
# CLPsych Twitter depression markers.
NEG_EMOTION: List[str] = [
    "sad", "sadness", "lonely", "alone", "empty", "hopeless", "worthless",
    "useless", "miserable", "depressed", "depression", "anxious", "anxiety",
    "tired", "exhausted", "suffer", "suffering", "hurt", "hurting", "pain",
    "painful", "broken", "lost", "scared", "afraid", "fear", "guilt",
    "guilty", "shame", "ashamed", "stupid", "horrible", "terrible", "awful",
    "hate", "hated", "hating", "angry", "anger", "upset", "crying", "tears",
]

# Positive-emotion vocabulary (inverse marker — depressed text uses LESS)
POS_EMOTION: List[str] = [
    "happy", "happiness", "glad", "joy", "joyful", "excited", "excitement",
    "love", "loved", "loving", "wonderful", "amazing", "great", "fantastic",
    "awesome", "fun", "laugh", "smile", "smiled", "smiling", "hope",
    "hopeful", "grateful", "thankful", "blessed",
]

# Death / self-harm vocabulary
DEATH_VOCAB: List[str] = [
    "die", "dying", "death", "dead", "suicide", "suicidal", "kill",
    "killing", "killed", "end it", "ending it", "give up", "no point",
    "no reason", "not worth", "ending my", "want to die", "wish i was dead",
    "wish i were dead", "would be better", "nobody would care",
]

# Hopelessness phrases — multi-word patterns
HOPELESSNESS_PHRASES: List[str] = [
    "no point", "what's the point", "whats the point", "no reason",
    "give up", "giving up", "given up", "tired of", "sick of",
    "can't anymore", "cant anymore", "doesn't matter", "doesnt matter",
    "won't get better", "wont get better", "never get better",
    "always like this", "no future", "no way out",
]

# Negation markers
NEGATION_WORDS: List[str] = [
    "not", "no", "nothing", "nobody", "no one", "never", "without",
    "n't", "don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't",
    "shouldn't", "isn't", "aren't", "wasn't", "weren't", "haven't",
    "hasn't", "hadn't", "cannot", "can't",
]

# Cognitive-mechanism words — LIWC CogMech subset. Pennebaker (2011)
# notes elevated cognitive-process words in rumination/depression.
COGNITIVE_WORDS: List[str] = [
    "think", "thinking", "thought", "thoughts", "know", "knew", "knowing",
    "realize", "realized", "understand", "understood", "wonder", "wondered",
    "suppose", "supposed", "guess", "guessed", "feel", "felt", "feeling",
    "feelings", "remember", "remembered", "forget", "forgot", "consider",
    "considered", "imagine", "imagined", "believe", "believed",
]

# Sleep vocabulary — depression-associated insomnia/hypersomnia markers
SLEEP_VOCAB: List[str] = [
    "sleep", "sleeping", "asleep", "awake", "insomnia", "nightmare",
    "nightmares", "tired", "exhausted", "fatigue", "fatigued", "rest",
    "bed", "bedtime", "dream", "dreaming", "lay", "laying", "lying",
    "in bed", "out of bed", "can't sleep", "cant sleep", "couldn't sleep",
]

# Body-state / somatic concerns
SOMATIC_VOCAB: List[str] = [
    "body", "head", "stomach", "heart", "chest", "weight", "food", "eat",
    "eating", "ate", "hungry", "appetite", "sick", "ill", "ache", "aches",
    "headache", "sore", "hurts", "throat",
]

# Hedging / uncertainty
HEDGE_WORDS: List[str] = [
    "maybe", "perhaps", "kind of", "kinda", "sort of", "sorta",
    "i guess", "i suppose", "i think", "probably", "possibly", "might",
    "i don't know", "i dont know", "idk",
]

# Other-person reference (used to compute self-vs-other ratio)
OTHER_REFERENCE: List[str] = [
    "you", "your", "yours", "yourself", "we", "us", "our", "ours",
    "ourselves", "they", "them", "their", "theirs", "themselves",
    "he", "she", "him", "her", "his", "hers",
]


# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------

# Pre-compile a token-finder once at import time. We use a simple
# unicode-aware word regex so that contractions tokenize as single tokens
# (e.g. "i'm" stays one token), matching how the depression literature
# typically tokenizes.
_WORD_RE = re.compile(r"[a-z']+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    """Lowercase token list. Keeps contractions intact ("i'm", "don't")."""
    return _WORD_RE.findall(text.lower())


def _token_density(tokens: List[str], vocab: set) -> float:
    """Fraction of tokens that appear in `vocab`."""
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in vocab) / len(tokens)


def _phrase_density(text_lower: str, phrases: List[str], n_words: int) -> float:
    """Hits of multi-word phrases per word — same definition as
    refusal_signals._phrase_density."""
    return sum(1 for p in phrases if p in text_lower) / max(1, n_words)


# Precompute sets for fast membership tests
_FIRST_PERSON_SET = set(FIRST_PERSON_SING)
_ABSOLUTIST_SET = set(w for w in ABSOLUTIST_WORDS if " " not in w)
_NEG_EMOTION_SET = set(NEG_EMOTION)
_POS_EMOTION_SET = set(POS_EMOTION)
_NEGATION_SET = set(w for w in NEGATION_WORDS if " " not in w)
_COGNITIVE_SET = set(COGNITIVE_WORDS)
_SLEEP_SET = set(w for w in SLEEP_VOCAB if " " not in w)
_SOMATIC_SET = set(SOMATIC_VOCAB)
_OTHER_REF_SET = set(OTHER_REFERENCE)


# --------------------------------------------------------------
# Public API
# --------------------------------------------------------------

def extract_depression_features(text: str) -> Dict[str, float]:
    """Compute the depression-detection feature vector.

    Features (alphabetical for stable ordering):
      - absolutist_density        (Al-Mosaiwi & Johnstone 2018)
      - cognitive_density         (LIWC CogMech)
      - death_phrase_density      (multi-word death/end-it markers)
      - first_person_singular_density  (Beck 1961, Rude 2004)
      - hedge_density
      - hopelessness_phrase_density
      - lexical_diversity         (type-token ratio)
      - log_word_count
      - mean_sentence_length
      - neg_emotion_density       (LIWC NegEmo)
      - negation_density
      - other_reference_density   (you/we/they/he/she)
      - pos_emotion_density       (LIWC PosEmo, inverse marker)
      - punctuation_density       (rumination signal)
      - question_mark_density
      - self_vs_other_ratio       (1p_sing / (1p_sing + other_ref))
      - sleep_density
      - somatic_density
      - starts_with_i             (self-referential opener)

    Args:
        text: the response text to analyze.

    Returns:
        dict mapping feature name to a float (densities in [0, 1] or
        log-space / count features).
    """
    if not text:
        text = ""
    lower = text.lower()
    tokens = _tokenize(text)
    n_tokens = max(1, len(tokens))
    n_words = max(1, len(text.split()))

    # Sentence-level features
    # Split on .!? — crude but matches the tokenization used by
    # mean_sentence_length in refusal_signals.
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    n_sents = max(1, len(sentences))
    mean_sent_len = sum(len(s.split()) for s in sentences) / n_sents

    # Lexical diversity — type-token ratio over the lowercased token list.
    # Note: TTR is length-sensitive; we cap by sampling first 100 tokens
    # to make it comparable across short and long posts (a standard fix
    # used in Pennebaker's LIWC studies).
    sample = tokens[:100]
    ttr = (len(set(sample)) / len(sample)) if sample else 0.0

    # Punctuation density — periods, commas, ellipses, dashes per word.
    # Ellipses and dashes are rumination markers in some studies.
    punct_count = sum(1 for c in text if c in ".,;:—–-")
    q_count = text.count("?")

    # First-person singular vs other-reference for self-focus ratio
    n_first_person = sum(1 for t in tokens if t in _FIRST_PERSON_SET)
    n_other_ref = sum(1 for t in tokens if t in _OTHER_REF_SET)
    denom_self_other = n_first_person + n_other_ref
    self_other_ratio = (n_first_person / denom_self_other) if denom_self_other else 0.5

    # First word — does the post start with "I" / "i'm" / etc.?
    first_word = tokens[0] if tokens else ""
    starts_with_i = 1.0 if first_word in _FIRST_PERSON_SET else 0.0

    return {
        "absolutist_density":           _token_density(tokens, _ABSOLUTIST_SET),
        "cognitive_density":            _token_density(tokens, _COGNITIVE_SET),
        "death_phrase_density":         _phrase_density(lower, DEATH_VOCAB, n_words),
        "first_person_singular_density": n_first_person / n_tokens,
        "hedge_density":                _phrase_density(lower, HEDGE_WORDS, n_words),
        "hopelessness_phrase_density":  _phrase_density(lower, HOPELESSNESS_PHRASES, n_words),
        "lexical_diversity":            ttr,
        "log_word_count":               math.log(n_words),
        "mean_sentence_length":         mean_sent_len,
        "neg_emotion_density":          _token_density(tokens, _NEG_EMOTION_SET),
        "negation_density":             _token_density(tokens, _NEGATION_SET),
        "other_reference_density":      n_other_ref / n_tokens,
        "pos_emotion_density":          _token_density(tokens, _POS_EMOTION_SET),
        "punctuation_density":          punct_count / n_words,
        "question_mark_density":        q_count / n_words,
        "self_vs_other_ratio":          self_other_ratio,
        "sleep_density":                _token_density(tokens, _SLEEP_SET),
        "somatic_density":              _token_density(tokens, _SOMATIC_SET),
        "starts_with_i":                starts_with_i,
    }


__all__ = [
    "FIRST_PERSON_SING",
    "ABSOLUTIST_WORDS",
    "NEG_EMOTION",
    "POS_EMOTION",
    "DEATH_VOCAB",
    "HOPELESSNESS_PHRASES",
    "NEGATION_WORDS",
    "COGNITIVE_WORDS",
    "SLEEP_VOCAB",
    "SOMATIC_VOCAB",
    "HEDGE_WORDS",
    "OTHER_REFERENCE",
    "extract_depression_features",
]

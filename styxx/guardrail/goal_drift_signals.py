# -*- coding: utf-8 -*-
"""
Goal-drift detection signals — text-only multi-turn features for `goal_check()`.

Goal drift here means: an agent receives a goal at session start, then
across multiple subsequent turns of work, gradually drifts away from
that goal — chasing tangents, redefining what the task is, or losing
sight of the original objective. Distinct from tool-call drift (v1):
tool-call drift is a single-call schema mismatch; goal drift is a
multi-turn intent migration.

Sibling to conversation-loop (#5): both are multi-turn instruments,
but loop measures stagnation (the agent says the same thing turn
after turn) while goal drift measures dispersion (the agent moves
further from its anchor turn after turn).

Detection inputs are `(turns: List[str])` where `turns[0]` is the
goal-statement turn and `turns[1:]` are subsequent agent
actions / observations / restated goals. Featurization is
multi-turn anchor-relative (compare each later turn's content to
turn 0 — the goal anchor).

Pure Python, no embeddings, no model weights. Pyodide-safe.

Calibration substrate: paired (anchored, drifted) multi-turn
sessions sampled from gpt-4o-mini under contrasting STANCE-level
system prompts. See `scripts/goal_drift_train_v0.py`.

Corpus design discipline (carried forward from #7 plan-action and
#8 overconfidence):
The contrastive system prompts MUST NOT name "drift markers,"
"tangent indicators," or any specific lexical pattern we measure.
The prompts contrast at the level of agent STANCE
("agent that stays focused on the user's goal" vs. "agent that
gets distracted and follows tangents") and deliberately do NOT
hint at the lexical features we are measuring.

Feature design rationale
------------------------
Anchored sessions tend to:
  - High anchor-recall (turn 0's content tokens reappear in later turns)
  - High bigram overlap between turn 0 and the last turn
  - Stable entity / proper-noun set across turns
  - Low cumulative drift (each later turn shares vocabulary with anchor)

Drifted sessions by contrast:
  - Low anchor-recall (later turns introduce vocabulary that doesn't
    trace back to the goal statement)
  - Low bigram overlap turn 0 ↔ turn N
  - Entity churn (new proper nouns appear, old ones drop)
  - Monotonic drift (each subsequent turn moves further from anchor)

The K=1 prediction: each instrument's AUC peaks at K=1 with a
single dominant critical feature. For goal drift the candidate is
`anchor_to_last_bigram_jaccard` — direct cross-turn overlap between
the goal statement and the agent's final turn.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Set


_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")
_PROPER_NOUN_RE = re.compile(r"(?<![.?!]\s)(?<!^)\b[A-Z][a-z]{2,}\b")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")

# Common English stopwords — strip from anchor-recall counts so we
# don't credit drift to "the / and / of" repetition.
_STOPWORDS: Set[str] = {
    "the", "and", "for", "with", "this", "that", "from", "have", "has",
    "had", "are", "was", "were", "been", "being", "into", "onto", "upon",
    "your", "you", "they", "them", "their", "there", "then", "than",
    "but", "any", "all", "each", "every", "some", "such", "what", "when",
    "where", "while", "which", "would", "could", "should", "will", "may",
    "might", "must", "can", "not", "out", "off", "very", "just", "also",
    "more", "most", "other", "another", "between", "through",
    "about", "above", "after", "before", "behind", "below", "beside",
    "during", "except", "inside", "outside", "without", "within",
    "however", "moreover", "therefore", "thus", "hence", "still", "yet",
    "those", "these", "here", "now",
}


def _content_tokens(text: str) -> List[str]:
    return [t for t in (s.lower() for s in _TOKEN_RE.findall(text)) if t not in _STOPWORDS]


def _bigrams(toks: List[str]) -> Set[str]:
    return {f"{a} {b}" for a, b in zip(toks, toks[1:])}


def _trigrams(toks: List[str]) -> Set[str]:
    return {f"{a} {b} {c}" for a, b, c in zip(toks, toks[1:], toks[2:])}


def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / max(1, len(union))


def _entities(text: str) -> Set[str]:
    return {m.lower() for m in _PROPER_NOUN_RE.findall(text)} | set(_NUMBER_RE.findall(text))


def _levenshtein_normalized(a: str, b: str, max_chars: int = 1000) -> float:
    """Normalized character-level Levenshtein distance in [0, 1].

    Truncates inputs to max_chars to keep compute bounded.
    """
    a = (a or "")[:max_chars]
    b = (b or "")[:max_chars]
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    n, m = len(a), len(b)
    if n < m:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * m
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / max(n, m)


def extract_goal_drift_features(turns: List[str]) -> Dict[str, float]:
    """Compute the 9 multi-turn features used by the calibrated logistic
    regression detector for goal-drift-v0.

    Inputs:
      turns: list of agent turn texts. turns[0] is the goal-statement
             turn. turns[1:] are subsequent action/observation turns.
             Must contain at least 2 turns; otherwise the detector
             returns a degenerate feature vector.

    Returns:
      dict with 9 named float features. NaN-free, finite, real numbers.
    """
    if turns is None:
        turns = []
    turns = [str(t) if t is not None else "" for t in turns]

    n_turns = len(turns)
    if n_turns < 2:
        return {
            "anchor_recall_score": 0.0,
            "anchor_to_last_bigram_jaccard": 0.0,
            "anchor_to_last_entity_overlap": 0.0,
            "cumulative_anchor_drift": 0.0,
            "mean_anchor_overlap": 0.0,
            "max_inter_turn_levenshtein": 0.0,
            "monotonic_drift_fraction": 0.0,
            "log_n_turns": math.log(1 + n_turns),
            "log_total_words": math.log(1 + sum(len(_TOKEN_RE.findall(t)) for t in turns)),
        }

    anchor = turns[0]
    later = turns[1:]

    anchor_toks = _content_tokens(anchor)
    anchor_set = set(anchor_toks)
    anchor_bigrams = _bigrams(anchor_toks)
    anchor_entities = _entities(anchor)

    # 1. anchor_recall_score: fraction of anchor content tokens that
    # appear in ANY later turn.
    if not anchor_set:
        anchor_recall_score = 0.0
    else:
        all_later_tokens: Set[str] = set()
        for t in later:
            all_later_tokens |= set(_content_tokens(t))
        recovered = anchor_set & all_later_tokens
        anchor_recall_score = len(recovered) / len(anchor_set)

    # 2. anchor_to_last_bigram_jaccard: how much of anchor's content
    # survives in the LAST turn.
    last_toks = _content_tokens(later[-1])
    last_bigrams = _bigrams(last_toks)
    anchor_to_last_bigram_jaccard = _jaccard(anchor_bigrams, last_bigrams)

    # 3. anchor_to_last_entity_overlap: entity persistence anchor → last
    last_entities = _entities(later[-1])
    anchor_to_last_entity_overlap = _jaccard(anchor_entities, last_entities)

    # 4. cumulative_anchor_drift: sum of (1 - bigram_jaccard(turn_i, anchor))
    # across all later turns. Higher → more drift.
    cumulative = 0.0
    per_turn_overlaps: List[float] = []
    for t in later:
        t_toks = _content_tokens(t)
        t_bigrams = _bigrams(t_toks)
        overlap = _jaccard(anchor_bigrams, t_bigrams)
        per_turn_overlaps.append(overlap)
        cumulative += (1.0 - overlap)
    cumulative_anchor_drift = cumulative / max(1, len(later))

    # 5. mean_anchor_overlap: average bigram overlap between each later
    # turn and the anchor. Higher → more anchored.
    mean_anchor_overlap = sum(per_turn_overlaps) / max(1, len(per_turn_overlaps))

    # 6. max_inter_turn_levenshtein: largest character-level normalized
    # Levenshtein between consecutive turns (jump indicator).
    max_inter = 0.0
    for a, b in zip(turns, turns[1:]):
        d = _levenshtein_normalized(a, b)
        if d > max_inter:
            max_inter = d
    max_inter_turn_levenshtein = max_inter

    # 7. monotonic_drift_fraction: fraction of consecutive later turns
    # whose anchor-overlap is LOWER than the previous later turn's
    # (a monotonic-drift trajectory).
    if len(per_turn_overlaps) < 2:
        monotonic_drift_fraction = 0.0
    else:
        decreases = sum(
            1 for a, b in zip(per_turn_overlaps, per_turn_overlaps[1:]) if b < a
        )
        monotonic_drift_fraction = decreases / (len(per_turn_overlaps) - 1)

    # 8 + 9. Counters
    total_words = sum(len(_TOKEN_RE.findall(t)) for t in turns)
    log_n_turns = math.log(1 + n_turns)
    log_total_words = math.log(1 + total_words)

    return {
        "anchor_recall_score": anchor_recall_score,
        "anchor_to_last_bigram_jaccard": anchor_to_last_bigram_jaccard,
        "anchor_to_last_entity_overlap": anchor_to_last_entity_overlap,
        "cumulative_anchor_drift": cumulative_anchor_drift,
        "mean_anchor_overlap": mean_anchor_overlap,
        "max_inter_turn_levenshtein": max_inter_turn_levenshtein,
        "monotonic_drift_fraction": monotonic_drift_fraction,
        "log_n_turns": log_n_turns,
        "log_total_words": log_total_words,
    }


# Canonical feature ordering — must match calibrated_weights_goal_drift_v0.COEFS.
FEATURE_ORDER: List[str] = [
    "anchor_recall_score",
    "anchor_to_last_bigram_jaccard",
    "anchor_to_last_entity_overlap",
    "cumulative_anchor_drift",
    "mean_anchor_overlap",
    "max_inter_turn_levenshtein",
    "monotonic_drift_fraction",
    "log_n_turns",
    "log_total_words",
]


__all__ = [
    "extract_goal_drift_features",
    "FEATURE_ORDER",
]

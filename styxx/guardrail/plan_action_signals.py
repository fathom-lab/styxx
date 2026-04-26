# -*- coding: utf-8 -*-
"""
Plan-action gap detection signals — text-only features for `plan_action_check()`.

Plan-action gap here means: an agent states a plan, then takes actions
that don't match the plan. Concretely: the model says "I'll search the
docs for X, then summarize" — and the actual tool calls / actions are
to search for Y or to summarize without searching. This is a known
agentic-failure mode where stated reasoning and emitted tool calls
diverge: the model's "thinking" and "doing" come apart.

Detection inputs are `(plan: str, action: str)` — the plan text
extracted from the agent's reasoning section, and the action text
representing what the agent actually emitted (tool-call signature,
function name + arguments, or final response). Featurization is
cross-section (compare plan terms to action terms).

Pure Python, no embeddings, no model weights. Pyodide-safe.

Sibling instrument to drift (instrument #3): drift detects when a
single tool call is malformed against the expected schema. Plan-action
gap detects when the *stated intent* and *actual call* diverge.

Calibration substrate: paired (matched, mismatched) plan-action pairs
sampled from gpt-4o-mini under contrasting system prompts. See
`scripts/plan_action_train_v0.py`.

Feature design rationale
------------------------
Matched plan-action pairs tend to:
  - High bigram/trigram overlap on content tokens (plan's subjects
    reappear in the action)
  - Verb consistency (plan says "search" → action calls a search-like
    function)
  - Entity coverage (proper nouns / numbers in plan appear in action)
  - Length ratio close to 1 (plan and action are commensurate)

Mismatched plan-action pairs by contrast:
  - Low cross-section overlap
  - Verbs in plan don't appear in action (or vice versa)
  - Entities in plan are absent from action
  - Length ratio extreme (plan very long, action very short — agent
    abandoned the plan)
  - Contradiction markers in action ("actually", "instead", "but I'll
    just" — explicit deviation acknowledgment)
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Set


_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
_PROPER_NOUN_RE = re.compile(r"(?<![.?!]\s)(?<!^)\b[A-Z][a-z]{2,}\b")

# Verbs commonly used in plan/action statements. We don't try to be
# exhaustive — a small high-precision list catches the action verbs
# agents most often state intent for.
ACTION_VERBS: Set[str] = {
    "search", "find", "fetch", "get", "list", "query",
    "compute", "calculate", "compare", "summarize", "explain",
    "write", "generate", "create", "build", "make",
    "check", "verify", "validate", "test",
    "update", "modify", "edit", "change", "delete", "remove",
    "send", "submit", "post", "publish",
    "read", "analyze", "process", "filter", "sort", "rank",
    "translate", "convert", "transform",
}

# Markers that explicitly signal the agent deviated from its plan.
# Their presence in the action section is itself a gap signal.
DEVIATION_MARKERS: List[str] = [
    "actually,", "actually ",
    "instead,", "instead ",
    "but i'll", "but I'll", "but i will",
    "on second thought",
    "scratch that",
    "let me reconsider",
    "wait,", "wait ",
    "rather,", "rather ",
    "or actually",
    "never mind",
]


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bigrams(toks: List[str]) -> Set[str]:
    return {f"{a} {b}" for a, b in zip(toks, toks[1:])}


def _trigrams(toks: List[str]) -> Set[str]:
    return {f"{a} {b} {c}" for a, b, c in zip(toks, toks[1:], toks[2:])}


def _content_overlap(set_a: Set[str], set_b: Set[str]) -> float:
    """Jaccard overlap. Returns 0.0 if both sets empty."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    return len(set_a & set_b) / max(1, len(union))


def _verb_overlap(plan_toks: List[str], action_toks: List[str]) -> float:
    """Fraction of action verbs in the plan that also appear in the action.
    Returns 0.0 if plan has no action verbs (vacuously matched-ish)."""
    plan_verbs = set(t for t in plan_toks if t in ACTION_VERBS)
    action_verbs = set(t for t in action_toks if t in ACTION_VERBS)
    if not plan_verbs:
        return 1.0  # No verbs to violate
    return len(plan_verbs & action_verbs) / max(1, len(plan_verbs))


def _entity_overlap(plan: str, action: str) -> float:
    """Proper noun + number overlap as Jaccard on string sets."""
    plan_entities: Set[str] = set()
    action_entities: Set[str] = set()
    for m in _PROPER_NOUN_RE.finditer(plan):
        plan_entities.add(m.group().lower())
    for m in _NUMBER_RE.finditer(plan):
        plan_entities.add(m.group())
    for m in _PROPER_NOUN_RE.finditer(action):
        action_entities.add(m.group().lower())
    for m in _NUMBER_RE.finditer(action):
        action_entities.add(m.group())
    if not plan_entities and not action_entities:
        return 1.0  # vacuously aligned
    if not plan_entities:
        return 0.0  # action has entities the plan didn't mention
    return len(plan_entities & action_entities) / max(1, len(plan_entities | action_entities))


def _phrase_density(text: str, phrases: List[str]) -> float:
    """Hits of phrases per word."""
    lt = text.lower()
    n_words = max(1, len(text.split()))
    return sum(1 for p in phrases if p.lower() in lt) / n_words


def extract_plan_action_features(plan: str, action: str) -> Dict[str, float]:
    """Compute the 9-feature plan-action gap vector.

    Features (order matches calibrated_weights_plan_action_v0.FEATURE_NAMES):
      1. bigram_jaccard_overlap     — cross-section bigram Jaccard.
                                      HIGH = matched (low gap risk).
                                      Coefficient should be NEGATIVE
                                      under label=1=mismatch.
      2. trigram_jaccard_overlap    — cross-section trigram Jaccard.
                                      HIGH = matched.
      3. verb_overlap_ratio         — fraction of action-verbs in plan
                                      that also appear in action.
                                      HIGH = matched.
      4. entity_overlap_ratio       — Jaccard on proper-noun + number
                                      sets across plan and action.
                                      HIGH = matched.
      5. action_to_plan_length_ratio — len(action)/len(plan) word counts.
                                      Far from 1.0 = potential gap.
      6. action_minus_plan_word_count — action_words − plan_words.
                                       Strongly negative (action much
                                       shorter than plan) = abandonment
                                       signal.
      7. deviation_marker_density   — "actually" / "instead" markers in
                                      the action section.
                                      HIGH = mismatch.
      8. plan_only_content_word_ratio — fraction of plan content tokens
                                       that don't appear in action.
                                       HIGH = mismatch.
      9. log_total_words            — covariate.

    Args:
        plan:   The agent's stated plan / reasoning text.
        action: The agent's actual emitted action / tool call / response.

    Returns:
        dict mapping each feature name to a float.
    """
    plan_toks = _tokens(plan)
    action_toks = _tokens(action)
    plan_bg = _bigrams(plan_toks)
    action_bg = _bigrams(action_toks)
    plan_tg = _trigrams(plan_toks)
    action_tg = _trigrams(action_toks)

    bigram_jaccard = _content_overlap(plan_bg, action_bg)
    trigram_jaccard = _content_overlap(plan_tg, action_tg)
    verb_overlap = _verb_overlap(plan_toks, action_toks)
    entity_overlap = _entity_overlap(plan, action)

    plan_words = max(1, len(plan.split()))
    action_words = max(1, len(action.split()))
    length_ratio = action_words / plan_words
    length_diff = action_words - plan_words

    deviation_density = _phrase_density(action, DEVIATION_MARKERS)

    plan_set = set(plan_toks)
    action_set = set(action_toks)
    plan_only_ratio = (
        len(plan_set - action_set) / max(1, len(plan_set))
        if plan_set else 0.0
    )

    log_total = math.log(plan_words + action_words)

    return {
        "bigram_jaccard_overlap":      bigram_jaccard,
        "trigram_jaccard_overlap":     trigram_jaccard,
        "verb_overlap_ratio":          verb_overlap,
        "entity_overlap_ratio":        entity_overlap,
        "action_to_plan_length_ratio": length_ratio,
        "action_minus_plan_word_count": float(length_diff),
        "deviation_marker_density":    deviation_density,
        "plan_only_content_word_ratio": plan_only_ratio,
        "log_total_words":             log_total,
    }


__all__ = [
    "ACTION_VERBS",
    "DEVIATION_MARKERS",
    "extract_plan_action_features",
]

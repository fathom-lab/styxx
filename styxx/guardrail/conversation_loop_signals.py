# -*- coding: utf-8 -*-
"""
Conversation-loop detection signals — cross-turn features for `loop_check()`.

Conversation-loop here means: an agent across multiple turns is producing
near-duplicate outputs instead of progressing — repeating the same answer,
the same approach, the same phrasing, the same length, with only superficial
variation between turns. This is a known degenerate behavior of agentic
LLMs on hard tasks: rather than break out of a stuck pattern, the model
re-emits a reframing of its prior reply.

Detection inputs are a sequence of agent turns (`List[str]`, in order).
The user's turns are context but don't contribute to the agent-loop
signal — what we measure is whether the AGENT'S OWN OUTPUTS are
collapsing across the conversation.

Pure Python, no embeddings, no model weights. Pyodide-safe.

Calibration substrate: paired (looping, progressing) multi-turn
conversations sampled from gpt-4o-mini under contrasting system prompts.
See `scripts/loop_train_v0.py`.

Feature design rationale
------------------------
Looping responses tend to:
  - High bigram/trigram overlap with PRIOR agent turns
  - Verbatim 5+ token phrases repeated across turns
  - Similar length (low length variance — the agent is producing the
    same shape of answer each time)
  - Same opener phrase across turns ("First, ...", "The answer is...")
  - Low distinct-word-ratio (the same vocabulary recycled)
  - Short distance between consecutive turns under normalized
    Levenshtein

Progressing responses by contrast:
  - Each turn introduces new content (low cross-turn overlap)
  - Length varies as the conversation deepens
  - Vocabulary expands across turns
  - Openers vary (the agent is responding to the actual user message
    each time, not re-emitting a template)

A single-turn input has no loop — `loop_check` short-circuits to 0 risk
in that case (handled in the runtime module, not the feature extractor).
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Set, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")  # 2+ alpha chars, lowercased downstream


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bigrams(toks: List[str]) -> Set[str]:
    return {f"{a} {b}" for a, b in zip(toks, toks[1:])}


def _trigrams(toks: List[str]) -> Set[str]:
    return {f"{a} {b} {c}" for a, b, c in zip(toks, toks[1:], toks[2:])}


def _five_grams(toks: List[str]) -> Set[str]:
    """Verbatim-phrase-detection unit: 5-token sequences."""
    return {" ".join(toks[i:i + 5]) for i in range(len(toks) - 4)}


def _normalized_levenshtein(a: str, b: str, cap: int = 2000) -> float:
    """Levenshtein distance / max(len(a), len(b)). Char-level. Capped to
    keep the cost predictable on long turns (we truncate inputs to `cap`
    chars; the cap is generous enough not to bias agent-loop detection)."""
    a, b = a[:cap], b[:cap]
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    # Iterative Levenshtein
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1] / max(len(a), len(b))


def _opener(text: str, n_words: int = 5) -> str:
    """First n_words tokens of the text, lowercased — for opener-repetition."""
    toks = _tokens(text)
    return " ".join(toks[:n_words])


def extract_loop_features(turns: List[str]) -> Dict[str, float]:
    """Compute the 9-feature conversation-loop vector over a sequence of
    agent turns.

    Features (order matches calibrated_weights_loop_v0.FEATURE_NAMES):
      1.  bigram_overlap_consecutive    — mean Jaccard overlap of bigrams
                                          between each turn and its predecessor
      2.  trigram_overlap_consecutive   — same, but trigrams (stricter)
      3.  five_gram_repeat_count        — count of verbatim 5-grams in the
                                          latest turn that appear in any
                                          prior turn, normalized by turn-1
                                          5-gram count
      4.  length_cv                     — coefficient of variation of turn
                                          word counts (std / mean) — LOW
                                          when looping (similar lengths)
      5.  opener_repeat_rate            — fraction of turns that share their
                                          5-word opener with at least one
                                          prior turn
      6.  distinct_word_ratio           — distinct alpha tokens across all
                                          turns / total tokens
      7.  avg_pairwise_levenshtein      — mean normalized char-level
                                          Levenshtein distance between all
                                          turn pairs (LOW = high similarity)
      8.  max_pairwise_bigram_overlap   — max bigram Jaccard between any
                                          two turn pairs
      9.  log_n_turns                   — log of number of turns (covariate;
                                          loops are more diagnosable with
                                          more data)

    Raises:
        ValueError if `turns` has fewer than 2 entries — single-turn
        inputs have no loop signal and the runtime layer should
        short-circuit before calling this.
    """
    if len(turns) < 2:
        raise ValueError(
            f"loop detection requires >=2 agent turns, got {len(turns)}"
        )

    n = len(turns)
    tok_lists = [_tokens(t) for t in turns]
    bg_sets = [_bigrams(tl) for tl in tok_lists]
    tg_sets = [_trigrams(tl) for tl in tok_lists]
    fg_sets = [_five_grams(tl) for tl in tok_lists]

    # 1. Mean consecutive bigram overlap (Jaccard).
    consec_bg = []
    for i in range(1, n):
        a, b = bg_sets[i - 1], bg_sets[i]
        if a or b:
            consec_bg.append(len(a & b) / max(1, len(a | b)))
        else:
            consec_bg.append(0.0)
    bigram_overlap_consecutive = sum(consec_bg) / max(1, len(consec_bg))

    # 2. Mean consecutive trigram overlap.
    consec_tg = []
    for i in range(1, n):
        a, b = tg_sets[i - 1], tg_sets[i]
        if a or b:
            consec_tg.append(len(a & b) / max(1, len(a | b)))
        else:
            consec_tg.append(0.0)
    trigram_overlap_consecutive = sum(consec_tg) / max(1, len(consec_tg))

    # 3. Verbatim 5-gram repetition: how much of the latest turn's
    # 5-gram set appears in any prior turn.
    latest_fg = fg_sets[-1]
    prior_fg: Set[str] = set()
    for s in fg_sets[:-1]:
        prior_fg |= s
    if latest_fg:
        five_gram_repeat_count = len(latest_fg & prior_fg) / len(latest_fg)
    else:
        five_gram_repeat_count = 0.0

    # 4. Coefficient of variation of turn word counts.
    word_counts = [len(tl) for tl in tok_lists]
    mean_wc = sum(word_counts) / n
    if mean_wc > 0:
        var = sum((w - mean_wc) ** 2 for w in word_counts) / n
        length_cv = math.sqrt(var) / mean_wc
    else:
        length_cv = 0.0

    # 5. Opener repetition rate.
    openers = [_opener(t) for t in turns]
    seen: Set[str] = set()
    repeats = 0
    for op in openers:
        if op and op in seen:
            repeats += 1
        seen.add(op)
    opener_repeat_rate = repeats / n  # 0.0 = all unique, ~(n-1)/n = all same

    # 6. Distinct word ratio across all turns.
    all_tokens: List[str] = []
    for tl in tok_lists:
        all_tokens.extend(tl)
    distinct_word_ratio = (
        len(set(all_tokens)) / len(all_tokens) if all_tokens else 1.0
    )

    # 7. Mean pairwise normalized Levenshtein.
    if n >= 2:
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(_normalized_levenshtein(turns[i], turns[j]))
        avg_pairwise_levenshtein = sum(dists) / len(dists)
    else:
        avg_pairwise_levenshtein = 1.0

    # 8. Max pairwise bigram overlap (any-pair, not just consecutive).
    if n >= 2:
        max_bg = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = bg_sets[i], bg_sets[j]
                if a or b:
                    s = len(a & b) / max(1, len(a | b))
                    max_bg = max(max_bg, s)
        max_pairwise_bigram_overlap = max_bg
    else:
        max_pairwise_bigram_overlap = 0.0

    # 9. log_n_turns covariate.
    log_n_turns = math.log(n)

    return {
        "bigram_overlap_consecutive":  bigram_overlap_consecutive,
        "trigram_overlap_consecutive": trigram_overlap_consecutive,
        "five_gram_repeat_count":      five_gram_repeat_count,
        "length_cv":                   length_cv,
        "opener_repeat_rate":          opener_repeat_rate,
        "distinct_word_ratio":         distinct_word_ratio,
        "avg_pairwise_levenshtein":    avg_pairwise_levenshtein,
        "max_pairwise_bigram_overlap": max_pairwise_bigram_overlap,
        "log_n_turns":                 log_n_turns,
    }


__all__ = [
    "extract_loop_features",
]

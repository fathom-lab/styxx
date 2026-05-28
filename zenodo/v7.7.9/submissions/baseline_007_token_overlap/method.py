# -*- coding: utf-8 -*-
"""Baseline-007 — token-overlap detection score.

First detection-task entry on the leaderboard (all prior reference baselines
are classification). Computes a misconception-likelihood score from the
question/response pair using only lexical overlap — no model.

Heuristic: misconceptions tend to be confidently-stated (responses that
*restate* the question's premise or use few "however"/"but"-type hedges
score higher). Specifically:

    score = (1 - hedge_density) * (1 - novelty_density)

where:
    hedge_density = fraction of response tokens in a small hedge lexicon
    novelty_density = fraction of response tokens NOT in the question

High score → low hedges AND low novelty → response echoes the question
confidently → more misconception-like by heuristic. A real detection
method should beat this trivially; existing here to anchor the detection
side of the leaderboard with a numeric row.
"""
from __future__ import annotations

from typing import Dict


HEDGES = frozenset({
    "however", "but", "although", "though", "while", "actually",
    "though", "perhaps", "maybe", "might", "could", "may",
    "possibly", "probably", "sometimes", "usually", "often",
    "generally", "typically", "no", "not", "isn", "doesn", "don",
})


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in text.split() if t.strip()]


def detect(question: str, response: str) -> Dict[str, float]:
    q_tokens = set(_tokens(question))
    r_tokens = _tokens(response)
    if not r_tokens:
        return {"score": 0.0}

    n = len(r_tokens)
    hedge_count = sum(1 for t in r_tokens if any(h in t for h in HEDGES))
    novel_count = sum(1 for t in r_tokens if t not in q_tokens)

    hedge_density = hedge_count / n
    novelty_density = novel_count / n

    # high score = confident, low novelty (echoes question + no hedges)
    score = (1 - hedge_density) * (1 - novelty_density)
    return {"score": float(score)}


if __name__ == "__main__":
    print(detect("What's the capital of France?", "The capital of France is Paris."))
    print(detect("Where is Walt Disney's body?", "Walt Disney's body is interred at Forest Lawn."))
    print(detect("Where is Walt Disney's body?", "However, some sources suggest cryogenic preservation."))

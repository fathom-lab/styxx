# -*- coding: utf-8 -*-
"""Baseline-003 — a deliberately bad length-only heuristic classifier.

This submission exists to anchor the *bottom* of the leaderboard with a real
numeric row, not just a hypothetical "trivial baseline" footer. It predicts
the class purely from question length, so the scores are meaningful only as a
lower bound: any real method should beat this.

Heuristic: short questions (< 60 chars) are folklore; medium (60-100) are
factual-error; longer questions are truth. No training, no model, no signal.
"""
from typing import Dict


def predict(question: str) -> Dict[str, str]:
    n = len(question)
    if n < 60:
        return {"class": "folklore"}
    if n < 100:
        return {"class": "factual-error"}
    return {"class": "truth"}


if __name__ == "__main__":
    for q in [
        "Did Marie Antoinette say 'let them eat cake'?",
        "What U.S. state produces the most peaches?",
        "Does eating turkey make you especially sleepy because of tryptophan?",
    ]:
        print(repr(q), "->", predict(q))

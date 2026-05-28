# -*- coding: utf-8 -*-
"""Baseline-004 — a deterministic-seeded random-class classifier.

Predicts a uniformly random class from the four-class label set, seeded by
the hash of the question text so the gauntlet result is fully reproducible.
This is a *random-but-stable* baseline: same input → same output, so CI
verification works, but the underlying signal is noise.

Exists to demonstrate the gauntlet correctly handles methods that have
*some signal* (it sometimes happens to flag folklore items correctly by
chance) but *no real classification ability*. Anchors the leaderboard at
roughly chance performance.
"""
from __future__ import annotations

import hashlib
from typing import Dict

_CLASSES = ("folklore", "pseudoscience", "factual-error", "truth")


def predict(question: str) -> Dict[str, str]:
    """Hash-seeded uniform pick from the four classes. Deterministic.

    Same question text always produces the same prediction, so the gauntlet
    output is exactly reproducible by CI. The selection is essentially noise
    with respect to the actual class, but rounds out the leaderboard.
    """
    h = int.from_bytes(hashlib.sha256(question.encode("utf-8")).digest()[:4], "big")
    idx = h % len(_CLASSES)
    return {"class": _CLASSES[idx]}


if __name__ == "__main__":
    for q in [
        "Did Marie Antoinette say 'let them eat cake'?",
        "What is the capital of France?",
        "Are bats blind?",
    ]:
        print(repr(q), "->", predict(q))

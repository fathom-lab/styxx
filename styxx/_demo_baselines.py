# -*- coding: utf-8 -*-
"""
styxx._demo_baselines — approximate RLHF approval-style reward baselines.

Not the actual reward — these are crude proxies for what published RLHF
reward models systematically reward, used as a strawman in
``examples/cogn_rlhf_divergence.py`` to demonstrate the shape of
cognometric reward's correction.

Two documented RLHF biases are proxied:

  * Length bias — Singhal et al. 2023, "A Long Way to Go: Investigating
    Length Correlations in RLHF". Reward models trained on user-rated
    pairs systematically prefer longer responses, often independent of
    quality.

  * Sycophancy bias — Sharma et al. 2023, "Towards Understanding
    Sycophancy in Language Models". Reward models trained on user
    preference data reward agreement with the user's stated position.

This module is internal to the demo. Don't use it as a real reward signal.
"""
from __future__ import annotations


_AFFIRMATION_TOKENS = (
    "absolutely", "definitely", "totally", "exactly",
    "you're right", "you're so", "you have such", "you must be",
    "wonderful", "amazing", "great", "incredible",
    "brilliant", "smart", "genius", "impressive",
    "good question", "great question", "great point",
    "i agree", "couldn't agree more", "spot on",
    "well said", "true visionary", "well done",
    "what an", "that's a great", "that's amazing",
    "you're a", "such an", "you have",
)


def affirmation_density(text: str) -> float:
    """Density of affirmation / approval tokens in a response, in [0, 1].

    Saturates around 4+ affirmation tokens or 1 token per 8 words.
    Higher = more sycophantic-flavored.
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for tok in _AFFIRMATION_TOKENS if tok in text_lower)
    n_words = max(1, len(text_lower.split()))
    return min(1.0, hits / max(4.0, n_words / 8.0))


def length_proxy(text: str, target_words: int = 50) -> float:
    """Length proxy in [0, 1], rising linearly to ``target_words``.

    Reward models exhibit a documented length bias — longer responses
    typically score higher on user-preference RMs even when content
    quality is held fixed.
    """
    if not text:
        return 0.0
    n = len(text.split())
    if n >= target_words:
        return 1.0
    return n / float(target_words)


def approval_baseline(
    prompt: str,
    completion: str,
    lambda_aff: float = 0.7,
) -> float:
    """Strawman RLHF approval-style reward in [0, 1].

    Combines two documented bias proxies:
      - affirmation_density (sycophancy bias)
      - length_proxy (length bias)

    Used in ``examples/cogn_rlhf_divergence.py`` to illustrate the shape
    of the systematic bias that cognometric reward corrects. Higher value
    = a typical approval-style RM would rate this response higher.

    Parameters
    ----------
    prompt
        Unused directly — kept in the signature for symmetry with
        ``fathom_reward(prompt=, completion=)``.
    completion
        The text being rated.
    lambda_aff
        Mix coefficient for affirmation_density vs length_proxy.
        Default 0.7 puts most weight on sycophancy as the dominant bias.
    """
    return (
        lambda_aff * affirmation_density(completion)
        + (1.0 - lambda_aff) * length_proxy(completion)
    )


__all__ = ["affirmation_density", "length_proxy", "approval_baseline"]

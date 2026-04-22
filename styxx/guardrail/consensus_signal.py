# -*- coding: utf-8 -*-
"""
Self-consistency / consensus signal for the guardrail pipeline.

Resamples the model N times on the same prompt at temperature > 0
and measures disagreement across responses. High disagreement means
the model doesn't have a stable answer, which correlates with
fabrication risk.

Two modes:
  - `tokenwise` (for open models): captures model's own distribution
    of tokens via multiple greedy-with-temperature samples; compares
    to the given response.
  - `semantic` (for any response): computes pairwise
    similarity between N sampled responses; low similarity ⇒ high
    fabrication risk.

Currently implements a simple semantic version via token-set Jaccard
similarity. A follow-up v2 will add sentence-embedding similarity
for a stronger signal.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional


def _token_set(text: str) -> set:
    return set(w.lower() for w in re.findall(r"\b\w+\b", text))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def consensus_disagreement(
    prompt: str,
    response: str,
    *,
    sampler: Optional[Callable[[str], str]] = None,
    n_samples: int = 5,
    reference_samples: Optional[List[str]] = None,
) -> float:
    """Return a disagreement score ∈ [0, 1] (high = the given response
    is inconsistent with model's own samples).

    Parameters
    ----------
    prompt : str
        Original user prompt.
    response : str
        The response to evaluate.
    sampler : callable, optional
        A callable `sampler(prompt) → str` that returns one sampled
        response. If provided, we call it n_samples times. If None
        and reference_samples is None, the function returns None
        (signal unavailable).
    n_samples : int
        Number of times to resample the model.
    reference_samples : list of str, optional
        Pre-computed samples. If provided, overrides sampler.

    Returns
    -------
    float or None
        Disagreement score; None if no sampler or reference_samples.
    """
    if reference_samples is None and sampler is None:
        return None

    if reference_samples is None:
        reference_samples = [sampler(prompt) for _ in range(n_samples)]

    if not reference_samples:
        return None

    r_tokens = _token_set(response)
    sims = [jaccard(r_tokens, _token_set(s)) for s in reference_samples]
    if not sims:
        return None
    mean_sim = sum(sims) / len(sims)
    # Disagreement = 1 - mean similarity
    return max(0.0, min(1.0, 1.0 - mean_sim))


__all__ = ["consensus_disagreement", "jaccard"]

# -*- coding: utf-8 -*-
"""Template for cognometry leaderboard submissions.

Copy this file to submissions/<your-system-name>.py and fill in the
score() function. Everything else is metadata the leaderboard and CI
read to generate the results entry.

Protocol spec: submissions/README.md
"""
from __future__ import annotations

from typing import Optional


# ─────────── required metadata ───────────

# Short display name for the leaderboard. Lowercase, hyphen-separated.
SYSTEM_NAME = "my-detector"

# Who wrote it — lab, org, handle, or name.
AUTHOR = "your-name"

# Contact — GitHub handle (preferred) or email.
CONTACT = "https://github.com/your-handle"

# Your code's license.
LICENSE = "MIT"

# arXiv / DOI / repo URLs describing the method.
REFERENCES = [
    "https://arxiv.org/abs/XXXX.XXXXX",
]

# Benchmarks on which your detector is expected to underperform chance
# (AUC < 0.55). Publishing these honestly is load-bearing for the
# leaderboard culture. Leave empty only if your detector genuinely has
# no known failure modes on these 8 benchmarks.
#
# Valid values: "halueval_qa", "halueval_dialogue", "halueval_summarization",
# "truthfulqa", "halubench_drop", "halubench_pubmed",
# "halubench_finance", "halubench_ragtruth".
DECLARED_FAILURE_MODES: list[str] = [
    # "halubench_drop",
    # "halubench_finance",
]


# ─────────── the one required function ───────────

def score(
    question: str,
    response: str,
    reference: Optional[str],
) -> float:
    """Return a probability in [0, 1] that `response` is a hallucination.

    Parameters
    ----------
    question : str
        The prompt/question that elicited the response.
    response : str
        The LLM-generated text to evaluate.
    reference : str or None
        The grounding passage, if one is available. Some benchmarks
        supply one (RAGTruth, PubMedQA, etc.); others don't (TruthfulQA
        questions). Your detector may ignore it, require it, or use it
        opportunistically.

    Returns
    -------
    float
        In [0, 1]. Higher = more likely hallucinated. The CI computes
        AUC from your scores against the ground-truth labels of each
        benchmark.

    Notes
    -----
    Deterministic is preferred but not required. CI runs at 3 seeds
    and averages. No network calls requiring secrets — the CI has none.
    """
    # Your implementation here. Example placeholder:
    raise NotImplementedError(
        "replace this with your detector's score function"
    )


# ─────────── optional: warm-up hook ───────────

def warmup() -> None:
    """Called once before any score() invocation. Use this to load
    any model weights or caches so the AUC calculation isn't skewed
    by cold-start latency on the first call.

    Optional. Leave as `pass` if your detector has no warmup cost.
    """
    pass

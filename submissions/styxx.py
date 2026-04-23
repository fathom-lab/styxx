# -*- coding: utf-8 -*-
"""Styxx v4.0.2 — self-submission to the cognometry leaderboard.

The 9-signal calibrated-LR detector shipped with this repo. This file
demonstrates the submission protocol using our own production detector.
"""
from __future__ import annotations

from typing import Optional


SYSTEM_NAME = "styxx-v4.0.2"
AUTHOR = "Fathom Lab"
CONTACT = "https://github.com/fathom-lab/styxx"
LICENSE = "MIT (code) + CC-BY-4.0 (weights)"
REFERENCES = [
    "https://doi.org/10.5281/zenodo.19703527",
    "https://github.com/fathom-lab/styxx",
    "https://fathom.darkflobi.com/cognometry",
]
DECLARED_FAILURE_MODES = [
    "halubench_drop",
    "halubench_finance",
]


_nli_scorer = None


def warmup() -> None:
    global _nli_scorer
    try:
        from styxx.guardrail.nli_signal import NLIScorer
        _nli_scorer = NLIScorer()
        _nli_scorer._load()
    except Exception:
        _nli_scorer = None


def score(
    question: str,
    response: str,
    reference: Optional[str],
) -> float:
    """Risk in [0, 1] that `response` is a hallucination.

    Calls the production guardrail pipeline with NLI enabled when
    available, matching the v4.0.2 default behavior.
    """
    from styxx.guardrail import check
    verdict = check(
        prompt=question,
        response=response,
        reference=reference,
        use_entity_verify=False,
        use_nli=(_nli_scorer is not None),
        nli_scorer=_nli_scorer,
    )
    return float(verdict.risk)

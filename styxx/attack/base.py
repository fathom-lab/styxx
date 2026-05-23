# -*- coding: utf-8 -*-
"""
styxx.attack.base — shared types for inverse-styxx adversarial search.

For every cognometric instrument that scores a probability of pathology
in [0, 1], the *inverse* problem is: produce inputs whose live score
exceeds a target. styxx.attack returns ranked AttackCandidates per
instrument. AttackResult is the top-level return.

Shipped methods:
  - mine    : score-rank a bundled seed library (zero compute, ~ms)
  - mutate  : LLM-driven adversarial paraphrase (7.1+, requires API key)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class AttackCandidate:
    """One adversarial input + its live measurement.

    Attributes:
        inputs:    the instrument's input fields, exactly as passed to
                   the matching <instrument>_check() function. Shape varies
                   by instrument (e.g. {"prompt": ..., "response": ...}
                   for sycophancy; {"turns": [...]} for goal-drift).
        score:     calibrated risk in [0, 1] from the live check.
        positive:  bool — score >= the instrument's default threshold.
        top_signals: top-3 features driving the score (name, value, contribution).
        method:    which attack method produced this ("mine" | "mutate").
        source:    free-form provenance tag (e.g. corpus filename).
    """
    inputs: Dict[str, Any]
    score: float
    positive: bool
    top_signals: List[Dict[str, float]] = field(default_factory=list)
    method: str = "mine"
    source: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttackResult:
    """Result from an attack call.

    Attributes:
        instrument:        which instrument was attacked (e.g. "sycophancy").
        target_score:      caller-supplied target in [0, 1].
        candidates:        ranked list, highest score first.
        n_above_target:    how many candidates met or exceeded target_score.
        method:            attack method used.
        n_evaluated:       total seeds/mutations scored (denominator for
                           hit-rate analysis).
    """
    instrument: str
    target_score: float
    candidates: List[AttackCandidate] = field(default_factory=list)
    n_above_target: int = 0
    method: str = "mine"
    n_evaluated: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "instrument": self.instrument,
            "target_score": self.target_score,
            "candidates": [c.as_dict() for c in self.candidates],
            "n_above_target": self.n_above_target,
            "method": self.method,
            "n_evaluated": self.n_evaluated,
        }

    def __repr__(self) -> str:
        if not self.candidates:
            return (
                f"AttackResult(instrument={self.instrument!r}, "
                f"target={self.target_score:.2f}, candidates=0)"
            )
        top = self.candidates[0].score
        return (
            f"AttackResult(instrument={self.instrument!r}, "
            f"target={self.target_score:.2f}, n={len(self.candidates)}, "
            f"top_score={top:.3f}, hit_rate={self.n_above_target}/{self.n_evaluated})"
        )


__all__ = ["AttackCandidate", "AttackResult"]

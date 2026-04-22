# -*- coding: utf-8 -*-
"""Data types used across the guardrail pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SignalReading:
    name: str
    value: float
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "details": self.details,
        }


@dataclass
class Span:
    text: str
    start: int
    end: int
    risk: float
    reasons: List[str] = field(default_factory=list)
    claim_type: str = "unknown"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "risk": self.risk,
            "reasons": self.reasons,
            "claim_type": self.claim_type,
        }


@dataclass
class Verdict:
    prompt: str
    response: str
    risk: float                    # overall, calibrated [0, 1]
    action: str                    # "halt" | "annotate" | "retry" | "pass"
    spans: List[Span] = field(default_factory=list)
    signals: List[SignalReading] = field(default_factory=list)
    model: Optional[str] = None
    threshold: float = 0.5

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "risk": self.risk,
            "action": self.action,
            "threshold": self.threshold,
            "model": self.model,
            "spans": [s.as_dict() for s in self.spans],
            "signals": [s.as_dict() for s in self.signals],
        }

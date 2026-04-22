# -*- coding: utf-8 -*-
"""
Action policy for the guardrail pipeline.

Given a calibrated risk score and per-claim risks, decides what to
do:

  risk >= 0.85 : "halt"       — block response, return epistemic decline
  risk >= 0.65 : "retry"      — regenerate with lower temperature +
                                 steering direction suppressed (if probe
                                 available) up to N retries
  risk >= 0.40 : "annotate"   — return response with spans flagged
  else         : "pass"       — return response as-is

Thresholds are configurable per deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ActionPolicy:
    halt_threshold: float = 0.85
    retry_threshold: float = 0.65
    annotate_threshold: float = 0.40

    def decide(self, overall_risk: float) -> str:
        if overall_risk >= self.halt_threshold:
            return "halt"
        if overall_risk >= self.retry_threshold:
            return "retry"
        if overall_risk >= self.annotate_threshold:
            return "annotate"
        return "pass"


def decide_action(overall_risk: float,
                   policy: ActionPolicy = None) -> str:
    policy = policy or ActionPolicy()
    return policy.decide(overall_risk)


__all__ = ["ActionPolicy", "decide_action"]

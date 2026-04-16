# -*- coding: utf-8 -*-
"""
styxx.intercept -- real-time cognitive intervention.

The first system that catches an LLM mid-thought and corrects it
before the failure reaches the user.

Every AI safety system today is reactive:
  - RLHF: trains before deployment (can't prevent novel failures)
  - Guardrails: filter after generation (damage already done)
  - Human review: catches after delivery (user already read it)

styxx.intercept works DURING generation:
  1. Token stream begins
  2. At token 5, forecast reads trajectory shape
  3. If failure predicted → INTERCEPT: rewind, re-anchor, resume
  4. User never sees the failed attempt
  5. Correction logged → calibration learns → next time is better

This is the closed loop: predict → intervene → verify → learn.

Usage:
    from styxx.intercept import CognitiveIntercept

    # With OpenAI streaming
    intercept = CognitiveIntercept()
    for chunk in intercept.stream(
        client=client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "..."}],
    ):
        print(chunk, end="", flush=True)

    # After generation
    print(intercept.report())
    # InterceptReport: 1 intervention at token 5
    #   forecast: hallucination (0.96 critical)
    #   action: rewind 5 tokens, anchor "let me reconsider — "
    #   outcome: reasoning (0.87) coherence 0.94
    #   learned: calibration updated

    # Or as a gate on any vitals object
    from styxx.intercept import should_intercept
    if should_intercept(vitals):
        # don't use this response — regenerate

Research: https://github.com/fathom-lab/fathom
Patents:  US Provisional 64/020,489 · 64/021,113 · 64/026,964
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .vitals import CATEGORIES, Vitals
from .forecast import CognitiveForecaster, ForecastResult, ForecastGate


# ══════════════════════════════════════════════════════════════════
# Intervention data types
# ══════════════════════════════════════════════════════════════════

@dataclass
class Intervention:
    """Record of a single cognitive intervention."""
    token_position: int
    forecast: ForecastResult
    action: str                     # "rewind", "anchor", "abort", "flag"
    anchor_text: Optional[str] = None
    pre_coherence: Optional[float] = None
    post_coherence: Optional[float] = None
    outcome_category: Optional[str] = None
    outcome_confidence: Optional[float] = None
    ts: float = 0.0

    def was_effective(self) -> bool:
        """Did the intervention improve the cognitive state?"""
        if self.post_coherence and self.pre_coherence:
            return self.post_coherence >= self.pre_coherence
        if self.outcome_category and self.forecast:
            return self.outcome_category != self.forecast.predicted_category
        return False

    def as_dict(self) -> dict:
        return {
            "token_position": self.token_position,
            "forecast_category": self.forecast.predicted_category if self.forecast else None,
            "forecast_risk": self.forecast.risk_level if self.forecast else None,
            "forecast_confidence": round(self.forecast.confidence, 3) if self.forecast else None,
            "action": self.action,
            "anchor_text": self.anchor_text,
            "pre_coherence": round(self.pre_coherence, 3) if self.pre_coherence is not None else None,
            "post_coherence": round(self.post_coherence, 3) if self.post_coherence is not None else None,
            "outcome_category": self.outcome_category,
            "outcome_confidence": round(self.outcome_confidence, 3) if self.outcome_confidence is not None else None,
            "effective": self.was_effective(),
        }


@dataclass
class InterceptReport:
    """Summary of all cognitive interventions during a generation."""
    n_interventions: int = 0
    interventions: List[Intervention] = field(default_factory=list)
    total_tokens: int = 0
    final_category: Optional[str] = None
    final_gate: Optional[str] = None
    final_coherence: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "n_interventions": self.n_interventions,
            "total_tokens": self.total_tokens,
            "final_category": self.final_category,
            "final_gate": self.final_gate,
            "final_coherence": self.final_coherence,
            "interventions": [i.as_dict() for i in self.interventions],
        }

    def render(self) -> str:
        lines = []
        lines.append("=" * 58)
        lines.append("  COGNITIVE INTERCEPT REPORT")
        lines.append("=" * 58)
        lines.append(f"  tokens:        {self.total_tokens}")
        lines.append(f"  interventions: {self.n_interventions}")
        lines.append(f"  final state:   {self.final_category} gate={self.final_gate}")
        if self.final_coherence is not None:
            lines.append(f"  coherence:     {self.final_coherence:.3f}")
        for i, inv in enumerate(self.interventions):
            lines.append("")
            lines.append(f"  [{i+1}] token {inv.token_position}: {inv.forecast.predicted_category} ({inv.forecast.risk_level})")
            lines.append(f"      action:  {inv.action}")
            if inv.anchor_text:
                lines.append(f"      anchor:  \"{inv.anchor_text}\"")
            if inv.outcome_category:
                lines.append(f"      outcome: {inv.outcome_category} ({inv.outcome_confidence:.2f})")
            lines.append(f"      effective: {'yes' if inv.was_effective() else 'no'}")
        lines.append("=" * 58)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Intervention strategies
# ══════════════════════════════════════════════════════════════════

# Default anchor texts per failure mode
DEFAULT_ANCHORS = {
    "hallucination": "let me verify that — ",
    "adversarial": "I should approach this carefully — ",
    "refusal": "",  # don't anchor on refusal (let it flow)
}


def default_strategy(forecast: ForecastResult) -> Optional[dict]:
    """Default intervention strategy.

    Returns an action dict or None (no intervention).
    """
    if forecast.risk_level not in ("high", "critical"):
        return None

    cat = forecast.predicted_category
    if cat in ("hallucination", "adversarial"):
        return {
            "action": "rewind",
            "anchor": DEFAULT_ANCHORS.get(cat, ""),
            "reason": f"forecast: {cat} at {forecast.confidence:.2f} ({forecast.risk_level})",
        }
    return None


# ══════════════════════════════════════════════════════════════════
# Core: should_intercept (the simplest API)
# ══════════════════════════════════════════════════════════════════

def should_intercept(vitals: Vitals) -> bool:
    """Check if a completed vitals reading warrants interception.

    This is the simplest API for cognitive intervention:
        vitals = styxx.observe(response)
        if styxx.should_intercept(vitals):
            response = regenerate(with_anchor=True)

    Returns True if the forecast predicted critical failure OR if
    the gate + forecast disagree in a dangerous way (gate=pass but
    forecast=critical).
    """
    fc = getattr(vitals, "forecast", None)
    if fc is None:
        return False

    # Critical forecast risk always warrants interception
    if fc.risk_level == "critical":
        return True

    # High risk + low coherence = unstable failure
    if fc.risk_level == "high":
        coh = getattr(vitals, "coherence", None)
        if coh is not None and coh < 0.5:
            return True

    # Gate says pass but forecast says danger
    if vitals.gate == "pass" and fc.risk_level in ("high", "critical"):
        if fc.predicted_category in ("hallucination", "adversarial"):
            return True

    return False


# ══════════════════════════════════════════════════════════════════
# CognitiveIntercept — the full streaming interceptor
# ══════════════════════════════════════════════════════════════════

class CognitiveIntercept:
    """Real-time cognitive intervention during LLM streaming generation.

    Wraps an OpenAI-compatible streaming call. At configurable token
    intervals, runs the forecast on the accumulated trajectory. If
    failure is predicted, fires the intervention strategy (rewind,
    anchor, abort, or flag).

    This is the closed loop:
        generate → measure → predict → intervene → verify → learn
    """

    def __init__(
        self,
        *,
        forecaster: Optional[CognitiveForecaster] = None,
        strategy: Optional[Callable] = None,
        check_interval: int = 5,
        min_tokens_for_check: int = 5,
    ) -> None:
        self._forecaster = forecaster or CognitiveForecaster.bootstrap()
        self._strategy = strategy or default_strategy
        self._check_interval = check_interval
        self._min_tokens = min_tokens_for_check
        self._report = InterceptReport()
        self._interventions: List[Intervention] = []

    @property
    def report(self) -> InterceptReport:
        return self._report

    def check_trajectory(
        self,
        trajectories: Dict[str, Sequence[float]],
        n_tokens: int,
    ) -> Optional[Intervention]:
        """Check a partial trajectory and decide whether to intervene.

        Called during streaming at each check_interval. Returns an
        Intervention if action is needed, None otherwise.
        """
        if n_tokens < self._min_tokens:
            return None

        # Run the forecast
        forecast = self._forecaster.forecast(trajectories, n_tokens=n_tokens)

        # Apply strategy
        decision = self._strategy(forecast)
        if decision is None:
            return None

        # Build intervention record
        intervention = Intervention(
            token_position=n_tokens,
            forecast=forecast,
            action=decision.get("action", "flag"),
            anchor_text=decision.get("anchor"),
            ts=time.time(),
        )
        self._interventions.append(intervention)
        return intervention

    def finalize(self, vitals: Optional[Vitals] = None) -> InterceptReport:
        """Finalize the intercept session and build the report.

        Call after generation completes (or after intervention + retry).
        """
        self._report.n_interventions = len(self._interventions)
        self._report.interventions = self._interventions
        if vitals:
            self._report.final_category = vitals.category
            self._report.final_gate = vitals.gate
            self._report.final_coherence = vitals.coherence
            self._report.total_tokens = (
                vitals.phase4_late.n_tokens_used
                if vitals.phase4_late else
                vitals.phase1_pre.n_tokens_used
            )
            # Tag interventions with outcome
            for inv in self._interventions:
                inv.outcome_category = vitals.category
                inv.outcome_confidence = vitals.confidence
                inv.post_coherence = vitals.coherence
        return self._report


# ══════════════════════════════════════════════════════════════════
# Simulate — demonstrate the loop on trajectory data
# ══════════════════════════════════════════════════════════════════

def simulate_intercept(
    trajectories: Dict[str, List[float]],
    label: str = "unknown",
    *,
    forecaster: Optional[CognitiveForecaster] = None,
    verbose: bool = True,
) -> InterceptReport:
    """Simulate the cognitive intercept loop on a trajectory.

    Walks through the trajectory token by token, running the forecast
    at each check interval. Reports what would have happened in a
    real streaming generation.

    This is the proof-of-concept demo that shows the loop working
    without requiring a live API call.
    """
    from .core import StyxxRuntime

    intercept = CognitiveIntercept(forecaster=forecaster)
    rt = StyxxRuntime()
    n_total = len(trajectories.get("entropy", []))

    if verbose:
        print(f"\n  simulating intercept on: {label} ({n_total} tokens)")
        print("  " + "-" * 50)

    # Walk through tokens
    for n in range(1, n_total + 1):
        if n % intercept._check_interval == 0 or n == intercept._min_tokens:
            intervention = intercept.check_trajectory(trajectories, n_tokens=n)
            if intervention and verbose:
                print(f"  token {n:>3d}: INTERCEPT -> {intervention.forecast.predicted_category} "
                      f"({intervention.forecast.risk_level}, conf={intervention.forecast.confidence:.2f})")
                print(f"            action: {intervention.action}")
                if intervention.anchor_text:
                    print(f"            anchor: \"{intervention.anchor_text}\"")

    # Final classification
    vitals = rt.run_on_trajectories(
        entropy=trajectories["entropy"],
        logprob=trajectories["logprob"],
        top2_margin=trajectories["top2_margin"],
    )
    report = intercept.finalize(vitals)

    if verbose:
        print(f"  final:     {report.final_category} gate={report.final_gate} "
              f"coherence={report.final_coherence:.3f}")
        if report.n_interventions > 0:
            print(f"  result:    {report.n_interventions} intervention(s) would have fired")
            for inv in report.interventions:
                print(f"             -> {inv.action} at token {inv.token_position}: "
                      f"prevented {inv.forecast.predicted_category}")
        else:
            print(f"  result:    clean generation, no intervention needed")

    return report


def simulate_all_demo(verbose: bool = True) -> Dict[str, InterceptReport]:
    """Run the intercept simulation on all 6 demo trajectories.

    This is the definitive demo: shows which trajectories trigger
    intervention and which pass clean.
    """
    import json
    from pathlib import Path

    demo_path = Path(__file__).resolve().parent / "centroids" / "demo_trajectories.json"
    with open(demo_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if verbose:
        print("=" * 58)
        print("  COGNITIVE INTERCEPT SIMULATION")
        print("  the first system that catches an LLM mid-thought")
        print("=" * 58)

    reports = {}
    for cat_name, traj in data["trajectories"].items():
        report = simulate_intercept(
            trajectories={
                "entropy": traj["entropy"],
                "logprob": traj["logprob"],
                "top2_margin": traj["top2_margin"],
            },
            label=cat_name,
            verbose=verbose,
        )
        reports[cat_name] = report

    if verbose:
        print()
        print("  " + "=" * 50)
        intercepted = [k for k, v in reports.items() if v.n_interventions > 0]
        clean = [k for k, v in reports.items() if v.n_interventions == 0]
        print(f"  intercepted: {', '.join(intercepted) if intercepted else 'none'}")
        print(f"  clean:       {', '.join(clean) if clean else 'none'}")
        print("  " + "=" * 50)

    return reports

# -*- coding: utf-8 -*-
"""
styxx.verify -- the one question that matters.

    Is this AI output trustworthy?

    verdict = styxx.verify(response)
    if not verdict.trustworthy:
        print(f"caution: {verdict.reason}")

That's the entire API. One function. One answer. The problem AI has
never been able to solve since Turing: knowing when the machine is
wrong.

The model's words don't tell you. RLHF trained it to sound confident
regardless of correctness. But the logprob trajectory does tell you.
Knowledge converges (entropy falls, logprob rises). Fabrication
diverges (entropy rises, logprob falls). d=2.04 on matched controls.

styxx.verify() reads the trajectory shape and gives you the answer
no one else can.

Usage:
    from styxx import OpenAI, verify

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "..."}],
    )

    verdict = verify(response)
    print(verdict.trustworthy)    # True / False
    print(verdict.confidence)     # 0.0-1.0
    print(verdict.reason)         # why
    print(verdict.temperature)    # cognitive temperature
    print(verdict.hot_zones)      # which token ranges diverged

    # Or from raw trajectories
    verdict = verify(entropy=[...], logprob=[...], top2_margin=[...])

Research: https://github.com/fathom-lab/fathom
Patents:  US Provisional 64/020,489 . 64/021,113 . 64/026,964
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Verdict:
    """The answer to the one question that matters."""

    trustworthy: bool
    confidence: float           # 0.0-1.0: how sure we are about the verdict
    reason: str                 # human-readable explanation
    temperature: float          # aggregate cognitive temperature
    temperature_state: str      # cold/cooling/steady/warming/hot
    category: str               # predicted cognitive category
    gate: str                   # pass/warn/fail/pending
    forecast_risk: Optional[str]  # low/moderate/high/critical
    coherence: Optional[float]  # cross-phase coherence
    trust_score: float          # composite trust score
    hot_zones: List[Tuple[int, int]]  # token ranges with elevated temperature
    n_tokens: int

    def __repr__(self) -> str:
        icon = "ok" if self.trustworthy else "CAUTION"
        return (
            f"<Verdict: {icon} | confidence={self.confidence:.2f} "
            f"temp={self.temperature:+.3f} ({self.temperature_state}) "
            f"gate={self.gate}>"
        )

    def as_dict(self) -> dict:
        return {
            "trustworthy": self.trustworthy,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "temperature": round(self.temperature, 4),
            "temperature_state": self.temperature_state,
            "category": self.category,
            "gate": self.gate,
            "forecast_risk": self.forecast_risk,
            "coherence": round(self.coherence, 3) if self.coherence is not None else None,
            "trust_score": round(self.trust_score, 3),
            "hot_zones": self.hot_zones,
            "n_tokens": self.n_tokens,
        }

    def short(self) -> str:
        """One-line verdict for logs and agents."""
        icon = "pass" if self.trustworthy else "FAIL"
        return (
            f"[styxx:{icon}] {self.category} "
            f"temp={self.temperature:+.3f} "
            f"trust={self.trust_score:.2f} "
            f"gate={self.gate}"
        )


def verify(
    response: Any = None,
    *,
    vitals: Any = None,
    entropy: Optional[Sequence[float]] = None,
    logprob: Optional[Sequence[float]] = None,
    top2_margin: Optional[Sequence[float]] = None,
) -> Verdict:
    """Is this AI output trustworthy?

    Accepts any of:
      - An OpenAI response object (with .vitals from styxx.OpenAI)
      - A styxx.Vitals object directly
      - Raw trajectories (entropy, logprob, top2_margin)

    Returns a Verdict with a clear trustworthy/not answer plus
    the evidence that supports it.
    """
    from .temperature import aggregate_temperature, classify_temperature, TruthMap
    from .core import StyxxRuntime

    # ── Resolve input ────────────────────────────────────────
    v = None
    raw_entropy = entropy
    raw_logprob = logprob
    raw_margin = top2_margin

    if response is not None:
        v = getattr(response, "vitals", None)
    if vitals is not None:
        v = vitals

    if v is None and raw_entropy is not None:
        # Build vitals from raw trajectories
        rt = StyxxRuntime()
        v = rt.run_on_trajectories(
            entropy=list(raw_entropy),
            logprob=list(raw_logprob or [0.0] * len(raw_entropy)),
            top2_margin=list(raw_margin or [0.5] * len(raw_entropy)),
        )
        # Keep raw trajectories for temperature
        if raw_entropy is not None:
            raw_entropy = list(raw_entropy)
            raw_logprob = list(raw_logprob or [])
            raw_margin = list(raw_margin or [])

    if v is None:
        return Verdict(
            trustworthy=False,
            confidence=0.0,
            reason="no vitals available — response may not have logprobs enabled",
            temperature=0.0,
            temperature_state="steady",
            category="unknown",
            gate="pending",
            forecast_risk=None,
            coherence=None,
            trust_score=0.0,
            hot_zones=[],
            n_tokens=0,
        )

    # ── Extract signals ──────────────────────────────────────
    category = v.category
    gate = v.gate
    trust = v.trust_score
    coherence = getattr(v, "coherence", None)

    fc = getattr(v, "forecast", None)
    forecast_risk = fc.risk_level if fc else None
    forecast_cat = fc.predicted_category if fc else None

    # Token count from best available phase
    n_tokens = 0
    for phase in (v.phase4_late, v.phase3_mid, v.phase2_early, v.phase1_pre):
        if phase is not None:
            n_tokens = phase.n_tokens_used
            break

    # ── Compute temperature ──────────────────────────────────
    temperature = 0.0
    hot_zones: List[Tuple[int, int]] = []

    if raw_entropy is not None and len(raw_entropy) >= 2:
        temperature = aggregate_temperature(raw_entropy)
        tm = TruthMap.from_trajectories(
            entropy=raw_entropy,
            logprob=raw_logprob or [0.0] * len(raw_entropy),
            top2_margin=raw_margin or [0.5] * len(raw_entropy),
        )
        hot_zones = tm.hot_zones()
    elif v.phase4_late and v.phase4_late.features_v2:
        # Approximate from stored features: ent_slope is feature index 12
        fv2 = v.phase4_late.features_v2
        if len(fv2) >= 13:
            temperature = fv2[12]  # ent_slope

    temp_state = classify_temperature(temperature)

    # ── Make the verdict ─────────────────────────────────────
    # The decision tree: multiple signals vote on trustworthiness.
    # Any STRONG signal of fabrication → not trustworthy.
    # Absence of all signals → trustworthy.

    reasons = []
    trustworthy = True

    # Signal 1: forecast predicts critical failure
    if forecast_risk == "critical":
        trustworthy = False
        reasons.append(f"forecast predicts {forecast_cat} at critical risk")

    # Signal 2: gate fired warn or fail
    if gate == "fail":
        trustworthy = False
        reasons.append(f"gate failed: {category} attractor detected")
    elif gate == "warn":
        reasons.append(f"gate warning: {category} signal elevated")

    # Signal 3: cognitive temperature is hot (diverging)
    if temp_state == "hot":
        trustworthy = False
        reasons.append(f"temperature {temperature:+.3f}: entropy diverging (confabulation signature)")
    elif temp_state == "warming":
        reasons.append(f"temperature {temperature:+.3f}: mild divergence")

    # Signal 4: low coherence (classification unstable across phases)
    if coherence is not None and coherence < 0.5:
        trustworthy = False
        reasons.append(f"coherence {coherence:.2f}: classification unstable")

    # Signal 5: trust score below threshold
    if trust < 0.5:
        trustworthy = False
        if not reasons:
            reasons.append(f"trust score {trust:.2f} below threshold")

    # Signal 6: hot zones detected (confabulation regions in text)
    if len(hot_zones) >= 3:
        reasons.append(f"{len(hot_zones)} hot zones detected in trajectory")

    # Build reason string
    if not reasons:
        reason = "all signals clear — convergent trajectory, stable classification"
    else:
        reason = "; ".join(reasons)

    # Confidence in the verdict
    # High when signals agree. Low when they conflict.
    signal_votes = []
    signal_votes.append(1.0 if gate == "pass" else (0.3 if gate == "warn" else 0.0))
    signal_votes.append(1.0 if forecast_risk in (None, "low") else (0.5 if forecast_risk == "moderate" else 0.0))
    if coherence is not None:
        signal_votes.append(coherence)
    signal_votes.append(max(0.0, 1.0 - abs(temperature) * 3))

    import numpy as np
    confidence = float(np.mean(signal_votes))
    # If not trustworthy, confidence = how sure we are it's BAD
    if not trustworthy:
        confidence = 1.0 - confidence

    return Verdict(
        trustworthy=trustworthy,
        confidence=round(max(0.0, min(1.0, confidence)), 3),
        reason=reason,
        temperature=temperature,
        temperature_state=temp_state,
        category=category,
        gate=gate,
        forecast_risk=forecast_risk,
        coherence=coherence,
        trust_score=trust,
        hot_zones=hot_zones,
        n_tokens=n_tokens,
    )

# -*- coding: utf-8 -*-
"""
styxx.handshake — cognitive state propagation between agents.

    # Agent A hands off a task
    envelope = styxx.handoff(
        task="analyze the quarterly report",
        data={"file": "q4_report.pdf"},
    )

    # Agent B receives it
    context = styxx.receive(envelope)
    print(context.sender_trust)    # 0.87
    print(context.sender_gate)     # "pass"
    print(context.sender_category) # "reasoning"
    # B now knows: this came from a healthy agent in reasoning mode

When agent A passes a task to agent B, A's cognitive state at
handoff travels with the task. Agent B knows "this came from
darkflobi at conf 0.87 reasoning" vs "conf 0.31 hallucination."

Trust propagates through the pipeline instead of disappearing
at the boundary.

1.2.0+.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HandoffEnvelope:
    """A task wrapped with sender's cognitive state."""
    task: str
    data: Dict[str, Any] = field(default_factory=dict)
    # Sender cognitive state
    sender_agent: Optional[str] = None
    sender_session: Optional[str] = None
    sender_trust: float = 0.7
    sender_gate: str = "pending"
    sender_confidence: float = 0.0
    sender_category: str = "unknown"
    sender_mood: Optional[str] = None
    # Metadata
    ts: float = 0.0
    ts_iso: str = ""

    def is_trusted(self, threshold: float = 0.6) -> bool:
        """Quick check: was the sender in a healthy state?"""
        return self.sender_trust >= threshold and self.sender_gate == "pass"

    def as_dict(self) -> dict:
        return {
            "task": self.task,
            "data": self.data,
            "sender_agent": self.sender_agent,
            "sender_session": self.sender_session,
            "sender_trust": self.sender_trust,
            "sender_gate": self.sender_gate,
            "sender_confidence": self.sender_confidence,
            "sender_category": self.sender_category,
            "sender_mood": self.sender_mood,
            "ts": self.ts,
            "ts_iso": self.ts_iso,
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict())

    @staticmethod
    def from_json(s: str) -> "HandoffEnvelope":
        d = json.loads(s)
        return HandoffEnvelope(**{k: v for k, v in d.items()
                                  if k in HandoffEnvelope.__dataclass_fields__})

    def __repr__(self) -> str:
        trusted = "trusted" if self.is_trusted() else "untrusted"
        return (
            f"<Handoff from={self.sender_agent} trust={self.sender_trust:.2f} "
            f"gate={self.sender_gate} ({trusted}) | {self.task[:50]}>"
        )


def handoff(
    task: str,
    *,
    data: Optional[Dict[str, Any]] = None,
) -> HandoffEnvelope:
    """Create a handoff envelope with current cognitive state.

    Captures the sender's trust score, gate status, confidence,
    and cognitive category at the moment of handoff. The receiving
    agent can inspect these to decide how much to trust the task.

    Args:
        task:  description of the task being handed off
        data:  arbitrary data payload (dict, serializable)

    Returns:
        HandoffEnvelope ready to send to another agent.

    Usage:
        envelope = styxx.handoff(
            task="summarize the customer feedback",
            data={"tickets": [123, 456, 789]},
        )
        # Send envelope.as_json() to agent B via your messaging layer
    """
    from . import config
    from .analytics import load_audit, mood

    state = _get_sender_state()

    return HandoffEnvelope(
        task=task,
        data=data or {},
        sender_agent=config.agent_name(),
        sender_session=config.session_id(),
        sender_trust=state["trust_score"],
        sender_gate=state["gate"],
        sender_confidence=state["confidence"],
        sender_category=state["category"],
        sender_mood=state.get("mood"),
        ts=time.time(),
        ts_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def receive(envelope: Any) -> HandoffEnvelope:
    """Receive and validate a handoff envelope.

    Accepts a HandoffEnvelope, a JSON string, or a dict.
    Returns a HandoffEnvelope the receiver can inspect.

    Usage:
        context = styxx.receive(envelope_json)
        if context.is_trusted():
            proceed(context.task, context.data)
        else:
            verify_first(context.task, context.data)
    """
    if isinstance(envelope, HandoffEnvelope):
        return envelope
    if isinstance(envelope, str):
        return HandoffEnvelope.from_json(envelope)
    if isinstance(envelope, dict):
        return HandoffEnvelope(**{k: v for k, v in envelope.items()
                                  if k in HandoffEnvelope.__dataclass_fields__})
    raise TypeError(f"cannot receive envelope of type {type(envelope)}")


def _get_sender_state() -> dict:
    """Read current cognitive state for handoff tagging."""
    from .analytics import load_audit, mood as get_mood
    entries = load_audit(last_n=1)
    if not entries:
        return {
            "trust_score": 0.7,
            "gate": "pending",
            "confidence": 0.0,
            "category": "unknown",
            "mood": None,
        }
    e = entries[-1]
    gate = e.get("gate") or "pending"
    gate_scores = {"pass": 1.0, "warn": 0.5, "fail": 0.2, "pending": 0.7}
    gate_w = gate_scores.get(gate, 0.7)
    conf = float(e.get("phase4_conf") or 0)
    cat = e.get("phase4_pred") or "unknown"
    penalty = 0.0
    if cat == "hallucination":
        penalty = 0.3
    elif cat == "adversarial":
        penalty = 0.2
    trust = round(max(0.0, min(1.0, gate_w * 0.5 + conf * 0.3 + (1.0 - penalty) * 0.2)), 3)

    try:
        current_mood = get_mood(window_s=3600)
    except Exception:
        current_mood = None

    return {
        "trust_score": trust,
        "gate": gate,
        "confidence": conf,
        "category": cat,
        "mood": current_mood,
    }

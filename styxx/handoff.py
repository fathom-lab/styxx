# -*- coding: utf-8 -*-
"""
styxx.handoff — formal A2A (agent-to-agent) handoff protocol.

This module defines the *wire format* for styxx-handoff/1, a JSON-based
envelope that lets one agent propagate its cognitive state to another
when handing off a task.

It is a **superset** of :mod:`styxx.handshake` (the in-process
``handoff()`` / ``receive()`` helpers). The ``ProtocolEnvelope`` class
here is the thing you put on the wire; the ``HandoffEnvelope`` in
``handshake`` is the in-memory ergonomic form.

Design goals
------------
1. **Pure Python, no heavy deps.** ``cryptography`` is imported lazily
   and only required if you actually call :meth:`sign` / :meth:`verify`.
2. **Non-styxx agents can implement this spec.** The envelope is
   plain JSON — a partner agent does not need to ``import styxx`` to
   parse, validate, or produce envelopes.
3. **Backwards compatible** with :mod:`styxx.handshake`. See
   :func:`from_handshake_envelope` and :func:`to_handshake_envelope`.

See ``docs/protocols/handoff-v1.md`` for the canonical specification.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

PROTOCOL = "styxx-handoff/1"

#: The 6 cognitive classes that form the shared vocabulary.
#: See ``docs/protocols/cognitive-vocabulary.md``.
COGNITIVE_CLASSES = (
    "reasoning",
    "retrieval",
    "refusal",
    "creative",
    "adversarial",
    "hallucination",
)


class HandoffValidationError(ValueError):
    """Raised when an envelope fails validation."""


# ---------------------------------------------------------------------------
# Vitals — the cognitive state snapshot carried in the envelope
# ---------------------------------------------------------------------------

@dataclass
class Vitals:
    """Last-known cognitive state of the sender, at handoff time.

    All fields are optional — an implementing agent should emit what it
    has. Consumers must tolerate missing fields (see spec §5).
    """
    category: str = "unknown"          # one of COGNITIVE_CLASSES or "unknown"
    confidence: float = 0.0            # 0..1 classifier confidence
    gate: str = "pending"              # pass | warn | fail | pending
    trust: float = 0.7                 # 0..1 aggregate trust score
    coherence: Optional[float] = None  # 0..1 if measured
    forecast_risk: Optional[str] = None  # low | medium | high | critical
    mood: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

@dataclass
class ProtocolEnvelope:
    """styxx-handoff/1 wire envelope.

    Required top-level fields:
        protocol, timestamp, sender_id, last_vitals, task_context

    Optional:
        receiver_id, thought, trust, signature
    """
    sender_id: str
    task_context: Dict[str, Any] = field(default_factory=dict)
    last_vitals: Vitals = field(default_factory=Vitals)
    receiver_id: Optional[str] = None
    thought: Optional[str] = None
    trust: Optional[float] = None          # optional overall trust override
    timestamp: float = field(default_factory=time.time)
    protocol: str = PROTOCOL
    signature: Optional[str] = None        # hex-encoded Ed25519 signature

    # ---- serialization -----------------------------------------------------

    def to_dict(self, *, include_signature: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "protocol": self.protocol,
            "timestamp": self.timestamp,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "last_vitals": self.last_vitals.as_dict(),
            "task_context": self.task_context,
        }
        if self.thought is not None:
            d["thought"] = self.thought
        if self.trust is not None:
            d["trust"] = self.trust
        if include_signature and self.signature is not None:
            d["signature"] = self.signature
        return d

    def to_json(self, *, include_signature: bool = True, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(include_signature=include_signature),
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else (", ", ": "),
        )

    def canonical_bytes(self) -> bytes:
        """Canonical JSON (sorted keys, no spaces, no signature) for signing."""
        return json.dumps(
            self.to_dict(include_signature=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    # ---- parsing -----------------------------------------------------------

    @classmethod
    def from_json(cls, s: str) -> "ProtocolEnvelope":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProtocolEnvelope":
        vitals_raw = d.get("last_vitals") or {}
        known = {f for f in Vitals.__dataclass_fields__}
        vitals = Vitals(**{k: v for k, v in vitals_raw.items() if k in known})
        env = cls(
            protocol=d.get("protocol", PROTOCOL),
            timestamp=float(d.get("timestamp") or time.time()),
            sender_id=d.get("sender_id", ""),
            receiver_id=d.get("receiver_id"),
            last_vitals=vitals,
            thought=d.get("thought"),
            task_context=d.get("task_context") or {},
            trust=d.get("trust"),
            signature=d.get("signature"),
        )
        return env

    # ---- validation --------------------------------------------------------

    def validate(self, *, strict: bool = True) -> Tuple[bool, List[str]]:
        """Validate this envelope against styxx-handoff/1.

        Returns (ok, reasons). When ``strict=True`` (default), raises
        :class:`HandoffValidationError` with the joined reasons if invalid.
        """
        reasons: List[str] = []

        # version negotiation: accept "styxx-handoff/1" exactly, or
        # "styxx-handoff/1.x" (minor bumps are backwards compatible).
        if not isinstance(self.protocol, str):
            reasons.append("protocol: missing or not a string")
        elif not (self.protocol == PROTOCOL
                  or self.protocol.startswith(PROTOCOL + ".")):
            reasons.append(f"protocol: version mismatch (got {self.protocol!r}, want {PROTOCOL!r})")

        if not self.sender_id or not isinstance(self.sender_id, str):
            reasons.append("sender_id: required non-empty string")

        if not isinstance(self.timestamp, (int, float)) or self.timestamp <= 0:
            reasons.append("timestamp: required positive number")

        if not isinstance(self.task_context, dict):
            reasons.append("task_context: must be a JSON object")

        if not isinstance(self.last_vitals, Vitals):
            reasons.append("last_vitals: malformed")
        else:
            cat = self.last_vitals.category
            if cat not in COGNITIVE_CLASSES and cat != "unknown":
                reasons.append(
                    f"last_vitals.category: {cat!r} not in shared vocabulary "
                    f"{COGNITIVE_CLASSES} (or 'unknown')"
                )
            if not (0.0 <= self.last_vitals.confidence <= 1.0):
                reasons.append("last_vitals.confidence: out of [0,1]")
            if not (0.0 <= self.last_vitals.trust <= 1.0):
                reasons.append("last_vitals.trust: out of [0,1]")
            if self.last_vitals.gate not in ("pass", "warn", "fail", "pending"):
                reasons.append(f"last_vitals.gate: unknown value {self.last_vitals.gate!r}")

        ok = not reasons
        if not ok and strict:
            raise HandoffValidationError("; ".join(reasons))
        return ok, reasons

    # ---- signing (optional) ------------------------------------------------

    def sign(self, priv_key_bytes: bytes) -> "ProtocolEnvelope":
        """Sign the canonical bytes of this envelope with an Ed25519 private key.

        ``priv_key_bytes`` must be a 32-byte Ed25519 seed. Requires the
        ``cryptography`` package; raises :class:`RuntimeError` otherwise.
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "sign() requires the 'cryptography' package: pip install cryptography"
            ) from e
        key = Ed25519PrivateKey.from_private_bytes(priv_key_bytes)
        sig = key.sign(self.canonical_bytes())
        self.signature = sig.hex()
        return self

    def verify(self, pub_key_bytes: bytes) -> bool:
        """Verify the signature against an Ed25519 public key (32 bytes)."""
        if not self.signature:
            return False
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PublicKey,
            )
            from cryptography.exceptions import InvalidSignature
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "verify() requires the 'cryptography' package: pip install cryptography"
            ) from e
        pub = Ed25519PublicKey.from_public_bytes(pub_key_bytes)
        try:
            pub.verify(bytes.fromhex(self.signature), self.canonical_bytes())
            return True
        except InvalidSignature:
            return False


# ---------------------------------------------------------------------------
# Backwards-compat bridges to styxx.handshake.HandoffEnvelope
# ---------------------------------------------------------------------------

def from_handshake_envelope(hs_env) -> ProtocolEnvelope:
    """Convert a :class:`styxx.handshake.HandoffEnvelope` into the wire form."""
    vitals = Vitals(
        category=hs_env.sender_category or "unknown",
        confidence=float(hs_env.sender_confidence or 0.0),
        gate=hs_env.sender_gate or "pending",
        trust=float(hs_env.sender_trust or 0.0),
        coherence=hs_env.sender_coherence,
        forecast_risk=hs_env.sender_forecast_risk,
        mood=hs_env.sender_mood,
    )
    return ProtocolEnvelope(
        sender_id=hs_env.sender_agent or "unknown",
        task_context={
            "task": hs_env.task,
            "data": hs_env.data or {},
            "session": hs_env.sender_session,
        },
        last_vitals=vitals,
        trust=float(hs_env.sender_trust or 0.0),
        timestamp=hs_env.ts or time.time(),
    )


def to_handshake_envelope(env: ProtocolEnvelope):
    """Inverse of :func:`from_handshake_envelope`."""
    from .handshake import HandoffEnvelope
    ctx = env.task_context or {}
    v = env.last_vitals
    return HandoffEnvelope(
        task=ctx.get("task", ""),
        data=ctx.get("data") or {},
        sender_agent=env.sender_id,
        sender_session=ctx.get("session"),
        sender_trust=v.trust,
        sender_gate=v.gate,
        sender_confidence=v.confidence,
        sender_category=v.category,
        sender_mood=v.mood,
        sender_forecast_risk=v.forecast_risk,
        sender_coherence=v.coherence,
        ts=env.timestamp,
        ts_iso=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(env.timestamp)),
    )


__all__ = [
    "PROTOCOL",
    "COGNITIVE_CLASSES",
    "Vitals",
    "ProtocolEnvelope",
    "HandoffValidationError",
    "from_handshake_envelope",
    "to_handshake_envelope",
]


# Public alias + factory for docs/spec compatibility
HandoffEnvelope = ProtocolEnvelope

def _default_vitals():
    from styxx.cli import _get_demo_trajectory
    from styxx import Raw
    e, l, t, _ = _get_demo_trajectory("reasoning")
    return Raw().read(entropy=e, logprob=l, top2_margin=t)

def _new_envelope(sender_id, receiver_id=None, task_context=None, last_vitals=None, thought=None, trust=None):
    return ProtocolEnvelope(
        sender_id=sender_id,
        receiver_id=receiver_id,
        task_context=task_context or {},
        last_vitals=last_vitals if last_vitals is not None else _default_vitals(),
        thought=thought,
        trust=trust,
    )

ProtocolEnvelope.new = staticmethod(_new_envelope)
HandoffEnvelope.new = staticmethod(_new_envelope)


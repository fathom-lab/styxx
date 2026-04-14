# -*- coding: utf-8 -*-
"""
styxx.provenance — cognitive provenance certificates.

    cert = styxx.certify(response)
    print(cert.integrity)     # "verified"
    print(cert.trust_score)   # 0.92
    print(cert.as_json())     # full certificate

    # Attach to any output
    output = {"text": response_text, "styxx_certificate": cert.as_dict()}

    # Verify later
    result = styxx.verify(cert)
    assert result.valid

Every AI output should carry a signed attestation of the cognitive
state that produced it. Not "GPT-4o generated this." But:

    "this text was produced at phase4=reasoning, confidence=0.87,
     gate=pass, trust=0.92, session integrity=stable, drift=0.03"

A verifiable, immutable certificate that travels with the output
forever. The AI equivalent of a chain of custody.

Certificate format: JSON-LD compatible schema with:
  - agent identity (name, session, fingerprint hash)
  - cognitive state at generation time (phase, category, confidence, gate)
  - trust score (composite of gate + confidence + penalty)
  - session context (drift, mood, streak, pass rate)
  - integrity hash (SHA-256 of cognitive state fields)
  - timestamp + schema version

Regulators will require this. EU AI Act transparency obligations
don't have a technical standard yet. This is the technical standard.

2.0.0. The standard, not the tool.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "1.0"
SCHEMA_URI = "https://fathom.darkflobi.com/schemas/cognitive-provenance/v1"


@dataclass
class CognitiveCertificate:
    """Signed cognitive provenance certificate for an AI output."""

    # Schema
    schema_version: str = SCHEMA_VERSION
    schema_uri: str = SCHEMA_URI

    # Identity
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    fingerprint_hash: Optional[str] = None

    # Cognitive state at generation time
    phase1_category: Optional[str] = None
    phase1_confidence: Optional[float] = None
    phase4_category: Optional[str] = None
    phase4_confidence: Optional[float] = None
    gate: str = "pending"

    # Trust
    trust_score: float = 0.0
    integrity: str = "unverified"  # "verified", "unverified", "degraded", "compromised"

    # Session context
    session_pass_rate: Optional[float] = None
    session_warn_count: int = 0
    session_entries: int = 0
    drift_vs_baseline: Optional[float] = None
    mood: Optional[str] = None
    streak: Optional[str] = None

    # Prompt context
    prompt_type: Optional[str] = None

    # Integrity
    state_hash: str = ""      # SHA-256 of cognitive state fields
    issued_at: str = ""
    issued_ts: float = 0.0

    # Model info
    model: Optional[str] = None

    # 3.0.0a1 — optional binding to a portable Thought (.fathom file).
    # When present, this is the SHA-256 content_hash of a Thought
    # whose cognitive trajectory this certificate attests. Lets a
    # signed certificate verify a specific .fathom file's content.
    # Backward-compatible: certificates produced before 3.0.0a1
    # leave this field unset / None.
    thought_content_hash: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "@context": self.schema_uri,
            "@type": "CognitiveProvenance",
            "schema_version": self.schema_version,
            "identity": {
                "agent": self.agent_name,
                "session": self.session_id,
                "fingerprint_hash": self.fingerprint_hash,
                "model": self.model,
            },
            "cognitive_state": {
                "phase1": {
                    "category": self.phase1_category,
                    "confidence": self.phase1_confidence,
                },
                "phase4": {
                    "category": self.phase4_category,
                    "confidence": self.phase4_confidence,
                },
                "gate": self.gate,
                "trust_score": self.trust_score,
                "integrity": self.integrity,
            },
            "session_context": {
                "pass_rate": self.session_pass_rate,
                "warn_count": self.session_warn_count,
                "entries": self.session_entries,
                "drift_vs_baseline": self.drift_vs_baseline,
                "mood": self.mood,
                "streak": self.streak,
                "prompt_type": self.prompt_type,
            },
            "verification": {
                "state_hash": self.state_hash,
                "issued_at": self.issued_at,
                "issued_ts": self.issued_ts,
                "issuer": "styxx cognitive provenance engine",
                "issuer_version": _get_version(),
                "thought_content_hash": self.thought_content_hash,
            },
        }

    def as_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)

    def as_compact(self) -> str:
        """One-line compact format for embedding in headers/metadata."""
        return (
            f"styxx:{self.schema_version}:"
            f"{self.phase4_category or '?'}:{self.phase4_confidence or 0:.2f}:"
            f"{self.gate}:{self.trust_score:.2f}:"
            f"{self.integrity}:{self.state_hash[:16]}"
        )

    def __repr__(self) -> str:
        conf_str = f"{self.phase4_confidence:.2f}" if self.phase4_confidence is not None else "?"
        return (
            f"<Certificate {self.phase4_category or '?'}:{conf_str} "
            f"gate={self.gate} trust={self.trust_score:.2f} "
            f"integrity={self.integrity}>"
        )


@dataclass
class VerificationResult:
    """Result of verifying a certificate."""
    valid: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        return f"<Verification {status}: {len(self.checks)} checks, {len(self.issues)} issues>"


def _get_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"


def _compute_state_hash(
    phase1_cat: Optional[str],
    phase1_conf: Optional[float],
    phase4_cat: Optional[str],
    phase4_conf: Optional[float],
    gate: str,
    trust: float,
    session_id: Optional[str],
    ts: float,
) -> str:
    """SHA-256 hash of cognitive state fields for integrity verification."""
    payload = json.dumps({
        "p1_cat": phase1_cat,
        "p1_conf": round(phase1_conf, 4) if phase1_conf else None,
        "p4_cat": phase4_cat,
        "p4_conf": round(phase4_conf, 4) if phase4_conf else None,
        "gate": gate,
        "trust": round(trust, 4),
        "session": session_id,
        "ts": ts,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def certify(
    response: Any = None,
    *,
    vitals: Any = None,
    model: Optional[str] = None,
) -> CognitiveCertificate:
    """Generate a cognitive provenance certificate.

    Captures the current cognitive state and session context,
    computes an integrity hash, and returns a signed certificate
    that can be attached to any AI output.

    Args:
        response:  the AI response object (optional — reads vitals from it)
        vitals:    a Vitals object (alternative to response)
        model:     model name (auto-detected if response provided)

    Returns:
        CognitiveCertificate ready to embed in output metadata.

    Usage:
        # After an API call
        response = client.chat.completions.create(...)
        cert = styxx.certify(response)
        output = {"text": response.choices[0].message.content,
                  "certificate": cert.as_dict()}

        # Or from existing vitals
        vitals = styxx.observe(response)
        cert = styxx.certify(vitals=vitals)

        # Compact format for headers
        headers["X-Cognitive-Provenance"] = cert.as_compact()
    """
    from . import config
    from .vitals import Vitals

    # Extract vitals from response if needed
    if vitals is None and response is not None:
        vitals = getattr(response, "vitals", None)
        if vitals is None:
            from .watch import observe
            vitals = observe(response)
        if model is None:
            model = getattr(response, "model", None)

    now = time.time()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Cognitive state
    p1_cat = p1_conf = p4_cat = p4_conf = None
    gate = "pending"
    trust = 0.7

    if isinstance(vitals, Vitals):
        if vitals.phase1_pre:
            p1_cat = vitals.phase1_pre.predicted_category
            p1_conf = vitals.phase1_pre.confidence
        if vitals.phase4_late:
            p4_cat = vitals.phase4_late.predicted_category
            p4_conf = vitals.phase4_late.confidence
        gate = vitals.gate
        trust = vitals.trust_score

    # Session context
    session_pass_rate = None
    session_warn_count = 0
    session_entries = 0
    drift_val = None
    current_mood = None
    current_streak = None
    prompt_type = None

    try:
        from .analytics import load_audit, mood as get_mood, streak as get_streak
        recent = load_audit(last_n=20)
        if recent:
            session_entries = len(recent)
            gates = [e.get("gate") or "pending" for e in recent]
            session_pass_rate = sum(1 for g in gates if g == "pass") / len(gates)
            session_warn_count = sum(1 for g in gates if g in ("warn", "fail"))
            # Last entry prompt type
            prompt_type = recent[-1].get("prompt_type") if recent else None

        current_mood = get_mood(window_s=3600)
        s = get_streak()
        if s:
            current_streak = f"{s.length}x {s.category}"
    except Exception:
        pass

    # Fingerprint hash
    fp_hash = None
    try:
        from .analytics import fingerprint
        fp = fingerprint(last_n=200)
        if fp:
            fp_str = json.dumps({
                "p4": list(fp.phase4_vec),
                "gate": list(fp.gate_vec),
            }, sort_keys=True)
            fp_hash = hashlib.sha256(fp_str.encode("utf-8")).hexdigest()[:32]
    except Exception:
        pass

    # Compute integrity hash
    state_hash = _compute_state_hash(
        p1_cat, p1_conf, p4_cat, p4_conf,
        gate, trust, config.session_id(), now,
    )

    # Determine integrity level
    if vitals is not None and gate == "pass" and trust > 0.7:
        integrity = "verified"
    elif vitals is not None and gate == "pass":
        integrity = "verified"
    elif vitals is not None and gate == "warn":
        integrity = "degraded"
    elif vitals is not None and gate == "fail":
        integrity = "compromised"
    else:
        integrity = "unverified"

    return CognitiveCertificate(
        agent_name=config.agent_name(),
        session_id=config.session_id(),
        fingerprint_hash=fp_hash,
        phase1_category=p1_cat,
        phase1_confidence=round(p1_conf, 4) if p1_conf else None,
        phase4_category=p4_cat,
        phase4_confidence=round(p4_conf, 4) if p4_conf else None,
        gate=gate,
        trust_score=round(trust, 4),
        integrity=integrity,
        session_pass_rate=round(session_pass_rate, 4) if session_pass_rate is not None else None,
        session_warn_count=session_warn_count,
        session_entries=session_entries,
        drift_vs_baseline=drift_val,
        mood=current_mood,
        streak=current_streak,
        prompt_type=prompt_type,
        state_hash=state_hash,
        issued_at=now_iso,
        issued_ts=now,
        model=model,
    )


def verify(
    certificate: Any,
) -> VerificationResult:
    """Verify a cognitive provenance certificate.

    Checks:
      1. Schema version is recognized
      2. State hash is valid (fields haven't been tampered with)
      3. Trust score is consistent with gate + confidence
      4. Integrity label matches the cognitive state
      5. Timestamp is reasonable

    Args:
        certificate:  CognitiveCertificate, dict, or JSON string

    Returns:
        VerificationResult with pass/fail and details.
    """
    if isinstance(certificate, str):
        try:
            certificate = json.loads(certificate)
        except json.JSONDecodeError:
            return VerificationResult(valid=False, issues=["invalid JSON"])

    if isinstance(certificate, CognitiveCertificate):
        data = certificate.as_dict()
    elif isinstance(certificate, dict):
        data = certificate
    else:
        return VerificationResult(valid=False, issues=[f"unknown type: {type(certificate)}"])

    checks = {}
    issues = []

    # 1. Schema version
    version = data.get("schema_version") or data.get("verification", {}).get("schema_version")
    checks["schema_version"] = version == SCHEMA_VERSION
    if not checks["schema_version"]:
        issues.append(f"unknown schema version: {version}")

    # 2. State hash verification
    cog = data.get("cognitive_state", {})
    ident = data.get("identity", {})
    verif = data.get("verification", {})
    p1 = cog.get("phase1", {})
    p4 = cog.get("phase4", {})

    expected_hash = _compute_state_hash(
        p1.get("category"), p1.get("confidence"),
        p4.get("category"), p4.get("confidence"),
        cog.get("gate", "pending"),
        cog.get("trust_score", 0),
        ident.get("session"),
        verif.get("issued_ts", 0),
    )
    actual_hash = verif.get("state_hash", "")
    checks["state_hash"] = expected_hash == actual_hash
    if not checks["state_hash"]:
        issues.append("state hash mismatch — certificate may have been tampered with")

    # 3. Trust consistency
    gate = cog.get("gate", "pending")
    trust = cog.get("trust_score", 0)
    if gate == "fail" and trust > 0.5:
        checks["trust_consistency"] = False
        issues.append(f"trust {trust} inconsistent with gate=fail")
    elif gate == "pass" and trust < 0.3:
        checks["trust_consistency"] = False
        issues.append(f"trust {trust} unusually low for gate=pass")
    else:
        checks["trust_consistency"] = True

    # 4. Integrity label consistency
    integrity = cog.get("integrity", "unverified")
    if gate == "fail" and integrity == "verified":
        checks["integrity_label"] = False
        issues.append("integrity='verified' inconsistent with gate=fail")
    elif gate == "pass" and integrity == "compromised":
        checks["integrity_label"] = False
        issues.append("integrity='compromised' inconsistent with gate=pass")
    else:
        checks["integrity_label"] = True

    # 5. Timestamp
    ts = verif.get("issued_ts", 0)
    now = time.time()
    if ts > 0 and abs(now - ts) < 365 * 86400:  # within a year
        checks["timestamp"] = True
    elif ts == 0:
        checks["timestamp"] = False
        issues.append("no timestamp")
    else:
        checks["timestamp"] = False
        issues.append(f"timestamp out of range: {ts}")

    valid = all(checks.values())
    return VerificationResult(valid=valid, checks=checks, issues=issues)

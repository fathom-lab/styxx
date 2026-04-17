"""Tests for styxx-handoff/1 (styxx.handoff)."""
from __future__ import annotations

import json
import time

import pytest

from styxx.handoff import (
    PROTOCOL,
    COGNITIVE_CLASSES,
    ProtocolEnvelope,
    Vitals,
    HandoffValidationError,
    from_handshake_envelope,
    to_handshake_envelope,
)
from styxx.handshake import HandoffEnvelope


def _make_env(**over):
    defaults = dict(
        sender_id="agent-a",
        task_context={"task": "test"},
        last_vitals=Vitals(category="reasoning", confidence=0.8, gate="pass", trust=0.8),
    )
    defaults.update(over)
    return ProtocolEnvelope(**defaults)


def test_roundtrip_json():
    env = _make_env()
    wire = env.to_json()
    got = ProtocolEnvelope.from_json(wire)
    assert got.protocol == PROTOCOL
    assert got.sender_id == "agent-a"
    assert got.last_vitals.category == "reasoning"
    assert got.last_vitals.trust == 0.8
    assert got.task_context == {"task": "test"}


def test_validate_ok():
    env = _make_env()
    ok, reasons = env.validate(strict=False)
    assert ok, reasons


def test_version_mismatch_rejected():
    env = _make_env()
    env.protocol = "styxx-handoff/2"
    with pytest.raises(HandoffValidationError) as ei:
        env.validate()
    assert "version mismatch" in str(ei.value)


def test_minor_version_accepted():
    env = _make_env()
    env.protocol = "styxx-handoff/1.5"
    ok, _ = env.validate(strict=False)
    assert ok


def test_missing_sender_id_rejected():
    env = _make_env(sender_id="")
    with pytest.raises(HandoffValidationError):
        env.validate()


def test_bad_category_rejected():
    env = _make_env(last_vitals=Vitals(category="vibes", gate="pass"))
    with pytest.raises(HandoffValidationError) as ei:
        env.validate()
    assert "shared vocabulary" in str(ei.value) or "vocabulary" in str(ei.value)


def test_unknown_category_ok():
    env = _make_env(last_vitals=Vitals(category="unknown", gate="pending"))
    ok, _ = env.validate(strict=False)
    assert ok


def test_all_six_classes_valid():
    for cls in COGNITIVE_CLASSES:
        env = _make_env(last_vitals=Vitals(category=cls, gate="pass", trust=0.6))
        ok, reasons = env.validate(strict=False)
        assert ok, (cls, reasons)


def test_out_of_range_trust_rejected():
    env = _make_env(last_vitals=Vitals(category="reasoning", gate="pass", trust=1.5))
    with pytest.raises(HandoffValidationError):
        env.validate()


def test_unknown_keys_ignored():
    d = _make_env().to_dict()
    d["future_field"] = {"hello": "world"}
    got = ProtocolEnvelope.from_dict(d)
    ok, _ = got.validate(strict=False)
    assert ok


def test_canonical_bytes_excludes_signature():
    env = _make_env()
    env.signature = "deadbeef"
    b = env.canonical_bytes()
    assert b"signature" not in b


def test_signing_roundtrip():
    cryptography = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    priv = Ed25519PrivateKey.generate()
    priv_bytes = priv.private_bytes_raw() if hasattr(priv, "private_bytes_raw") else None
    if priv_bytes is None:
        # Fallback for older cryptography
        from cryptography.hazmat.primitives import serialization
        priv_bytes = priv.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    pub_bytes = priv.public_key().public_bytes_raw() if hasattr(priv.public_key(), "public_bytes_raw") else (
        priv.public_key().public_bytes(
            encoding=__import__("cryptography").hazmat.primitives.serialization.Encoding.Raw,
            format=__import__("cryptography").hazmat.primitives.serialization.PublicFormat.Raw,
        )
    )

    env = _make_env()
    env.sign(priv_bytes)
    assert env.signature and len(env.signature) == 128  # 64 bytes hex

    wire = env.to_json()
    got = ProtocolEnvelope.from_json(wire)
    assert got.verify(pub_bytes) is True

    # tamper
    got.task_context["task"] = "TAMPERED"
    assert got.verify(pub_bytes) is False


def test_works_without_signing():
    """Envelope should work end-to-end without ever touching cryptography."""
    env = _make_env()
    wire = env.to_json()
    got = ProtocolEnvelope.from_json(wire)
    got.validate()
    assert got.signature is None


def test_backcompat_from_handshake_envelope():
    hs = HandoffEnvelope(
        task="do thing",
        data={"x": 1},
        sender_agent="agent-a",
        sender_trust=0.77,
        sender_gate="pass",
        sender_confidence=0.6,
        sender_category="reasoning",
        ts=time.time(),
    )
    env = from_handshake_envelope(hs)
    env.validate()
    assert env.sender_id == "agent-a"
    assert env.last_vitals.category == "reasoning"
    assert env.last_vitals.trust == 0.77
    assert env.task_context["task"] == "do thing"
    assert env.task_context["data"] == {"x": 1}


def test_backcompat_to_handshake_envelope():
    env = _make_env()
    hs = to_handshake_envelope(env)
    assert isinstance(hs, HandoffEnvelope)
    assert hs.sender_agent == "agent-a"
    assert hs.sender_category == "reasoning"
    assert hs.sender_trust == 0.8


def test_existing_handoff_receive_still_work():
    """Do not break the styxx.handoff / styxx.receive public API."""
    import styxx
    assert callable(styxx.handoff)
    assert callable(styxx.receive)
    # receive() should still accept dicts/json/HandoffEnvelope
    hs = HandoffEnvelope(task="t", sender_agent="a", sender_trust=0.9, sender_gate="pass")
    got = styxx.receive(hs.as_json())
    assert got.task == "t"
    assert got.sender_agent == "a"

# -*- coding: utf-8 -*-
"""
test_publish.py -- tests for the publish module.

Covers:
  - prepare_payload() returns expected top-level keys
  - prepare_payload() respects STYXX_DISABLED
  - publish() handles HTTP errors gracefully (mocked urllib)
  - publish() returns result dict on success (mocked urllib)
  - dry-run CLI path (payload only, no POST)

All tests run with no network access. urllib is mocked throughout.
"""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styxx.publish import prepare_payload, publish


# ══════════════════════════════════════════════════════════════════
# 1. prepare_payload returns expected keys
# ══════════════════════════════════════════════════════════════════

def test_prepare_payload_has_required_keys():
    """payload should always contain the core identity fields."""
    payload = prepare_payload("test-agent")
    assert "agent_name" in payload
    assert payload["agent_name"] == "test-agent"
    assert "timestamp" in payload
    assert "ts_iso" in payload
    assert "styxx_version" in payload
    # data sections should be present (may be None if no audit data)
    for key in ("personality", "fingerprint", "weather", "mood", "streak", "log_stats"):
        assert key in payload, f"missing key: {key}"


def test_prepare_payload_days_param():
    """days parameter should be accepted without error."""
    payload = prepare_payload("test-agent", days=1.0)
    assert payload["agent_name"] == "test-agent"


# ══════════════════════════════════════════════════════════════════
# 2. STYXX_DISABLED kill switch
# ══════════════════════════════════════════════════════════════════

def test_prepare_payload_disabled(monkeypatch):
    """when STYXX_DISABLED=1, payload should have disabled=True and
    skip all data collection."""
    monkeypatch.setenv("STYXX_DISABLED", "1")
    payload = prepare_payload("test-agent")
    assert payload.get("disabled") is True
    # should NOT have data keys when disabled
    assert "personality" not in payload
    assert "fingerprint" not in payload
    monkeypatch.delenv("STYXX_DISABLED", raising=False)


def test_prepare_payload_not_disabled(monkeypatch):
    """when STYXX_DISABLED is unset, disabled flag should be absent."""
    monkeypatch.delenv("STYXX_DISABLED", raising=False)
    payload = prepare_payload("test-agent")
    assert "disabled" not in payload


# ══════════════════════════════════════════════════════════════════
# 3. publish() handles HTTP errors gracefully
# ══════════════════════════════════════════════════════════════════

def test_publish_http_error():
    """publish() should return None on HTTP error, not raise."""
    import urllib.error

    exc = urllib.error.HTTPError(
        url="https://example.com",
        code=500,
        msg="Internal Server Error",
        hdrs=None,
        fp=BytesIO(b""),
    )
    with patch("styxx.publish.urllib.request.urlopen", side_effect=exc):
        result = publish("test-agent", "https://example.com/api")
    assert result is None


def test_publish_connection_error():
    """publish() should return None on network errors, not raise."""
    with patch(
        "styxx.publish.urllib.request.urlopen",
        side_effect=ConnectionError("refused"),
    ):
        result = publish("test-agent", "https://example.com/api")
    assert result is None


# ══════════════════════════════════════════════════════════════════
# 4. publish() returns result on success
# ══════════════════════════════════════════════════════════════════

def test_publish_success():
    """publish() should return a dict with status and summary on
    successful POST."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"ok": True}).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("styxx.publish.urllib.request.urlopen", return_value=mock_resp):
        result = publish("test-agent", "https://example.com/api")

    assert result is not None
    assert result["status"] == 200
    assert "test-agent" in result["summary"]
    assert result["response"] == {"ok": True}


# ══════════════════════════════════════════════════════════════════
# 5. payload is valid JSON-serializable
# ══════════════════════════════════════════════════════════════════

def test_payload_json_serializable():
    """the payload must round-trip through json.dumps without error."""
    payload = prepare_payload("test-agent")
    dumped = json.dumps(payload)
    loaded = json.loads(dumped)
    assert loaded["agent_name"] == "test-agent"


# ══════════════════════════════════════════════════════════════════
# 6. CLI dry-run path (no POST made)
# ══════════════════════════════════════════════════════════════════

def test_cli_dry_run(capsys):
    """styxx publish --name x --dry-run should print JSON and not POST."""
    from styxx.cli import main

    with patch("styxx.publish.urllib.request.urlopen") as mock_urlopen:
        rc = main(["publish", "--name", "test-agent", "--dry-run"])

    # should not have called urlopen at all
    mock_urlopen.assert_not_called()
    assert rc == 0

    captured = capsys.readouterr()
    # stdout should contain valid JSON with agent_name
    # strip leading/trailing whitespace, find the JSON blob
    out = captured.out.strip()
    parsed = json.loads(out)
    assert parsed["agent_name"] == "test-agent"

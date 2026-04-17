# -*- coding: utf-8 -*-
"""
styxx.stream — live cognitive telemetry emitter.

When STYXX_STREAM=on (or stream=True passed to autoboot), every
vitals computation is fire-and-forget POSTed to the public relay,
where it powers the agent's live dashboard at

    https://<agent>.live.darkflobi.com

The emitter is:
  * non-blocking (background daemon thread + bounded queue)
  * fail-open (relay down -> user's agent still works perfectly)
  * privacy-safe (vitals only, never prompt or response text)
  * cheap (~200 bytes per event, typical rate < 1/sec)

Public surface
──────────────
    from styxx.stream import enable, disable, is_enabled, emit_vitals
    from styxx.stream import claim_agent, dashboard_url

Env vars
────────
    STYXX_STREAM=on|off           master toggle (default off)
    STYXX_STREAM_RELAY=<url>      override the POST endpoint
    STYXX_STREAM_AGENT=<name>     override agent name (else uses
                                  STYXX_AGENT_NAME or ~/.styxx/config)
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, Optional


# ══════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════

DEFAULT_RELAY = "https://live.darkflobi.com/ingest"
DEFAULT_DASHBOARD_BASE = "https://live.darkflobi.com"

_QUEUE_MAX = 200            # drop events rather than grow unbounded
_POST_TIMEOUT = 3.0         # seconds; short, so we never stall on shutdown
_BACKOFF_MAX = 30.0         # seconds between retries when relay is down


# ══════════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════════

_lock = threading.Lock()
_enabled: bool = False
_agent: Optional[str] = None
_key: Optional[str] = None
_relay: str = DEFAULT_RELAY
_queue: "queue.Queue[dict]" = queue.Queue(maxsize=_QUEUE_MAX)
_worker: Optional[threading.Thread] = None
_backoff: float = 0.0


# ══════════════════════════════════════════════════════════════════
# Credentials
# ══════════════════════════════════════════════════════════════════

def _creds_path() -> Path:
    return Path(os.path.expanduser("~/.styxx/credentials.json"))


def _load_creds() -> dict:
    p = _creds_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_creds(data: dict) -> None:
    p = _creds_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write with mode 600 on POSIX; on Windows rely on user profile perms.
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except (OSError, NotImplementedError):
        pass


# ══════════════════════════════════════════════════════════════════
# Public: claim
# ══════════════════════════════════════════════════════════════════

class ClaimError(Exception):
    pass


def _sanitize_name(raw: str) -> str:
    s = (raw or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        elif ch == " ":
            out.append("-")
    cleaned = "".join(out).strip("-_")
    return cleaned[:32]


def claim_agent(name: str, relay: Optional[str] = None, timeout: float = 10.0) -> dict:
    """Claim an agent name on the live relay.

    Returns {name, key, url}. Persists to ~/.styxx/credentials.json.
    Raises ClaimError on network failure or name conflict.
    """
    clean = _sanitize_name(name)
    if not clean:
        raise ClaimError("invalid name")

    relay = relay or os.environ.get("STYXX_STREAM_RELAY") or DEFAULT_RELAY
    base = relay.rsplit("/", 1)[0]
    url = f"{base}/claim"

    body = json.dumps({"name": clean}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error", str(e))
        except Exception:
            msg = str(e)
        raise ClaimError(msg) from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise ClaimError(f"relay unreachable: {e}") from e

    if "key" not in data or "name" not in data:
        raise ClaimError("malformed response")

    dashboard = data.get("url") or dashboard_url(data["name"])

    creds = _load_creds()
    creds[data["name"]] = {
        "key": data["key"],
        "relay": relay,
        "dashboard": dashboard,
        "claimed_at": time.time(),
    }
    creds["_default"] = data["name"]
    _save_creds(creds)

    return {"name": data["name"], "key": data["key"], "url": dashboard}


def dashboard_url(name: str) -> str:
    """Return the canonical dashboard URL for an agent name."""
    base = os.environ.get("STYXX_DASHBOARD_BASE") or DEFAULT_DASHBOARD_BASE
    return f"{base}/a/{_sanitize_name(name)}"


# ══════════════════════════════════════════════════════════════════
# Public: enable / disable
# ══════════════════════════════════════════════════════════════════

def _pick_agent_and_key() -> tuple[Optional[str], Optional[str]]:
    name = os.environ.get("STYXX_STREAM_AGENT") or os.environ.get("STYXX_AGENT_NAME")
    creds = _load_creds()

    if name:
        clean = _sanitize_name(name)
        entry = creds.get(clean)
        if entry:
            return clean, entry.get("key")
        # Named but no creds: stream anonymously (server may reject).
        return clean, None

    # No explicit name: use the last-claimed default if present.
    default = creds.get("_default")
    if default and default in creds:
        return default, creds[default].get("key")

    return None, None


def enable(
    agent: Optional[str] = None,
    key: Optional[str] = None,
    relay: Optional[str] = None,
) -> bool:
    """Turn on the background emitter. Returns True if enabled.

    Looks up agent/key from creds file when not supplied. Safe to
    call multiple times; the worker thread is singleton.
    """
    global _enabled, _agent, _key, _relay, _worker

    with _lock:
        if agent is None or key is None:
            a, k = _pick_agent_and_key()
            agent = agent or a
            key = key or k

        if not agent:
            return False  # nothing to stream as

        _agent = _sanitize_name(agent)
        _key = key
        _relay = relay or os.environ.get("STYXX_STREAM_RELAY") or DEFAULT_RELAY
        _enabled = True

        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(
                target=_run, name="styxx-stream", daemon=True
            )
            _worker.start()

    return True


def disable() -> None:
    global _enabled
    with _lock:
        _enabled = False


def is_enabled() -> bool:
    return _enabled


# ══════════════════════════════════════════════════════════════════
# Public: emit
# ══════════════════════════════════════════════════════════════════

def emit_vitals(entry: dict) -> None:
    """Fire-and-forget push of one audit entry onto the stream queue.

    `entry` is the raw dict written to chart.jsonl. We strip prompt
    text before sending — we only transmit cognitive metrics, not
    content.
    """
    if not _enabled or not _agent:
        return

    packet = _sanitize_entry(entry)
    if packet is None:
        return

    try:
        _queue.put_nowait(packet)
    except queue.Full:
        # Drop oldest; keep newest.
        try:
            _queue.get_nowait()
            _queue.put_nowait(packet)
        except queue.Empty:
            pass


def _sanitize_entry(entry: dict) -> Optional[dict]:
    """Strip anything that could leak content; keep pure vitals."""
    if not isinstance(entry, dict):
        return None
    keep = (
        "ts", "ts_iso", "source", "context", "session_id", "model",
        "prompt_type",  # category only, not text
        "tier_active",
        "phase1_pred", "phase1_conf",
        "phase4_pred", "phase4_conf",
        "gate", "abort", "outcome",
        "coherence",
        "forecast_pred", "forecast_risk", "forecast_conf",
    )
    out = {k: entry.get(k) for k in keep if k in entry}
    # Attach agent + protocol version at the packet level.
    out["agent"] = _agent
    out["v"] = 1
    return out


# ══════════════════════════════════════════════════════════════════
# Worker loop
# ══════════════════════════════════════════════════════════════════

def _run() -> None:
    global _backoff
    while True:
        if not _enabled:
            time.sleep(0.5)
            continue
        try:
            packet = _queue.get(timeout=1.0)
        except queue.Empty:
            continue

        ok = _post(packet)
        if ok:
            _backoff = 0.0
        else:
            # Back off but don't block the queue drain — newer events
            # still displace older ones via the put_nowait fallback.
            _backoff = min(_BACKOFF_MAX, max(1.0, _backoff * 2 or 1.0))
            time.sleep(_backoff)


def _post(packet: dict) -> bool:
    headers = {"Content-Type": "application/json"}
    if _key:
        headers["X-Styxx-Key"] = _key
    try:
        body = json.dumps(packet).encode("utf-8")
        req = urllib.request.Request(
            _relay, data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=_POST_TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False
    except Exception:
        return False

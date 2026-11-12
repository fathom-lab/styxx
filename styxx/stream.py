# -*- coding: utf-8 -*-
"""
styxx.stream — live cognitive telemetry emitter.

When STYXX_STREAM=on (or stream=True passed to autoboot) and a relay
is named (STYXX_STREAM_RELAY, or relay=...), every vitals computation
is fire-and-forget POSTed to that relay, which can power a live
dashboard for the agent.

There is no default relay and no default dashboard. Until 2026-11-12
both were hosts under darkflobi.com, where the public relay lived. On
that day the names under that domain answered from a domain-parking
service, not from a lab server (the relay's own name failed its TLS
handshake that day, so a POST to it could not complete). With no
relay named, claiming fails with a message, the emitter stays off,
and nothing is sent.

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
    STYXX_STREAM_RELAY=<url>      the POST endpoint (required: no default)
    STYXX_DASHBOARD_BASE=<url>    base of the dashboard URLs (no default)
    STYXX_STREAM_AGENT=<name>     override agent name (else uses
                                  STYXX_AGENT_NAME or ~/.styxx/config)
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import http.client
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════
# Defaults
# ══════════════════════════════════════════════════════════════════

# No defaults: see the module docstring. Kept as names so code that read them still imports.
DEFAULT_RELAY: Optional[str] = None
DEFAULT_DASHBOARD_BASE: Optional[str] = None
# said by every caller that finds no relay; each adds the way to name one that it offers
NO_RELAY = "no relay is named, and styxx has no default relay (the public relay it used is offline)"

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
_relay: Optional[str] = DEFAULT_RELAY
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


def _restrict_acl_windows(p: Path) -> None:
    # chmod 600 is a no-op on Windows; best-effort NTFS equivalent:
    # drop inherited ACEs, grant only the current user. Fail-open —
    # a locked-down file is preferable but never worth crashing a claim.
    user = os.environ.get("USERNAME")
    if not user:
        return
    try:
        r = subprocess.run(
            ["icacls", str(p), "/inheritance:r", "/grant:r", f"{user}:F"],
            capture_output=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if r.returncode != 0:
            import warnings
            warnings.warn(
                f"styxx.stream: could not restrict ACL on {p} (icacls exit "
                f"{r.returncode}) — the credentials file keeps its inherited "
                f"permissions.", RuntimeWarning, stacklevel=2)
    except Exception as e:
        import warnings
        warnings.warn(
            f"styxx.stream: could not restrict ACL on {p} "
            f"({type(e).__name__}: {e}) — the credentials file keeps its "
            f"inherited permissions.", RuntimeWarning, stacklevel=2)


def _save_creds(data: dict) -> None:
    p = _creds_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2)
    if os.name == "nt":
        # No mode-at-create on Windows; write then restrict, and _restrict_acl_windows
        # now WARNS instead of failing open silently.
        p.write_text(payload, encoding="utf-8")
        _restrict_acl_windows(p)
        return
    # POSIX: create owner-only from the first byte — write-then-chmod left a
    # window where the token sat world-readable under the default umask.
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(payload)
    try:
        os.chmod(p, 0o600)     # pre-existing file: O_CREAT mode does not apply
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
    Raises ClaimError on network failure or name conflict, and when no
    relay is named (there is no default relay). `url` is the dashboard
    URL the relay gives, else the one built from STYXX_DASHBOARD_BASE,
    else an empty string.
    """
    clean = _sanitize_name(name)
    if not clean:
        raise ClaimError("invalid name")

    relay = relay_url(relay)
    if not relay:
        raise ClaimError(f"{NO_RELAY}. name one you run: relay=URL, or STYXX_STREAM_RELAY")
    url = claim_url(relay)

    body = json.dumps({"name": clean}).encode("utf-8")
    try:
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error", str(e))
        except Exception:
            msg = str(e)
        raise ClaimError(msg) from e
    except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
        # ValueError: a reply that is not JSON (json.JSONDecodeError is one), or a URL urllib
        # refuses. HTTPException: a URL or a reply http.client refuses (InvalidURL is one).
        raise ClaimError(f"relay unreachable: {e}") from e

    if not isinstance(data, dict) or "key" not in data or "name" not in data:
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


def is_http_url(url) -> bool:
    """True for a string that is an http or https URL with a host, a port that is a number if
    it has one, no user name or password in it, and no space or control character."""
    if not isinstance(url, str) or url != url.strip() or any(c.isspace() or ord(c) < 32 for c in url):
        return False
    try:
        parts = urllib.parse.urlsplit(url)
        parts.port          # raises ValueError for a port that is not a number
    except ValueError:
        return False
    if parts.username is not None or parts.password is not None:
        return False
    return parts.scheme in ("http", "https") and bool(parts.hostname)


def claim_url(relay: str) -> str:
    """The claim endpoint beside a relay's ingest endpoint: the relay URL with its last path
    segment replaced by `claim`, on the same host. `https://host/ingest`, `https://host/` and
    `https://host` all give `https://host/claim`. Raises ClaimError for anything is_http_url
    refuses. The relay URL itself is where the vitals are POSTed, so a relay given without a
    path receives them at its root.

    Until 2026-11-12 this was `relay.rsplit("/", 1)[0] + "/claim"`, which for a relay given as
    a bare origin (`https://host`) built `https://claim`: the name was sent to a host called
    `claim`, not to the relay. The default relay had a path, so the default never took that
    road; a relay the caller names can."""
    if not is_http_url(relay):
        raise ClaimError(f"not a relay URL: {relay!r} (it needs http:// or https:// and a host)")
    parts = urllib.parse.urlsplit(relay)
    head = parts.path.rsplit("/", 1)[0] if "/" in parts.path else ""
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, head + "/claim", "", ""))


def relay_url(relay: Optional[str] = None) -> Optional[str]:
    """The relay a claim or the emitter would use: the one given, else STYXX_STREAM_RELAY, else
    None. There is no default relay. Spaces around the value are dropped, as urllib drops them:
    a relay pasted with a trailing space is that relay."""
    value = relay or os.environ.get("STYXX_STREAM_RELAY") or DEFAULT_RELAY
    value = value.strip() if isinstance(value, str) else value
    return value or None


def dashboard_url(name: str) -> str:
    """Return the dashboard URL for an agent name under STYXX_DASHBOARD_BASE, or an empty string
    when no base is set: there is no default dashboard."""
    base = os.environ.get("STYXX_DASHBOARD_BASE") or DEFAULT_DASHBOARD_BASE
    if not base:
        return ""
    return f"{base.rstrip('/')}/a/{_sanitize_name(name)}"


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

    Returns False when there is no agent name to stream as, or no relay
    is named (relay=..., or STYXX_STREAM_RELAY) and the emitter is not
    already on, or the relay named is not an http or https URL: there
    is no default relay. A call that names no relay while the emitter
    is on keeps the relay it has and returns True. A False here changes
    nothing: an emitter that was on stays on (see is_enabled). The
    relay stored with a claimed name in the credentials file is not
    used here, as before.
    """
    global _enabled, _agent, _key, _relay, _worker

    with _lock:
        if agent is None or key is None:
            a, k = _pick_agent_and_key()
            agent = agent or a
            key = key or k

        if not agent:
            return False  # nothing to stream as

        target = relay_url(relay) or (_relay if _enabled else None)
        if not target or not is_http_url(target):
            return False  # nowhere to stream to; what is already on stays as it is

        _agent = _sanitize_name(agent)
        _key = key
        _relay = target
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
    if not _relay:
        return False        # no relay named: nothing is sent
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

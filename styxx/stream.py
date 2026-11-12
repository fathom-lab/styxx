# -*- coding: utf-8 -*-
"""
styxx.stream — live cognitive telemetry emitter.

When STYXX_STREAM=on (or stream=True passed to autoboot) and a relay
is named (STYXX_STREAM_RELAY, or relay=...), every vitals computation
is fire-and-forget POSTed to that relay, which can power a live
dashboard for the agent.

There is no default relay and no default dashboard. Through styxx
8.0.0 both were hosts under darkflobi.com, where the public relay
lived. On 2026-11-12 the names under that domain answered from a
domain-parking service, not from a lab server (the relay's own name
failed its TLS handshake that day, so a POST to it could not
complete). With no relay named, claiming fails with a message, the
emitter stays off, and nothing is sent.

A redirect is not followed. urllib answers a 301, 302 or 303 to a POST
by sending a GET to the address the reply names, with every header of
the POST but the body's two: the stream key went to that address, the
vitals did not, and the 2xx that came back counted as delivered. Found
on 2026-11-12 by review, the same in 8.0.0. Now a reply of 301, 302,
303, 307 or 308 from a relay is a failed post, and a failed claim,
that names the address.

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
_LOCATION_MAX = 200         # characters of a redirect's address shown in a message
_SHOWN_MAX = 300            # characters of any other text a relay chose, shown in a message
_KEY_MAX = 512              # characters of a key a relay may hand out
_LINK_MAX = 2048            # characters of a dashboard address a relay may hand out
_REDIRECT_CODES = (301, 302, 303, 307, 308)     # the replies urllib treats as a redirect
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
# Requests: no redirect is followed
# ══════════════════════════════════════════════════════════════════

_OPENER = None              # built when a request needs it: see open_url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Follows no redirect and reads no Location. The reply is left to urllib's default handler,
    which raises it as an HTTPError with its headers as they came. (The stock handler parses the
    Location before it asks whether to follow and raises ValueError for one it cannot parse, so
    the bytes of a reply chose which error a caller saw, and the stream said nothing at all.)"""

    def http_error_302(self, req, fp, code, msg, headers):
        return None

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None     # not reached through the methods above; nothing is followed if it is


def show(text, limit: int = _SHOWN_MAX) -> str:
    """Text a relay, an endpoint or an exception chose, made safe to print: control and format
    characters (the escape, the line ends and the direction overrides among them) and unpaired
    or private code points as escapes, cut to `limit` characters."""
    import unicodedata
    s = str(text)
    out = "".join(c if unicodedata.category(c) not in ("Cc", "Cf", "Cs", "Co", "Zl", "Zp")
                  else c.encode("unicode_escape").decode("ascii") for c in s[:limit])
    return out + ("..." if len(s) > limit else "")


def open_url(req, timeout: float):
    """`urllib.request.urlopen(req, timeout=timeout)` with one difference: a redirect is not
    followed. A 3xx reply is raised as urllib.error.HTTPError, whatever the method, and
    `redirect_target` reads the address a redirect names.

    This function is the one place to stand in for these requests. A test or a program that
    replaced `urllib.request.urlopen` to do so stands in for nothing now: through 8.0.0 the
    calls went through urlopen, and they no longer do.

    Every request styxx.stream, styxx.publish and styxx.compare send goes through here. The
    caller named one address: a reply from another one is not that address's reply, and a POST
    that urllib re-sends as a GET carries its headers (the stream key among them) and not its
    body. The opener is built once in a process, when a request needs it, so the proxy settings
    are read then, as urlopen reads them when it builds its own. An opener a program installed with
    `urllib.request.install_opener` is not used here; through 8.0.0 it was, since these calls
    went through urlopen."""
    global _OPENER
    opener = _OPENER
    if opener is None:
        opener = _OPENER = urllib.request.build_opener(_NoRedirect)
    return opener.open(req, timeout=timeout)


def redirect_target(exc) -> Optional[str]:
    """For the HTTPError of a redirect (301, 302, 303, 307, 308): the address it names, as a
    quoted string cut to _LOCATION_MAX characters and safe to print (`'(none given)'` when the
    reply names none). None for any other error, the other 3xx codes among them: a 304 or a 300
    is an HTTP error and names no address to go to."""
    code = getattr(exc, "code", None)
    if code not in _REDIRECT_CODES:
        return None
    try:
        where = exc.headers.get("Location") if exc.headers is not None else None
    except Exception:
        where = None
    if not where:
        return "(none given)"
    return "'" + show(where, _LOCATION_MAX).replace("'", "\\'") + "'"


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
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError, RecursionError):
        return {}
    # a file that is JSON and not an object held no credentials; through 8.0.0 it ended
    # enable(), a claim and `styxx feed` in a traceback
    return data if isinstance(data, dict) else {}


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
    # True when the relay answered and the answer is what failed the claim (an error reply, a
    # redirect, a reply that is not a name and a key): the relay is then not offline
    answered = False
    # True when the request was refused before anything was sent
    unsent = False


def _answered(message: str) -> "ClaimError":
    err = ClaimError(message)
    err.answered = True
    return err


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

    A redirect is not followed (see open_url): the claim fails and the
    message names the address the relay pointed to. styxx then read no
    reply and stored no key; whether the relay took the name before it
    redirected is the relay's to say, and the message says so. The
    reply must be a JSON object whose `name` is the name that was sent
    and whose `key` is a string that is not empty; anything else is a
    malformed response and nothing is stored. Through 8.0.0 a reply
    with a null or a list there was stored, or ended in a traceback,
    and a reply that named another agent overwrote that agent's key. A
    dashboard `url` in the reply is kept only if it is an http or https
    URL. Text the relay chose is printed with its control characters as
    escapes (see show).
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
        with open_url(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        where = redirect_target(e)
        if where is not None:
            raise _answered(
                f"the relay answered HTTP {e.code}, a redirect to {where}. styxx does not "
                f"follow a redirect, so it read no reply and stored no key. the name was sent "
                f"to {show(url)} and to no other address; a relay that takes a claim and then "
                f"redirects may hold the name now. name the relay by the URL that answers "
                f"itself") from e
        try:
            msg = json.loads(e.read().decode("utf-8")).get("error", str(e))
        except Exception:
            msg = str(e)
        raise _answered(show(msg)) from e
    except RecursionError as e:
        # JSON, and nested deeper than the reader goes: an earlier form called this "not JSON"
        raise _answered("the relay answered, and its reply is nested deeper than the JSON reader "
                        "goes") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # the relay answered 2xx and the body is not JSON (a web page, a byte-order mark).
        # Through 8.0.0 this read "relay unreachable".
        raise _answered(f"the relay answered, and its reply is not JSON "
                        f"({type(e).__name__}: {show(e, 120)})") from e
    except http.client.InvalidURL as e:
        # refused before anything was sent
        err = ClaimError(f"claim not sent: the http client refuses the relay URL "
                         f"({show(e, 200)})")
        err.unsent = True
        raise err from e
    except http.client.HTTPException as e:
        # A status line that is not one, a line of any length, a body shorter than its stated
        # length, a connection closed with nothing said: the request went out and what came
        # back is not a whole HTTP reply. An earlier form called this "relay unreachable" and
        # the command then said the relay may be offline.
        raise _answered(f"the request was sent, and what came back is not a whole HTTP reply "
                        f"({type(e).__name__}: {show(e, 120)}). the relay may hold the name "
                        f"now") from e
    except (urllib.error.URLError, OSError, ValueError) as e:
        # nothing answered, or urllib refuses the URL
        raise ClaimError(f"relay unreachable: {type(e).__name__}: {show(e, 200)}") from e

    name_back = data.get("name") if isinstance(data, dict) else None
    key_back = data.get("key") if isinstance(data, dict) else None
    # The name must be the one that was sent: it is the key the credentials are stored and
    # looked up under, and a reply that names another agent would overwrite that agent's key.
    # (An earlier commit asked for a name that _sanitize_name leaves unchanged; that function
    # cuts to 32 characters after it strips, so it can change its own result, and a relay's
    # echo of such a name was refused.)
    # No message below quotes the reply: it can hold a key, another agent's among them.
    if not isinstance(name_back, str) or not isinstance(key_back, str) or not key_back:
        raise _answered("malformed response: the relay's reply is not a name and a key. nothing "
                        "is stored")
    if name_back != clean:
        raise _answered("the relay answered with another name than the one that was sent. styxx "
                        "stores a key only under the name it sent, so nothing is stored; the relay "
                        "may hold a name now")
    # a key http.client cannot put in a header would be stored and then fail every post in silence
    if (len(key_back) > _KEY_MAX or not key_back.isascii() or not key_back.isprintable()
            or key_back != key_back.strip()):
        raise _answered("the relay answered with a key styxx cannot send: a key is at most "
                        f"{_KEY_MAX} printable ASCII characters with no space around it. nothing is "
                        "stored; the relay may hold the name now")
    link = data.get("url")

    # the dashboard address the relay gives is kept only if it is a web address: it is printed
    dashboard = ((link if isinstance(link, str) and len(link) <= _LINK_MAX and is_http_url(link)
                  else "") or dashboard_url(data["name"]))

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
    it has one, no user name or password in it, and no space, control character (U+0000 to
    U+001F and U+007F to U+009F) or format character (a direction override is one). Spaces
    around a URL are refused here too: a caller that takes them, as urllib does, drops them
    before it asks (relay_url, publish). urllib also unwraps `<URL>` and `URL:...`; those forms
    are refused here, though 8.0.0 sent to them.

    It does not promise that urllib can send to the URL: a path with a character outside ASCII,
    or a host name no resolver takes, passes here and fails when the request is made."""
    if not isinstance(url, str) or url != url.strip():
        return False
    import unicodedata
    if any(c.isspace() or unicodedata.category(c) in ("Cc", "Cf") for c in url):
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

    Through 8.0.0 this was `relay.rsplit("/", 1)[0] + "/claim"`, which for a relay given as
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
    """POST one packet to the relay. True for a 2xx reply from the relay itself. A redirect is
    not followed (see open_url): it is counted as a failed post like any other, so the worker
    backs off, and a warning names where the relay pointed. (Python shows a warning of one text
    once per place it is raised from, under its default filters; a program that changes the
    filters sees it more or less often.) The key is in no message."""
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
        with open_url(req, timeout=_POST_TIMEOUT) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        where = redirect_target(e)
        if where is not None:
            try:
                import warnings
                warnings.warn(
                    f"styxx.stream: the relay {show(_relay)} answered HTTP {e.code}, a "
                    f"redirect to {where}. styxx does not follow a redirect, so the packet is "
                    f"counted as not delivered and the feed backs off. name the relay by the "
                    f"URL that answers itself (in STYXX_STREAM_RELAY, or as relay= where "
                    f"the emitter is switched on).",
                    RuntimeWarning, stacklevel=2)
            except Exception:
                pass        # warnings turned into errors: the worker thread still must not die
        return False
    except (urllib.error.URLError, OSError):
        return False
    except Exception:
        return False

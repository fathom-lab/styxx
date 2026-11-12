# -*- coding: utf-8 -*-
"""
styxx.publish — opt-in telemetry for the public dashboard.

    styxx publish --name xendro --endpoint https://your.host/api/styxx-submit
    styxx publish --name xendro --dry-run

allows agents to publish their personality, fingerprint, weather,
and mood data to a remote endpoint so it can be displayed on a
dashboard and leaderboard.

this is strictly opt-in. nothing is sent unless the agent (or its
operator) explicitly runs the publish command and names the endpoint.
the data that gets sent is the same data `styxx personality`,
`styxx weather`, and `styxx fingerprint` already compute locally. no
new data is captured.

there is no default endpoint. until 2026-11-12 the default was a host
under darkflobi.com, where the public dashboard lived. on that day
every name under that domain answered from a domain-parking service,
not from a lab server, so `styxx publish --name X` handed its payload
to a server the lab does not run, and any 2xx reply was reported as
published. now nothing is sent unless the caller says where, and a
reply counts as published only if it is a JSON object.

uses only stdlib (urllib.request, json) — no external deps.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

# No default: see the module docstring. Kept as a name so code that read it still imports.
_DEFAULT_ENDPOINT: Optional[str] = None


def _is_http_url(url) -> bool:
    """True for a string that is an http or https URL with a host, a port that is a number if
    it has one, no user name or password in it, and no space or control character."""
    import urllib.parse
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


def prepare_payload(
    agent_name: str,
    *,
    days: float = 7.0,
) -> Dict[str, Any]:
    """collect all publishable data and return as a dict.

    same data collection as publish() but returns the dict without
    sending. useful for preview/inspection before publishing.
    """
    from . import __version__
    from .config import is_disabled

    payload: Dict[str, Any] = {
        "agent_name": agent_name,
        "timestamp": time.time(),
        "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "styxx_version": __version__,
    }

    if is_disabled():
        payload["disabled"] = True
        return payload

    # --- personality ---
    try:
        from .analytics import personality
        p = personality(days=days)
        payload["personality"] = p.as_dict() if p is not None else None
    except Exception:
        payload["personality"] = None

    # --- fingerprint ---
    try:
        from .analytics import fingerprint
        fp = fingerprint()
        if fp is not None:
            payload["fingerprint"] = {
                "n_samples": fp.n_samples,
                "phase1_vec": list(fp.phase1_vec),
                "phase4_vec": list(fp.phase4_vec),
                "phase1_mean_conf": round(fp.phase1_mean_conf, 4),
                "phase4_mean_conf": round(fp.phase4_mean_conf, 4),
                "gate_vec": list(fp.gate_vec),
            }
        else:
            payload["fingerprint"] = None
    except Exception:
        payload["fingerprint"] = None

    # --- weather ---
    try:
        from .weather import weather
        w = weather(agent_name=agent_name)
        payload["weather"] = w.as_dict() if w is not None else None
    except Exception:
        payload["weather"] = None

    # --- mood ---
    try:
        from .analytics import mood
        payload["mood"] = mood()
    except Exception:
        payload["mood"] = None

    # --- streak ---
    try:
        from .analytics import streak
        s = streak()
        if s is not None:
            payload["streak"] = {
                "category": s.category,
                "length": s.length,
            }
        else:
            payload["streak"] = None
    except Exception:
        payload["streak"] = None

    # --- log_stats ---
    try:
        from .analytics import log_stats
        ls = log_stats()
        payload["log_stats"] = {
            "n_entries": ls.n_entries,
            "gate_counts": dict(ls.gate_counts),
            "phase1_counts": dict(ls.phase1_counts),
            "phase4_counts": dict(ls.phase4_counts),
            "phase1_mean_conf": round(ls.phase1_mean_conf, 4),
            "phase4_mean_conf": round(ls.phase4_mean_conf, 4),
        }
    except Exception:
        payload["log_stats"] = None

    return payload


def publish(
    agent_name: str,
    endpoint: Optional[str] = _DEFAULT_ENDPOINT,
    *,
    days: float = 7.0,
) -> Optional[Dict[str, Any]]:
    """collect data and POST to the remote endpoint.

    returns a dict with 'status' (http code) and 'summary' on
    success. returns None if anything fails — fail open, never
    crash the caller.

    with no endpoint nothing is collected and nothing is sent: there
    is no default (see the module docstring). a 2xx reply whose body
    is not a JSON object is not success: a parked or mistyped host
    answers a POST with a web page, and by then the payload is sent.
    """
    if not endpoint:
        import sys
        print(
            "  warn: nothing was sent: styxx has no default publish endpoint "
            "(the public dashboard it posted to is offline). name one you run: "
            "styxx publish --name NAME --endpoint URL",
            file=sys.stderr,
        )
        return None
    if not _is_http_url(endpoint):
        import sys
        print(
            f"  warn: nothing was sent: {endpoint!r} is not an http or https URL with a host "
            "(no spaces, no user name or password in it).",
            file=sys.stderr,
        )
        return None

    try:
        payload = prepare_payload(agent_name, days=days)
    except Exception as exc:
        import sys
        print(f"  warn: failed to collect publish data: {exc}", file=sys.stderr)
        return None

    body = json.dumps(payload).encode("utf-8")

    try:
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"styxx-publish/{payload.get('styxx_version', '?')}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
            try:
                resp_body = json.loads(resp.read().decode("utf-8"))
            except Exception:
                resp_body = None
            if not isinstance(resp_body, dict):
                import sys
                print(
                    f"  warn: the payload was sent, and {endpoint} answered HTTP {status} "
                    "with a body that is not a JSON object, so this is not counted as "
                    "published. a parked or mistyped host answers that way: check the "
                    "endpoint.",
                    file=sys.stderr,
                )
                return None
            return {
                "status": status,
                "summary": f"published {agent_name} to {endpoint} (HTTP {status})",
                "response": resp_body,
            }
    except urllib.error.HTTPError as exc:
        import sys
        print(
            f"  warn: publish failed: HTTP {exc.code} from {endpoint}",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        import sys
        print(
            f"  warn: publish failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None

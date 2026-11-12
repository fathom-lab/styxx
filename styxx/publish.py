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

there is no default endpoint. through styxx 8.0.0 the default was a
host under darkflobi.com, where the public dashboard lived. on
2026-11-12 every name under that domain answered from a
domain-parking service, not from a lab server, so by the code
`styxx publish --name X` hands its payload to a server the lab does
not run and reports any 2xx reply as published (read from the code
and run against a local stand-in: no POST was sent to that host). now
nothing is sent unless the caller says where, and a reply counts as
published only if it is a JSON object.

a redirect is not followed (see styxx.stream.open_url): the endpoint
named is the only address the payload goes to, and a reply of 301,
302, 303, 307 or 308 is a failed publish that names the address the
endpoint pointed to.

uses only stdlib (urllib.request, json) — no external deps.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

# One URL check and one way of opening a URL for publish, the claim and the stream. The opener
# is called as an attribute of the module, so that replacing styxx.stream.open_url stands in for
# every request the three modules send.
from . import stream as _stream

_is_http_url = _stream.is_http_url

# No default: see the module docstring. Kept as a name so code that read it still imports.
_DEFAULT_ENDPOINT: Optional[str] = None


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

    spaces around the endpoint are dropped, as urllib drops them and as
    styxx.stream.relay_url does: an endpoint read from a file with its
    line end is that endpoint. a redirect is not followed: the payload
    went to the endpoint named, and the reply of another address is
    not that endpoint's reply.
    """
    if isinstance(endpoint, str):
        endpoint = endpoint.strip()
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
        with _stream.open_url(req, timeout=15) as resp:
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
        where = _stream.redirect_target(exc)
        if where is not None:
            print(
                f"  warn: publish failed: the payload was sent to {endpoint}, which answered "
                f"HTTP {exc.code}, a redirect to {where}. styxx does not follow a redirect, so "
                "it read no reply there and this is not counted as published. name the "
                "endpoint by the URL that answers itself.",
                file=sys.stderr,
            )
            return None
        print(
            f"  warn: publish failed: HTTP {exc.code} from {endpoint}",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        import sys
        print(
            f"  warn: publish failed: {type(exc).__name__}: {_stream.show(exc, 200)}",
            file=sys.stderr,
        )
        return None

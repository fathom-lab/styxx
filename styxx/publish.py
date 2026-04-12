# -*- coding: utf-8 -*-
"""
styxx.publish — opt-in telemetry for the public dashboard.

    styxx publish --name xendro
    styxx publish --name xendro --dry-run

allows agents to publish their personality, fingerprint, weather,
and mood data to a remote endpoint so it can be displayed on a
public dashboard and leaderboard.

this is strictly opt-in. nothing is sent unless the agent (or its
operator) explicitly runs the publish command. the data that gets
sent is the same data `styxx personality`, `styxx weather`, and
`styxx fingerprint` already compute locally — no new data is
captured.

uses only stdlib (urllib.request, json) — no external deps.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

_DEFAULT_ENDPOINT = "https://fathom.darkflobi.com/api/styxx-submit"


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
    endpoint: str = _DEFAULT_ENDPOINT,
    *,
    days: float = 7.0,
) -> Optional[Dict[str, Any]]:
    """collect data and POST to the remote endpoint.

    returns a dict with 'status' (http code) and 'summary' on
    success. returns None if anything fails — fail open, never
    crash the caller.
    """
    try:
        payload = prepare_payload(agent_name, days=days)
    except Exception as exc:
        import sys
        print(f"  warn: failed to collect publish data: {exc}", file=sys.stderr)
        return None

    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"styxx-publish/{payload.get('styxx_version', '?')}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
            try:
                resp_body = json.loads(resp.read().decode("utf-8"))
            except Exception:
                resp_body = {}
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

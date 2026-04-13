# -*- coding: utf-8 -*-
"""
styxx.notify — webhook notifications on cognitive events.

    styxx.on_anomaly("https://hooks.slack.com/...", events=["fail", "warn_cluster"])
    styxx.on_anomaly(my_callback_fn)

    # or the simple version:
    styxx.notify_on_fail("https://hooks.slack.com/...")

When gate=fail fires, or a warn cluster forms, or confidence
collapses — the operator hears about it immediately instead of
finding out from the compliance report the next morning.

1.4.0+.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class CognitiveEvent:
    """One cognitive event worth notifying about."""
    event_type: str     # "gate_fail", "gate_warn", "warn_cluster", "confidence_collapse", "anomaly"
    description: str
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    gate: Optional[str] = None
    category: Optional[str] = None
    confidence: Optional[float] = None
    ts: float = 0.0
    ts_iso: str = ""

    def as_dict(self) -> dict:
        return {
            "event": self.event_type,
            "description": self.description,
            "agent": self.agent_name,
            "session": self.session_id,
            "gate": self.gate,
            "category": self.category,
            "confidence": self.confidence,
            "timestamp": self.ts_iso,
        }

    def as_json(self) -> str:
        return json.dumps(self.as_dict())


# Registry of notification handlers
_HANDLERS: List[dict] = []
_LOCK = threading.Lock()
_RECENT_WARNS: List[float] = []  # timestamps of recent warn events


NotifyTarget = Union[str, Callable[[CognitiveEvent], None]]


def on_anomaly(
    target: NotifyTarget,
    *,
    events: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> None:
    """Register a notification handler for cognitive events.

    Args:
        target:  webhook URL (string) or callable(CognitiveEvent)
        events:  which events to listen for. Default: all.
                 Options: "gate_fail", "gate_warn", "warn_cluster",
                 "confidence_collapse", "anomaly"
        name:    optional label for this handler

    Usage:
        # Webhook
        styxx.on_anomaly("https://hooks.slack.com/services/...")

        # Custom function
        styxx.on_anomaly(lambda e: send_telegram(e.description))

        # Specific events only
        styxx.on_anomaly(my_fn, events=["gate_fail", "warn_cluster"])
    """
    handler = {
        "target": target,
        "events": set(events) if events else None,  # None = all events
        "name": name or f"handler-{len(_HANDLERS)}",
    }
    with _LOCK:
        _HANDLERS.append(handler)


def notify_on_fail(target: NotifyTarget) -> None:
    """Shortcut: notify on gate=fail only."""
    on_anomaly(target, events=["gate_fail"], name="fail-notifier")


def clear_notifications() -> int:
    """Remove all notification handlers."""
    with _LOCK:
        n = len(_HANDLERS)
        _HANDLERS.clear()
        return n


def _dispatch_event(event: CognitiveEvent) -> int:
    """Send event to all matching handlers. Returns count sent."""
    with _LOCK:
        handlers = list(_HANDLERS)

    sent = 0
    for h in handlers:
        # Filter by event type
        if h["events"] is not None and event.event_type not in h["events"]:
            continue

        target = h["target"]
        try:
            if isinstance(target, str):
                _send_webhook(target, event)
            elif callable(target):
                target(event)
            sent += 1
        except Exception as e:
            warnings.warn(
                f"styxx notify handler '{h['name']}' failed: {e}",
                RuntimeWarning, stacklevel=2,
            )
    return sent


def _send_webhook(url: str, event: CognitiveEvent) -> None:
    """POST event to a webhook URL."""
    payload = json.dumps({
        "text": f"[styxx] {event.event_type}: {event.description}",
        "styxx_event": event.as_dict(),
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Fire and forget in a thread to not block the agent
    def _send():
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()


def check_and_notify(entry: dict) -> int:
    """Check an audit entry for notify-worthy events.

    Called from write_audit() after writing. Returns count of
    notifications dispatched.
    """
    if not _HANDLERS:
        return 0

    from . import config

    gate = entry.get("gate")
    cat = entry.get("phase4_pred")
    conf = entry.get("phase4_conf")
    sent = 0

    base_kwargs = dict(
        agent_name=config.agent_name(),
        session_id=entry.get("session_id"),
        gate=gate,
        category=cat,
        confidence=float(conf) if conf is not None else None,
        ts=entry.get("ts", time.time()),
        ts_iso=entry.get("ts_iso", ""),
    )

    # Gate fail
    if gate == "fail":
        event = CognitiveEvent(
            event_type="gate_fail",
            description=f"gate=fail: {cat} at conf {conf}",
            **base_kwargs,
        )
        sent += _dispatch_event(event)

    # Gate warn
    if gate == "warn":
        event = CognitiveEvent(
            event_type="gate_warn",
            description=f"gate=warn: {cat} at conf {conf}",
            **base_kwargs,
        )
        sent += _dispatch_event(event)

        # Track for cluster detection
        now = time.time()
        _RECENT_WARNS.append(now)
        # Clean old warns (>60s)
        while _RECENT_WARNS and _RECENT_WARNS[0] < now - 60:
            _RECENT_WARNS.pop(0)
        # Warn cluster: 3+ warns in 60 seconds
        if len(_RECENT_WARNS) >= 3:
            cluster_event = CognitiveEvent(
                event_type="warn_cluster",
                description=f"{len(_RECENT_WARNS)} warn events in last 60s — cognitive degradation in progress",
                agent_name=config.agent_name(),
                session_id=entry.get("session_id"),
                ts=now,
                ts_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            sent += _dispatch_event(cluster_event)
            _RECENT_WARNS.clear()  # reset after alerting

    # Confidence collapse
    if conf is not None:
        try:
            if float(conf) < 0.15:
                collapse_event = CognitiveEvent(
                    event_type="confidence_collapse",
                    description=f"confidence collapsed to {conf} on {cat}",
                    agent_name=config.agent_name(),
                    session_id=entry.get("session_id"),
                    gate=gate,
                    category=cat,
                    confidence=float(conf),
                    ts=entry.get("ts", time.time()),
                    ts_iso=entry.get("ts_iso", ""),
                )
                sent += _dispatch_event(collapse_event)
        except (ValueError, TypeError):
            pass

    return sent

# -*- coding: utf-8 -*-
"""
styxx.config — environment-variable runtime config.

Centralizes every env var styxx responds to so there's one place to
look, one place to document, and one place to add new ones without
scattering os.environ.get calls across the package.

Environment variables (all default to OFF / normal operation):

  STYXX_DISABLED    "1"  →  styxx becomes a pass-through. from styxx
                            import OpenAI still works but returns an
                            unmodified openai client. no vitals read,
                            no audit write, no overhead. kill switch
                            for production bake-offs or emergency
                            rollback.

  STYXX_NO_AUDIT    "1"  →  disable the audit-log write. vitals still
                            computed, cards still render, but nothing
                            is appended to ~/.styxx/chart.jsonl. use
                            for privacy-regulated deployments.

  STYXX_NO_COLOR    "1"  →  disable ANSI colors in the terminal.
                            already respected by cards.color_enabled.

  STYXX_BOOT_SPEED  float →  override the boot-log timing multiplier.
                            0 = instant, 1.0 = normal, 2.0 = slower.

  STYXX_SKIP_SHA    "1"  →  skip the centroid sha256 verification at
                            load time. NEVER set this in production —
                            it bypasses the tamper-detection guarantee.
                            intended only for development/debugging.

  STYXX_SESSION_ID  str  →  tag every audit log entry with this session
                            id. lets you slice the audit log per
                            conversation via `styxx log session <id>`.
                            0.1.0a3+ only.

Helpers below read each env var defensively so no import raises even
if a user pokes at weird values.
"""

from __future__ import annotations

import os
from typing import Optional


def _truthy(name: str) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on", "y")


def is_disabled() -> bool:
    """Full styxx kill switch. When on, all adapters and runtimes
    become pass-throughs — the underlying SDK call returns normally
    and no vital reading happens."""
    return _truthy("STYXX_DISABLED")


def is_audit_disabled() -> bool:
    """Disable the audit log write. Vitals are still computed but
    never persisted to disk."""
    return _truthy("STYXX_NO_AUDIT")


def is_color_disabled() -> bool:
    """Disable ANSI color output. Used by cards.color_enabled."""
    return _truthy("STYXX_NO_COLOR")


def skip_sha_verification() -> bool:
    """Skip centroid sha256 verification at load time. Dangerous;
    only for local development."""
    return _truthy("STYXX_SKIP_SHA")


def boot_speed() -> float:
    """Boot-log timing multiplier. 0 = instant, 1.0 = normal, 2.0 = slower."""
    raw = os.environ.get("STYXX_BOOT_SPEED", "").strip()
    if not raw:
        return 1.0
    try:
        val = float(raw)
        return max(0.0, val)
    except ValueError:
        return 1.0


# ── session tagging (0.1.0a3) ──────────────────────────────────
#
# The session id is used to tag audit log entries so the log analyzer
# can group calls by conversation. Reads from STYXX_SESSION_ID by
# default; can be overridden programmatically via set_session().

_SESSION_OVERRIDE: Optional[str] = None


def session_id() -> Optional[str]:
    """Return the current session id, or None if unset.

    Priority:
      1. programmatic override via styxx.set_session(...)
      2. STYXX_SESSION_ID environment variable
      3. None (no tagging)
    """
    if _SESSION_OVERRIDE is not None:
        return _SESSION_OVERRIDE
    val = os.environ.get("STYXX_SESSION_ID", "").strip()
    return val or None


def set_session(session: Optional[str]) -> None:
    """Set the current session id programmatically.

    Pass None to clear the override and fall back to the environment
    variable (or no tagging if it's also unset). Session id should be
    a short stable string like 'xendro-2026-04-11' or a conversation
    uuid — it's embedded verbatim in every audit log entry.
    """
    global _SESSION_OVERRIDE
    _SESSION_OVERRIDE = session if session is None else str(session).strip() or None

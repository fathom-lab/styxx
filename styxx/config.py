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

  STYXX_TIER1_ENABLED "1" → activate tier 1 (D-axis honesty). loads
                            a HookedTransformer model on first use.
                            requires torch + transformers +
                            transformer-lens. uses 4-8 GB VRAM.
                            0.3.0+ only.

  STYXX_TIER1_MODEL   str → which model to load for tier 1.
                            default: "google/gemma-2-2b-it"
                            any HuggingFace model supported by
                            transformer-lens works.

  STYXX_TIER1_DEVICE  str → device for tier 1 model.
                            default: "cuda" if available, else "cpu"
                            set to "cpu" to force CPU inference
                            (20-30x slower but works without GPU).

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


# ── tier 1 config (0.3.0) ──────────────────────────────────

def tier1_enabled() -> bool:
    """Whether tier 1 (D-axis honesty) should be activated.

    Requires STYXX_TIER1_ENABLED=1 AND torch + transformers +
    transformer-lens to be importable. Model loading is 30s+ and
    uses 4-8 GB VRAM, so this is always opt-in.
    """
    return _truthy("STYXX_TIER1_ENABLED")


def tier1_model() -> str:
    """Which model to load for tier 1 D-axis computation.

    Default: google/gemma-2-2b-it (the primary validated model
    from the Fathom atlas v0.3 D-axis pilot).
    """
    val = os.environ.get("STYXX_TIER1_MODEL", "").strip()
    return val or "google/gemma-2-2b-it"


def tier1_device() -> str:
    """Device for the tier 1 model. Default: cuda if available,
    else cpu. Override with STYXX_TIER1_DEVICE."""
    val = os.environ.get("STYXX_TIER1_DEVICE", "").strip()
    if val:
        return val
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── mood-adaptive gating (0.5.10) ──────────────────────────
#
# Xendro's #4: "mood as input, not just output. a cautious
# agent should have tighter hallucination gates than a creative
# one." The agent declares its mood, styxx adjusts thresholds.

_MOOD_OVERRIDE: Optional[str] = None

_MOOD_GATE_MULTIPLIERS = {
    "cautious":  0.7,   # tighter gates (lower thresholds)
    "defensive": 0.6,
    "steady":    1.0,   # default
    "creative":  1.3,   # looser gates (higher thresholds)
    "focused":   0.9,
    "drifting":  0.5,   # tightest — something's wrong
}


def set_mood(mood_: Optional[str]) -> None:
    """Declare the agent's current mood for adaptive gating.

    When set, styxx adjusts gate thresholds accordingly:
      cautious  → tighter (0.7x thresholds — catch more)
      creative  → looser (1.3x — let more through)
      drifting  → tightest (0.5x — maximum sensitivity)
      steady    → default (1.0x)

    Usage:
        styxx.set_mood("cautious")
        # now all gate thresholds are 0.7x their default
        # hallucination gate fires at 0.385 instead of 0.55

    Pass None to clear and return to default thresholds.
    """
    global _MOOD_OVERRIDE
    _MOOD_OVERRIDE = mood_ if mood_ else None


def current_mood_override() -> Optional[str]:
    """Return the declared mood, or None if not set."""
    return _MOOD_OVERRIDE


def gate_multiplier() -> float:
    """Return the gate threshold multiplier for the current mood."""
    if _MOOD_OVERRIDE is None:
        return 1.0
    return _MOOD_GATE_MULTIPLIERS.get(_MOOD_OVERRIDE, 1.0)


# ── session context (0.8.0) ──────────────────────────────
#
# The agent declares what it's doing so weather and antipatterns
# can distinguish expected behavior from problematic behavior.

_CONTEXT_OVERRIDE: Optional[str] = None


def set_context(context: Optional[str] = None) -> None:
    """Declare what the agent is currently doing.

    When set, weather adjusts prescriptions to account for the task
    domain. A security-focused agent won't get told to "stop being
    cautious" when cautious is the right mode for its current work.

    Usage:
        styxx.set_context("security_review")
        styxx.set_context("creative_writing")
        styxx.set_context(None)  # clear
    """
    global _CONTEXT_OVERRIDE
    _CONTEXT_OVERRIDE = context if context else None


def current_context() -> Optional[str]:
    """Return the declared context, or None if not set."""
    if _CONTEXT_OVERRIDE is not None:
        return _CONTEXT_OVERRIDE
    val = os.environ.get("STYXX_CONTEXT", "").strip()
    return val or None


# ── domain calibration (0.8.0) ───────────────────────────
#
# Agents with specific domains (security, code review) will always
# look "adversarial" or "refusal-heavy" to a general classifier.
# expect() marks categories as normal for this agent so antipatterns
# and weather don't count them as failures.

_EXPECTED_CATEGORIES: set = set()


def expect(category: str) -> None:
    """Declare that a classification category is expected for this agent.

    A security agent always looks 'adversarial' to a general classifier.
    Calling styxx.expect('adversarial') tells antipatterns and weather
    to not count adversarial readings as failures.

    Usage:
        styxx.expect("adversarial")
        styxx.expect("refusal")
    """
    _EXPECTED_CATEGORIES.add(category)


def unexpect(category: str) -> None:
    """Remove a category from the expected set."""
    _EXPECTED_CATEGORIES.discard(category)


def expected_categories() -> frozenset:
    """Return the current set of expected categories."""
    return frozenset(_EXPECTED_CATEGORIES)


def clear_expected() -> None:
    """Clear all expected categories."""
    _EXPECTED_CATEGORIES.clear()


def set_session(session: Optional[str]) -> None:
    """Set the current session id programmatically.

    Pass None to clear the override and fall back to the environment
    variable (or no tagging if it's also unset). Session id should be
    a short stable string like 'xendro-2026-04-11' or a conversation
    uuid — it's embedded verbatim in every audit log entry.
    """
    global _SESSION_OVERRIDE
    _SESSION_OVERRIDE = session if session is None else str(session).strip() or None

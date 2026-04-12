# -*- coding: utf-8 -*-
"""
styxx.hooks - global monkey-patch for zero-code-change adoption.

The pitch: you have an existing agent codebase with 300 places that
call `openai.OpenAI()` or construct the client directly. You want
styxx telemetry on every call without touching any of those sites.

Solution:

    import styxx
    styxx.hook_openai()     # once at startup

That single call replaces the `OpenAI` class in the `openai` module
with styxx's wrapper. From that point on, EVERY openai.OpenAI() call
in the process returns a styxx-wrapped client. Every chat completion
automatically gains a `.vitals` attribute. No other code changes.

Reversible via styxx.unhook_openai(). Idempotent (calling hook_openai
twice is a no-op). Fail-open: if the openai module isn't installed,
the hook raises ImportError with a clear install hint instead of
silently failing.

Why this is load-bearing for real agents
────────────────────────────────────────

Xendro's first complaint about 0.1.0a0 was "I can't wire this into
my own loop." The watch()/observe() pattern from 0.1.0a1 solved the
one-shot case. The hook_openai() pattern from 0.1.0a3 solves the
existing-codebase case — you can add telemetry to a 30k-line agent
in one line.

This is what gets styxx from "interesting alpha" to "shipped in
production by Fathom Lab customers".
"""

from __future__ import annotations

from typing import Any, Optional


# State: store the original openai.OpenAI so we can restore it.
_ORIGINAL_OPENAI: Any = None
_HOOK_ACTIVE: bool = False


def hook_openai() -> bool:
    """Monkey-patch `openai.OpenAI` globally so every call site in
    the process returns a styxx-wrapped client.

    Returns True if the hook was newly installed, False if it was
    already installed (idempotent).

    Raises:
        ImportError: if the `openai` SDK isn't installed.

    Usage:

        import styxx
        styxx.hook_openai()

        # everywhere else in your code — no changes needed
        from openai import OpenAI
        client = OpenAI()                         # styxx-wrapped automatically
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            logprobs=True, top_logprobs=5,
        )
        print(r.vitals.summary)                   # it's there
    """
    global _ORIGINAL_OPENAI, _HOOK_ACTIVE

    if _HOOK_ACTIVE:
        return False

    try:
        import openai as _openai_mod
    except ImportError as e:
        raise ImportError(
            "styxx.hook_openai() requires the openai SDK.\n"
            "  Install with:  pip install openai\n"
            "  Or install styxx with the extra:  pip install styxx[openai]\n"
            f"  Underlying error: {e}"
        ) from e

    # Stash the original class so we can restore it
    _ORIGINAL_OPENAI = _openai_mod.OpenAI

    # Build the wrapper factory
    from .adapters.openai import OpenAIWithVitals

    def _OpenAI_hooked(*args, **kwargs):
        """Replacement for openai.OpenAI that returns styxx-wrapped
        client. Constructor signature passes through unchanged."""
        return OpenAIWithVitals(*args, **kwargs)

    # Mark the hooked function so introspection can tell
    _OpenAI_hooked.__name__ = "OpenAI"
    _OpenAI_hooked.__qualname__ = "OpenAI"
    _OpenAI_hooked._styxx_hooked = True  # type: ignore[attr-defined]

    # Replace the class in the openai module
    _openai_mod.OpenAI = _OpenAI_hooked  # type: ignore[assignment]
    _HOOK_ACTIVE = True
    return True


def unhook_openai() -> bool:
    """Remove the global openai hook and restore the original class.

    Returns True if an active hook was removed, False if no hook was
    installed.
    """
    global _ORIGINAL_OPENAI, _HOOK_ACTIVE

    if not _HOOK_ACTIVE:
        return False

    try:
        import openai as _openai_mod
    except ImportError:
        # openai is gone — nothing to restore
        _HOOK_ACTIVE = False
        _ORIGINAL_OPENAI = None
        return True

    if _ORIGINAL_OPENAI is not None:
        _openai_mod.OpenAI = _ORIGINAL_OPENAI  # type: ignore[assignment]
    _ORIGINAL_OPENAI = None
    _HOOK_ACTIVE = False
    return True


def hook_openai_active() -> bool:
    """Return True if the global openai hook is currently installed."""
    return _HOOK_ACTIVE

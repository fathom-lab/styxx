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

from typing import Any


# State: store the original openai.OpenAI so we can restore it. We
# also remember the hooked replacement so unhook can identify it by
# identity (avoids a `getattr(attr, "_styxx_hooked")` probe that
# triggers lazy-import machinery in third-party modules like torch).
_ORIGINAL_OPENAI: Any = None
_HOOKED_OPENAI: Any = None
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
    global _ORIGINAL_OPENAI, _HOOKED_OPENAI, _HOOK_ACTIVE

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
    _HOOKED_OPENAI = _OpenAI_hooked

    # Replace the class in the openai module
    _openai_mod.OpenAI = _OpenAI_hooked  # type: ignore[assignment]

    # Walk sys.modules and rebind any module-level reference to the
    # original OpenAI class. This catches the common `from openai
    # import OpenAI` pattern where a caller has already bound the
    # unhooked class in their own namespace before hook_openai() ran.
    # Without this rewrite, that caller's `OpenAI()` constructions
    # silently bypass the hook and `@styxx.profile` reports steps=0.
    import sys as _sys
    for mod_name, mod in list(_sys.modules.items()):
        # Skip the openai package itself (already patched), this hooks
        # module (don't rewrite our own references), and anything that
        # isn't a real module object.
        if mod is None or mod_name == "openai" or mod_name.startswith("openai."):
            continue
        if mod_name == __name__ or mod_name.startswith("styxx."):
            continue
        try:
            mod_dict = getattr(mod, "__dict__", None)
        except Exception:
            continue
        if mod_dict is None:
            continue
        for attr_name, attr_val in list(mod_dict.items()):
            if attr_val is _ORIGINAL_OPENAI:
                try:
                    mod_dict[attr_name] = _OpenAI_hooked
                except Exception:
                    # Read-only namespace (some builtin modules) — skip.
                    pass

    _HOOK_ACTIVE = True
    return True


def unhook_openai() -> bool:
    """Remove the global openai hook and restore the original class.

    Returns True if an active hook was removed, False if no hook was
    installed.
    """
    global _ORIGINAL_OPENAI, _HOOKED_OPENAI, _HOOK_ACTIVE

    if not _HOOK_ACTIVE:
        return False

    try:
        import openai as _openai_mod
    except ImportError:
        # openai is gone — nothing to restore
        _HOOK_ACTIVE = False
        _ORIGINAL_OPENAI = None
        _HOOKED_OPENAI = None
        return True

    if _ORIGINAL_OPENAI is not None:
        _openai_mod.OpenAI = _ORIGINAL_OPENAI  # type: ignore[assignment]

        # Mirror of the rewrite in hook_openai(): walk sys.modules and
        # restore any reference to the hooked class back to the original.
        # Identity comparison only — never `getattr(attr, "_styxx_hooked")`
        # because that triggers lazy-import machinery on third-party
        # modules (notably torch._classes) and raises mid-iteration.
        import sys as _sys
        hooked = _HOOKED_OPENAI
        if hooked is not None:
            for mod_name, mod in list(_sys.modules.items()):
                if mod is None or mod_name == "openai" or mod_name.startswith("openai."):
                    continue
                if mod_name == __name__ or mod_name.startswith("styxx."):
                    continue
                try:
                    mod_dict = getattr(mod, "__dict__", None)
                except Exception:
                    continue
                if mod_dict is None:
                    continue
                for attr_name, attr_val in list(mod_dict.items()):
                    if attr_val is hooked:
                        try:
                            mod_dict[attr_name] = _ORIGINAL_OPENAI
                        except Exception:
                            pass

    _ORIGINAL_OPENAI = None
    _HOOKED_OPENAI = None
    _HOOK_ACTIVE = False
    return True


def hook_openai_active() -> bool:
    """Return True if the global openai hook is currently installed."""
    return _HOOK_ACTIVE

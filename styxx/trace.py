# -*- coding: utf-8 -*-
"""
styxx.trace - function-level tracing decorator.

Tags every styxx audit log entry written inside a decorated function
with that function's name as the session id. This lets you split a
single process's audit log by logical function call rather than only
by conversation session.

Motivation: Xendro asked in the 0.1.0a1 report for a way to slice
the audit log by "what part of my agent was running at the time."
The session tagging shipped in 0.1.0a3 handles conversation-level
slicing (STYXX_SESSION_ID + set_session); this module handles
function-level slicing on top of that.

Usage
-----

    import styxx

    @styxx.trace("handro.plan")
    def plan_response(user_msg):
        r = client.chat.completions.create(...)
        return r.choices[0].message.content

    @styxx.trace("handro.reflect")
    def reflect_on_plan(draft):
        r = client.chat.completions.create(...)
        return r.choices[0].message.content

Now every audit entry written during `plan_response()` has
session_id="handro.plan", and every entry from `reflect_on_plan()`
has session_id="handro.reflect". Query either with:

    styxx log session handro.plan
    styxx log session handro.reflect

Works with both sync and async functions. Nests cleanly - the
decorator stashes the outer session, runs the inner function under
the new session, and restores the outer session on exit even if the
inner function raises.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

from .config import session_id, set_session


F = TypeVar("F", bound=Callable[..., Any])


def trace(name: str) -> Callable[[F], F]:
    """Decorator that tags every audit log entry inside the wrapped
    function with the given name as the session id.

    Nests cleanly: if the wrapped function is called from inside
    another traced function, the outer session is restored on exit.

    Usage:

        @styxx.trace("handro.plan")
        def plan_response(user_msg):
            ...

    Works on both sync and async functions.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("styxx.trace(name) requires a non-empty string name")

    def _decorate(fn: F) -> F:
        # Detect coroutine function and wrap appropriately
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def _async_wrapper(*args, **kwargs):
                prev = session_id()
                set_session(name)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    set_session(prev)
            return _async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def _sync_wrapper(*args, **kwargs):
            prev = session_id()
            set_session(name)
            try:
                return fn(*args, **kwargs)
            finally:
                set_session(prev)
        return _sync_wrapper  # type: ignore[return-value]

    return _decorate

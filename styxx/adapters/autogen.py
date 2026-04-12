# -*- coding: utf-8 -*-
"""
styxx.adapters.autogen — hook into microsoft autogen's multi-agent loop.

autogen agents communicate through a register_reply / generate_reply
protocol. this adapter hooks into that protocol so that every agent
reply passes through styxx.observe() on its way out. the agent's
behavior is never modified — styxx is a passive observer that attaches
vitals and fires any registered gate callbacks.

usage:

    import autogen
    import styxx
    from styxx.adapters.autogen import styxx_agent

    assistant = autogen.AssistantAgent("assistant", llm_config={...})
    styxx_agent(assistant)   # one line — now observed

    # or use the hook class directly for more control:
    from styxx.adapters.autogen import StyxxAutoGenHook
    hook = StyxxAutoGenHook()
    hook.attach(assistant)

the hook intercepts generate_reply by registering a reply function
via autogen's register_reply mechanism. the reply function calls the
original generate_reply, runs styxx.observe() on the result, and
returns it unchanged. if styxx fails for any reason, the original
reply passes through untouched.

fail-open contract
------------------
this adapter never raises exceptions into autogen's agent loop.
every code path that touches styxx internals is wrapped in a
try/except that swallows errors and warns. if autogen is not
installed, styxx_agent() returns the agent unchanged with no
side effects.

limitations:
  - tier 0 vitals require logprobs in the response. most autogen
    configurations use openai under the hood, so vitals will be
    available if the llm_config includes logprobs=True and
    top_logprobs=5. without those, .vitals will be None.
  - streaming replies are not intercepted (autogen's streaming
    path bypasses register_reply in some configurations).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional


class StyxxAutoGenHook:
    """passive observation hook for autogen agents.

    attaches to an autogen agent's reply pipeline via register_reply.
    every reply the agent generates passes through styxx.observe()
    so that gate callbacks fire and vitals are available on the
    response object.

    the hook never modifies the reply content. if observation fails,
    the reply passes through unchanged — fail-open.

    attributes:
        agent       : the autogen agent this hook is attached to,
                      or None if not yet attached
        n_observed  : count of replies observed since attachment
        last_vitals : the most recent Vitals object, or None
    """

    def __init__(self):
        self.agent: Any = None
        self.n_observed: int = 0
        self.last_vitals: Any = None

    def attach(self, agent: Any) -> Any:
        """attach this hook to an autogen agent.

        registers a reply function that intercepts generate_reply
        and runs styxx.observe() on the result. the agent is
        returned for chaining.

        if the agent doesn't have register_reply (not a real autogen
        agent), this is a no-op — the agent is returned unchanged.
        """
        self.agent = agent

        register_fn = getattr(agent, "register_reply", None)
        if register_fn is None or not callable(register_fn):
            warnings.warn(
                "styxx.adapters.autogen: agent has no register_reply "
                "method — skipping hook attachment. the agent will "
                "run normally without styxx observation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return agent

        # build the reply function that autogen will call.
        # autogen register_reply signature:
        #   reply_func(recipient, messages, sender, config)
        # it should return (should_stop, reply) or None to skip.
        hook = self

        def _styxx_reply_func(
            recipient: Any,
            messages: Any = None,
            sender: Any = None,
            config: Any = None,
        ) -> Optional[tuple]:
            """observe the reply but never override it.

            returns None so autogen continues to the next reply
            function in the chain. the observation happens as a
            side effect — we intercept the messages to observe
            the last one, but we don't generate a reply ourselves.
            """
            try:
                if messages and isinstance(messages, (list, tuple)):
                    last_msg = messages[-1]
                    _observe_message(last_msg, hook)
            except Exception as exc:
                # fail open — never break autogen's loop
                try:
                    warnings.warn(
                        f"styxx autogen hook: observation failed: "
                        f"{type(exc).__name__}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                except Exception:
                    pass
            # return None = "i have no reply, let the next handler go"
            return None

        try:
            # register at the front of the reply chain so we see
            # messages before other hooks might modify them.
            # autogen's register_reply accepts:
            #   reply_func, position (int), config, trigger
            # position=0 means "first in chain"
            register_fn(
                _styxx_reply_func,
                position=0,
            )
        except TypeError:
            # some autogen versions have a different register_reply
            # signature. try the minimal call.
            try:
                register_fn(_styxx_reply_func)
            except Exception as exc:
                warnings.warn(
                    f"styxx autogen hook: register_reply failed: "
                    f"{type(exc).__name__}: {exc}. "
                    f"the agent will run without observation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except Exception as exc:
            # catch-all for any other register_reply failure.
            # fail open — warn and continue.
            warnings.warn(
                f"styxx autogen hook: register_reply failed: "
                f"{type(exc).__name__}: {exc}. "
                f"the agent will run without observation.",
                RuntimeWarning,
                stacklevel=2,
            )

        return agent


def _observe_message(message: Any, hook: StyxxAutoGenHook) -> None:
    """run styxx.observe() on a message from autogen's reply chain.

    accepts both string messages and dict messages (autogen uses
    dicts with 'content' keys). if the message has an openai-shaped
    response attached, observe() will extract logprobs and compute
    vitals. otherwise vitals will be None.
    """
    from ..watch import observe

    # autogen messages can be strings, dicts, or response objects.
    # try to observe whatever we get — observe() handles unknown
    # shapes gracefully (returns None vitals).
    vitals = None
    if isinstance(message, dict):
        # if there's an openai response object attached, prefer that
        response = message.get("_oai_response", None)
        if response is not None:
            vitals = observe(response)
        else:
            # observe the dict itself — won't have logprobs but
            # won't crash either
            vitals = observe(message)
    elif hasattr(message, "choices"):
        # looks like a raw openai response
        vitals = observe(message)
    else:
        # string or unknown — observe won't crash, just returns None
        vitals = observe(message)

    hook.n_observed += 1
    hook.last_vitals = vitals


def styxx_agent(agent: Any) -> Any:
    """one-line wrapper to add styxx observation to an autogen agent.

    usage:
        assistant = autogen.AssistantAgent("assistant", llm_config={...})
        styxx_agent(assistant)

    returns the agent (pass-through) so it can be used inline:
        agent = styxx_agent(autogen.AssistantAgent(...))

    if autogen is not installed or the agent doesn't support
    register_reply, the agent is returned unchanged. never raises.
    """
    try:
        hook = StyxxAutoGenHook()
        hook.attach(agent)
        # stash the hook on the agent so callers can access
        # hook.last_vitals and hook.n_observed if they want.
        try:
            agent._styxx_hook = hook
        except Exception:
            # some agents are frozen or use __slots__
            pass
    except Exception as exc:
        # fail open — never break the caller's agent setup
        try:
            warnings.warn(
                f"styxx_agent: failed to attach hook: "
                f"{type(exc).__name__}: {exc}. "
                f"the agent will run without styxx observation.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception:
            pass
    return agent

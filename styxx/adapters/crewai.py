# -*- coding: utf-8 -*-
"""
styxx.adapters.crewai — thin bridge for crewai multi-agent crews.

crewai uses langchain internally for its LLM calls. this adapter
injects a styxx-aware langchain callback handler into a crew's
agent configuration so that every LLM call made by the crew's
agents gets cognitive vitals computed transparently.

usage:

    from crewai import Agent, Task, Crew
    from styxx.adapters.crewai import styxx_crew

    crew = Crew(agents=[...], tasks=[...])
    crew = styxx_crew(crew)   # inject styxx monitoring
    result = crew.kickoff()   # every LLM call now has vitals

the injection walks each agent in the crew and appends a
StyxxCrewCallback to its llm's callback list. the callback
captures per-token logprob data when available and feeds it
through the styxx runtime to produce vitals readings.

fail-open contract: if crewai is not installed, if the crew
object doesn't have the expected shape, or if callback injection
fails for any reason, styxx_crew() returns the crew unchanged.
the crew runs exactly as it would without styxx — no crash, no
side effect, no performance hit.

known limitations:
  - crewai's internal langchain usage may not always expose
    logprobs depending on the underlying LLM provider.
  - when logprobs are not available, the callback still fires
    but vitals will be None (same as the anthropic adapter).
  - tested against crewai >= 0.28.0. older versions may have
    a different agent/llm structure and will fail open.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional

from .. import config


def _get_langchain_base():
    """try to import langchain's BaseCallbackHandler. returns the
    class if available, None otherwise. never raises."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
        return BaseCallbackHandler
    except ImportError:
        pass
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        return BaseCallbackHandler
    except ImportError:
        pass
    return None


# build the base class tuple at import time. if langchain is
# available, StyxxCrewCallback inherits from BaseCallbackHandler
# so crewai's internal dispatch recognizes it. if langchain is
# absent, it's a plain object — still works as a duck-typed
# callback handler (langchain dispatches on method names, not
# strict isinstance checks in most code paths).
_LCBase = _get_langchain_base()
_BASES = (_LCBase,) if _LCBase is not None else (object,)


class StyxxCrewCallback(*_BASES):
    """langchain callback handler that computes styxx vitals on each
    LLM generation within a crewai crew.

    when langchain is installed, inherits from BaseCallbackHandler
    so crewai's internal plumbing recognizes it natively. when
    langchain is absent, works as a standalone duck-typed handler.
    either way, the callback is safe to instantiate and use.
    """

    def __init__(self):
        if _LCBase is not None:
            super().__init__()
        self._vitals_log: List[Any] = []
        # lazy import the runtime — only when we actually need it
        self._runtime = None

    def _get_runtime(self):
        if self._runtime is None:
            from ..core import StyxxRuntime
            self._runtime = StyxxRuntime()
        return self._runtime

    @property
    def vitals_log(self) -> List[Any]:
        """list of vitals objects computed during the crew run.
        each entry corresponds to one LLM generation that had
        logprob data available. entries are None when logprobs
        were not available."""
        return list(self._vitals_log)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """called by langchain after each LLM generation completes.

        attempts to extract logprob trajectories from the response
        and compute vitals. if extraction fails (no logprobs, wrong
        response shape, etc), appends None and moves on.
        """
        try:
            vitals = self._extract_and_compute(response)
            self._vitals_log.append(vitals)
        except Exception:
            # fail open — never break the crew's execution
            self._vitals_log.append(None)

    def _extract_and_compute(self, response: Any) -> Any:
        """attempt to pull logprob data from a langchain LLMResult
        and compute styxx vitals. returns None if the data is not
        available."""
        # langchain LLMResult has .generations which is a list of
        # lists of Generation objects. each Generation may have a
        # .generation_info dict with logprobs data.
        generations = getattr(response, "generations", None)
        if not generations or not generations[0]:
            return None

        gen = generations[0][0]
        info = getattr(gen, "generation_info", None) or {}
        logprobs = info.get("logprobs")
        if not logprobs:
            return None

        # try to extract token-level data from the logprobs dict.
        # openai-style logprobs have a "content" key with per-token
        # objects, or a flat "token_logprobs" list.
        import math
        entropy_traj = []
        logprob_traj = []
        top2_traj = []

        content = logprobs.get("content", None)
        if content and isinstance(content, list):
            for tok in content:
                chosen_lp = float(tok.get("logprob", 0.0))
                logprob_traj.append(chosen_lp)
                top_lps = tok.get("top_logprobs", [])
                if top_lps:
                    lps = [float(t.get("logprob", 0.0)) for t in top_lps]
                    probs = [math.exp(lp) for lp in lps]
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                        ent = -sum(
                            p * math.log(p + 1e-12)
                            for p in probs if p > 0
                        )
                    else:
                        ent = 0.0
                    sorted_probs = sorted(probs, reverse=True)
                    margin = (
                        float(sorted_probs[0] - sorted_probs[1])
                        if len(sorted_probs) >= 2
                        else 1.0
                    )
                    entropy_traj.append(float(ent))
                    top2_traj.append(margin)
                else:
                    entropy_traj.append(0.0)
                    top2_traj.append(1.0)
        else:
            # flat token_logprobs style (legacy openai)
            token_lps = logprobs.get("token_logprobs", [])
            top_lps_list = logprobs.get("top_logprobs", [])
            if not token_lps:
                return None
            for i, chosen_lp in enumerate(token_lps):
                if chosen_lp is None:
                    continue
                logprob_traj.append(float(chosen_lp))
                tops = top_lps_list[i] if i < len(top_lps_list) else {}
                if tops:
                    lps = [float(v) for v in tops.values()]
                    probs = [math.exp(lp) for lp in lps]
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                        ent = -sum(
                            p * math.log(p + 1e-12)
                            for p in probs if p > 0
                        )
                    else:
                        ent = 0.0
                    sorted_probs = sorted(probs, reverse=True)
                    margin = (
                        float(sorted_probs[0] - sorted_probs[1])
                        if len(sorted_probs) >= 2
                        else 1.0
                    )
                    entropy_traj.append(float(ent))
                    top2_traj.append(margin)
                else:
                    entropy_traj.append(0.0)
                    top2_traj.append(1.0)

        if not entropy_traj:
            return None

        runtime = self._get_runtime()
        return runtime.run_on_trajectories(
            entropy=entropy_traj,
            logprob=logprob_traj,
            top2_margin=top2_traj,
        )


def styxx_crew(crew: Any) -> Any:
    """inject styxx cognitive monitoring into a crewai crew.

    walks each agent in the crew and appends a StyxxCrewCallback
    to its LLM's callback list. returns the same crew object
    (mutated in place) so the caller can chain:

        result = styxx_crew(crew).kickoff()

    fail-open: if anything goes wrong — crewai not installed,
    unexpected crew structure, callback injection error — the
    crew is returned unchanged with a RuntimeWarning.

    Parameters
    ----------
    crew : crewai.Crew
        the crew instance to instrument.

    Returns
    -------
    crew : crewai.Crew
        the same crew object, with styxx callbacks injected.
    """
    # respect the global kill switch
    if config.is_disabled():
        return crew

    try:
        return _inject_callbacks(crew)
    except Exception as e:
        warnings.warn(
            f"styxx_crew: callback injection failed: "
            f"{type(e).__name__}: {e}. "
            f"returning crew unchanged.",
            RuntimeWarning,
        )
        return crew


def _inject_callbacks(crew: Any) -> Any:
    """internal: walk crew.agents and inject a StyxxCrewCallback
    into each agent's llm callback list."""
    # lazy-check that the crew looks like a crewai Crew
    agents = getattr(crew, "agents", None)
    if agents is None:
        warnings.warn(
            "styxx_crew: crew has no .agents attribute. "
            "returning unchanged.",
            RuntimeWarning,
        )
        return crew

    callback = StyxxCrewCallback()

    for agent in agents:
        llm = getattr(agent, "llm", None)
        if llm is None:
            continue

        # crewai agents expose their LLM as a langchain-compatible
        # object. callbacks can be on .callbacks (list) or we may
        # need to set it.
        existing = getattr(llm, "callbacks", None)
        if existing is None:
            try:
                llm.callbacks = [callback]
            except Exception:
                pass
        elif isinstance(existing, list):
            # avoid double-injection
            if not any(isinstance(cb, StyxxCrewCallback) for cb in existing):
                existing.append(callback)
        else:
            # callbacks is some other type — don't touch it
            pass

    # stash the callback on the crew so users can access vitals_log
    try:
        crew._styxx_callback = callback
    except Exception:
        pass

    return crew

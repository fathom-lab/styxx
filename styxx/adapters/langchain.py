# -*- coding: utf-8 -*-
"""
styxx.adapters.langchain — callback handler for langchain agents.

Drop a StyxxCallbackHandler into any langchain agent's callback list
and every LLM generation that carries logprobs gets cognitive vitals
attached automatically. No changes to your chain code, no changes to
your prompt templates. Just add the handler and read the vitals.

    from styxx.adapters.langchain import StyxxCallbackHandler
    handler = StyxxCallbackHandler()

    llm = ChatOpenAI(
        model="gpt-4o",
        callbacks=[handler],
        model_kwargs={"logprobs": True, "top_logprobs": 5},
    )
    result = llm.invoke("why is the sky blue?")

    print(handler.last_vitals)          # most recent Vitals or None
    print(handler.vitals_history)       # list of all Vitals computed

The handler hooks into langchain's callback system via on_llm_end and
on_llm_error. On every LLM completion, it attempts to extract an
openai-shaped response from the LLMResult and run it through
styxx.observe(). If logprobs are available, vitals get computed. If
not, the handler stays quiet and the agent continues unaffected.

Langchain is imported lazily inside methods — NOT at the top level —
so styxx never forces a hard dependency on langchain. If langchain
isn't installed, you can still import this module; the handler will
raise a clear error only when you try to instantiate it.

Fail-open contract
──────────────────
Like every styxx adapter, this handler never raises, never breaks
the host framework. Every public method is wrapped in try/except.
If vitals can't be computed (no logprobs, wrong response shape,
runtime error), the handler stores None and moves on. The agent
never notices.

This adapter was built for the xendro langchain integration path.
The shape was designed so any langchain agent — sequential chains,
agents with tools, LCEL pipelines — can get cognitive vitals with
one line of config.

credits: xendro for the shape request, darkflobi for the build.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from .. import config
from ..vitals import Vitals


class StyxxCallbackHandler:
    """Langchain callback handler that computes styxx cognitive vitals
    on every LLM generation.

    Instantiate and pass as a callback to any langchain LLM or chain.
    After each generation, check .last_vitals for the most recent
    reading or .vitals_history for the full sequence.

    The handler extends langchain's BaseCallbackHandler. The base
    class is resolved lazily at instantiation time so that langchain
    is not a hard import dependency of styxx.

    Example:
        from styxx.adapters.langchain import StyxxCallbackHandler
        handler = StyxxCallbackHandler()
        llm = ChatOpenAI(callbacks=[handler], ...)
        result = llm.invoke("hello")
        if handler.last_vitals is not None:
            print(handler.last_vitals.summary)
    """

    def __init__(self):
        # Verify langchain is available
        try:
            from langchain_core.callbacks import BaseCallbackHandler
        except ImportError:
            try:
                from langchain.callbacks.base import BaseCallbackHandler
            except ImportError as e:
                raise ImportError(
                    "styxx.adapters.langchain requires langchain.\n"
                    "  Install with:  pip install langchain-core\n"
                    "  Or:            pip install langchain\n"
                    f"  Underlying error: {e}"
                ) from e

        # Store the base class for isinstance checks if needed
        self._base_cls = BaseCallbackHandler

        # State
        self._vitals_history: List[Optional[Vitals]] = []
        self._last_vitals: Optional[Vitals] = None
        self._runtime = None

        # Respect the STYXX_DISABLED kill switch
        if not config.is_disabled():
            try:
                from ..core import StyxxRuntime
                self._runtime = StyxxRuntime()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────
    # public API
    # ─────────────────────────────────────────────────────────────

    @property
    def last_vitals(self) -> Optional[Vitals]:
        """The most recently computed Vitals, or None if no vitals
        have been computed yet or the last generation had no logprobs."""
        return self._last_vitals

    @property
    def vitals_history(self) -> List[Optional[Vitals]]:
        """Full history of vitals from every on_llm_end call.
        Entries are None for generations where logprobs were not
        available."""
        return list(self._vitals_history)

    # ─────────────────────────────────────────────────────────────
    # langchain callback interface
    # ─────────────────────────────────────────────────────────────

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called by langchain when an LLM generation completes.

        Extracts logprobs from the LLMResult if available, runs them
        through the styxx runtime, and stores the resulting vitals.
        Fails open on any error.
        """
        try:
            self._process_llm_result(response)
        except Exception:
            # fail open — never break the agent
            self._last_vitals = None
            self._vitals_history.append(None)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called by langchain when an LLM call errors.

        Logs a fail gate in vitals history. Never raises.
        """
        try:
            self._last_vitals = None
            self._vitals_history.append(None)
        except Exception:
            pass

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM starts generating. No-op for styxx."""
        pass

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called for each new token in streaming. No-op for styxx
        in the callback handler path (streaming vitals are handled
        by the reflex session if needed)."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any],
        *, run_id: Optional[UUID] = None, **kwargs: Any,
    ) -> None:
        pass

    def on_chain_end(
        self, outputs: Dict[str, Any],
        *, run_id: Optional[UUID] = None, **kwargs: Any,
    ) -> None:
        pass

    def on_chain_error(
        self, error: BaseException,
        *, run_id: Optional[UUID] = None, **kwargs: Any,
    ) -> None:
        pass

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str,
        *, run_id: Optional[UUID] = None, **kwargs: Any,
    ) -> None:
        pass

    def on_tool_end(
        self, output: str,
        *, run_id: Optional[UUID] = None, **kwargs: Any,
    ) -> None:
        pass

    def on_tool_error(
        self, error: BaseException,
        *, run_id: Optional[UUID] = None, **kwargs: Any,
    ) -> None:
        pass

    def on_text(
        self, text: str,
        *, run_id: Optional[UUID] = None, **kwargs: Any,
    ) -> None:
        pass

    # ─────────────────────────────────────────────────────────────
    # internal
    # ─────────────────────────────────────────────────────────────

    def _process_llm_result(self, response: Any) -> None:
        """Extract logprobs from a langchain LLMResult and compute
        vitals via styxx.observe().

        The LLMResult may contain an openai-shaped response object
        in its generation_info or llm_output. We try multiple
        extraction paths and fall back gracefully.
        """
        if self._runtime is None:
            self._last_vitals = None
            self._vitals_history.append(None)
            return

        # Path 1: try to extract from the LLMResult's generation_info
        # which often contains the raw openai response dict.
        openai_response = self._extract_openai_response(response)
        if openai_response is not None:
            from ..watch import observe
            vitals = observe(openai_response)
            self._last_vitals = vitals
            self._vitals_history.append(vitals)
            return

        # Path 2: check if the response already has a .vitals attribute
        # (e.g. if the LLM was wrapped by styxx.OpenAI and the result
        # was passed through langchain). Only trust it if it's a real
        # Vitals instance — not a MagicMock or arbitrary truthy object.
        pre_attached = getattr(response, "vitals", None)
        if isinstance(pre_attached, Vitals):
            self._last_vitals = pre_attached
            self._vitals_history.append(pre_attached)
            return

        # No logprobs found — store None, move on.
        self._last_vitals = None
        self._vitals_history.append(None)

    def _extract_openai_response(self, response: Any) -> Any:
        """Try to pull an openai-shaped response or raw logprob data
        out of a langchain LLMResult.

        Langchain's LLMResult structure:
          - response.generations: list of list of Generation objects
          - each Generation has .generation_info (dict with raw response data)
          - response.llm_output: dict with token usage and raw response

        For ChatOpenAI with logprobs=True, the generation_info often
        contains the openai logprobs in a structure we can feed
        directly to styxx.observe().
        """
        try:
            # Try generations[0][0].generation_info
            generations = getattr(response, "generations", None)
            if generations and len(generations) > 0:
                gen_list = generations[0]
                if gen_list and len(gen_list) > 0:
                    gen = gen_list[0]
                    info = getattr(gen, "generation_info", None)
                    if info and isinstance(info, dict):
                        # Check if there's a direct logprobs field
                        logprobs = info.get("logprobs", None)
                        if logprobs is not None:
                            # Build an openai-shaped response object
                            # that styxx.observe() can parse
                            return _build_openai_shaped(logprobs)
        except Exception:
            pass

        try:
            # Try llm_output for raw response data
            llm_output = getattr(response, "llm_output", None)
            if llm_output and isinstance(llm_output, dict):
                # Some providers store the full response here
                logprobs = llm_output.get("logprobs", None)
                if logprobs is not None:
                    return _build_openai_shaped(logprobs)
        except Exception:
            pass

        return None


# ──────────────────────────────────────────────────────────────────
# internal helpers
# ──────────────────────────────────────────────────────────────────

class _FakeTopLogprob:
    """Minimal stand-in for openai's TopLogprob."""
    def __init__(self, logprob: float):
        self.logprob = logprob


class _FakeTokenLogprob:
    """Minimal stand-in for openai's ChatCompletionTokenLogprob."""
    def __init__(self, token: str, logprob: float, top_logprobs: list):
        self.token = token
        self.logprob = logprob
        self.top_logprobs = top_logprobs


class _FakeLogprobsBlock:
    """Minimal stand-in for openai's ChoiceLogprobs."""
    def __init__(self, content: list):
        self.content = content


class _FakeChoice:
    """Minimal stand-in for openai's Choice."""
    def __init__(self, logprobs):
        self.logprobs = logprobs


class _FakeResponse:
    """Minimal openai-shaped response for styxx.observe() to parse."""
    def __init__(self, choices: list):
        self.choices = choices


def _build_openai_shaped(logprobs: Any) -> Optional[_FakeResponse]:
    """Build an openai-shaped response from langchain's logprob data.

    Langchain passes through the openai logprobs structure in various
    formats. We handle the common ones:
    1. A dict with a "content" key containing per-token logprob data
    2. An object with a .content attribute (already openai-shaped)
    """
    try:
        content = None

        if isinstance(logprobs, dict):
            content = logprobs.get("content", None)
        elif hasattr(logprobs, "content"):
            content = logprobs.content

        if content is None or not isinstance(content, (list, tuple)):
            return None

        # Build the token logprob list
        token_logprobs = []
        for tok_data in content:
            if isinstance(tok_data, dict):
                token = tok_data.get("token", "")
                lp = float(tok_data.get("logprob", 0.0))
                top_lps_raw = tok_data.get("top_logprobs", [])
                top_lps = [
                    _FakeTopLogprob(float(t.get("logprob", 0.0)))
                    if isinstance(t, dict) else t
                    for t in (top_lps_raw or [])
                ]
                token_logprobs.append(
                    _FakeTokenLogprob(token=token, logprob=lp, top_logprobs=top_lps)
                )
            elif hasattr(tok_data, "logprob"):
                # Already an openai-shaped object
                token_logprobs.append(tok_data)

        if not token_logprobs:
            return None

        return _FakeResponse(
            choices=[_FakeChoice(logprobs=_FakeLogprobsBlock(content=token_logprobs))]
        )
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────
# StyxxHallucinationGuard — the @trust equivalent for LangChain
# ────────────────────────────────────────────────────────────────────
#
# The callback handler above is passive observation. This class is
# active interception — wrap any LangChain Runnable and every output
# gets cognometrically verified before the chain returns. Based on the
# same 9-signal cross-validated detector that ships in @trust.
#
# Usage:
#
#     from langchain_openai import ChatOpenAI
#     from langchain_core.runnables import RunnablePassthrough
#     from styxx.adapters.langchain import StyxxHallucinationGuard
#
#     rag_chain = (
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | ChatOpenAI(model="gpt-4o")
#         | StrOutputParser()
#     )
#
#     guarded = StyxxHallucinationGuard(threshold=0.7).wrap(rag_chain)
#     answer = guarded.invoke("who directed inception?",
#                              config={"metadata": {
#                                  "context": "Inception (2010) was..."
#                              }})
#     # If the model's response scored ≥ 0.7 risk, answer is the styxx
#     # fallback text. Otherwise, pass-through.
#
# Works on any Runnable that produces a string or a dict with a
# reference-y key (context / passage / etc — same auto-detect list
# as @trust). Input metadata carries the grounding reference.
# ────────────────────────────────────────────────────────────────────

_REFERENCE_METADATA_KEYS = (
    "reference", "context", "passage", "passages",
    "docs", "documents", "source", "sources",
    "knowledge", "grounding", "retrieved", "retrieval",
)

_DEFAULT_FALLBACK = (
    "I'm not confident enough in my answer to give you one. "
    "Please verify with a trusted source."
)


def _extract_reference(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not metadata:
        return None
    for k in _REFERENCE_METADATA_KEYS:
        v = metadata.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, (list, tuple)) and all(
            isinstance(x, str) for x in v
        ) and v:
            return "\n".join(v)
    return None


class StyxxHallucinationGuard:
    """Wrap a LangChain Runnable with @trust-style hallucination gating.

    Verifies the Runnable's output via styxx.guardrail.check() with
    NLI + novelty signals, intercepts on high risk.

    Parameters mirror @trust: threshold, on_halt, fallback, max_retries,
    use_nli, use_entity_verify. See styxx.trust for details.

    Cross-validated numbers (3-seed averaged, n=150/dataset):

        HaluEval-QA           AUC 0.998
        TruthfulQA            AUC 0.994
        HaluBench-RAGTruth    AUC 0.807  (direct RAG faithfulness)
        HaluBench-PubMedQA    AUC 0.719
        HaluEval-Dialog       AUC 0.676
        HaluEval-Summ         AUC 0.643
        HaluBench-FinanceBench AUC 0.492  (published failure)
        HaluBench-DROP        AUC 0.424  (published failure)

    https://fathom.darkflobi.com/cognometry/failures documents the two
    failure modes — avoid this guard for extractive-span reading-comp
    or financial arithmetic.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        on_halt: str = "fallback",      # "fallback" | "raise" | "annotate"
        fallback: str = _DEFAULT_FALLBACK,
        max_retries: int = 2,
        use_nli: Optional[bool] = None,  # None = auto-detect styxx[nli]
        use_entity_verify: bool = False,
    ):
        if on_halt not in {"fallback", "raise", "annotate"}:
            raise ValueError(
                "on_halt must be one of: fallback, raise, annotate"
            )
        self._threshold = threshold
        self._on_halt = on_halt
        self._fallback = fallback
        self._max_retries = max_retries
        self._use_nli = use_nli
        self._use_entity_verify = use_entity_verify

    def wrap(self, runnable):
        """Wrap a LangChain Runnable. Returns a Runnable with identical
        input interface but gated output.
        """
        try:
            from langchain_core.runnables import (
                RunnableLambda, RunnablePassthrough,
            )
            from langchain_core.runnables.config import RunnableConfig
        except ImportError as e:
            raise ImportError(
                "StyxxHallucinationGuard.wrap() requires langchain-core. "
                "Install with: pip install langchain-core"
            ) from e

        guard = self  # closure over self for RunnableLambda

        def _check(output, config=None):
            from ..guardrail import check
            metadata = (config or {}).get("metadata", {}) if config else {}
            reference = _extract_reference(metadata)
            prompt = metadata.get("prompt") or metadata.get("question", "")
            response_text = (
                output if isinstance(output, str)
                else str(output.get("answer") if isinstance(output, dict)
                         else output)
            )
            verdict = check(
                prompt=str(prompt),
                response=response_text,
                reference=reference,
                use_nli=guard._use_nli,
                use_entity_verify=guard._use_entity_verify,
            )
            if verdict.risk < guard._threshold:
                return output
            if guard._on_halt == "raise":
                raise RuntimeError(
                    f"styxx hallucination risk {verdict.risk:.3f} "
                    f"exceeded threshold {guard._threshold:.2f}"
                )
            if guard._on_halt == "annotate":
                return {
                    "output": output,
                    "styxx_risk": verdict.risk,
                    "styxx_action": verdict.action,
                    "styxx_halted": True,
                }
            # fallback
            if isinstance(output, dict):
                out = dict(output)
                out["answer"] = guard._fallback
                return out
            return guard._fallback

        checker = RunnableLambda(_check)
        # Compose: run the original runnable, then apply the check
        return runnable | checker


__all__ = ["StyxxCallbackHandler", "StyxxHallucinationGuard"]

# -*- coding: utf-8 -*-
"""LlamaIndex evaluator — styxx-backed hallucination detection for RAG.

Drops into the LlamaIndex Evaluator protocol. Strongest benchmark
number (HaluBench-RAGTruth AUC 0.807) maps directly onto LlamaIndex's
core RAG pipelines.

Usage:

    from llama_index.core.evaluation import EvaluationResult
    from styxx.adapters.llamaindex import StyxxHallucinationEvaluator

    evaluator = StyxxHallucinationEvaluator(threshold=0.7)

    # `response` is a Response object from a LlamaIndex query engine
    result = evaluator.evaluate_response(query=query, response=response)
    result.passing    # True/False
    result.score      # 1.0 - risk (for consistency with other evaluators)
    result.feedback   # human-readable explanation
    result.metadata   # {"risk": float, "action": str, "signals": {...}}

Or by components:

    result = evaluator.evaluate(
        query="who directed inception?",
        response="Inception was directed by Christopher Nolan.",
        contexts=["Inception is a 2010 film directed by Christopher Nolan."],
    )

Requires: pip install styxx[nli] llama-index-core
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Sequence

try:
    from llama_index.core.evaluation import BaseEvaluator, EvaluationResult
except ImportError as e:
    raise ImportError(
        "styxx.adapters.llamaindex requires `llama-index-core`. "
        "Install with: pip install llama-index-core"
    ) from e


class StyxxHallucinationEvaluator(BaseEvaluator):
    """Hallucination detection backed by styxx's 9-signal
    cross-validated detector.

    Reported benchmark AUCs on RAG-relevant datasets (3-seed averaged):

        HaluEval-QA           0.998
        TruthfulQA            0.994
        HaluBench-RAGTruth    0.807   <-- the direct RAG faithfulness hit
        HaluBench-PubMedQA    0.719

    Known failure modes (not suitable for these domains):

        HaluBench-DROP        0.424   extractive-span reading comp
        HaluBench-FinanceBench 0.492  financial arithmetic

    See https://fathom.darkflobi.com/cognometry/failures for structural
    explanation and probe evidence.

    Parameters
    ----------
    threshold : float
        Risk threshold in [0, 1]. A response with risk >= threshold is
        classified as hallucinated. Default 0.7 (same as ``@trust``).
    use_nli : bool, optional
        Enable NLI contradiction signal. Default None = auto-enable if
        ``styxx[nli]`` is installed.
    use_entity_verify : bool
        Run Wikipedia entity verification. Off by default for speed.
    use_probe : bool
        Use the residual-level confab probe. Off by default.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        use_nli: Optional[bool] = None,
        use_entity_verify: bool = False,
        use_probe: bool = False,
    ):
        self._threshold = threshold
        self._use_nli = use_nli
        self._use_entity_verify = use_entity_verify
        self._use_probe = use_probe

    def _get_prompts(self):
        return {}

    def _update_prompts(self, prompts):
        pass

    @classmethod
    def class_name(cls) -> str:
        return "StyxxHallucinationEvaluator"

    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        sleep_time_in_seconds: int = 0,
        **kwargs: Any,
    ) -> EvaluationResult:
        from ..guardrail import check

        response_text = str(response) if response is not None else ""
        if not response_text.strip():
            return EvaluationResult(
                query=query,
                response=response_text,
                passing=True,
                score=1.0,
                feedback="empty response",
            )

        # Merge contexts (LlamaIndex native) and reference
        # (explicit override) into one reference string.
        ref_parts = []
        if contexts:
            ref_parts.extend(str(c) for c in contexts if c)
        if reference:
            ref_parts.append(str(reference))
        ref = "\n\n".join(ref_parts) if ref_parts else None

        verdict = check(
            prompt=str(query) if query else "",
            response=response_text,
            reference=ref,
            use_entity_verify=self._use_entity_verify,
            use_probe=self._use_probe,
            use_nli=self._use_nli,
        )

        risk = float(verdict.risk)
        passing = risk < self._threshold
        signals = {
            s.name: (round(float(s.value), 4)
                     if isinstance(s.value, (int, float)) else s.value)
            for s in verdict.signals
        }

        return EvaluationResult(
            query=query,
            response=response_text,
            contexts=list(contexts) if contexts else None,
            passing=passing,
            score=round(1.0 - risk, 4),  # high score = faithful
            feedback=(
                f"hallucination risk {risk:.3f} (threshold {self._threshold:.2f}). "
                f"action={verdict.action}. "
                f"signals={sorted(signals.keys())}"
            ),
            metadata={
                "risk": risk,
                "action": verdict.action,
                "threshold": self._threshold,
                "signals": signals,
            },
        )

    def evaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Synchronous convenience wrapper around aevaluate."""
        # If we're already inside a running loop, asyncio.run() would raise a
        # cryptic "cannot be called from a running event loop". Detect that and
        # raise the actionable guidance instead. (The previous version raised
        # this message then immediately swallowed it with `except RuntimeError:
        # pass` and fell through to the cryptic error anyway.)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no running loop — safe to use asyncio.run below
        else:
            raise RuntimeError(
                "evaluate() called from within a running event loop; "
                "await aevaluate() instead."
            )
        return asyncio.run(
            self.aevaluate(
                query=query,
                response=response,
                contexts=contexts,
                reference=reference,
                **kwargs,
            )
        )


__all__ = ["StyxxHallucinationEvaluator"]

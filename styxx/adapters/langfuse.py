# -*- coding: utf-8 -*-
"""
styxx.adapters.langfuse — inject cognitive vitals into Langfuse traces.

    from styxx.adapters.langfuse import StyxxLangfuseHandler
    handler = StyxxLangfuseHandler()

    llm = ChatOpenAI(
        model="gpt-4o",
        callbacks=[handler],
        model_kwargs={"logprobs": True, "top_logprobs": 5},
    )
    result = llm.invoke("why is the sky blue?")

    # vitals appear as Langfuse scores:
    #   styxx_gate: 1.0 (pass)
    #   styxx_phase4_confidence: 0.45

When used with Langfuse tracing, every LLM call gets styxx vitals
posted as numeric SCORES on the trace — visible in dashboards, charts,
and filters. The full vitals dict is also attached as observation
metadata for drill-down.

Gate mapping:
    pass    → 1.0
    warn    → 0.5
    fail    → 0.0
    pending → None (no score posted)

For users NOT using LangChain, the standalone function
enrich_langfuse_trace(client, trace_id, vitals) posts vitals to an
existing trace via the Langfuse Python SDK.

Fail-open on everything. If Langfuse is not installed, not configured,
or the API call fails, the handler still computes vitals (accessible
via .last_vitals). The agent never notices.

0.7.0+.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..vitals import Vitals
from .langchain import StyxxCallbackHandler


# ══════════════════════════════════════════════════════════════════
# Gate → numeric score
# ══════════════════════════════════════════════════════════════════

GATE_SCORES: Dict[str, float] = {
    "pass": 1.0,
    "warn": 0.5,
    "fail": 0.0,
}


def _gate_to_score(gate: str) -> Optional[float]:
    """Map a styxx gate label to a Langfuse numeric score.
    Returns None for 'pending' (don't post a score)."""
    return GATE_SCORES.get(gate)


# ══════════════════════════════════════════════════════════════════
# Standalone enrichment function
# ══════════════════════════════════════════════════════════════════

def enrich_langfuse_trace(
    client: Any,
    trace_id: str,
    vitals: Optional[Vitals],
) -> None:
    """Post styxx vitals as scores + metadata to an existing Langfuse trace.

    For users who are NOT using LangChain but have a trace_id from
    the Langfuse Python SDK. Works with styxx.OpenAI() or
    styxx.observe() output.

    Args:
        client:   a langfuse.Langfuse instance
        trace_id: the Langfuse trace ID to enrich
        vitals:   a Vitals object (or None, in which case this is a no-op)

    Usage:
        from langfuse import Langfuse
        from styxx.adapters.langfuse import enrich_langfuse_trace

        lf = Langfuse()
        trace = lf.trace(name="my-agent-call")

        response = styxx_client.chat.completions.create(...)
        enrich_langfuse_trace(lf, trace.id, response.vitals)
    """
    if vitals is None or client is None:
        return

    try:
        # Post gate score
        gate_val = _gate_to_score(vitals.gate)
        if gate_val is not None:
            client.score(
                trace_id=trace_id,
                name="styxx_gate",
                value=gate_val,
                comment=f"gate={vitals.gate}",
            )

        # Post phase 4 confidence
        if vitals.phase4_late is not None:
            client.score(
                trace_id=trace_id,
                name="styxx_phase4_confidence",
                value=round(vitals.phase4_late.confidence, 4),
                comment=f"category={vitals.phase4_late.predicted_category}",
            )

        # Post phase 1 confidence
        if vitals.phase1_pre is not None:
            client.score(
                trace_id=trace_id,
                name="styxx_phase1_confidence",
                value=round(vitals.phase1_pre.confidence, 4),
                comment=f"category={vitals.phase1_pre.predicted_category}",
            )

    except Exception:
        pass  # fail open


# ══════════════════════════════════════════════════════════════════
# Handler
# ══════════════════════════════════════════════════════════════════

class StyxxLangfuseHandler(StyxxCallbackHandler):
    """LangChain callback handler that computes styxx cognitive vitals
    AND posts them as Langfuse scores on the active trace.

    Subclasses StyxxCallbackHandler. All vitals computation is
    inherited. This class adds the Langfuse score posting on top.

    The Langfuse client is created lazily from LANGFUSE_PUBLIC_KEY
    and LANGFUSE_SECRET_KEY env vars (same convention as the Langfuse
    SDK). Or pass a pre-configured client to the constructor.

    If Langfuse is not installed, not configured, or any API call fails,
    the handler still works as a normal StyxxCallbackHandler. Fail-open.

    Example:
        from styxx.adapters.langfuse import StyxxLangfuseHandler
        handler = StyxxLangfuseHandler()
        llm = ChatOpenAI(callbacks=[handler], ...)
        result = llm.invoke("hello")

        # vitals in handler.last_vitals (always)
        # vitals as Langfuse scores (when configured)
    """

    def __init__(self, client: Any = None):
        super().__init__()
        self._langfuse_client = client
        self._langfuse_available: Optional[bool] = None

    def _get_langfuse(self) -> Any:
        """Lazy-init the Langfuse client. Returns None if not available."""
        if self._langfuse_client is not None:
            return self._langfuse_client

        if self._langfuse_available is False:
            return None

        try:
            from langfuse import Langfuse
            self._langfuse_client = Langfuse()
            self._langfuse_available = True
            return self._langfuse_client
        except ImportError:
            self._langfuse_available = False
            return None
        except Exception:
            self._langfuse_available = False
            return None

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Compute vitals (via parent), then post to Langfuse."""
        # Step 1: compute vitals via the parent handler
        super().on_llm_end(
            response,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

        # Step 2: post to Langfuse (fail-open)
        try:
            self._post_to_langfuse(run_id)
        except Exception:
            pass  # never break the agent

    def _post_to_langfuse(self, run_id: Optional[UUID] = None) -> None:
        """Post vitals scores to the Langfuse trace."""
        if self._last_vitals is None:
            return

        client = self._get_langfuse()
        if client is None:
            return

        # Use run_id as trace identifier
        trace_id = str(run_id) if run_id else None
        if trace_id is None:
            return

        enrich_langfuse_trace(client, trace_id, self._last_vitals)

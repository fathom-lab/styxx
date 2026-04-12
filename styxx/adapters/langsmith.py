# -*- coding: utf-8 -*-
"""
styxx.adapters.langsmith — inject cognitive vitals into LangSmith traces.

    from styxx.adapters.langsmith import StyxxLangSmithHandler
    handler = StyxxLangSmithHandler()

    llm = ChatOpenAI(
        model="gpt-4o",
        callbacks=[handler],
        model_kwargs={"logprobs": True, "top_logprobs": 5},
    )
    result = llm.invoke("why is the sky blue?")

    # vitals appear in your LangSmith trace as flat metadata:
    #   styxx_phase4_category: "reasoning"
    #   styxx_gate: "pass"
    #   styxx_phase4_confidence: 0.45

When used with LangSmith tracing enabled, every LLM call in your
trace gets styxx cognitive vitals as searchable/filterable metadata
on the run. No changes to your chain code — just swap the handler.

The handler extends the existing StyxxCallbackHandler (which does
the actual vitals computation), then patches the active LangSmith
run tree with a flat metadata dict. Flat keys because LangSmith's
UI filters work on top-level metadata, not nested structures.

If LangSmith is not configured or not installed, the handler still
computes vitals (accessible via .last_vitals) — the LangSmith
enrichment simply becomes a no-op. Fail-open on everything.

0.7.0+.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..vitals import Vitals
from .langchain import StyxxCallbackHandler


# ══════════════════════════════════════════════════════════════════
# Metadata helper
# ══════════════════════════════════════════════════════════════════

def langsmith_metadata(vitals: Optional[Vitals]) -> Dict[str, Any]:
    """Convert a Vitals object to a flat dict suitable for LangSmith
    run metadata.

    Flat keys (not nested) because LangSmith's filtering, search,
    and dashboard columns work on top-level metadata keys.

    Returns an empty dict if vitals is None.

    Usage:
        from styxx.adapters.langsmith import langsmith_metadata
        metadata = langsmith_metadata(vitals)
        # {"styxx_phase1_category": "reasoning", "styxx_gate": "pass", ...}
    """
    if vitals is None:
        return {}

    meta: Dict[str, Any] = {
        "styxx_tier": vitals.tier_active,
        "styxx_gate": vitals.gate,
    }

    # Phase 1
    if vitals.phase1_pre is not None:
        meta["styxx_phase1_category"] = vitals.phase1_pre.predicted_category
        meta["styxx_phase1_confidence"] = round(vitals.phase1_pre.confidence, 4)

    # Phase 4
    if vitals.phase4_late is not None:
        meta["styxx_phase4_category"] = vitals.phase4_late.predicted_category
        meta["styxx_phase4_confidence"] = round(vitals.phase4_late.confidence, 4)

    # D-axis (tier 1)
    d = vitals.d_honesty
    if d is not None:
        meta["styxx_d_honesty"] = d

    # Abort reason
    if vitals.abort_reason is not None:
        meta["styxx_abort_reason"] = vitals.abort_reason

    return meta


# ══════════════════════════════════════════════════════════════════
# Handler
# ══════════════════════════════════════════════════════════════════

class StyxxLangSmithHandler(StyxxCallbackHandler):
    """LangChain callback handler that computes styxx cognitive vitals
    AND injects them into the active LangSmith trace as flat metadata.

    Subclasses StyxxCallbackHandler — all vitals computation is
    inherited. This class adds the LangSmith enrichment on top.

    If LangSmith is not installed or no trace is active, the handler
    still works as a normal StyxxCallbackHandler (vitals are computed
    and stored in .last_vitals / .vitals_history). The LangSmith
    metadata injection is purely additive.

    Example:
        from styxx.adapters.langsmith import StyxxLangSmithHandler
        handler = StyxxLangSmithHandler()
        llm = ChatOpenAI(callbacks=[handler], ...)
        result = llm.invoke("hello")

        # vitals in handler.last_vitals (always)
        # vitals in LangSmith trace metadata (when tracing is active)
    """

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Compute vitals (via parent), then patch the LangSmith run."""
        # Step 1: compute vitals via the parent handler
        super().on_llm_end(
            response,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

        # Step 2: inject into LangSmith (fail-open)
        try:
            self._patch_langsmith_run()
        except Exception:
            pass  # never break the agent for a metadata write

    def _patch_langsmith_run(self) -> None:
        """Attempt to patch the current LangSmith run tree with vitals
        metadata. Silently returns if langsmith is not available or
        no run is active."""
        if self._last_vitals is None:
            return

        try:
            from langsmith.run_helpers import get_current_run_tree
        except ImportError:
            return  # langsmith not installed — that's fine

        run_tree = get_current_run_tree()
        if run_tree is None:
            return  # no active trace — that's fine

        meta = langsmith_metadata(self._last_vitals)
        if not meta:
            return

        # Merge into existing metadata
        existing = getattr(run_tree, "metadata", None) or {}
        existing.update(meta)

        # Patch the run tree
        if hasattr(run_tree, "patch"):
            run_tree.patch(metadata=existing)
        elif hasattr(run_tree, "metadata"):
            run_tree.metadata = existing

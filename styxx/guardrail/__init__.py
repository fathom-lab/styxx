# -*- coding: utf-8 -*-
"""
styxx.guardrail — multi-signal hallucination-prevention system.

Not a detector. A prevention pipeline. A response from any LLM is
decomposed into atomic claims; each claim is scored by multiple
independent signals (text features, entity verification, optional
probe readings, optional self-consistency); signals are fused by
an isotonic-calibrated regressor; an action policy decides what
to do with the result.

Top-level entry points:

    from styxx.guardrail import check, Verdict

    verdict = check(
        prompt="Who wrote Hamlet?",
        response="Hamlet was written by William Shakespeare...",
        model="meta-llama/Llama-3.2-1B-Instruct",  # optional
        use_probe=True,           # residual-level signal if model loaded
        use_consensus=False,      # self-consistency resampling (slow)
        use_entity_verify=True,   # Wikipedia entity grounding
    )
    # verdict.risk          ∈ [0, 1], calibrated
    # verdict.spans         = [{span, risk, reasons}, ...]
    # verdict.action        = "halt" | "annotate" | "retry" | "pass"
    # verdict.signal_details = per-signal readings

Sub-modules:

    claim_decomposer  — response → atomic claims
    entity_verify     — Wikipedia-based claim grounding
    text_signals      — text-feature prior (length-ratio, entity-density)
    probe_signal      — residual-level (requires HF model)
    consensus_signal  — self-consistency resampling
    fusion            — isotonic-calibrated signal combination
    policy            — action decision
"""
from __future__ import annotations

from .types import Verdict, Span, SignalReading  # noqa: F401
from .entry import check  # noqa: F401
# Second cognometric instrument (v5.0): text-only refusal detector.
# Cross-validated law II: trained on JBB/Llama-1B, XSTest/GPT-4 AUC 0.976,
# mean cross-model AUC 0.794 across 5 model families, n=2,250 held-out.
from .refusal import refuse_check, RefusalVerdict  # noqa: F401
# Third cognometric instrument (v6.0): tool-call drift detection.
# Calibrated on BFCL v3, 5-fold CV AUC 0.916. First text-only detector
# to beat the Healy et al. 2026 hidden-state baseline (AUC 0.72 on Glaive).
from .drift import drift_check, DriftVerdict  # noqa: F401

__all__ = [
    "check",
    "Verdict",
    "Span",
    "SignalReading",
    "refuse_check",
    "RefusalVerdict",
    "drift_check",
    "DriftVerdict",
]

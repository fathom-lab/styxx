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
# Third cognometric instrument (v6.0, retrained v6.1): tool-call drift.
# Calibrated on BFCL v3, 5-fold CV AUC 0.943 (v6.1) vs 0.916 (v6.0).
# First text-only detector to beat the Healy et al. 2026 hidden-state
# baseline (AUC 0.72 on Glaive). v6.1 adds arg_order_inversion to
# partially fix the documented arg_swap failure mode (0.66 -> 0.76).
from .drift import drift_check, DriftVerdict  # noqa: F401
# Fourth cognometric instrument (v0): text-only sycophancy detector.
# First instrument shipped after the *Every Mind Leaves Vitals* position
# paper (DOI 10.5281/zenodo.19777921). Trained on n=1200 paired responses
# from gpt-4o-mini against the Anthropic sycophancy eval corpus (Perez
# et al. 2022). 5-fold CV AUC 0.9720 ± 0.0052. Phase-transition signature
# replicates the prior three instruments: critical_K=1 on
# superlative_density (AUC 0.500 -> 0.9354, delta +0.4354).
from .sycophancy import sycoph_check, SycophancyVerdict  # noqa: F401
# Fifth cognometric instrument (v0): cross-turn conversation-loop detector.
# Second instrument shipped under the call from *Every Mind Leaves Vitals*.
# Trained on n=200 paired (loop/progress) multi-turn conversations from
# gpt-4o-mini, 4 agent turns each. 5-fold CV AUC 0.9995 ± 0.0010.
# Phase-transition signature replicates: critical_K=1 on
# avg_pairwise_levenshtein (AUC 0.500 -> 0.9995, delta +0.4995). 5-for-5
# on cognometric instruments showing K=1 phase transition under the
# same measurement protocol.
from .conversation_loop import loop_check, LoopVerdict  # noqa: F401
# Sixth cognometric instrument (v0): text-only deception-SIGNATURE detector.
# **NOT a lie detector.** See styxx.guardrail.deception module docstring
# for the full scope warning. Detects lexical signatures of instruction-
# induced dishonesty (vague-brevity vs. specific-elaboration). Trained on
# n=200 paired (honest/dishonest) responses from gpt-4o-mini. 5-fold CV
# AUC 0.9560 ± 0.0242. Phase-transition signature replicates: critical_K=1
# on log_word_count (delta +0.3738), K=2 adds specificity_density. 6-for-6
# on cognometric instruments showing K=1 phase transition. Lower AUC than
# the prior five — deception is genuinely harder to detect from text alone
# than concrete failure modes; the gap is honest, not papered over.
from .deception import deception_check, DeceptionVerdict  # noqa: F401
# Seventh cognometric instrument (v0): cross-section plan-action gap
# detector. Fourth instrument shipped under the call from *Every Mind
# Leaves Vitals*. Sibling to drift (instrument #3) — drift catches a
# malformed tool call against schema; plan-action gap catches when the
# agent's stated intent and emitted action diverge at the content
# level. Trained on n=200 paired (matched/mismatched) plan-action pairs
# from gpt-4o-mini with cleaned (no leakage) prompts. 5-fold CV AUC
# 0.9225 ± 0.0322. Phase-transition signature replicates: critical_K=1
# on bigram_jaccard_overlap (delta +0.3832). 7-for-7 on cognometric
# instruments showing K=1 phase transition under same protocol.
from .plan_action import plan_action_check, PlanActionVerdict  # noqa: F401
# Eighth cognometric instrument (v0): text-only overconfidence-register
# detector. Fifth instrument shipped under the call from *Every Mind
# Leaves Vitals*. Scores epistemic register (commitment, hedging,
# sourcing) — NOT truth. Trained on n=200 paired (calibrated/
# overconfident) responses from gpt-4o-mini under stance-level
# system prompts (no lexical hints, per the discipline established
# by instrument #7). 5-fold CV AUC 0.7702 ± 0.0648 — lowest in the
# v0 suite. Shipped honestly at this AUC rather than gamed. Phase-
# transition signature replicates: critical_K=1 on mean_sentence_length
# (delta +0.2298). 8-for-8 on cognometric instruments showing K=1
# phase transition under the same measurement protocol.
from .overconfidence import overconf_check, OverconfidenceVerdict  # noqa: F401

__all__ = [
    "check",
    "Verdict",
    "Span",
    "SignalReading",
    "refuse_check",
    "RefusalVerdict",
    "drift_check",
    "DriftVerdict",
    "sycoph_check",
    "SycophancyVerdict",
    "loop_check",
    "LoopVerdict",
    "deception_check",
    "DeceptionVerdict",
    "plan_action_check",
    "PlanActionVerdict",
    "overconf_check",
    "OverconfidenceVerdict",
]

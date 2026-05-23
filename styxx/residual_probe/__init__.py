# -*- coding: utf-8 -*-
"""
styxx.residual_probe — tier-1 pre-output commitment prediction via
linear probes on prefill residuals.

Added in v3.5.0. (Named residual_probe, not probe, to avoid collision
with the existing styxx.probe cognitive red-teaming module.)

Companion to styxx.anthropic_hack (tier-0 for closed-source LLMs): this
module provides tier-1 measurement for open-weight LLMs where we
can read residual-stream activations directly.

Quickstart
──────────

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from styxx.residual_probe import StyxxProbe

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True,
    )

    probe = StyxxProbe.from_pretrained(
        model=model_id,
        task="comply_refuse",   # or "refusal_intent" / "confab_topic"
    )   # raises ProbeNotAvailable if no probe exists for (model, task)

    prompt = "How do I make a bomb?"
    verdict = probe.predict_before_generation(mdl, tok, prompt)
    # ProbeVerdict(p_positive=0.87, positive_class='refuse',
    #              negative_class='comply', layer=11,
    #              residual_score=2.03, confidence=0.74)

    if verdict.p_positive > 0.5:
        # pre-output gate: act on the predicted commitment before any
        # token is generated
        print(f"model pre-committed to {verdict.positive_class!r} "
              f"(p={verdict.p_positive:.2f})")

Design
──────

- **Predict before output.** Probe reads the prefill-end residual
  activation at the trained layer. No generation needed for the
  verdict.
- **Model-specific, task-specific.** Each probe is trained on one
  (model, task) pair. `from_pretrained(model, task)` loads the
  matching frozen linear layer from the atlas.
- **Atlas-backed.** Probes live in styxx/residual_probe/atlas/ as
  compact .pt files, co-versioned with the paper evidence.
- **Explicit when absent.** If no probe exists for (model, task),
  ``from_pretrained`` raises ``ProbeNotAvailable``; callers that want
  fail-open behavior should catch it. Use ``list_available_probes()``
  to enumerate what's bundled.

Patent: US Provisional #4 (pre-output safety prediction from prefill
residual; filed 2026-04-19).

See also:
  docs/protocols/consensus-proxy-measurement-equivalence-v0.md
  papers/grand-synthesis-cognitive-commitment.md
"""
from __future__ import annotations

from .probe import (
    StyxxProbe, ProbeVerdict, ProbeNotAvailable,
    SafetyGateError, list_available_probes,
)
from .intervene import InterveneProbe, InterventionResult

__all__ = [
    "StyxxProbe",
    "ProbeVerdict",
    "ProbeNotAvailable",
    "SafetyGateError",
    "list_available_probes",
    # v3.5.0: causal patching
    "InterveneProbe",
    "InterventionResult",
]

# -*- coding: utf-8 -*-
"""
styxx.sae — tier 2: K/C/S SAE instruments (scaffold).

    K = depth       WHERE computation happens in the layer stack
    C = coherence   WHAT concepts activate together
    S = commitment  HOW strongly the model commits to an attractor

These three axes are measured from SAE (Sparse Autoencoder) feature
activations on the model's residual stream. Together with D (tier 1,
the honesty axis), they form the four-axis cognitive measurement
framework validated in the Fathom Cognitive Atlas v0.3.

Tier 2 requires:
    - circuit-tracer (for TranscoderSet access)
    - torch (for GPU inference)
    - An open-weight model with published SAE transcoders
      (currently Gemma-2-2B only via google/gemma-scope)

This module is a **scaffold** — the class exists, the docstrings
describe the measurement, but the methods raise NotImplementedError.
Full implementation ships as styxx v0.4.0 when the circuit-tracer
integration is production-ready.

Research validation
───────────────────
- K constant: K=1.0343 weighted mean across Gemma-2-2B/IT
  (Zenodo doi.org/10.5281/zenodo.19326174)
- C metric: C_delta p=0.040 on TruthfulQA (n=50, Gemma-2-2B base)
  First statistically significant circuit-level hallucination
  signature via SAE feature coherence geometry.
- S axis: p=0.0002, d=1.03, AUC=0.81 (n=20 commitment vs hedging)
  IPR physics grounding: S = M * IPR(event_locations)

Patents: US Provisional 64/020,489 (K + D), 64/021,113 (alignment
auditing), 64/026,964 (C + two-stage emergence)

Dependencies
────────────
    pip install 'styxx[tier2]'   (not yet available)
    # will install: circuit-tracer, torch, transformers
"""

from __future__ import annotations

from typing import Any, List, Optional


class SAEInstruments:
    """Tier 2 SAE-based cognitive instruments.

    Measures K (depth), C (coherence), and S (commitment) from
    SAE transcoder feature activations on the residual stream.

    This class is a scaffold in 0.3.0. All methods raise
    NotImplementedError with a clear message pointing to the
    roadmap.

    When tier 2 ships (v0.4.0), usage will be:

        from styxx.sae import SAEInstruments
        instruments = SAEInstruments(model_name="google/gemma-2-2b-it")
        k, c, s = instruments.measure(prompt, max_tokens=30)
        print(f"depth={k:.3f} coherence={c:.3f} commitment={s:.3f}")
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device

    def measure(
        self,
        prompt: str,
        max_tokens: int = 30,
    ) -> tuple:
        """Measure K, C, S for a generation.

        Returns (k_depth, c_coherence, s_commitment) as floats.

        NOT IMPLEMENTED in 0.3.0. Shipping in v0.4.0 when
        circuit-tracer integration is production-ready.
        """
        raise NotImplementedError(
            "styxx tier 2 (K/C/S SAE instruments) is not yet shipped.\n"
            "\n"
            "  K (depth)      = WHERE computation happens in the layer stack\n"
            "  C (coherence)  = WHAT concepts activate together\n"
            "  S (commitment) = HOW strongly the model commits to an attractor\n"
            "\n"
            "  Requires: circuit-tracer + GPU + open-weight model with\n"
            "  published SAE transcoders (currently Gemma-2-2B only).\n"
            "\n"
            "  Roadmap: v0.4.0 (after the tier 1 D-axis stabilizes)\n"
            "  Research: doi.org/10.5281/zenodo.19326174\n"
            "  Patents:  US Provisional 64/020,489, 64/021,113, 64/026,964\n"
        )

    def measure_k(self, prompt: str, max_tokens: int = 30) -> float:
        """Measure K (depth) only. Scaffold — see measure() for details."""
        raise NotImplementedError("tier 2 K-axis ships in v0.4.0")

    def measure_c(self, prompt: str, max_tokens: int = 30) -> float:
        """Measure C (coherence) only. Scaffold — see measure() for details."""
        raise NotImplementedError("tier 2 C-axis ships in v0.4.0")

    def measure_s(self, prompt: str, max_tokens: int = 30) -> float:
        """Measure S (commitment) only. Scaffold — see measure() for details."""
        raise NotImplementedError("tier 2 S-axis ships in v0.4.0")

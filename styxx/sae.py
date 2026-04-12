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

    0.4.0: upgraded from scaffold to real implementation.
    Delegates to styxx.kcs.KCSAxis for the actual computation.

    Usage:

        from styxx.sae import SAEInstruments
        instruments = SAEInstruments(model_name="google/gemma-2-2b-it")
        result = instruments.measure("why is the sky blue?")
        print(f"K={result.weighted_depth:.2f}")
        print(f"C_delta={result.c_delta:.4f}")
        print(f"S_early={result.s_early:.4f}")
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b-it",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            from .kcs import KCSAxis
            self._engine = KCSAxis(
                model_name=self.model_name,
                device=self.device,
            )

    def measure(self, prompt: str, max_tokens: int = 30):
        """Full K/C/S measurement. Returns a KCSResult."""
        self._ensure_engine()
        return self._engine.score(prompt)

    def measure_trajectory(self, prompt: str, max_tokens: int = 30):
        """K/C/S with per-token trajectories. Returns a KCSResult."""
        self._ensure_engine()
        return self._engine.score_trajectory(prompt, max_tokens=max_tokens)

    def measure_k(self, prompt: str, max_tokens: int = 30) -> float:
        """K (depth) only."""
        result = self.measure(prompt)
        return result.weighted_depth

    def measure_c(self, prompt: str, max_tokens: int = 30) -> Optional[float]:
        """C (coherence) only."""
        result = self.measure(prompt)
        return result.c_delta

    def measure_s(self, prompt: str, max_tokens: int = 30) -> Optional[float]:
        """S (commitment) only. Requires trajectory-mode measurement."""
        result = self.measure_trajectory(prompt, max_tokens=max_tokens)
        return result.s_early

    def unload(self):
        if self._engine is not None:
            self._engine.unload()
            self._engine = None

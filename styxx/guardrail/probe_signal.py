# -*- coding: utf-8 -*-
"""
Residual-level probe signal for the guardrail pipeline.

Uses the behavioral confabulation probe (`confab_behavioral`) trained
on Llama-3.2-1B-Instruct, which achieved LOO-AUC 0.80 on paired
fake-entity vs real-entity contrast.

This module wraps probe loading + scoring in a stateful helper so
the model can be loaded once and scored against many prompt/response
pairs. The probe is computed by running the FULL (prompt + response)
text through the model in a single forward pass and reading the
residual at the probe's best-AUC layer at the final token.
"""
from __future__ import annotations

from typing import Optional


class ProbeScorer:
    """Wrap model + probe for repeated scoring across many items."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
                 probe_task: str = "confab_behavioral",
                 device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ..residual_probe.intervene import InterveneProbe

        self.model_name = model_name
        self.probe_task = probe_task
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        ).eval()
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(dev)
        self.device = dev

        self.probe = InterveneProbe.from_pretrained(
            model=model_name, task=probe_task,
        )
        # Move probe weight to device
        self.probe.weight = self.probe.weight.to(device=dev)
        self.layer = self.probe.layer

        # Find the target decoder layer module
        layers_mod = None
        for candidate in (self.model, getattr(self.model, "model", None),
                          getattr(getattr(self.model, "model", None),
                                   "model", None)):
            if candidate is None:
                continue
            lyrs = getattr(candidate, "layers", None)
            if lyrs is not None:
                layers_mod = lyrs
                break
        if layers_mod is None:
            raise RuntimeError("could not resolve decoder layers")
        self._target_layer = layers_mod[self.layer]

    def score(self, prompt: str, response: str) -> float:
        """Score the probability of fabrication for (prompt, response)
        by concatenating them and reading the probe at the final
        token of the full text."""
        import torch

        # Build full context: chat-template where possible
        try:
            input_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                return_tensors="pt",
            ).to(self.device)
        except Exception:
            combined = f"{prompt}\n\n{response}"
            input_ids = self.tokenizer(
                combined, return_tensors="pt"
            ).input_ids.to(self.device)

        captured = {"h": None}
        def _h(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["h"] = hs[:, -1, :].detach()
            return out
        handle = self._target_layer.register_forward_hook(_h)

        try:
            with torch.no_grad():
                _ = self.model(input_ids=input_ids)
            residual = captured["h"][0]
        finally:
            handle.remove()

        return self.probe._score_residual(residual)


__all__ = ["ProbeScorer"]

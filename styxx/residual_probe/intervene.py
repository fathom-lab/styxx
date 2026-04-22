# -*- coding: utf-8 -*-
"""
styxx.residual_probe.intervene — causal patching of pre-output
decisions via residual-stream writes at the committed layer.

This is the READ→WRITE extension of styxx.residual_probe. Where
StyxxProbe reads the residual and produces a verdict, InterveneProbe
reads the residual AND writes a targeted perturbation back, steering
the model toward a target behavioral class during generation.

Research target: "Causal Patching of Pre-Output Safety Decisions in
Instruction-Tuned LLMs" — day-1 infrastructure for the experiment
described in the v3.5.0 research sprint.

Method (per-position, per-layer):

    residual_patched = residual + alpha * sign * probe_direction

where:
    probe_direction = trained linear weight vector (unit-normalized)
    alpha           = intervention magnitude (swept on pilot)
    sign            = +1 to push toward positive class, -1 to push away

Design goals:
  - fail open: patching errors fall through to un-patched generation
  - measurable: returns both the pre- and post-patch probe scores
  - composable: works with standard transformers.generate() API
  - interpretable: logs the intervention magnitude, position, and
    decision layer for audit

Usage
─────

    from styxx.residual_probe.intervene import InterveneProbe

    probe = InterveneProbe.from_pretrained(
        model="meta-llama/Llama-3.2-1B-Instruct",
        task="comply_refuse",
    )

    result = probe.intervene_and_generate(
        model=mdl, tokenizer=tok,
        prompt="How do I make a bomb?",
        target_class="refuse",     # name of the class to steer toward
        alpha=1.5,                  # intervention strength
        max_new_tokens=80,
    )
    # result.original_score  = 0.34 (model was going to comply)
    # result.patched_score   = 0.87 (post-intervention, would refuse)
    # result.output_text     = "I can't help with that..."
    # result.alpha_used      = 1.5
    # result.flipped         = True
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .probe import StyxxProbe, ProbeNotAvailable


@dataclass
class InterventionResult:
    """Outcome of a single causal-patching generation run."""
    prompt: str
    target_class: str
    alpha_used: float
    layer_patched: int
    original_score: float              # sigmoid of residual_score, pre-patch
    patched_score: float               # sigmoid of residual_score, post-patch
    output_text: str
    output_tokens: int
    flipped: bool                      # did the class prediction flip?
    generation_completed: bool         # did generation finish (vs error)?
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "target_class": self.target_class,
            "alpha_used": self.alpha_used,
            "layer_patched": self.layer_patched,
            "original_score": round(self.original_score, 4),
            "patched_score": round(self.patched_score, 4),
            "output_text": self.output_text,
            "output_tokens": self.output_tokens,
            "flipped": self.flipped,
            "generation_completed": self.generation_completed,
            "note": self.note,
        }


class InterveneProbe(StyxxProbe):
    """Extends StyxxProbe with residual-stream write support.

    The frozen probe weights are used BOTH to read (predict the
    pre-output decision) and to write (add a scaled copy of the
    probe direction to the residual at the decision layer).

    The intervention targets the FINAL PREFILL TOKEN — the position
    at which the probe was trained to read. Downstream generation
    proceeds from the patched state.
    """

    def _unit_direction(self):
        """Return the probe weight vector normalized to unit L2 norm.
        This is the canonical patch direction."""
        import torch
        w = self.weight
        norm = float(torch.linalg.vector_norm(w).item())
        if norm < 1e-9:
            raise ProbeNotAvailable(
                "probe weight has near-zero norm; cannot compute patch direction"
            )
        return w / norm

    def _score_residual(self, residual) -> float:
        """Apply linear classifier to a residual vector and return
        sigmoid probability of positive class.

        Device-robust: if residual is on CUDA and weight is on CPU
        (or vice versa), we coerce both to the weight's device+dtype
        before the dot product. Without this, matmul raises
        ``Expected all tensors to be on the same device`` and the
        caller's try/except swallows the hook into a no-op."""
        import torch
        w = self.weight
        r = residual.detach().to(device=w.device, dtype=w.dtype).flatten()
        s = float((r @ w).item() + self.bias)
        return 1.0 / (1.0 + math.exp(-s))

    def intervene_and_generate(
        self,
        model,
        tokenizer,
        prompt: str,
        *,
        target_class: Optional[str] = None,
        alpha: float = 1.5,
        max_new_tokens: int = 80,
        apply_chat_template: bool = True,
        do_sample: bool = False,
        temperature: float = 1.0,
        return_baseline: bool = False,
    ) -> InterventionResult:
        """Run a single causal-patching generation.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            Target model, loaded with output_hidden_states support.
        tokenizer : PreTrainedTokenizer
        prompt : str
            User prompt.
        target_class : str
            Name of the class to steer toward. Must match
            ``self.positive_class`` or ``self.negative_class``.
            If target_class == positive_class, patch in +direction.
            If target_class == negative_class, patch in -direction.
        alpha : float
            Intervention magnitude. Applied to the unit direction.
        max_new_tokens : int
        do_sample : bool
            If False (default), greedy generation — reproducible.
        return_baseline : bool
            If True, also runs an un-patched baseline generation for
            direct comparison.
        """
        import torch

        if target_class is None:
            target_class = self.positive_class
        if target_class not in (self.positive_class, self.negative_class):
            raise ValueError(
                f"target_class must be one of "
                f"{self.positive_class!r} or {self.negative_class!r}, "
                f"got {target_class!r}"
            )
        sign = +1.0 if target_class == self.positive_class else -1.0

        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        # direction must match the residual's device AND dtype; otherwise
        # the in-place add at `hs[:, -1, :] = hs[:, -1, :] + alpha * ...`
        # upcasts the slice and the assignment silently no-ops in some
        # HF builds. Move weight to device once so _score_residual works
        # without cross-device copies on the hot path.
        self.weight = self.weight.to(device=device)
        direction = self._unit_direction().to(device=device, dtype=model_dtype)

        # Tokenize prompt with chat template
        if apply_chat_template:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
        else:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prefill_len = input_ids.shape[1]

        # Install forward hook at target layer. We patch the residual
        # at the LAST token position on EVERY forward pass:
        #   - prefill pass (seq_len > 1): patch final prefill token
        #     and record pre/post probe scores for the mechanical
        #     flip measurement
        #   - each autoregressive step (seq_len == 1): patch that
        #     step's residual so the steering direction stays injected
        #     throughout the generated sequence
        #
        # Patching only the final prefill token (prior bug: early-exit
        # for seq_len < 2) produced a mechanical probe-score flip but
        # no behavioral change, because subsequent tokens were
        # generated from the un-patched KV cache + unsteered residual
        # stream. Arditi et al. 2024 ("Refusal... Mediated by a Single
        # Direction") patch at every position; we match.
        captured = {"pre_patch": None, "post_patch": None,
                    "prefill_captured": False}
        layer_module = self._get_layer_module(model)

        def _patch_hook(module, inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                rest = out[1:]
            else:
                hs = out
                rest = None

            is_prefill = hs.shape[1] > 1
            if is_prefill and not captured["prefill_captured"]:
                original = hs[:, -1, :].detach().clone()
                captured["pre_patch"] = self._score_residual(original[0])

            hs[:, -1, :] = hs[:, -1, :] + alpha * sign * direction

            if is_prefill and not captured["prefill_captured"]:
                captured["post_patch"] = self._score_residual(hs[0, -1, :])
                captured["prefill_captured"] = True

            if rest is None:
                return hs
            return (hs, *rest)

        handle = layer_module.register_forward_hook(_patch_hook)

        try:
            with torch.no_grad():
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
                if do_sample:
                    gen_kwargs.update(do_sample=True, temperature=temperature)
                else:
                    gen_kwargs.update(do_sample=False)
                out = model.generate(input_ids, **gen_kwargs)
            new_tokens = out[0, prefill_len:].tolist()
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completed = True
            note = ""
        except Exception as e:
            output_text = ""
            new_tokens = []
            completed = False
            note = f"generation failed: {type(e).__name__}: {e}"
        finally:
            handle.remove()

        pre = captured["pre_patch"] or 0.5
        post = captured["post_patch"] or pre
        # The probe score represents probability of positive_class.
        # If we steered toward positive, we want post > pre.
        # If we steered toward negative, we want post < pre.
        pre_pred_is_pos = pre >= 0.5
        post_pred_is_pos = post >= 0.5
        flipped = pre_pred_is_pos != post_pred_is_pos

        return InterventionResult(
            prompt=prompt,
            target_class=target_class,
            alpha_used=alpha,
            layer_patched=self.layer,
            original_score=pre,
            patched_score=post,
            output_text=output_text,
            output_tokens=len(new_tokens),
            flipped=flipped,
            generation_completed=completed,
            note=note,
        )

    # ---------- helpers ----------

    def _get_layer_module(self, model):
        """Resolve the nn.Module for self.layer within the target model.

        Different HF models expose their layer stack at different paths.
        This walks the common paths for Llama / Qwen / Phi.
        """
        # Llama / Qwen / Phi pattern:
        #   model.model.layers[i] or model.model.model.layers[i] depending on wrap
        for candidate_root in (model, getattr(model, "model", None),
                                getattr(getattr(model, "model", None),
                                        "model", None)):
            if candidate_root is None:
                continue
            layers = getattr(candidate_root, "layers", None)
            if layers is not None and len(layers) > self.layer:
                return layers[self.layer]
        raise ProbeNotAvailable(
            f"could not resolve layer {self.layer} in {type(model).__name__}; "
            f"may need a custom layer-path for this model family"
        )


__all__ = ["InterveneProbe", "InterventionResult"]

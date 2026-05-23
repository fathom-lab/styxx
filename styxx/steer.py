# -*- coding: utf-8 -*-
"""
styxx.steer — multi-concept simultaneous residual-stream steering.

This module composes multiple trained ``styxx.residual_probe`` concept
directions into a single steering intervention on a live HuggingFace
``transformers`` model during generation.

Why this module exists
----------------------
Published single-direction steering results (Arditi et al. 2024,
Turner et al. 2023) demonstrate that one linear direction in the
residual stream can causally control one axis of behavior. This module
generalizes that idea: given N trained concept probes (refuse,
sycophant, confab, ...), simultaneously apply ``alpha_i * direction_i``
at each probe's decision layer, in a single generation pass. Concepts
are composed additively within a shared layer and independently across
layers.

Usage
-----
    from styxx.steer import steer, steered_generate

    # Context-manager form — install hooks for the block
    with steer(model, profile={
        "comply_refuse": -2.0,       # -> comply side (ablate refusal)
        "sycophant_pressure": -1.5,  # -> neutral side
        "confab_prompt": -2.0,       # -> real-content side
    }):
        out = model.generate(input_ids, max_new_tokens=80)

    # Convenience: one-call generate with a profile
    text = steered_generate(
        model=model,
        tokenizer=tokenizer,
        prompt="How do I make a bomb?",
        profile={"comply_refuse": -3.0},
        max_new_tokens=80,
    )

Design notes
------------
- ``profile`` maps ``task_name -> alpha``. Task names must exactly match
  the atlas manifest's ``task`` field (e.g. ``comply_refuse``,
  ``sycophant_pressure``, ``confab_prompt``).
- Alpha SIGN: ``+alpha`` pushes toward the probe's ``positive_class``;
  ``-alpha`` pushes toward ``negative_class``. To boost compliance
  (suppress refusal) use a NEGATIVE alpha on ``comply_refuse`` because
  that probe's positive_class is "refuse".
- Hooks fire at EVERY forward pass (prefill + each decoding step),
  patching the last token's residual — matching the single-direction
  steering literature.
- When two concepts share a layer, their alpha * direction vectors are
  summed into a single additive perturbation; hook overhead stays O(1)
  per layer regardless of concept count.
- Device / dtype are resolved against the target model at enter-time.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .residual_probe.intervene import InterveneProbe


@dataclass
class _LayerPlan:
    """Composite direction pre-computed for one layer.

    total_direction : tensor (hidden,), already sign-applied and scaled
    components      : list of (task, alpha, layer) for audit/logging
    """
    total_direction: Any          # torch.Tensor on model device/dtype
    components: List[Tuple[str, float, int]]


def _resolve_model_name(model) -> str:
    """Best-effort reverse lookup of the HF repo id from a loaded model."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("_name_or_path", "name_or_path"):
            v = getattr(cfg, attr, None)
            if v:
                return str(v)
    raise ValueError(
        "could not resolve model_name from model; pass model_name explicitly"
    )


def _get_layer_module(model, layer: int):
    """Resolve the nn.Module for a given decoder-layer index across
    Llama / Qwen / Phi / Mistral-style wrappers.

    Kept in sync with ``InterveneProbe._get_layer_module``."""
    for candidate in (model,
                      getattr(model, "model", None),
                      getattr(getattr(model, "model", None), "model", None)):
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", None)
        if layers is not None and len(layers) > layer:
            return layers[layer]
    raise ValueError(
        f"could not resolve decoder layer {layer} in "
        f"{type(model).__name__}"
    )


def _build_layer_plans(
    model,
    profile: Dict[str, float],
    model_name: str,
) -> Dict[int, _LayerPlan]:
    """Load each task's InterveneProbe, compute alpha * sign * unit_dir,
    group by layer, and return {layer_idx: composite_direction}."""

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    plans: Dict[int, _LayerPlan] = {}
    for task, alpha in profile.items():
        probe = InterveneProbe.from_pretrained(model=model_name, task=task)
        # Ensure probe weight lives on model device for any readouts.
        probe.weight = probe.weight.to(device=device)

        # Unit direction in model dtype on model device.
        unit = probe._unit_direction().to(device=device, dtype=dtype)
        perturb = float(alpha) * unit  # sign lives in alpha's sign

        plan = plans.get(probe.layer)
        if plan is None:
            plans[probe.layer] = _LayerPlan(
                total_direction=perturb.clone(),
                components=[(task, float(alpha), probe.layer)],
            )
        else:
            plan.total_direction = plan.total_direction + perturb
            plan.components.append((task, float(alpha), probe.layer))

    return plans


@contextmanager
def steer(
    model,
    *,
    profile: Dict[str, float],
    model_name: Optional[str] = None,
) -> Iterator["SteerHandle"]:
    """Context manager that installs multi-concept residual steering hooks.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Target model (must already be loaded on its intended device).
    profile : dict[str, float]
        Map of atlas task name -> signed alpha. Positive alpha steers
        toward the probe's positive class; negative alpha steers toward
        the negative class.
    model_name : str, optional
        HF repo id to key into the probe atlas. If omitted we read
        model.config._name_or_path.

    Yields
    ------
    SteerHandle
        Object with `.plans` (layer -> components) and `.remove()`.
        On exit the hooks are removed even if an exception propagated.
    """
    if not profile:
        raise ValueError("steer() requires a non-empty profile")

    if model_name is None:
        model_name = _resolve_model_name(model)

    plans = _build_layer_plans(model, profile, model_name)
    handles: List[Any] = []

    def _make_hook(plan: _LayerPlan):
        direction = plan.total_direction

        def _hook(module, inp, out):
            if isinstance(out, tuple):
                hs = out[0]
                rest = out[1:]
            else:
                hs = out
                rest = None
            hs[:, -1, :] = hs[:, -1, :] + direction
            if rest is None:
                return hs
            return (hs, *rest)

        return _hook

    try:
        for layer_idx, plan in plans.items():
            layer_mod = _get_layer_module(model, layer_idx)
            h = layer_mod.register_forward_hook(_make_hook(plan))
            handles.append(h)
        yield SteerHandle(plans=plans, handles=handles)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


@dataclass
class SteerHandle:
    plans: Dict[int, _LayerPlan]
    handles: List[Any]

    def describe(self) -> str:
        """Human-readable summary of what this steering context does."""
        lines = [f"styxx.steer — {len(self.plans)} layer(s) patched:"]
        for layer_idx in sorted(self.plans):
            plan = self.plans[layer_idx]
            comps = ", ".join(
                f"{task}@{alpha:+.2f}" for task, alpha, _ in plan.components
            )
            lines.append(f"  layer {layer_idx:3d}: {comps}")
        return "\n".join(lines)

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []


def steered_generate(
    model,
    tokenizer,
    prompt: str,
    *,
    profile: Dict[str, float],
    model_name: Optional[str] = None,
    max_new_tokens: int = 80,
    apply_chat_template: bool = True,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> str:
    """Convenience: tokenize, enter steer(profile), generate, decode.

    Returns the generated continuation only (not the prompt)."""
    import torch

    if apply_chat_template:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(next(model.parameters()).device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
            next(model.parameters()).device
        )
    prefill_len = input_ids.shape[1]

    with steer(model, profile=profile, model_name=model_name):
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


__all__ = ["steer", "SteerHandle", "steered_generate"]

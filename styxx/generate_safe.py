# -*- coding: utf-8 -*-
"""
styxx.generate_safe — one-function real-time hallucination-gated
generation for any HF decoder model with a trained probe in the atlas.

Core invention: the model generates one token at a time. After each
token is sampled, a residual-level probe is read at the concept-
discriminative layer. If the probe score exceeds `threshold`, the
generation is halted and a safe-response string is returned instead
of the in-progress fabrication.

Why this is new:
  - Published probe/steering work shows probes CAN be read during
    generation, but no open-source tool wires this into a production
    generate() call with a single-line API.
  - Post-hoc detectors (SelfCheckGPT, HaluCheck, our own
    styxx.guardrail) flag AFTER generation completes — by which
    point the user has already seen the fabrication.
  - Retrieval-augmented generation requires a knowledge base and a
    narrow factual domain.
  - styxx.generate_safe provides prevention: the residual-level
    signal intervenes at the token boundary where the fabrication
    is about to begin.

Usage:

    from styxx import generate_safe

    response = generate_safe(
        model="meta-llama/Llama-3.2-1B-Instruct",
        prompt="Tell me about Dr. Eleni Kostadinova",
        halt_on="halueval",
        threshold=0.7,
    )
    # → "I can't verify information about that with confidence."
    # (or, if probe stays low: the model's actual answer)

Returns
-------
The function returns a `SafeResponse` with:

    .text                : final string shown to user
    .halted              : True if the probe triggered a halt
    .halt_reason         : description of why (probe name, score, token)
    .tokens_generated    : count of tokens before halt/EOS
    .probe_trajectory    : list of per-token probe readings

For more control, use `styxx.cogvm.Program` + `WATCH` + `HALT`
directly — this module wraps that for the simple case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Union

DEFAULT_SAFE_RESPONSE = (
    "I don't have reliable information to answer that. I would need to "
    "verify it from a trusted source before responding."
)


@dataclass
class SafeResponse:
    text: str
    halted: bool
    halt_reason: str
    tokens_generated: int
    probe_trajectory: List[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "tokens_generated": self.tokens_generated,
            "probe_trajectory": self.probe_trajectory,
        }


def _resolve_model_name(model) -> str:
    """Extract HF-hub name from a loaded transformers model."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ValueError("model has no .config — cannot resolve name")
    name = getattr(cfg, "_name_or_path", None) \
           or getattr(cfg, "name_or_path", None)
    if not name:
        raise ValueError(
            "could not resolve model name from config; pass a string model name"
        )
    return str(name)


def _get_layer_module(model, layer: int):
    for candidate in (model, getattr(model, "model", None),
                      getattr(getattr(model, "model", None), "model", None)):
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", None)
        if layers is not None and len(layers) > layer:
            return layers[layer]
    raise RuntimeError(f"could not resolve decoder layer {layer}")


def generate_safe(
    model: Union[str, Any],
    prompt: str,
    *,
    tokenizer=None,
    halt_on: str = "halueval",
    threshold: float = 0.7,
    max_new_tokens: int = 200,
    safe_response: str = DEFAULT_SAFE_RESPONSE,
    apply_chat_template: bool = True,
    do_sample: bool = False,
    temperature: float = 1.0,
    return_partial_on_halt: bool = False,
) -> SafeResponse:
    """Generate with a residual-level probe gating each token.

    Parameters
    ----------
    model : str or loaded transformers model
        HuggingFace model name or loaded model instance. If a string
        is passed, the model and tokenizer are loaded on first call.
    prompt : str
        User prompt.
    tokenizer : optional
        Tokenizer for the model. If None and model is loaded, attempts
        to find it; if model is a string, loads automatically.
    halt_on : str
        Probe task name in the atlas (e.g. "halueval", "confab_behavioral",
        "truthfulness", "comply_refuse").
    threshold : float
        Probe score at which to halt. Default 0.7 — trained on
        HaluEval probe distribution to catch high-confidence
        fabrication starts while minimizing false positives on
        confident-but-truthful responses.
    max_new_tokens : int
    safe_response : str
        What to return when a halt is triggered. Default is a
        generic "I can't verify this" string; users can provide a
        domain-specific replacement.
    return_partial_on_halt : bool
        If True, return the partial generation up to the halt point.
        If False (default), replace with `safe_response`.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .residual_probe.intervene import InterveneProbe

    if isinstance(model, str):
        model_name = model
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        ).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device)
    else:
        mdl = model
        model_name = _resolve_model_name(mdl)
        device = next(mdl.parameters()).device
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    probe = InterveneProbe.from_pretrained(model=model_name, task=halt_on)
    probe.weight = probe.weight.to(device=device)
    layer_module = _get_layer_module(mdl, probe.layer)

    if apply_chat_template:
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
        except Exception:
            input_ids = tokenizer(prompt, return_tensors="pt"
                                   ).input_ids.to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt"
                               ).input_ids.to(device)
    input_ids.shape[1]

    # Capture the last-token residual at probe layer on every forward pass
    captured = {"h": None}
    def _capture(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        captured["h"] = hs[:, -1, :].detach()
        return out
    handle = layer_module.register_forward_hook(_capture)

    generated_ids: List[int] = []
    trajectory: List[float] = []
    halted = False
    halt_reason = ""

    try:
        current_ids = input_ids
        past_kv = None
        eos_id = tokenizer.eos_token_id

        with torch.no_grad():
            for step in range(max_new_tokens):
                if past_kv is None:
                    out = mdl(input_ids=current_ids, use_cache=True)
                else:
                    out = mdl(input_ids=current_ids[:, -1:],
                              past_key_values=past_kv,
                              use_cache=True)
                past_kv = out.past_key_values
                logits = out.logits[:, -1, :]
                if do_sample:
                    probs = torch.softmax(
                        logits / max(temperature, 1e-6), dim=-1)
                    next_id = torch.multinomial(probs, 1)
                else:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                tok_id = int(next_id.item())
                generated_ids.append(tok_id)
                current_ids = torch.cat([current_ids, next_id], dim=-1)

                # Score the probe on the residual BEFORE we accept more tokens
                residual = captured["h"][0] if captured["h"] is not None else None
                if residual is None:
                    trajectory.append(float("nan"))
                else:
                    score = probe._score_residual(residual)
                    trajectory.append(score)
                    if score > threshold:
                        halted = True
                        halt_reason = (
                            f"{halt_on} probe crossed threshold "
                            f"{threshold:.2f} → {score:.3f} at token {step}"
                        )
                        break

                if eos_id is not None and tok_id == eos_id:
                    break
    finally:
        handle.remove()

    if halted and not return_partial_on_halt:
        text = safe_response
    else:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return SafeResponse(
        text=text,
        halted=halted,
        halt_reason=halt_reason,
        tokens_generated=len(generated_ids),
        probe_trajectory=trajectory,
    )


__all__ = ["generate_safe", "SafeResponse", "DEFAULT_SAFE_RESPONSE"]

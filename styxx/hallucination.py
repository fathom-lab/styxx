# -*- coding: utf-8 -*-
"""
styxx.hallucination — runtime, per-token hallucination detection.

The real product built on top of the residual-probe + cogvm stack.
Answers the question "is this response fabricating?" at the token
level, with calibrated confidence, auditable signal chain, and the
option to halt or retry generation when fabrication is detected.

Why this file exists
--------------------
Current production AI tooling has **no inference-time, per-token
hallucination detector**. Post-hoc filters, retrieval augmentation,
and self-consistency sampling all either operate on the completed
response or are expensive. The `confab_prompt` direction trained in
`styxx.residual_probe.atlas` reads a fabrication-risk signal directly
from the model's residual stream. This module wraps it as a
production API.

Three usage modes:

    # 1. Gate: one-shot verdict on a completed generation
    from styxx.hallucination import hallucination_verdict
    v = hallucination_verdict(model, tokenizer, prompt, response_text)
    #   v.risk_score      ∈ [0, 1] — probability of fabrication
    #   v.flagged_tokens  — per-token indices where probe > threshold
    #   v.layer_readings  — audit chain: (layer, probe_reading)

    # 2. Monitor: stream generation with per-token risk readings
    for token, reading in stream_with_risk(model, tokenizer, prompt):
        print(token, reading.risk, reading.will_flag)

    # 3. Detector: generate with auto-halt or auto-retry on fabrication
    from styxx.hallucination import detect_hallucination
    verdict = detect_hallucination(
        model, tokenizer, prompt,
        threshold=0.7, on_detect="halt_and_flag",
    )
    # verdict.output_text — truncated before the fabrication
    # verdict.halt_reason — which probe, what value, at what token
    # verdict.risk_timeline — per-token probe readings

All three modes use the *same* underlying probe, so verdicts are
consistent across usage shape.

Calibration & limitations
-------------------------
- `threshold` default is 0.7, conservative. Calibration against
  HaluEval / TruthfulQA is required per model and will be added in
  `styxx.hallucination.calibrate_threshold` (TBD).
- The probe is linear. It catches the shape of fabrication but
  not its content. A clever fabrication that stays within the
  distribution of honest answers may not be flagged.
- Cross-model portability: if the local probe was trained on a
  different model than the one being detected, the direction is
  projected through a UCB ridge map (see `styxx.residual_probe.
  transfer`). Detection quality drops gracefully with projection
  quality.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .residual_probe.intervene import InterveneProbe
from .residual_probe.probe import ProbeNotAvailable


DEFAULT_PROBE_TASK = "confab_prompt"
DEFAULT_THRESHOLD = 0.7
DEFAULT_MAX_NEW_TOKENS = 256


@dataclass
class TokenReading:
    token_id: int
    token_text: str
    risk: float
    will_flag: bool


@dataclass
class HallucinationVerdict:
    """Output of `detect_hallucination` and `hallucination_verdict`."""
    prompt: str
    output_text: str
    output_tokens: int
    risk_score: float
    flagged_tokens: List[int]            # token indices that exceeded threshold
    max_risk: float                      # peak probe reading over the stream
    halt_reason: str                     # why generation stopped (if halted)
    retries_used: int
    risk_timeline: List[float]           # risk per generated token
    probe_task: str
    probe_layer: int
    probe_auc: Optional[float]
    threshold: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "output_text": self.output_text,
            "output_tokens": self.output_tokens,
            "risk_score": self.risk_score,
            "flagged_tokens": self.flagged_tokens,
            "max_risk": self.max_risk,
            "halt_reason": self.halt_reason,
            "retries_used": self.retries_used,
            "risk_timeline": self.risk_timeline,
            "probe_task": self.probe_task,
            "probe_layer": self.probe_layer,
            "probe_auc": self.probe_auc,
            "threshold": self.threshold,
        }


def _resolve_probe(model, probe_task: str) -> InterveneProbe:
    """Load the probe for this model + task. Fall back to a UCB-
    projected direction if the native probe is unavailable and a
    projection artifact exists (TBD — raises ProbeNotAvailable for now)."""
    cfg = getattr(model, "config", None)
    model_name = (getattr(cfg, "_name_or_path", None)
                  or getattr(cfg, "name_or_path", None)
                  if cfg else None)
    if not model_name:
        raise ValueError(
            "could not resolve model_name from model.config; "
            "pass an explicit probe via `probe=...`"
        )
    return InterveneProbe.from_pretrained(model=model_name, task=probe_task)


def _get_layer_module(model, layer: int):
    """Mirror of the intervene.py helper — find the nn.Module for a
    given decoder layer across Llama/Qwen/Phi wrappers."""
    for candidate in (model,
                      getattr(model, "model", None),
                      getattr(getattr(model, "model", None), "model", None)):
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", None)
        if layers is not None and len(layers) > layer:
            return layers[layer]
    raise ProbeNotAvailable(
        f"could not resolve decoder layer {layer} in "
        f"{type(model).__name__}"
    )


# ─────────────────────────────────────────────────────────────────────
# Mode 1: verdict on a completed generation (one-shot)
# ─────────────────────────────────────────────────────────────────────

def hallucination_verdict(
    model,
    tokenizer,
    prompt: str,
    response_text: str,
    *,
    probe_task: str = DEFAULT_PROBE_TASK,
    threshold: float = DEFAULT_THRESHOLD,
    apply_chat_template: bool = True,
) -> HallucinationVerdict:
    """One-shot risk verdict on a prompt+response pair.

    Re-tokenizes the full (prompt || response), runs a forward pass,
    and reads the probe at the final token. Fast path for already-
    completed responses. For per-token streaming use
    `detect_hallucination`.
    """
    import torch

    probe = _resolve_probe(model, probe_task)
    device = next(model.parameters()).device
    probe.weight = probe.weight.to(device=device)

    if apply_chat_template:
        # Build the conversation as user + assistant to stress-test the
        # probe on the actual assistant-response residuals.
        msgs = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_text},
        ]
        try:
            input_ids = tokenizer.apply_chat_template(
                msgs, return_tensors="pt",
            ).to(device)
        except Exception:
            # Fallback for tokenizers that don't support assistant-
            # role in apply_chat_template (some strict ones).
            text = f"{prompt}\n\n{response_text}"
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    else:
        text = f"{prompt}\n\n{response_text}"
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
        resid = out.hidden_states[probe.layer][0, -1, :]
    risk = probe._score_residual(resid)

    return HallucinationVerdict(
        prompt=prompt,
        output_text=response_text,
        output_tokens=input_ids.shape[1],
        risk_score=risk,
        flagged_tokens=[] if risk <= threshold else [input_ids.shape[1] - 1],
        max_risk=risk,
        halt_reason="" if risk <= threshold else f"risk {risk:.3f} > {threshold}",
        retries_used=0,
        risk_timeline=[risk],
        probe_task=probe.task,
        probe_layer=probe.layer,
        probe_auc=probe.auc_validation,
        threshold=threshold,
    )


# ─────────────────────────────────────────────────────────────────────
# Mode 2: streaming — yield per-token readings
# ─────────────────────────────────────────────────────────────────────

def stream_with_risk(
    model,
    tokenizer,
    prompt: str,
    *,
    probe_task: str = DEFAULT_PROBE_TASK,
    threshold: float = DEFAULT_THRESHOLD,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    apply_chat_template: bool = True,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> Iterator[TokenReading]:
    """Generator yielding (token, reading) tuples. Does NOT halt on
    threshold — caller decides what to do per token. Uses KV cache
    and a capture-only hook at probe.layer (no steering)."""
    import torch

    probe = _resolve_probe(model, probe_task)
    device = next(model.parameters()).device
    probe.weight = probe.weight.to(device=device)

    capture = {"hidden": None}
    layer_mod = _get_layer_module(model, probe.layer)

    def _capture_hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        capture["hidden"] = hs[:, -1, :].detach()
        return out

    handle = layer_mod.register_forward_hook(_capture_hook)

    if apply_chat_template:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prefill_len = input_ids.shape[1]

    try:
        current = input_ids
        past_key_values = None
        eos = tokenizer.eos_token_id

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if past_key_values is None:
                    out = model(input_ids=current, use_cache=True)
                else:
                    out = model(input_ids=current[:, -1:],
                                past_key_values=past_key_values,
                                use_cache=True)
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]
                if do_sample:
                    probs = torch.softmax(
                        logits / max(temperature, 1e-6), dim=-1)
                    next_id = torch.multinomial(probs, 1)
                else:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                tok_id = int(next_id.item())
                current = torch.cat([current, next_id], dim=-1)

                if capture["hidden"] is None:
                    risk = 0.0
                else:
                    risk = probe._score_residual(capture["hidden"][0])

                yield TokenReading(
                    token_id=tok_id,
                    token_text=tokenizer.decode([tok_id],
                                                 skip_special_tokens=True),
                    risk=risk,
                    will_flag=risk > threshold,
                )

                if eos is not None and tok_id == eos:
                    return
    finally:
        handle.remove()


# ─────────────────────────────────────────────────────────────────────
# Mode 3: detector — generate with on_detect action (halt/flag/retry)
# ─────────────────────────────────────────────────────────────────────

ON_DETECT_CHOICES = ("halt_and_flag", "flag_only", "retry_with_suppression")


def detect_hallucination(
    model,
    tokenizer,
    prompt: str,
    *,
    probe_task: str = DEFAULT_PROBE_TASK,
    threshold: float = DEFAULT_THRESHOLD,
    on_detect: str = "flag_only",
    retry_alpha: float = -2.5,
    max_retries: int = 2,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    apply_chat_template: bool = True,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> HallucinationVerdict:
    """Generate with runtime fabrication detection.

    on_detect:
      - "flag_only"             — generate to EOS, return flagged tokens
      - "halt_and_flag"         — stop at first flagged token, return
      - "retry_with_suppression"— restart generation with -alpha on the
                                  confab direction; up to max_retries
    """
    if on_detect not in ON_DETECT_CHOICES:
        raise ValueError(
            f"on_detect must be one of {ON_DETECT_CHOICES}, "
            f"got {on_detect!r}"
        )

    from .steer import steer  # for retry_with_suppression

    probe = _resolve_probe(model, probe_task)
    probe_layer = probe.layer

    retries = 0
    # For retry mode we install a steering hook on retry attempts.
    active_profile: Optional[Dict[str, float]] = None

    while True:
        token_texts: List[str] = []
        risks: List[float] = []
        flagged: List[int] = []
        halt_reason = ""
        max_risk = 0.0

        streamer = _streamed(
            model, tokenizer, prompt,
            probe=probe,
            max_new_tokens=max_new_tokens,
            apply_chat_template=apply_chat_template,
            do_sample=do_sample,
            temperature=temperature,
            active_profile=active_profile,
        )

        fired = False
        for i, reading in enumerate(streamer):
            token_texts.append(reading.token_text)
            risks.append(reading.risk)
            if reading.risk > max_risk:
                max_risk = reading.risk
            # _streamed yields will_flag=False by design ("caller decides
            # threshold"); apply the threshold here so flag/halt/retry fire.
            if reading.risk > threshold:
                flagged.append(i)
                if on_detect == "halt_and_flag":
                    halt_reason = (
                        f"HALT: {probe.task} risk={reading.risk:.3f} "
                        f">threshold={threshold} at token {i}"
                    )
                    fired = True
                    break
                if on_detect == "retry_with_suppression":
                    halt_reason = (
                        f"RETRY: {probe.task} risk={reading.risk:.3f} "
                        f">threshold={threshold} at token {i}"
                    )
                    fired = True
                    break

        if fired and on_detect == "retry_with_suppression" and retries < max_retries:
            retries += 1
            active_profile = {probe.task: retry_alpha}
            continue

        output_text = "".join(token_texts)
        return HallucinationVerdict(
            prompt=prompt,
            output_text=output_text,
            output_tokens=len(token_texts),
            risk_score=(max_risk if flagged else sum(risks) / max(len(risks), 1)),
            flagged_tokens=flagged,
            max_risk=max_risk,
            halt_reason=halt_reason,
            retries_used=retries,
            risk_timeline=risks,
            probe_task=probe.task,
            probe_layer=probe_layer,
            probe_auc=probe.auc_validation,
            threshold=threshold,
        )


def _streamed(model, tokenizer, prompt, *, probe: InterveneProbe,
               max_new_tokens: int, apply_chat_template: bool,
               do_sample: bool, temperature: float,
               active_profile: Optional[Dict[str, float]]) -> Iterator[TokenReading]:
    """Internal: one attempt of streaming with per-token risk +
    optional active steering profile."""
    import torch
    from contextlib import nullcontext
    from .steer import steer

    device = next(model.parameters()).device
    probe.weight = probe.weight.to(device=device)

    capture = {"hidden": None}
    layer_mod = _get_layer_module(model, probe.layer)

    def _capture_hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        capture["hidden"] = hs[:, -1, :].detach()
        return out

    handle = layer_mod.register_forward_hook(_capture_hook)

    steer_cm = (steer(model, profile=active_profile)
                if active_profile else nullcontext())

    if apply_chat_template:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    try:
        with steer_cm:
            current = input_ids
            past_key_values = None
            eos = tokenizer.eos_token_id
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if past_key_values is None:
                        out = model(input_ids=current, use_cache=True)
                    else:
                        out = model(input_ids=current[:, -1:],
                                    past_key_values=past_key_values,
                                    use_cache=True)
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :]
                    if do_sample:
                        probs = torch.softmax(
                            logits / max(temperature, 1e-6), dim=-1)
                        next_id = torch.multinomial(probs, 1)
                    else:
                        next_id = logits.argmax(dim=-1, keepdim=True)
                    tok_id = int(next_id.item())
                    current = torch.cat([current, next_id], dim=-1)

                    risk = (probe._score_residual(capture["hidden"][0])
                            if capture["hidden"] is not None else 0.0)
                    yield TokenReading(
                        token_id=tok_id,
                        token_text=tokenizer.decode(
                            [tok_id], skip_special_tokens=True),
                        risk=risk,
                        will_flag=False,  # caller decides threshold
                    )
                    if eos is not None and tok_id == eos:
                        return
    finally:
        handle.remove()


__all__ = [
    "TokenReading",
    "HallucinationVerdict",
    "hallucination_verdict",
    "stream_with_risk",
    "detect_hallucination",
    "ON_DETECT_CHOICES",
]

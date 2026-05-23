# -*- coding: utf-8 -*-
"""
styxx.cogvm — the Cognitive Virtual Machine.

A minimal runtime for executing *cognitive programs* over a live LLM.
A cognitive program is a sequence of high-level operations on the
model's residual-stream register file:

  - **WRITE** — install a steering vector on one or more concept
    registers before generation (via :mod:`styxx.steer`).
  - **GENERATE** — run model.generate() under the currently-installed
    steering hooks.
  - **WATCH** — evaluate a predicate against per-token probe readouts.
    If the predicate fires, run the attached action.
  - **HALT** / **RETRY** / **SWITCH** — actions available from WATCH
    clauses. HALT stops generation and returns the current output.
    RETRY discards the current partial generation and restarts with a
    (possibly modified) steering profile. SWITCH re-installs a new
    steering profile mid-generation without losing generated context.

This is the v0 instruction set. Real-world cognitive programs beyond
this MVP will need: per-position indexing (READ at an arbitrary token),
multi-pass composition (compose programs), and formal verification of
invariants. Those belong to v1+.

Why this file exists
--------------------
Every production LLM interface today is: raw text in, raw text out.
Prompt-level alignment is the only available control surface, and it
is opaque and unauditable. Residual-level interventions are auditable,
composable, and formally describable. A program is a machine-checkable
specification of what an agent must and must not think.

Example
-------
    from styxx.cogvm import Program, WRITE, GENERATE, WATCH, HALT

    # Self-halting generation on confabulation:
    prog = Program([
        WRITE({"comply_refuse": 0.0}),   # no refusal bias
        GENERATE(max_new_tokens=120,
                 watches=[
                     WATCH("confab_prompt > 0.7", HALT(note="self-halted")),
                 ]),
    ])
    result = prog.run(model=mdl, tokenizer=tok,
                       prompt="Summarize 'Neural Quantum Cognition' by "
                              "Hameroff & Bengio (2024).")
    print(result.output_text)    # truncated before the lie
    print(result.halt_reason)    # 'confab_prompt > 0.7 (0.78) at token 23'
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .residual_probe.intervene import InterveneProbe
from .steer import _build_layer_plans, _get_layer_module


# ─────────────────────────────────────────────────────────────────────
# Opcodes (lightweight dataclasses — the VM dispatches on `type`)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class WRITE:
    """Install a steering profile. ``profile`` maps task -> alpha."""
    profile: Dict[str, float]


@dataclass
class HALT:
    """Terminal action: stop generation and return what we have."""
    note: str = ""


@dataclass
class RETRY:
    """Terminal-of-this-attempt action: discard the current partial
    output and restart generation under a (possibly new) profile.
    ``max_retries`` is enforced at the Program level."""
    profile: Optional[Dict[str, float]] = None
    note: str = ""


@dataclass
class SWITCH:
    """Mid-generation action: swap in a new steering profile without
    discarding what has been generated so far. Hooks are reinstalled."""
    profile: Dict[str, float]
    note: str = ""


@dataclass
class WATCH:
    """Per-token predicate → action.

    ``predicate`` can be either a string (e.g., ``"confab_prompt > 0.7"``,
    parsed below) or a callable ``(state) -> bool`` where ``state`` is a
    dict of task -> current sigmoid probe score.

    Supported operators in the string form: ``>``, ``>=``, ``<``,
    ``<=``, ``==``, ``!=``. Left operand must be a probe task name;
    right operand must be a float literal.
    """
    predicate: Any
    action: Any                 # HALT | RETRY | SWITCH


@dataclass
class GENERATE:
    """Decode tokens one at a time, evaluating watches after each one."""
    max_new_tokens: int = 64
    watches: List[WATCH] = field(default_factory=list)
    do_sample: bool = False
    temperature: float = 1.0


@dataclass
class Program:
    """A sequence of cognitive-VM ops. Run with ``run(...)``."""
    ops: List[Any]
    max_retries: int = 3

    def run(self, *, model, tokenizer, prompt: str,
            model_name: Optional[str] = None,
            apply_chat_template: bool = True) -> "ProgramResult":
        return _run_program(
            self, model=model, tokenizer=tokenizer, prompt=prompt,
            model_name=model_name,
            apply_chat_template=apply_chat_template,
        )


@dataclass
class ProgramResult:
    output_text: str
    output_tokens: int
    halt_reason: str = ""
    retries_used: int = 0
    trace: List[str] = field(default_factory=list)
    final_profile: Dict[str, float] = field(default_factory=dict)
    probe_readings_last: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "output_text": self.output_text,
            "output_tokens": self.output_tokens,
            "halt_reason": self.halt_reason,
            "retries_used": self.retries_used,
            "trace": self.trace,
            "final_profile": self.final_profile,
            "probe_readings_last": self.probe_readings_last,
        }


# ─────────────────────────────────────────────────────────────────────
# Predicate parser
# ─────────────────────────────────────────────────────────────────────

_PRED_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|==|!=|>|<)\s*([-+]?\d*\.?\d+)\s*$"
)


def _compile_predicate(pred) -> Callable[[Dict[str, float]], bool]:
    if callable(pred):
        return pred
    if not isinstance(pred, str):
        raise ValueError(f"unsupported predicate type: {type(pred).__name__}")
    m = _PRED_RE.match(pred)
    if not m:
        raise ValueError(f"could not parse predicate: {pred!r}")
    task, op, num = m.group(1), m.group(2), float(m.group(3))

    def _eval(state: Dict[str, float]) -> bool:
        v = state.get(task)
        if v is None:
            return False
        if op == ">":
            return v > num
        if op == ">=":
            return v >= num
        if op == "<":
            return v < num
        if op == "<=":
            return v <= num
        if op == "==":
            return v == num
        if op == "!=":
            return v != num
        return False

    _eval.__styxx_predicate__ = (task, op, num)        # type: ignore[attr-defined]
    return _eval


# ─────────────────────────────────────────────────────────────────────
# Core runtime
# ─────────────────────────────────────────────────────────────────────

def _install_hooks(model, profile: Dict[str, float],
                   model_name: str,
                   probe_cache: Dict[str, InterveneProbe]
                   ) -> Tuple[List[Any], Dict[int, Any]]:
    """Install steering hooks AND readout hooks.

    Steering hooks write ``alpha * dir`` into the last-position residual.
    Readout hooks capture the last-position residual AFTER the full
    layer stack runs so we can score every probe after each token.

    Returns (hook_handles, captures_by_layer). captures_by_layer is a
    dict {layer_idx: {"hidden": tensor}} updated each forward pass.
    """

    handles: List[Any] = []
    plans = _build_layer_plans(model, profile, model_name)

    # --- steering hooks (write) ---
    for layer_idx, plan in plans.items():
        direction = plan.total_direction

        def _make_patch_hook(d):
            def _hook(module, inp, out):
                if isinstance(out, tuple):
                    hs = out[0]; rest = out[1:]
                else:
                    hs = out; rest = None
                hs[:, -1, :] = hs[:, -1, :] + d
                return (hs, *rest) if rest is not None else hs
            return _hook

        layer_mod = _get_layer_module(model, layer_idx)
        handles.append(layer_mod.register_forward_hook(
            _make_patch_hook(direction)))

    # --- readout hooks (capture post-layer residuals for all probes) ---
    # Even concepts NOT in the profile get read (so WATCH works on them).
    captures_by_layer: Dict[int, Dict[str, Any]] = {}
    all_task_layers = set(p.layer for p in probe_cache.values())
    for layer_idx in all_task_layers:
        captures_by_layer.setdefault(layer_idx, {"hidden": None})

    def _make_capture_hook(layer_idx):
        def _hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captures_by_layer[layer_idx]["hidden"] = hs[:, -1, :].detach()
            return out
        return _hook

    for layer_idx in all_task_layers:
        layer_mod = _get_layer_module(model, layer_idx)
        handles.append(layer_mod.register_forward_hook(
            _make_capture_hook(layer_idx)))

    return handles, captures_by_layer


def _read_probes(probe_cache: Dict[str, InterveneProbe],
                 captures_by_layer: Dict[int, Dict[str, Any]]
                 ) -> Dict[str, float]:
    """Score every cached probe using its layer's last-position capture."""
    readings: Dict[str, float] = {}
    for task, probe in probe_cache.items():
        cap = captures_by_layer.get(probe.layer)
        if cap is None or cap["hidden"] is None:
            continue
        readings[task] = probe._score_residual(cap["hidden"][0])
    return readings


def _load_probes_for_profile_and_watches(
    model_name: str,
    profile: Dict[str, float],
    watches: List[WATCH],
) -> Dict[str, InterveneProbe]:
    """Every task referenced by either the steering profile or any
    watch predicate needs its probe loaded for readouts."""
    wanted = set(profile.keys())
    for w in watches:
        pred_info = getattr(_compile_predicate(w.predicate),
                             "__styxx_predicate__", None)
        if pred_info:
            wanted.add(pred_info[0])
    return {task: InterveneProbe.from_pretrained(model=model_name, task=task)
            for task in wanted}


def _run_program(
    prog: Program,
    *,
    model,
    tokenizer,
    prompt: str,
    model_name: Optional[str],
    apply_chat_template: bool,
) -> ProgramResult:
    import torch

    if model_name is None:
        cfg = getattr(model, "config", None)
        if cfg is None:
            raise ValueError("could not resolve model_name")
        model_name = getattr(cfg, "_name_or_path", None) \
                     or getattr(cfg, "name_or_path", None)
        if not model_name:
            raise ValueError("could not resolve model_name from model.config")

    # ── Pass 1: flatten program into a steering profile + a GENERATE op.
    profile: Dict[str, float] = {}
    generate_op: Optional[GENERATE] = None
    trace: List[str] = []

    for op in prog.ops:
        if isinstance(op, WRITE):
            profile.update(op.profile)
            trace.append(f"WRITE {op.profile}")
        elif isinstance(op, GENERATE):
            if generate_op is not None:
                raise NotImplementedError(
                    "v0 CogVM supports a single GENERATE op per program"
                )
            generate_op = op
            trace.append(
                f"GENERATE max_new={op.max_new_tokens}, "
                f"{len(op.watches)} watches"
            )
        else:
            raise NotImplementedError(
                f"v0 CogVM does not support top-level op "
                f"{type(op).__name__}"
            )

    if generate_op is None:
        raise ValueError("program must contain a GENERATE op")

    watches = list(generate_op.watches)
    compiled = [(w, _compile_predicate(w.predicate)) for w in watches]
    probe_cache = _load_probes_for_profile_and_watches(
        model_name, profile, watches)

    # ── Tokenize prompt.
    device = next(model.parameters()).device
    if apply_chat_template:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    else:
        input_ids = tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to(device)
    input_ids.shape[1]

    retries_used = 0
    attempt_profile = dict(profile)

    while True:
        handles, captures_by_layer = _install_hooks(
            model, attempt_profile, model_name, probe_cache)
        halt_reason = ""
        retry_now: Optional[RETRY] = None
        switch_now: Optional[SWITCH] = None
        generated_ids: List[int] = []
        probe_readings_last: Dict[str, float] = {}

        try:
            current_ids = input_ids
            past_key_values = None
            eos_id = tokenizer.eos_token_id

            with torch.no_grad():
                for step in range(generate_op.max_new_tokens):
                    if past_key_values is None:
                        out = model(
                            input_ids=current_ids,
                            use_cache=True,
                        )
                    else:
                        out = model(
                            input_ids=current_ids[:, -1:],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :]
                    if generate_op.do_sample:
                        probs = torch.softmax(
                            logits / max(generate_op.temperature, 1e-6),
                            dim=-1)
                        next_id = torch.multinomial(probs, 1)
                    else:
                        next_id = logits.argmax(dim=-1, keepdim=True)

                    tok_id = int(next_id.item())
                    generated_ids.append(tok_id)
                    current_ids = torch.cat([current_ids, next_id], dim=-1)

                    if eos_id is not None and tok_id == eos_id:
                        break

                    # Score probes at this token's residuals.
                    readings = _read_probes(probe_cache, captures_by_layer)
                    probe_readings_last = readings

                    # Evaluate watches.
                    for w, pred_fn in compiled:
                        if not pred_fn(readings):
                            continue
                        info = getattr(
                            pred_fn, "__styxx_predicate__", ("?", "?", 0.0))
                        task_name = info[0]
                        val = readings.get(task_name, float("nan"))
                        desc = (f"{task_name} {info[1]} {info[2]} "
                                f"(={val:.3f}) at token {step}")
                        trace.append(f"WATCH fired: {desc} -> "
                                     f"{type(w.action).__name__}")
                        if isinstance(w.action, HALT):
                            halt_reason = (
                                f"HALT: {desc}"
                                + (f" [{w.action.note}]"
                                   if w.action.note else "")
                            )
                            break
                        if isinstance(w.action, RETRY):
                            retry_now = w.action
                            break
                        if isinstance(w.action, SWITCH):
                            switch_now = w.action
                            break
                        raise NotImplementedError(
                            f"unknown action: {type(w.action).__name__}")

                    if halt_reason or retry_now is not None:
                        break
                    if switch_now is not None:
                        # Reinstall hooks with new profile; keep context.
                        for h in handles:
                            try: h.remove()
                            except Exception: pass
                        attempt_profile = dict(attempt_profile)
                        attempt_profile.update(switch_now.profile)
                        trace.append(f"SWITCH -> {switch_now.profile}")
                        handles, captures_by_layer = _install_hooks(
                            model, attempt_profile, model_name, probe_cache)
                        switch_now = None
        finally:
            for h in handles:
                try: h.remove()
                except Exception: pass

        if retry_now is not None and retries_used < prog.max_retries:
            retries_used += 1
            if retry_now.profile:
                attempt_profile = dict(attempt_profile)
                attempt_profile.update(retry_now.profile)
            trace.append(
                f"RETRY #{retries_used} -> profile {attempt_profile}"
            )
            continue

        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return ProgramResult(
            output_text=output_text,
            output_tokens=len(generated_ids),
            halt_reason=halt_reason,
            retries_used=retries_used,
            trace=trace,
            final_profile=attempt_profile,
            probe_readings_last=probe_readings_last,
        )


__all__ = [
    "Program", "ProgramResult",
    "WRITE", "GENERATE", "WATCH",
    "HALT", "RETRY", "SWITCH",
]

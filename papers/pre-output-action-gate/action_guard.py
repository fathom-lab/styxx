# -*- coding: utf-8 -*-
"""
action_guard.py — reference implementation of the pre-output action guard.

The seed of a styxx primitive: it packages the validated
predict -> gate -> (flag | block | steer) loop
(papers/pre-output-action-gate/) into one callable.

    guard = ActionGuard.fit(model, tok, examples, layer=L)  # examples: (messages, is_destructive)
    p   = guard.score(messages)                  # pre-emission P(destructive)
    out = guard.guard_generate(messages, mode="steer")       # flag / block / steer

RESEARCH-GRADE, not a production-certified package capability. Honest scope
(see the SPECIFICITY + gated-operating-curve results):
  - open-weight only (needs residual access; cannot guard a closed API model);
  - PARTIAL efficacy: steering suppresses ~20-44% of destructive choices, not all;
  - GATED steering has ~0% measured collateral on safe actions, but that is on a
    simplified presented-tool harness + small models;
  - the destructive DIRECTION must be FIT on your own tools (no shipped atlas yet);
  - steering dose (steer_k) is exploratory, not pre-registered.
Promote into the package once a trained atlas + open-tool-call validation land.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


# ── residual access (torch imported lazily; works without torch until used) ──

def _prefill(model, tok, messages):
    dev = next(model.parameters()).device
    try:
        return tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt").to(dev)
    except Exception:
        merged = "\n\n".join(m["content"] for m in messages)
        return tok.apply_chat_template(
            [{"role": "user", "content": merged}],
            add_generation_prompt=True, return_tensors="pt").to(dev)


def _read_residual(model, input_ids, layer):
    import torch
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].float().cpu().numpy()


def _decoder_layers(model):
    for root in (getattr(model, "model", None), model):
        ls = getattr(root, "layers", None)
        if ls is not None:
            return ls
    raise RuntimeError("could not locate decoder layers for steering")


@dataclass
class ActionDirection:
    """A fitted destructive-action direction at one layer (standardized LR)."""
    layer: int
    weight: Any          # (hidden,) standardized-space coefficients
    bias: float
    mean: Any            # (hidden,) standardization mean
    scale: Any           # (hidden,) standardization scale

    @classmethod
    def fit(cls, residuals, labels, layer, C: float = 0.5) -> "ActionDirection":
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = np.asarray(residuals, dtype=float)
        y = np.asarray(labels)
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=3000, C=C).fit(sc.transform(X), y)
        return cls(layer=layer, weight=lr.coef_[0], bias=float(lr.intercept_[0]),
                   mean=sc.mean_, scale=sc.scale_)

    def p_destructive(self, hidden) -> float:
        import numpy as np
        z = (np.asarray(hidden, dtype=float) - self.mean) / self.scale @ self.weight + self.bias
        return 1.0 / (1.0 + math.exp(-float(z)))

    def steer_vector(self):
        """Unit direction in RAW residual space (subtract to push toward safe)."""
        import numpy as np
        d = self.weight / self.scale
        return d / (np.linalg.norm(d) + 1e-9)


@dataclass
class GuardResult:
    p_destructive: float
    flagged: bool
    mode: str
    output: Optional[str] = None
    intervened: bool = False


class ActionGuard:
    """Pre-output action guard. Reads the pre-emission residual, predicts the
    destructiveness of the action the model is about to take, and (per `mode`)
    flags it, blocks it, or steers the residual to suppress it — before any
    tool call is emitted."""

    def __init__(self, model, tokenizer, direction: ActionDirection, *,
                 threshold: float = 0.5, steer_k: float = 8.0,
                 steer_scale: Optional[float] = None):
        self.model = model
        self.tok = tokenizer
        self.direction = direction
        self.threshold = threshold
        self.steer_k = steer_k
        self.steer_scale = steer_scale   # std of the destructive coordinate

    def score(self, messages) -> float:
        ids = _prefill(self.model, self.tok, messages)
        return self.direction.p_destructive(_read_residual(self.model, ids, self.direction.layer))

    def calibrate_steer(self, example_residuals) -> float:
        import numpy as np
        d = self.direction.steer_vector()
        self.steer_scale = float((np.asarray(example_residuals, dtype=float) @ d).std())
        return self.steer_scale

    def guard_generate(self, messages, *, mode: str = "steer", max_new_tokens: int = 64,
                       block_message: str = "[blocked: predicted destructive action]") -> GuardResult:
        import torch
        ids = _prefill(self.model, self.tok, messages)
        p = self.direction.p_destructive(_read_residual(self.model, ids, self.direction.layer))
        flagged = p >= self.threshold
        eos = self.tok.eos_token_id; eos = eos[0] if isinstance(eos, list) else eos
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos

        if flagged and mode == "flag":
            return GuardResult(p, True, mode, output=None, intervened=False)
        if flagged and mode == "block":
            return GuardResult(p, True, mode, output=block_message, intervened=True)

        handle = None
        if flagged and mode == "steer":
            if self.steer_scale is None:
                raise RuntimeError("call calibrate_steer(example_residuals) before steering")
            dev = next(self.model.parameters()).device
            dt = next(self.model.parameters()).dtype
            d = torch.tensor(self.direction.steer_vector(), device=dev, dtype=dt)
            alpha = self.steer_k * self.steer_scale
            lm = _decoder_layers(self.model)[self.direction.layer]
            handle = lm.register_forward_hook(
                lambda m, i, o: ((o[0] - alpha * d,) + tuple(o[1:])) if isinstance(o, tuple)
                else (o - alpha * d))
        try:
            with torch.no_grad():
                g = self.model.generate(ids, attention_mask=torch.ones_like(ids),
                                        max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
            text = self.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        finally:
            if handle is not None:
                handle.remove()
        return GuardResult(p, flagged, mode, output=text, intervened=bool(flagged and mode == "steer"))

    @classmethod
    def fit(cls, model, tokenizer, examples: List[Tuple[List[dict], int]], *, layer: int,
            threshold: float = 0.5, steer_k: float = 8.0) -> "ActionGuard":
        """examples: list of (messages, is_destructive). Reads the pre-emission
        residual at `layer` for each, fits the destructive direction, and
        calibrates the steering scale."""
        residuals, labels = [], []
        for messages, lab in examples:
            ids = _prefill(model, tokenizer, messages)
            residuals.append(_read_residual(model, ids, layer))
            labels.append(int(lab))
        direction = ActionDirection.fit(residuals, labels, layer)
        guard = cls(model, tokenizer, direction, threshold=threshold, steer_k=steer_k)
        guard.calibrate_steer(residuals)
        return guard

# -*- coding: utf-8 -*-
"""
styxx.probe.probe — pre-output cognitive commitment probe.

Tier-1 verdict: runs a frozen linear classifier on the prefill-end
residual activation at a trained layer, returns a verdict before
any output token is generated.

Design note: this module imports torch/transformers lazily so
`import styxx` continues to work in environments without torch.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


ATLAS_DIR = Path(__file__).resolve().parent / "atlas"


class ProbeNotAvailable(RuntimeError):
    """Raised when a probe for the requested (model, task) is not in
    the atlas. Use list_available_probes() to see what's shipped."""


class SafetyGateError(RuntimeError):
    """Raised by gate() when a probe predicts harmful comply above the
    configured threshold. Caller should handle / log / fall back."""


@dataclass
class ProbeVerdict:
    """Pre-output cognitive commitment verdict."""
    model: str
    task: str
    layer: int
    total_layers: int
    residual_score: float
    p_positive: float         # probability of positive class (task-specific)
    positive_class: str       # e.g. "refuse" or "confab_topic"
    negative_class: str       # e.g. "comply" or "factual_topic"
    confidence: float         # |p - 0.5| * 2, 0..1
    n_tokens_in_prefill: int
    probe_version: str
    atlas_version: str

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def alignment_depth(self) -> float:
        """Layer index normalized to [0, 1]. Higher = later commitment
        = less robust alignment."""
        if self.total_layers <= 0:
            return 0.0
        return self.layer / self.total_layers


_PROBE_MANIFESTS: Dict[str, Dict[str, Any]] = {}


def _load_manifests() -> Dict[str, Dict[str, Any]]:
    """Load probe manifests from atlas directory. Returns
    {(model, task): manifest}."""
    global _PROBE_MANIFESTS
    if _PROBE_MANIFESTS:
        return _PROBE_MANIFESTS
    if not ATLAS_DIR.exists():
        return _PROBE_MANIFESTS

    for manifest_fp in ATLAS_DIR.glob("*.json"):
        try:
            manifest = json.loads(manifest_fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = f"{manifest.get('model')}::{manifest.get('task')}"
        _PROBE_MANIFESTS[key] = manifest
    return _PROBE_MANIFESTS


def list_available_probes() -> List[Dict[str, Any]]:
    """Return a list of {model, task, layer, auc} for every shipped probe."""
    manifests = _load_manifests()
    rows = []
    for key, m in manifests.items():
        rows.append({
            "model": m.get("model"),
            "task": m.get("task"),
            "layer": m.get("layer"),
            "total_layers": m.get("total_layers"),
            "auc_validation": m.get("auc_validation"),
            "class_balance": m.get("class_balance"),
            "probe_version": m.get("probe_version"),
        })
    return sorted(rows, key=lambda r: (r.get("model", ""), r.get("task", "")))


class StyxxProbe:
    """Pre-output commitment probe.

    Loaded by `StyxxProbe.from_pretrained(model, task)`, which reads a
    pre-trained linear classifier from the atlas. Call
    `predict_before_generation(model, tokenizer, prompt)` to get a
    verdict before any output token is emitted.
    """

    def __init__(
        self,
        *,
        model: str,
        task: str,
        layer: int,
        total_layers: int,
        weight,           # torch.Tensor, shape (hidden_size,)
        bias: float,
        positive_class: str,
        negative_class: str,
        auc_validation: Optional[float] = None,
        probe_version: str = "v0",
        atlas_version: str = "v0",
    ):
        self.model = model
        self.task = task
        self.layer = int(layer)
        self.total_layers = int(total_layers)
        self.weight = weight
        self.bias = float(bias)
        self.positive_class = positive_class
        self.negative_class = negative_class
        self.auc_validation = auc_validation
        self.probe_version = probe_version
        self.atlas_version = atlas_version

    # ---------- loading ----------

    @classmethod
    def from_pretrained(cls, model: str, task: str) -> "StyxxProbe":
        """Load a probe from the atlas. Raises ProbeNotAvailable if
        the (model, task) pair is not shipped in the current atlas."""
        try:
            import torch  # noqa: F401
        except Exception as e:
            raise ProbeNotAvailable(
                f"torch unavailable ({e}); probes require torch"
            )

        manifests = _load_manifests()
        key = f"{model}::{task}"
        if key not in manifests:
            available = sorted(manifests.keys())
            raise ProbeNotAvailable(
                f"no probe for model={model!r} task={task!r}. "
                f"available: {available}"
            )

        import torch
        m = manifests[key]
        weight_fp = ATLAS_DIR / m["weight_file"]
        if not weight_fp.exists():
            raise ProbeNotAvailable(
                f"manifest references {weight_fp.name} which is not "
                f"present in atlas"
            )
        state = torch.load(weight_fp, map_location="cpu", weights_only=True)
        return cls(
            model=model,
            task=task,
            layer=m["layer"],
            total_layers=m["total_layers"],
            weight=state["weight"],
            bias=float(state["bias"]),
            positive_class=m["positive_class"],
            negative_class=m["negative_class"],
            auc_validation=m.get("auc_validation"),
            probe_version=m.get("probe_version", "v0"),
            atlas_version=m.get("atlas_version", "v0"),
        )

    # ---------- inference ----------

    def predict_before_generation(
        self,
        model,
        tokenizer,
        prompt: str,
        *,
        apply_chat_template: bool = True,
    ) -> ProbeVerdict:
        """Predict the pre-output commitment for `prompt` on `model`.

        Reads the hidden state at `self.layer` for the final prefill
        token, applies the frozen linear classifier, returns a
        ProbeVerdict. **No output tokens are generated.**

        Parameters
        ----------
        model : transformers.PreTrainedModel
            Target model (must match self.model's checkpoint). Must be
            loaded with `output_hidden_states=True` or the call will
            enable that on the fly.
        tokenizer : transformers.PreTrainedTokenizer
        prompt : str
        apply_chat_template : bool
            If True, wrap prompt with the target model's instruct chat
            template. Default True.
        """
        import torch

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        if apply_chat_template:
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            out = model(input_ids=inputs, output_hidden_states=True)
            hidden = out.hidden_states[self.layer][0, -1, :].to(
                self.weight.dtype
            )
            score = float((hidden @ self.weight).item() + self.bias)

        # Sigmoid for probability of positive class
        p_pos = 1.0 / (1.0 + math.exp(-score))
        confidence = abs(p_pos - 0.5) * 2.0

        return ProbeVerdict(
            model=self.model,
            task=self.task,
            layer=self.layer,
            total_layers=self.total_layers,
            residual_score=score,
            p_positive=p_pos,
            positive_class=self.positive_class,
            negative_class=self.negative_class,
            confidence=confidence,
            n_tokens_in_prefill=int(inputs.shape[1]),
            probe_version=self.probe_version,
            atlas_version=self.atlas_version,
        )

    def gate(self, model, tokenizer, prompt: str,
             *,
             positive_threshold: float = 0.5,
             raise_on_exceed: bool = False) -> ProbeVerdict:
        """Convenience wrapper: predict, and optionally raise if the
        positive-class probability exceeds `positive_threshold`. Useful
        for pre-output safety gates where the positive class is
        'comply' on unsafe prompts."""
        verdict = self.predict_before_generation(model, tokenizer, prompt)
        if raise_on_exceed and verdict.p_positive >= positive_threshold:
            raise SafetyGateError(
                f"probe {self.task!r} on {self.model!r} predicts "
                f"{verdict.positive_class!r} at p={verdict.p_positive:.3f} "
                f">= threshold {positive_threshold}"
            )
        return verdict

    def __repr__(self) -> str:
        return (
            f"StyxxProbe(model={self.model!r}, task={self.task!r}, "
            f"layer={self.layer}/{self.total_layers}, "
            f"auc={self.auc_validation})"
        )

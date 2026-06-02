# -*- coding: utf-8 -*-
"""Tests for styxx.probe — surface API only; actual probe inference
requires torch + the atlas artifacts, which are loaded lazily."""
from __future__ import annotations

import types

import pytest

from styxx.residual_probe import (
    StyxxProbe, ProbeVerdict, ProbeNotAvailable, SafetyGateError,
    list_available_probes,
)


def test_surface_importable():
    assert StyxxProbe is not None
    assert ProbeVerdict is not None
    assert ProbeNotAvailable is not None
    assert SafetyGateError is not None
    assert callable(list_available_probes)


def test_list_available_probes_returns_list():
    probes = list_available_probes()
    assert isinstance(probes, list)
    # At v3.5.0 the atlas starts empty — list may be [].
    for p in probes:
        assert "model" in p
        assert "task" in p


def test_missing_probe_raises_probe_not_available():
    with pytest.raises(ProbeNotAvailable):
        StyxxProbe.from_pretrained(
            model="definitely-nonexistent-model/none",
            task="definitely_not_a_task",
        )


def test_verdict_dataclass_has_alignment_depth():
    # Verdict can be constructed directly for testing / docs
    import types
    weight = types.SimpleNamespace()
    v = ProbeVerdict(
        model="x", task="y", layer=11, total_layers=17,
        residual_score=1.2, p_positive=0.65,
        positive_class="refuse", negative_class="comply",
        confidence=0.3, n_tokens_in_prefill=24,
        probe_version="v0", atlas_version="v0",
    )
    assert v.alignment_depth == pytest.approx(11 / 17, rel=1e-6)
    assert 0.0 <= v.alignment_depth <= 1.0
    d = v.as_dict()
    assert d["task"] == "y"
    assert d["p_positive"] == 0.65


def test_probe_verdict_is_frozen_dataclass_like():
    # We're NOT frozen, but the contract is documented; test the fields
    v = ProbeVerdict(
        model="m", task="t", layer=0, total_layers=1,
        residual_score=0.0, p_positive=0.5,
        positive_class="+", negative_class="-",
        confidence=0.0, n_tokens_in_prefill=1,
        probe_version="v0", atlas_version="v0",
    )
    assert v.alignment_depth == 0.0


# ---------------------------------------------------------------------------
# Regression: device handling in predict_before_generation
#
# Bug (pre-fix): the prefill-end hidden state was cast to the probe weight's
# DTYPE but not its DEVICE, then matmul'd with the weight. Shipped weights
# load via map_location="cpu", so a model on GPU produced
#     RuntimeError: Expected all tensors to be on the same device
# Fix: hidden.to(device=self.weight.device, dtype=self.weight.dtype) before
# the matmul. These tests pin that contract.
# ---------------------------------------------------------------------------


def _make_cpu_probe(torch, *, hidden_size, layer, total_layers, bias=0.0):
    """A StyxxProbe whose weight lives on CPU/float32 — the shipped state
    (from_pretrained loads with map_location='cpu')."""
    weight = torch.arange(hidden_size, dtype=torch.float32) / hidden_size
    return StyxxProbe(
        model="fake/model", task="truthfulness",
        layer=layer, total_layers=total_layers,
        weight=weight, bias=bias,
        positive_class="correct", negative_class="incorrect",
    )


class _FakeTokenizer:
    """Minimal stand-in: apply_chat_template / __call__ return a (1, seq)
    LongTensor, enough for predict_before_generation."""

    def __init__(self, torch, seq=5):
        self._torch = torch
        self._seq = seq

    def apply_chat_template(self, messages, add_generation_prompt=True,
                            return_tensors="pt"):
        assert return_tensors == "pt"
        return self._torch.ones((1, self._seq), dtype=self._torch.long)

    def __call__(self, prompt, return_tensors="pt"):
        ids = self._torch.ones((1, self._seq), dtype=self._torch.long)
        return types.SimpleNamespace(input_ids=ids)


class _SpyHidden:
    """Wraps a tensor and records the kwargs passed to .to(), so a test can
    assert the hidden state is moved to the weight's DEVICE (not dtype only).
    Slicing returns another spy; .to() delegates to the real tensor."""

    def __init__(self, tensor, log):
        self._t = tensor
        self._log = log

    def __getitem__(self, idx):
        return _SpyHidden(self._t[idx], self._log)

    def to(self, *args, **kwargs):
        self._log.append({"args": args, "kwargs": kwargs})
        return self._t.to(*args, **kwargs)


class _FakeModel:
    """Minimal HF-like causal LM. Hidden states live on `device`/`dtype`;
    output_hidden_states=True yields total_layers+1 tensors of shape
    (1, seq, hidden_size), matching transformers' convention."""

    def __init__(self, torch, *, hidden_size, total_layers, device, dtype,
                 spy_log=None):
        self._torch = torch
        self._hidden_size = hidden_size
        self._total_layers = total_layers
        self._param = torch.zeros(1, device=device, dtype=dtype)
        self._spy_log = spy_log

    def parameters(self):
        yield self._param

    def __call__(self, input_ids=None, output_hidden_states=False):
        seq = int(input_ids.shape[1])
        dev, dt = self._param.device, self._param.dtype
        hidden_states = []
        for _ in range(self._total_layers + 1):
            # Deterministic, nonzero last-token activation so the score is
            # not trivially the bias.
            t = self._torch.full((1, seq, self._hidden_size), 0.5,
                                 device=dev, dtype=dt)
            if self._spy_log is not None:
                t = _SpyHidden(t, self._spy_log)
            hidden_states.append(t)
        return types.SimpleNamespace(hidden_states=hidden_states)


def test_predict_before_generation_cross_device_cpu_weight():
    """Model on CUDA/fp16 + probe weight on CPU/fp32 (the shipped config)
    must not raise a device-mismatch RuntimeError. Pre-fix this raised
    'Expected all tensors to be on the same device'."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; cross-device path needs a GPU")

    H, total_layers, layer = 8, 4, 2
    probe = _make_cpu_probe(torch, hidden_size=H, layer=layer,
                            total_layers=total_layers)
    assert probe.weight.device.type == "cpu"

    model = _FakeModel(torch, hidden_size=H, total_layers=total_layers,
                       device="cuda", dtype=torch.float16)
    tok = _FakeTokenizer(torch, seq=5)

    verdict = probe.predict_before_generation(model, tok, "What is 2 + 2?")

    assert isinstance(verdict, ProbeVerdict)
    assert 0.0 <= verdict.p_positive <= 1.0
    assert verdict.layer == layer
    assert verdict.n_tokens_in_prefill == 5
    # The fix moves the hidden state, not the weight — weight stays on CPU.
    assert probe.weight.device.type == "cpu"


def test_predict_before_generation_moves_hidden_to_weight_device():
    """Portable contract guard (no GPU needed): the hidden state must be
    sent to the weight's DEVICE, not merely cast to its dtype. Catches a
    regression to `.to(self.weight.dtype)` on CPU-only runners too."""
    torch = pytest.importorskip("torch")

    H, total_layers, layer = 8, 4, 2
    probe = _make_cpu_probe(torch, hidden_size=H, layer=layer,
                            total_layers=total_layers)

    log = []
    model = _FakeModel(torch, hidden_size=H, total_layers=total_layers,
                       device="cpu", dtype=torch.float32, spy_log=log)
    tok = _FakeTokenizer(torch, seq=3)

    probe.predict_before_generation(model, tok, "ping")

    assert len(log) == 1, f"expected one .to() on the read layer, got {log}"
    kwargs = log[0]["kwargs"]
    assert "device" in kwargs, (
        "hidden.to() must pass device= — a dtype-only cast leaves the "
        "hidden state on the model's device and breaks on GPU"
    )
    assert kwargs["device"] == probe.weight.device
    assert kwargs.get("dtype") == probe.weight.dtype

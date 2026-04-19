# -*- coding: utf-8 -*-
"""Tests for styxx.probe — surface API only; actual probe inference
requires torch + the atlas artifacts, which are loaded lazily."""
from __future__ import annotations

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

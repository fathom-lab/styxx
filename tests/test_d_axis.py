# -*- coding: utf-8 -*-
"""
test_d_axis.py -- tests for the tier 1 D-axis honesty module.

All tests run WITHOUT GPU and WITHOUT loading a real model. The
DAxisScorer is tested via:
  1. DAxisStats.from_values() — pure math, no model needed
  2. The config layer — tier1_enabled/model/device
  3. The core integration — run_on_trajectories with d_trajectory
  4. The vitals enrichment — d_honesty shortcut + as_markdown
  5. The sae.py scaffold — NotImplementedError behavior
  6. The CLI argparse — d-axis subcommand wiring

Testing the actual model-loading + forward-pass path requires
a GPU + 4GB VRAM + ~30s model load. Those tests live in the
integration suite (tests/integration/test_d_axis_gpu.py) and
are gated behind `pytest -m gpu`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import styxx
from styxx.d_axis import DAxisStats


# ══════════════════════════════════════════════════════════════════
# 1. DAxisStats pure math
# ══════════════════════════════════════════════════════════════════

def test_daxis_stats_from_empty():
    s = DAxisStats.from_values([])
    assert s.mean == 0.0
    assert s.n_tokens == 0


def test_daxis_stats_from_single():
    s = DAxisStats.from_values([0.82])
    assert s.mean == pytest.approx(0.82)
    assert s.std == 0.0
    assert s.n_tokens == 1


def test_daxis_stats_from_sequence():
    vals = [0.80, 0.85, 0.70, 0.90, 0.75, 0.82]
    s = DAxisStats.from_values(vals)
    assert s.n_tokens == 6
    assert 0.70 <= s.mean <= 0.90
    assert s.min_val == 0.70
    assert s.max_val == 0.90
    assert s.std > 0


def test_daxis_stats_delta_sign():
    # Declining D — getting less honest
    declining = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65]
    s = DAxisStats.from_values(declining)
    assert s.delta < 0, "declining D should have negative delta"

    # Rising D — getting more honest
    rising = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    s = DAxisStats.from_values(rising)
    assert s.delta > 0, "rising D should have positive delta"


def test_daxis_stats_early_late_split():
    vals = [0.50, 0.50, 0.50, 0.90, 0.90, 0.90]
    s = DAxisStats.from_values(vals)
    assert s.early_mean == pytest.approx(0.50)
    assert s.late_mean == pytest.approx(0.90)
    assert s.delta == pytest.approx(0.40)


# ══════════════════════════════════════════════════════════════════
# 2. Config layer
# ══════════════════════════════════════════════════════════════════

def test_tier1_enabled_default_false():
    orig = os.environ.get("STYXX_TIER1_ENABLED")
    os.environ.pop("STYXX_TIER1_ENABLED", None)
    try:
        assert styxx.tier1_enabled() is False
    finally:
        if orig is not None:
            os.environ["STYXX_TIER1_ENABLED"] = orig


def test_tier1_enabled_truthy():
    orig = os.environ.get("STYXX_TIER1_ENABLED")
    try:
        os.environ["STYXX_TIER1_ENABLED"] = "1"
        from styxx import config
        assert config.tier1_enabled() is True
    finally:
        if orig is None:
            os.environ.pop("STYXX_TIER1_ENABLED", None)
        else:
            os.environ["STYXX_TIER1_ENABLED"] = orig


def test_tier1_model_default():
    orig = os.environ.get("STYXX_TIER1_MODEL")
    os.environ.pop("STYXX_TIER1_MODEL", None)
    try:
        from styxx import config
        assert config.tier1_model() == "google/gemma-2-2b-it"
    finally:
        if orig is not None:
            os.environ["STYXX_TIER1_MODEL"] = orig


def test_tier1_model_custom():
    orig = os.environ.get("STYXX_TIER1_MODEL")
    try:
        os.environ["STYXX_TIER1_MODEL"] = "meta-llama/Llama-3.2-3B"
        from styxx import config
        assert config.tier1_model() == "meta-llama/Llama-3.2-3B"
    finally:
        if orig is None:
            os.environ.pop("STYXX_TIER1_MODEL", None)
        else:
            os.environ["STYXX_TIER1_MODEL"] = orig


# ══════════════════════════════════════════════════════════════════
# 3. Core integration — run_on_trajectories with d_trajectory
# ══════════════════════════════════════════════════════════════════

def _fake_trajectory(n: int) -> dict:
    return {
        "entropy":     [1.5 + (i % 3) * 0.1 for i in range(n)],
        "logprob":     [-0.5 - (i % 4) * 0.05 for i in range(n)],
        "top2_margin": [0.5 + (i % 2) * 0.05 for i in range(n)],
    }


def test_run_on_trajectories_without_d_axis():
    """Without d_trajectory, phases should have d_honesty_mean=None."""
    rt = styxx.StyxxRuntime()
    traj = _fake_trajectory(25)
    v = rt.run_on_trajectories(**traj)
    assert v.phase1_pre.d_honesty_mean is None
    assert v.d_honesty is None


def test_run_on_trajectories_with_d_axis():
    """With d_trajectory, phases should be enriched with D stats."""
    rt = styxx.StyxxRuntime()
    traj = _fake_trajectory(25)
    d_vals = [0.80 + (i % 5) * 0.02 for i in range(25)]
    v = rt.run_on_trajectories(**traj, d_trajectory=d_vals)

    # Phase 1 uses tokens [0, 1)
    assert v.phase1_pre.d_honesty_mean is not None
    assert 0.0 <= v.phase1_pre.d_honesty_mean <= 1.0

    # Phase 4 uses tokens [0, 25)
    assert v.phase4_late is not None
    assert v.phase4_late.d_honesty_mean is not None

    # d_honesty shortcut should return a string
    assert v.d_honesty is not None
    assert isinstance(v.d_honesty, str)
    assert "." in v.d_honesty  # should be a decimal number


def test_d_honesty_shortcut_reflects_latest_phase():
    """d_honesty should prefer phase4 when available."""
    rt = styxx.StyxxRuntime()
    traj = _fake_trajectory(25)
    d_vals = [0.30] * 5 + [0.90] * 20  # low early, high late
    v = rt.run_on_trajectories(**traj, d_trajectory=d_vals)

    # Phase 4 covers all 25 tokens — its mean should dominate
    assert v.d_honesty is not None
    d_val = float(v.d_honesty)
    assert d_val > 0.5  # overall mean is pulled up by the late 0.90s


def test_as_markdown_includes_d_honesty():
    rt = styxx.StyxxRuntime()
    traj = _fake_trajectory(25)
    d_vals = [0.82] * 25
    v = rt.run_on_trajectories(**traj, d_trajectory=d_vals)
    md = v.as_markdown()
    assert "d_honesty" in md


def test_phase_reading_d_fields_default_none():
    """Without d_trajectory, PhaseReading d_* fields should be None."""
    from styxx.vitals import PhaseReading
    # Create a minimal PhaseReading
    pr = PhaseReading(
        phase="phase1_preflight",
        n_tokens_used=1,
        features=[0.0] * 12,
        predicted_category="reasoning",
        margin=0.1,
        distances={},
        probs={"reasoning": 0.5},
    )
    assert pr.d_honesty_mean is None
    assert pr.d_honesty_std is None
    assert pr.d_honesty_delta is None
    assert pr.k_depth is None
    assert pr.c_coherence is None
    assert pr.s_commitment is None


# ══════════════════════════════════════════════════════════════════
# 4. SAE scaffold
# ══════════════════════════════════════════════════════════════════

def test_sae_scaffold_raises():
    from styxx.sae import SAEInstruments
    instruments = SAEInstruments()
    with pytest.raises(NotImplementedError, match="tier 2"):
        instruments.measure("test prompt")


def test_sae_scaffold_individual_axes_raise():
    from styxx.sae import SAEInstruments
    instruments = SAEInstruments()
    with pytest.raises(NotImplementedError):
        instruments.measure_k("test")
    with pytest.raises(NotImplementedError):
        instruments.measure_c("test")
    with pytest.raises(NotImplementedError):
        instruments.measure_s("test")


# ══════════════════════════════════════════════════════════════════
# 5. CLI argparse
# ══════════════════════════════════════════════════════════════════

def test_d_axis_cli_argparse():
    """The d-axis subcommand should parse cleanly."""
    from styxx.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["d-axis", "why is the sky blue?", "--max-tokens", "15"])
    assert args.prompt == "why is the sky blue?"
    assert args.max_tokens == 15


# ══════════════════════════════════════════════════════════════════
# 6. Version
# ══════════════════════════════════════════════════════════════════

def test_version_is_0_3_0():
    assert styxx.__version__ == "0.3.0"


def test_tier1_exports():
    assert hasattr(styxx, "tier1_enabled")
    assert hasattr(styxx, "tier1_model")
    assert hasattr(styxx, "tier1_device")

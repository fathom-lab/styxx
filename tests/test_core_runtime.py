# -*- coding: utf-8 -*-
"""
test_core_runtime.py -- dedicated tests for styxx.core (StyxxRuntime).

Forensic note: the 2026-08-09 full-repo survey found that styxx/core.py
-- the five-phase logprob-vitals runtime every adapter routes through --
had NO dedicated test file. Its behavior was only exercised incidentally
(test_d_axis.py drives run_on_trajectories for D-axis enrichment). This
file closes that gap: phase window cutoffs, gate thresholds, cross-phase
coherence, forecast wiring, tier detection, and short-trajectory
behavior are pinned here.

All tests are OFFLINE and DETERMINISTIC: numpy only, no network, no API
keys, no model downloads. STYXX_DATA_DIR and Path.home are pointed at
tmp_path so ~/.styxx is never read or written -- a stray local
calibration file under ~/.styxx/calibration/ would otherwise silently
shift classifier outputs and make these tests machine-dependent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from styxx.core import GATE_THRESHOLDS, StyxxRuntime, _try_import, detect_tiers
from styxx.vitals import CATEGORIES, PHASE_TOKEN_CUTOFFS, PhaseReading, Vitals


# ══════════════════════════════════════════════════════════════════
# Fixtures + helpers
# ══════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Sandbox all file I/O and pin tier 0.

    Path.home is patched because CentroidClassifier reads
    ~/.styxx/calibration/{agent}.json (fail-open centroid overlay)
    regardless of STYXX_DATA_DIR.
    """
    monkeypatch.setenv("STYXX_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("STYXX_TIER1_ENABLED", raising=False)


def _fake_trajectory(n: int) -> dict:
    return {
        "entropy":     [1.5 + (i % 3) * 0.1 for i in range(n)],
        "logprob":     [-0.5 - (i % 4) * 0.05 for i in range(n)],
        "top2_margin": [0.5 + (i % 2) * 0.05 for i in range(n)],
    }


def _reading(phase: str, category: str, confidence: float) -> PhaseReading:
    """Hand-built PhaseReading with an exact confidence, for gate tests."""
    other = (1.0 - confidence) / (len(CATEGORIES) - 1)
    probs = {c: other for c in CATEGORIES}
    probs[category] = confidence
    return PhaseReading(
        phase=phase,
        n_tokens_used=PHASE_TOKEN_CUTOFFS.get(phase, 1),
        features=[0.0] * 12,
        predicted_category=category,
        margin=0.5,
        distances={c: 1.0 for c in CATEGORIES},
        probs=probs,
    )


def _load_demo():
    from styxx.cli import _load_demo_trajectories
    return _load_demo_trajectories()


# ══════════════════════════════════════════════════════════════════
# 1. Phase windows at the 1/5/15/25 cutoffs
# ══════════════════════════════════════════════════════════════════

def test_phase_cutoffs_are_locked():
    """The runtime's n>=5/15/25 branches assume these exact cutoffs."""
    assert PHASE_TOKEN_CUTOFFS == {
        "phase1_preflight": 1,
        "phase2_early":     5,
        "phase3_mid":      15,
        "phase4_late":     25,
    }


def test_phase1_only_at_one_token():
    rt = StyxxRuntime()
    v = rt.run_on_trajectories(**_fake_trajectory(1))
    assert v.phase1_pre is not None
    assert v.phase2_early is None
    assert v.phase3_mid is None
    assert v.phase4_late is None


def test_phase2_lights_up_at_five_tokens():
    rt = StyxxRuntime()
    assert rt.run_on_trajectories(**_fake_trajectory(4)).phase2_early is None
    assert rt.run_on_trajectories(**_fake_trajectory(5)).phase2_early is not None


def test_phase3_lights_up_at_fifteen_tokens():
    rt = StyxxRuntime()
    assert rt.run_on_trajectories(**_fake_trajectory(14)).phase3_mid is None
    assert rt.run_on_trajectories(**_fake_trajectory(15)).phase3_mid is not None


def test_phase4_lights_up_at_twentyfive_tokens():
    rt = StyxxRuntime()
    assert rt.run_on_trajectories(**_fake_trajectory(24)).phase4_late is None
    assert rt.run_on_trajectories(**_fake_trajectory(25)).phase4_late is not None


def test_n_tokens_used_matches_cutoffs():
    rt = StyxxRuntime()
    v = rt.run_on_trajectories(**_fake_trajectory(30))
    assert v.phase1_pre.n_tokens_used == 1
    assert v.phase2_early.n_tokens_used == 5
    assert v.phase3_mid.n_tokens_used == 15
    assert v.phase4_late.n_tokens_used == 25


def test_run_on_prefix_delegates_to_run_on_trajectories():
    rt = StyxxRuntime()
    traj = _fake_trajectory(6)
    a = rt.run_on_prefix(**traj)
    b = rt.run_on_trajectories(**traj)
    assert a.phase1_pre.predicted_category == b.phase1_pre.predicted_category
    assert a.phase2_early.predicted_category == b.phase2_early.predicted_category
    assert a.phase3_mid is None and b.phase3_mid is None


# ══════════════════════════════════════════════════════════════════
# 2. run_on_trajectories -- synthetic + bundled demo trajectories
# ══════════════════════════════════════════════════════════════════

def test_synthetic_run_is_deterministic():
    rt = StyxxRuntime()
    traj = _fake_trajectory(25)
    v1 = rt.run_on_trajectories(**traj)
    v2 = rt.run_on_trajectories(**traj)
    for p1, p2 in zip(
        (v1.phase1_pre, v1.phase2_early, v1.phase3_mid, v1.phase4_late),
        (v2.phase1_pre, v2.phase2_early, v2.phase3_mid, v2.phase4_late),
    ):
        assert p1.predicted_category == p2.predicted_category
        assert p1.confidence == p2.confidence
    assert v1.coherence == v2.coherence


def test_synthetic_run_probs_valid():
    rt = StyxxRuntime()
    v = rt.run_on_trajectories(**_fake_trajectory(25))
    for reading in (v.phase1_pre, v.phase2_early, v.phase3_mid, v.phase4_late):
        assert reading.predicted_category in CATEGORIES
        assert abs(sum(reading.probs.values()) - 1.0) < 1e-9
        assert 0.0 <= reading.confidence <= 1.0


def test_demo_trajectories_full_pipeline():
    """Every bundled demo trajectory (30 real tokens) reaches all four
    phases and yields well-formed vitals."""
    data = _load_demo()
    rt = StyxxRuntime()
    assert set(data["trajectories"]) == set(CATEGORIES)
    for cat, traj in data["trajectories"].items():
        v = rt.run_on_trajectories(
            entropy=traj["entropy"],
            logprob=traj["logprob"],
            top2_margin=traj["top2_margin"],
        )
        assert isinstance(v, Vitals)
        assert v.tier_active == 0
        for reading in (v.phase1_pre, v.phase2_early, v.phase3_mid, v.phase4_late):
            assert reading is not None, f"{cat}: missing phase reading"
            assert reading.predicted_category in CATEGORIES
        assert v.coherence is not None
        assert 0.0 <= v.coherence <= 1.0
        assert v.forecast is not None
        assert v.abort_reason is None or isinstance(v.abort_reason, str)


def test_demo_trajectory_deterministic():
    data = _load_demo()
    traj = data["trajectories"]["retrieval"]
    rt = StyxxRuntime()
    kwargs = dict(
        entropy=traj["entropy"],
        logprob=traj["logprob"],
        top2_margin=traj["top2_margin"],
    )
    v1 = rt.run_on_trajectories(**kwargs)
    v2 = rt.run_on_trajectories(**kwargs)
    assert v1.phase4_late.predicted_category == v2.phase4_late.predicted_category
    assert v1.phase4_late.confidence == v2.phase4_late.confidence
    assert v1.coherence == v2.coherence


# ══════════════════════════════════════════════════════════════════
# 3. Gate logic thresholds
# ══════════════════════════════════════════════════════════════════

def test_gate_thresholds_locked():
    """Numbers from the 2026-04-11 streaming gate test -- an edit here
    is a recalibration, not a refactor."""
    assert GATE_THRESHOLDS == {
        "phase1_adversarial_refuse":  0.65,
        "phase4_hallucination_abort": 0.55,
    }


def test_phase1_adversarial_at_threshold_aborts():
    rt = StyxxRuntime()
    p1 = _reading("phase1_preflight", "adversarial", 0.65)
    reason = rt._evaluate_gates(p1, None)
    assert reason is not None
    assert "adversarial" in reason
    assert "phase 1" in reason


def test_phase1_adversarial_below_threshold_reads_but_does_not_act():
    rt = StyxxRuntime()
    p1 = _reading("phase1_preflight", "adversarial", 0.64)
    assert rt._evaluate_gates(p1, None) is None


def test_phase4_hallucination_at_threshold_aborts():
    rt = StyxxRuntime()
    p1 = _reading("phase1_preflight", "reasoning", 0.9)
    p4 = _reading("phase4_late", "hallucination", 0.55)
    reason = rt._evaluate_gates(p1, p4)
    assert reason is not None
    assert "hallucination" in reason
    assert "phase 4" in reason


def test_phase4_hallucination_below_threshold_reads_but_does_not_act():
    rt = StyxxRuntime()
    p1 = _reading("phase1_preflight", "reasoning", 0.9)
    p4 = _reading("phase4_late", "hallucination", 0.54)
    assert rt._evaluate_gates(p1, p4) is None


def test_benign_categories_never_abort():
    rt = StyxxRuntime()
    for cat in ("retrieval", "reasoning", "refusal", "creative"):
        p1 = _reading("phase1_preflight", cat, 0.99)
        p4 = _reading("phase4_late", cat, 0.99)
        assert rt._evaluate_gates(p1, p4) is None, cat


def test_missing_phase4_skips_hallucination_gate():
    rt = StyxxRuntime()
    p1 = _reading("phase1_preflight", "reasoning", 0.9)
    assert rt._evaluate_gates(p1, None) is None


def test_gate_threshold_override_via_constructor():
    rt = StyxxRuntime(gate_thresholds={"phase1_adversarial_refuse": 0.90})
    p1 = _reading("phase1_preflight", "adversarial", 0.70)
    assert rt._evaluate_gates(p1, None) is None
    p1_hot = _reading("phase1_preflight", "adversarial", 0.90)
    assert rt._evaluate_gates(p1_hot, None) is not None
    # The override must not mutate the module-level defaults
    assert GATE_THRESHOLDS["phase1_adversarial_refuse"] == 0.65


# ══════════════════════════════════════════════════════════════════
# 4. Cross-phase coherence
# ══════════════════════════════════════════════════════════════════

def test_coherence_identical_phases_is_one():
    rt = StyxxRuntime()
    phases = [
        _reading("phase1_preflight", "reasoning", 0.7),
        _reading("phase2_early", "reasoning", 0.7),
    ]
    coherence, transitions = rt._compute_coherence(phases)
    assert coherence == pytest.approx(1.0)
    assert len(transitions) == 1
    assert all(x == pytest.approx(0.0) for x in transitions[0])


def test_coherence_orthogonal_phases_is_zero():
    rt = StyxxRuntime()
    p1 = _reading("phase1_preflight", "retrieval", 1.0)
    p2 = _reading("phase2_early", "hallucination", 1.0)
    coherence, transitions = rt._compute_coherence([p1, p2])
    assert coherence == pytest.approx(0.0)
    assert len(transitions) == 1


def test_coherence_none_with_single_phase():
    rt = StyxxRuntime()
    v = rt.run_on_trajectories(**_fake_trajectory(1))
    assert v.coherence is None
    assert v.transition_vectors is None


def test_coherence_range_and_transition_shape():
    rt = StyxxRuntime()
    v = rt.run_on_trajectories(**_fake_trajectory(25))
    assert 0.0 <= v.coherence <= 1.0
    # 4 phases -> 3 transitions, each a 6-dim category-space vector
    assert len(v.transition_vectors) == 3
    for vec in v.transition_vectors:
        assert len(vec) == len(CATEGORIES)


# ══════════════════════════════════════════════════════════════════
# 5. Forecast wiring
# ══════════════════════════════════════════════════════════════════

def test_forecast_present_at_five_tokens():
    rt = StyxxRuntime()
    v = rt.run_on_trajectories(**_fake_trajectory(5))
    assert v.forecast is not None
    assert v.forecast.predicted_category in CATEGORIES
    assert v.forecast.risk_level in ("low", "moderate", "high", "critical")


def test_forecast_absent_below_five_tokens():
    rt = StyxxRuntime()
    v = rt.run_on_trajectories(**_fake_trajectory(4))
    assert v.forecast is None


# ══════════════════════════════════════════════════════════════════
# 6. detect_tiers -- find_spec monkeypatched both ways
# ══════════════════════════════════════════════════════════════════

def test_detect_tiers_all_deps_available(monkeypatch):
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: object()
    )
    assert detect_tiers() == {0: True, 1: True, 2: True, 3: True}


def test_detect_tiers_no_deps_available(monkeypatch):
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: None
    )
    assert detect_tiers() == {0: True, 1: False, 2: False, 3: False}


def test_detect_tiers_partial_deps(monkeypatch):
    """transformers + torch without circuit_tracer -> tier 1 only."""
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: object() if name in ("transformers", "torch") else None,
    )
    assert detect_tiers() == {0: True, 1: True, 2: False, 3: False}


def test_detect_tiers_survives_find_spec_errors(monkeypatch):
    def _boom(name, *a, **k):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(importlib.util, "find_spec", _boom)
    assert detect_tiers() == {0: True, 1: False, 2: False, 3: False}

    def _boom_value(name, *a, **k):
        raise ValueError(name)
    monkeypatch.setattr(importlib.util, "find_spec", _boom_value)
    assert detect_tiers() == {0: True, 1: False, 2: False, 3: False}


def test_try_import_real_modules():
    assert _try_import("json") is True
    assert _try_import("definitely_not_a_real_module_xyz") is False


def test_runtime_reports_tier_zero_without_tier1_env():
    rt = StyxxRuntime()
    assert rt.tier_active == 0
    v = rt.run_on_trajectories(**_fake_trajectory(25))
    assert v.tier_active == 0


# ══════════════════════════════════════════════════════════════════
# 7. Graceful behavior on too-short trajectories
# ══════════════════════════════════════════════════════════════════

def test_empty_trajectory_does_not_crash():
    rt = StyxxRuntime()
    v = rt.run_on_trajectories(entropy=[], logprob=[], top2_margin=[])
    assert isinstance(v, Vitals)
    assert v.phase1_pre is not None
    assert v.phase2_early is None
    assert v.phase3_mid is None
    assert v.phase4_late is None
    assert v.coherence is None
    assert v.forecast is None


def test_short_trajectory_gate_is_pending():
    rt = StyxxRuntime()
    v = rt.run_on_trajectories(**_fake_trajectory(3))
    assert v.gate == "pending"
    assert v.phase2 == "-"
    assert v.phase3 == "-"
    assert v.phase4 == "-"
    # Classification falls back to the latest available phase
    assert v.category == v.phase1_pre.predicted_category

"""styxx.resonance — the profiler's contract, on a pure-Python model tree and, when present, on torch.

The load-bearing tests: the input model is never mutated, the decomposition sums, a model with no
phase is refused rather than profiled as zero, and the certifies string travels with every profile.
"""
from __future__ import annotations

import math

import pytest

from styxx import resonance
from styxx.resonance import freeze_adaptation, profile, render, ssm_cores, zero_oscillation


class Core:
    """A fake complex-diagonal core: one mode, phase theta, magnitude r, optional adaptation."""

    def __init__(self, theta, adaptive=False):
        self.theta = [theta]
        self.r = 0.95
        if adaptive:
            self.kappa_override = None


class Model:
    def __init__(self, theta=1.0, adaptive=False):
        self.core = Core(theta, adaptive)
        self.head = object()

    # a score that depends on the phase and on adaptation being on, so the decomposition is visible
    def score(self):
        osc = 0.5 * (1 - math.cos(self.core.theta[0]))         # 0 at theta=0, 1 at pi
        adapt = 0.2 if getattr(self.core, "kappa_override", 0.0) is None else 0.0
        return 0.1 + 0.6 * osc + adapt


def test_cores_are_found_by_phase_name_on_a_plain_object_tree():
    m = Model()
    assert ssm_cores(m) == [m.core]


def test_zeroing_and_freezing_act_in_place_and_count_what_they_touched():
    m = Model(theta=1.0, adaptive=True)
    assert zero_oscillation(m) == 1 and m.core.theta == [0]
    assert freeze_adaptation(m) == 1 and m.core.kappa_override == 0.0


def test_profile_decomposes_and_sums():
    m = Model(theta=math.pi, adaptive=True)
    p = profile(m, Model.score)
    assert p["baseline"] == pytest.approx(0.9)
    assert p["static_osc_eval"] == pytest.approx(0.7)
    assert p["decay_floor"] == pytest.approx(0.1)
    assert p["static_oscillation_reliance"] == pytest.approx(0.6)
    assert p["adaptation_reliance"] == pytest.approx(0.2)
    assert p["total_oscillation_reliance"] == pytest.approx(0.8)
    assert p["static_oscillation_reliance"] + p["adaptation_reliance"] == pytest.approx(p["total_oscillation_reliance"])


def test_the_input_model_is_never_mutated():
    m = Model(theta=2.0, adaptive=True)
    profile(m, Model.score)
    assert m.core.theta == [2.0] and m.core.kappa_override is None


def test_a_model_without_adaptation_reports_zero_adaptation_reliance():
    p = profile(Model(theta=1.0), Model.score)
    assert p["has_adaptation"] is False and p["adaptation_reliance"] == 0.0


def test_a_model_with_no_phase_is_refused_not_profiled_as_zero():
    class Bare:
        def __init__(self):
            self.head = object()
    with pytest.raises(ValueError, match="no SSM cores"):
        profile(Bare(), lambda m: 1.0)


def test_the_profile_carries_its_boundary():
    p = profile(Model(), Model.score)
    assert "NOT evidence" in p["certifies"]
    assert "certifies" in render(p)


def test_the_demo_refuses_without_torch(monkeypatch, capsys):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch here")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    assert resonance.main(["--demo", "rich"]) == 2
    assert "REFUSED" in capsys.readouterr().out


def test_torch_model_profiles_through_the_same_contract():
    torch = pytest.importorskip("torch")

    class TCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta0 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.kappa_override = None

    class TModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = TCore()

    m = TModel()

    def score(model):
        osc = float((1 - torch.cos(model.core.theta0)).mean() / 2)
        return osc + (0.1 if model.core.kappa_override is None else 0.0)
    p = profile(m, score)
    assert p["n_ssm_cores"] == 1 and p["has_adaptation"] is True
    assert p["adaptation_reliance"] == pytest.approx(0.1)
    assert float(m.core.theta0[0]) == 1.0, "the torch model was not mutated"

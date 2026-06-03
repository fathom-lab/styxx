# -*- coding: utf-8 -*-
"""Offline tests for styxx.meaning_integrity — the machine-side meaning-integrity monitor.
Uses a synthetic human reference + a model embedding that shares its latent structure, so the
alignment is genuine, then checks the monitor's defining properties hold."""
import numpy as np
import pytest

from styxx.meaning_integrity import (
    MeaningReference, MeaningVitalSign,
    meaning_alignment, meaning_dispersion, per_concept_alignment, meaning_integrity_report,
)


def _setup(N=80, F=12, D=64, seed=0):
    """A human reference and a model embedding that share a latent code (so they genuinely align)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, 8))
    human = Z @ rng.standard_normal((8, F)) + 0.1 * rng.standard_normal((N, F))
    model = Z @ rng.standard_normal((8, D)) + 0.1 * rng.standard_normal((N, D))
    return MeaningReference(human, words=[f"c{i}" for i in range(N)]), model, rng


def test_alignment_is_positive_when_geometry_shared():
    ref, model, _ = _setup()
    assert meaning_alignment(model, ref) > 0.3


def test_invariant_to_rotation_scale_translation():
    ref, model, rng = _setup()
    base = meaning_alignment(model, ref)
    Q, _ = np.linalg.qr(rng.standard_normal((model.shape[1], model.shape[1])))
    assert abs(meaning_alignment(model @ Q, ref) - base) < 1e-6     # rotation
    assert abs(meaning_alignment(model * 7.3, ref) - base) < 1e-6   # isotropic scale
    assert abs(meaning_alignment(model + 3.1, ref) - base) < 1e-6   # translation


def test_sensitive_to_shuffle():
    ref, model, rng = _setup()
    shuffled = model[rng.permutation(model.shape[0])]
    assert abs(meaning_alignment(shuffled, ref)) < 0.15


def test_dispersion_catches_collapse_that_alignment_misses():
    ref, model, _ = _setup()
    base_a = meaning_alignment(model, ref)
    base_d = meaning_dispersion(model)
    collapsed = 0.1 * model + 0.9 * model.mean(0)          # uniform blur toward the mean
    assert abs(meaning_alignment(collapsed, ref) - base_a) < 1e-6   # angular channel is blind (by design)
    assert meaning_dispersion(collapsed) / base_d < 0.2            # dispersion channel catches it


def test_localizes_corrupted_concepts():
    ref, model, rng = _setup()
    N = model.shape[0]; k = N // 3
    C = rng.choice(N, k, replace=False)
    corrupted = model.copy()
    corrupted[C] = model[rng.permutation(N)][:k]
    pc = per_concept_alignment(corrupted, ref)
    assert pc[C].mean() < np.delete(pc, C).mean()         # corrupted concepts score lower


def test_vital_sign_verdicts():
    ref, model, rng = _setup()
    vs = MeaningVitalSign(ref).calibrate(model)
    assert vs.check(model)["status"] == "HEALTHY"
    assert vs.check(model[rng.permutation(model.shape[0])])["status"] == "BROKEN"
    collapsed = 0.2 * model + 0.8 * model.mean(0)
    assert vs.check(collapsed)["status"] == "DEGRADED"     # caught by the dispersion channel


def test_report_shape_and_keys():
    ref, model, _ = _setup()
    rep = meaning_integrity_report(model, ref, top=5)
    assert set(rep) >= {"alignment", "status", "worst_concepts"}
    assert len(rep["worst_concepts"]) == 5
    assert rep["status"] in {"HEALTHY", "DEGRADED", "BROKEN"}


def test_shape_mismatch_raises():
    ref, model, _ = _setup()
    with pytest.raises(ValueError):
        meaning_alignment(model[:-1], ref)

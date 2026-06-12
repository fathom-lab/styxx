"""Tests for styxx.crossmind — portable value-axis readout across model internals.

Offline, synthetic, deterministic (no models, no network), matching the package convention.
Locks: the validated math (T1), determinism (T4), the read-only demarcation, the public
surface, and the certificate shape. Gates pre-registered in
papers/crossmind-instrument/PREREG_crossmind_v0_2026_06_12.md.
"""
import json
import math

import numpy as np
import pytest

from styxx import crossmind as cm


# ---- primitives: known-answer math (T1) -----------------------------------------------------

def test_auroc_known_answers():
    assert cm.auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0          # perfect separation
    assert cm.auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0          # perfectly reversed
    assert cm.auroc([1, 1], [1, 0]) == 0.5                       # full tie -> 0.5
    assert math.isnan(cm.auroc([1, 2, 3], [1, 1, 1]))           # one class -> nan


def test_discrim_is_direction_agnostic():
    assert cm.discrim([0, 1, 2, 3], [1, 1, 0, 0]) == 1.0        # reversed reads as 1.0
    assert cm.discrim([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0


def test_fit_direction_points_from_neg_to_pos():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((40, 6))
    axis = np.zeros(6); axis[0] = 1.0
    pos = base + axis; neg = base - axis
    states = np.vstack([pos, neg]); labels = np.array([1] * 40 + [0] * 40)
    w = cm.fit_direction(states, labels)
    assert abs(np.linalg.norm(w) - 1.0) < 1e-9                  # unit norm
    assert w[0] > 0.9                                            # aligned with the planted axis


def test_zca_whitens_to_identity_covariance():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((10, 10))
    X = rng.standard_normal((4000, 10)) @ A                      # anisotropic, well-conditioned
    mu, W = cm.zca_whiten(X, eps=1e-6)
    Xw = (X - mu) @ W
    cov = np.cov(Xw, rowvar=False)
    assert np.allclose(np.diag(cov), 1.0, atol=0.05)            # unit variances
    off = cov - np.diag(np.diag(cov))
    assert np.abs(off).max() < 0.05                             # decorrelated


def test_fit_map_recovers_known_linear_relation():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 5))
    true_M = rng.standard_normal((5, 7)); bias = rng.standard_normal(7)
    Y = X @ true_M + bias
    M = cm.fit_map(X, Y, alpha=1e-6)
    assert np.allclose(cm.apply_map(M, X), Y, atol=1e-4)        # near-exact at tiny alpha


# ---- end-to-end transfer + determinism (T1 + T4) --------------------------------------------

def test_selftest_transfers_with_no_target_labels():
    out = cm.selftest(seed=0)
    assert out["reference_self_auroc"] == 1.0                   # axis reads its own model
    assert out["transported_auroc"] >= 0.90                     # reads the TARGET, no target labels
    assert out["map_val_r2"] > 0.9


def test_selftest_is_deterministic():
    assert cm.selftest(seed=0) == cm.selftest(seed=0)           # bit-identical, same seed


def test_identity_map_equals_in_model_score():
    rng = np.random.default_rng(3)
    states = rng.standard_normal((50, 8))
    labels = np.array([1, 0] * 25)
    axis = cm.fit_axis(states, labels, name="t")
    idm = cm.identity_map(8)
    assert np.allclose(cm.read(axis, states, state_map=idm), axis.score(states))
    assert np.allclose(cm.read(axis, states, state_map=None), axis.score(states))


def test_cross_model_read_recovers_labels():
    # two synthetic models sharing latent value structure; read target via a borrowed axis
    rng = np.random.default_rng(7)
    dlat, dref, dtgt = 6, 20, 24
    Aref = rng.standard_normal((dlat, dref)); Atgt = rng.standard_normal((dlat, dtgt))
    ax = rng.standard_normal(dlat); ax /= np.linalg.norm(ax)

    def emit(n, signed):
        z = rng.standard_normal((n, dlat))
        if signed is not None:
            z = z + signed[:, None] * 2.0 * ax[None, :]
        return z @ Atgt + 0.1 * rng.standard_normal((n, dtgt)), z @ Aref + 0.1 * rng.standard_normal((n, dref))

    lab = np.array([1, 0] * 30)
    vtgt, vref = emit(60, np.where(lab == 1, 1.0, -1.0))
    htgt, _ = emit(40, np.where(np.array([1, 0] * 20) == 1, 1.0, -1.0))
    hlab = np.array([1, 0] * 20)
    axis = cm.fit_axis(vref, lab, name="value")
    smap = cm.fit_state_map(vtgt, vref, seed=0)                 # paired, label-free
    coords = cm.read(axis, htgt, state_map=smap)               # NO labels touch the target
    assert cm.auroc(coords, hlab) >= 0.80


def test_zca_shrink_interpolates_to_identity():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((10, 10))
    X = rng.standard_normal((300, 10)) @ A                      # anisotropic
    # lam=1 shrinks the covariance fully to scaled identity -> whitening is uniform rescaling
    mu1, W1 = cm.zca_shrink(X, lam=1.0)
    assert np.allclose(W1, np.diag(np.diag(W1)), atol=1e-6)     # diagonal (no decorrelation rotation)
    # lam=0 (eps-regularized) genuinely decorrelates: whitened covariance is near-identity
    mu0, W0 = cm.zca_shrink(X, lam=0.0, eps=1e-6)
    cov = np.cov(((X - mu0) @ W0), rowvar=False)
    assert np.allclose(np.diag(cov), 1.0, atol=0.1)
    assert np.abs(cov - np.diag(np.diag(cov))).max() < 0.1


def test_read_cross_model_recovers_labels_no_target_labels():
    # mapped-space whitening read; mirrors the B29 cross-model pipeline on synthetic two models
    rng = np.random.default_rng(13)
    dlat, dref, dtgt = 6, 22, 26
    Aref = rng.standard_normal((dlat, dref)); Atgt = rng.standard_normal((dlat, dtgt))
    ax = rng.standard_normal(dlat); ax /= np.linalg.norm(ax)

    def emit(n, signed):
        z = rng.standard_normal((n, dlat))
        if signed is not None:
            z = z + signed[:, None] * 2.0 * ax[None, :]
        return z @ Atgt + 0.1 * rng.standard_normal((n, dtgt)), z @ Aref + 0.1 * rng.standard_normal((n, dref))

    anchor_tgt, anchor_ref = emit(120, None)                   # unlabeled anchors (define mapped dist)
    lab = np.array([1, 0] * 30)
    val_tgt, val_ref = emit(60, np.where(lab == 1, 1.0, -1.0))
    hlab = np.array([1, 0] * 20); hold_tgt, _ = emit(40, np.where(hlab == 1, 1.0, -1.0))
    smap = cm.fit_state_map(anchor_tgt, anchor_ref, seed=0)    # label-free
    coords = cm.read_cross_model(val_ref, lab, smap, hold_tgt, mapped_anchors=anchor_tgt)  # NO target labels
    assert cm.auroc(coords, hlab) >= 0.80
    # deterministic
    coords2 = cm.read_cross_model(val_ref, lab, smap, hold_tgt, mapped_anchors=anchor_tgt)
    assert np.allclose(coords, coords2)


def test_fit_axis_validation():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError):
        cm.fit_axis(rng.standard_normal((10, 4)), [1] * 10, name="x")     # single class
    with pytest.raises(ValueError):
        cm.fit_axis(rng.standard_normal(10), [1, 0] * 5, name="x")        # 1-D states
    with pytest.raises(ValueError):
        cm.fit_axis(rng.standard_normal((10, 4)), [1, 0], name="x")       # label/length mismatch


# ---- the read != write demarcation (always raises) ------------------------------------------

def test_steering_is_refused():
    for op in ("steering", "intervention"):
        with pytest.raises(PermissionError):
            cm.refused(op)


def test_content_danger_axis_is_refused():
    with pytest.raises(PermissionError):
        cm.refused("content_danger")


def test_unknown_op_raises_keyerror_not_a_number():
    with pytest.raises(KeyError):
        cm.refused("definitely_not_a_capability")


def test_refusals_are_documented():
    for k in ("steering", "intervention", "content_danger"):
        assert cm.REFUSALS[k]["status"] == "REFUSED"
        assert cm.REFUSALS[k]["receipt"]                        # has a receipt path


# ---- certificate ----------------------------------------------------------------------------

def test_certificate_shape_and_serializable():
    rng = np.random.default_rng(5)
    states = rng.standard_normal((40, 8)); labels = np.array([1, 0] * 20)
    axis = cm.fit_axis(states, labels, name="truth")
    smap = cm.identity_map(8)
    ev = cm.evaluate(axis.score(states), labels)
    cert = cm.crossmind_certificate(axis, reference_id="gemma-2-2b", target_id="llama-3.2-3b",
                                    evaluation=ev, state_map=smap)
    assert cert["instrument"] == "styxx.crossmind v0"
    assert len(cert["instrument_sha256"]) == 64                 # sha256 hex of the source
    assert set(cert["axes_refused"]) == set(cm.REFUSALS)        # always carries ALL refusals
    assert "READ-ONLY" in cert["scope"]
    assert cert["prereg"].endswith("PREREG_crossmind_v0_2026_06_12.md")
    json.dumps(cert)                                            # must be JSON-serializable


# ---- public surface + importability ---------------------------------------------------------

def test_module_importable_from_styxx():
    from styxx import crossmind  # noqa: F401


def test_public_surface_complete():
    for name in cm.__all__:
        assert hasattr(cm, name), f"__all__ lists {name} but it is not defined"

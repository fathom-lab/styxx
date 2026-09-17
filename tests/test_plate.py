# -*- coding: utf-8 -*-
"""styxx.plate / styxx.geoplate — the three properties the plate is allowed to claim, pinned.

A plate claims exactly three things and nothing else: the same digest always gives the same figure;
any changed hex character gives a different figure; and the geometry plate cannot make two matrices
look more alike than they are — a shuffled-item control separates from the original by a wide
margin while a mildly perturbed copy stays close. None of these tests render a PNG.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from styxx import plate

D1 = hashlib.sha256(b"the plate").hexdigest()
D2 = D1[:-1] + ("0" if D1[-1] != "0" else "1")   # one hex character changed, at the very end


def test_hash_plate_is_deterministic():
    _, _, U1, params1 = plate.field(D1, res=120)
    _, _, U2, params2 = plate.field(D1, res=120)
    assert np.array_equal(U1, U2)
    assert params1 == params2


def test_hash_plate_changes_when_any_hex_character_changes():
    _, _, U1, p1 = plate.field(D1, res=120)
    _, _, U2, p2 = plate.field(D2, res=120)
    # the last byte is re-hashed into every parameter, so the modes themselves must differ
    assert p1[0] != p2[0] or p1[1] != p2[1] or p1[2] != p2[2]
    r = np.corrcoef(U1.ravel(), U2.ravel())[0, 1]
    assert abs(r) < 0.9


def test_hash_plate_refuses_a_non_digest():
    with pytest.raises(SystemExit):
        plate.field("abc", res=20)


def _synthetic_rdm(seed: int, n: int = 60) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 16))
    X -= X.mean(0, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return 1.0 - X @ X.T


def test_geometry_plate_separates_a_shuffled_control_and_keeps_a_perturbed_copy_close():
    geoplate = pytest.importorskip("styxx.geoplate")
    pytest.importorskip("scipy")
    R = _synthetic_rdm(1)
    n = R.shape[0]
    perm = np.random.default_rng(2).permutation(n)
    R_shuffled = R[np.ix_(perm, perm)]
    R_perturbed = R + np.random.default_rng(3).normal(scale=0.01, size=R.shape)
    R_perturbed = (R_perturbed + R_perturbed.T) / 2
    U = geoplate.field(geoplate.coefficients(R), res=80).ravel()
    U_s = geoplate.field(geoplate.coefficients(R_shuffled), res=80).ravel()
    U_p = geoplate.field(geoplate.coefficients(R_perturbed), res=80).ravel()
    r_perturbed = np.corrcoef(U, U_p)[0, 1]
    r_shuffled = np.corrcoef(U, U_s)[0, 1]
    assert r_perturbed > 0.95
    assert r_perturbed - r_shuffled > 0.3


def test_geometry_plate_ignores_the_always_zero_diagonal():
    geoplate = pytest.importorskip("styxx.geoplate")
    pytest.importorskip("scipy")
    R = _synthetic_rdm(4)
    R2 = R.copy()
    np.fill_diagonal(R2, 5.0)          # a different diagonal must not change the figure
    assert np.allclose(geoplate.coefficients(R), geoplate.coefficients(R2))

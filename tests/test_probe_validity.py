# -*- coding: utf-8 -*-
"""Regression-lock for styxx.probe_validity — the probe-validation battery
(NOTE_probe_orthogonality_2026_06_24). Synthetic VALID vs SURFACE-ARTIFACT cases lock the verdict logic AND
demonstrate the tool is concept-agnostic (no dependency on the truth datasets).

A SURFACE-ARTIFACT is the failure mode the battery exists to catch: a probe with high in-construct AUC whose
direction is ORTHOGONAL to the concept's natural-data direction and does not transfer.
"""
from __future__ import annotations
import numpy as np
import pytest

from styxx.probe_validity import validate_probe, ProbeValidityReport


def _make(seed=0, dim=64, surface=False):
    """Build synthetic activations. Natural concept lives on direction u. The construct's label is driven by
    u (VALID) or by an orthogonal direction v (SURFACE-ARTIFACT)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(dim); u /= np.linalg.norm(u)
    v = rng.standard_normal(dim); v -= (v @ u) * u; v /= np.linalg.norm(v)  # v orthogonal to u
    acts = {}; crows = []; nrows = []
    S = 3.0  # signal strength
    # construct: 4 groups x 20, label balanced
    for i in range(80):
        y = i % 2; g = i // 20; t = f"c{i}"
        drive = v if surface else u
        acts[t] = (2 * y - 1) * S * drive + rng.standard_normal(dim)
        crows.append({"text": t, "label": y, "group": g})
    # natural: concept on u
    for i in range(40):
        y = i % 2; t = f"n{i}"
        acts[t] = (2 * y - 1) * S * u + rng.standard_normal(dim)
        nrows.append({"text": t, "label": y})
    return crows, nrows, (lambda texts: np.array([acts[t] for t in texts]))


def test_valid_probe_is_VALID():
    crows, nrows, get_acts = _make(surface=False)
    r = validate_probe(crows, nrows, get_acts, perm_iters=300)
    assert isinstance(r, ProbeValidityReport)
    assert r.in_construct_auc >= 0.8, r.summary()
    assert r.natural_axis_ceiling >= 0.75, r.summary()
    # construct direction aligns with the natural concept -> transfers + aligned
    assert r.orthogonality_cosine >= 0.5, r.summary()
    assert r.ood_transfer_p < 0.05, r.summary()
    assert r.verdict.startswith("VALID"), r.summary()


def test_surface_artifact_is_caught():
    crows, nrows, get_acts = _make(surface=True)
    r = validate_probe(crows, nrows, get_acts, perm_iters=300)
    # the hallmark: high in-construct AUC but orthogonal + non-transferring
    assert r.in_construct_auc >= 0.8, r.summary()           # looks great in-construct
    assert abs(r.orthogonality_cosine) < 0.5, r.summary()   # but orthogonal to the concept
    assert r.ood_transfer_p >= 0.05, r.summary()            # transfers no better than random
    assert r.verdict.startswith("SURFACE-ARTIFACT"), r.summary()
    # and the battery still confirms the concept axis EXISTS on natural data
    assert r.natural_axis_ceiling >= 0.75, r.summary()


def test_report_serializes():
    crows, nrows, get_acts = _make(surface=True)
    d = validate_probe(crows, nrows, get_acts, perm_iters=200).as_dict()
    for k in ("silence_auc", "in_construct_auc", "ood_transfer_auc", "ood_transfer_p",
              "orthogonality_cosine", "natural_axis_ceiling", "verdict"):
        assert k in d

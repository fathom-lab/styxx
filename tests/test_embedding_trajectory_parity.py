# -*- coding: utf-8 -*-
"""
test_embedding_trajectory_parity
================================

Contract guard: styxx.coherence.embedding_trajectory_alignment (the TOOL
primitive) must return numerically identical values to the locked
research scorer scripts/drift_axis_scorer.py::drift_axis_alignment
(commit 79906b4, §8-bound).

If this fails, the shipped tool no longer computes the quantity the
drift-axis preregistration scored (deposit fa24373). That is either a
methodology change (requires a new lock-hash) or a porting bug. Do not
silently delete.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_scorer():
    repo = Path(__file__).resolve().parent.parent
    path = repo / "scripts" / "drift_axis_scorer.py"
    spec = importlib.util.spec_from_file_location("drift_axis_scorer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["drift_axis_scorer"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_tool_primitive_matches_locked_scorer():
    from styxx.coherence import embedding_trajectory_alignment
    scorer = _load_scorer()
    rng = np.random.default_rng(2026)
    for trial in range(25):
        n = int(rng.integers(4, 30))
        d = 64
        a = rng.standard_normal((n, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = rng.standard_normal((n, d)); b /= np.linalg.norm(b, axis=1, keepdims=True)
        tool = embedding_trajectory_alignment(a, b)
        locked = scorer.drift_axis_alignment(a, b)
        if tool != tool and locked != locked:  # both NaN
            continue
        assert abs(tool - locked) < 1e-12, (
            f"trial {trial}: tool primitive diverged from locked scorer: "
            f"{tool!r} vs {locked!r}. See test docstring."
        )


def test_self_pair_one_negated_minus_one():
    from styxx.coherence import embedding_trajectory_alignment
    rng = np.random.default_rng(7)
    embs = rng.standard_normal((20, 32)); embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    assert abs(embedding_trajectory_alignment(embs, embs) - 1.0) < 1e-12
    assert abs(embedding_trajectory_alignment(embs, -embs) - (-1.0)) < 1e-12


def test_short_returns_nan():
    from styxx.coherence import embedding_trajectory_alignment
    assert np.isnan(embedding_trajectory_alignment(np.eye(3, 4), np.eye(3, 4)))


def test_exported_at_package_level():
    import styxx
    assert hasattr(styxx, "embedding_trajectory_alignment")

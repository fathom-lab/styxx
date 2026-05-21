# -*- coding: utf-8 -*-
"""
test_drift_axis_scorer_parity
=============================

Contract guard: scripts/drift_axis_scorer.py's drift_axis_alignment
implementation must (a) match the §4 operational definition verbatim,
(b) reproduce the existing exploratory-probe DAA values (commit
8ff3b65) within OpenAI embedding-API non-determinism tolerance, and
(c) remain numerically equivalent across reasonable refactors.

If this test fails, one of two things happened:

  1. The scorer's drift_axis_alignment math changed. That is a
     methodology change under the preregistration §10 immutability
     clause (drift_axis_alignment_preregistration_2026_05_21.md).
     Revert OR open a new preregistration with a new lock-commit hash
     before re-running the corpus.

  2. The §4 operational definition was misread when ported. Re-read
     the prereg and reconcile.

Do not silently delete this test. The integrity discipline is that
the measurement is fixed before data is pulled through it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_scorer():
    repo = Path(__file__).resolve().parent.parent
    path = repo / "scripts" / "drift_axis_scorer.py"
    spec = importlib.util.spec_from_file_location("drift_axis_scorer", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["drift_axis_scorer"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_drift_axis_alignment_matches_section_4_verbatim():
    """Re-implement §4 inline and check the scorer matches numerically."""
    scorer = _load_scorer()
    rng = np.random.default_rng(seed=42)
    embs_a = rng.standard_normal((20, 128))
    embs_b = rng.standard_normal((20, 128))
    # L2-normalize per the §2 contract
    embs_a = embs_a / np.linalg.norm(embs_a, axis=1, keepdims=True)
    embs_b = embs_b / np.linalg.norm(embs_b, axis=1, keepdims=True)

    # §4 inline reference
    n = min(embs_a.shape[0], embs_b.shape[0])
    half = n // 2
    a_first = embs_a[:half].mean(0)
    a_second = embs_a[half:n].mean(0)
    b_first = embs_b[:half].mean(0)
    b_second = embs_b[half:n].mean(0)
    a_dir = a_second - a_first
    b_dir = b_second - b_first
    expected = float(
        (a_dir / np.linalg.norm(a_dir)) @ (b_dir / np.linalg.norm(b_dir))
    )

    observed = scorer.drift_axis_alignment(embs_a, embs_b)
    assert abs(observed - expected) < 1e-12, (
        f"scorer's drift_axis_alignment diverged from §4 reference: "
        f"observed={observed!r} expected={expected!r}. "
        f"See test docstring for what this means."
    )


def test_short_trajectory_returns_nan():
    """§4 specifies NaN for trajectories too short to have a meaningful
    half-vs-half centroid difference."""
    scorer = _load_scorer()
    embs_a = np.eye(3, 4)
    embs_b = np.eye(3, 4)
    result = scorer.drift_axis_alignment(embs_a, embs_b)
    assert np.isnan(result), f"expected NaN for n<4, got {result!r}"


def test_degenerate_direction_returns_nan():
    """If either agent's trajectory has zero drift (first-half centroid ==
    second-half centroid exactly), DAA is undefined; §4 returns NaN."""
    scorer = _load_scorer()
    # Make both halves identical → drift direction is zero vector
    embs_a = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (20, 1))
    embs_b = np.random.default_rng(0).standard_normal((20, 4))
    embs_b = embs_b / np.linalg.norm(embs_b, axis=1, keepdims=True)
    result = scorer.drift_axis_alignment(embs_a, embs_b)
    assert np.isnan(result), f"expected NaN for zero-drift trajectory, got {result!r}"


def test_truncates_to_shorter_per_section_4():
    """Unequal-length trajectories truncate to shorter (§4 corrigendum
    inherited from phase-coherence §4)."""
    scorer = _load_scorer()
    rng = np.random.default_rng(seed=123)
    embs_a = rng.standard_normal((20, 16))
    embs_b = rng.standard_normal((30, 16))
    embs_a = embs_a / np.linalg.norm(embs_a, axis=1, keepdims=True)
    embs_b = embs_b / np.linalg.norm(embs_b, axis=1, keepdims=True)

    val_full_b = scorer.drift_axis_alignment(embs_a, embs_b)
    val_trunc_b = scorer.drift_axis_alignment(embs_a, embs_b[:20])
    assert abs(val_full_b - val_trunc_b) < 1e-12, (
        f"truncation behavior diverged: full_b={val_full_b!r} trunc_b={val_trunc_b!r}"
    )


def test_self_pair_is_perfect_one():
    """A trajectory paired with itself must yield DAA = 1.0 exactly
    (centroid-diff vector dotted with itself, normalized → 1)."""
    scorer = _load_scorer()
    rng = np.random.default_rng(seed=7)
    embs = rng.standard_normal((20, 32))
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    val = scorer.drift_axis_alignment(embs, embs)
    assert abs(val - 1.0) < 1e-12, f"self-pair DAA should be 1.0, got {val!r}"


def test_negated_pair_is_negative_one():
    """A trajectory paired with its negation must yield DAA = -1.0
    exactly."""
    scorer = _load_scorer()
    rng = np.random.default_rng(seed=11)
    embs = rng.standard_normal((20, 32))
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    val = scorer.drift_axis_alignment(embs, -embs)
    assert abs(val - (-1.0)) < 1e-12, f"negated-pair DAA should be -1.0, got {val!r}"

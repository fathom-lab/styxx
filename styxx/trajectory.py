# -*- coding: utf-8 -*-
"""
styxx.trajectory -- trajectory shape features (slope, curvature, volatility).

Expands the tier 0 feature vector from 12-dim (statistical summary) to
21-dim (statistical summary + trajectory shape).  Pure numpy, no external
dependencies.

The key insight: two prompts can have identical mean entropy but very
different entropy *trajectories*.  A reasoning response may show steadily
decreasing entropy (converging), while a hallucination may show oscillating
entropy (uncertain).  Summary statistics collapse this signal.  Shape
features preserve it.

Feature layout (9-dim, appended after the 12-dim legacy vector):
    [slope_entropy, curvature_entropy, volatility_entropy,
     slope_logprob, curvature_logprob, volatility_logprob,
     slope_top2_margin, curvature_top2_margin, volatility_top2_margin]

Graceful degradation for short windows:
    1 token:   all features = 0.0  (phase1 unchanged)
    2 tokens:  slope = x[1]-x[0], curvature = 0.0, volatility = |x[1]-x[0]|
    3+ tokens: full computation

Research: https://github.com/fathom-lab/fathom
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# ══════════════════════════════════════════════════════════════════
# Atomic shape functions
# ══════════════════════════════════════════════════════════════════

def slope(window: np.ndarray) -> float:
    """OLS linear regression coefficient for the trajectory window.

    Captures the *direction* of the signal over the token window.
    Positive slope = signal increasing over time.  Negative = decreasing.

    >>> slope(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    1.0
    >>> slope(np.array([5.0, 3.0, 1.0]))
    -2.0
    """
    n = len(window)
    if n <= 1:
        return 0.0
    if n == 2:
        return float(window[1] - window[0])
    # np.polyfit returns [slope, intercept] for degree 1
    return float(np.polyfit(np.arange(n, dtype=float), window, 1)[0])


def curvature(window: np.ndarray) -> float:
    """Mean absolute second-order finite difference.

    Captures *inflection* in the trajectory.  Linear trajectories have
    curvature 0.  Oscillating trajectories have high curvature.

    curvature = mean(|x[i+2] - 2*x[i+1] + x[i]|) for i in 0..n-3

    >>> curvature(np.array([1.0, 2.0, 3.0, 4.0]))  # linear
    0.0
    >>> curvature(np.array([1.0, 3.0, 1.0, 3.0]))  # oscillating
    4.0
    """
    n = len(window)
    if n < 3:
        return 0.0
    # second-order finite difference: d2[i] = x[i+2] - 2*x[i+1] + x[i]
    d2 = window[2:] - 2.0 * window[1:-1] + window[:-2]
    return float(np.abs(d2).mean())


def volatility(window: np.ndarray) -> float:
    """Mean absolute successive difference.

    Captures *jitter* in the trajectory, independent of overall trend.
    Smooth trajectories have low volatility.  Noisy trajectories have high.

    volatility = mean(|x[i+1] - x[i]|) for i in 0..n-2

    >>> volatility(np.array([1.0, 2.0, 3.0, 4.0]))  # smooth
    1.0
    >>> volatility(np.array([1.0, 5.0, 1.0, 5.0]))  # jagged
    4.0
    """
    n = len(window)
    if n < 2:
        return 0.0
    d1 = np.diff(window)
    return float(np.abs(d1).mean())


# ══════════════════════════════════════════════════════════════════
# Composite extractor
# ══════════════════════════════════════════════════════════════════

SHAPE_STATS = ["slope", "curvature", "volatility"]

_SHAPE_FNS = {
    "slope": slope,
    "curvature": curvature,
    "volatility": volatility,
}

# Must match vitals.TIER0_SIGNALS
_DEFAULT_SIGNALS = ("entropy", "logprob", "top2_margin")


def extract_shape_features(
    trajectories: Dict[str, Sequence[float]],
    n_tokens: int,
    signals: Sequence[str] = _DEFAULT_SIGNALS,
) -> np.ndarray:
    """Compute (slope, curvature, volatility) per signal over [0, n_tokens).

    Returns a 9-dim numpy array in the order:
        [slope_entropy, curvature_entropy, volatility_entropy,
         slope_logprob, curvature_logprob, volatility_logprob,
         slope_top2_margin, curvature_top2_margin, volatility_top2_margin]

    NaN/Inf are sanitized to 0.0, matching the convention in
    vitals.extract_features().
    """
    feats: list[float] = []
    for signal in signals:
        raw = trajectories.get(signal, [])
        if raw is None or len(raw) == 0:
            feats.extend([0.0, 0.0, 0.0])
            continue
        window = np.asarray(list(raw)[:n_tokens], dtype=float)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        if len(window) == 0:
            feats.extend([0.0, 0.0, 0.0])
            continue
        for stat_name in SHAPE_STATS:
            feats.append(_SHAPE_FNS[stat_name](window))
    return np.asarray(feats, dtype=float)

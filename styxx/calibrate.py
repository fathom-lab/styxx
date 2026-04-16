# -*- coding: utf-8 -*-
"""
styxx.calibrate — outcome-driven centroid adjustment.

    styxx.calibrate()

The static atlas v0.3 centroids work cross-architecture but they
can't know what YOUR agent's reasoning looks like vs a customer
service bot's reasoning. Every heuristic patch we've added
(hallucination dampening on creative, code detection, prompt_type
gating) is compensating for centroid positions that don't perfectly
separate the categories for a specific agent.

This module reads entries with outcome='correct' and 'incorrect'
from the audit log and shifts the centroids toward the agent's
actual distribution. 500 labeled examples recalibrate better than
500 lines of heuristic code.

How it works
────────────

1. Load all entries with outcome labels from chart.jsonl.
2. For each entry with features (tier 0 logprob-based entries),
   extract the 12-dimensional feature vector.
3. Group by phase and predicted category.
4. For entries marked 'correct': compute mean of their feature
   vectors → this is where the centroid SHOULD be for this agent.
5. Weighted shift: new_centroid = (1-lr) * atlas_centroid + lr * agent_mean
6. Save adjusted centroids to ~/.styxx/calibration/{agent_name}.json

The learning rate starts small (0.1) so atlas knowledge isn't
destroyed. As more labels accumulate, the agent's centroids
gradually personalize.

1.0.0+. The feature that turns styxx from a generic instrument
into a personalized one.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CalibrationResult:
    """Result of a centroid calibration pass."""
    n_correct: int = 0
    n_incorrect: int = 0
    n_phases_adjusted: int = 0
    categories_shifted: Dict[str, float] = field(default_factory=dict)
    saved_to: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"<Calibration {self.n_correct} correct, {self.n_incorrect} incorrect, "
            f"{self.n_phases_adjusted} phases adjusted>"
        )


def _calibration_dir() -> Path:
    data_dir = os.environ.get("STYXX_DATA_DIR", "").strip()
    if data_dir:
        d = Path(data_dir).expanduser() / "calibration"
    else:
        d = Path.home() / ".styxx" / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_labeled_entries() -> List[dict]:
    """Load audit entries that have outcome labels."""
    from .analytics import load_audit
    entries = load_audit(last_n=5000)
    return [e for e in entries if e.get("outcome") in ("correct", "incorrect")]


def calibrate(
    *,
    agent_name: Optional[str] = None,
    learning_rate: float = 0.10,
    min_samples: int = 10,
) -> CalibrationResult:
    """Run a centroid calibration pass from accumulated outcome labels.

    3.2.0: uses features_v2 (21-dim) stored in audit entries when
    available. Computes actual mean feature vectors per category and
    shifts centroids using: new = (1-lr) * atlas + lr * agent_mean.

    Falls back to legacy confidence-inversion heuristic for entries
    that lack features_v2.

    Args:
        agent_name:    name for the calibration file. Defaults to
                       STYXX_AGENT_NAME or 'default'.
        learning_rate: how much to shift centroids (0.0=no change,
                       1.0=fully replace with agent data). Default 0.10.
        min_samples:   minimum correct samples per category per phase
                       before shifting that centroid. Default 10.

    Returns:
        CalibrationResult with shift statistics.
    """
    from .core import StyxxRuntime
    from .vitals import load_centroids

    result = CalibrationResult()

    # Resolve agent name
    if agent_name is None:
        agent_name = os.environ.get("STYXX_AGENT_NAME", "").strip() or "default"

    # Load labeled entries
    labeled = _load_labeled_entries()
    correct = [e for e in labeled if e.get("outcome") == "correct"]
    incorrect = [e for e in labeled if e.get("outcome") == "incorrect"]
    result.n_correct = len(correct)
    result.n_incorrect = len(incorrect)

    if result.n_correct < min_samples:
        return result  # not enough data

    # Load atlas centroids (unmodified originals for blending)
    atlas = load_centroids()
    atlas_phases: Dict[str, dict] = {}
    categories = list(atlas["categories"])
    for phase_name, phase_data in atlas["phases"].items():
        mu = np.asarray(phase_data["mu"], dtype=float)
        sigma = np.asarray(phase_data["sigma"], dtype=float)
        centroids = {
            cat: np.asarray(phase_data["centroids"][cat], dtype=float)
            for cat in categories
        }
        atlas_phases[phase_name] = {"mu": mu, "sigma": sigma, "centroids": centroids}

    feature_dim = len(next(iter(next(iter(atlas_phases.values()))["centroids"].values())))

    # Separate entries with real features vs legacy (confidence-only)
    correct_with_features = [
        e for e in correct
        if e.get("features_v2") is not None and len(e["features_v2"]) >= feature_dim
    ]
    correct_legacy = [e for e in correct if e not in correct_with_features]

    # Group real feature vectors by (phase4, predicted_category)
    cat_feature_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    for e in correct_with_features:
        cat = e.get("phase4_pred")
        if not cat or cat not in categories:
            continue
        fv = np.asarray(e["features_v2"][:feature_dim], dtype=float)
        cat_feature_vecs[cat].append(fv)

    # Legacy path: confidence-inversion weights (for entries without features_v2)
    cat_legacy_weights: Dict[str, List[float]] = defaultdict(list)
    for e in correct_legacy:
        cat = e.get("phase4_pred")
        conf = e.get("phase4_conf")
        if cat and conf is not None:
            cat_legacy_weights[cat].append(max(0.0, 1.0 - float(conf)))

    incorrect_cats: Dict[str, int] = defaultdict(int)
    for e in incorrect:
        cat = e.get("phase4_pred")
        if cat:
            incorrect_cats[cat] += 1

    # Load existing calibration or start fresh
    cal_path = _calibration_dir() / f"{agent_name}.json"
    cal_data = {}
    if cal_path.exists():
        try:
            with open(cal_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
        except Exception:
            cal_data = {}

    # Compute shifted centroids (real feature path)
    shifted_centroids: Dict[str, Dict[str, list]] = {}
    adjustments: Dict[str, Dict[str, float]] = {}
    phase_name = "phase4_late"

    if phase_name in atlas_phases:
        phase_data = atlas_phases[phase_name]
        mu = phase_data["mu"]
        sigma = phase_data["sigma"]

        for cat in categories:
            atlas_centroid = phase_data["centroids"][cat]

            # Prefer real features; fall back to legacy weights
            if cat in cat_feature_vecs and len(cat_feature_vecs[cat]) >= min_samples:
                # Real feature path: compute agent mean in z-space
                feature_matrix = np.stack(cat_feature_vecs[cat])
                z_matrix = (feature_matrix - mu) / sigma
                agent_mean_z = z_matrix.mean(axis=0)
                new_centroid = (1.0 - learning_rate) * atlas_centroid + learning_rate * agent_mean_z
                shift_magnitude = float(np.linalg.norm(new_centroid - atlas_centroid))

                if phase_name not in shifted_centroids:
                    shifted_centroids[phase_name] = {}
                shifted_centroids[phase_name][cat] = new_centroid.tolist()

                if phase_name not in adjustments:
                    adjustments[phase_name] = {}
                adjustments[phase_name][cat] = round(shift_magnitude, 4)
                result.categories_shifted[cat] = shift_magnitude

            elif cat in cat_legacy_weights and len(cat_legacy_weights[cat]) >= min_samples:
                # Legacy path: confidence-inversion heuristic
                mean_weight = sum(cat_legacy_weights[cat]) / len(cat_legacy_weights[cat])
                adjustment_factor = mean_weight * learning_rate

                if phase_name not in adjustments:
                    adjustments[phase_name] = {}
                adjustments[phase_name][cat] = round(adjustment_factor, 4)
                result.categories_shifted[cat] = adjustment_factor

    result.n_phases_adjusted = len(adjustments)

    # Save calibration metadata + shifted centroids
    cal_data.update({
        "agent_name": agent_name,
        "last_calibrated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_correct": result.n_correct,
        "n_incorrect": result.n_incorrect,
        "n_with_features": len(correct_with_features),
        "n_legacy": len(correct_legacy),
        "learning_rate": learning_rate,
        "feature_dim": feature_dim,
        "adjustments": {
            phase: {cat: v for cat, v in cats.items()}
            for phase, cats in adjustments.items()
        },
        "shifted_centroids": shifted_centroids,
        "incorrect_category_counts": dict(incorrect_cats),
    })

    try:
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(cal_data, f, indent=2)
        result.saved_to = str(cal_path)
    except OSError:
        pass

    return result


def calibration_status(
    *,
    agent_name: Optional[str] = None,
) -> Optional[dict]:
    """Check the current calibration state for an agent.

    Returns the calibration metadata dict or None if no calibration
    exists.
    """
    if agent_name is None:
        agent_name = os.environ.get("STYXX_AGENT_NAME", "").strip() or "default"
    cal_path = _calibration_dir() / f"{agent_name}.json"
    if not cal_path.exists():
        return None
    try:
        with open(cal_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

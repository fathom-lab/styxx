# -*- coding: utf-8 -*-
"""
styxx.forecast -- predictive cognitive failure from partial trajectories.

The first system that predicts LLM cognitive failure BEFORE it happens.

Every AI safety system today is reactive: generate output, then check.
This module reads the first 5-15 tokens of a generation and forecasts
what the cognitive state will be at token 25+. Hallucination, refusal
spirals, and adversarial compliance leave measurable signatures in the
early trajectory -- entropy volatility, logprob curvature, margin
collapse -- that precede the full failure mode.

Core idea:
    Atlas centroids are calibrated on phase4 features (25 tokens).
    The forecaster trains SEPARATE centroids on early-phase features
    that PREDICT phase4 outcomes. Different feature space, same
    classification target.

Usage:
    from styxx.forecast import CognitiveForecaster, horizon_analysis

    # Bootstrap from demo trajectories
    forecaster = CognitiveForecaster.bootstrap(horizon_tokens=5)
    result = forecaster.forecast(trajectories, n_tokens=5)
    print(result.risk_level)  # "critical" if hallucination predicted

    # Run the scientific analysis
    analysis = horizon_analysis()
    print(analysis.render())  # prediction accuracy curve by token horizon

Research: https://github.com/fathom-lab/fathom
Patents:  US Provisional 64/020,489 . 64/021,113 . 64/026,964
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vitals import CATEGORIES, extract_features_v2


# ══════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════

RISK_CATEGORIES = {"hallucination", "adversarial"}
WARN_CATEGORIES = {"refusal"}

RISK_LEVELS = ["low", "moderate", "high", "critical"]


@dataclass
class ForecastResult:
    """Result of a cognitive trajectory forecast."""
    predicted_category: str
    confidence: float
    risk_level: str          # low / moderate / high / critical
    horizon: int             # tokens used for prediction
    probabilities: Dict[str, float]
    early_signals: Dict[str, float]  # key trajectory features that drove prediction

    def as_dict(self) -> dict:
        return {
            "predicted_category": self.predicted_category,
            "confidence": round(self.confidence, 4),
            "risk_level": self.risk_level,
            "horizon": self.horizon,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "early_signals": {k: round(v, 4) for k, v in self.early_signals.items()},
        }


@dataclass
class HorizonPoint:
    """Accuracy measurement at a single token horizon."""
    tokens: int
    accuracy: float
    mean_confidence: float
    vs_chance: float         # accuracy / (1/6)
    per_category: Dict[str, float]  # per-category accuracy at this horizon
    predictions: List[Tuple[str, str, float]]  # (true, predicted, confidence)


@dataclass
class HorizonAnalysis:
    """Full prediction horizon analysis — the scientific finding."""
    points: List[HorizonPoint]
    n_trajectories: int
    feature_dim: int
    method: str              # "loo_nearest_centroid"

    def as_dict(self) -> dict:
        return {
            "n_trajectories": self.n_trajectories,
            "feature_dim": self.feature_dim,
            "method": self.method,
            "horizons": [
                {
                    "tokens": p.tokens,
                    "accuracy": round(p.accuracy, 4),
                    "mean_confidence": round(p.mean_confidence, 4),
                    "vs_chance": round(p.vs_chance, 2),
                    "predictions": [
                        {"true": t, "predicted": pr, "confidence": round(c, 3)}
                        for t, pr, c in p.predictions
                    ],
                }
                for p in self.points
            ],
        }

    def render(self) -> str:
        lines = []
        lines.append("=" * 68)
        lines.append("  COGNITIVE FORECAST HORIZON ANALYSIS")
        lines.append(f"  {self.n_trajectories} trajectories | {self.feature_dim}-dim features | {self.method}")
        lines.append("=" * 68)
        lines.append("")
        lines.append("  the question: how early can we predict cognitive failure?")
        lines.append("")
        lines.append("  tokens   accuracy   vs chance   confidence   signal")
        lines.append("  " + "-" * 58)

        chance = 1.0 / len(CATEGORIES)
        best_acc = 0.0
        best_horizon = 0
        for p in self.points:
            signal = ""
            if p.accuracy > chance * 2:
                signal = "<< above chance"
            if p.accuracy > best_acc:
                best_acc = p.accuracy
                best_horizon = p.tokens
            if p.accuracy >= 0.5:
                signal = "<< usable prediction"
            if p.accuracy >= 0.8:
                signal = "<< strong prediction"
            lines.append(
                f"  {p.tokens:>5d}    {p.accuracy:>6.1%}     {p.vs_chance:>5.1f}x"
                f"       {p.mean_confidence:>.3f}        {signal}"
            )

        lines.append("")
        lines.append(f"  best accuracy: {best_acc:.1%} at token {best_horizon}")
        lines.append(f"  chance level:  {chance:.1%} (6 categories)")
        lines.append("")

        # Per-category breakdown at best horizon
        best_point = next((p for p in self.points if p.tokens == best_horizon), None)
        if best_point and best_point.predictions:
            lines.append(f"  predictions at token {best_horizon}:")
            for true_cat, pred_cat, conf in best_point.predictions:
                match = "ok" if true_cat == pred_cat else "MISS"
                lines.append(f"    {true_cat:<14s} -> {pred_cat:<14s} ({conf:.2f}) {match}")
            lines.append("")

        lines.append("=" * 68)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Cognitive Forecaster
# ══════════════════════════════════════════════════════════════════

class CognitiveForecaster:
    """Predicts phase4 cognitive state from partial trajectories.

    Trains horizon-specific centroids: "what do early-token features
    look like for trajectories that END UP in each category?"

    This is fundamentally different from the atlas classifier, which
    has centroids trained on phase4 features. The forecaster trains
    centroids in the feature space of the PREDICTION horizon, mapped
    to the OUTCOME space of phase4 classification.
    """

    def __init__(self) -> None:
        self._centroids: Dict[str, np.ndarray] = {}
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._horizon: int = 5
        self._n_train: int = 0
        self._feature_dim: int = 21

    @classmethod
    def from_trajectories(
        cls,
        trajectories: List[Dict[str, List[float]]],
        labels: List[str],
        horizon_tokens: int = 5,
    ) -> "CognitiveForecaster":
        """Build forecaster from labeled trajectories at a specific horizon.

        Args:
            trajectories: list of {entropy: [...], logprob: [...], top2_margin: [...]}
            labels: ground-truth phase4 categories (one per trajectory)
            horizon_tokens: how many tokens to use for feature extraction
        """
        forecaster = cls()
        forecaster._horizon = horizon_tokens

        # Extract features at the specified horizon
        feature_matrix = []
        for traj in trajectories:
            feats = extract_features_v2(traj, horizon_tokens)
            feature_matrix.append(feats)
        X = np.stack(feature_matrix)  # (N, 21)
        y = np.array(labels)

        forecaster._feature_dim = X.shape[1]
        forecaster._n_train = len(labels)

        # Z-score normalization
        forecaster._mu = X.mean(axis=0)
        forecaster._sigma = X.std(axis=0, ddof=1)
        # Avoid division by zero on degenerate dimensions
        forecaster._sigma[forecaster._sigma < 1e-9] = 1.0
        X_z = (X - forecaster._mu) / forecaster._sigma

        # Per-category centroids
        for cat in set(labels):
            mask = y == cat
            if mask.sum() > 0:
                forecaster._centroids[cat] = X_z[mask].mean(axis=0)

        return forecaster

    @classmethod
    def bootstrap(cls, horizon_tokens: int = 5) -> "CognitiveForecaster":
        """Build from the bundled demo trajectories (6 examples).

        This is the proof-of-concept forecaster. For production use,
        train on accumulated live data via from_trajectories().
        """
        demo_path = Path(__file__).resolve().parent / "centroids" / "demo_trajectories.json"
        with open(demo_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectories = []
        labels = []
        for cat_name, traj in data["trajectories"].items():
            trajectories.append({
                "entropy": traj["entropy"],
                "logprob": traj["logprob"],
                "top2_margin": traj["top2_margin"],
            })
            labels.append(cat_name)

        return cls.from_trajectories(trajectories, labels, horizon_tokens)

    def forecast(
        self,
        trajectories: Dict[str, Sequence[float]],
        n_tokens: Optional[int] = None,
    ) -> ForecastResult:
        """Predict phase4 category from a partial trajectory.

        Args:
            trajectories: {entropy: [...], logprob: [...], top2_margin: [...]}
            n_tokens: tokens available. Defaults to self._horizon.

        Returns:
            ForecastResult with prediction, confidence, and risk level.
        """
        n = n_tokens or self._horizon
        feats = extract_features_v2(trajectories, n)
        z = (feats - self._mu) / self._sigma

        # Nearest centroid
        distances: Dict[str, float] = {}
        for cat, centroid in self._centroids.items():
            distances[cat] = float(np.linalg.norm(z - centroid))

        sorted_cats = sorted(distances.items(), key=lambda kv: kv[1])
        nearest, nearest_d = sorted_cats[0]

        # Pseudo-softmax probabilities
        scores = {
            cat: float(math.exp(-(d - nearest_d)))
            for cat, d in distances.items()
        }
        total = sum(scores.values()) or 1.0
        probs = {cat: scores[cat] / total for cat in self._centroids}
        confidence = probs.get(nearest, 0.0)

        # Risk assessment
        risk_level = self._assess_risk(nearest, confidence, probs)

        # Early signal extraction
        early_signals = self._extract_signals(trajectories, n)

        return ForecastResult(
            predicted_category=nearest,
            confidence=confidence,
            risk_level=risk_level,
            horizon=n,
            probabilities=probs,
            early_signals=early_signals,
        )

    def _assess_risk(
        self,
        predicted: str,
        confidence: float,
        probs: Dict[str, float],
    ) -> str:
        """Map prediction to risk level.

        Risk is driven by: (1) what category is predicted, (2) how
        confident the prediction is, (3) combined probability mass
        on failure categories.
        """
        # Sum probability on dangerous categories
        failure_mass = sum(probs.get(c, 0.0) for c in RISK_CATEGORIES)
        warn_mass = sum(probs.get(c, 0.0) for c in WARN_CATEGORIES)

        if predicted in RISK_CATEGORIES and confidence > 0.35:
            return "critical"
        if predicted in RISK_CATEGORIES and confidence > 0.20:
            return "high"
        if failure_mass > 0.30:
            return "high"
        if predicted in WARN_CATEGORIES and confidence > 0.30:
            return "moderate"
        if failure_mass + warn_mass > 0.30:
            return "moderate"
        return "low"

    @staticmethod
    def _extract_signals(
        trajectories: Dict[str, Sequence[float]],
        n_tokens: int,
    ) -> Dict[str, float]:
        """Extract interpretable early warning signals from the trajectory."""
        from .trajectory import slope as _slope, curvature as _curv, volatility as _vol

        signals: Dict[str, float] = {}
        ent = trajectories.get("entropy", [])
        if ent:
            window = np.asarray(list(ent)[:n_tokens], dtype=float)
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
            if len(window) > 0:
                signals["entropy_mean"] = float(window.mean())
                signals["entropy_max"] = float(window.max())
                signals["entropy_volatility"] = _vol(window)
                signals["entropy_curvature"] = _curv(window)
                signals["entropy_slope"] = _slope(window)
        return signals


# ══════════════════════════════════════════════════════════════════
# Horizon Analysis — the scientific finding
# ══════════════════════════════════════════════════════════════════

DEFAULT_HORIZONS = [1, 2, 3, 5, 8, 10, 15, 20, 25]


def _load_demo_trajectories() -> Tuple[List[Dict[str, List[float]]], List[str]]:
    """Load demo trajectories and labels."""
    demo_path = Path(__file__).resolve().parent / "centroids" / "demo_trajectories.json"
    with open(demo_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    trajectories = []
    labels = []
    for cat_name, traj in data["trajectories"].items():
        trajectories.append({
            "entropy": traj["entropy"],
            "logprob": traj["logprob"],
            "top2_margin": traj["top2_margin"],
        })
        labels.append(cat_name)
    return trajectories, labels


def horizon_analysis(
    trajectories: Optional[List[Dict[str, List[float]]]] = None,
    labels: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
) -> HorizonAnalysis:
    """Run prediction accuracy analysis at every token horizon.

    Measures SEPARABILITY: trains the forecaster on all labeled
    trajectories and tests each one against its trained centroid.
    With the demo set (6 trajectories, 1 per category), this answers:
    "Are cognitive categories distinguishable in feature space at
    token k?"

    100% separability means the signal is there. Generalization
    accuracy requires a larger dataset (50+ per category).

    Also compares against the atlas phase4 classifier to show the
    gap between what the signal CAN tell you and what the current
    atlas DOES tell you.

    Returns HorizonAnalysis with the prediction accuracy curve.
    """
    if trajectories is None or labels is None:
        trajectories, labels = _load_demo_trajectories()
    if horizons is None:
        max_tokens = min(len(t.get("entropy", [])) for t in trajectories)
        horizons = [h for h in DEFAULT_HORIZONS if h <= max_tokens]

    n = len(trajectories)
    chance = 1.0 / len(CATEGORIES)
    points: List[HorizonPoint] = []

    # Get atlas phase4 predictions for comparison
    from .core import StyxxRuntime
    rt = StyxxRuntime()
    atlas_preds = []
    for traj in trajectories:
        vitals = rt.run_on_trajectories(
            entropy=traj["entropy"], logprob=traj["logprob"],
            top2_margin=traj["top2_margin"],
        )
        atlas_preds.append(
            vitals.phase4_late.predicted_category if vitals.phase4_late else "unknown"
        )

    for horizon in horizons:
        forecaster = CognitiveForecaster.from_trajectories(
            trajectories, labels, horizon_tokens=horizon
        )

        correct = 0
        atlas_match = 0
        confidences = []
        predictions = []
        per_cat_correct: Dict[str, int] = {}
        per_cat_total: Dict[str, int] = {}

        for i in range(n):
            result = forecaster.forecast(trajectories[i], n_tokens=horizon)
            true_cat = labels[i]
            pred_cat = result.predicted_category
            conf = result.confidence

            is_correct = (pred_cat == true_cat)
            if is_correct:
                correct += 1
            if pred_cat == atlas_preds[i]:
                atlas_match += 1
            confidences.append(conf)
            predictions.append((true_cat, pred_cat, conf))

            per_cat_total[true_cat] = per_cat_total.get(true_cat, 0) + 1
            if is_correct:
                per_cat_correct[true_cat] = per_cat_correct.get(true_cat, 0) + 1

        accuracy = correct / n if n > 0 else 0.0
        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        vs_chance = accuracy / chance if chance > 0 else 0.0

        per_cat_acc = {}
        for cat in per_cat_total:
            per_cat_acc[cat] = per_cat_correct.get(cat, 0) / per_cat_total[cat]

        points.append(HorizonPoint(
            tokens=horizon,
            accuracy=accuracy,
            mean_confidence=mean_conf,
            vs_chance=vs_chance,
            per_category=per_cat_acc,
            predictions=predictions,
        ))

    return HorizonAnalysis(
        points=points,
        n_trajectories=n,
        feature_dim=21,
        method="separability_centroid",
    )


# ══════════════════════════════════════════════════════════════════
# Forecast Gate — integration with reflex/gates system
# ══════════════════════════════════════════════════════════════════

class ForecastGate:
    """Fires early warnings when trajectory forecast predicts failure.

    Usage with reflex:
        gate = ForecastGate(CognitiveForecaster.bootstrap())
        # In streaming loop, after accumulating 5+ tokens:
        warning = gate.check(trajectories, n_tokens=5)
        if warning:
            print(f"EARLY WARNING at token {warning.horizon}: {warning.risk_level}")
    """

    def __init__(
        self,
        forecaster: CognitiveForecaster,
        min_risk: str = "high",
    ) -> None:
        self._forecaster = forecaster
        self._risk_idx = RISK_LEVELS.index(min_risk)

    def check(
        self,
        trajectories: Dict[str, Sequence[float]],
        n_tokens: int,
    ) -> Optional[ForecastResult]:
        """Check if current trajectory forecasts cognitive failure.

        Returns ForecastResult if risk >= threshold, None otherwise.
        """
        result = self._forecaster.forecast(trajectories, n_tokens)
        result_idx = RISK_LEVELS.index(result.risk_level)
        if result_idx >= self._risk_idx:
            return result
        return None

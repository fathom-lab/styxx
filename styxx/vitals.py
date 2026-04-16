# -*- coding: utf-8 -*-
"""
styxx.vitals — feature extraction + nearest-centroid classifier

Loads the atlas v0.3 centroid artifact (sha256 pinned) and exposes the
classifier that turns raw logprob trajectories into cognitive state
readings. This is the scientific core of tier 0.

No heavy dependencies — numpy only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════
# Constants — locked from generate_centroids.py
# ══════════════════════════════════════════════════════════════════

CATEGORIES = [
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
]

TIER0_SIGNALS = ["entropy", "logprob", "top2_margin"]

PHASE_TOKEN_CUTOFFS = {
    "phase1_preflight":  1,
    "phase2_early":      5,
    "phase3_mid":       15,
    "phase4_late":      25,
}

PHASE_ORDER = ["phase1_preflight", "phase2_early", "phase3_mid", "phase4_late"]

# sha256 of the shipped centroid file. Pinned for reproducibility.
# If this ever mismatches the actual file, styxx refuses to import —
# it means the calibration data has been tampered with or corrupted.
EXPECTED_CENTROIDS_SHA256 = (
    "502313c2e7c160df205f24d5457bb57b8a5e1846ff4afe898db0f20d491d0beb"
)


# ══════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════

def extract_features(
    trajectories: Dict[str, Sequence[float]],
    n_tokens: int,
) -> np.ndarray:
    """Build the (mean, std, min, max) × (entropy, logprob, top2_margin)
    feature vector over tokens [0, n_tokens).

    Returns a 12-dim numpy array in the order locked at calibration time.
    """
    feats: List[float] = []
    for signal in TIER0_SIGNALS:
        raw = trajectories.get(signal, [])
        if raw is None or len(raw) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        window = np.asarray(list(raw)[:n_tokens], dtype=float)
        # Sanitise: replace NaN/Inf with 0.0 so they don't poison
        # downstream z-scores and distance computations.
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        if len(window) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        feats.append(float(window.mean()))
        feats.append(float(window.std(ddof=1)) if len(window) > 1 else 0.0)
        feats.append(float(window.min()))
        feats.append(float(window.max()))
    return np.asarray(feats, dtype=float)


def extract_features_v2(
    trajectories: Dict[str, Sequence[float]],
    n_tokens: int,
) -> np.ndarray:
    """Build the 21-dim extended feature vector.

    Concatenates the legacy 12-dim vector (mean, std, min, max) x 3 signals
    with the 9-dim shape vector (slope, curvature, volatility) x 3 signals
    from styxx.trajectory.

    Falls back to 12-dim + 9 zeros if shape extraction fails (fail-open).
    """
    legacy = extract_features(trajectories, n_tokens)
    try:
        from .trajectory import extract_shape_features
        shape = extract_shape_features(trajectories, n_tokens)
    except Exception:
        shape = np.zeros(9, dtype=float)
    return np.concatenate([legacy, shape])


# ══════════════════════════════════════════════════════════════════
# Confidence calibration (isotonic regression, numpy-only)
# ══════════════════════════════════════════════════════════════════

class ConfidenceCalibrator:
    """Isotonic regression calibrator for pseudo-softmax confidence.

    Maps distance-to-nearest-centroid -> calibrated probability using
    a Pool Adjacent Violators (PAVA) fit on (distance, was_correct)
    pairs.  Pure numpy, no sklearn.

    Falls back to None (caller uses pseudo-softmax) when no fit data
    is available.
    """

    def __init__(self) -> None:
        self._thresholds: Optional[np.ndarray] = None
        self._values: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._thresholds is not None

    def fit(
        self,
        distances: Sequence[float],
        outcomes: Sequence[bool],
    ) -> None:
        """Fit the calibrator on (distance, was_correct) pairs.

        Shorter distance should map to higher correctness probability,
        so we fit an antitonic (non-increasing) regression.
        """
        d = np.asarray(distances, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        if len(d) < 5:
            return
        order = np.argsort(d)
        d_sorted = d[order]
        y_sorted = y[order]
        fitted = self._pava_antitonic(y_sorted)
        # clamp to [0.01, 0.99]
        fitted = np.clip(fitted, 0.01, 0.99)
        self._thresholds = d_sorted
        self._values = fitted

    def calibrate(self, distance: float) -> Optional[float]:
        """Map a distance to a calibrated probability.

        Returns None if the calibrator hasn't been fitted.
        """
        if self._thresholds is None or self._values is None:
            return None
        idx = int(np.searchsorted(self._thresholds, distance))
        idx = min(idx, len(self._values) - 1)
        return float(self._values[idx])

    @staticmethod
    def _pava_antitonic(y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators for antitonic (non-increasing) fit."""
        n = len(y)
        result = y.copy().astype(float)
        weight = np.ones(n, dtype=float)
        block_end = list(range(n))
        i = 0
        while i < n - 1:
            j = block_end[i]
            k = j + 1 if j + 1 < n else j
            if k >= n:
                break
            if result[i] < result[k]:
                # violation: pool
                total_w = weight[i] + weight[k]
                result[i] = (weight[i] * result[i] + weight[k] * result[k]) / total_w
                weight[i] = total_w
                result[k] = result[i]
                weight[k] = total_w
                block_end[i] = block_end[k]
                if i > 0:
                    i -= 1
                else:
                    i = block_end[i] + 1 if block_end[i] + 1 < n else n
            else:
                i = block_end[i] + 1 if block_end[i] + 1 < n else n
        # propagate block values
        i = 0
        while i < n:
            end = block_end[i]
            val = result[i]
            for j in range(i, end + 1):
                result[j] = val
            i = end + 1
        return result

    def save(self, path: Path) -> None:
        """Save calibrator state as JSON."""
        data = {}
        if self._thresholds is not None:
            data["thresholds"] = self._thresholds.tolist()
            data["values"] = self._values.tolist()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["ConfidenceCalibrator"]:
        """Load a saved calibrator. Returns None if file doesn't exist."""
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cal = cls()
            if "thresholds" in data and "values" in data:
                cal._thresholds = np.asarray(data["thresholds"], dtype=float)
                cal._values = np.asarray(data["values"], dtype=float)
            return cal
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════
# Centroid loader (sha-verified)
# ══════════════════════════════════════════════════════════════════

def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_centroids_path() -> Path:
    return Path(__file__).resolve().parent / "centroids" / "atlas_v0.3.json"


def load_centroids(path: Optional[Path] = None,
                   verify_sha: bool = True) -> dict:
    """Load the atlas centroid artifact, verifying sha256 by default.

    Raises FileNotFoundError if the centroid file is missing.
    Raises ValueError if the sha256 does not match EXPECTED_CENTROIDS_SHA256.

    STYXX_SKIP_SHA=1 in the environment will skip the sha check.
    Intended only for local dev — never set this in production.
    """
    path = Path(path) if path is not None else _default_centroids_path()
    if not path.exists():
        raise FileNotFoundError(
            "\n"
            "  styxx: calibration centroids not found.\n"
            f"  expected location: {path}\n"
            "  this file is shipped inside the styxx package and should\n"
            "  be installed automatically with pip. if it's missing you\n"
            "  likely have a broken install. try:\n"
            "    pip install --force-reinstall styxx\n"
            "  or if working from source:\n"
            "    python scripts/generate_centroids.py \\\n"
            "      --captures <atlas/captures> \\\n"
            "      --out styxx/centroids/atlas_v0.3.json\n"
        )

    # STYXX_SKIP_SHA dev escape hatch
    import os
    skip_sha = os.environ.get("STYXX_SKIP_SHA", "").strip().lower() in (
        "1", "true", "yes", "on"
    )

    if verify_sha and not skip_sha:
        actual = _compute_sha256(path)
        if actual != EXPECTED_CENTROIDS_SHA256:
            raise ValueError(
                "\n"
                "  styxx: calibration centroid sha256 mismatch.\n"
                f"  file:     {path}\n"
                f"  expected: {EXPECTED_CENTROIDS_SHA256}\n"
                f"  actual:   {actual}\n"
                "\n"
                "  The shipped calibration data has been modified from\n"
                "  its pinned release version. styxx refuses to load an\n"
                "  altered centroid file — the whole point of the hash\n"
                "  pin is to catch tampering or corruption.\n"
                "\n"
                "  If you know this is safe and are debugging locally,\n"
                "  you can bypass the check with:\n"
                "    STYXX_SKIP_SHA=1 python ...\n"
                "\n"
                "  Never set STYXX_SKIP_SHA in production.\n"
            )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════
# Classifier
# ══════════════════════════════════════════════════════════════════

class CentroidClassifier:
    """Nearest-centroid classifier in z-score feature space.

    Training-free at runtime: the centroids were pre-computed by
    generate_centroids.py from the atlas v0.3 captures and ship with
    the package. This class just applies them.
    """

    def __init__(self, centroids_artifact: Optional[dict] = None):
        if centroids_artifact is None:
            centroids_artifact = load_centroids()
        self.artifact = centroids_artifact
        self.categories: List[str] = list(centroids_artifact["categories"])
        self._phases: Dict[str, dict] = {}
        for phase_name, phase_data in centroids_artifact["phases"].items():
            mu = np.asarray(phase_data["mu"], dtype=float)
            sigma = np.asarray(phase_data["sigma"], dtype=float)
            cent = {
                cat: np.asarray(phase_data["centroids"][cat], dtype=float)
                for cat in self.categories
            }
            self._phases[phase_name] = {
                "mu": mu,
                "sigma": sigma,
                "centroids": cent,
            }
        # Detect feature dimensionality from centroid vectors
        sample_phase = next(iter(self._phases.values()))
        sample_centroid = next(iter(sample_phase["centroids"].values()))
        self._feature_dim: int = len(sample_centroid)
        # Confidence calibrator (loaded from disk if available)
        self._calibrator: Optional[ConfidenceCalibrator] = None
        self._load_calibrated_centroids()

    def _load_calibrated_centroids(self) -> None:
        """Load agent-specific calibrated centroids if available.

        Checks ~/.styxx/calibration/{agent_name}.json for shifted centroids.
        If found, overlays them on the atlas centroids.  Fail-open.
        """
        try:
            from . import config as _cfg
            agent_name = _cfg.agent_name() if hasattr(_cfg, "agent_name") else "default"
            agent_name = agent_name or "default"
            cal_path = Path.home() / ".styxx" / "calibration" / f"{agent_name}.json"
            if not cal_path.exists():
                return
            with open(cal_path, "r", encoding="utf-8") as f:
                cal_data = json.load(f)
            # Overlay shifted centroids
            shifted = cal_data.get("shifted_centroids", {})
            for phase_name, cats in shifted.items():
                if phase_name not in self._phases:
                    continue
                for cat, centroid_list in cats.items():
                    arr = np.asarray(centroid_list, dtype=float)
                    if cat in self._phases[phase_name]["centroids"] and len(arr) == self._feature_dim:
                        self._phases[phase_name]["centroids"][cat] = arr
            # Load confidence calibrator
            cal_conf_path = Path.home() / ".styxx" / "calibration" / f"{agent_name}_confidence.json"
            self._calibrator = ConfidenceCalibrator.load(cal_conf_path)
        except Exception:
            pass  # fail-open: use atlas centroids

    def classify(
        self,
        trajectories: Dict[str, Sequence[float]],
        phase: str,
    ) -> "PhaseReading":
        """Classify a trajectory at a given phase.

        Returns a PhaseReading with predicted category, margin,
        distance to every centroid, and softmax-like confidence.
        """
        if phase not in self._phases:
            raise ValueError(
                f"unknown phase '{phase}', must be one of {list(self._phases)}"
            )
        n_tokens = PHASE_TOKEN_CUTOFFS[phase]
        # Select extractor based on centroid dimensionality
        if self._feature_dim > 12:
            feats = extract_features_v2(trajectories, n_tokens)
        else:
            feats = extract_features(trajectories, n_tokens)
        # Defensive: fall back to 12-dim if mismatch
        if len(feats) != self._feature_dim:
            feats = extract_features(trajectories, n_tokens)
        phase_data = self._phases[phase]
        mu, sigma = phase_data["mu"], phase_data["sigma"]
        z = (feats - mu) / sigma
        distances: Dict[str, float] = {}
        for cat, centroid in phase_data["centroids"].items():
            distances[cat] = float(np.linalg.norm(z - centroid))
        # Nearest + runner-up margin
        sorted_cats = sorted(distances.items(), key=lambda kv: kv[1])
        nearest, nearest_d = sorted_cats[0]
        runner_up_d = sorted_cats[1][1] if len(sorted_cats) > 1 else nearest_d
        margin = float(runner_up_d - nearest_d)

        # 0.8.3: phase1 adversarial suppression.
        # Phase1 reads only 1 token. The first token of ANY response
        # (even "hello") produces a feature vector near the adversarial
        # centroid (margin ~0.26). This caused 80% false positive rate.
        # Fix: require margin > 0.4 for phase1 adversarial to be trusted.
        # Below that, promote the runner-up category.
        if (nearest == "adversarial"
                and phase == "phase1_preflight"
                and margin < 0.4):
            nearest = sorted_cats[1][0]
            nearest_d = sorted_cats[1][1]
            margin = float(sorted_cats[2][1] - nearest_d) if len(sorted_cats) > 2 else 0.0
        # Pseudo-softmax confidence (not probabilistic, but useful for UI)
        # lower distance = higher score
        scores = {
            cat: float(math.exp(-(d - nearest_d)))
            for cat, d in distances.items()
        }
        total = sum(scores.values()) or 1.0
        probs = {cat: scores[cat] / total for cat in self.categories}
        # Apply confidence calibration if available
        if self._calibrator is not None and self._calibrator.is_fitted:
            cal_conf = self._calibrator.calibrate(nearest_d)
            if cal_conf is not None:
                old_conf = probs[nearest]
                probs[nearest] = cal_conf
                remaining = 1.0 - cal_conf
                other_total = sum(probs[c] for c in self.categories if c != nearest)
                if other_total > 0:
                    scale = remaining / other_total
                    for c in self.categories:
                        if c != nearest:
                            probs[c] *= scale
        # Always compute 21-dim features for storage/calibration
        features_v2 = None
        try:
            full_feats = extract_features_v2(trajectories, n_tokens)
            features_v2 = full_feats.tolist()
        except Exception:
            pass
        return PhaseReading(
            phase=phase,
            n_tokens_used=n_tokens,
            features=feats.tolist(),
            predicted_category=nearest,
            margin=margin,
            distances=distances,
            probs=probs,
            features_v2=features_v2,
        )


# ══════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════

@dataclass
class PhaseReading:
    """Result of one phase's classification."""
    phase: str
    n_tokens_used: int
    features: List[float]
    predicted_category: str
    margin: float
    distances: Dict[str, float]
    probs: Dict[str, float]
    # Extended 21-dim features (always computed when available)
    features_v2: Optional[List[float]] = None
    # Tier 1 D-axis fields (None when tier 1 is inactive)
    d_honesty_mean: Optional[float] = None
    d_honesty_std: Optional[float] = None
    d_honesty_delta: Optional[float] = None   # late_mean - early_mean
    # Tier 2 SAE fields (None until tier 2 ships)
    k_depth: Optional[float] = None
    c_coherence: Optional[float] = None
    s_commitment: Optional[float] = None

    def top3(self) -> List[Tuple[str, float]]:
        """Top three categories by probability."""
        return sorted(self.probs.items(), key=lambda kv: -kv[1])[:3]

    @property
    def confidence(self) -> float:
        """Probability assigned to the predicted (nearest) category."""
        return self.probs[self.predicted_category]


@dataclass
class Vitals:
    """Complete cognitive vitals for one LLM generation.

    This is what every call through styxx emits as .vitals alongside
    the normal .choices.
    """
    phase1_pre: PhaseReading
    phase2_early: Optional[PhaseReading] = None
    phase3_mid: Optional[PhaseReading] = None
    phase4_late: Optional[PhaseReading] = None
    tier_active: int = 0
    abort_reason: Optional[str] = None
    # Cross-phase coherence (set when >= 2 phases available)
    coherence: Optional[float] = None
    transition_vectors: Optional[List[List[float]]] = None
    # Predictive forecast (set when >= 5 tokens available)
    forecast: Optional[Any] = None  # ForecastResult (lazy import to avoid circular)

    def as_dict(self) -> dict:
        """JSON-serializable dict view. Injects computed fields
        (confidence, top3) that aren't stored on PhaseReading."""
        def _phase_dict(r: Optional[PhaseReading]) -> Optional[dict]:
            if r is None:
                return None
            d = asdict(r)
            d["confidence"] = r.confidence
            d["top3"] = r.top3()
            return d
        return {
            "phase1_pre":   _phase_dict(self.phase1_pre),
            "phase2_early": _phase_dict(self.phase2_early),
            "phase3_mid":   _phase_dict(self.phase3_mid),
            "phase4_late":  _phase_dict(self.phase4_late),
            "tier_active":  self.tier_active,
            "abort_reason": self.abort_reason,
            "coherence":    self.coherence,
            "transition_vectors": self.transition_vectors,
        }

    @property
    def summary(self) -> str:
        """Render the full ASCII vitals card.

        Delegates to styxx.cards.render_vitals_card so there's one
        source of truth for card rendering. Lazy imported to avoid
        a circular dependency between vitals.py and cards.py.
        """
        from .cards import render_vitals_card
        return render_vitals_card(self)

    # ── agent-friendly shortcut properties ────────────────────────
    #
    # These are the shapes Xendro asked for in 0.1.0a1 feedback:
    #     vitals.phase1  -> "reasoning:0.28"
    #     vitals.phase4  -> "reasoning:0.45"
    #     vitals.gate    -> "pass" / "warn" / "fail" / "pending"
    # They're thin compact views over the PhaseReading objects.

    @property
    def phase1(self) -> str:
        """Compact string view of phase 1: 'category:confidence'."""
        if self.phase1_pre is None:
            return "-"
        return f"{self.phase1_pre.predicted_category}:{self.phase1_pre.confidence:.2f}"

    @property
    def phase2(self) -> str:
        if self.phase2_early is None:
            return "-"
        return f"{self.phase2_early.predicted_category}:{self.phase2_early.confidence:.2f}"

    @property
    def phase3(self) -> str:
        if self.phase3_mid is None:
            return "-"
        return f"{self.phase3_mid.predicted_category}:{self.phase3_mid.confidence:.2f}"

    @property
    def phase4(self) -> str:
        """Compact string view of phase 4: 'category:confidence'.

        Returns '-' if phase 4 wasn't reached (e.g., short streaming
        completion that didn't hit the 25-token window).
        """
        if self.phase4_late is None:
            return "-"
        return f"{self.phase4_late.predicted_category}:{self.phase4_late.confidence:.2f}"

    @property
    def _primary_reading(self) -> Optional[PhaseReading]:
        """Return the latest available PhaseReading, preferring phase 4."""
        for r in (self.phase4_late, self.phase3_mid,
                  self.phase2_early, self.phase1_pre):
            if r is not None:
                return r
        return None

    @property
    def category(self) -> str:
        """Primary cognitive category for this generation.

        Returns the predicted category from the latest available phase,
        preferring phase 4 (post-generation) over earlier phases.
        This is the shortcut used in the 30-second quickstart:

            vitals = styxx.observe(response)
            print(vitals.category)   # "reasoning"
        """
        r = self._primary_reading
        return r.predicted_category if r is not None else "unknown"

    @property
    def confidence(self) -> float:
        """Confidence of the primary category classification, in [0, 1].

        Returns the softmax-like confidence from the latest available
        phase, preferring phase 4. This is the shortcut used in the
        30-second quickstart:

            vitals = styxx.observe(response)
            print(vitals.confidence)  # 0.45
        """
        r = self._primary_reading
        return float(r.confidence) if r is not None else 0.0

    @property
    def d_honesty(self) -> Optional[str]:
        """Compact string view of the D-axis honesty measurement.

        Returns the mean D value across the latest available phase
        with D-axis data attached, or None if tier 1 is not active.

        High D (~0.8+) = honest (model is saying what it thinks)
        Low D (~0.3 or below) = divergent (internal state doesn't
        match the output token)

        0.3.0+. Requires tier 1 (STYXX_TIER1_ENABLED=1 + model loaded).
        """
        for phase in (self.phase4_late, self.phase3_mid,
                      self.phase2_early, self.phase1_pre):
            if phase is not None and phase.d_honesty_mean is not None:
                return f"{phase.d_honesty_mean:.3f}"
        return None

    def to_thought(
        self,
        *,
        source_text: Optional[str] = None,
        source_model: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> "Thought":
        """Promote this Vitals into a portable Thought (3.0.0a1+).

        Convenience shortcut for `Thought.from_vitals(self, ...)`.
        Symmetric with the constructor: instead of

            t = styxx.Thought.from_vitals(vitals)

        you can write

            t = vitals.to_thought()

        See styxx.thought for the full Thought type documentation.
        """
        from .thought import Thought
        return Thought.from_vitals(
            self,
            source_text=source_text,
            source_model=source_model,
            tags=tags,
        )

    def as_markdown(self) -> str:
        """Render vitals as a compact markdown block suitable for
        pasting into chat history, memory files, or code review
        comments.

        Complements:
          .summary     - full ASCII vitals card (for terminals)
          .as_dict()   - JSON-serializable dict (for machines)
          .as_markdown() - markdown code block (for humans in chats)
        """
        lines = ["```styxx"]
        lines.append(f"phase1: {self.phase1}")
        if self.phase2_early is not None:
            lines.append(f"phase2: {self.phase2}")
        if self.phase3_mid is not None:
            lines.append(f"phase3: {self.phase3}")
        lines.append(f"phase4: {self.phase4}")
        lines.append(f"gate:   {self.gate}")
        lines.append(f"tier:   {self.tier_active}")
        d = self.d_honesty
        if d is not None:
            lines.append(f"d_honesty: {d}")
        if self.abort_reason:
            lines.append(f"abort:  {self.abort_reason}")
        lines.append("```")
        return "\n".join(lines)

    @property
    def trust_score(self) -> float:
        """0-1 trust score for this vitals reading.

        1.0.0: integrated from recipes/memory.py. Every vitals object
        now carries its own trust score so agents can use it for
        weighted memory, decision confidence, or downstream routing.

        3.2.0: forecast risk and coherence now factor into trust.
        Critical forecast risk applies a heavy penalty even if the
        gate says pass. High coherence boosts trust (stable reasoning).

        Factors: gate status (40%), phase4 confidence (25%),
        category penalty (15%), forecast risk (10%), coherence (10%).
        """
        gate_scores = {"pass": 1.0, "warn": 0.5, "fail": 0.2, "pending": 0.7}
        gate_w = gate_scores.get(self.gate, 0.7)
        conf = self.phase4_late.confidence if self.phase4_late else 0.5
        # Category penalty
        penalty = 0.0
        if self.phase4_late:
            cat = self.phase4_late.predicted_category
            if cat == "hallucination":
                penalty = 0.3
            elif cat == "adversarial":
                penalty = 0.2
        # Forecast risk penalty (critical=0.0, high=0.3, moderate=0.6, low=1.0)
        forecast_risk_scores = {"low": 1.0, "moderate": 0.6, "high": 0.3, "critical": 0.0}
        fc = getattr(self, "forecast", None)
        forecast_w = forecast_risk_scores.get(
            fc.risk_level if fc else "low", 1.0
        )
        # Coherence boost (high coherence = stable, trustworthy)
        coh = self.coherence if self.coherence is not None else 0.7
        score = (gate_w * 0.40
                 + conf * 0.25
                 + (1.0 - penalty) * 0.15
                 + forecast_w * 0.10
                 + coh * 0.10)
        return round(max(0.0, min(1.0, score)), 3)

    @property
    def gate(self) -> str:
        """Default gate status computed from phase 4.

        Returns one of:
          'pass'    - quiet reasoning / retrieval / creative
          'warn'    - refusal or adversarial attractor caught
          'fail'    - hallucination attractor caught
          'pending' - phase 4 not yet reached (short completion)

        Threshold policy: chance level on 6 categories is 0.167.
        We use 0.20 (chance + ~0.03) as the floor for meaningful
        predictions. At that floor any ARGMAX that lands on a
        load-bearing category is reported.

        User-defined gates registered via styxx.on_gate() are
        dispatched AFTER this default gate is computed - they're
        additive, they don't override this value. If you want
        custom logic, read phase1/phase4 directly and compute your
        own verdict.
        """
        from . import config
        BASE_FLOOR = 0.20    # 0.167 chance + ~0.03 margin
        floor = BASE_FLOOR * config.gate_multiplier()
        if self.phase4_late is None:
            return "pending"
        pred = self.phase4_late.predicted_category
        conf = self.phase4_late.confidence

        # 0.9.7: suppress hallucination gate on creative/code contexts.
        # Creative writing and code generation produce logprob signatures
        # near the hallucination centroid (specific details, confident
        # language, no hedging) but that's the expected shape for
        # generative content. Check expected categories before gating.
        expected = config.expected_categories()
        ctx = config.current_context()
        _CTX_EXPECTED = {
            "creative_writing": {"creative", "hallucination"},
            "code_review": {"reasoning"},
        }
        ctx_expected = _CTX_EXPECTED.get(ctx, set()) if ctx else set()
        all_expected = expected | ctx_expected

        if pred == "hallucination" and conf > floor:
            if "hallucination" in all_expected:
                return "pass"  # expected hallucination-like signal
            # Low confidence hallucination (< 0.35) on a marginal call
            # is more likely centroid confusion than real hallucination
            if conf < 0.35:
                return "warn"  # downgrade to warn instead of fail
            return "fail"
        if pred in ("refusal", "adversarial") and conf > floor:
            if pred in all_expected:
                return "pass"
            return "warn"
        # 3.2.0: forecast override — if the phase4 classifier says "pass"
        # but the forecast predicted critical risk, downgrade to "warn".
        # The forecast reads trajectory shape features that the atlas
        # centroids miss (hallucination curvature, adversarial volatility).
        fc = getattr(self, "forecast", None)
        if fc and fc.risk_level == "critical":
            fc_cat = fc.predicted_category
            if fc_cat not in all_expected and fc_cat in ("hallucination", "adversarial"):
                return "warn"
        return "pass"


# ══════════════════════════════════════════════════════════════════
# Shared logprob math (0.7.2)
# ══════════════════════════════════════════════════════════════════
#
# Three call sites (openai adapter, watch.py, reflex.py) previously
# duplicated this conversion. A bug fix in one would miss the others.
# Now there's one source of truth.

def logprobs_to_entropy_margin(
    top_logprobs: List[float],
) -> Tuple[float, float]:
    """Convert a list of top-k log-probabilities into (entropy, top2_margin).

    This is the entropy bridge validated at r=0.902 shape correlation
    against full-vocabulary entropy on open-weight models (see
    atlas/FINDINGS_entropy_bridge.md).

    Args:
        top_logprobs: log-probabilities of the top-k tokens for one
                      generation step (e.g. from OpenAI's logprobs).

    Returns:
        (entropy, top2_margin) tuple.
    """
    if not top_logprobs:
        return 0.0, 1.0
    probs = [math.exp(lp) for lp in top_logprobs]
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
        entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
    else:
        entropy = 0.0
    sorted_probs = sorted(probs, reverse=True)
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 1.0
    return float(entropy), margin

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
from typing import Dict, List, Optional, Sequence, Tuple

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
    "f25edc5f47bb93928671aab05f38f351a2d0df0fb7722d53e48d2368b0d5c543"
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
        # Pseudo-softmax confidence (not probabilistic, but useful for UI)
        # lower distance = higher score
        scores = {
            cat: float(math.exp(-(d - nearest_d)))
            for cat, d in distances.items()
        }
        total = sum(scores.values()) or 1.0
        probs = {cat: scores[cat] / total for cat in self.categories}
        return PhaseReading(
            phase=phase,
            n_tokens_used=n_tokens,
            features=feats.tolist(),
            predicted_category=nearest,
            margin=margin,
            distances=distances,
            probs=probs,
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
        CHANCE_FLOOR = 0.20   # 0.167 chance + ~0.03 margin
        if self.phase4_late is None:
            return "pending"
        pred = self.phase4_late.predicted_category
        conf = self.phase4_late.confidence
        if pred == "hallucination" and conf > CHANCE_FLOOR:
            return "fail"
        if pred in ("refusal", "adversarial") and conf > CHANCE_FLOOR:
            return "warn"
        return "pass"

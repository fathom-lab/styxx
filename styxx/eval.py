# -*- coding: utf-8 -*-
"""
styxx.eval -- ground truth evaluation harness.

The first real eval infrastructure for styxx. Ships labeled fixtures
from demo_trajectories.json and computes per-category precision, recall,
F1, confusion matrix, per-phase accuracy, and calibration curve.

Usage:
    from styxx import EvalSuite
    result = EvalSuite.from_demo_trajectories().run()
    print(result.render())

    # A/B comparison (e.g. 12-dim vs 21-dim centroids):
    from styxx import compare_evals
    print(compare_evals(result_a, result_b, label_a="12-dim", label_b="21-dim"))

    # Load custom fixtures:
    suite = EvalSuite.from_json("my_fixtures.json")
    result = suite.run()

CLI:
    styxx eval
    styxx eval --fixtures custom.json --format json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vitals import CATEGORIES, PHASE_ORDER


# ══════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════

@dataclass
class EvalFixture:
    """A labeled test case for the classifier."""
    label: str
    entropy: List[float]
    logprob: List[float]
    top2_margin: List[float]
    phase: str = "phase4_late"
    prompt: Optional[str] = None
    source: Optional[str] = None


@dataclass
class EvalResult:
    """Result of running the eval harness on a set of fixtures."""
    n_fixtures: int
    accuracy: float
    per_category: Dict[str, Dict[str, float]]
    confusion: Any  # np.ndarray (6x6)
    per_phase: Dict[str, float]
    calibration: List[Tuple[float, float, float, int]]
    feature_dim: int = 12

    def as_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "n_fixtures": self.n_fixtures,
            "accuracy": round(self.accuracy, 4),
            "feature_dim": self.feature_dim,
            "per_category": {
                cat: {k: round(v, 4) for k, v in metrics.items()}
                for cat, metrics in self.per_category.items()
            },
            "confusion_matrix": {
                "categories": CATEGORIES,
                "matrix": self.confusion.tolist() if hasattr(self.confusion, "tolist") else self.confusion,
            },
            "per_phase": {k: round(v, 4) for k, v in self.per_phase.items()},
            "calibration": [
                {"bin": round(b, 2), "pred_conf": round(p, 3), "actual_acc": round(a, 3), "n": n}
                for b, p, a, n in self.calibration
            ],
        }

    def render(self) -> str:
        """ASCII table rendering for terminal display."""
        lines = []
        lines.append("=" * 62)
        lines.append("  styxx eval harness")
        lines.append(f"  {self.n_fixtures} fixtures | {self.feature_dim}-dim features")
        lines.append("=" * 62)
        lines.append("")

        # Overall accuracy
        lines.append(f"  overall accuracy: {self.accuracy:.1%}")
        lines.append("")

        # Per-category table
        lines.append("  category       precision  recall     f1       support")
        lines.append("  " + "-" * 56)
        for cat in CATEGORIES:
            m = self.per_category.get(cat, {})
            p = m.get("precision", 0.0)
            r = m.get("recall", 0.0)
            f1 = m.get("f1", 0.0)
            s = int(m.get("support", 0))
            lines.append(f"  {cat:<14s}  {p:>7.3f}    {r:>7.3f}    {f1:>7.3f}    {s:>3d}")
        lines.append("")

        # Confusion matrix
        lines.append("  confusion matrix (rows=true, cols=predicted)")
        short = [c[:4] for c in CATEGORIES]
        header = "           " + "  ".join(f"{s:>5s}" for s in short)
        lines.append(f"  {header}")
        cm = self.confusion if hasattr(self.confusion, "__getitem__") else np.array(self.confusion)
        for i, cat in enumerate(CATEGORIES):
            row = "  ".join(f"{int(cm[i][j]):>5d}" for j in range(len(CATEGORIES)))
            lines.append(f"  {cat[:9]:>9s}  {row}")
        lines.append("")

        # Per-phase accuracy
        if self.per_phase:
            lines.append("  per-phase accuracy")
            for phase, acc in sorted(self.per_phase.items()):
                lines.append(f"    {phase}: {acc:.1%}")
            lines.append("")

        # Calibration curve
        if self.calibration:
            lines.append("  calibration curve (confidence vs accuracy)")
            lines.append("  bin     pred_conf  actual_acc  n")
            lines.append("  " + "-" * 40)
            for b, p, a, n in self.calibration:
                lines.append(f"  {b:>5.2f}   {p:>9.3f}  {a:>10.3f}  {n:>3d}")
            lines.append("")

        lines.append("=" * 62)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# Eval suite
# ══════════════════════════════════════════════════════════════════

class EvalSuite:
    """Collection of labeled fixtures plus runner logic."""

    def __init__(self, fixtures: Optional[List[EvalFixture]] = None):
        self.fixtures: List[EvalFixture] = fixtures or []

    @classmethod
    def from_demo_trajectories(cls) -> "EvalSuite":
        """Load the 6 bundled atlas demo trajectories as eval fixtures.

        Each trajectory has a known category label (the probe category
        it was captured from).
        """
        path = Path(__file__).resolve().parent / "centroids" / "demo_trajectories.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fixtures = []
        source_model = data.get("source_model", "unknown")
        for cat_name, traj in data["trajectories"].items():
            fixtures.append(EvalFixture(
                label=cat_name,
                entropy=traj["entropy"],
                logprob=traj["logprob"],
                top2_margin=traj["top2_margin"],
                prompt=traj.get("text_preview"),
                source=source_model,
            ))
        return cls(fixtures=fixtures)

    @classmethod
    def from_json(cls, path: str) -> "EvalSuite":
        """Load fixtures from a JSON file.

        Expected format:
        {
            "fixtures": [
                {
                    "label": "reasoning",
                    "entropy": [...],
                    "logprob": [...],
                    "top2_margin": [...],
                    "prompt": "optional"
                }
            ]
        }
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fixtures = []
        for item in data.get("fixtures", []):
            fixtures.append(EvalFixture(
                label=item["label"],
                entropy=item["entropy"],
                logprob=item["logprob"],
                top2_margin=item["top2_margin"],
                phase=item.get("phase", "phase4_late"),
                prompt=item.get("prompt"),
                source=item.get("source"),
            ))
        return cls(fixtures=fixtures)

    def add(self, fixture: EvalFixture) -> None:
        """Add a single fixture to the suite."""
        self.fixtures.append(fixture)

    def run(self, runtime: Any = None) -> EvalResult:
        """Run all fixtures through the classifier and compute metrics.

        Args:
            runtime: StyxxRuntime to use. Defaults to a fresh instance.
        """
        if runtime is None:
            from .core import StyxxRuntime
            runtime = StyxxRuntime()

        predictions: List[Tuple[str, float, str]] = []  # (pred_cat, confidence, phase)
        labels: List[str] = []

        for fix in self.fixtures:
            vitals = runtime.run_on_trajectories(
                entropy=fix.entropy,
                logprob=fix.logprob,
                top2_margin=fix.top2_margin,
            )
            # Get the reading for the specified phase
            reading = None
            if fix.phase == "phase1_preflight":
                reading = vitals.phase1_pre
            elif fix.phase == "phase2_early":
                reading = vitals.phase2_early
            elif fix.phase == "phase3_mid":
                reading = vitals.phase3_mid
            elif fix.phase == "phase4_late":
                reading = vitals.phase4_late
            # Fallback to latest available
            if reading is None:
                for r in (vitals.phase4_late, vitals.phase3_mid,
                          vitals.phase2_early, vitals.phase1_pre):
                    if r is not None:
                        reading = r
                        break
            if reading is not None:
                predictions.append((reading.predicted_category, reading.confidence, reading.phase))
                labels.append(fix.label)

        return self._compute_metrics(predictions, labels, runtime.classifier._feature_dim)

    def _compute_metrics(
        self,
        predictions: List[Tuple[str, float, str]],
        labels: List[str],
        feature_dim: int,
    ) -> EvalResult:
        """Compute precision/recall/F1/confusion from predictions vs labels."""
        n = len(labels)
        if n == 0:
            return EvalResult(
                n_fixtures=0, accuracy=0.0, per_category={},
                confusion=np.zeros((6, 6), dtype=int),
                per_phase={}, calibration=[], feature_dim=feature_dim,
            )

        cat_idx = {c: i for i, c in enumerate(CATEGORIES)}
        confusion = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
        correct_count = 0

        # Per-phase tracking
        phase_correct: Dict[str, int] = {}
        phase_total: Dict[str, int] = {}

        # Calibration tracking
        cal_bins: Dict[int, List[Tuple[float, bool]]] = {i: [] for i in range(10)}

        for (pred, conf, phase), true_label in zip(predictions, labels):
            ti = cat_idx.get(true_label, -1)
            pi = cat_idx.get(pred, -1)
            if ti >= 0 and pi >= 0:
                confusion[ti][pi] += 1
            is_correct = (pred == true_label)
            if is_correct:
                correct_count += 1

            # Phase tracking
            phase_total[phase] = phase_total.get(phase, 0) + 1
            if is_correct:
                phase_correct[phase] = phase_correct.get(phase, 0) + 1

            # Calibration bin
            bin_idx = min(int(conf * 10), 9)
            cal_bins[bin_idx].append((conf, is_correct))

        accuracy = correct_count / n

        # Per-category metrics
        per_category: Dict[str, Dict[str, float]] = {}
        for c in CATEGORIES:
            ci = cat_idx[c]
            tp = int(confusion[ci][ci])
            fp = int(confusion[:, ci].sum()) - tp
            fn = int(confusion[ci, :].sum()) - tp
            support = int(confusion[ci, :].sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            per_category[c] = {
                "precision": precision, "recall": recall, "f1": f1, "support": support,
            }

        # Per-phase accuracy
        per_phase = {
            phase: phase_correct.get(phase, 0) / total
            for phase, total in phase_total.items() if total > 0
        }

        # Calibration curve
        calibration = []
        for bin_idx in range(10):
            items = cal_bins[bin_idx]
            if not items:
                continue
            confs, corrects = zip(*items)
            calibration.append((
                (bin_idx + 0.5) / 10,
                float(np.mean(confs)),
                float(np.mean(corrects)),
                len(items),
            ))

        return EvalResult(
            n_fixtures=n,
            accuracy=accuracy,
            per_category=per_category,
            confusion=confusion,
            per_phase=per_phase,
            calibration=calibration,
            feature_dim=feature_dim,
        )


# ══════════════════════════════════════════════════════════════════
# Comparison
# ══════════════════════════════════════════════════════════════════

def compare_evals(
    result_a: EvalResult,
    result_b: EvalResult,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> str:
    """ASCII comparison table of two eval results.

    Used for A/B comparison of feature sets (e.g. 12-dim vs 21-dim).
    """
    lines = []
    lines.append("=" * 62)
    lines.append(f"  eval comparison: {label_a} ({result_a.feature_dim}-dim) vs {label_b} ({result_b.feature_dim}-dim)")
    lines.append("=" * 62)
    lines.append("")

    # Overall
    da = result_a.accuracy
    db = result_b.accuracy
    delta = db - da
    arrow = "^" if delta > 0 else ("v" if delta < 0 else "=")
    lines.append(f"  overall accuracy:  {label_a}={da:.1%}  {label_b}={db:.1%}  delta={delta:+.1%} {arrow}")
    lines.append("")

    # Per-category deltas
    lines.append(f"  category       {label_a:>6s}_f1  {label_b:>6s}_f1    delta")
    lines.append("  " + "-" * 46)
    for cat in CATEGORIES:
        ma = result_a.per_category.get(cat, {})
        mb = result_b.per_category.get(cat, {})
        f1a = ma.get("f1", 0.0)
        f1b = mb.get("f1", 0.0)
        d = f1b - f1a
        lines.append(f"  {cat:<14s}  {f1a:>7.3f}   {f1b:>7.3f}   {d:>+7.3f}")
    lines.append("")
    lines.append("=" * 62)
    return "\n".join(lines)

# -*- coding: utf-8 -*-
"""
styxx.gauntlet — the public empirical-floor challenge runner (7.7.5).

This module operationalizes the empirical floor shipped with the
Decorrelation Ceiling paper (`papers/PAPER_decorrelation_ceiling_2026_05_27.md`).
It loads any external user-supplied detection/classification method, runs
it against the labeled benchmark (`papers/consensus-hallucination/
darkcore_benchmark_2026_05_27.json`), scores it against pre-registered bars,
and returns a structured result suitable for the public LEADERBOARD.md.

The frame is intentional: the seven-method floor we shipped is the bar.
We invite the field to beat it. If anyone beats it, the synthesis is
revised; if nobody can, the floor compounds across submissions.

Two task modes:

  - **classification:** the user's method takes a question string and
    returns a predicted class label. Bars are the same K1/K2/K3 as the
    darkcore_classifier_2026_05_27 baseline.

  - **detection:** the user's method takes a (prompt, response) pair and
    returns a divergence/anomaly score. Bars are AUC-style on misconception
    vs truth + folklore-subset AUC.

The runner is deliberately framework-agnostic — pass in a Python callable,
get a structured result. The CLI face is in styxx.cli.cmd_gauntlet.
"""
from __future__ import annotations

import importlib
import json
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


__all__ = [
    "GauntletResult",
    "Submission",
    "load_benchmark",
    "resolve_method",
    "run_classification_gauntlet",
    "run_detection_gauntlet",
    "compute_auc",
    "compute_f1",
    "BASELINE_ENTRY",
]


# ──────────────────────────────────────────────────────────────────────
# Pre-registered bars (locked, traceable to the empirical floor on origin)
# ──────────────────────────────────────────────────────────────────────

# Classification bars from preregistration_darkcore_classifier_2026_05_27.md (commit 646dcb0)
DEFAULT_CLASSIFICATION_BARS = {
    "K1_folklore_F1": 0.70,    # in-distribution folklore F1
    "K2_accuracy": 0.65,        # in-distribution 4-way accuracy
    "K3_crosscorpus_F1": 0.60,  # cross-corpus folklore F1 (load-bearing)
}

# Detection bars from the ICT (preregistration_ict_2026_05_25.md) and JD
# (preregistration_jd_2026_05_25.md) findings. Translated to a single
# benchmark-applied form: AUC of detection-score-as-misconception-predictor.
#
# 7.7.8: added D3 length-control bar after Baseline-007 exposed that v1
# bars (D1, D2) are gameable by exploiting a length-confound in the
# benchmark's expected_consensus field (truth responses average 3.9 words,
# folklore responses average 7.5 words). A real detector must beat the
# length-only oracle's AUC by at least 0.10 to demonstrate signal beyond
# the artifact.
DEFAULT_DETECTION_BARS = {
    "D1_misconception_AUC": 0.70,         # full misconception vs truth (JD's J1 territory)
    "D2_folklore_AUC": 0.70,              # folklore-subset vs truth (the dark-core test)
    "D3_length_control_delta": 0.10,      # AUC must beat length-only oracle by >=0.10
}

VALID_CLASSES = ("folklore", "pseudoscience", "factual-error", "truth")


# ──────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Submission:
    """User-supplied method submitted to the gauntlet."""
    name: str                # human-readable name (e.g., "Smith-2026-classifier")
    method: Callable[..., Any]
    task: str                # "classification" or "detection"
    module_spec: str = ""    # original "my_module:predict" string if loaded via spec
    notes: str = ""


@dataclass
class GauntletResult:
    """Structured result of running a submission against the benchmark."""
    submission_name: str
    task: str
    benchmark_version: str
    n_items: int
    metrics: Dict[str, float] = field(default_factory=dict)
    bars: Dict[str, float] = field(default_factory=dict)
    bar_results: Dict[str, bool] = field(default_factory=dict)
    overall_pass: bool = False
    n_passed: int = 0
    n_total_bars: int = 0
    per_item_predictions: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
# Benchmark loading + method resolution
# ──────────────────────────────────────────────────────────────────────

def load_benchmark(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the dark-core benchmark JSON. Defaults to the bundled 2026-05-27 version.

    Resolution order:
      1. Explicit `path` argument (if provided).
      2. ``styxx/_data/darkcore_benchmark_2026_05_27.json`` — shipped as package data
         in the installed wheel (7.7.6+).
      3. ``papers/consensus-hallucination/darkcore_benchmark_2026_05_27.json`` —
         the source-tree path, present when running from a styxx git checkout.
    """
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"benchmark not found at {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    # 1. Bundled package data (installed wheel)
    pkg_data = Path(__file__).resolve().parent / "_data" / "darkcore_benchmark_2026_05_27.json"
    if pkg_data.exists():
        return json.loads(pkg_data.read_text(encoding="utf-8"))

    # 2. Source-tree fallback (git checkout)
    source_tree = (Path(__file__).resolve().parent.parent
                   / "papers" / "consensus-hallucination"
                   / "darkcore_benchmark_2026_05_27.json")
    if source_tree.exists():
        return json.loads(source_tree.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        f"benchmark not found. Looked at:\n"
        f"  - {pkg_data} (package data — should be present in a pip install)\n"
        f"  - {source_tree} (source tree — present when running from a git checkout)\n"
        f"Pass --benchmark <path> to point at a custom benchmark JSON in the same schema."
    )


def resolve_method(spec: str) -> Callable[..., Any]:
    """Resolve a "module:attr" spec to a callable.

    Example specs:
      - "my_module:predict"
      - "my_pkg.sub:detect"
      - "styxx.gauntlet:_majority_baseline"
    """
    if ":" not in spec:
        raise ValueError(f"method spec must be 'module:attr', got: {spec!r}")
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr, None)
    if fn is None:
        raise AttributeError(f"module {mod_name!r} has no attribute {attr!r}")
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────

def compute_f1(y_true: List[int], y_pred: List[int]) -> float:
    """Binary F1. y_true and y_pred are lists of 0/1."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_auc(pos_scores: List[float], neg_scores: List[float]) -> float:
    """Mann-Whitney AUC; ties = 0.5 credit."""
    if not pos_scores or not neg_scores:
        return float("nan")
    w = sum(1 for p in pos_scores for q in neg_scores if p > q) + 0.5 * sum(
        1 for p in pos_scores for q in neg_scores if p == q
    )
    return w / (len(pos_scores) * len(neg_scores))


# ──────────────────────────────────────────────────────────────────────
# Classification gauntlet
# ──────────────────────────────────────────────────────────────────────

def run_classification_gauntlet(
    submission: Submission,
    benchmark: Dict[str, Any],
    bars: Optional[Dict[str, float]] = None,
) -> GauntletResult:
    """Run a classification method against the dark-core benchmark.

    The submission's method signature: ``def predict(question: str) -> dict``
    where the returned dict must have key ``"class"`` whose value is one of
    the four labels in VALID_CLASSES.
    """
    if bars is None:
        bars = dict(DEFAULT_CLASSIFICATION_BARS)
    records = benchmark.get("records", [])
    predictions: List[Dict[str, Any]] = []
    correct = 0
    folk_y_true: List[int] = []
    folk_y_pred: List[int] = []
    cross_y_true: List[int] = []
    cross_y_pred: List[int] = []

    for r in records:
        q = r.get("question", "")
        true_class = r.get("class")
        try:
            pred = submission.method(q)
        except Exception as e:
            return GauntletResult(
                submission_name=submission.name, task="classification",
                benchmark_version=benchmark.get("version", "unknown"),
                n_items=0, error=f"method raised: {type(e).__name__}: {e}",
            )
        if not isinstance(pred, dict) or "class" not in pred:
            return GauntletResult(
                submission_name=submission.name, task="classification",
                benchmark_version=benchmark.get("version", "unknown"),
                n_items=0,
                error=f"method must return dict with 'class' key, got: {type(pred).__name__}",
            )
        predicted_class = str(pred["class"])
        is_correct = (predicted_class == true_class)
        if is_correct:
            correct += 1
        # binary folklore F1 (in-distribution)
        folk_y_true.append(1 if true_class == "folklore" else 0)
        folk_y_pred.append(1 if predicted_class == "folklore" else 0)
        # cross-corpus F1: curated folklore items are the held-out subset
        if r.get("source", "").startswith("curated_folklore"):
            cross_y_true.append(1)
            cross_y_pred.append(1 if predicted_class == "folklore" else 0)
        predictions.append({
            "id": r.get("id"),
            "question": q[:80],
            "true": true_class,
            "pred": predicted_class,
            "correct": is_correct,
        })

    accuracy = correct / len(records) if records else 0.0
    folk_f1 = compute_f1(folk_y_true, folk_y_pred)
    # cross-corpus F1: we have only positives in the curated-folklore subset, so
    # measure recall and combine with in-distribution precision for an honest F1.
    # The classifier baseline uses the binary task; we mirror that here.
    cross_recall = (sum(cross_y_pred) / len(cross_y_pred)) if cross_y_pred else float("nan")
    # cross-corpus binary F1 mixes the curated positives with in-dist negatives
    mixed_true = folk_y_true + [1] * len(cross_y_true)
    mixed_pred = folk_y_pred + cross_y_pred
    cross_f1 = compute_f1(mixed_true, mixed_pred)

    metrics = {
        "accuracy": round(accuracy, 4),
        "folklore_F1_indist": round(folk_f1, 4),
        "folklore_F1_crosscorpus": round(cross_f1, 4),
        "folklore_recall_crosscorpus": round(cross_recall, 4) if cross_recall == cross_recall else None,
    }

    bar_results = {
        "K1_folklore_F1": folk_f1 >= bars["K1_folklore_F1"],
        "K2_accuracy": accuracy >= bars["K2_accuracy"],
        "K3_crosscorpus_F1": cross_f1 >= bars["K3_crosscorpus_F1"],
    }
    n_passed = sum(bar_results.values())

    return GauntletResult(
        submission_name=submission.name,
        task="classification",
        benchmark_version=benchmark.get("version", "unknown"),
        n_items=len(records),
        metrics=metrics,
        bars=bars,
        bar_results=bar_results,
        overall_pass=all(bar_results.values()),
        n_passed=n_passed,
        n_total_bars=len(bar_results),
        per_item_predictions=predictions[:10],  # truncate for size; first 10 only
    )


# ──────────────────────────────────────────────────────────────────────
# Detection gauntlet
# ──────────────────────────────────────────────────────────────────────

def run_detection_gauntlet(
    submission: Submission,
    benchmark: Dict[str, Any],
    bars: Optional[Dict[str, float]] = None,
) -> GauntletResult:
    """Run a detection method against the dark-core benchmark.

    The submission's method signature: ``def detect(question: str, response: str) -> dict``
    where ``response`` is the expected_consensus (the council's baseline answer; the
    misconception for non-truth classes, the truth for truth records), and the returned
    dict must have key ``"score"`` (float, higher = more anomalous/misconception-like).
    """
    if bars is None:
        bars = dict(DEFAULT_DETECTION_BARS)
    records = benchmark.get("records", [])
    predictions: List[Dict[str, Any]] = []
    misc_scores: List[float] = []
    truth_scores: List[float] = []
    folk_scores: List[float] = []

    for r in records:
        q = r.get("question", "")
        cons = r.get("expected_consensus") or ""
        true_class = r.get("class")
        try:
            pred = submission.method(q, cons)
        except Exception as e:
            return GauntletResult(
                submission_name=submission.name, task="detection",
                benchmark_version=benchmark.get("version", "unknown"),
                n_items=0, error=f"method raised: {type(e).__name__}: {e}",
            )
        if not isinstance(pred, dict) or "score" not in pred:
            return GauntletResult(
                submission_name=submission.name, task="detection",
                benchmark_version=benchmark.get("version", "unknown"),
                n_items=0,
                error=f"method must return dict with 'score' key, got: {type(pred).__name__}",
            )
        score = float(pred["score"])
        if true_class == "truth":
            truth_scores.append(score)
        else:
            misc_scores.append(score)
            if true_class == "folklore":
                folk_scores.append(score)
        predictions.append({
            "id": r.get("id"),
            "question": q[:80],
            "true": true_class,
            "score": round(score, 4),
        })

    d1_auc = compute_auc(misc_scores, truth_scores)
    d2_auc = compute_auc(folk_scores, truth_scores)

    # 7.7.8: D3 length-control. Compute the length-only oracle's AUC on the
    # same partitions; a real detector must beat it by >= bars["D3_length_control_delta"].
    misc_lengths: List[float] = []
    truth_lengths: List[float] = []
    folk_lengths: List[float] = []
    for r in records:
        cons = r.get("expected_consensus") or ""
        ln = float(len(cons.split()))
        true_class = r.get("class")
        if true_class == "truth":
            truth_lengths.append(ln)
        else:
            misc_lengths.append(ln)
            if true_class == "folklore":
                folk_lengths.append(ln)
    length_d1_auc = compute_auc(misc_lengths, truth_lengths)
    length_d2_auc = compute_auc(folk_lengths, truth_lengths)

    d1_delta = (d1_auc - length_d1_auc) if (d1_auc == d1_auc and length_d1_auc == length_d1_auc) else float("nan")
    d2_delta = (d2_auc - length_d2_auc) if (d2_auc == d2_auc and length_d2_auc == length_d2_auc) else float("nan")
    d3_delta_required = bars["D3_length_control_delta"]
    d3_pass = ((d1_delta == d1_delta) and d1_delta >= d3_delta_required
               and (d2_delta == d2_delta) and d2_delta >= d3_delta_required)

    metrics = {
        "n_misconception": len(misc_scores),
        "n_truth": len(truth_scores),
        "n_folklore": len(folk_scores),
        "mean_misconception_score": round(statistics.fmean(misc_scores), 4) if misc_scores else None,
        "mean_truth_score": round(statistics.fmean(truth_scores), 4) if truth_scores else None,
        "mean_folklore_score": round(statistics.fmean(folk_scores), 4) if folk_scores else None,
        "D1_misconception_AUC": round(d1_auc, 4) if d1_auc == d1_auc else None,
        "D2_folklore_AUC": round(d2_auc, 4) if d2_auc == d2_auc else None,
        # 7.7.8 length-control diagnostics:
        "length_oracle_misconception_AUC": round(length_d1_auc, 4) if length_d1_auc == length_d1_auc else None,
        "length_oracle_folklore_AUC": round(length_d2_auc, 4) if length_d2_auc == length_d2_auc else None,
        "D1_minus_length_AUC": round(d1_delta, 4) if d1_delta == d1_delta else None,
        "D2_minus_length_AUC": round(d2_delta, 4) if d2_delta == d2_delta else None,
    }

    bar_results = {
        "D1_misconception_AUC": (d1_auc == d1_auc) and d1_auc >= bars["D1_misconception_AUC"],
        "D2_folklore_AUC": (d2_auc == d2_auc) and d2_auc >= bars["D2_folklore_AUC"],
        "D3_length_control_delta": bool(d3_pass),
    }
    n_passed = sum(bar_results.values())

    return GauntletResult(
        submission_name=submission.name,
        task="detection",
        benchmark_version=benchmark.get("version", "unknown"),
        n_items=len(records),
        metrics=metrics,
        bars=bars,
        bar_results=bar_results,
        overall_pass=all(bar_results.values()),
        n_passed=n_passed,
        n_total_bars=len(bar_results),
        per_item_predictions=predictions[:10],
    )


# ──────────────────────────────────────────────────────────────────────
# Baseline entry — the seven-method floor as Baseline-001
# ──────────────────────────────────────────────────────────────────────

BASELINE_ENTRY = {
    "rank": "Baseline-001 (the floor)",
    "submitter": "Fathom Lab",
    "method": "seven-method pre-registered arc",
    "tasks": ["detection (4 methods)", "classification (1 method)", "constructive injection (2 methods)"],
    "n_bars_passed": "0 / 7",
    "summary": (
        "Four detection methods (Dark Matter perturbation-fragility, CVPD agreement-fracture, "
        "JD justification-divergence, ICT neutral injection) — all closed-negative on the dark "
        "core. One classification method (sentence-transformer + balanced LR) — FAIL K2 + K3, "
        "20% recall on cross-corpus folklore. Two constructive variants (ICT-folklore, "
        "ICT-authoritative) — SHORTFALL on the same corpus (28/30 already corrected baseline)."
    ),
    "commits": "bcd4208..a6d7a7e",
    "paper": "papers/PAPER_decorrelation_ceiling_2026_05_27.md",
    "capstone": "papers/REPORT_decorrelation_ceiling_v2_2026_05_27.md",
    "submission_date": "2026-05-27",
}


# ──────────────────────────────────────────────────────────────────────
# Internal — majority-class baseline for tests
# ──────────────────────────────────────────────────────────────────────

def _majority_baseline_predict(question: str) -> Dict[str, str]:
    """A trivial majority-class classifier: predicts 'truth' for every input.

    Used as a sanity-check submission in the test suite. By construction this
    cannot pass any of the bars (folklore F1 = 0; cross-corpus folklore F1 = 0;
    accuracy ≈ truth-class frequency in the benchmark).
    """
    return {"class": "truth", "confidence": 1.0}


def _zero_baseline_detect(question: str, response: str) -> Dict[str, float]:
    """A trivial constant-score detector: returns 0 for everything.

    Cannot pass any detection bar (AUCs will be 0.5 by definition of constant)."""
    return {"score": 0.0}


def _length_oracle_detect(question: str, response: str) -> Dict[str, float]:
    """The length-only detector: scores equal to the response word count.

    Used as the D3 length-control floor in run_detection_gauntlet. A real
    detector must beat this oracle's AUC by at least D3_length_control_delta
    (0.10 by default) to demonstrate signal beyond the documented benchmark
    length-confound. Added 7.7.8 after Baseline-007's accidental PASS exposed
    the artifact (see LEADERBOARD.md Baseline-007 row + the 7.7.8 CHANGELOG)."""
    return {"score": float(len((response or "").split()))}

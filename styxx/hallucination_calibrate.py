# -*- coding: utf-8 -*-
"""
styxx.hallucination_calibrate — threshold calibration for production.

Given a labeled validation set of (prompt, expected_label) pairs where
expected_label ∈ {"fabrication", "honest"}, this module computes the
probe readings, then picks the threshold that maximizes a chosen
objective (F1 by default; also supports precision-at-high-recall for
safety-critical deployments that cannot tolerate false negatives).

Produces a `CalibratedThreshold` object that the caller passes to
`styxx.hallucination.detect_hallucination(..., threshold=ct.value)`.

Usage
-----
    from styxx.hallucination_calibrate import calibrate_from_labels

    validation_set = [
        ("What is the capital of France?", "honest"),
        ("Summarize paper X by author Y published in ...", "fabrication"),
        ...
    ]

    ct = calibrate_from_labels(
        model=mdl, tokenizer=tok,
        validation_set=validation_set,
        probe_task="confab_behavioral",
        objective="f1",          # or "precision_at_recall_0.95"
    )
    print(ct.value, ct.auc, ct.f1_at_threshold)

    # Now use ct.value as threshold
    verdict = detect_hallucination(..., threshold=ct.value)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .hallucination import hallucination_verdict
from .residual_probe.intervene import InterveneProbe


@dataclass
class CalibratedThreshold:
    """Result of threshold calibration."""
    value: float
    objective: str
    auc: float
    f1_at_threshold: float
    precision_at_threshold: float
    recall_at_threshold: float
    n_positives: int
    n_negatives: int
    probe_task: str
    probe_layer: int
    per_prompt_scores: List[Tuple[str, str, float]] = field(default_factory=list)


def calibrate_from_labels(
    model,
    tokenizer,
    validation_set: List[Tuple[str, str]],
    *,
    probe_task: str = "confab_behavioral",
    objective: str = "f1",
    apply_chat_template: bool = True,
    max_new_tokens: int = 80,
) -> CalibratedThreshold:
    """Compute the best threshold over a validation set.

    Parameters
    ----------
    validation_set : list of (prompt, label)
        label ∈ {"fabrication", "honest"}
    objective : str
        "f1" — maximize F1 at the chosen threshold
        "precision_at_recall_0.95" — max precision with recall ≥ 0.95
        "precision_at_recall_0.99" — recall ≥ 0.99 (safety-critical)
        "max_separation" — maximize gap between class means
    """
    import torch

    if objective not in (
        "f1", "precision_at_recall_0.95",
        "precision_at_recall_0.99", "max_separation",
    ):
        raise ValueError(f"unknown objective {objective!r}")

    # Run model on each prompt, collect probe reading + expected label.
    scores: List[Tuple[str, str, float]] = []
    for prompt, label in validation_set:
        if label not in ("fabrication", "honest"):
            raise ValueError(
                f"label must be 'fabrication' or 'honest', got {label!r}"
            )
        # Generate a short response, then score per our stream API.
        # For calibration, a short generation suffices; we take the
        # max risk observed.
        from .hallucination import stream_with_risk
        max_risk = 0.0
        for reading in stream_with_risk(
            model=model, tokenizer=tokenizer, prompt=prompt,
            probe_task=probe_task,
            threshold=1.0,    # disable flagging during stream
            max_new_tokens=max_new_tokens,
            apply_chat_template=apply_chat_template,
        ):
            if reading.risk > max_risk:
                max_risk = reading.risk
        scores.append((prompt, label, max_risk))

    n_pos = sum(1 for _, l, _ in scores if l == "fabrication")
    n_neg = sum(1 for _, l, _ in scores if l == "honest")
    if n_pos < 2 or n_neg < 2:
        raise ValueError(
            "need at least 2 positives and 2 negatives; got "
            f"{n_pos}/{n_neg}"
        )

    # Compute AUC using the rank test.
    import numpy as np
    pos_scores = np.array([s for _, l, s in scores if l == "fabrication"])
    neg_scores = np.array([s for _, l, s in scores if l == "honest"])
    wins = sum(1 for p in pos_scores for n in neg_scores if p > n)
    ties = sum(1 for p in pos_scores for n in neg_scores if p == n)
    auc = (wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores))

    # Scan thresholds and compute (precision, recall, f1) for each.
    candidates = sorted(set(float(s) for _, _, s in scores))
    # Use midpoints to avoid boundary ties:
    mids = []
    for i in range(len(candidates) - 1):
        mids.append((candidates[i] + candidates[i + 1]) / 2.0)
    if not mids:
        mids = candidates

    best_thr = candidates[len(candidates) // 2] if not mids else mids[0]
    best_metric = -1.0
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0

    for thr in mids:
        tp = sum(1 for _, l, s in scores
                 if l == "fabrication" and s > thr)
        fp = sum(1 for _, l, s in scores
                 if l == "honest" and s > thr)
        fn = sum(1 for _, l, s in scores
                 if l == "fabrication" and s <= thr)
        tn = sum(1 for _, l, s in scores
                 if l == "honest" and s <= thr)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        if objective == "f1":
            metric = f1
        elif objective == "precision_at_recall_0.95":
            metric = prec if rec >= 0.95 else -1.0
        elif objective == "precision_at_recall_0.99":
            metric = prec if rec >= 0.99 else -1.0
        elif objective == "max_separation":
            pos_above = sum(1 for _, l, s in scores
                            if l == "fabrication" and s > thr)
            neg_above = sum(1 for _, l, s in scores
                            if l == "honest" and s > thr)
            metric = pos_above / max(n_pos, 1) - neg_above / max(n_neg, 1)

        if metric > best_metric:
            best_metric = metric
            best_thr = thr
            best_f1 = f1
            best_prec = prec
            best_rec = rec

    # Resolve probe layer (for the result record)
    probe = None
    try:
        cfg = getattr(model, "config", None)
        name = (getattr(cfg, "_name_or_path", None)
                if cfg else None)
        if name:
            probe = InterveneProbe.from_pretrained(model=name,
                                                    task=probe_task)
    except Exception:
        pass
    probe_layer = probe.layer if probe is not None else -1

    return CalibratedThreshold(
        value=best_thr,
        objective=objective,
        auc=auc,
        f1_at_threshold=best_f1,
        precision_at_threshold=best_prec,
        recall_at_threshold=best_rec,
        n_positives=n_pos,
        n_negatives=n_neg,
        probe_task=probe_task,
        probe_layer=probe_layer,
        per_prompt_scores=scores,
    )


__all__ = ["CalibratedThreshold", "calibrate_from_labels"]

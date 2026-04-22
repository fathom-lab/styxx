# -*- coding: utf-8 -*-
"""
Fit a logistic-regression meta-classifier on top of the guardrail
signal vector, trained on a HaluEval dev split and evaluated on a
held-out test split. This replaces hand-tuned fusion weights with
learned weights and gives a proper calibration curve.

Usage
-----
  python benchmarks/hallucination_test/guardrail_calibrate.py \\
    --n_dev 200 --n_test 100 --seed_dev 11 --seed_test 12 \\
    --probe_task halueval
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_halueval_seeded(n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data",
                      streaming=True)
    rng = random.Random(seed)
    rows = []
    for row in ds:
        if not all(row.get(k) for k in
                   ("knowledge", "question", "right_answer",
                    "hallucinated_answer")):
            continue
        rows.append(row)
        if len(rows) >= n * 4:
            break
    rng.shuffle(rows)
    return rows[:n]


def collect_signals(rows, probe_scorer=None, entity_verify=True):
    """For each HaluEval item, score both right and hallucinated answers
    and return (feature_matrix, labels, action_list)."""
    from styxx.guardrail import check
    import numpy as np

    X = []
    y = []
    actions = []
    t0 = time.time()
    for i, row in enumerate(rows):
        prompt = f"Question: {row['question']}"
        for answer, lbl in ((row["right_answer"], 0),
                             (row["hallucinated_answer"], 1)):
            v = check(
                prompt=prompt,
                response=answer,
                reference=row["knowledge"],
                use_entity_verify=entity_verify,
                use_grounding=True,
                use_probe=(probe_scorer is not None),
                probe_scorer=probe_scorer,
            )
            # Pick out signal values
            sig = {s.name: s.value for s in v.signals}
            feat = [
                sig.get("text_claim_risk", 0.0),
                sig.get("entity_unverified_frac", 0.0),
                sig.get("knowledge_grounding", 0.5),
                sig.get("probe_confab", 0.5),
            ]
            X.append(feat)
            y.append(lbl)
            actions.append(v.action)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  [{time.time()-t0:.0f}s]")
    return np.array(X), np.array(y), actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_dev", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=100)
    ap.add_argument("--seed_dev", type=int, default=11)
    ap.add_argument("--seed_test", type=int, default=12)
    ap.add_argument("--probe_task", default="halueval")
    ap.add_argument("--entity_verify", action="store_true", default=True)
    args = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    dev_rows = load_halueval_seeded(args.n_dev, args.seed_dev)
    test_rows = load_halueval_seeded(args.n_test, args.seed_test)
    # Dedupe test against dev to avoid leakage
    dev_q = {r["question"] for r in dev_rows}
    test_rows = [r for r in test_rows if r["question"] not in dev_q]
    print(f"[1/4] dev={len(dev_rows)} test={len(test_rows)} "
          f"(dedupped by question)")

    print("[2/4] loading probe scorer ...")
    from styxx.guardrail.probe_signal import ProbeScorer
    probe_scorer = ProbeScorer(probe_task=args.probe_task)
    print(f"  probe layer {probe_scorer.layer} "
          f"AUC {probe_scorer.probe.auc_validation}")

    print(f"[3/4] scoring dev signals ...")
    X_dev, y_dev, _ = collect_signals(dev_rows, probe_scorer=probe_scorer,
                                       entity_verify=args.entity_verify)
    print(f"  dev X={X_dev.shape}")

    print(f"  fitting logistic regression meta-classifier ...")
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                              solver="liblinear")
    clf.fit(X_dev, y_dev)
    print(f"  coefficients: {dict(zip(['text','entity','grounding','probe'], clf.coef_[0].tolist()))}")
    print(f"  intercept: {clf.intercept_[0]:.3f}")

    # Self-AUC on dev
    dev_probs = clf.predict_proba(X_dev)[:, 1]
    dev_auc = roc_auc_score(y_dev, dev_probs)
    print(f"  dev AUC: {dev_auc:.3f}")

    print(f"\n[4/4] evaluating on test (n={len(test_rows)}) ...")
    X_test, y_test, _ = collect_signals(test_rows, probe_scorer=probe_scorer,
                                         entity_verify=args.entity_verify)
    test_probs = clf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)

    # Per-threshold analysis on test
    print()
    print("=" * 72)
    print(f"TEST AUC (held out, n={len(test_rows)}): {test_auc:.4f}")
    print(f"(dev AUC: {dev_auc:.4f})")
    print()
    print("Test-set risk distribution:")
    import numpy as np
    right_probs = test_probs[y_test == 0]
    halluc_probs = test_probs[y_test == 1]
    print(f"  Right answers:         mean={right_probs.mean():.3f}  "
          f"median={np.median(right_probs):.3f}  "
          f"95%ile={np.quantile(right_probs, 0.95):.3f}")
    print(f"  Hallucinated answers:  mean={halluc_probs.mean():.3f}  "
          f"median={np.median(halluc_probs):.3f}  "
          f"5%ile={np.quantile(halluc_probs, 0.05):.3f}")

    print()
    print("Threshold analysis (precision / recall on test set):")
    for thresh in (0.3, 0.5, 0.6, 0.7, 0.8):
        tp = int(np.sum((test_probs > thresh) & (y_test == 1)))
        fp = int(np.sum((test_probs > thresh) & (y_test == 0)))
        fn = int(np.sum((test_probs <= thresh) & (y_test == 1)))
        tn = int(np.sum((test_probs <= thresh) & (y_test == 0)))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"  thresh={thresh:.1f}  TP={tp:>3d} FP={fp:>3d} "
              f"FN={fn:>3d} TN={tn:>3d}  "
              f"prec={prec:.3f} rec={rec:.3f} F1={f1:.3f}")

    out_path = (ROOT / "benchmarks" / "hallucination_test"
                / "results" / "guardrail_calibrated.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_dev": len(dev_rows),
        "n_test": len(test_rows),
        "dev_auc": float(dev_auc),
        "test_auc": float(test_auc),
        "probe_task": args.probe_task,
        "coefficients": dict(zip(
            ['text_claim_risk', 'entity_unverified_frac',
             'knowledge_grounding', 'probe_confab'],
            clf.coef_[0].tolist()
        )),
        "intercept": float(clf.intercept_[0]),
        "right_mean": float(right_probs.mean()),
        "halluc_mean": float(halluc_probs.mean()),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

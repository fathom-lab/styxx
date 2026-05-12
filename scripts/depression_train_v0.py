"""depression_train_v0.py — train the first cognometric instrument on
biological-cognition data.

This is the substrate-bridge experiment from *Every Mind Leaves Vitals* §3.
The 9 LLM cognometric instruments all show K=1 phase-transition signature.
Does the same signature replicate when the substrate is biological cognition
(Reddit r/depression posts vs other-psychiatric subreddits)?

Pipeline (mirror of scripts/sycophancy_train_v0.py):
  1. Load benchmarks/data/depression/responses_v0.jsonl
     (6000 r/depression POS + 6000 r/{adhd,aspergers,ocd,ptsd} NEG, length-matched)
  2. Featurize via extract_depression_features
  3. 5-fold CV with calibrated logistic regression
  4. Greedy forward feature ablation — record AUC at each K
  5. find_critical_k → critical_K, critical_feature, ΔAUC@K
  6. Write benchmarks/depression_feature_scaling.json + weights JSON

If a low-K phase transition appears (Δ ≥ 0.20 in a single feature added),
the substrate-bridge claim is empirically confirmed for depression. If it
doesn't appear, that's a falsification — equally publishable, constrains
the universality claim.

Usage:
    python scripts/depression_train_v0.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from styxx.guardrail.depression_signals import extract_depression_features  # noqa: E402

DATA_PATH = ROOT / "benchmarks" / "data" / "depression" / "responses_v0.jsonl"
OUT_DIR = ROOT / "benchmarks"


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def featurize(rows: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix + labels from text rows."""
    if not rows:
        raise SystemExit("no rows to featurize")
    feature_names = sorted(extract_depression_features("x").keys())
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        feats = extract_depression_features(r["text"])
        for j, name in enumerate(feature_names):
            X[i, j] = feats[name]
        y[i] = int(r["label_depression"])
    return X, y, feature_names


def train_full(X: np.ndarray, y: np.ndarray, feature_names: List[str], seed: int = 0) -> Dict:
    """Train calibrated LR with 5-fold CV. Return fold AUCs + final fit."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(
            C=1.0, max_iter=2000, random_state=seed, solver="lbfgs"
        ).fit(scaler.transform(X[tr]), y[tr])
        proba = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        auc = roc_auc_score(y[te], proba)
        fold_aucs.append(auc)
        print(f"  fold {fold}: AUC {auc:.4f}")
    print(f"  mean AUC: {np.mean(fold_aucs):.4f} (std {np.std(fold_aucs):.4f})")

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(
        C=1.0, max_iter=2000, random_state=seed, solver="lbfgs"
    ).fit(scaler.transform(X), y)
    return {
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "feature_names": feature_names,
        "coefs": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def feature_ablation(X: np.ndarray, y: np.ndarray, feature_names: List[str], seed: int = 0) -> List[Dict]:
    """Greedy forward feature selection, recording AUC at each K."""
    n_features = X.shape[1]
    selected: List[int] = []
    remaining = list(range(n_features))
    history = []
    prev_auc = 0.5

    def auc_for(idxs):
        if not idxs:
            return 0.5
        Xi = X[:, idxs]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        aucs = []
        for tr, te in skf.split(Xi, y):
            scaler = StandardScaler().fit(Xi[tr])
            clf = LogisticRegression(
                C=1.0, max_iter=2000, random_state=seed
            ).fit(scaler.transform(Xi[tr]), y[tr])
            p = clf.predict_proba(scaler.transform(Xi[te]))[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        return float(np.mean(aucs))

    while remaining:
        best_idx, best_auc = None, -1.0
        for cand in remaining:
            auc = auc_for(selected + [cand])
            if auc > best_auc:
                best_idx, best_auc = cand, auc
        selected.append(best_idx)
        remaining.remove(best_idx)
        K = len(selected)
        history.append({
            "K": K,
            "feature_added": feature_names[best_idx],
            "mean_auc": best_auc,
            "delta_from_prev": best_auc - prev_auc,
            "selected_features": [feature_names[i] for i in selected],
        })
        print(f"  K={K:2d} +{feature_names[best_idx]:32s} AUC={best_auc:.4f} (Δ {best_auc-prev_auc:+.4f})")
        prev_auc = best_auc
    return history


def find_critical_k(history: List[Dict]) -> Dict:
    """Largest single-feature jump = critical-K phase transition."""
    biggest = max(history, key=lambda h: h["delta_from_prev"])
    return {
        "critical_K": biggest["K"],
        "critical_feature": biggest["feature_added"],
        "delta_auc_at_K": biggest["delta_from_prev"],
        "auc_at_K": biggest["mean_auc"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"=== depression v0 training (seed={args.seed}) ===")
    print(f"loading {DATA_PATH}...")
    rows = load_rows(DATA_PATH)
    print(f"  loaded {len(rows)} rows")
    print(f"  pos (label=1): {sum(1 for r in rows if r['label_depression']==1)}")
    print(f"  neg (label=0): {sum(1 for r in rows if r['label_depression']==0)}")

    print("\n[featurize]")
    X, y, feature_names = featurize(rows)
    print(f"  shape: X={X.shape}, y={y.shape}")
    print(f"  features ({len(feature_names)}): {feature_names}")

    print("\n[full-feature 5-fold CV]")
    full = train_full(X, y, feature_names, seed=args.seed)

    print("\n[K-ablation: greedy forward feature selection]")
    history = feature_ablation(X, y, feature_names, seed=args.seed)
    crit = find_critical_k(history)
    print(f"\n  CRITICAL: K={crit['critical_K']} feature={crit['critical_feature']}")
    print(f"           AUC@K = {crit['auc_at_K']:.4f}, ΔAUC = {crit['delta_auc_at_K']:+.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scaling_path = OUT_DIR / "depression_feature_scaling.json"
    with open(scaling_path, "w", encoding="utf-8") as f:
        json.dump({
            "instrument": "depression_v0",
            "dataset": "solomonk/reddit_mental_health_posts (r/depression vs r/{adhd,aspergers,ocd,ptsd})",
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "n_features": len(feature_names),
            "seed": args.seed,
            "full_mean_auc": full["mean_auc"],
            "full_std_auc": full["std_auc"],
            "history": history,
            "critical": crit,
        }, f, indent=2)
    print(f"\n  wrote {scaling_path}")

    weights_path = OUT_DIR / "depression_weights_v0.json"
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)
    print(f"  wrote {weights_path}")


if __name__ == "__main__":
    main()

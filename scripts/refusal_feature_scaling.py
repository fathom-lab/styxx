"""Cross-instrument phase-transition probe — refusal detector.

Research question: the drift phase-transition result (2026-04-23)
found that each drift class becomes detectable at a specific K
(feature count) rather than smoothly with K. Does the pattern
generalize? This script runs the same top-K ablation on the v1
refusal detector (cognometric instrument #2).

If refusal also phase-transitions, the pattern is NOT drift-specific
— it's a property of cognometric instruments in general, which
strengthens the Law II claim (cross-substrate universality) in
our manifesto.

Dataset: JBB-Llama-1B, n=80 labeled (prompt, response, refuse)
samples, same as refusal v1 training set.

Features: the 18 refusal_signals features shipped in
styxx.guardrail.calibrated_weights_refusal_v1.

Ablations:
  (1) TOP-K by |coef| from full-model baseline
  (2) RANDOM subsets (3 seeds per K) — null expectation

CV: 5-fold stratified, RANDOM_STATE=0 (consistent with drift).

Phase-transition criterion: overall AUC jump >= 0.10 between
consecutive K values in the top-K ablation.

Usage:
  python scripts/refusal_feature_scaling.py

Output:
  benchmarks/refusal_feature_scaling.json  — raw numbers
  papers/refusal_phase_transitions.md      — writeup
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# scripts/ is not a package — import the sibling by prepending to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from refusal_baseline_v1 import extract_refusal_features  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA_PATH = REPO / "styxx" / "residual_probe" / "atlas" / "compliance_labels_llama_1b.json"
OUT_JSON = REPO / "benchmarks" / "refusal_feature_scaling.json"
OUT_MD = REPO / "papers" / "refusal_phase_transitions.md"

RANDOM_STATE = 0
N_SPLITS = 5

FEATURE_NAMES = [
    "refusal_density", "hedge_density", "confidence_density",
    "uncertainty_density", "entity_density", "claim_density",
    "reasoning_marker_density", "sentence_length_mean", "unique_ratio",
    "disclaimer_density", "safety_flag_density", "redirect_density",
    "normative_density", "short_response", "starts_with_sorry",
    "starts_with_i_cant", "starts_with_normative", "log_word_count",
]
N_FEATURES = len(FEATURE_NAMES)


def load_rows():
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for item in json.load(f):
            prompt = item.get("prompt", "") or ""
            response = item.get("response_excerpt", "") or ""
            label = int(item.get("label", item.get("complied", 0)))
            # label=1 means refusal in this dataset (aligned with v1 convention)
            rows.append({"prompt": prompt, "response": response, "label": label})
    return rows


def feature_matrix(rows):
    X = []
    for r in rows:
        feats = extract_refusal_features(r["prompt"], r["response"])
        X.append([feats[name] for name in FEATURE_NAMES])
    return np.array(X, dtype=float)


def cv_auc(X, y, feature_idx):
    """5-fold CV pooled AUC restricted to a feature subset."""
    if len(feature_idx) == 0:
        return 0.5  # no features -> chance
    Xs = X[:, feature_idx]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    labels = []
    for tr, te in skf.split(Xs, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(Xs[tr])
        X_te = sc.transform(Xs[te])
        clf = LogisticRegression(
            C=1.0, max_iter=2000, random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        clf.fit(X_tr, y[tr])
        p = clf.predict_proba(X_te)[:, 1]
        scores.extend(p)
        labels.extend(y[te])
    return float(roc_auc_score(labels, scores))


def full_model_importance(X, y):
    """Train the full 18-feature model, return (feature, |coef|) ranked."""
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(
        C=1.0, max_iter=2000, random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    clf.fit(Xs, y)
    ranked = sorted(
        zip(FEATURE_NAMES, clf.coef_[0], range(N_FEATURES)),
        key=lambda kv: -abs(kv[1]),
    )
    return ranked  # list of (name, coef, orig_idx)


def main():
    rows = load_rows()
    print(f"loaded {len(rows)} samples")
    y = np.array([r["label"] for r in rows])
    print(f"  class balance: {sum(y==0)} no-refuse, {sum(y==1)} refuse")

    X = feature_matrix(rows)
    print(f"  feature matrix: {X.shape}")

    # Full-model importance ranking
    ranked = full_model_importance(X, y)
    print()
    print("Full 18-feature importance ranking (|coef| desc):")
    for name, c, idx in ranked:
        print(f"  {name:<26s}  {c:+.3f}")
    full_auc = cv_auc(X, y, list(range(N_FEATURES)))
    print(f"\nFull-model 5-fold CV AUC: {full_auc:.4f}")

    # TOP-K ablation
    print()
    print(f"{'='*70}")
    print("Ablation 1: TOP-K by |coef|")
    print(f"{'='*70}")
    top_k_results = []
    prev_auc = None
    for K in range(0, N_FEATURES + 1):
        if K == 0:
            auc = 0.5
            added = None
        else:
            idx = [r[2] for r in ranked[:K]]
            auc = cv_auc(X, y, idx)
            added = ranked[K-1][0]
        delta = (auc - prev_auc) if prev_auc is not None else 0.0
        flag = "  <-- phase transition (>= 0.10)" if abs(delta) >= 0.10 else ""
        print(f"  K={K:2d}  AUC={auc:.4f}  delta={delta:+.4f}  added={added}{flag}")
        top_k_results.append({
            "K": K,
            "auc": auc,
            "added": added,
            "delta_from_prev": delta,
            "phase_transition": bool(abs(delta) >= 0.10),
        })
        prev_auc = auc

    # RANDOM subsets (3 seeds per K, K in [1, N])
    print()
    print(f"{'='*70}")
    print("Ablation 2: RANDOM subsets (3 seeds per K)")
    print(f"{'='*70}")
    rng = random.Random(RANDOM_STATE)
    random_results = []
    for K in range(1, N_FEATURES + 1):
        aucs = []
        for seed in range(3):
            r_rng = random.Random(100 * K + seed)
            idx = r_rng.sample(range(N_FEATURES), K)
            auc = cv_auc(X, y, idx)
            aucs.append(auc)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))
        print(f"  K={K:2d}  random AUC mean={mean_auc:.4f} ± {std_auc:.4f}")
        random_results.append({
            "K": K,
            "random_aucs": aucs,
            "random_mean": mean_auc,
            "random_std": std_auc,
        })

    # Save artifact
    out = {
        "methodology": "refusal feature scaling — 18-feature top-K + random-subsets, 5-fold CV seed=0",
        "instrument": "refusal (cognometric instrument #2)",
        "dataset": "JBB-Llama-1B (n=80)",
        "n_samples": len(rows),
        "class_balance_no_refuse_refuse": [int(sum(y == 0)), int(sum(y == 1))],
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "full_model_auc": full_auc,
        "full_model_importance": [
            {"rank": i + 1, "feature": r[0], "coef": r[1]}
            for i, r in enumerate(ranked)
        ],
        "top_k": top_k_results,
        "random_subsets": random_results,
        "phase_transition_threshold": 0.10,
        "phase_transitions_found": [
            {"K": r["K"], "added": r["added"], "delta": r["delta_from_prev"]}
            for r in top_k_results if r["phase_transition"]
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"wrote -> {OUT_JSON.relative_to(REPO)}")

    # Summary verdict
    print()
    print(f"{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    n_jumps = len(out["phase_transitions_found"])
    if n_jumps > 0:
        print(f"Found {n_jumps} phase transition(s):")
        for pt in out["phase_transitions_found"]:
            print(f"  K={pt['K']}  +{pt['added']}  delta={pt['delta']:+.4f}")
    else:
        print("No phase transitions found (no K->K+1 AUC jumps >= 0.10).")


if __name__ == "__main__":
    main()

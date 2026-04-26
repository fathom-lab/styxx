"""Per-substrate ablation for sycophancy v0.

Tests whether the K=1 phase-transition on `superlative_density` holds
within each substrate (NLP survey, philpapers2020, political typology),
or is an artifact of pooling across substrates.

Reads the cached responses from training (no new API calls).
Outputs: benchmarks/sycophancy_per_substrate_ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from styxx.guardrail.sycophancy_signals import extract_sycophancy_features

CACHE = ROOT / "benchmarks" / "data" / "sycophancy" / "responses_v0.jsonl"
OUT = ROOT / "benchmarks" / "sycophancy_per_substrate_ablation.json"


def load_responses():
    with open(CACHE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def featurize(rows):
    feature_names = list(extract_sycophancy_features("x", "x").keys())
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        feats = extract_sycophancy_features(r["question"], r["response"])
        for j, name in enumerate(feature_names):
            X[i, j] = feats[name]
        y[i] = int(r["label_sycophantic"])
    return X, y, feature_names


def cv_auc(X, y, idxs, seed=0, n_splits=5):
    """5-fold CV mean AUC over a feature subset (by index)."""
    if not idxs:
        return 0.5
    Xi = X[:, idxs]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(Xi, y):
        scaler = StandardScaler().fit(Xi[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
            scaler.transform(Xi[tr]), y[tr]
        )
        p = clf.predict_proba(scaler.transform(Xi[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


def greedy_forward_ablation(X, y, feature_names, seed=0):
    n = X.shape[1]
    selected, remaining = [], list(range(n))
    history, prev = [], 0.5
    while remaining:
        best_i, best_auc = None, -1.0
        for cand in remaining:
            a = cv_auc(X, y, selected + [cand], seed=seed)
            if a > best_auc:
                best_i, best_auc = cand, a
        selected.append(best_i)
        remaining.remove(best_i)
        history.append({
            "K": len(selected),
            "feature_added": feature_names[best_i],
            "mean_auc": best_auc,
            "delta_from_prev": best_auc - prev,
        })
        prev = best_auc
    return history


def find_critical(history):
    biggest = max(history, key=lambda h: h["delta_from_prev"])
    return {
        "critical_K": biggest["K"],
        "critical_feature": biggest["feature_added"],
        "delta_auc_at_K": biggest["delta_from_prev"],
        "auc_at_K": biggest["mean_auc"],
    }


def main():
    rows = load_responses()
    print(f"loaded {len(rows)} responses")

    by_sub = {}
    for r in rows:
        by_sub.setdefault(r["substrate"], []).append(r)

    out = {"substrates": {}}
    for sub, sub_rows in sorted(by_sub.items()):
        print(f"\n=== substrate: {sub}  (n={len(sub_rows)}) ===")
        X, y, names = featurize(sub_rows)
        print(f"  feature matrix: {X.shape}, balance: {y.mean():.3f}")
        history = greedy_forward_ablation(X, y, names)
        crit = find_critical(history)
        for h in history:
            print(f"  K={h['K']:2d} +{h['feature_added']:30s} AUC={h['mean_auc']:.4f} (Δ {h['delta_from_prev']:+.4f})")
        print(f"  → critical_K={crit['critical_K']} on {crit['critical_feature']} (Δ {crit['delta_auc_at_K']:+.4f})")
        out["substrates"][sub] = {
            "n": int(y.shape[0]),
            "full_auc": history[-1]["mean_auc"],
            "history": history,
            "critical": crit,
        }

    # Pooled (all substrates)
    print(f"\n=== pooled (all substrates) ===")
    X, y, names = featurize(rows)
    history = greedy_forward_ablation(X, y, names)
    crit = find_critical(history)
    for h in history:
        print(f"  K={h['K']:2d} +{h['feature_added']:30s} AUC={h['mean_auc']:.4f}")
    print(f"  → critical_K={crit['critical_K']} on {crit['critical_feature']} (Δ {crit['delta_auc_at_K']:+.4f})")
    out["pooled"] = {"n": int(y.shape[0]), "history": history, "critical": crit}

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")
    print(f"\nsubstrate_K_var summary:")
    for sub, data in out["substrates"].items():
        c = data["critical"]
        print(f"  {sub:30s} K={c['critical_K']} {c['critical_feature']:25s} Δ{c['delta_auc_at_K']:+.4f} AUC={c['auc_at_K']:.4f}")


if __name__ == "__main__":
    main()

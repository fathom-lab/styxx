"""Refusal detector v2 — scaled training.

Training set (combined):
  - JBB-Llama-1B         n=80   (our original, committed)
  - JBB judge_comparison n=300  (public, 12 models × 3 human annotators)
  Total: n=380

Held-out test (unchanged from v1 — avoids double-dipping):
  - XSTest v2 × 5 model families, n=2,250

Label convention:
  - Our data: label=1 for refusal, label=0 for compliance
  - JBB judge: human_majority=0 means SAFE (model refused) → label=1 for us
              human_majority=1 means UNSAFE (model complied) → label=0 for us
  (inverted semantics since JBB labels "safety score" not "refusal")

Re-fit the same 18 features; compare v1 (n=80) vs v2 (n=380) AUC per
held-out model family. Document the delta.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.refusal_baseline_v1 import extract_refusal_features
from scripts.refusal_xstest_heldout import binarize_xstest_label, prepare

REPO = Path(__file__).resolve().parents[1]
JBB_LLAMA_PATH = REPO / "styxx" / "residual_probe" / "atlas" / "compliance_labels_llama_1b.json"


def load_jbb_judge_comparison():
    """Returns list of dicts with {prompt, response, label} where
    label=1 is refusal and label=0 is compliance."""
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", split="test")
    out = []
    for s in ds:
        # human_majority: 0 means safe (refused), 1 means unsafe (complied)
        hm = s.get("human_majority")
        if hm is None:
            continue
        refused = 1 if int(hm) == 0 else 0
        prompt = s.get("prompt") or s.get("goal") or ""
        response = s.get("target_response") or ""
        if not prompt or not response:
            continue
        out.append({"prompt": prompt, "response": response, "label": refused})
    return out


def load_jbb_llama_1b():
    """Our existing 80 samples."""
    raw = json.loads(JBB_LLAMA_PATH.read_text(encoding="utf-8"))
    out = []
    for s in raw:
        out.append({
            "prompt": s["prompt"],
            "response": s.get("response_excerpt", ""),
            "label": int(s["label"]),  # already 1=refuse, 0=comply
        })
    return out


def features_matrix(samples):
    """Build (X, y) matrix from a list of {prompt, response, label}."""
    X, y = [], []
    feat_names = None
    for s in samples:
        f = extract_refusal_features(s["prompt"], s["response"])
        if feat_names is None:
            feat_names = list(f.keys())
        X.append([f[k] for k in feat_names])
        y.append(s["label"])
    return np.array(X), np.array(y), feat_names


def main():
    print("=" * 72)
    print("REFUSAL DETECTOR v2 — SCALED TRAINING")
    print("=" * 72)

    # Load both training sources
    llama = load_jbb_llama_1b()
    jbb_judge = load_jbb_judge_comparison()
    combined = llama + jbb_judge

    print(f"\nTraining-set composition:")
    print(f"  JBB-Llama-1B            n={len(llama):>4d}   refuse={sum(s['label'] for s in llama):>3d}")
    print(f"  JBB judge_comparison    n={len(jbb_judge):>4d}   refuse={sum(s['label'] for s in jbb_judge):>3d}")
    print(f"  combined (v2)           n={len(combined):>4d}   refuse={sum(s['label'] for s in combined):>3d}")

    X_train, y_train, feat_names = features_matrix(combined)
    print(f"\n  feature matrix: {X_train.shape}  class_bal={int(y_train.mean() * 100)}% refuse")

    # Fit the same LR (C=1.0, max_iter=1000, random_state=0 for reproducibility)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    clf.fit(X_train_s, y_train)

    train_probs = clf.predict_proba(X_train_s)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs)
    print(f"  training-set fit AUC (reference): {train_auc:.4f}")

    # Hold out: XSTest × 5 model families (unchanged from v1 eval)
    print("\n" + "=" * 72)
    print("HELD-OUT EVAL — XSTest v2, 5 model families (same as v1)")
    print("=" * 72)

    splits = ["gpt4", "llama2new", "llama2orig", "mistralguard", "mistralinstruct"]
    # v1 numbers for side-by-side
    v1_aucs = {
        "gpt4":             0.9759,
        "llama2new":        0.8741,
        "llama2orig":       0.7832,
        "mistralguard":     0.7797,
        "mistralinstruct":  0.5971,
    }

    per_split = {}
    for split in splits:
        ds = load_dataset("natolambert/xstest-v2-copy", split=split)
        X_test, y_test, _, dropped = prepare(
            ds,
            get_prompt=lambda s: s["prompt"],
            get_response=lambda s: s.get("completion") or "",
            get_label=lambda s: binarize_xstest_label(s.get("final_label")),
        )
        if len(y_test) == 0 or len(set(y_test)) < 2:
            continue
        X_test_s = scaler.transform(X_test)
        probs = clf.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, probs)
        per_split[split] = {"auc": float(auc), "n": len(y_test), "refuse_frac": float(y_test.mean())}
        delta = auc - v1_aucs.get(split, 0)
        bar = "█" * int(auc * 40)
        print(f"  {split:<18s}  AUC {auc:.4f} [{bar:<40s}]  n={len(y_test)}  v1→v2 delta: {delta:+.4f}")

    v1_mean = float(np.mean(list(v1_aucs.values())))
    v2_mean = float(np.mean([v["auc"] for v in per_split.values()]))
    print(f"\n  MEAN cross-model AUC:  v1={v1_mean:.4f}  →  v2={v2_mean:.4f}  (delta: {v2_mean - v1_mean:+.4f})")

    # Failure-mode check
    failures_v2 = [k for k, v in per_split.items() if v["auc"] < 0.65]
    print(f"\n  documented failure modes (AUC < 0.65):")
    print(f"    v1: ['mistralinstruct']")
    print(f"    v2: {failures_v2}")

    # Feature importance — what changed with more data?
    print("\n" + "=" * 72)
    print("FEATURE WEIGHTS — v2 vs v1 coefficients")
    print("=" * 72)
    v1_coefs = {
        "refusal_density": 1.4580, "hedge_density": -0.0340, "confidence_density": -0.0198,
        "uncertainty_density": 0.0, "entity_density": -0.0748, "claim_density": 0.1161,
        "reasoning_marker_density": 0.0, "sentence_length_mean": 0.3462, "unique_ratio": 0.3035,
        "disclaimer_density": 0.4744, "safety_flag_density": 0.2386, "redirect_density": 0.0,
        "normative_density": 0.4429, "short_response": 0.0, "starts_with_sorry": 2.0616,
        "starts_with_i_cant": 0.0, "starts_with_normative": 0.0, "log_word_count": 0.2098,
    }
    print(f"  {'feature':<28s}  {'v1':>8s}  {'v2':>8s}  {'delta':>8s}")
    for name, v2c in sorted(zip(feat_names, clf.coef_[0]), key=lambda kv: -abs(kv[1])):
        v1c = v1_coefs.get(name, 0.0)
        print(f"  {name:<28s}  {v1c:>+8.3f}  {v2c:>+8.3f}  {v2c - v1c:>+8.3f}")
    print(f"  {'intercept':<28s}  {1.685:>+8.3f}  {clf.intercept_[0]:>+8.3f}")

    # Save v2 calibration artifact
    result = {
        "methodology": "refusal_v2 — text-only LR, train=JBB-Llama-1B+JBB-judge n=380, test=XSTest-v2 n=2250",
        "train_n": len(combined),
        "train_auc_fit": float(train_auc),
        "per_split_auc_v2": per_split,
        "per_split_auc_v1": v1_aucs,
        "mean_cross_model_auc_v2": v2_mean,
        "mean_cross_model_auc_v1": v1_mean,
        "delta_mean_auc": float(v2_mean - v1_mean),
        "features": feat_names,
        "coefficients": dict(zip(feat_names, clf.coef_[0].tolist())),
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "documented_failure_modes": failures_v2,
    }
    out_path = REPO / "benchmarks" / "refusal_xstest_heldout_v2.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote results -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()

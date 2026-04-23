"""Refusal detector — held-out evaluation on XSTest.

Trains a text-only LR on the 80 labeled (prompt, response, refuse)
samples from JailbreakBench/Llama-3.2-1B, then evaluates on XSTest
with responses from 4 *different* models (GPT-4, Llama 2, Mistral-*).

This is the critical cross-model universality check for law II of
cognometry: "vitals transfer across substrates." If the JBB-Llama-1B
weights hit >0.85 AUC on XSTest-GPT-4 responses, we've empirically
validated the claim for refusal detection — a second cognometric
instrument on the same methodology as hallucination (styxx v4).

XSTest reference: Röttger et al. (2023), "XSTest: A Test Suite for
Identifying Exaggerated Safety Behaviours in Large Language Models."
HuggingFace: natolambert/xstest-v2-copy

Label mapping:
  1_full_compliance            -> COMPLY (0)
  2_full_refusal               -> REFUSE (1)
  3_partial_refusal            -> REFUSE (1)  (any refusal)
  (other / ambiguous)          -> skip
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from scripts.refusal_baseline_v1 import extract_refusal_features

REPO = Path(__file__).resolve().parents[1]
JBB_PATH = REPO / "styxx" / "residual_probe" / "atlas" / "compliance_labels_llama_1b.json"


def binarize_xstest_label(lbl: str) -> int | None:
    lbl = (lbl or "").lower()
    if "full_compliance" in lbl:
        return 0
    if "refusal" in lbl:
        return 1
    return None  # skip partial / ambiguous


def prepare(samples, get_prompt, get_response, get_label):
    X, y, dropped = [], [], 0
    feature_names = None
    for s in samples:
        lbl = get_label(s)
        if lbl is None:
            dropped += 1
            continue
        f = extract_refusal_features(get_prompt(s), get_response(s))
        if feature_names is None:
            feature_names = list(f.keys())
        X.append([f[k] for k in feature_names])
        y.append(lbl)
    return np.array(X), np.array(y), feature_names, dropped


def main():
    # ---- Train: JBB-Llama-1B (80 samples) ----
    jbb = json.loads(JBB_PATH.read_text(encoding="utf-8"))
    X_train, y_train, feat_names, _ = prepare(
        jbb,
        get_prompt=lambda s: s["prompt"],
        get_response=lambda s: s.get("response_excerpt", ""),
        get_label=lambda s: int(s["label"]),
    )
    print(f"TRAIN (JBB-Llama-1B): n={len(X_train)}, refuse={int(y_train.sum())}/{len(y_train)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    clf.fit(X_train_s, y_train)

    # Training-set AUC (sanity)
    train_probs = clf.predict_proba(X_train_s)[:, 1]
    print(f"  train-set AUC (for reference): {roc_auc_score(y_train, train_probs):.4f}")
    print()

    # ---- Held-out: XSTest across 4 model completions ----
    splits = ["gpt4", "llama2new", "llama2orig", "mistralguard", "mistralinstruct"]
    print("HELD-OUT (XSTest v2, completions by 4 different model families):")
    print()

    per_split = {}
    for split in splits:
        try:
            ds = load_dataset("natolambert/xstest-v2-copy", split=split)
        except Exception as e:
            print(f"  {split}: load failed ({e})")
            continue
        X_test, y_test, _, dropped = prepare(
            ds,
            get_prompt=lambda s: s["prompt"],
            get_response=lambda s: s.get("completion") or "",
            get_label=lambda s: binarize_xstest_label(s.get("final_label")),
        )
        if len(y_test) == 0 or len(set(y_test)) < 2:
            print(f"  {split}: degenerate — all one class or empty (skipped)")
            continue
        X_test_s = scaler.transform(X_test)
        probs = clf.predict_proba(X_test_s)[:, 1]
        auc = roc_auc_score(y_test, probs)
        per_split[split] = {
            "auc": float(auc),
            "n": len(y_test),
            "refuse_frac": float(y_test.mean()),
            "dropped": int(dropped),
        }
        bar = "█" * int(auc * 40)
        print(f"  {split:<18s}  AUC {auc:.4f}  [{bar:<40s}]  n={len(y_test)} refuse={int(y_test.sum())}  dropped={dropped}")

    print()
    print("COGNOMETRY LAW II empirical check (refusal):")
    print(f"  trained on JBB / Llama-3.2-1B responses (n={len(X_train)})")
    print(f"  evaluated on XSTest / {len(per_split)} different model families (total n={sum(v['n'] for v in per_split.values())})")

    if per_split:
        mean_cross = np.mean([v["auc"] for v in per_split.values()])
        print(f"  MEAN CROSS-MODEL AUC: {mean_cross:.4f}")
        print()
        # Published failure modes logic (mirroring calibrated_weights_v4 pattern)
        failures = [k for k, v in per_split.items() if v["auc"] < 0.65]
        if failures:
            print(f"  ! documented failure modes (AUC < 0.65): {failures}")

    # Feature importance from the trained weights
    print()
    print("TRAINED WEIGHTS (text-only LR, n_train=80):")
    coefs = sorted(zip(feat_names, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    for name, coef in coefs:
        print(f"  {name:<25s} {coef:+.3f}")
    print(f"  intercept                 {clf.intercept_[0]:+.3f}")

    # Save results for paper/release artifact
    result = {
        "methodology": "text-only LR, train=JBB-Llama-1B n=80, test=XSTest-v2",
        "train_auc": float(roc_auc_score(y_train, train_probs)),
        "per_split_auc": per_split,
        "mean_cross_model_auc": float(np.mean([v["auc"] for v in per_split.values()])) if per_split else None,
        "features": feat_names,
        "coefficients": dict(zip(feat_names, clf.coef_[0].tolist())),
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    out_path = REPO / "benchmarks" / "refusal_xstest_heldout_v1.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote results -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()

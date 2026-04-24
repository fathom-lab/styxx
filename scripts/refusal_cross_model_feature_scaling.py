"""Cross-model phase-transition probe — refusal detector on XSTest × 5 families.

Extends refusal_feature_scaling.py (which found the in-distribution K=1
phase transition on JBB-Llama-1B) to held-out XSTest responses from
5 different model families (gpt4, llama2new, llama2orig, mistralguard,
mistralinstruct). Question: does each substrate have the same critical
K, or does it shift per model family?

This is a direct test of cognometry's Law II (cross-substrate universality)
combined with the phase-transition finding. If the critical K is stable
across families, Law II is stronger. If K shifts, we have a calibration
fingerprint per family.

Protocol:
  - Train LR on full JBB-Llama-1B (n=80) with TOP-K features by |coef|
    from a full 18-feature model (same ranking as the within-sample
    ablation).
  - For each K in [0..18], evaluate on each XSTest split (n=450).
  - Record per-split AUC at each K.
  - Phase-transition flag: |AUC[K] - AUC[K-1]| >= 0.10 per split.

Dataset:
  Train: styxx/residual_probe/atlas/compliance_labels_llama_1b.json (n=80)
  Test:  natolambert/xstest-v2-copy splits — gpt4, llama2new, llama2orig,
         mistralguard, mistralinstruct (~450 each)

Label binarization (XSTest): full_refusal -> 1, full_compliance -> 0,
partial -> skip.

Output:
  benchmarks/refusal_cross_model_feature_scaling.json
  papers/refusal_cross_model_phase_transitions.md

Usage:
  python scripts/refusal_cross_model_feature_scaling.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from refusal_baseline_v1 import extract_refusal_features  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
JBB_PATH = REPO / "styxx" / "residual_probe" / "atlas" / "compliance_labels_llama_1b.json"
OUT_JSON = REPO / "benchmarks" / "refusal_cross_model_feature_scaling.json"
OUT_MD = REPO / "papers" / "refusal_cross_model_phase_transitions.md"

RANDOM_STATE = 0
PT_DELTA = 0.10

FEATURE_NAMES = [
    "refusal_density", "hedge_density", "confidence_density",
    "uncertainty_density", "entity_density", "claim_density",
    "reasoning_marker_density", "sentence_length_mean", "unique_ratio",
    "disclaimer_density", "safety_flag_density", "redirect_density",
    "normative_density", "short_response", "starts_with_sorry",
    "starts_with_i_cant", "starts_with_normative", "log_word_count",
]
N_FEATURES = len(FEATURE_NAMES)

XSTEST_SPLITS = ["gpt4", "llama2new", "llama2orig", "mistralguard", "mistralinstruct"]


def binarize_xstest(lbl):
    lbl = (lbl or "").lower()
    if "full_compliance" in lbl:
        return 0
    if "refusal" in lbl:
        return 1
    return None


def prepare(samples, get_prompt, get_response, get_label):
    X, y, dropped = [], [], 0
    for s in samples:
        lbl = get_label(s)
        if lbl is None:
            dropped += 1
            continue
        f = extract_refusal_features(get_prompt(s), get_response(s))
        X.append([f[name] for name in FEATURE_NAMES])
        y.append(lbl)
    return np.array(X, dtype=float), np.array(y), dropped


def main():
    # --- Training set ---
    jbb = json.loads(JBB_PATH.read_text(encoding="utf-8"))
    X_train, y_train, _ = prepare(
        jbb,
        lambda s: s["prompt"],
        lambda s: s.get("response_excerpt", ""),
        lambda s: int(s["label"]),
    )
    print(f"train (JBB-Llama-1B): n={len(X_train)}, refuse={int(y_train.sum())}/{len(y_train)}")

    # --- Full model importance ranking (same method as in-sample ablation) ---
    sc_full = StandardScaler()
    X_tr_s = sc_full.fit_transform(X_train)
    clf_full = LogisticRegression(
        C=1.0, max_iter=2000, random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    clf_full.fit(X_tr_s, y_train)
    ranked = sorted(
        zip(FEATURE_NAMES, clf_full.coef_[0], range(N_FEATURES)),
        key=lambda kv: -abs(kv[1]),
    )
    print()
    print("Feature importance ranking (full 18-feature model):")
    for rank, (name, c, _idx) in enumerate(ranked, 1):
        print(f"  {rank:2d}. {name:<26s}  {c:+.3f}")

    # --- Load all XSTest splits up-front ---
    print()
    print("loading XSTest splits ...")
    split_data = {}
    for split in XSTEST_SPLITS:
        try:
            ds = load_dataset("natolambert/xstest-v2-copy", split=split)
        except Exception as e:
            print(f"  {split}: load FAILED ({e})")
            continue
        X, y, dropped = prepare(
            ds,
            lambda s: s["prompt"],
            lambda s: s.get("completion") or "",
            lambda s: binarize_xstest(s.get("final_label")),
        )
        if len(set(y)) < 2:
            print(f"  {split}: degenerate (single class) — skipped")
            continue
        split_data[split] = (X, y)
        print(f"  {split:<16s} n={len(y)} refuse={int(y.sum())}/{len(y)} (dropped {dropped})")

    if not split_data:
        sys.exit("no usable splits")

    # --- Top-K sweep, eval on each split ---
    print()
    print(f"{'='*78}")
    print("Top-K cross-model sweep (train JBB / test XSTest per split)")
    print(f"{'='*78}")

    rows = []
    header = f"  {'K':>3} {'added':<26s} " + " ".join(
        f"{s:>16s}" for s in split_data.keys()
    )
    print(header)
    prev_per_split = {s: None for s in split_data}
    pt_per_split = {s: [] for s in split_data}

    for K in range(0, N_FEATURES + 1):
        if K == 0:
            # chance for all
            row = {"K": 0, "added": None, "aucs": {s: 0.5 for s in split_data}}
        else:
            feat_idx = [r[2] for r in ranked[:K]]
            added = ranked[K-1][0]
            # Train on full JBB with only these features
            sc = StandardScaler()
            X_tr_k = sc.fit_transform(X_train[:, feat_idx])
            clf = LogisticRegression(
                C=1.0, max_iter=2000, random_state=RANDOM_STATE,
                class_weight="balanced",
            )
            clf.fit(X_tr_k, y_train)
            aucs = {}
            for split, (X_te, y_te) in split_data.items():
                X_te_k = sc.transform(X_te[:, feat_idx])
                probs = clf.predict_proba(X_te_k)[:, 1]
                aucs[split] = float(roc_auc_score(y_te, probs))
            row = {"K": K, "added": added, "aucs": aucs}

        # flag per-split phase transitions
        flags = []
        for split in split_data:
            prev = prev_per_split[split]
            cur = row["aucs"][split]
            if prev is not None and abs(cur - prev) >= PT_DELTA:
                flags.append(f"{split}:+{cur - prev:+.2f}")
                pt_per_split[split].append({
                    "K": row["K"], "added": row["added"],
                    "from_auc": prev, "to_auc": cur,
                    "delta": cur - prev,
                })
            prev_per_split[split] = cur

        line = (
            f"  {row['K']:>3d} {str(row['added'])[:26]:<26s} "
            + " ".join(f"{row['aucs'][s]:>16.4f}" for s in split_data)
        )
        if flags:
            line += "  *PT: " + ", ".join(flags)
        print(line)
        rows.append(row)

    # --- Summary: per-split critical K ---
    print()
    print(f"{'='*78}")
    print("Per-split critical-K summary")
    print(f"{'='*78}")
    for split, pts in pt_per_split.items():
        if pts:
            for pt in pts:
                print(f"  {split:<16s}  K={pt['K']:<3d}  +{pt['added']:<26s}  "
                      f"{pt['from_auc']:.3f} -> {pt['to_auc']:.3f} (delta={pt['delta']:+.3f})")
        else:
            final_auc = rows[-1]["aucs"][split]
            print(f"  {split:<16s}  no phase transition (final AUC {final_auc:.3f})")

    # --- Save artifact ---
    out = {
        "methodology": (
            "Refusal cross-model phase-transition — train LR top-K features "
            "on JBB-Llama-1B (n=80), eval on XSTest v2 per model family "
            "(n=~450 each). top-K ordering from full-18-feature coef ranking "
            "(seed=0, class_weight=balanced)."
        ),
        "train_dataset": "JBB-Llama-1B n=80",
        "test_dataset": "natolambert/xstest-v2-copy",
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "full_model_ranking": [
            {"rank": i + 1, "feature": r[0], "coef": r[1]}
            for i, r in enumerate(ranked)
        ],
        "splits": {
            s: {"n": int(len(y)), "refuse": int(y.sum()), "no_refuse": int((y == 0).sum())}
            for s, (X, y) in split_data.items()
        },
        "top_k_sweep": [
            {"K": r["K"], "added": r["added"], "aucs": r["aucs"]}
            for r in rows
        ],
        "phase_transitions_per_split": pt_per_split,
        "phase_transition_threshold": PT_DELTA,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"wrote -> {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()

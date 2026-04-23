"""Refusal-detection text-only baseline — Instrument #2 spike.

Loads the 80 labeled (prompt, response, complied) samples committed
in `styxx/residual_probe/atlas/compliance_labels_llama_1b.json` and
evaluates a text-only logistic-regression classifier on top of
existing `text_features` signals.

Goal: establish a baseline AUC on the data that was used to train
the v0 residual-stream probe (which hit 0.9015 LOO on Llama 3.2 1B).
A competitive text-only baseline would let us ship a second cognometric
instrument that works on any LLM's output without requiring model
weights.

Methodology:
  - 5-fold stratified CV  (80 samples: adequate, small)
  - Features: existing text_features + handful of new heuristics
  - LR with L2 reg
  - Report mean AUC, per-fold AUCs, feature importances

Reference: calibrated_weights_v4.py methodology — same 3-seed pooled
LR pattern we used for hallucination.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from styxx.anthropic_hack.text_features import extract_features

REPO = Path(__file__).resolve().parents[1]
DATA_PATH = REPO / "styxx" / "residual_probe" / "atlas" / "compliance_labels_llama_1b.json"


# ----------------------------------------------------------------
# Additional text heuristics not in text_features.py
# ----------------------------------------------------------------

DISCLAIMER_PHRASES = [
    "as an ai", "as a language model", "as an ai assistant",
    "as an ai language", "i'm an ai", "i am an ai",
    "it's important to note", "it is important to note",
    "it's worth noting", "please note that",
]

SAFETY_FLAG_PHRASES = [
    "illegal", "unethical", "harmful", "dangerous", "violence",
    "harm to others", "against the law", "inappropriate content",
    "safety", "ethical",
]

REDIRECT_PHRASES = [
    "instead, i can", "instead i can", "however, i can",
    "however i can", "i can help with", "let me suggest",
    "i'd recommend", "i would recommend", "consider reaching out",
    "please consult", "speak to a professional",
]

# Moralizing / lecturing patterns — Mistral-instruct's typical
# refusal style. Adds recall for models that don't apologize but
# instead lecture on ethics/safety.
NORMATIVE_PHRASES = [
    "it's important to note", "it is important to note",
    "it's important to understand", "it is important to understand",
    "it's crucial to", "it is crucial to",
    "it's essential to", "it is essential to",
    "never acceptable", "is never acceptable",
    "not acceptable", "is not acceptable",
    "please understand", "please consider",
    "must always", "should always respect",
    "proper consent", "respect the privacy",
    "healthy and respectful", "safe and respectful",
    "against the law", "legal and ethical",
]

NORMATIVE_OPENERS = (
    "it's important", "it is important",
    "it's crucial", "it is crucial",
    "it's essential", "it is essential",
    "first, it's", "first, it is",
    "please note", "please be aware",
    "first and foremost", "it should be noted",
)

SHORT_RESPONSE_THRESHOLD = 50  # words


def _phrase_density(text: str, phrases: List[str]) -> float:
    lt = text.lower()
    n_words = max(1, len(text.split()))
    return sum(1 for p in phrases if p in lt) / n_words


def extract_refusal_features(prompt: str, response: str) -> Dict[str, float]:
    """Build the feature vector for refusal classification."""
    # Existing anthropic_hack text features (returns a dataclass)
    tf = extract_features(response)

    n_words = max(1, len(response.split()))

    feats = {
        # Existing signals (ported out of the dataclass)
        "refusal_density": tf.refusal_density,
        "hedge_density": tf.hedge_density,
        "confidence_density": tf.confidence_density,
        "uncertainty_density": tf.uncertainty_density,
        "entity_density": tf.entity_density,
        "claim_density": tf.claim_density,
        "reasoning_marker_density": tf.reasoning_marker_density,
        "sentence_length_mean": tf.sentence_length_mean,
        "unique_ratio": tf.unique_ratio,

        # New heuristics
        "disclaimer_density": _phrase_density(response, DISCLAIMER_PHRASES),
        "safety_flag_density": _phrase_density(response, SAFETY_FLAG_PHRASES),
        "redirect_density": _phrase_density(response, REDIRECT_PHRASES),
        "normative_density": _phrase_density(response, NORMATIVE_PHRASES),
        "short_response": 1.0 if n_words < SHORT_RESPONSE_THRESHOLD else 0.0,
        "starts_with_sorry": 1.0 if response.strip().lower().startswith(("i'm sorry", "i am sorry", "sorry"))
                             else 0.0,
        "starts_with_i_cant": 1.0 if response.strip().lower().startswith(("i can't", "i cannot", "i can not"))
                               else 0.0,
        "starts_with_normative": 1.0 if response.strip().lower().startswith(NORMATIVE_OPENERS)
                                  else 0.0,
        "log_word_count": math.log(max(1, n_words)),
    }
    return feats


def main():
    # Load labeled data
    samples = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    print(f"loaded {len(samples)} labeled samples from {DATA_PATH.name}")
    print(f"  class balance: {sum(s['label'] for s in samples)}/{len(samples)} refuse")

    # Extract features
    X_list = []
    y_list = []
    feature_names = None
    for s in samples:
        f = extract_refusal_features(s["prompt"], s.get("response_excerpt", ""))
        if feature_names is None:
            feature_names = list(f.keys())
        X_list.append([f[k] for k in feature_names])
        y_list.append(s["label"])

    X = np.array(X_list)
    y = np.array(y_list)
    print(f"  feature matrix: {X.shape}  features: {feature_names}")

    # 5-fold stratified CV with LR
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    fold_aucs = []
    all_test_scores = []
    all_test_labels = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
        clf.fit(X_train, y[train_idx])
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y[test_idx], probs)
        fold_aucs.append(auc)
        all_test_scores.extend(probs)
        all_test_labels.extend(y[test_idx])
        print(f"  fold {fold+1}: AUC {auc:.4f}  n_test={len(test_idx)}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    pooled_auc = roc_auc_score(all_test_labels, all_test_scores)

    print()
    print(f"  MEAN fold AUC: {mean_auc:.4f}  (std {std_auc:.4f})")
    print(f"  POOLED AUC (concat across folds): {pooled_auc:.4f}")
    print()

    # Fit final model on all data for feature importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    clf.fit(X_scaled, y)
    coefs = list(zip(feature_names, clf.coef_[0]))
    coefs.sort(key=lambda kv: -abs(kv[1]))
    print("  feature importance (scaled, |coef| desc):")
    for name, coef in coefs:
        print(f"    {name:<25s} {coef:+.3f}")
    print(f"    intercept           {clf.intercept_[0]:+.3f}")

    # Compare against the v0 residual probe on the same 80 samples
    print()
    print("  -- comparison --")
    print(f"    residual-probe v0 (Llama 3.2 1B layer 10, LOO): AUC 0.9015")
    print(f"    text-only baseline (this run, 5-fold CV):         AUC {mean_auc:.4f}  (pooled {pooled_auc:.4f})")


if __name__ == "__main__":
    main()

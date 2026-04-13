# -*- coding: utf-8 -*-
"""
styxx.learned_classifier — trained text classifier from audit data.

The text heuristic in conversation.py is regex. It works but it has
a hard ceiling. This module trains a simple classifier (logistic
regression on TF-IDF features) from the accumulated audit log with
outcome labels.

chart.jsonl IS the training set. Every entry with a prompt field
and an outcome='correct' label is a training example:
  - input: the prompt text
  - label: the phase4_pred category

The trained model replaces the regex heuristic for text-based
classification on providers that don't expose logprobs (Anthropic,
local models).

Why logistic regression / TF-IDF:
  - No GPU required, trains in <1 second on 1000 examples
  - scikit-learn is the only dependency (already common)
  - Interpretable — you can inspect which words drive each category
  - Outperforms regex patterns after ~200 labeled examples
  - Falls back to the regex heuristic if sklearn isn't available
    or there aren't enough training examples

1.0.0+.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TrainResult:
    """Result of training the text classifier."""
    n_train: int = 0
    n_categories: int = 0
    accuracy: float = 0.0
    saved_to: Optional[str] = None
    error: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"<TrainResult {self.n_train} examples, "
            f"{self.n_categories} categories, "
            f"acc {self.accuracy:.2f}>"
        )


def _model_dir() -> Path:
    data_dir = os.environ.get("STYXX_DATA_DIR", "").strip()
    if data_dir:
        d = Path(data_dir).expanduser() / "models"
    else:
        d = Path.home() / ".styxx" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_training_data() -> Tuple[List[str], List[str]]:
    """Load (prompt_text, category) pairs from correct-labeled entries."""
    from .analytics import load_audit
    entries = load_audit(last_n=5000)
    texts, labels = [], []
    for e in entries:
        if e.get("outcome") != "correct":
            continue
        prompt = e.get("prompt")
        cat = e.get("phase4_pred")
        if prompt and cat and len(prompt) > 5:
            texts.append(prompt)
            labels.append(cat)
    return texts, labels


def train_text_classifier(
    *,
    min_samples: int = 50,
    agent_name: Optional[str] = None,
) -> TrainResult:
    """Train a text classifier from accumulated audit data.

    Reads all entries with outcome='correct' and a non-empty prompt,
    trains a logistic regression on TF-IDF features, and saves the
    model for use by the text heuristic fallback path.

    Args:
        min_samples:  minimum training examples required. Default 50.
        agent_name:   name for the model file. Defaults to
                      STYXX_AGENT_NAME or 'default'.

    Returns:
        TrainResult with accuracy and save path.

    Requires scikit-learn. Falls back gracefully if not installed.
    """
    result = TrainResult()

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        result.error = (
            "scikit-learn required for trained text classifier. "
            "Install with: pip install scikit-learn"
        )
        return result

    if agent_name is None:
        agent_name = os.environ.get("STYXX_AGENT_NAME", "").strip() or "default"

    texts, labels = _load_training_data()
    result.n_train = len(texts)

    if len(texts) < min_samples:
        result.error = (
            f"need at least {min_samples} labeled examples, "
            f"have {len(texts)}. log styxx.feedback('correct') "
            f"after calls to build the training set."
        )
        return result

    result.n_categories = len(set(labels))

    # Train
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        multi_class="multinomial",
        solver="lbfgs",
    )
    model.fit(X, labels)

    # Cross-validate if enough data
    if len(texts) >= 100:
        scores = cross_val_score(model, X, labels, cv=min(5, len(set(labels))),
                                 scoring="accuracy")
        result.accuracy = float(scores.mean())
    else:
        result.accuracy = model.score(X, labels)

    # Save model + vectorizer
    model_path = _model_dir() / f"{agent_name}_text_clf.pkl"
    try:
        with open(model_path, "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "model": model,
                         "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "n_train": len(texts), "accuracy": result.accuracy}, f)
        result.saved_to = str(model_path)
    except OSError as e:
        result.error = f"could not save model: {e}"

    return result


def classify_with_trained_model(
    text: str,
    *,
    agent_name: Optional[str] = None,
) -> Optional[Tuple[str, float]]:
    """Classify text using the trained model if available.

    Returns (category, confidence) or None if no trained model exists.
    Falls back to None so the caller can use the regex heuristic.
    """
    if agent_name is None:
        agent_name = os.environ.get("STYXX_AGENT_NAME", "").strip() or "default"

    model_path = _model_dir() / f"{agent_name}_text_clf.pkl"
    if not model_path.exists():
        return None

    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        vectorizer = data["vectorizer"]
        model = data["model"]
        X = vectorizer.transform([text])
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        best_idx = proba.argmax()
        return (classes[best_idx], float(proba[best_idx]))
    except Exception:
        return None

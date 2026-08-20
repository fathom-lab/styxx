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

Persistence is JSON (vocabulary, idf, coefficients), never pickle:
the model dir may be attacker-writable, and pickle.load executes
arbitrary code — same class as the 7.17.x RCE fix. Legacy .pkl
models are never loaded; retrain to regenerate.

1.0.0+.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


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


# Vectorizer construction params — stored in the JSON payload and
# re-applied on load so transform() matches training exactly.
_VECTORIZER_PARAMS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
}

# Legacy pickle files we already noted on stderr (once per process).
_LEGACY_PKL_NOTED: set = set()


def _model_dir() -> Path:
    data_dir = os.environ.get("STYXX_DATA_DIR", "").strip()
    if data_dir:
        d = Path(data_dir).expanduser() / "models"
    else:
        d = Path.home() / ".styxx" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _logreg():
    """LogisticRegression with the training config.

    multi_class was removed in sklearn 1.7+ (multinomial is the only
    behavior there); older versions need it pinned explicitly so
    binary predict_proba matches across versions.
    """
    from sklearn.linear_model import LogisticRegression
    try:
        return LogisticRegression(max_iter=1000, C=1.0,
                                  multi_class="multinomial", solver="lbfgs")
    except TypeError:
        return LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")


def _load_training_data() -> Tuple[List[str], List[str]]:
    """Load (prompt_text, category) pairs from HUMAN-confirmed entries.

    AUTO-STAMPED OUTCOMES ARE REFUSED. With auto-feedback enabled the outcome is
    derived from the entry's own gate (pass -> correct), and the label used here
    is ``phase4_pred`` -- the classifier's own prediction. Training on those
    pairs is a closed loop with no external signal at all: prompts the gate
    liked, labelled with what the classifier already said. It would relearn its
    own mistakes and read the agreement as accuracy.

    Standing caveat even for human labels: ``outcome="correct"`` means the
    ANSWER was right, not that the CATEGORY was. This selects entries a human
    vouched for; it does not verify the label itself. A properly supervised
    corpus would carry category labels a human actually assigned.
    """
    from .analytics import load_audit
    entries = load_audit(last_n=5000)
    texts, labels = [], []
    for e in entries:
        if e.get("outcome") != "correct" or e.get("outcome_source") == "auto":
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
    vectorizer = TfidfVectorizer(**_VECTORIZER_PARAMS)
    X = vectorizer.fit_transform(texts)
    model = _logreg()
    model.fit(X, labels)

    # Cross-validate if enough data
    if len(texts) >= 100:
        scores = cross_val_score(model, X, labels, cv=min(5, len(set(labels))),
                                 scoring="accuracy")
        result.accuracy = float(scores.mean())
    else:
        result.accuracy = model.score(X, labels)

    # Save model + vectorizer as plain JSON (see module docstring —
    # pickle here would be an RCE vector)
    model_path = _model_dir() / f"{agent_name}_text_clf.json"
    payload = {
        "format": "styxx-text-clf-v1",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train": len(texts),
        "accuracy": result.accuracy,
        "vectorizer": {
            "params": _VECTORIZER_PARAMS,
            "vocabulary": {t: int(i) for t, i in vectorizer.vocabulary_.items()},
            "idf": [float(v) for v in vectorizer.idf_],
        },
        "model": {
            "coef": [[float(v) for v in row] for row in model.coef_],
            "intercept": [float(v) for v in model.intercept_],
            "classes": [str(c) for c in model.classes_],
        },
    }
    try:
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        result.saved_to = str(model_path)
    except OSError as e:
        result.error = f"could not save model: {e}"

    return result


def _load_json_model(model_path: Path):
    """Rebuild the fitted vectorizer + model from the JSON payload."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vparams = dict(data["vectorizer"]["params"])
    if "ngram_range" in vparams:
        vparams["ngram_range"] = tuple(vparams["ngram_range"])
    vocabulary = {t: int(i) for t, i in data["vectorizer"]["vocabulary"].items()}
    vectorizer = TfidfVectorizer(vocabulary=vocabulary, **vparams)
    vectorizer.idf_ = np.asarray(data["vectorizer"]["idf"], dtype=np.float64)

    model = _logreg()
    model.classes_ = np.asarray(data["model"]["classes"])
    model.coef_ = np.asarray(data["model"]["coef"], dtype=np.float64)
    model.intercept_ = np.asarray(data["model"]["intercept"], dtype=np.float64)
    return vectorizer, model


def classify_with_trained_model(
    text: str,
    *,
    agent_name: Optional[str] = None,
) -> Optional[Tuple[str, float]]:
    """Classify text using the trained model if available.

    Returns (category, confidence) or None if no trained model exists.
    Falls back to None so the caller can use the regex heuristic.

    Legacy pickle models (*.pkl, written before the JSON format) are
    treated as absent — unpickling from a possibly attacker-writable
    dir is arbitrary code execution. A one-line stderr note asks for
    a retrain instead.
    """
    if agent_name is None:
        agent_name = os.environ.get("STYXX_AGENT_NAME", "").strip() or "default"

    model_path = _model_dir() / f"{agent_name}_text_clf.json"
    if not model_path.exists():
        legacy = model_path.with_suffix(".pkl")
        if legacy.exists() and str(legacy) not in _LEGACY_PKL_NOTED:
            _LEGACY_PKL_NOTED.add(str(legacy))
            sys.stderr.write(
                f"styxx: ignoring legacy pickle model {legacy.name}; "
                "retrain with styxx.train_text_classifier()\n"
            )
        return None

    try:
        vectorizer, model = _load_json_model(model_path)
        X = vectorizer.transform([text])
        proba = model.predict_proba(X)[0]
        best_idx = proba.argmax()
        return (str(model.classes_[best_idx]), float(proba[best_idx]))
    except Exception:
        return None

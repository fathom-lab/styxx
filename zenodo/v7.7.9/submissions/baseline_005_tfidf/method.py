# -*- coding: utf-8 -*-
"""Baseline-005 — TF-IDF + linear classifier.

A classical NLP baseline. Word/char n-gram TF-IDF features over the question
text, balanced one-vs-rest logistic regression on top. No model weights to
download (unlike Baseline-002's sentence-transformer), no GPU needed.
Trains on the same corpus as Baseline-002.

This is a real attempt at the gauntlet, not a deliberately bad anchor.
Question: does classical bag-of-tokens beat the embedding-based baseline
on K3 (cross-corpus folklore F1)?
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict

warnings.filterwarnings("ignore")

_TRAIN_SPLIT = None
_CLASSIFIER = None
_VECTORIZER = None


def _load_training_corpus():
    """Build the training corpus from ICT receipts + curated truth controls.
    Curated FOLKLORE items are held out (cross-corpus test set)."""
    global _TRAIN_SPLIT
    if _TRAIN_SPLIT is not None:
        return _TRAIN_SPLIT

    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    ict_path = repo_root / "papers" / "consensus-hallucination" / "probe_ict_results.json"
    corpus_path = repo_root / "papers" / "consensus-hallucination" / "corpus_folklore_2026_05_27.py"

    if not ict_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            f"baseline_005 needs the ICT results + curated corpus from the source tree. "
            f"expected: {ict_path} + {corpus_path}. run from a git checkout of fathom-lab/styxx."
        )

    ict = json.loads(ict_path.read_text())
    import importlib.util
    spec = importlib.util.spec_from_file_location("corpus_folklore", corpus_path)
    corpus_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corpus_mod)
    curated_truths = corpus_mod.TRUTHS

    records = []
    for r in ict.get("rows", []):
        q = r.get("q", "")
        cat = r.get("cat", "")
        label = r.get("label")
        if cat == "self-referential":
            continue
        if label == 0:
            records.append((q, "truth"))
        elif cat == "folklore/legend":
            records.append((q, "folklore"))
        elif cat == "pseudoscience/supernatural":
            records.append((q, "pseudoscience"))
        elif cat == "factual-error":
            records.append((q, "factual-error"))
    for q, _, _ in curated_truths:
        records.append((q, "truth"))

    _TRAIN_SPLIT = records
    return records


def _ensure_classifier():
    """Train TF-IDF + balanced one-vs-rest LR. Same fit-once protocol as Baseline-002."""
    global _CLASSIFIER, _VECTORIZER
    if _CLASSIFIER is not None:
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
    except ImportError as e:
        raise ImportError(
            f"baseline_005 requires scikit-learn. install: pip install scikit-learn. "
            f"original: {e}"
        )

    records = _load_training_corpus()
    class_counts = Counter(c for _, c in records)
    keep = {c for c, n in class_counts.items() if n >= 2}
    X_text = [q for q, c in records if c in keep]
    y = [c for q, c in records if c in keep]

    # Word + char n-gram TF-IDF (classical strong baseline for short text)
    _VECTORIZER = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
        norm="l2",
    )
    X = _VECTORIZER.fit_transform(X_text)

    _CLASSIFIER = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260527)
    )
    _CLASSIFIER.fit(X, y)


def predict(question: str) -> Dict[str, str]:
    _ensure_classifier()
    x = _VECTORIZER.transform([question])
    pred = _CLASSIFIER.predict(x)[0]
    return {"class": str(pred)}


if __name__ == "__main__":
    print(predict("What is the capital of France?"))
    print(predict("Where is Walt Disney's body?"))
    print(predict("Did Marie Antoinette say 'let them eat cake'?"))

# -*- coding: utf-8 -*-
"""Baseline-006 — char-level TF-IDF + linear classifier.

Sibling of Baseline-005 but with character n-grams instead of word n-grams.
Same training corpus, same classifier head. Tests whether sub-word /
character-level lexical features pick up the folklore signature where
word-level cannot.

**Pre-stated prediction (recorded BEFORE running, see submission.json):**
- K3 cross-corpus F1 < 0.30 (i.e., does NOT beat Baseline-002's 0.36)
- folklore recall < 0.17 (Baseline-002's cross-corpus recall)
- If char-level beats word-level Baseline-005, the "word abstraction is
  wrong" claim is partially refuted.
- If char-level loses to or ties Baseline-005, the "classical NLP cannot
  see the dark core" claim has n=2 confirmation.
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
    global _TRAIN_SPLIT
    if _TRAIN_SPLIT is not None:
        return _TRAIN_SPLIT
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    ict_path = repo_root / "papers" / "consensus-hallucination" / "probe_ict_results.json"
    corpus_path = repo_root / "papers" / "consensus-hallucination" / "corpus_folklore_2026_05_27.py"
    if not ict_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(f"baseline_006 needs source tree paths: {ict_path}, {corpus_path}")
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
    global _CLASSIFIER, _VECTORIZER
    if _CLASSIFIER is not None:
        return
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    records = _load_training_corpus()
    class_counts = Counter(c for _, c in records)
    keep = {c for c, n in class_counts.items() if n >= 2}
    X_text = [q for q, c in records if c in keep]
    y = [c for q, c in records if c in keep]

    # KEY DIFFERENCE FROM BASELINE-005: analyzer is "char_wb" (char n-grams
    # within word boundaries), not "word". n-grams of length 3-5.
    _VECTORIZER = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
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

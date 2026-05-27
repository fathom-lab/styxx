# -*- coding: utf-8 -*-
"""Baseline-002 — the dark-core classifier from `darkcore_classifier_2026_05_27.py`,
wrapped in the gauntlet's classification interface.

This is the sentence-transformer + balanced one-vs-rest logistic regression
classifier shipped as the closed-negative baseline in
`papers/consensus-hallucination/darkcore_classifier_2026_05_27.py` (commit a3dc813).

It is shipped as a baseline on the public leaderboard to make the empirical
floor concretely visible: a real classifier with real scores against the
locked K1/K2/K3 bars. Replace this with your method in your own
`submissions/<your-name>/method.py` and run `styxx gauntlet --method
submissions.<your-name>.method:predict --task classification` to see if you
beat it.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict

warnings.filterwarnings("ignore")


_TRAIN_SPLIT = None  # cached at first call
_CLASSIFIER = None
_EMBEDDER = None


def _load_training_corpus():
    """Build the training corpus from ICT receipts + curated truth controls.
    The curated FOLKLORE items are held out (cross-corpus test set), so the
    classifier sees only 4 in-distribution folklore items in training."""
    global _TRAIN_SPLIT
    if _TRAIN_SPLIT is not None:
        return _TRAIN_SPLIT

    # Resolve paths in a way that works whether installed or running from source.
    # The benchmark + the source ICT results are needed.
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent  # submissions/baseline_002_classifier/method.py -> repo root
    ict_path = repo_root / "papers" / "consensus-hallucination" / "probe_ict_results.json"
    corpus_path = repo_root / "papers" / "consensus-hallucination" / "corpus_folklore_2026_05_27.py"

    if not ict_path.exists() or not corpus_path.exists():
        raise FileNotFoundError(
            f"baseline_002 needs the ICT results + curated corpus from the source tree "
            f"(not bundled in the pip wheel). expected: {ict_path} + {corpus_path}. "
            f"run from a git checkout of fathom-lab/styxx."
        )

    ict = json.loads(ict_path.read_text())
    # Import the corpus module by path
    import importlib.util
    spec = importlib.util.spec_from_file_location("corpus_folklore", corpus_path)
    corpus_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corpus_mod)
    curated_truths = corpus_mod.TRUTHS

    # Pull (question, class) pairs from ICT receipts
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
    # Add curated truths (curated folklore is held out)
    for q, _, _ in curated_truths:
        records.append((q, "truth"))

    _TRAIN_SPLIT = records
    return records


def _ensure_classifier():
    """Lazily train the classifier on first call. Same spec as
    `darkcore_classifier_2026_05_27.py`: all-MiniLM-L6-v2 + balanced
    one-vs-rest logistic regression, fit-once."""
    global _CLASSIFIER, _EMBEDDER
    if _CLASSIFIER is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
    except ImportError as e:
        raise ImportError(
            f"baseline_002 requires sentence-transformers and scikit-learn. "
            f"install with `pip install styxx[nli]` or `pip install sentence-transformers scikit-learn`. "
            f"original: {e}"
        )

    records = _load_training_corpus()
    # Drop classes with <2 items (stratify requires it)
    class_counts = Counter(c for _, c in records)
    keep = {c for c, n in class_counts.items() if n >= 2}
    X_text = [q for q, c in records if c in keep]
    y = [c for q, c in records if c in keep]

    _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = _EMBEDDER.encode(X_text, normalize_embeddings=True, show_progress_bar=False)

    _CLASSIFIER = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=20260527)
    )
    _CLASSIFIER.fit(X, y)


def predict(question: str) -> Dict[str, str]:
    """Classification interface for the styxx gauntlet."""
    _ensure_classifier()
    x = _EMBEDDER.encode([question], normalize_embeddings=True, show_progress_bar=False)
    pred = _CLASSIFIER.predict(x)[0]
    return {"class": str(pred)}


if __name__ == "__main__":
    # quick smoke test
    print(predict("What is the capital of France?"))
    print(predict("Where is Walt Disney's body?"))

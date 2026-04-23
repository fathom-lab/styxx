# -*- coding: utf-8 -*-
"""
NLI-based contradiction signal.

v3.9.1's response-novelty signals cannot distinguish faithful dialog
responses from hallucinated ones: both add content not verbatim in
the reference. Dialog/summarization hallucinations are mostly
CONTRADICTIONS (the response asserts something the reference
denies), not ADDITIONS (the response asserts something the reference
is silent on).

This module adds a proper NLI signal: a small entailment model
runs on (reference → response) pairs and returns the probability
that the response is contradicted by the reference.

Model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  - 184M params, CPU-friendly (~1s per pair on a modern CPU)
  - trained on MNLI + FEVER + ANLI-R3
  - 3-class output: entailment / neutral / contradiction

Usage:

    from styxx.guardrail.nli_signal import NLIScorer
    scorer = NLIScorer()           # downloads model on first call
    p_contradict = scorer.score(
        premise=reference_text,
        hypothesis=response_text,
    )
    # p_contradict in [0, 1]; higher = more likely contradicted

Optional dependency: requires `pip install styxx[nli]`
(torch + transformers).
"""
from __future__ import annotations

import threading
from typing import List, Optional


_DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
_MAX_LEN = 512


class NLIScorer:
    """Lazy-loaded NLI contradiction scorer.

    Thread-safe, process-local. First call triggers model download
    + load (~1GB on disk, ~700MB RAM). Subsequent calls are fast.

    On CUDA-available systems the model runs on GPU. Falls back to
    CPU automatically.
    """

    _lock = threading.Lock()

    def __init__(self, model_name: str = _DEFAULT_MODEL,
                  device: Optional[str] = None):
        self.model_name = model_name
        self._device = device
        self._tokenizer = None
        self._model = None
        self._label_idx = None   # index of contradiction label

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import (
                    AutoTokenizer, AutoModelForSequenceClassification,
                )
            except ImportError as e:
                raise ImportError(
                    "styxx.guardrail.nli_signal requires torch and "
                    "transformers. Install with `pip install "
                    "styxx[nli]`."
                ) from e

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
            ).to(self._device)
            self._model.eval()

            # Find contradiction label index. Different NLI models
            # use different label orderings.
            id2label = self._model.config.id2label
            for idx, name in id2label.items():
                if "contradict" in name.lower():
                    self._label_idx = int(idx)
                    break
            if self._label_idx is None:
                # Fallback: MNLI standard ordering (0=entailment, 1=neutral, 2=contradiction)
                self._label_idx = 2

    def score(self, premise: str, hypothesis: str) -> float:
        """Return probability that ``premise`` contradicts ``hypothesis``.

        Returns 0.0 if premise or hypothesis is empty (fail-open).
        """
        if not premise or not hypothesis:
            return 0.0
        self._load()

        import torch  # local
        inputs = self._tokenizer(
            premise, hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_LEN,
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        return float(probs[self._label_idx])

    def score_batch(self, pairs: List[tuple]) -> List[float]:
        """Score many (premise, hypothesis) pairs at once.

        Faster than looping ``score()`` because the tokenizer and
        model fire once.
        """
        if not pairs:
            return []
        self._load()

        import torch
        premises, hypotheses = zip(*[
            (p or "", h or "") for (p, h) in pairs
        ])
        inputs = self._tokenizer(
            list(premises), list(hypotheses),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=_MAX_LEN,
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        return [float(p[self._label_idx]) for p in probs]


# Module-level singleton for easy re-use
_default_scorer: Optional[NLIScorer] = None


def get_default_scorer() -> NLIScorer:
    """Return a lazily-initialized module-level NLIScorer."""
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = NLIScorer()
    return _default_scorer


def nli_contradiction_score(reference: str, response: str,
                              scorer: Optional[NLIScorer] = None) -> float:
    """Convenience: score contradiction of a single (ref, resp) pair.

    Uses the module-level default scorer unless one is passed.
    Returns 0.0 on empty inputs or model load failure (fail-open).
    """
    if not reference or not response:
        return 0.0
    try:
        s = scorer or get_default_scorer()
        return s.score(premise=reference, hypothesis=response)
    except Exception:
        return 0.0


__all__ = [
    "NLIScorer", "get_default_scorer", "nli_contradiction_score",
]

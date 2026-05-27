# -*- coding: utf-8 -*-
"""Baseline-008 — embedding-similarity detector.

The real semantic-axis bet. Scores each (question, response) pair by the
cosine similarity between sentence-transformer embeddings of the question
and the response. Hypothesis: misconception responses (full council
restatements) share more semantic content with the question than truth
responses (terse direct answers), so similarity is higher on misconceptions.

The question this baseline answers: is "semantic similarity" just a smarter
version of "length / lexical overlap" (and therefore captured by D3's
length-control bar), or does it add real signal above the length floor?

If D3 fails: n=1 evidence that Q-R similarity is structurally equivalent
to length on this benchmark — the artifact applies in any feature space
that measures Q-R correlation. The dark core stays dark to similarity-
based detection.

If D3 passes: the first real PASS on the gauntlet, demonstrating semantic
similarity adds signal beyond the artifact. The synthesis would get its
first deployable positive on the detection axis.

Pre-stated prediction (committed BEFORE the gauntlet run, see
PRE_STATED_PREDICTION.md): ~15-25% probability of clearing D3.
"""
from __future__ import annotations

import warnings
from typing import Dict

warnings.filterwarnings("ignore")

_MODEL = None


def _ensure_model():
    global _MODEL
    if _MODEL is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            f"baseline_008 requires sentence-transformers. install: "
            f"pip install sentence-transformers (or pip install styxx[nli]). original: {e}"
        )
    _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def detect(question: str, response: str) -> Dict[str, float]:
    """Cosine similarity between question and response embeddings.

    Higher = more semantically similar = more misconception-like under the
    hypothesis. The exact metric value is in [-1, 1] but in practice for
    normalized sentence-transformer embeddings on natural text it stays
    in roughly [0.0, 1.0]."""
    _ensure_model()
    embs = _MODEL.encode([question, response], normalize_embeddings=True, show_progress_bar=False)
    sim = float(embs[0] @ embs[1])
    return {"score": sim}


if __name__ == "__main__":
    for q, r in [
        ("What is the capital of France?", "Paris"),
        ("Where is Walt Disney's body?", "Walt Disney's body is interred at Forest Lawn Memorial Park."),
        ("Did Marie Antoinette say 'let them eat cake'?", "Marie Antoinette is often attributed with saying 'let them eat cake'."),
    ]:
        print(f"sim={detect(q, r)['score']:.4f}  q={q!r}  r={r!r}")

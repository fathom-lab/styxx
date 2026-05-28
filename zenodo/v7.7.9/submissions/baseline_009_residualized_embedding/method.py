# -*- coding: utf-8 -*-
"""Baseline-009 — length-residualized embedding similarity.

The obvious "smart fix" after Baseline-008's D3 failure: take the same
sentence-transformer cosine similarity, then subtract the length-predicted
component, and score on the residual. The hypothesis being tested: does
this give signal that clears D3, or does residualization just destroy
signal without escaping the artifact?

The detector is unsupervised: the (intercept, slope) for the linear
regression `sim ~ word_length` is fit on the benchmark's (q, response)
pairs WITHOUT using class labels. The fit captures the length-prediction
of similarity; the residual is what remains after that prediction is
subtracted.

Pre-stated prediction (committed BEFORE the gauntlet run, see
PRE_STATED_PREDICTION.md): modal outcome D2 passes, D1 fails, D3 fails
wide. Probability of clearing D3 cleanly: ~3%.

If D3 fails wide (predicted ~65%): n=1 evidence that length-
residualization is not the path through the D3 bar. The synthesis
argues for genuinely orthogonal semantic features as Baseline-010 next.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

_MODEL = None
_FIT: Optional[Tuple[float, float]] = None


def _ensure_model():
    global _MODEL
    if _MODEL is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            f"baseline_009 requires sentence-transformers. install: "
            f"pip install sentence-transformers (or pip install styxx[nli]). original: {e}"
        )
    _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _ensure_fit():
    """Pre-fit the linear regression sim ~ word_length on the benchmark.

    UNSUPERVISED: class labels are never read. Only (sim, length) pairs are
    used. The fit is deterministic and one-time; cached in _FIT for the
    duration of the process.
    """
    global _FIT
    if _FIT is not None:
        return
    _ensure_model()
    from styxx.gauntlet import load_benchmark
    bench = load_benchmark()
    records = bench.get("records", [])
    sims = []
    lengths = []
    for r in records:
        q = r.get("question", "") or ""
        cons = r.get("expected_consensus", "") or ""
        embs = _MODEL.encode([q, cons], normalize_embeddings=True, show_progress_bar=False)
        sim = float(embs[0] @ embs[1])
        ln = float(len(cons.split()))
        sims.append(sim)
        lengths.append(ln)
    n = len(sims)
    if n == 0:
        _FIT = (0.0, 0.0)
        return
    mean_s = sum(sims) / n
    mean_l = sum(lengths) / n
    cov = sum((s - mean_s) * (l - mean_l) for s, l in zip(sims, lengths)) / n
    var_l = sum((l - mean_l) ** 2 for l in lengths) / n
    slope = (cov / var_l) if var_l > 0 else 0.0
    intercept = mean_s - slope * mean_l
    _FIT = (intercept, slope)


def detect(question: str, response: str) -> Dict[str, float]:
    """Length-residualized cosine similarity.

    score = cosine(embed(q), embed(r)) − (intercept + slope · word_length(r))

    The intercept and slope are pre-fit on the benchmark's (sim, length)
    distribution without class labels. Higher residual = "more similar
    than length alone would predict" = more misconception-like under the
    hypothesis. By construction, the residual's correlation with length
    is zero on the fit corpus.
    """
    _ensure_model()
    _ensure_fit()
    intercept, slope = _FIT  # type: ignore[misc]
    embs = _MODEL.encode([question, response], normalize_embeddings=True, show_progress_bar=False)
    sim = float(embs[0] @ embs[1])
    ln = float(len((response or "").split()))
    predicted_sim = intercept + slope * ln
    residual = sim - predicted_sim
    return {"score": residual}


if __name__ == "__main__":
    for q, r in [
        ("What is the capital of France?", "Paris"),
        ("Where is Walt Disney's body?", "Walt Disney's body is interred at Forest Lawn Memorial Park."),
        ("Did Marie Antoinette say 'let them eat cake'?", "Marie Antoinette is often attributed with saying 'let them eat cake'."),
    ]:
        print(f"residual={detect(q, r)['score']:.4f}  q={q!r}  r={r!r}")

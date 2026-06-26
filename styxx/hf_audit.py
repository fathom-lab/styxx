# -*- coding: utf-8 -*-
"""styxx.audit_hf_model — audit a HuggingFace text-classifier for length bias in one call.

``audit_confound`` answers *is this score riding a confound?* but it asks you to bring an
orthogonal corpus and wire up the scoring. For the two most common cases — sentiment and
toxicity classifiers — styxx ships a *validated, length-orthogonal boundary corpus* (the concept
and response-length are decorrelated by construction; a plain bag-of-words recovers the concept,
so a length-biased verdict means the model's SCORE rides length, not that the signal is missing).
This wraps the whole pipeline into a single call::

    from styxx import audit_hf_model
    report = audit_hf_model("cardiffnlp/twitter-roberta-base-sentiment-latest", construct="sentiment")
    print(report.verdict)          # THRESHOLD-BIASED / ROBUST / CONFOUND-DEPENDENT / INCONCLUSIVE
    print(report.confound_score_coef, report.confound_score_coef_ci95)

or from the CLI::

    styxx audit-model cardiffnlp/twitter-roberta-base-sentiment-latest --construct sentiment

It loads the model with ``transformers.pipeline`` (``trust_remote_code`` stays **off** — auditing a
model never executes its repo code), maps each output to a single positive/toxic scalar with a
robust, *verifiable* label mapping (override with ``score_label`` or ``score_fn`` for non-standard
heads), computes the bag-of-words construct-recoverability ceiling, and runs ``audit_confound``.

Requires ``transformers`` + ``torch`` (``pip install 'styxx[hf]'``) — unless you pass your own
``score_fn``, in which case only the (base) styxx install is needed.

Honest scope: single frontier generator, single seed, n=200, ONE confound (length); the concept is a
model-instantiated stance verified by a BoW refit, not gold human labels; verdicts are read at the
decision boundary (confounds hide at saturation). For any other construct, build your own grid with
``styxx.build_confound_grid`` and call ``styxx.audit_confound`` directly.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .confound_audit import ConfoundAuditReport, audit_confound

#: bundled, validated length-orthogonal boundary corpora (shipped under ``styxx/_data``)
_CORPORA: Dict[str, str] = {
    "sentiment": "confound_boundary_sentiment.jsonl",
    "toxicity": "confound_boundary_toxicity.jsonl",
}
# Semantic label keys searched (in order) to derive the positive/toxic scalar from a model's
# output. We deliberately do NOT fall back to opaque ``label_0``/``label_1`` indices: their
# polarity is not knowable from the label alone, and guessing could silently invert the score.
# A model with only opaque labels returns None here -> the caller asks for an explicit
# ``score_label`` rather than risk a wrong verdict.
_SENTIMENT_KEYS = ("positive", "pos")
_TOXICITY_KEYS = ("toxic", "toxicity")


def available_constructs() -> List[str]:
    """The constructs with a bundled boundary corpus."""
    return sorted(_CORPORA)


def _load_corpus(construct: str) -> List[Dict[str, Any]]:
    fn = _CORPORA.get(construct)
    if fn is None:
        raise ValueError(
            f"construct must be one of {available_constructs()} (got {construct!r}). "
            "For any other construct, build a grid with styxx.build_confound_grid and call "
            "styxx.audit_confound directly."
        )
    # importlib.resources keeps this working from a wheel, a zip, or a source checkout.
    from importlib import resources

    text = resources.files("styxx._data").joinpath(fn).read_text(encoding="utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:  # pragma: no cover — corrupt install
        raise RuntimeError(f"bundled corpus {fn!r} is empty; reinstall styxx")
    return rows


def _bow_recoverable_auc(rows: List[Dict[str, Any]]) -> Optional[float]:
    """Model-agnostic ceiling: can a plain bag-of-words recover the concept from the text?
    This separates a *broken model* (construct recoverable, model can't) from a *degenerate
    corpus* (construct not in the text). Returns None if sklearn is unavailable."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_predict
        from sklearn.pipeline import make_pipeline
    except ImportError:  # pragma: no cover — sklearn is a base dep, but stay graceful
        return None
    texts = [r["text"] for r in rows]
    y = [int(r["label"]) for r in rows]
    oof = cross_val_predict(
        make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1, 2)), LogisticRegression(max_iter=2000)),
        texts, y, cv=5, method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(y, oof))


def _default_target(prob: Dict[str, float], construct: str) -> Optional[float]:
    """Map a model's per-label probabilities (lower-cased label -> prob) to a single
    positive/toxic scalar. Star-rating sentiment heads ('1 star'..'5 stars') map to a
    normalized expected-star polarity. Returns None if no known key matches (the caller then
    asks for an explicit ``score_label``/``score_fn`` rather than guessing)."""
    if construct == "sentiment":
        star_like = [k for k in prob if "star" in k and k.split()[:1] and k.split()[0].isdigit()]
        # Only treat as a star-rating head if EVERY label is a numeric-star label — otherwise a
        # mixed head would be silently renormalized over a subset of its classes.
        if star_like and len(star_like) == len(prob):
            tot = w = 0.0
            for k in star_like:
                tot += int(k.split()[0]) * prob[k]
                w += prob[k]
            if w <= 0:
                return None
            return (tot / w - 1.0) / 4.0  # 1..5 stars -> 0..1 positivity
        keys = _SENTIMENT_KEYS
    else:
        keys = _TOXICITY_KEYS
    for k in keys:
        if k in prob:
            return prob[k]
    return None


def audit_hf_model(
    model_id: str,
    construct: str = "sentiment",
    *,
    score_label: Optional[str] = None,
    score_fn: Optional[Callable[[str], float]] = None,
    device: int = -1,
) -> ConfoundAuditReport:
    """Audit a HuggingFace text-classification model for length bias on a bundled boundary corpus.

    Parameters
    ----------
    model_id : str
        A HuggingFace model id (or local path) for a ``text-classification`` head — e.g.
        ``"distilbert-base-uncased-finetuned-sst-2-english"``.
    construct : {"sentiment", "toxicity"}
        Which bundled, validated boundary corpus to audit against.
    score_label : str, optional
        Force the output label whose probability is the score (e.g. ``"POSITIVE"``, ``"toxic"``).
        Use when a model's labels don't match the defaults. Case-insensitive.
    score_fn : callable, optional
        ``str -> float`` scorer. If given, the model is **not** loaded (no transformers/torch
        needed) — useful for closed/remote scorers or testing. Overrides ``model_id`` scoring.
    device : int
        transformers device index (``-1`` = CPU, the portable default; ``0`` = first GPU).

    Returns
    -------
    ConfoundAuditReport
        Same report as :func:`styxx.audit_confound` — ``.verdict``, ``.confound_score_coef``
        (+ ``.confound_score_coef_ci95``), ``.within_stratum_auc``, and a ready-to-use
        ``.guard(score, confound)`` when the verdict is ``THRESHOLD-BIASED``.

    Raises
    ------
    ImportError
        If ``transformers`` is needed (no ``score_fn``) but not installed — ``pip install 'styxx[hf]'``.
    ValueError
        On an unknown ``construct`` or labels that can't be mapped without ``score_label``/``score_fn``.
    """
    rows = _load_corpus(construct)

    if score_fn is not None:
        scores = [float(score_fn(r["text"])) for r in rows]
    else:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "audit_hf_model needs transformers + torch to load a model. "
                "Install them with:  pip install 'styxx[hf]'   "
                "(or pass score_fn=... to audit a scorer you call yourself)."
            ) from exc

        # trust_remote_code is left at its safe default (False): auditing a model must never
        # execute code shipped in the model repo.
        clf = pipeline(
            "text-classification", model=model_id, top_k=None,
            truncation=True, max_length=256, device=device,
        )
        want = score_label.lower() if score_label else None

        def _score(text: str) -> float:
            out = clf(text)[0]  # top_k=None -> list of {label, score} dicts
            prob = {p["label"].lower(): float(p["score"]) for p in out}
            if want is not None:
                if want not in prob:
                    raise ValueError(
                        f"score_label {score_label!r} not in model labels {sorted(prob)}"
                    )
                return prob[want]
            v = _default_target(prob, construct)
            if v is None:
                raise ValueError(
                    f"could not map labels {sorted(prob)} to a {construct!r} score automatically. "
                    "Pass score_label='<the positive/toxic label>' or score_fn=..."
                )
            return v

        scores = [_score(r["text"]) for r in rows]

    refit = _bow_recoverable_auc(rows)
    return audit_confound(
        rows, scores=scores, instrument=model_id, confound="log_words",
        construct_recoverable_auc=refit,
    )

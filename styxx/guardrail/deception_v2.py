# -*- coding: utf-8 -*-
"""
styxx.guardrail.deception_v2 — semantic-grounding deception detection.

============================================================================
WHY v2 EXISTS
============================================================================
The v0 detector measures the *lexical signature* of instructed-dishonesty
under contrastive prompting (gpt-4o-mini training corpus). On 2026-05-10
we re-evaluated v0 on TruthfulQA and observed:

  - v0 published in-corpus AUC:    0.956
  - v0 on TruthfulQA (out-of-corpus): **0.590** (chance ≈ 0.5)
  - v1 lexical retrain ceiling on TQA: 0.667
  - **v2 NLI contradiction**:        **0.818**
  - v2 embedding similarity:         0.736

The lexical-signature approach has a ceiling; semantic grounding clears
it. v2 is the upgrade that addresses v0's documented "single-source
corpus" failure mode by scoring the response against a *correct
reference* instead of against a calibrated bag of lexical features.

See `.styxx/DECEPTION_V2_BREAKTHROUGH_2026_05_10.md` and
`.styxx/DECEPTION_V1_FINDING_2026_05_10.md` for the full evidence trail.
============================================================================

Three modes:
  - "nli"  (default when a reference is provided): score response against
           reference using NLI cross-encoder. P(contradiction) is the
           deception score. Most rigorous; requires `transformers` +
           `sentence-transformers` and ~184M-param model
           (`cross-encoder/nli-deberta-v3-base`).
  - "emb"  (lighter): embed (response, correct_ref, incorrect_ref) and
           score = cos(response, incorrect_ref) - cos(response,
           correct_ref). Requires `sentence-transformers` and
           `all-MiniLM-L6-v2` (~22M params).
  - "v0_fallback": no reference available — fall back to v0 lexical
           detector. Returns the v0 verdict with a `scope_warning`
           field flagging the AUC-0.59 generalization gap.

The "auto" mode (default for `deception_check_v2`) picks nli if a
reference is provided AND `transformers` is importable; falls back
to emb if only `sentence-transformers` is available; falls back to
v0_fallback if no reference is provided OR no semantic backend is
available.

All semantic backends are loaded lazily on first use (model files
download from HuggingFace Hub on first call; cached afterwards).

License: MIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# v0 fallback path — always available
from .deception import deception_check as _v0_check


Mode = Literal["nli", "emb", "v0_fallback", "auto"]
DEFAULT_DECEPTION_V2_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Lazy model loaders (singletons)
# ---------------------------------------------------------------------------
_EMB_MODEL = None
_NLI_MODEL = None
_EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


def _get_emb_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "deception_v2 emb mode requires sentence-transformers. "
                "Install: pip install sentence-transformers"
            ) from e
        _EMB_MODEL = SentenceTransformer(_EMB_MODEL_NAME)
    return _EMB_MODEL


def _get_nli_model():
    global _NLI_MODEL
    if _NLI_MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "deception_v2 nli mode requires sentence-transformers. "
                "Install: pip install sentence-transformers"
            ) from e
        _NLI_MODEL = CrossEncoder(_NLI_MODEL_NAME, max_length=256)
    return _NLI_MODEL


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass
class DeceptionV2Verdict:
    """Verdict from `deception_check_v2()`.

    Attributes:
        prompt:               echoed back
        response:             response under test
        deception_risk:       calibrated [0, 1] probability
        shows_signature:      bool — risk >= threshold
        threshold:            decision threshold used
        mode:                 "nli" | "emb" | "v0_fallback"
        reference_provided:   bool — was a reference available
        features:             mode-specific feature dict
            nli mode: {"contradiction", "entailment", "neutral"}
            emb mode: {"sim_correct", "sim_incorrect", "delta"}
            v0_fallback: forwards v0's 9 lexical features
        top_signals:          mode-specific top firing items (for advice)
        scope_warning:        non-empty for v0_fallback (the AUC 0.59 gap)
        models_used:          dict of {role: model_name} for reproducibility
    """
    prompt: str
    response: str
    deception_risk: float
    shows_signature: bool
    threshold: float
    mode: str
    reference_provided: bool
    features: Dict[str, Any] = field(default_factory=dict)
    top_signals: List[Tuple[str, float, float]] = field(default_factory=list)
    scope_warning: str = ""
    models_used: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "deception_risk": self.deception_risk,
            "shows_signature": self.shows_signature,
            "threshold": self.threshold,
            "mode": self.mode,
            "reference_provided": self.reference_provided,
            "features": dict(self.features),
            "top_signals": [
                {"name": n, "value": v, "contribution": c}
                for n, v, c in self.top_signals
            ],
            "scope_warning": self.scope_warning,
            "models_used": dict(self.models_used),
        }


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def _check_nli(
    prompt: str,
    response: str,
    correct_reference: str,
    threshold: float,
) -> DeceptionV2Verdict:
    """NLI contradiction-against-correct-reference scoring.

    Premise = correct_reference (what the model SHOULD have said).
    Hypothesis = response (what the model DID say).
    P(contradiction) = the response contradicts the correct answer.
    """
    nli = _get_nli_model()
    # nli-deberta-v3-base output order: contradiction, entailment, neutral
    raw = nli.predict([(correct_reference, response)],
                      apply_softmax=True, convert_to_numpy=True)[0]
    contradiction, entailment, neutral = float(raw[0]), float(raw[1]), float(raw[2])

    deception_risk = contradiction
    feats = {
        "contradiction": contradiction,
        "entailment": entailment,
        "neutral": neutral,
    }
    top = [
        ("contradiction", contradiction, contradiction),
        ("entailment", entailment, -entailment),    # entailment lowers risk
        ("neutral", neutral, 0.0),
    ]
    return DeceptionV2Verdict(
        prompt=prompt,
        response=response,
        deception_risk=deception_risk,
        shows_signature=bool(deception_risk >= threshold),
        threshold=threshold,
        mode="nli",
        reference_provided=True,
        features=feats,
        top_signals=top,
        models_used={"nli_cross_encoder": _NLI_MODEL_NAME},
    )


def _check_emb(
    prompt: str,
    response: str,
    correct_reference: str,
    incorrect_reference: Optional[str],
    threshold: float,
) -> DeceptionV2Verdict:
    """Embedding cosine-similarity scoring.

    If `incorrect_reference` is provided:
        score = cos(response, incorrect_ref) - cos(response, correct_ref)
        higher = more like the incorrect ref = more deceptive
    Otherwise:
        score = 1 - cos(response, correct_ref)
        higher = less similar to correct = more deceptive
    """
    import numpy as np
    emb = _get_emb_model()
    refs = [response, correct_reference]
    if incorrect_reference:
        refs.append(incorrect_reference)
    vecs = emb.encode(refs, show_progress_bar=False, batch_size=8)

    def cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    sim_correct = cos(vecs[0], vecs[1])
    sim_incorrect = cos(vecs[0], vecs[2]) if incorrect_reference else None

    if sim_incorrect is not None:
        delta = sim_incorrect - sim_correct  # in [-2, 2], usually small
        # Map [-1, 1] -> [0, 1] sigmoid-style for a probability-shaped output
        deception_risk = 1.0 / (1.0 + pow(2.71828, -2.0 * delta))
        feats = {
            "sim_correct": sim_correct,
            "sim_incorrect": sim_incorrect,
            "delta": delta,
        }
    else:
        deception_risk = max(0.0, min(1.0, 1.0 - sim_correct))
        feats = {"sim_correct": sim_correct, "sim_incorrect": None, "delta": None}

    top = [
        ("sim_correct", sim_correct, -sim_correct),
        ("sim_incorrect", sim_incorrect or 0.0, sim_incorrect or 0.0),
    ]
    return DeceptionV2Verdict(
        prompt=prompt,
        response=response,
        deception_risk=deception_risk,
        shows_signature=bool(deception_risk >= threshold),
        threshold=threshold,
        mode="emb",
        reference_provided=True,
        features=feats,
        top_signals=top,
        models_used={"sentence_transformer": _EMB_MODEL_NAME},
    )


def _check_v0_fallback(
    prompt: str,
    response: str,
    threshold: float,
    *,
    reference_provided: bool = False,
    reason: str = "no_reference",
) -> DeceptionV2Verdict:
    """Fall back to v0 lexical detector with explicit scope warning.

    `reason` tells the caller WHY we fell back so the scope_warning can be
    accurate: 'no_reference' (no correct_reference passed) vs
    'no_semantic_backend' (reference provided but sentence-transformers
    isn't installed) vs 'explicit' (mode='v0_fallback' chosen by caller).
    """
    v0 = _v0_check(prompt, response, threshold=threshold)
    if reason == "no_semantic_backend":
        warning = (
            "v2 fell back to v0 lexical detector — a `correct_reference` was "
            "provided but the NLI backend isn't installed. v0 has documented "
            "AUC 0.59 on TruthfulQA (near chance) for ground-truth deception "
            "detection. To enable nli mode (AUC 0.82), install: "
            "pip install styxx[nli]  (or styxx-mcp[nli])."
        )
    elif reason == "explicit":
        warning = (
            "v0_fallback mode selected explicitly. AUC 0.59 on TruthfulQA — "
            "the score reflects the v0 lexical signature, not factuality. "
            "For ground-truth scoring, pass `correct_reference` and use "
            "mode='nli'."
        )
    else:
        warning = (
            "v2 fell back to v0 lexical detector — no correct reference "
            "was provided. v0 has documented AUC 0.59 on TruthfulQA "
            "(near chance) for ground-truth deception detection. The "
            "score reflects the Pennebaker/Newman vague-brevity lexical "
            "signature, NOT factuality. For ground-truth scoring, "
            "provide a `correct_reference` argument and use mode='nli'."
        )
    return DeceptionV2Verdict(
        prompt=prompt,
        response=response,
        deception_risk=v0.deception_risk,
        shows_signature=v0.shows_signature,
        threshold=v0.threshold,
        mode="v0_fallback",
        reference_provided=reference_provided,
        features=dict(v0.features),
        top_signals=list(v0.top_signals),
        scope_warning=warning,
        models_used={"v0_lexical": "calibrated_weights_deception_v0"},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deception_check_v2(
    prompt: str,
    response: str,
    *,
    correct_reference: Optional[str] = None,
    incorrect_reference: Optional[str] = None,
    mode: Mode = "auto",
    threshold: Optional[float] = None,
) -> DeceptionV2Verdict:
    """Semantic-grounding deception verdict for (prompt, response).

    Args:
        prompt:               The user's prompt.
        response:             The model's response under test.
        correct_reference:    A known-correct answer the response should
                              agree with. Required for "nli" and "emb"
                              modes; if absent in "auto", falls back to
                              v0 lexical detector.
        incorrect_reference:  Optional known-incorrect answer. Used by
                              "emb" mode for differential scoring.
        mode:                 "nli" | "emb" | "v0_fallback" | "auto".
                              "auto" picks nli if reference provided +
                              sentence-transformers available; emb if
                              only sentence-transformers available;
                              v0_fallback if neither.
        threshold:            Decision threshold (default 0.5).

    Returns:
        DeceptionV2Verdict with mode-specific features and (when mode is
        v0_fallback) a populated `scope_warning`.

    Bench (TruthfulQA, n=746 leakage-corrected):
        nli (deberta-v3-base contradiction):  AUC 0.818
        emb (MiniLM-L6-v2 differential):      AUC 0.736
        v0_fallback (lexical):                AUC 0.590 (near chance)

    Example (NLI mode, the recommended path):

        >>> from styxx.guardrail import deception_check_v2
        >>> v = deception_check_v2(
        ...     prompt="When was the Treaty of Versailles signed?",
        ...     response="The treaty was signed in 1815.",  # wrong
        ...     correct_reference="The Treaty of Versailles was signed "
        ...                        "on June 28, 1919.",
        ...     mode="nli",
        ... )
        >>> v.deception_risk > 0.5
        True
        >>> v.mode
        'nli'

    Example (no reference — v0 fallback with scope warning):

        >>> v = deception_check_v2(
        ...     prompt="Q",
        ...     response="The treaty was signed in 1815.",
        ... )
        >>> v.mode
        'v0_fallback'
        >>> "AUC 0.59" in v.scope_warning
        True
    """
    th = float(threshold) if threshold is not None else DEFAULT_DECEPTION_V2_THRESHOLD

    # Resolve mode + reason for any fallback
    fallback_reason = "no_reference"
    if mode == "auto":
        if correct_reference and _has_sentence_transformers():
            mode_resolved = "nli"
        elif correct_reference:
            mode_resolved = "v0_fallback"
            fallback_reason = "no_semantic_backend"
        else:
            mode_resolved = "v0_fallback"
            fallback_reason = "no_reference"
    else:
        mode_resolved = mode
        if mode == "v0_fallback":
            fallback_reason = "explicit"

    # Validate mode prerequisites
    if mode_resolved in ("nli", "emb") and not correct_reference:
        raise ValueError(
            f"deception_check_v2 mode={mode_resolved!r} requires "
            f"`correct_reference`. Either pass one, or use mode='v0_fallback' "
            f"or mode='auto'."
        )

    if mode_resolved == "nli":
        return _check_nli(prompt, response, correct_reference, th)
    elif mode_resolved == "emb":
        return _check_emb(prompt, response, correct_reference,
                           incorrect_reference, th)
    elif mode_resolved == "v0_fallback":
        return _check_v0_fallback(
            prompt, response, th,
            reference_provided=bool(correct_reference),
            reason=fallback_reason,
        )
    else:
        raise ValueError(f"unknown mode: {mode_resolved!r}")


CALIBRATION_FINGERPRINT_V2: Dict[str, Any] = {
    "instrument": "deception-v2",
    "modes": ["nli", "emb", "v0_fallback"],
    "training_corpus": "TruthfulQA n=746 leakage-corrected pairs",
    "auc_TQA": {
        "nli": 0.818,
        "emb": 0.736,
        "v0_fallback": 0.590,
    },
    "models": {
        "nli_cross_encoder": _NLI_MODEL_NAME,
        "embedding": _EMB_MODEL_NAME,
    },
    "evidence": (
        ".styxx/DECEPTION_V2_BREAKTHROUGH_2026_05_10.md  ·  "
        ".styxx/DECEPTION_V1_FINDING_2026_05_10.md  ·  "
        ".styxx/out_deception_v2_semantic.json"
    ),
}


__all__ = [
    "DeceptionV2Verdict",
    "deception_check_v2",
    "DEFAULT_DECEPTION_V2_THRESHOLD",
    "CALIBRATION_FINGERPRINT_V2",
]

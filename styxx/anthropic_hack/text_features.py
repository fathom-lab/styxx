# -*- coding: utf-8 -*-
"""
Text-feature classifier for Anthropic responses.

Extracts surface-level linguistic features from generated text and maps
them to a Vitals-compatible reading. Useful when logprobs, ensembles,
and companion models are all unavailable (tier -2 fallback).

Output Vitals are labelled with `phase="text-heuristic"` and
`tier_active=-1`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


HEDGES = {
    "maybe", "perhaps", "might", "possibly", "could", "seems", "appears",
    "likely", "unlikely", "uncertain", "unclear", "probably", "suggest",
    "suggests", "potentially", "apparently", "presumably", "i think",
    "i believe", "i'm not sure", "not certain", "roughly", "approximately",
    "around", "about", "kind of", "sort of",
}

CONFIDENCE = {
    "definitely", "definitively", "certainly", "clearly", "obviously",
    "undoubtedly", "always", "never", "exactly", "precisely",
    "without a doubt", "in fact", "indeed", "well-known", "established",
    "confirmed", "documented",
}

UNCERTAINTY = {
    "i don't know", "i'm not sure", "unsure", "unclear", "hard to say",
    "difficult to tell", "unknown", "cannot determine", "can't tell",
    "no idea", "unfamiliar",
}

REFUSAL_MARKERS = {
    # First-person explicit refusals
    "i can't", "i cannot", "i won't", "i will not", "i shouldn't",
    "i should not", "i don't think i", "i do not think i",
    "i'm not going to", "i am not going to", "i'd rather not",
    "i would rather not", "i'm not able", "i am not able",
    "unable to", "not able to",
    # Contextualized refuse/decline — the bare verbs are too aggressive
    # ("refuse mediocrity", "the decline of", "an offer I can decline")
    # so we require agent-facing context.
    "i refuse", "i decline", "must decline", "must refuse",
    "i'd decline", "i'd refuse", "would decline", "would refuse",
    "refuses to answer", "declines to answer", "refused to answer",
    "declined to answer",
    # Soft refusal wrappers
    "not appropriate", "against my", "sorry, but", "sorry, i",
    "prefer not to", "rather not",
    # Soft refusal / epistemic decline — Claude's default when it
    # won't confabulate on unverifiable prompts
    "i don't have", "i do not have", "i'm not familiar",
    "i am not familiar", "i don't recognize", "i do not recognize",
    "i cannot verify", "i can't verify", "don't have reliable",
    "do not have reliable", "don't have information",
    "do not have information", "no record of",
}

# Step-by-step reasoning markers
REASONING_MARKERS = {
    "first", "second", "third", "then", "next", "therefore", "thus",
    "hence", "because", "since", "consider", "let's", "let me",
    "step by step", "step-by-step", "follows that", "it follows",
    "we can", "we should", "premise", "premises", "conclude",
    "conclusion", "setup", "solution", "given", "substituting",
    "combined", "combined approach", "approach speed", "solve",
    "equation", "equations", "calculate", "calculating", "compute",
    "apply", "applying", "answer:",
}

# Named-entity-ish: capitalized tokens (very rough)
ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")
SENT_RE = re.compile(r"[^.!?]+[.!?]+")
WORD_RE = re.compile(r"\b\w+\b")
# Strip leading markdown header markers ("# Title") so the title text
# doesn't mislead the classifier.
MD_HEADER_RE = re.compile(r"^\s*#+\s+", re.MULTILINE)
# Strip markdown bullets at line start
MD_BULLET_RE = re.compile(r"^\s*[\-*+]\s+", re.MULTILINE)


@dataclass
class TextFeatures:
    n_words: int
    n_sentences: int
    n_lines: int
    hedge_density: float
    confidence_density: float
    uncertainty_density: float
    refusal_density: float
    entity_density: float
    claim_density: float          # sentences containing confidence markers
    reasoning_marker_density: float
    sentence_length_mean: float
    sentence_length_std: float
    unique_ratio: float
    line_density: float           # non-empty lines / n_words
    mean_line_length: float       # words per non-empty line

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _count_phrases(text: str, vocab) -> int:
    tl = text.lower()
    total = 0
    for phrase in vocab:
        if " " in phrase:
            total += tl.count(phrase)
        else:
            total += sum(1 for w in WORD_RE.findall(tl) if w == phrase)
    return total


def extract_features(text: str) -> TextFeatures:
    text = text or ""
    # Strip markdown header markers and bullets so they don't count as
    # sentence-internal entities or mislead structural features.
    clean = MD_HEADER_RE.sub("", text)
    clean = MD_BULLET_RE.sub("", clean)

    words = WORD_RE.findall(clean)
    n_words = max(len(words), 1)

    # Lines: non-empty, stripped
    raw_lines = [ln.strip() for ln in clean.split("\n") if ln.strip()]
    n_lines = max(len(raw_lines), 1)
    line_word_counts = [len(WORD_RE.findall(ln)) for ln in raw_lines]
    mean_line_len = (sum(line_word_counts) / len(line_word_counts)
                     if line_word_counts else 0.0)

    # Sentences: keep the period-delimited view for prose, but ALSO
    # treat each newline-separated line as a "sentence" for entity
    # detection (so line-initial capitals don't get counted as
    # entities in poetry / lists / markdown).
    sentences = [s.strip() for s in SENT_RE.findall(clean) if s.strip()]
    n_sent = max(len(sentences), 1)

    hedges = _count_phrases(clean, HEDGES)
    conf = _count_phrases(clean, CONFIDENCE)
    unc = _count_phrases(clean, UNCERTAINTY)
    refusal = _count_phrases(clean, REFUSAL_MARKERS)
    reason_m = _count_phrases(clean, REASONING_MARKERS)

    # Entity detection: skip first token of every LINE (not every
    # period-delimited sentence) so poetry doesn't get false entities.
    entities = 0
    for ln in raw_lines:
        toks = WORD_RE.findall(ln)
        for i, w in enumerate(toks):
            if i == 0:
                continue
            if w and w[0].isupper() and not w.isupper():
                entities += 1
            elif w.isupper() and len(w) <= 4:  # acronyms: Au, DNA, USA
                entities += 1

    claim_sents = sum(
        1 for s in sentences
        if any(c in s.lower() for c in CONFIDENCE)
    )
    slens = [len(WORD_RE.findall(s)) for s in sentences] or [len(words)]
    mean_len = sum(slens) / len(slens)
    if len(slens) > 1:
        var = sum((x - mean_len) ** 2 for x in slens) / (len(slens) - 1)
        std_len = math.sqrt(var)
    else:
        std_len = 0.0
    unique_ratio = len(set(w.lower() for w in words)) / n_words

    return TextFeatures(
        n_words=n_words,
        n_sentences=n_sent,
        n_lines=n_lines,
        hedge_density=hedges / n_words,
        confidence_density=conf / n_words,
        uncertainty_density=unc / n_words,
        refusal_density=refusal / n_words,
        entity_density=entities / n_words,
        claim_density=claim_sents / n_sent,
        reasoning_marker_density=reason_m / n_words,
        sentence_length_mean=mean_len,
        sentence_length_std=std_len,
        unique_ratio=unique_ratio,
        line_density=n_lines / n_words,
        mean_line_length=mean_line_len,
    )


# ---------- classifier ----------

CATEGORIES = [
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
]


def classify(text: str) -> Dict[str, object]:
    """Classify a text response into a styxx category using surface
    heuristics. Returns dict with predicted, probs, features.

    Design note: retrieval vs. hallucination cannot be distinguished from
    surface features alone — both are "confident claim with entities".
    This classifier returns "retrieval" for the confident-claim shape
    and relies on downstream systems (verify, forecast, truth-checking)
    to separate true from false claims. Callers who need that
    distinction should treat text-mode retrieval as "unverified claim".
    """
    f = extract_features(text)

    scores: Dict[str, float] = {c: 0.0 for c in CATEGORIES}

    # refusal: explicit refusal markers dominate
    # a single refusal marker should swamp stylistic noise
    scores["refusal"] = (
        25.0 * f.refusal_density
        + 1.5 * f.uncertainty_density
        + (3.0 if f.refusal_density > 0 else 0.0)
    )

    # retrieval: confident claim shape — entities + short sentences
    # + confidence/evidence markers + low hedging. Same shape as
    # hallucination; truth-check belongs downstream.
    # Requires at least SOME positive signal (entity or confidence) to
    # fire — otherwise any low-hedge text defaults to retrieval.
    has_claim_signal = (f.entity_density > 0.0 or
                        f.confidence_density > 0.0)
    scores["retrieval"] = (
        (
            3.0 * f.confidence_density
            + 5.0 * f.entity_density
            + (0.4 if f.sentence_length_mean < 18 else 0.0)
        ) if has_claim_signal else 0.0
    ) - 1.5 * f.refusal_density - 1.0 * f.reasoning_marker_density

    # reasoning: step markers + hedges + longer sentences. The step
    # markers are the high-signal feature — "first...then...therefore".
    scores["reasoning"] = (
        6.0 * f.reasoning_marker_density
        + 2.0 * f.hedge_density
        + 0.5 * f.claim_density
        + (0.3 if f.sentence_length_mean >= 12 else 0.0)
        - 2.0 * f.refusal_density
    )

    # creative: prose fiction OR poetic structure. Two shapes:
    #  - prose: ≥30 words, variable sentence length, high unique ratio
    #  - poetry: multiple short lines AND no reasoning markers
    # Reasoning answers often use bulleted structure (short lines) but
    # contain "setup", "solution", "substituting" etc. — gate the
    # poetic shape bonus on the absence of reasoning markers.
    length_bonus = 1.0 if f.n_words >= 30 else (f.n_words / 30.0)
    poetic_shape = (
        1.0 if (
            f.n_lines >= 3
            and f.mean_line_length <= 8.0
            and f.reasoning_marker_density < 0.02
        ) else 0.0
    )
    prose_creative = length_bonus * (
        0.8 * (f.sentence_length_std / max(f.sentence_length_mean, 1.0))
        + 0.8 * f.unique_ratio
    )
    scores["creative"] = (
        prose_creative
        + 2.5 * poetic_shape
        - 4.0 * f.reasoning_marker_density
        - 3.0 * f.confidence_density
        - 3.0 * f.refusal_density
    )

    # hallucination: deliberately under-weighted vs. retrieval. Without
    # truth-check we cannot separate them from surface features. Keep a
    # non-zero score so consensus/companion readings can upweight it.
    scores["hallucination"] = (
        1.0 * f.entity_density
        + 0.5 * f.confidence_density
        - 0.5 * f.hedge_density
        - 3.0 * f.refusal_density
    )

    # adversarial: high uncertainty + confusion
    scores["adversarial"] = (
        3.0 * f.uncertainty_density
        + 1.0 * f.hedge_density
        - 1.0 * f.confidence_density
        - 2.0 * f.refusal_density
    )

    # softmax to probs
    m = max(scores.values())
    exps = {c: math.exp(v - m) for c, v in scores.items()}
    Z = sum(exps.values())
    probs = {c: exps[c] / Z for c in CATEGORIES}
    predicted = max(probs, key=probs.get)
    top = sorted(probs.values(), reverse=True)
    margin = top[0] - top[1]

    return {
        "predicted": predicted,
        "probs": probs,
        "margin": margin,
        "features": f.as_dict(),
        "mode": "text-heuristic",
    }


def build_vitals(text: str):
    """Produce a styxx Vitals object with a text-heuristic reading."""
    from ..vitals import Vitals, PhaseReading

    result = classify(text)
    probs = result["probs"]
    predicted = result["predicted"]
    distances = {c: (1.0 - p) * 5.0 for c, p in probs.items()}
    feats = list(result["features"].values())

    reading = PhaseReading(
        phase="text-heuristic",
        n_tokens_used=result["features"]["n_words"],
        features=feats,
        predicted_category=predicted,
        margin=result["margin"],
        distances=distances,
        probs=probs,
    )
    v = Vitals(
        phase1_pre=reading,
        phase2_early=None,
        phase3_mid=None,
        phase4_late=reading,
        tier_active=-1,
    )
    # attach label so callers can branch on it
    try:
        v.mode = "text-heuristic"  # type: ignore[attr-defined]
    except Exception:
        pass
    return v


__all__ = ["extract_features", "classify", "build_vitals",
           "TextFeatures", "CATEGORIES"]

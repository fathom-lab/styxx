# -*- coding: utf-8 -*-
"""
styxx.divergence — confabulation & fabrication signals from semantic divergence.

Two pure-function primitives from the 2026-05-25 behavioral-knowledge-boundary
arc (papers/tier3-confident-confabulation, papers/council-reference-free-truth):

    semantic_entropy(samples)   — ACROSS-SAMPLE divergence of ONE model's answers
        to the same prompt. High = the model invents a different fact each sample
        = confident confabulation. Low = consistent (it knows the answer, or
        abstains consistently). Its niche is logprob-LESS settings: on TriviaQA it
        matched the literature (AUC 0.785) but did NOT beat single-response logprob
        (0.817) there — a sampling-only signal, not a logprob-beater (see Evidence).

    council_agreement(answers)  — ACROSS-MODEL agreement, one answer per
        independent model. High = models converge (real / shared knowledge);
        low = each invents differently (fabrication). Reference-free.

    grounded_honesty(samples, claim) — is a factual SELF-CLAIM honest? Grounds the
        stated claim against the model's OWN resampled belief:
        g = Stability x Concordance. Breaks the text-only register ceiling on
        factual self-claims (AUC 0.97 grounded vs 0.50 text-only deception) and is
        self-calibrating — ``stability`` gates when to trust the verdict vs abstain
        (papers/grounded-honesty-axis/). Single-model self-consistency, NOT a truth
        oracle (a confidently-wrong belief yields a confidently-wrong verdict).

Both rest on one mechanism: a fact is a shared attractor (convergent), a
fabrication has none (divergent).

Evidence (FEASIBILITY-GRADE — NOT production-validated; small n, OpenAI-only,
single pre-registered runs; see papers/):
  - semantic_entropy: AUC 0.88–0.95 separating confident confabulation from
    correct answers, cross-model (gpt-4o-mini / gpt-4o / gpt-3.5-turbo); and
    VALIDATED on TriviaQA (n=150, judge clustering) at AUC 0.785 — in the
    ~0.75–0.79 semantic-entropy literature band. IMPORTANT: on TriviaQA
    single-response logprob beat it (0.817), so its niche is logprob-LESS settings
    (e.g. the Anthropic API), NOT beating logprob where available
    (papers/benchmark-validation/FINDING_triviaqa_2026_05_25.md).
  - council_agreement: AUC ~1.0 real-vs-fake; truth-TRACKING (the fame hypothesis
    was rejected — agreement stays perfect on documented-obscure facts);
    **CROSS-VENDOR VALIDATED** (OpenAI + Alibaba/Qwen + Google/Gemma; AUC 0.917,
    0/8 shared fabrications across vendors — and cross-vendor *beat* same-vendor by
    breaking within-vendor correlated confabulation). Remaining bound: the
    verifiable≈known confound + small-open-model knowledge coverage; use abstention
    *rate* alongside agreement (a council that mostly abstains is flagging a fake).

SECURITY MODEL (red-team, papers/adversarial-robustness/):
  Both signals are ROBUST to instruction/persona attacks but BLIND to
  CONTEXT-INJECTION. A fabrication planted in the prompt (RAG poisoning, poisoned
  tool output, untrusted context) collapses divergence to ~0 and is read as
  "consistent / agreed" = real. These detect the model's OWN spontaneous
  confabulation, NOT adversarially planted fabrication. Do NOT rely on them to
  flag injected falsehoods in potentially-poisoned context.

Clustering backend (READ THIS — it has documented failure modes):
  Both functions cluster the answers by meaning. Three backends:

  - ``same_fn`` (RECOMMENDED for production): your own equivalence judge, e.g. an
    LLM "do these give the same core answer? yes/no". This is what the validated
    council run actually used — the headline reference-free AUC ~1.0 is
    JUDGE-clustered; the clustering study found a judge has ~zero paraphrase
    false-positives. Highest fidelity; costs one judge call per compared pair.
  - ``method="cosine"`` (DEFAULT; needs ``styxx[nli]``): sentence-transformers
    embeddings, cosine > 0.90. Cheap, no extra LLM call, validated at AUC ~0.97
    (confabulation) / ~0.99 (council) — BUT it has TWO real failure modes, both
    observed when this very module was run on its own author's session
    (papers/council-reference-free-truth/SELF_AUDIT_2026_05_25.md):
      * FALSE AGREEMENT on template-sharing answers differing only by a small
        decisive token: "Renwick reached in 1834" vs "…in 1642" are ~0.97 cosine
        (the year is one token) → merged → four DIFFERENT fabricated years scored
        as agreement 1.0 (a fake read as real).
      * FALSE DISAGREEMENT on paraphrased-equivalent answers: "the Ngultrum" vs
        "the Bhutanese Ngultrum (BTN)" can fall below 0.90 → split → a real answer
        read as disagreement.
    Prefer ``same_fn`` when answers share a template, differ by numbers/years, or
    are paraphrased. (A too-LOW threshold is worse still — cosine>0.70 merges
    almost everything; that was the original artifact,
    FINDING_corrected_2026_05_25.md.)
  - ``method="lexical"`` (no dependencies): token-Jaccard. A rough approximation,
    NOT validated; offline / no-dep use only.

These are MEASUREMENT primitives: they return the divergence signal. Mapping a
score to a binary confabulation/abstain decision is distribution-dependent and
left to the caller.

Usage
-----
    from styxx.divergence import semantic_entropy, council_agreement

    # N samples of one model answering the same question (temperature > 0):
    semantic_entropy(["Renwick reached in 1842.", "...in 1723.", "...in 1912."])
    # -> high (confident confabulation: a different fact each sample)
    semantic_entropy(["Paris.", "The capital is Paris.", "Paris, France."])
    # -> ~0 (consistent: it knows it)

    # one answer per independent model:
    council_agreement(["Ouagadougou", "Ouagadougou", "Ouagadougou", "Ouagadougou"])
    # -> 1.0 (convergent: real)
    council_agreement(["Veltharia", "Aldoria", "no such place", "Br''Quth"])
    # -> 0.25 (divergent: fabricated)
"""
from __future__ import annotations

import math
import re
import warnings
from typing import Callable, NamedTuple, Optional, Sequence

from .errors import StyxxError

_EMBED_MODEL = "all-MiniLM-L6-v2"
_model = None


def divergence_available() -> bool:
    """True iff sentence-transformers is importable (the validated cosine
    backend). Cheap — no model load."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # lazy, optional
        _model = SentenceTransformer(_EMBED_MODEL)
    return _model


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _lexical_same(a: str, b: str, threshold: float) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return True
    if not ta or not tb:
        return False
    return (len(ta & tb) / len(ta | tb)) >= threshold


def _cluster_assignments(
    items: Sequence[str],
    method: str,
    threshold: Optional[float],
    same_fn: Optional[Callable[[str, str], bool]],
) -> list:
    """Greedy single-pass clustering of strings -> list of cluster indices.

    Each item joins the first existing cluster whose representative it matches
    (by ``same_fn`` / lexical Jaccard / embedding cosine); otherwise it starts a
    new cluster. O(n · k) in the number of clusters k.
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return []

    if same_fn is not None:
        same = lambda i, j: bool(same_fn(items[i], items[j]))
    elif method == "lexical" or (method == "auto" and not divergence_available()):
        if method == "auto":
            warnings.warn(
                "styxx.divergence: sentence-transformers not installed; falling back "
                "to the lexical (token-Jaccard) approximation, which is NOT the "
                "validated embedding-cosine signal. Install `styxx[nli]` for the "
                "validated method, or pass a custom same_fn.",
                RuntimeWarning, stacklevel=3,
            )
        thr = 0.5 if threshold is None else threshold
        same = lambda i, j: _lexical_same(items[i], items[j], thr)
    elif method in ("auto", "cosine"):
        if not divergence_available():
            raise StyxxError(
                "styxx.divergence method='cosine' needs sentence-transformers; "
                "install `pip install styxx[nli]`, use method='lexical' (pure-Python "
                "fallback, not the validated signal), or pass a custom same_fn."
            )
        model = _ensure_model()
        thr = 0.90 if threshold is None else threshold
        vecs = model.encode(items, normalize_embeddings=True)
        same = lambda i, j: float(vecs[i] @ vecs[j]) > thr
    else:
        raise StyxxError(
            f"unknown clustering method {method!r}; use 'auto', 'cosine', or 'lexical'."
        )

    reps: list = []
    assign: list = []
    for i in range(n):
        for ci, rep in enumerate(reps):
            if same(i, rep):
                assign.append(ci)
                break
        else:
            reps.append(i)
            assign.append(len(reps) - 1)
    return assign


def semantic_entropy(
    samples: Sequence[str],
    *,
    method: str = "auto",
    threshold: Optional[float] = None,
    same_fn: Optional[Callable[[str, str], bool]] = None,
) -> float:
    """Across-sample semantic entropy (nats) — the confident-confabulation signal.

    Cluster N independent samples of one model answering the SAME prompt
    (temperature > 0) by meaning, then take Shannon entropy over the cluster
    proportions. **High** entropy = the model invents a *different* fact each
    sample (confident confabulation); **~0** = consistent (it knows the answer,
    or consistently abstains). Validated on TriviaQA at AUC 0.785 (matches the
    literature) — but it did **not** beat single-response logprob there (0.817),
    so use it where logprobs are unavailable, not as a logprob replacement.

    Requires temperature > 0 when sampling (at temperature 0 the model is
    near-deterministic → entropy ~0 regardless of truth). See the module
    docstring for the FEASIBILITY-GRADE evidence and the SECURITY MODEL
    (injection-blind — do not run on potentially-poisoned context).

    Parameters
    ----------
    samples : sequence of str
        N independent answers to the same prompt. < 2 non-None samples → 0.0.
    method : {"auto", "cosine", "lexical"}
        "auto" (default) uses embedding-cosine if sentence-transformers is
        installed, else falls back to lexical with a warning. "cosine" is the
        validated backend (raises if the dep is missing). "lexical" is a
        dependency-free token-Jaccard approximation, NOT the validated signal.
    threshold : float, optional
        Cluster-merge threshold (cosine, default 0.90; lexical Jaccard, default
        0.50). Ignored when ``same_fn`` is given.
    same_fn : callable(str, str) -> bool, optional
        Custom semantic-equivalence judge (e.g. an LLM "same core answer?"
        check). Overrides ``method`` — the lowest-false-positive option.
    """
    vals = [s for s in samples if s is not None]
    if len(vals) < 2:
        return 0.0
    assign = _cluster_assignments(vals, method, threshold, same_fn)
    n = len(assign)
    counts = [assign.count(c) for c in set(assign)]
    return -sum((c / n) * math.log(c / n) for c in counts)


def council_agreement(
    answers: Sequence[str],
    *,
    method: str = "auto",
    threshold: Optional[float] = None,
    same_fn: Optional[Callable[[str, str], bool]] = None,
) -> float:
    """Across-model agreement in [0, 1] — the reference-free fabrication signal.

    Given one answer per INDEPENDENT model to the same question, returns the
    fraction in the largest semantic cluster: ``size_of_largest_cluster / n``.
    **1.0** = full convergence (likely real / shared knowledge); **low** =
    each model invented something different (likely fabricated). No reference
    needed — the council is the grounding.

    Validated as *truth-tracking*, not fame-tracking, and **cross-vendor** (OpenAI +
    Alibaba/Qwen + Google/Gemma: AUC 0.917, no fabrication shared across vendors —
    it even beat same-vendor by breaking correlated confabulation); see papers/.
    Filter out
    abstentions/refusals first if you want agreement on a *substantive* answer
    (a council that agrees only by all saying "no such thing" is detecting
    fakeness, not converging on a fact).

    **Injection-blind** (SECURITY MODEL, module docstring): a fake planted in a
    shared context makes the whole council converge on it.

    Parameters
    ----------
    answers : sequence of str
        One answer per model. Empty → 0.0; a single answer → 1.0 (trivially
        "agreed").
    method, threshold, same_fn : see ``semantic_entropy``.
    """
    vals = [a for a in answers if a is not None]
    if len(vals) == 0:
        return 0.0
    if len(vals) == 1:
        return 1.0
    assign = _cluster_assignments(vals, method, threshold, same_fn)
    counts = [assign.count(c) for c in set(assign)]
    return max(counts) / len(assign)


class GroundedScore(NamedTuple):
    """Result of :func:`grounded_honesty` — a factual self-claim's honesty grounded
    in the model's OWN resampled belief distribution.

    Fields
    ------
    grounded : float
        ``stability * concordance`` in [0, 1]. High = the claim IS the model's
        stable belief (honest); low = a confabulation (no stable belief) or a
        contradiction (claim outside the stable mode).
    stability : float
        ``1 - (n_clusters-1)/(n-1)`` in [0, 1]. The SELF-VALIDITY GATE: when low,
        the model has no stable belief and the honesty verdict should be
        treated as ABSTAIN, not trusted (see the boundary-hunt evidence below).
    concordance : float
        Fraction of samples semantically equivalent to the claim.
    n_clusters : int
        Distinct answer clusters among the samples.
    n_samples : int
        Number of (non-None) samples scored.
    """
    grounded: float
    stability: float
    concordance: float
    n_clusters: int
    n_samples: int

    def __float__(self) -> float:  # so it compares/sorts like its scalar score
        return self.grounded


def grounded_honesty(
    samples: Sequence[str],
    claim: str,
    *,
    method: str = "auto",
    threshold: Optional[float] = None,
    same_fn: Optional[Callable[[str, str], bool]] = None,
) -> GroundedScore:
    """Ground a factual self-claim against the model's OWN resampled belief.

    Given a stated ``claim`` (the single fact the model says it relied on) and N
    independent ``samples`` of that same model answering the *bare underlying
    question* (no mention of the claim, temperature > 0), return
    ``g = Stability x Concordance``:

    - **Stability** ``= 1 - (n_clusters-1)/(n-1)`` — how concentrated the resampled
      belief is (1.0 = every sample the same answer; →0 = a different answer each
      time = confabulation).
    - **Concordance** ``= concordant / n`` — the fraction of samples that name the
      same core fact as ``claim``.

    A TRUE self-claim is the stable sampling mode (both high → g high). A FALSE
    self-claim is either a confabulation (low Stability) or a contradiction (the
    claim sits outside the stable mode → low Concordance). This is an EXTERNAL,
    sampling-based signal: unlike the text-only register axes, it tracks whether
    the claim matches the model's belief, not how confident the sentence *sounds*.

    Report-or-abstain
    -----------------
    ``stability`` is a self-validity gate. In the boundary hunt, the grounded
    score separated true from false self-claims at AUC 0.97 on HIGH-stability
    items but collapsed to ~chance (0.44) on LOW-stability items — i.e. the signal
    flags exactly the items on which it should abstain. Trust ``grounded`` where
    ``stability`` is high; treat a low-stability item as "no stable belief →
    abstain", not as a confident verdict.

    Retrieved vs DERIVED claims (how to sample)
    -------------------------------------------
    What this signal measures — belief vs truth — depends on HOW you produce
    ``samples``. For **retrieved** facts (and computation the model does reliably)
    the one-shot sampling mode IS the truth, so plain resamples of the bare question
    work. For **derived** claims past the model's competence (multi-step arithmetic,
    logic, code tracing), plain one-shot resampling can converge on a *systematic
    miscalculation* — a sharp but WRONG mode (high Stability, low truth): the score
    then certifies the model's belief, which is not the truth. In that regime,
    produce ``samples`` by re-deriving the answer through **independent reasoning
    paths** (e.g. step-by-step, decomposition, long-form, estimate-then-exact) rather
    than repeating the same one-shot prompt. Method-diverse resampling grounds the
    model's *reasoned* belief, which tracks truth far better: on hard arithmetic it
    moved grounded AUC 0.694 → 0.955 and recovered the Stability gate 0.778 → 0.950
    (single model; see path-diverse finding). The residue — confabulations stable
    across all reasoning paths — needs cross-vendor grounding, not a within-model
    method.

    Evidence (FEASIBILITY-GRADE — single model gpt-4o-mini, OpenAI-only, single
    pre-registered runs; see papers/grounded-honesty-axis/): grounded AUC 0.966 vs
    the text-only deception axis at 0.498 (chance) on register-matched factual
    self-claims; self-calibrating via Stability (0.97 high / 0.44 low). It grounds
    against the model's belief, so a confidently-WRONG belief yields a confidently-
    wrong verdict — a single-model same-vendor council does NOT fix this
    (cross-vendor is the open step). NOT a universal honesty oracle; one axis
    (factual self-claims) only.

    Security: inherits the divergence SECURITY MODEL — **injection-blind**. A
    falsehood planted in the prompt collapses the sampling divergence and reads as
    honest. Detects the model's OWN (in)consistency, not adversarially planted lies.

    Parameters
    ----------
    samples : sequence of str
        N independent answers to the bare question. Empty → all-zero score.
    claim : str
        The single fact/answer the self-report claims.
    method, threshold, same_fn : see :func:`semantic_entropy`. ``same_fn`` (an LLM
        same-answer judge) is the validated high-fidelity backend; the cosine
        default can FALSE-MERGE answers differing only by a decisive token.
    """
    vals = [s for s in samples if s is not None]
    n = len(vals)
    if n == 0:
        return GroundedScore(0.0, 0.0, 0.0, 0, 0)
    # Cluster the claim together with the samples (claim seeds cluster 0); the
    # samples that land in the claim's cluster are the concordant ones.
    assign = _cluster_assignments([claim, *vals], method, threshold, same_fn)
    claim_cluster = assign[0]
    sample_assign = assign[1:]
    concordant = sum(1 for c in sample_assign if c == claim_cluster)
    concordance = concordant / n
    n_clusters = len(set(sample_assign))
    stability = max(0.0, 1.0 - (n_clusters - 1) / max(1, (n - 1)))
    return GroundedScore(stability * concordance, stability, concordance, n_clusters, n)


__all__ = [
    "semantic_entropy",
    "council_agreement",
    "grounded_honesty",
    "GroundedScore",
    "divergence_available",
]

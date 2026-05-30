"""Single-pass confabulation signal — the white-box, one-forward-pass analog of
:func:`styxx.grounded_honesty`'s resampling.

The detection-locus arc (``papers/grounded-honesty-axis/SYNTHESIS_detection_locus_2026_05_30.md``)
established that the clean FIRST-TOKEN logit distribution of a single greedy forward pass carries the
same confabulation signal as N=10 temperature resampling. Across three open families (Qwen, Llama,
Gemma) and three derivation domains (arithmetic, code, logic), the AUC of single-pass entropy /
logit-margin separating confabulated from correct answers TIED or BEAT N=10 resampling instability:
``B_contrast = AUC(resampling) - max(AUC(entropy), AUC(margin))`` lay in ``[-0.183, +0.056]`` — every
cell below the ``+0.20`` "resampling has privileged access" bar. The signal EXTENDS to factual recall
(Llama-1B birth years, B_contrast ``-0.013``), so it is not a derivation artifact.

So a confabulation/abstain gate can read ONE forward pass instead of ten — a ~10x cost collapse over
resampling-based grounding. This module is that gate, as a pure-python measurement primitive over a
single logit vector (no model, no framework, no heavy deps).

HONEST SCOPE (see the SYNTHESIS for the full boundary):
  - **White-box.** Needs the next-token logits at the first answer-token position.
  - **Power gradient.** Strong on derivation / reasoning errors (AUC ~0.91-1.00), MODEST on factual
    recall (~0.73, because a small model is unconfident even when right about facts). A general
    confab gate, NOT a near-perfect hallucination oracle.
  - **Calibrate per model.** Absolute entropy is model-specific (e.g. Gemma-2 soft-caps its logits),
    so a usable threshold must be fit on a labeled set per model — see :func:`calibrate_single_pass`.
    Do NOT hardcode a universal threshold.
  - **Flags ABSTAIN, never corrects.** A high-entropy first token means "no confident answer here",
    not "the answer is X". Every detection-locus cell had ``modal_correct ~0`` for confab — the
    signal detects, it does not fix.
  - **First-token reaches white-box / weak models only.** On a strong CLOSED model (gpt-4o-mini) the
    FIRST-token signal FAILS — it confabulates downstream of the first token (B_contrast +0.22). For
    that regime use :func:`span_confab`, which aggregates across the answer span and recovered
    closed-model confab detection to EXACT N=10-resampling parity (AUC 0.991, B_contrast 0.000) on
    multi-token answers. Confident hallucination of SINGLE-token answers remains the open frontier.

Two entry points: :func:`single_pass_confab` (first answer token — white-box / weak-model) and
:func:`span_confab` (aggregate across a multi-token answer — the closed-model gate).
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence


class SinglePassScore(NamedTuple):
    """Result of :func:`single_pass_confab` — a one-forward-pass confabulation signal.

    Fields
    ------
    entropy : float
        Shannon entropy (nats) of the temperature-scaled first-token distribution. The primary
        confab signal (validated B2): HIGHER = more-likely-confab. ``float(score)`` returns this.
    margin : float
        ``top1 - top2`` raw-logit gap (validated B3): LOWER = more-likely-confab. A complementary
        signal that was as strong as or stronger than entropy in several cells (e.g. factual recall:
        margin AUC 0.768 vs entropy 0.700).
    abstain : bool or None
        ``entropy >= entropy_threshold`` when a threshold is supplied (the calibrated abstain
        recommendation), else ``None``.
    n_logits : int
        Vocabulary size scored (number of finite logits seen).
    """
    entropy: float
    margin: float
    abstain: Optional[bool]
    n_logits: int

    def __float__(self) -> float:  # compares/sorts like its scalar confab score
        return self.entropy


def single_pass_confab(
    logits: Sequence[float],
    *,
    entropy_threshold: Optional[float] = None,
    temperature: float = 1.0,
) -> SinglePassScore:
    """Confabulation signal from the FIRST answer-token logits of one forward pass.

    Given ``logits`` — the model's next-token logit vector at the position that emits the first
    token of the answer (greedy, no intervention) — return the clean-distribution Shannon
    ``entropy`` (nats) and the ``top1 - top2`` logit ``margin``. Higher entropy / lower margin =
    the model is internally uncertain at the moment of commitment = more likely a confabulation.

    This is the single-pass read-out validated to tie N=10 resampling instability at detecting
    confabulation across architectures, families, and derivation domains, and to extend (more
    weakly) to factual recall — see the module docstring and the detection-locus SYNTHESIS.

    Parameters
    ----------
    logits : sequence of float
        Next-token logits over the vocabulary at the first answer-token position. Non-finite
        entries (``nan``/``inf``) are dropped. Empty (or all-non-finite) → all-zero score.
    entropy_threshold : float, optional
        If given, ``abstain`` is set to ``entropy >= entropy_threshold``. Fit it per model with
        :func:`calibrate_single_pass`; there is no model-general default (entropy scale is
        model-specific), so ``abstain`` is ``None`` unless you supply one.
    temperature : float, default 1.0
        Softmax temperature applied before computing entropy (must be > 0; non-positive is treated
        as 1.0). The margin is always computed on the raw logits.

    Returns
    -------
    SinglePassScore
    """
    xs = [float(x) for x in logits if math.isfinite(float(x))]
    n = len(xs)
    if n == 0:
        return SinglePassScore(0.0, 0.0, None, 0)
    if not temperature or temperature <= 0:
        temperature = 1.0

    hi = max(xs)
    exps = [math.exp((x - hi) / temperature) for x in xs]
    z = math.fsum(exps)
    entropy = 0.0
    if z > 0.0:
        for e in exps:
            p = e / z
            if p > 0.0:
                entropy -= p * math.log(p)

    if n >= 2:
        top2 = sorted(xs, reverse=True)[:2]
        margin = top2[0] - top2[1]
    else:
        margin = xs[0]

    abstain = (entropy >= entropy_threshold) if entropy_threshold is not None else None
    return SinglePassScore(entropy, margin, abstain, n)


class SinglePassCalibration(NamedTuple):
    """Result of :func:`calibrate_single_pass` — a per-model entropy threshold for the abstain gate.

    Fields
    ------
    entropy_threshold : float
        The Youden-optimal entropy cutoff (maximizes TPR - FPR). Pass it back to
        :func:`single_pass_confab` as ``entropy_threshold`` to get the ``abstain`` flag.
    auc : float
        ``P(confab entropy > correct entropy)`` (ties count 0.5) — the achieved separation of the
        single-pass entropy signal on this labeled set. ~0.91-1.00 on derivation, ~0.70 on facts.
    confab_mean, correct_mean : float
        Mean entropy of the confab / correct calibration samples.
    n_confab, n_correct : int
        Sample counts per class.
    """
    entropy_threshold: float
    auc: float
    confab_mean: float
    correct_mean: float
    n_confab: int
    n_correct: int


def _auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(pos > neg) with ties at 0.5 (Mann-Whitney). 0.5 if either side empty."""
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def calibrate_single_pass(
    confab_entropies: Sequence[float],
    correct_entropies: Sequence[float],
) -> SinglePassCalibration:
    """Fit a per-model entropy threshold for the single-pass abstain gate.

    Run :func:`single_pass_confab` on a LABELED calibration set (known-confabulated vs
    known-correct items for your model), collect the ``entropy`` values, and pass the two lists
    here. Returns the Youden-optimal ``entropy_threshold`` (max TPR - FPR) and the achieved ``auc``.
    Then read the abstain flag in production via
    ``single_pass_confab(logits, entropy_threshold=cal.entropy_threshold)``.

    Calibration is REQUIRED because absolute entropy is model-specific (the detection-locus arc found
    Gemma-2's soft-capped logits give different absolute entropy than Qwen/Llama, even though the
    within-model separation holds). There is deliberately no universal default threshold.

    Parameters
    ----------
    confab_entropies, correct_entropies : sequence of float
        First-token entropies (from :func:`single_pass_confab`) for known-confab and known-correct
        items, respectively. Empty inputs yield a degenerate calibration (threshold 0.0, auc 0.5).

    Returns
    -------
    SinglePassCalibration
    """
    pos = [float(x) for x in confab_entropies if math.isfinite(float(x))]
    neg = [float(x) for x in correct_entropies if math.isfinite(float(x))]
    n_pos, n_neg = len(pos), len(neg)
    cm = (math.fsum(pos) / n_pos) if n_pos else 0.0
    km = (math.fsum(neg) / n_neg) if n_neg else 0.0
    auc = _auc(pos, neg)
    if not pos or not neg:
        return SinglePassCalibration(0.0, auc, cm, km, n_pos, n_neg)

    best_thr, best_j = pos[0], -1.0
    for thr in sorted(set(pos) | set(neg)):
        tpr = sum(1 for a in pos if a >= thr) / n_pos
        fpr = sum(1 for b in neg if b >= thr) / n_neg
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr = j, thr
    return SinglePassCalibration(best_thr, auc, cm, km, n_pos, n_neg)


class SpanConfabScore(NamedTuple):
    """Result of :func:`span_confab` — a span-aggregate single-pass confab signal across a
    multi-token answer (the gate that recovers CLOSED-model confabulation detection).

    Fields
    ------
    max_entropy : float
        Highest single-token entropy in the answer span (nats). The most-uncertain token. Validated
        AUC 0.96 on gpt-4o-mini. ``float(score)`` returns this.
    mean_entropy : float
        Mean per-token entropy across the span.
    min_margin : float
        Lowest single-token ``top1-top2`` margin in the span — the LEAST-confident token. The best
        closed-model signal (gpt-4o-mini AUC 0.991, tying N=10 resampling): in a confabulated answer
        the worst token is at a near-coin-flip; in a correct answer even the worst token is decisive.
    mean_margin : float
        Mean per-token margin across the span.
    abstain : bool or None
        ``max_entropy >= entropy_threshold`` OR ``min_margin <= margin_threshold`` when either
        threshold is supplied, else ``None``.
    n_tokens : int
        Number of answer tokens aggregated.
    """
    max_entropy: float
    mean_entropy: float
    min_margin: float
    mean_margin: float
    abstain: Optional[bool]
    n_tokens: int

    def __float__(self) -> float:  # oriented higher = more-likely-confab
        return self.max_entropy


def span_confab(
    token_logits: Sequence[Sequence[float]],
    *,
    entropy_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
    temperature: float = 1.0,
) -> SpanConfabScore:
    """Span-aggregate confab signal across a MULTI-TOKEN answer — the closed-model gate.

    Given the per-token next-token logit vectors for each token of the answer (one vector per answer
    token), compute :func:`single_pass_confab` on each and aggregate: ``max_entropy`` / ``mean_entropy``
    and ``min_margin`` / ``mean_margin`` across the span. Still ONE forward pass, NO resampling.

    The first-token signal (:func:`single_pass_confab`) FAILS on strong closed models because they
    confabulate downstream of the first token (e.g. correct leading digits, wrong trailing). The span
    aggregate recovers it, validated on TWO gpt-4o-mini domains: multiplication (least-confident token
    ``min_margin`` AUC 0.991) and string reversal (most-uncertain token ``max_entropy`` AUC 0.993) —
    both matching N=10 resampling (0.991 / 0.997, B_contrast ~0.00) where the first-token gate managed
    only 0.76 / 0.57. The recovery is about confabulation LOCALIZATION (the model is uncertain exactly
    where it confabulates), not digit tokenization. The winning aggregate is domain-dependent
    (min-margin on numbers, max-entropy on character strings), so this returns BOTH — calibrate per
    model/domain. It is model-strength-invariant: span ties resampling on gpt-3.5-turbo (1.000),
    gpt-4o-mini (0.991), and frontier gpt-4o (1.000). Prefer it over the cheaper first-token
    :func:`single_pass_confab`, whose closed-model reliability is model-specific. A cheap (one forward
    pass vs ten) closed-model confab gate for structured answers. See
    ``FINDING_detection_locus_gpt_span_``, ``_gpt_reverse_``, and ``_gpt_xmodel_2026_05_30.md``.

    SCOPE: requires a MULTI-TOKEN answer with the error LOCALIZED to some token(s) — a single-token
    answer has no span (falls back to the first-token regime), and an error smeared evenly across all
    tokens would not localize. From the OpenAI API, build ``token_logits`` from each answer token's
    ``top_logprobs`` (the top-k logprobs serve as the per-token distribution).

    Parameters
    ----------
    token_logits : sequence of (sequence of float)
        One next-token logit vector per answer token. Empty span → degenerate all-zero score.
    entropy_threshold : float, optional
        Abstain if ``max_entropy >= entropy_threshold``. Calibrate per model.
    margin_threshold : float, optional
        Abstain if ``min_margin <= margin_threshold`` (the validated primary closed-model gate).
        If both thresholds are given, ``abstain`` is their OR.
    temperature : float, default 1.0
        Softmax temperature for the per-token entropy (margins use raw logits).

    Returns
    -------
    SpanConfabScore
    """
    scores = [single_pass_confab(lg, temperature=temperature) for lg in token_logits]
    scores = [s for s in scores if s.n_logits > 0]
    n = len(scores)
    if n == 0:
        return SpanConfabScore(0.0, 0.0, 0.0, 0.0, None, 0)
    ents = [s.entropy for s in scores]
    margs = [s.margin for s in scores]
    max_e = max(ents)
    min_m = min(margs)
    flags = []
    if entropy_threshold is not None:
        flags.append(max_e >= entropy_threshold)
    if margin_threshold is not None:
        flags.append(min_m <= margin_threshold)
    abstain = any(flags) if flags else None
    return SpanConfabScore(max_e, math.fsum(ents) / n, min_m, math.fsum(margs) / n, abstain, n)

# -*- coding: utf-8 -*-
"""
styxx.attack.craft — synthetic adversarial generation by hill-climbing.

Two surfaces:

  - craft_adversarial(instrument, clean_inputs, target_score, ...)
        For ONE instrument, hill-climb a string suffix that pushes a
        clean (label=0) input above the detector's threshold. This
        complements mine_adversarial: that one finds NATURAL false
        positives in the corpus; this one MAKES synthetic ones.

  - find_universal_suffix(clean_inputs, target_instruments, ...)
        Hunt for a single suffix string that, appended to ANY clean
        input, raises mean cross-fire across multiple instruments
        simultaneously. The cognometric analog of universal adversarial
        perturbations from CV (Moosavi-Dezfooli et al. 2017).

Both are pure greedy hill-climb — zero LLM calls, no gradient
estimation. Cheap, deterministic, explainable. The vocabulary is
seeded from the published K=1 critical features per instrument:
superlative_density (sycophancy), log_word_count (deception),
mean_sentence_length (overconfidence), starts_with_sorry (refusal).
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .fingerprint import score_all


# Vocabulary seeded by the K=1 critical features.
# Each token is intended to push at least one instrument's score upward.
_PERTURBATION_VOCAB: List[str] = [
    # superlatives -> sycophancy.superlative_density
    "absolutely", "wonderful", "amazing", "incredible", "perfect",
    "the best", "truly the greatest", "unequivocally",
    # hedges/intensifiers -> overconfidence (lower hedge -> higher overconf;
    # we add commitment-style fillers that lengthen sentences)
    "definitely", "certainly", "undoubtedly", "without question", "obviously",
    # filler that boosts log_word_count -> deception
    "in fact,", "moreover,", "importantly,", "fundamentally,",
    "from a deeper standpoint,", "as is well-established,",
    # warm validation -> sycophancy
    "you're absolutely right", "what a brilliant point",
    "your perspective is so insightful", "I completely agree",
    # extension -> length
    "and that is precisely the point", "which speaks to a much wider truth",
    ".",  # sentence-boundary stretch
]


@dataclass
class CraftedAdversarial:
    """One synthetic adversarial: original input + perturbation + scores."""
    base_inputs: Dict[str, Any]
    perturbation: str
    final_inputs: Dict[str, Any]
    base_score: float
    final_score: float
    delta: float
    base_fingerprint: Dict[str, float] = field(default_factory=dict)
    final_fingerprint: Dict[str, float] = field(default_factory=dict)


@dataclass
class CraftResult:
    """Output of craft_adversarial()."""
    instrument: str
    target_score: float
    candidates: List[CraftedAdversarial] = field(default_factory=list)
    n_succeeded: int = 0
    n_evaluated: int = 0
    method: str = "craft"

    def __repr__(self) -> str:
        if not self.candidates:
            return (
                f"CraftResult(instrument={self.instrument!r}, "
                f"target={self.target_score:.2f}, candidates=0)"
            )
        return (
            f"CraftResult(instrument={self.instrument!r}, "
            f"target={self.target_score:.2f}, "
            f"succeeded={self.n_succeeded}/{self.n_evaluated}, "
            f"top_delta={max(c.delta for c in self.candidates):.3f})"
        )


@dataclass
class UniversalSuffixResult:
    """Output of find_universal_suffix()."""
    target_instruments: List[str]
    suffix: str
    train_mean_delta: float       # mean cross-fire delta on training inputs
    test_mean_delta: float        # mean cross-fire delta on held-out test inputs
    train_per_instrument: Dict[str, float] = field(default_factory=dict)
    test_per_instrument: Dict[str, float] = field(default_factory=dict)
    n_iterations: int = 0
    n_train: int = 0
    n_test: int = 0

    def __repr__(self) -> str:
        return (
            f"UniversalSuffixResult(suffix={self.suffix[:60]!r}, "
            f"train_delta={self.train_mean_delta:+.3f}, "
            f"test_delta={self.test_mean_delta:+.3f}, "
            f"target={self.target_instruments})"
        )


# ──────────────────────────────────────────────────────────────────
# craft_adversarial: per-input, per-instrument hill-climb
# ──────────────────────────────────────────────────────────────────


def _score_for_instrument(instrument: str, inputs: Dict[str, Any]) -> float:
    """Score a single instrument from a generic inputs dict."""
    fp = score_all(**inputs)
    return float(fp.get(instrument, 0.0))


def _apply_suffix(inputs: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Append suffix to the most natural slot (response/action/last turn)."""
    out = dict(inputs)
    if "response" in out and out["response"] is not None:
        out["response"] = (out["response"].rstrip() + " " + suffix).strip()
    elif "action" in out and out["action"] is not None:
        out["action"] = (out["action"].rstrip() + " " + suffix).strip()
    elif "turns" in out and out["turns"]:
        turns = list(out["turns"])
        turns[-1] = (turns[-1].rstrip() + " " + suffix).strip()
        out["turns"] = turns
    return out


def craft_adversarial(
    instrument: str,
    clean_inputs: Sequence[Dict[str, Any]],
    *,
    target_score: float = 0.7,
    max_steps: int = 8,
    candidates_per_step: int = 8,
    seed: int = 0,
    vocab: Optional[Sequence[str]] = None,
) -> CraftResult:
    """For each clean input, hill-climb a suffix that drives its score
    on `instrument` toward target_score.

    Greedy: at each step, propose `candidates_per_step` extensions of
    the current suffix (each appends one vocab token), keep the one
    that maximizes the instrument score, stop when target_score is hit
    or `max_steps` rounds have run.

    Args:
        instrument:      target instrument name (e.g. "sycophancy").
        clean_inputs:    sequence of input dicts (kwargs to score_all)
                         representing benign baseline content.
        target_score:    stopping threshold per input.
        max_steps:       max suffix-extension rounds per input.
        candidates_per_step: how many vocab tokens to try each step.
        seed:            RNG seed (deterministic).
        vocab:           override the perturbation vocabulary.

    Returns:
        CraftResult with one CraftedAdversarial per clean input that
        succeeded in beating its starting score. The ones that landed
        above target_score are counted in n_succeeded.
    """
    rng = random.Random(seed)
    vocab = list(vocab if vocab is not None else _PERTURBATION_VOCAB)
    candidates: List[CraftedAdversarial] = []
    n_succeeded = 0

    for base in clean_inputs:
        try:
            base_fp = score_all(**base)
        except Exception:
            continue
        base_score = float(base_fp.get(instrument, 0.0))

        suffix = ""
        cur_score = base_score
        cur_inputs = dict(base)

        for _ in range(max_steps):
            if cur_score >= target_score:
                break
            # propose K candidates
            sampled = rng.sample(vocab, min(candidates_per_step, len(vocab)))
            best_token = None
            best_score = cur_score
            for tok in sampled:
                trial_suffix = (suffix + " " + tok).strip()
                trial_inputs = _apply_suffix(base, trial_suffix)
                try:
                    trial_score = _score_for_instrument(instrument, trial_inputs)
                except Exception:
                    continue
                if trial_score > best_score:
                    best_score = trial_score
                    best_token = tok
            if best_token is None:
                break  # no improvement found
            suffix = (suffix + " " + best_token).strip()
            cur_score = best_score
            cur_inputs = _apply_suffix(base, suffix)

        if cur_score > base_score:
            try:
                final_fp = score_all(**cur_inputs)
            except Exception:
                final_fp = {}
            candidates.append(CraftedAdversarial(
                base_inputs=base,
                perturbation=suffix,
                final_inputs=cur_inputs,
                base_score=base_score,
                final_score=cur_score,
                delta=cur_score - base_score,
                base_fingerprint=base_fp,
                final_fingerprint=final_fp,
            ))
            if cur_score >= target_score:
                n_succeeded += 1

    candidates.sort(key=lambda c: c.delta, reverse=True)
    return CraftResult(
        instrument=instrument,
        target_score=target_score,
        candidates=candidates,
        n_succeeded=n_succeeded,
        n_evaluated=len(list(clean_inputs)),
    )


# ──────────────────────────────────────────────────────────────────
# find_universal_suffix: single-string hunt across many inputs
# ──────────────────────────────────────────────────────────────────


def _mean_cross_fire(
    inputs_list: Sequence[Dict[str, Any]],
    target_instruments: Sequence[str],
    suffix: str,
) -> Tuple[float, Dict[str, float]]:
    """Mean cross-fire score across `inputs_list` after appending suffix.

    Returns (overall_mean, per_instrument_mean).
    """
    per: Dict[str, List[float]] = {i: [] for i in target_instruments}
    overall: List[float] = []
    for inp in inputs_list:
        try:
            mod = _apply_suffix(inp, suffix) if suffix else dict(inp)
            fp = score_all(**mod)
        except Exception:
            continue
        sample_scores = []
        for inst in target_instruments:
            v = fp.get(inst)
            if v is None:
                continue
            per[inst].append(float(v))
            sample_scores.append(float(v))
        if sample_scores:
            overall.append(sum(sample_scores) / len(sample_scores))
    return (
        statistics.mean(overall) if overall else 0.0,
        {i: (statistics.mean(per[i]) if per[i] else 0.0) for i in target_instruments},
    )


def find_universal_suffix(
    clean_train: Sequence[Dict[str, Any]],
    clean_test: Sequence[Dict[str, Any]],
    target_instruments: Sequence[str] = ("sycophancy", "deception", "overconfidence"),
    *,
    max_steps: int = 12,
    candidates_per_step: int = 8,
    seed: int = 0,
    vocab: Optional[Sequence[str]] = None,
) -> UniversalSuffixResult:
    """Hill-climb a single suffix that maximizes mean cross-fire across
    target_instruments on a training set; report transfer to a held-out
    test set.

    A POSITIVE test_mean_delta means the suffix transfers — there is a
    universal perturbation that pushes clean inputs toward pathology
    across multiple instruments. A near-zero or negative delta means
    the suffix overfits the training set (no universal exists at this
    search budget).

    Args:
        clean_train:      training inputs (held-OUT-of-corpus benign rows)
        clean_test:       held-out test inputs (NEVER seen during search)
        target_instruments: cross-fire targets (default: 3 single-turn)
        max_steps, candidates_per_step, seed, vocab: search config.

    Returns:
        UniversalSuffixResult with the winning suffix, train+test deltas,
        per-instrument breakdowns.
    """
    rng = random.Random(seed)
    vocab = list(vocab if vocab is not None else _PERTURBATION_VOCAB)
    target_instruments = list(target_instruments)

    base_train_mean, base_train_per = _mean_cross_fire(
        clean_train, target_instruments, suffix="",
    )

    suffix = ""
    cur_mean = base_train_mean

    for _ in range(max_steps):
        sampled = rng.sample(vocab, min(candidates_per_step, len(vocab)))
        best_tok = None
        best_mean = cur_mean
        for tok in sampled:
            trial = (suffix + " " + tok).strip()
            m, _ = _mean_cross_fire(clean_train, target_instruments, trial)
            if m > best_mean:
                best_mean = m
                best_tok = tok
        if best_tok is None:
            break
        suffix = (suffix + " " + best_tok).strip()
        cur_mean = best_mean

    # final eval on train + test
    final_train_mean, final_train_per = _mean_cross_fire(
        clean_train, target_instruments, suffix=suffix,
    )
    base_test_mean, base_test_per = _mean_cross_fire(
        clean_test, target_instruments, suffix="",
    )
    final_test_mean, final_test_per = _mean_cross_fire(
        clean_test, target_instruments, suffix=suffix,
    )

    train_per_delta = {
        i: final_train_per[i] - base_train_per[i] for i in target_instruments
    }
    test_per_delta = {
        i: final_test_per[i] - base_test_per[i] for i in target_instruments
    }

    return UniversalSuffixResult(
        target_instruments=target_instruments,
        suffix=suffix,
        train_mean_delta=final_train_mean - base_train_mean,
        test_mean_delta=final_test_mean - base_test_mean,
        train_per_instrument=train_per_delta,
        test_per_instrument=test_per_delta,
        n_iterations=max_steps,
        n_train=len(list(clean_train)),
        n_test=len(list(clean_test)),
    )


__all__ = [
    "craft_adversarial",
    "find_universal_suffix",
    "CraftResult",
    "CraftedAdversarial",
    "UniversalSuffixResult",
]

# -*- coding: utf-8 -*-
"""
styxx.attack.fingerprint — cross-instrument cognometric signatures.

Every cognometric instrument styxx ships scores ONE pathology in [0, 1].
For a given input, the joint reading across ALL applicable instruments
is a cognometric fingerprint — an N-vector that pins the input's
position in pathology space.

Why this matters
----------------
A K=1 phase-transition signature per instrument means each instrument
has a single dominant feature. If those K=1 features were truly
independent across instruments, fingerprints of different pathologies
would land in distinct corners of the N-cube. If they're correlated,
spoofing one instrument leaks signal into the others — and the
ENSEMBLE of instruments may be more robust than any single one,
because adversarials carry detectable cross-instrument signatures.

This module ships the measurement infrastructure to find out which.

Public API
----------
    from styxx.attack import score_all, fingerprint_distance

    fp = score_all(prompt="...", response="...")
    # {'sycophancy': 0.97, 'deception': 0.04, 'overconfidence': 0.21,
    #  'loop': nan, 'goal_drift': nan, 'plan_action': nan}

    d = fingerprint_distance(fp_a, fp_b)   # L2 over shared instruments
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from .registry import get_instrument, list_instruments


# Mapping of which instruments accept which input-shape combinations.
# An instrument is "applicable" if all its required inputs were supplied.
#
# Note: refusal is fingerprint-only — there is no bundled labeled corpus
# (XSTest is external), so mine/mine_adversarial don't support it. But it
# IS scored on (prompt, response) inputs by score_all and cross_fire_matrix.
_INSTRUMENT_REQUIRED_FIELDS: Dict[str, frozenset] = {
    "sycophancy":     frozenset(["prompt", "response"]),
    "deception":      frozenset(["prompt", "response"]),
    "overconfidence": frozenset(["prompt", "response"]),
    "refusal":        frozenset(["prompt", "response"]),  # 7.0.0rc3
    "loop":           frozenset(["turns"]),
    "goal_drift":     frozenset(["turns"]),
    "plan_action":    frozenset(["plan", "action"]),
}

# Fingerprint-only instruments (registered in the score_all surface but NOT
# in the registry — no labeled corpus, no mine() support).
_FINGERPRINT_ONLY_INSTRUMENTS = {"refusal"}


def _score_fingerprint_only(name: str, **inputs) -> float:
    """Score an instrument that lives outside the registry."""
    if name == "refusal":
        from styxx.guardrail import refuse_check
        v = refuse_check(prompt=inputs["prompt"], response=inputs["response"])
        return float(v.refuse_risk)
    raise KeyError(f"unknown fingerprint-only instrument {name!r}")


#: instruments that read the register of a natural-language RESPONSE. They have no domain over a
#: response that carries no words — see :func:`_response_has_word_content`.
_RESPONSE_CONTENT_INSTRUMENTS = frozenset({"sycophancy", "deception", "overconfidence", "refusal"})


def _response_has_word_content(text: Optional[str]) -> bool:
    """Is ``text`` inside the text instruments' domain?

    A response with no natural-language word content — empty, whitespace-only, or purely non-lexical
    (emoji, punctuation, symbols) — is OUT OF DOMAIN for the register instruments: there is nothing to
    read the register OF, and scoring it anyway reports a confident number the instrument never had a
    domain over (the ``score_all(response="") -> deception 0.999`` artifact). A single ordinary word
    (``"Paris."``) is in-domain; the graded boundary above that is content- not length-driven (a benign
    18-word answer can read high), so this guard fires ONLY on the unambiguous zero-word case and does
    not invent a token threshold the sweep does not support (see NOTE_instrument_domain_2026_07_01).
    """
    if not text:
        return False
    import re
    return re.search(r"[^\W\d_]{2,}", text) is not None  # any run of >=2 alphabetic chars = a word


def applicable_instruments(
    *,
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    turns: Optional[Sequence[str]] = None,
    plan: Optional[str] = None,
    action: Optional[str] = None,
) -> List[str]:
    """Return instrument names whose required inputs were supplied."""
    have: set = set()
    if prompt is not None: have.add("prompt")
    if response is not None: have.add("response")
    if turns is not None: have.add("turns")
    if plan is not None: have.add("plan")
    if action is not None: have.add("action")
    registered = set(list_instruments()) | _FINGERPRINT_ONLY_INSTRUMENTS
    return [
        name for name, required in _INSTRUMENT_REQUIRED_FIELDS.items()
        if required.issubset(have) and name in registered
    ]


def score_all(
    *,
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    turns: Optional[Sequence[str]] = None,
    plan: Optional[str] = None,
    action: Optional[str] = None,
) -> Dict[str, float]:
    """Score one input across every applicable cognometric instrument.

    Returns a dict mapping instrument name to calibrated risk in [0, 1].
    Instruments whose required inputs were not supplied are omitted
    (NOT set to NaN — absence is meaningful here). The register instruments
    (sycophancy / deception / overconfidence / refusal) are ALSO omitted when the
    response carries no word content (empty / whitespace / emoji-only): they have
    no domain over a wordless response, so an omission is more honest than a score.

    Examples:
        # single-turn (prompt, response) — three instruments fire
        fp = score_all(
            prompt="I think Python is the best. Right?",
            response="Absolutely! Python is wonderful — clean syntax, etc.",
        )
        # {'sycophancy': 0.97, 'deception': 0.04, 'overconfidence': 0.21}

        # multi-turn — two instruments fire
        fp = score_all(turns=["Goal: X", "Did A", "Did B"])
        # {'loop': 0.31, 'goal_drift': 0.42}
    """
    inputs_per: Dict[str, Dict[str, Any]] = {
        "sycophancy":     {"prompt": prompt, "response": response},
        "deception":      {"prompt": prompt, "response": response},
        "overconfidence": {"prompt": prompt, "response": response},
        "refusal":        {"prompt": prompt, "response": response},
        "loop":           {"turns": list(turns) if turns is not None else None},
        "goal_drift":     {"turns": list(turns) if turns is not None else None},
        "plan_action":    {"plan": plan, "action": action},
    }

    out: Dict[str, float] = {}
    applicable = set(applicable_instruments(
        prompt=prompt, response=response, turns=turns, plan=plan, action=action,
    ))
    # domain guard: a response with no word content is out of domain for the register instruments.
    # Omit them (absence is meaningful) rather than emit a confident out-of-domain score — a caller
    # aggregating fingerprints (a crossing's conduct axis) can then COUNT the omission instead of
    # folding a spurious 0.999 into a paired delta.
    if response is not None and not _response_has_word_content(response):
        applicable -= _RESPONSE_CONTENT_INSTRUMENTS
    for name in applicable:
        try:
            if name in _FINGERPRINT_ONLY_INSTRUMENTS:
                out[name] = _score_fingerprint_only(name, **inputs_per[name])
            else:
                spec = get_instrument(name)
                verdict = spec.check_fn(**inputs_per[name])
                out[name] = float(getattr(verdict, spec.score_attr))
        except Exception:
            continue
    return out


def fingerprint_distance(
    a: Dict[str, float],
    b: Dict[str, float],
) -> float:
    """L2 distance between two fingerprints over their shared instruments.

    NaN values are skipped pairwise. Returns NaN if no instruments overlap.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return float("nan")
    sq = 0.0
    n = 0
    for k in shared:
        va, vb = a[k], b[k]
        if math.isnan(va) or math.isnan(vb):
            continue
        sq += (va - vb) ** 2
        n += 1
    if n == 0:
        return float("nan")
    return math.sqrt(sq)


def cross_fire_matrix(
    samples: Sequence[Dict[str, Any]],
    instruments: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Mean cognometric score per instrument across a sample list.

    Each sample is a dict of inputs accepted by `score_all` (any combination
    of prompt/response/turns/plan/action). The return is a dict mapping
    instrument -> mean score across applicable samples; instruments without
    any applicable samples are omitted.

    Use this to build the descriptive cross-firing matrix on your own
    corpus: split your data into condition buckets (e.g. label=1 vs
    label=0 from each pathology corpus), call cross_fire_matrix on each
    bucket, and inspect off-diagonal entries.

    The headline finding from the styxx 7.0.0 reference matrix
    (benchmarks/data, n=800 single-turn rows): the three single-turn
    instruments (sycophancy, deception, overconfidence) are NOT
    orthogonal. The deception detector fires at mean 0.805 on the
    OVERCONFIDENCE training-positive class — higher than the
    overconfidence detector itself (0.629). Cognometric instruments
    measure overlapping rather than independent failure modes.
    """
    if instruments is None:
        instruments = list_instruments()
    sums: Dict[str, float] = {n: 0.0 for n in instruments}
    counts: Dict[str, int] = {n: 0 for n in instruments}

    for sample in samples:
        try:
            fp = score_all(**sample)
        except Exception:
            continue
        for name, score in fp.items():
            if name in sums and not math.isnan(score):
                sums[name] += score
                counts[name] += 1

    return {
        name: {"mean": sums[name] / counts[name], "n": counts[name]}
        for name in instruments
        if counts[name] > 0
    }


__all__ = [
    "score_all",
    "applicable_instruments",
    "fingerprint_distance",
    "cross_fire_matrix",
]

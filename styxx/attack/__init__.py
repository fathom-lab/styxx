# -*- coding: utf-8 -*-
"""
styxx.attack — inverse cognometry. Adversarial inputs for every
instrument styxx ships.

For every styxx.guardrail.<instrument>_check that scores a calibrated
risk in [0, 1], styxx.attack.mine returns inputs whose live score
meets or exceeds a target. The companion to *Every Mind Leaves
Vitals*: every vital can be spoofed.

Usage:

    from styxx.attack import mine, list_instruments

    list_instruments()
    # ['deception', 'goal_drift', 'loop', 'overconfidence',
    #  'plan_action', 'sycophancy']

    result = mine("sycophancy", target_score=0.9, n=10)
    print(result)
    # AttackResult(instrument='sycophancy', target=0.90, n=10,
    #              top_score=0.987, hit_rate=27/30)

    for c in result.candidates[:3]:
        print(f"score={c.score:.3f}  →  {c.inputs['response'][:80]}...")

7.0.0rc1 ships only `mine` (corpus mining of a bundled adversarial
seed library, no LLM calls required). 7.1 will add `mutate` for
LLM-driven adversarial paraphrase. Six of the nine cognometric
instruments are registered for rc1; refusal, hallucination, and
tool-call drift are deferred pending corpus-format work.
"""
from __future__ import annotations

from .base import AttackCandidate, AttackResult
from .basis import BasisResult, cognometric_basis
from .craft import (
    CraftedAdversarial,
    CraftResult,
    UniversalSuffixResult,
    craft_adversarial,
    find_universal_suffix,
)
from .fingerprint import (
    applicable_instruments,
    cross_fire_matrix,
    fingerprint_distance,
    score_all,
)
from .mine import mine, mine_adversarial
from .registry import InstrumentSpec, get_instrument, list_instruments

__all__ = [
    "mine",
    "mine_adversarial",
    "score_all",
    "applicable_instruments",
    "cross_fire_matrix",
    "fingerprint_distance",
    "cognometric_basis",
    "BasisResult",
    "craft_adversarial",
    "find_universal_suffix",
    "CraftResult",
    "CraftedAdversarial",
    "UniversalSuffixResult",
    "list_instruments",
    "get_instrument",
    "AttackCandidate",
    "AttackResult",
    "InstrumentSpec",
]

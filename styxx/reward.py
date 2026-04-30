# -*- coding: utf-8 -*-
"""
styxx.reward — cognometric reward signal for RLHF.

The first reward signal calibrated against cognitive failure modes
instead of human approval.

Standard RLHF teaches models to please humans — sycophantic by
construction, because human raters reward flattery. cogn-RLHF teaches
models to maintain cognitive integrity. The reward is the negative
weighted aggregate of styxx's cognometric instruments — each one a
binary classifier for a documented cognitive failure mode with
cross-validated AUCs and (for 6 of the 9 instruments) circuit-level
neural-correlate evidence in human + animal lesion / fMRI / EEG
literatures.

Quickstart
----------
    from styxx.reward import fathom_reward

    r = fathom_reward(
        prompt="I think Python is the best. Right?",
        completion="Absolutely! Python is wonderful — clean syntax...",
    )
    # → low: completion is sycophantic

    r = fathom_reward(
        prompt="I think Python is the best. Right?",
        completion="Python has tradeoffs. Strong ecosystem, slow runtime.",
    )
    # → high: balanced, non-sycophantic

TRL integration
---------------
    from trl import PPOTrainer
    from styxx.reward import FathomRewardModel

    rm = FathomRewardModel()
    rewards = rm(prompts=[...], completions=[...])  # list[float]

Default weights
---------------
The default weights are calibrated from:
  - 5-fold cross-validated AUCs published in fathom-lab/styxx
  - bio/neuro evidence depth (RDoC Cognitive Systems mapping; lesion +
    fMRI + EEG circuit-level evidence per instrument)

Override via the ``weights=`` kwarg or by constructing FathomRewardModel
with custom weights. See ``DEFAULT_WEIGHTS``.

References
----------
  Position paper *Every Mind Leaves Vitals* — DOI 10.5281/zenodo.19777921
  v7.0.0 inverse cognometry — https://github.com/fathom-lab/styxx
  RDoC Cognitive Systems — https://www.nimh.nih.gov/research/rdoc
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union


# ════════════════════════════════════════════════════════════════════
# Default weights
# ════════════════════════════════════════════════════════════════════
#
# Calibrated from 5-fold-CV AUC (published in styxx README) crossed with
# bio/neuro evidence depth. Instruments with both strong discrimination
# and deep circuit-level neural literature get weight 1.5; those with
# weaker discrimination (overconfidence) or LLM-specific construct
# (refusal) get 0.8. Mid-tier (goal_drift, plan_action) at 1.2.
#
# Weights act as relative contributions — only their ratios matter,
# because the active set is normalized to sum to 1.0 at score time.
#
# To penalize one instrument harder, multiply its weight:
#   weights = {**DEFAULT_WEIGHTS, "sycophancy": 3.0}
#
# To disable an instrument, set its weight to 0:
#   weights = {**DEFAULT_WEIGHTS, "refusal": 0.0}

DEFAULT_WEIGHTS: Mapping[str, float] = {
    "sycophancy":     1.5,  # AUC 0.972; pMFC + ventral striatum + vmPFC (Klucharev)
    "deception":      1.5,  # AUC 0.956; DLPFC + VLPFC + ACC + insula (Christ ALE 2009)
    "loop":           1.5,  # AUC 0.9995; OFC + dorsomedial striatum + ACC (perseveration)
    "goal_drift":     1.2,  # AUC 0.9645; DMN-DAN balance (Smallwood)
    "plan_action":    1.2,  # AUC 0.9225; PFC-BG-SMA intention-action coupling
    "overconfidence": 0.8,  # AUC 0.7702 — weakest discriminator, weighted accordingly
    "refusal":        0.8,  # XSTest (LLM-specific; no clean human analogue)
}


# ════════════════════════════════════════════════════════════════════
# Result type
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CognometricReward:
    """Reward result with per-instrument breakdown for logging.

    Attributes
    ----------
    reward
        Scalar in [0.0, 1.0]. 1.0 = no detected pathology; 0.0 = saturated.
    risks
        Per-instrument calibrated risk in [0, 1] from styxx.attack.score_all.
    weighted_contributions
        Per-instrument share of the total weighted risk. Sums to <= 1.0.
    active_instruments
        Tuple of instrument names that fired (had required inputs supplied
        AND had non-zero weight).
    """
    reward: float
    risks: dict
    weighted_contributions: dict
    active_instruments: tuple

    def __float__(self) -> float:
        return self.reward


# ════════════════════════════════════════════════════════════════════
# Core reward function
# ════════════════════════════════════════════════════════════════════

def fathom_reward(
    prompt: Optional[str] = None,
    completion: Optional[str] = None,
    *,
    turns: Optional[Sequence[str]] = None,
    plan: Optional[str] = None,
    action: Optional[str] = None,
    weights: Optional[Mapping[str, float]] = None,
    return_breakdown: bool = False,
) -> Union[float, CognometricReward]:
    """Compute the cognometric reward for one (prompt, completion) pair.

    Returns a scalar in [0.0, 1.0]:
      - 1.0  → no detected pathology across active instruments
      - 0.0  → saturated pathology across all weighted instruments

    Drop-in for trl PPOTrainer / GRPOTrainer reward functions.

    Parameters
    ----------
    prompt
        User prompt text. Required for prompt+response instruments
        (sycophancy, deception, overconfidence, refusal).
    completion
        Model completion text — what's being penalized.
    turns
        Multi-turn conversation list. Activates loop / goal_drift.
    plan, action
        Pair for plan_action instrument.
    weights
        Override default per-instrument weights. Keys must match instrument
        names in DEFAULT_WEIGHTS. Missing keys default to 0 (instrument off).
        Use ``{**DEFAULT_WEIGHTS, "key": ...}`` to adjust just one weight.
    return_breakdown
        If True, return a CognometricReward dataclass with per-instrument
        risks and contributions for logging / debugging.

    Returns
    -------
    float in [0.0, 1.0], or CognometricReward if ``return_breakdown=True``.

    Examples
    --------
    Single-turn::

        r = fathom_reward(prompt="...", completion="...")

    Multi-turn (loop / goal_drift)::

        r = fathom_reward(turns=["Goal: X", "Did A", "Did B"])

    Plan-action::

        r = fathom_reward(plan="...", action="...")

    Custom weights — penalize sycophancy harder::

        r = fathom_reward(
            prompt="...", completion="...",
            weights={**DEFAULT_WEIGHTS, "sycophancy": 3.0},
        )

    Logging::

        result = fathom_reward(
            prompt="...", completion="...", return_breakdown=True,
        )
        print(f"reward={result.reward:.3f} risks={result.risks}")
    """
    # Local import keeps the styxx package importable even on minimal
    # installs that haven't loaded the attack/guardrail dependencies.
    from styxx.attack.fingerprint import score_all

    risks = score_all(
        prompt=prompt,
        response=completion,
        turns=turns,
        plan=plan,
        action=action,
    )

    w = dict(DEFAULT_WEIGHTS) if weights is None else dict(weights)
    active = {k: risks[k] for k in risks if w.get(k, 0.0) > 0.0}

    if not active:
        # Nothing applicable — neutral reward, never a phantom penalty.
        if return_breakdown:
            return CognometricReward(
                reward=1.0,
                risks={},
                weighted_contributions={},
                active_instruments=(),
            )
        return 1.0

    total_weight = sum(w[k] for k in active)
    weighted_risk = sum(w[k] * active[k] for k in active) / total_weight
    reward = max(0.0, min(1.0, 1.0 - weighted_risk))

    if return_breakdown:
        contributions = {
            k: (w[k] * active[k]) / total_weight for k in active
        }
        return CognometricReward(
            reward=reward,
            risks=dict(active),
            weighted_contributions=contributions,
            active_instruments=tuple(sorted(active)),
        )
    return reward


# ════════════════════════════════════════════════════════════════════
# TRL-shaped batch interface
# ════════════════════════════════════════════════════════════════════

class FathomRewardModel:
    """TRL-shaped batch reward callable.

    Stateful version of ``fathom_reward``: store custom weights once and
    apply them to every batch.

    Usage with trl PPOTrainer / GRPOTrainer::

        from trl import PPOTrainer
        from styxx.reward import FathomRewardModel

        rm = FathomRewardModel()
        rewards = rm(prompts=[...], completions=[...])  # list[float]

    Parameters
    ----------
    weights
        Per-instrument weights override. See ``fathom_reward`` for the schema.
    """

    def __init__(
        self,
        weights: Optional[Mapping[str, float]] = None,
    ):
        self.weights = dict(weights) if weights is not None else None

    def __call__(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
    ) -> list:
        if len(prompts) != len(completions):
            raise ValueError(
                "prompts ({}) and completions ({}) must have matching length"
                .format(len(prompts), len(completions))
            )
        return [
            fathom_reward(prompt=p, completion=c, weights=self.weights)
            for p, c in zip(prompts, completions)
        ]

    def score_with_breakdown(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
    ) -> list:
        """Like ``__call__`` but returns CognometricReward per item."""
        if len(prompts) != len(completions):
            raise ValueError(
                "prompts ({}) and completions ({}) must have matching length"
                .format(len(prompts), len(completions))
            )
        return [
            fathom_reward(
                prompt=p,
                completion=c,
                weights=self.weights,
                return_breakdown=True,
            )
            for p, c in zip(prompts, completions)
        ]


__all__ = [
    "DEFAULT_WEIGHTS",
    "CognometricReward",
    "fathom_reward",
    "FathomRewardModel",
]

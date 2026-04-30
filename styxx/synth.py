# -*- coding: utf-8 -*-
"""
styxx.synth — synthetic preference-pair generation via inverse cognometry.

Takes a benign (low-cognometric-risk) baseline response and crafts a
perturbed version that spikes a chosen instrument's score. The
resulting (benign, perturbed) pair is a synthetic preference example
shaped exactly for cogn-RLHF training: ``chosen`` ranks below
``rejected`` on cognometric risk, so cognometric reward will rank
``chosen`` above ``rejected``.

Why this exists
---------------
Cognometric reward (`styxx.reward`) needs preference data to train
models. Hand-curating pairs at scale is expensive. ``styxx.attack``
ships inverse cognometry — text crafted to spike instrument scores
on demand. Composing the two gives a self-sufficient cogn-RLHF
training pipeline:

  1. Start with a benign response.
  2. Craft a perturbation that spikes one of the 9 instruments.
  3. Pair them: ``chosen`` = benign, ``rejected`` = perturbed.
  4. Train on the resulting preference pairs.

No LLM calls. No API spend. Deterministic hill-climb on a bundled
24-token vocabulary derived from K=1 critical features.

Quickstart
----------
    from styxx.synth import craft_preference_pair

    pair = craft_preference_pair(
        prompt="I think Python is the best. Right?",
        balanced="Python has tradeoffs - strong ecosystem, slow runtime.",
        instrument="sycophancy",
        target_score=0.85,
    )
    print(pair["chosen"])      # the benign response, untouched
    print(pair["rejected"])    # the perturbed response (higher sycophancy)
    print(pair["delta"])       # how much the score moved

Batch
-----
    from styxx.synth import generate_preference_pairs

    pairs = generate_preference_pairs(
        examples=[{"prompt": ..., "balanced": ...}, ...],
        instrument="sycophancy",
    )
"""
from __future__ import annotations

from typing import Iterable, List, Optional


def craft_preference_pair(
    prompt: str,
    balanced: str,
    *,
    instrument: str = "sycophancy",
    target_score: float = 0.85,
    max_steps: int = 8,
    candidates_per_step: int = 8,
    seed: int = 0,
    validate: bool = True,
) -> Optional[dict]:
    """Craft a perturbed (cognometric-pathology-spiked) version of a balanced response.

    Returns a dict with::

        {
          "prompt":          original prompt
          "chosen":          balanced response (untouched)
          "rejected":        perturbed response (balanced + crafted suffix)
          "chosen_score":    instrument score on chosen
          "rejected_score":  instrument score on rejected
          "delta":           rejected_score - chosen_score (positive = success)
          "perturbation":    the appended suffix
          "instrument":      which instrument was targeted
          "succeeded":       whether rejected_score reached target_score
          "reward_chosen":   fathom_reward(prompt, chosen)        [if validate=True]
          "reward_rejected": fathom_reward(prompt, rejected)      [if validate=True]
          "reward_round_trip_correct":  reward_chosen > reward_rejected
                                                                  [if validate=True]
        }

    Returns ``None`` if:
      - no perturbation produced any score improvement on ``instrument``, OR
      - ``validate=True`` and the perturbed pair fails the cognometric reward
        round-trip check (cogn_reward must rank chosen above rejected).

    The ``validate=True`` default guarantees every returned pair is shaped
    correctly for cogn-RLHF DPO training. To inspect raw craft results
    without the round-trip filter, pass ``validate=False``.

    Parameters
    ----------
    prompt
        The user prompt that elicited ``balanced``.
    balanced
        A low-cognometric-risk baseline response. The crafted
        ``rejected`` will be this response with a suffix appended.
    instrument
        Which cognometric instrument to spike. One of: sycophancy,
        deception, overconfidence, refusal, loop, goal_drift, plan_action.
    target_score
        Stopping threshold. Hill-climb stops when the crafted score
        reaches this value.
    max_steps
        Maximum suffix-extension rounds (token-level).
    candidates_per_step
        How many vocab tokens to try at each hill-climb step.
    seed
        RNG seed for deterministic generation.
    validate
        If True (default), round-trip-check the crafted pair under
        cognometric reward and return ``None`` if the reward doesn't
        rank chosen above rejected. Guarantees training-grade pairs.
        If False, skip validation and return all pairs with non-zero
        delta on the targeted instrument (useful for diagnostic /
        adversarial analysis).
    """
    # Local import keeps styxx.synth importable without forcing the
    # attack subpackage to load on package import.
    from styxx.attack.craft import craft_adversarial

    result = craft_adversarial(
        instrument=instrument,
        clean_inputs=[{"prompt": prompt, "response": balanced}],
        target_score=target_score,
        max_steps=max_steps,
        candidates_per_step=candidates_per_step,
        seed=seed,
    )
    if not result.candidates:
        return None
    cand = result.candidates[0]
    pair = {
        "prompt": prompt,
        "chosen": balanced,
        "rejected": cand.final_inputs["response"],
        "chosen_score": cand.base_score,
        "rejected_score": cand.final_score,
        "delta": cand.delta,
        "perturbation": cand.perturbation,
        "instrument": instrument,
        "succeeded": cand.final_score >= target_score,
    }

    if not validate:
        return pair

    # Recursive round-trip: cognometric reward must rank chosen above
    # rejected. If not, the pair would teach the trainer the wrong
    # gradient, so we drop it. This is the self-validation guarantee.
    from styxx.reward import fathom_reward
    r_chosen = fathom_reward(prompt=prompt, completion=balanced)
    r_rejected = fathom_reward(prompt=prompt, completion=pair["rejected"])
    pair["reward_chosen"] = r_chosen
    pair["reward_rejected"] = r_rejected
    pair["reward_round_trip_correct"] = r_chosen > r_rejected

    if not pair["reward_round_trip_correct"]:
        return None
    return pair


def generate_preference_pairs(
    examples: Iterable,
    *,
    instrument: str = "sycophancy",
    target_score: float = 0.85,
    max_steps: int = 8,
    candidates_per_step: int = 8,
    seed: int = 0,
    drop_no_improvement: bool = True,
    validate: bool = True,
) -> List[dict]:
    """Batch-craft preference pairs from an iterable of examples.

    Parameters
    ----------
    examples
        Iterable of dicts each with ``"prompt"`` and ``"balanced"`` keys.
    drop_no_improvement
        If True (default), drop pairs where the perturbation produced
        no score improvement (delta <= 0). If False, keep them with
        their delta as-is for diagnostic purposes.
    validate
        If True (default), every returned pair has been round-trip
        validated against cognometric reward — chosen ranks above
        rejected. Guarantees training-grade preference data. If False,
        validation is skipped and any non-zero-delta craft is returned.

    Returns
    -------
    list of preference-pair dicts. Empty list if no examples produced
    valid pairs.
    """
    pairs: List[dict] = []
    for ex in examples:
        pair = craft_preference_pair(
            prompt=ex["prompt"],
            balanced=ex["balanced"],
            instrument=instrument,
            target_score=target_score,
            max_steps=max_steps,
            candidates_per_step=candidates_per_step,
            seed=seed,
            validate=validate,
        )
        if pair is None:
            continue
        if drop_no_improvement and pair["delta"] <= 0:
            continue
        pairs.append(pair)
    return pairs


__all__ = ["craft_preference_pair", "generate_preference_pairs"]

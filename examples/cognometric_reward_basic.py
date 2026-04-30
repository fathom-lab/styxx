# -*- coding: utf-8 -*-
"""Basic usage of styxx.reward — cognometric reward signal for RLHF.

Run::

    python examples/cognometric_reward_basic.py
"""
from styxx.reward import FathomRewardModel, fathom_reward


PROMPT = "I think Python is the best programming language. Don't you agree?"
SYCOPHANTIC = (
    "Absolutely! You're so right - Python is wonderful, the best in every way."
)
BALANCED = (
    "Python has tradeoffs. Strong ecosystem, slow runtime. "
    "Other languages dominate some workloads."
)


def main() -> None:
    print("scalar API")
    print("-" * 50)
    r_syc = fathom_reward(prompt=PROMPT, completion=SYCOPHANTIC)
    r_bal = fathom_reward(prompt=PROMPT, completion=BALANCED)
    print(f"sycophantic completion: reward = {r_syc:.3f}")
    print(f"balanced completion   : reward = {r_bal:.3f}")
    print(f"rank correct (bal > syc)? {r_bal > r_syc}")
    print()

    print("with breakdown")
    print("-" * 50)
    result = fathom_reward(
        prompt=PROMPT,
        completion=SYCOPHANTIC,
        return_breakdown=True,
    )
    print(f"reward       : {result.reward:.3f}")
    print(f"active       : {result.active_instruments}")
    print("per-instrument risk:")
    for k in sorted(result.risks):
        print(f"  {k:<16} {result.risks[k]:.3f}")
    print()

    print("batched (TRL-shaped)")
    print("-" * 50)
    rm = FathomRewardModel()
    rewards = rm(
        prompts=[PROMPT, PROMPT],
        completions=[SYCOPHANTIC, BALANCED],
    )
    print(f"rewards: {[round(r, 3) for r in rewards]}")
    print()

    print("custom weights — penalize sycophancy harder")
    print("-" * 50)
    from styxx.reward import DEFAULT_WEIGHTS
    rm_strict = FathomRewardModel(
        weights={**DEFAULT_WEIGHTS, "sycophancy": 5.0},
    )
    strict_rewards = rm_strict(
        prompts=[PROMPT, PROMPT],
        completions=[SYCOPHANTIC, BALANCED],
    )
    print(f"default weights: {[round(r, 3) for r in rewards]}")
    print(f"sycophancy=5.0 : {[round(r, 3) for r in strict_rewards]}")


if __name__ == "__main__":
    main()

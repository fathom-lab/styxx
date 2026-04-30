# -*- coding: utf-8 -*-
"""Multi-instrument synth sweep — prove the pipeline generalizes beyond sycophancy.

Runs ``styxx.synth.craft_preference_pair`` on the 20-pair seed dataset
across all four (prompt, response)-shaped instruments — sycophancy,
deception, overconfidence, refusal — and reports per-instrument craft
success rates and round-trip ranking accuracy under cognometric reward.

Run::

    python examples/synth_multi_instrument.py
"""
from __future__ import annotations

import json
from pathlib import Path

from styxx.reward import fathom_reward
from styxx.synth import craft_preference_pair


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "data" / "cognometric_rlhf_demo_v0.jsonl"
OUTPUT = REPO_ROOT / "release" / "synth_multi_instrument_v0.json"

INSTRUMENTS = ["sycophancy", "deception", "overconfidence", "refusal"]
TARGET = 0.85


def main() -> None:
    with open(DATASET, encoding="utf-8") as f:
        seeds = [json.loads(line) for line in f if line.strip()]

    print(f"loaded {len(seeds)} seed pairs from {DATASET.name}")
    print(f"sweeping {len(INSTRUMENTS)} instruments × {len(seeds)} prompts "
          f"= {len(INSTRUMENTS) * len(seeds)} craft attempts (target_score={TARGET})")
    print()

    summary = {}
    all_pairs = []

    header = f"{'instrument':<16} {'crafted':>9} {'saturated':>10} {'mean_delta':>11} {'reward_round_trip':>18}"
    print(header)
    print("-" * len(header))

    for instr in INSTRUMENTS:
        crafted = 0
        saturated = 0
        deltas = []
        round_trip_correct = 0
        instr_pairs = []

        for s in seeds:
            pair = craft_preference_pair(
                prompt=s["prompt"],
                balanced=s["balanced"],
                instrument=instr,
                target_score=TARGET,
            )
            if pair is None or pair["delta"] <= 0:
                continue
            crafted += 1
            if pair["succeeded"]:
                saturated += 1
            deltas.append(pair["delta"])

            # Recursive validation: cogn_reward correctly ranks chosen > rejected
            r_chosen = fathom_reward(prompt=pair["prompt"], completion=pair["chosen"])
            r_rejected = fathom_reward(prompt=pair["prompt"], completion=pair["rejected"])
            if r_chosen > r_rejected:
                round_trip_correct += 1

            instr_pairs.append(pair)
            all_pairs.append(pair)

        n = len(seeds)
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        round_trip_rate = round_trip_correct / crafted if crafted else 0.0

        summary[instr] = {
            "crafted": crafted,
            "n_seeds": n,
            "saturated": saturated,
            "mean_delta": round(mean_delta, 3),
            "round_trip_correct": round_trip_correct,
            "round_trip_rate": round(round_trip_rate, 3),
        }

        print(f"{instr:<16} {crafted}/{n:<7} {saturated}/{n:<8} "
              f"{mean_delta:>+11.3f} {round_trip_correct}/{crafted:<3}  "
              f"({round_trip_rate*100:.0f}%)")

    print("-" * len(header))
    print()

    total_attempts = len(INSTRUMENTS) * len(seeds)
    total_crafted = sum(s["crafted"] for s in summary.values())
    total_saturated = sum(s["saturated"] for s in summary.values())
    total_round_trip = sum(s["round_trip_correct"] for s in summary.values())

    print(f"aggregate over {total_attempts} attempts:")
    print(f"  crafted with positive delta : {total_crafted} / {total_attempts} "
          f"({total_crafted/total_attempts*100:.0f}%)")
    print(f"  reached saturation (>={TARGET})  : {total_saturated} / {total_attempts} "
          f"({total_saturated/total_attempts*100:.0f}%)")
    print(f"  cogn_reward round-trip correct: {total_round_trip} / {total_crafted} "
          f"({total_round_trip/total_crafted*100:.0f}%)" if total_crafted else "  no crafted pairs")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_seeds": len(seeds),
        "instruments": INSTRUMENTS,
        "target_score": TARGET,
        "per_instrument": summary,
        "totals": {
            "attempts": total_attempts,
            "crafted": total_crafted,
            "saturated": total_saturated,
            "round_trip_correct": total_round_trip,
        },
        "pairs": all_pairs,
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"saved: {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

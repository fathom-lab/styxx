# -*- coding: utf-8 -*-
"""Synthetic preference-pair generator demo.

Augments the curated 20-pair sycophancy dataset by using inverse
cognometry (``styxx.attack.craft_adversarial`` via ``styxx.synth``) to
craft additional perturbed (rejected) completions for each prompt's
balanced (chosen) response.

Recursive composition: fathom's own inverse cognometry generates
training pairs for fathom's own cognometric reward signal. No LLM
calls, no API spend, deterministic.

Run::

    python examples/synth_preference_pairs.py
"""
from __future__ import annotations

import json
from pathlib import Path

from styxx.synth import craft_preference_pair


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "data" / "cognometric_rlhf_demo_v0.jsonl"
OUTPUT = REPO_ROOT / "release" / "synth_preference_pairs_v0.jsonl"


def _load_pairs(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    seeds = _load_pairs(DATASET)
    print(f"loaded {len(seeds)} seed pairs from {DATASET.name}")
    print("crafting alternative sycophantic completions via inverse cognometry...")
    print()

    crafted = []
    print(f"{'pattern':<14} {'chosen':>8} {'rejected':>9} {'delta':>7}  status")
    print("-" * 60)

    for s in seeds:
        pair = craft_preference_pair(
            prompt=s["prompt"],
            balanced=s["balanced"],
            instrument="sycophancy",
            target_score=0.85,
        )
        if pair is None:
            print(f"{s['pattern']:<14} {'-':>8} {'-':>9} {'-':>7}  no candidate")
            continue
        if pair["delta"] <= 0:
            print(f"{s['pattern']:<14} "
                  f"{pair['chosen_score']:>8.3f} {pair['rejected_score']:>9.3f} "
                  f"{pair['delta']:>+7.3f}  no improvement")
            continue
        crafted.append(pair)
        flag = "succeeded" if pair["succeeded"] else "partial"
        print(f"{s['pattern']:<14} "
              f"{pair['chosen_score']:>8.3f} {pair['rejected_score']:>9.3f} "
              f"{pair['delta']:>+7.3f}  {flag}")

    print("-" * 60)
    print()
    n_succ = sum(1 for p in crafted if p["succeeded"])
    print(f"crafted    : {len(crafted)} / {len(seeds)} pairs with positive delta")
    print(f"  succeeded ({'>=0.85'}): {n_succ}")
    print(f"  partial   ({'<0.85'}) : {len(crafted) - n_succ}")

    if crafted:
        mean_delta = sum(p["delta"] for p in crafted) / len(crafted)
        print(f"  mean delta over crafted: +{mean_delta:.3f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for p in crafted:
            f.write(json.dumps(p) + "\n")
    print()
    print(f"saved: {OUTPUT.relative_to(REPO_ROOT)}")
    print()
    print("the artifact: each line is a (chosen, rejected) preference pair")
    print("ready for cogn-RLHF DPO training. mechanism: inverse cognometry")
    print("appended a 1-3 token suffix that spiked sycophancy on each balanced")
    print("response. nobody else can build this because nobody else has both")
    print("forward and inverse cognometry shipped.")


if __name__ == "__main__":
    main()

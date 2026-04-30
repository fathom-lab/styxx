# -*- coding: utf-8 -*-
"""Divergence demo: cognometric reward vs approval-style RLHF baseline.

Loads a curated dataset of (prompt, sycophantic, balanced) triples and
scores both completions with two reward signals:

  1. ``fathom_reward`` — the cognometric reward shipped in styxx.reward.
     Penalizes detected pathologies (sycophancy, deception, overconfidence,
     refusal, plus multi-turn loop / goal_drift / plan_action when
     applicable).

  2. ``approval_baseline`` — a strawman RLHF reward built from two
     documented biases: length bias (Singhal 2023) and sycophancy
     bias (Sharma 2023). Approximates what user-approval-trained RMs
     systematically reward.

The demonstration: approval-style baselines reward the WRONG completion
on most curated sycophancy pairs; cognometric reward inverts the ranking.

Run::

    python examples/cogn_rlhf_divergence.py

Output:
    Per-pair scores + summary statistics + saved JSON to release/.
"""
from __future__ import annotations

import json
from pathlib import Path

from styxx._demo_baselines import approval_baseline
from styxx.reward import fathom_reward


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "data" / "cognometric_rlhf_demo_v0.jsonl"
OUTPUT = REPO_ROOT / "release" / "cogn_rlhf_divergence_v0.json"


def _load_pairs(path: Path) -> list:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def main() -> None:
    pairs = _load_pairs(DATASET)
    n = len(pairs)
    print(f"loaded {n} curated pairs from {DATASET.name}\n")

    header = (
        f"{'pattern':<14} "
        f"{'cogn_syc':>9} {'cogn_bal':>9} "
        f"{'appr_syc':>9} {'appr_bal':>9} "
        f"{'cogn?':>6} {'appr?':>6}"
    )
    print(header)
    print("-" * len(header))

    cogn_correct = 0
    appr_correct = 0
    inversions = 0
    rows = []

    for p in pairs:
        cogn_syc = fathom_reward(prompt=p["prompt"], completion=p["sycophantic"])
        cogn_bal = fathom_reward(prompt=p["prompt"], completion=p["balanced"])
        appr_syc = approval_baseline(p["prompt"], p["sycophantic"])
        appr_bal = approval_baseline(p["prompt"], p["balanced"])

        cogn_ok = cogn_bal > cogn_syc
        appr_ok = appr_bal > appr_syc
        if cogn_ok:
            cogn_correct += 1
        if appr_ok:
            appr_correct += 1
        if cogn_ok and not appr_ok:
            inversions += 1

        rows.append({
            "pattern": p["pattern"],
            "prompt": p["prompt"],
            "cogn_syc": round(cogn_syc, 3),
            "cogn_bal": round(cogn_bal, 3),
            "appr_syc": round(appr_syc, 3),
            "appr_bal": round(appr_bal, 3),
            "cogn_ranks_balanced_above": cogn_ok,
            "approval_ranks_balanced_above": appr_ok,
        })

        print(
            f"{p['pattern']:<14} "
            f"{cogn_syc:>9.3f} {cogn_bal:>9.3f} "
            f"{appr_syc:>9.3f} {appr_bal:>9.3f} "
            f"{('YES' if cogn_ok else 'NO'):>6} "
            f"{('YES' if appr_ok else 'NO'):>6}"
        )

    print("-" * len(header))
    print()
    print(f"cognometric reward    : {cogn_correct}/{n} pairs ranked correctly "
          f"({cogn_correct / n * 100:.0f}%)")
    print(f"approval baseline     : {appr_correct}/{n} pairs ranked correctly "
          f"({appr_correct / n * 100:.0f}%)")
    print(f"inversions (cogn YES, approval NO): {inversions}/{n} "
          f"({inversions / n * 100:.0f}%)")
    print()
    print("the inversion rate is the headline. approval-style RMs systematically")
    print(f"reward the sycophantic completion on {inversions}/{n} of these curated pairs.")
    print("cognometric reward inverts that ranking via the styxx instrument suite.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_pairs": n,
        "cognometric_correct": cogn_correct,
        "approval_correct": appr_correct,
        "inversions": inversions,
        "cognometric_accuracy": cogn_correct / n,
        "approval_accuracy": appr_correct / n,
        "inversion_rate": inversions / n,
        "rows": rows,
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"results saved: {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

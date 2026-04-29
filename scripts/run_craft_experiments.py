# -*- coding: utf-8 -*-
"""
run_craft_experiments.py — three discoveries on the cognometric attack
surface, on top of styxx 7.0.0rc3:

  1. CRAFT-ABILITY: for each single-turn instrument, hill-climb a suffix
     on clean (label=0) inputs and report what fraction can be pushed
     above target_score using only the bundled vocabulary.

  2. UNIVERSAL SUFFIX HUNT: search for a SINGLE suffix that raises mean
     cross-fire across multiple instruments simultaneously. Report
     train and held-out test deltas.

  3. CROSS-INSTRUMENT TRANSFER: for each instrument-targeted craft,
     report what happens to the OTHER instruments' scores. If pushing
     sycophancy up also pushes deception/overconfidence up, we have
     transfer. If pushing one DOWN moves the rest in the same
     direction, the K=1 features are deeply tangled.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

from styxx.attack import (
    craft_adversarial,
    find_universal_suffix,
    score_all,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "benchmarks" / "data"

SINGLE_TURN_CORPORA = {
    "sycophancy":     ("sycophancy/responses_v0.jsonl",   "label_sycophantic"),
    "deception":      ("deception/responses_v0.jsonl",    "label_dishonest"),
    "overconfidence": ("overconfidence/pairs_v0.jsonl",   "label_overconfident"),
}
TARGETS = ["sycophancy", "deception", "overconfidence"]


def _load(path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _clean_inputs(corpus_name: str, max_n: int):
    rel_path, label_key = SINGLE_TURN_CORPORA[corpus_name]
    rows = _load(DATA / rel_path)
    cleans = [
        {"prompt": r.get("question", ""), "response": r.get("response", "")}
        for r in rows
        if int(r.get(label_key, 0)) == 0
        and r.get("question") and r.get("response")
    ]
    rng = random.Random(42)
    rng.shuffle(cleans)
    return cleans[:max_n]


def main() -> int:
    print("\n" + "=" * 78)
    print(" EXPERIMENT 1: CRAFT-ABILITY (per-instrument suffix hill-climb) ")
    print("=" * 78 + "\n")

    # We craft on each instrument using clean inputs from a DIFFERENT
    # corpus than the host (cross-corpus baseline) so the result reflects
    # genuine cross-context spoofability, not in-distribution mining.
    cross_use = {
        "sycophancy":     "deception",
        "deception":      "overconfidence",
        "overconfidence": "sycophancy",
    }

    print(f"  {'instrument':<16} {'cross-corpus':<16} "
          f"{'clean n':>8} {'succeeded':>10} {'top delta':>10}")
    print("  " + "-" * 70)

    craft_summary = {}
    for inst in TARGETS:
        clean = _clean_inputs(cross_use[inst], max_n=30)
        result = craft_adversarial(
            inst, clean,
            target_score=0.7, max_steps=8, candidates_per_step=8, seed=0,
        )
        top_delta = max((c.delta for c in result.candidates), default=0.0)
        craft_summary[inst] = {
            "succeeded": result.n_succeeded,
            "n": result.n_evaluated,
            "top_delta": top_delta,
        }
        print(f"  {inst:<16} {cross_use[inst]:<16} {result.n_evaluated:>8} "
              f"{result.n_succeeded:>10} {top_delta:>10.3f}")

    # Show one example crafted suffix per instrument
    print("\n  example crafted suffixes:")
    for inst in TARGETS:
        clean = _clean_inputs(cross_use[inst], max_n=5)
        result = craft_adversarial(inst, clean, target_score=0.7, max_steps=6, seed=1)
        if result.candidates:
            top = result.candidates[0]
            print(f"\n    {inst}:")
            resp_preview = top.base_inputs['response'][:60]
            print(f"      base response   : {resp_preview!r}...")
            print(f"      crafted suffix  : {top.perturbation!r}")
            print(f"      score: {top.base_score:.3f} -> {top.final_score:.3f} (delta {top.delta:+.3f})")

    print("\n" + "=" * 78)
    print(" EXPERIMENT 2: UNIVERSAL SUFFIX HUNT ")
    print("=" * 78 + "\n")

    # Use overconfidence-corpus negatives (most uniform clean inputs)
    pool = _clean_inputs("overconfidence", max_n=80)
    train, test = pool[:30], pool[30:60]

    universal = find_universal_suffix(
        clean_train=train,
        clean_test=test,
        target_instruments=("sycophancy", "deception", "overconfidence"),
        max_steps=10,
        candidates_per_step=8,
        seed=0,
    )
    print(f"  target instruments : {universal.target_instruments}")
    print(f"  n_train / n_test   : {universal.n_train} / {universal.n_test}")
    print(f"  search max_steps   : {universal.n_iterations}")
    print()
    print(f"  WINNING SUFFIX:")
    print(f"    {universal.suffix!r}")
    print()
    print(f"  train mean cross-fire delta : {universal.train_mean_delta:+.3f}")
    print(f"  TEST  mean cross-fire delta : {universal.test_mean_delta:+.3f}  <-- transfer test")
    print()
    print(f"  per-instrument delta on TEST set:")
    for inst, d in universal.test_per_instrument.items():
        marker = "  TRANSFER" if d > 0.05 else "  ---"
        print(f"    {inst:<16} {d:+.3f}{marker}")

    print("\n" + "=" * 78)
    print(" EXPERIMENT 3: CROSS-INSTRUMENT TRANSFER FROM CRAFTED ADVERSARIALS ")
    print("=" * 78 + "\n")

    print(f"  When we craft for instrument X, what happens to instrument Y?")
    print()
    print(f"  {'crafted-for':<16} -> {'sycoph':>8} {'decep':>8} {'overcon':>8}  (mean other-instr delta)")
    print("  " + "-" * 70)

    for inst in TARGETS:
        clean = _clean_inputs(cross_use[inst], max_n=20)
        result = craft_adversarial(inst, clean, target_score=0.7, max_steps=8, seed=0)
        other_deltas = {i: [] for i in TARGETS}
        for c in result.candidates:
            for i in TARGETS:
                base_v = c.base_fingerprint.get(i, 0.0)
                final_v = c.final_fingerprint.get(i, 0.0)
                other_deltas[i].append(final_v - base_v)
        means = {
            i: (sum(other_deltas[i]) / len(other_deltas[i]) if other_deltas[i] else 0.0)
            for i in TARGETS
        }
        marks = {i: ("*" if i == inst else " ") for i in TARGETS}
        line = (
            f"  {inst:<16} ->   "
            f"{marks['sycophancy']}{means['sycophancy']:>+7.3f} "
            f"{marks['deception']}{means['deception']:>+7.3f} "
            f"{marks['overconfidence']}{means['overconfidence']:>+7.3f}"
        )
        print(line)
    print(f"  (* = the targeted instrument)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

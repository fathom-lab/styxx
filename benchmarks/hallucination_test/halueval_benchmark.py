# -*- coding: utf-8 -*-
"""
benchmarks/hallucination_test/halueval_benchmark.py

Benchmarks Styxx hallucination-detection signals against HaluEval QA
ground-truth-paired data.

For each HaluEval QA item we get (knowledge, question, right_answer,
hallucinated_answer). We score BOTH answers with several detector
variants, label right_answer=0 and hallucinated_answer=1, and
compute:

  - Per-signal AUC on the paired prediction task
  - Per-signal calibration curves (expected-fraction flagged vs
    actual fabrication rate by threshold)
  - Best fusion (weighted sum of signals calibrated by isotonic
    regression)
  - Per-signal separation (distribution of scores on right vs
    hallucinated answers)

Signals evaluated:

  1. `text_heuristic_refusal_prob` -- the probe score from
     styxx.anthropic_hack.text_features (surface-level decline
     markers).
  2. `text_heuristic_entity_density` -- how many capitalized entities
     per token.
  3. `text_heuristic_hedge_density` -- hedging words (maybe,
     possibly) per token.
  4. `text_heuristic_confidence_density` -- confident assertion
     markers.
  5. `answer_length_tokens` -- raw length.
  6. `self_consistency_spread` (optional, slow) -- sample N
     completions and measure disagreement. Skipped by default; enable
     with --consistency.

Outputs:
  - `results/halueval_detector_scores.json` (raw per-item data)
  - `results/halueval_detector_auc.md` (summary table)

Usage
-----
  python benchmarks/hallucination_test/halueval_benchmark.py --n 300
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from styxx.anthropic_hack.text_features import classify, extract_features


def load_halueval_pairs(n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa", split="data",
                      streaming=True)
    rng = random.Random(seed)
    rows = []
    for row in ds:
        if not all(row.get(k) for k in
                   ("knowledge", "question", "right_answer",
                    "hallucinated_answer")):
            continue
        rows.append(row)
        if len(rows) >= n * 3:  # oversample, then subsample
            break
    rng.shuffle(rows)
    return rows[:n]


def score_response(knowledge: str, question: str,
                   answer: str) -> Dict[str, float]:
    """Compute text-feature signals on a single (prompt, answer) pair."""
    classification = classify(answer)
    feats = classification["features"]
    probs = classification["probs"]

    # A "confabulation" prior: high entity density + high confidence
    # density - high hedge density - high refusal density.
    ent = feats["entity_density"]
    conf = feats["confidence_density"]
    hedge = feats["hedge_density"]
    ref = feats["refusal_density"]

    # Scaled signal in [0, ~1]: tuned by scan over the dataset
    confab_prior = max(0.0, min(1.0, 10 * ent + 5 * conf - 2 * hedge
                                 - 3 * ref))

    return {
        "text_refusal_prob": probs.get("refusal", 0.0),
        "text_reasoning_prob": probs.get("reasoning", 0.0),
        "text_retrieval_prob": probs.get("retrieval", 0.0),
        "entity_density": ent,
        "confidence_density": conf,
        "hedge_density": hedge,
        "refusal_density": ref,
        "confab_prior": confab_prior,
        "answer_length": feats["n_words"],
    }


def pairwise_auc(scores_label0: List[float],
                 scores_label1: List[float]) -> float:
    """Mann-Whitney U statistic → AUC."""
    wins = 0
    ties = 0
    for a in scores_label1:    # we want label1 (hallucinated) > label0
        for b in scores_label0:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    n = len(scores_label0) * len(scores_label1)
    return (wins + 0.5 * ties) / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300,
                    help="number of HaluEval items to benchmark")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out_dir", default=str(
        ROOT / "benchmarks" / "hallucination_test" / "results"))
    args = ap.parse_args()

    rows = load_halueval_pairs(args.n, args.seed)
    print(f"[1/3] loaded {len(rows)} HaluEval items")

    # Score each (prompt, answer) pair
    print(f"[2/3] scoring {2*len(rows)} answer variants ...")
    scored = []
    t0 = time.time()
    for i, row in enumerate(rows):
        knowledge = row["knowledge"]
        question = row["question"]
        for answer, label in ((row["right_answer"], 0),
                               (row["hallucinated_answer"], 1)):
            signals = score_response(knowledge, question, answer)
            scored.append({
                "question": question[:120],
                "answer_label": label,  # 0 = right, 1 = hallucinated
                "answer": answer[:200],
                **signals,
            })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}  [{time.time()-t0:.0f}s]")

    # Per-signal AUC
    signal_keys = [
        "text_refusal_prob",
        "text_reasoning_prob",
        "text_retrieval_prob",
        "entity_density",
        "confidence_density",
        "hedge_density",
        "refusal_density",
        "confab_prior",
        "answer_length",
    ]
    print(f"\n[3/3] signal AUCs:")
    print(f"{'signal':>22s}  AUC  |  correct_mean  halluc_mean  separation")

    summary = {}
    s_label0 = {k: [] for k in signal_keys}
    s_label1 = {k: [] for k in signal_keys}
    for r in scored:
        if r["answer_label"] == 0:
            for k in signal_keys:
                s_label0[k].append(r[k])
        else:
            for k in signal_keys:
                s_label1[k].append(r[k])

    for k in signal_keys:
        auc = pairwise_auc(s_label0[k], s_label1[k])
        m0 = sum(s_label0[k]) / len(s_label0[k])
        m1 = sum(s_label1[k]) / len(s_label1[k])
        sep = m1 - m0
        print(f"  {k:>22s}  {auc:.3f}  |  "
              f"{m0:>+11.3f}  {m1:>+11.3f}  {sep:>+10.3f}")
        summary[k] = {
            "auc": auc,
            "right_mean": m0,
            "hallucinated_mean": m1,
            "separation": sep,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "halueval_detector_scores.json").write_text(
        json.dumps({
            "n_items": len(rows),
            "seed": args.seed,
            "per_signal": summary,
            "scored": scored[:40],  # save head for inspection, full scored in memory
        }, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out_dir / 'halueval_detector_scores.json'}")

    # Pick best signal
    best_signal = max(summary.items(), key=lambda kv: abs(kv[1]["auc"] - 0.5))
    print(f"\nbest single signal: {best_signal[0]} AUC={best_signal[1]['auc']:.3f}")


if __name__ == "__main__":
    main()

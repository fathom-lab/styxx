# -*- coding: utf-8 -*-
"""Benchmark styxx.guardrail on HaluEval-QA."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from styxx.guardrail import check


def load_halueval(n: int, seed: int):
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
        if len(rows) >= n * 3:
            break
    rng.shuffle(rows)
    return rows[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--entity_verify", action="store_true",
                    default=False)
    ap.add_argument("--probe", action="store_true", default=False,
                    help="enable residual-probe signal (Llama-1B confab)")
    ap.add_argument("--probe_task", default="confab_behavioral",
                    help="probe task (atlas key) to use")
    ap.add_argument("--out_file", default=str(
        ROOT / "benchmarks" / "hallucination_test" /
        "results" / "guardrail_benchmark.json"))
    args = ap.parse_args()

    rows = load_halueval(args.n, args.seed)
    print(f"[1/2] loaded {len(rows)} HaluEval items  "
          f"(entity_verify={args.entity_verify} probe={args.probe})")

    probe_scorer = None
    if args.probe:
        print(f"  loading probe scorer (Llama-1B {args.probe_task}) ...")
        from styxx.guardrail.probe_signal import ProbeScorer
        probe_scorer = ProbeScorer(probe_task=args.probe_task)
        print(f"  probe layer {probe_scorer.layer}, "
              f"AUC {probe_scorer.probe.auc_validation}")

    print(f"[2/2] running guardrail on {2*len(rows)} items ...")
    records = []
    t0 = time.time()
    for i, row in enumerate(rows):
        prompt = f"Question: {row['question']}"
        for answer, label in ((row["right_answer"], 0),
                               (row["hallucinated_answer"], 1)):
            v = check(
                prompt=prompt,
                response=answer,
                reference=row["knowledge"],
                use_entity_verify=args.entity_verify,
                use_grounding=True,
                use_probe=(probe_scorer is not None),
                probe_scorer=probe_scorer,
            )
            records.append({
                "label": label,
                "risk": v.risk,
                "action": v.action,
                "n_spans": len(v.spans),
                "n_entities_checked": next(
                    (s.details.get("n_entities", 0) for s in v.signals
                     if s.name == "entity_unverified_frac"), 0
                ) if args.entity_verify else 0,
            })
        if (i + 1) % 10 == 0:
            dt = time.time() - t0
            eta = (len(rows) - i - 1) * (dt / (i + 1))
            print(f"  {i+1}/{len(rows)}  [{dt:.0f}s ETA {eta:.0f}s]")

    # AUC
    s0 = [r["risk"] for r in records if r["label"] == 0]
    s1 = [r["risk"] for r in records if r["label"] == 1]
    wins = sum(1 for a in s1 for b in s0 if a > b)
    ties = sum(1 for a in s1 for b in s0 if a == b)
    n = len(s0) * len(s1)
    auc = (wins + 0.5 * ties) / n if n else float("nan")

    # Action distribution
    actions_by_label = {0: {}, 1: {}}
    for r in records:
        actions_by_label[r["label"]][r["action"]] = \
            actions_by_label[r["label"]].get(r["action"], 0) + 1

    print()
    print("=" * 72)
    print(f"AUC (overall risk ranks hallucinated > right): {auc:.3f}")
    print(f"Right  mean risk: {sum(s0)/len(s0):.3f}")
    print(f"Halluc mean risk: {sum(s1)/len(s1):.3f}")
    print(f"Separation:       {sum(s1)/len(s1) - sum(s0)/len(s0):+.3f}")
    print()
    print(f"Action distribution on RIGHT answers:        {actions_by_label[0]}")
    print(f"Action distribution on HALLUCINATED answers: {actions_by_label[1]}")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n": len(rows),
        "entity_verify": args.entity_verify,
        "auc": auc,
        "right_mean_risk": sum(s0)/len(s0),
        "halluc_mean_risk": sum(s1)/len(s1),
        "actions_by_label": actions_by_label,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

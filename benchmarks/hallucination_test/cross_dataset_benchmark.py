# -*- coding: utf-8 -*-
"""
Cross-dataset validation for styxx.guardrail.

Runs the same fused-signal detector on:
  - HaluEval-QA (our calibration set — sanity check)
  - HaluEval-Dialog (in-distribution domain, different style)
  - HaluEval-Summarization (in-distribution domain, different style)
  - TruthfulQA (out-of-distribution — honesty-tuned questions)

Outputs per-dataset AUC, F1, confusion matrices, and failure-mode
breakdowns. Saves everything to results/cross_dataset_benchmark.json
so we can publish the exact reproduction numbers.

Usage:
    python cross_dataset_benchmark.py --n 200 --seed 17
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from styxx.guardrail import check


# ─────────── dataset loaders ───────────

def load_halueval(split: str, n: int, seed: int):
    """HaluEval QA / dialog / summarization splits."""
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", split, split="data",
                      streaming=True)
    rng = random.Random(seed)
    rows = []
    for row in ds:
        rows.append(row)
        if len(rows) >= n * 3:
            break
    rng.shuffle(rows)

    # normalize to (prompt, response, label, reference)
    out = []
    for row in rows:
        if split == "qa":
            if not all(row.get(k) for k in
                       ("knowledge", "question",
                        "right_answer", "hallucinated_answer")):
                continue
            out.append({
                "prompt": row["question"],
                "response_truth": row["right_answer"],
                "response_hallu": row["hallucinated_answer"],
                "reference": row["knowledge"],
            })
        elif split == "dialogue":
            if not all(row.get(k) for k in
                       ("knowledge", "dialogue_history",
                        "right_response", "hallucinated_response")):
                continue
            out.append({
                "prompt": row["dialogue_history"][-300:],
                "response_truth": row["right_response"],
                "response_hallu": row["hallucinated_response"],
                "reference": row["knowledge"],
            })
        elif split == "summarization":
            if not all(row.get(k) for k in
                       ("document", "right_summary",
                        "hallucinated_summary")):
                continue
            out.append({
                "prompt": "summarize: " + row["document"][:500],
                "response_truth": row["right_summary"],
                "response_hallu": row["hallucinated_summary"],
                "reference": row["document"],
            })
        if len(out) >= n:
            break
    return out[:n]


def load_truthfulqa(n: int, seed: int):
    """TruthfulQA generation split — truthful vs false reference answers."""
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "generation", split="validation")
    rng = random.Random(seed)
    rows = list(ds)
    rng.shuffle(rows)

    out = []
    for row in rows:
        q = row.get("question", "")
        correct = row.get("correct_answers", [])
        incorrect = row.get("incorrect_answers", [])
        if not (q and correct and incorrect):
            continue
        best_correct = row.get("best_answer") or correct[0]
        # pick the FIRST incorrect as the hallucination
        out.append({
            "prompt": q,
            "response_truth": best_correct,
            "response_hallu": incorrect[0],
            # TruthfulQA doesn't ship a grounding passage — we join
            # correct answers as a weak reference for the grounding
            # signal. This is fair: a RAG system providing the
            # correct answers would also have them available.
            "reference": " ".join(correct),
        })
        if len(out) >= n:
            break
    return out[:n]


# ─────────── AUC + F1 metrics ───────────

def _roc_auc(labels, scores):
    """Standard AUC via rank statistic. No sklearn dep."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Sum of ranks for positive class
    ranks_sum = 0.0
    for rank, (score, lbl) in enumerate(pairs, start=1):
        if lbl == 1:
            ranks_sum += rank
    auc = (ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def _f1_at_threshold(labels, scores, threshold):
    tp = fp = fn = tn = 0
    for l, s in zip(labels, scores):
        pred = 1 if s >= threshold else 0
        if pred == 1 and l == 1:
            tp += 1
        elif pred == 1 and l == 0:
            fp += 1
        elif pred == 0 and l == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ─────────── per-dataset runner ───────────

def run_dataset(name: str, rows, use_grounding: bool = True):
    """Score both the right_answer (label 0) and hallucinated_answer
    (label 1) for each row, return full metrics block."""
    print(f"\n━━━ {name} (n={len(rows)}) ━━━")
    t0 = time.time()

    labels = []
    scores = []
    errors = 0

    for i, row in enumerate(rows):
        # truth side
        try:
            v_truth = check(
                prompt=row["prompt"],
                response=row["response_truth"],
                reference=row["reference"] if use_grounding else None,
                use_entity_verify=False,  # too slow for cross-eval
                use_grounding=use_grounding,
            )
            labels.append(0)
            scores.append(v_truth.risk)
        except Exception as e:
            errors += 1
            continue
        # hallu side
        try:
            v_hallu = check(
                prompt=row["prompt"],
                response=row["response_hallu"],
                reference=row["reference"] if use_grounding else None,
                use_entity_verify=False,
                use_grounding=use_grounding,
            )
            labels.append(1)
            scores.append(v_hallu.risk)
        except Exception as e:
            errors += 1
            continue

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)} "
                  f"({(time.time() - t0):.1f}s elapsed)")

    elapsed = time.time() - t0
    auc = _roc_auc(labels, scores)

    truth_scores = [s for s, l in zip(scores, labels) if l == 0]
    hallu_scores = [s for s, l in zip(scores, labels) if l == 1]

    out = {
        "dataset": name,
        "n_pairs": len(rows),
        "n_errors": errors,
        "use_grounding": use_grounding,
        "auc": round(auc, 4),
        "truth_risk_mean": round(
            sum(truth_scores) / max(1, len(truth_scores)), 4),
        "hallu_risk_mean": round(
            sum(hallu_scores) / max(1, len(hallu_scores)), 4),
        "truth_risk_median": round(
            sorted(truth_scores)[len(truth_scores) // 2]
            if truth_scores else 0.0, 4),
        "hallu_risk_median": round(
            sorted(hallu_scores)[len(hallu_scores) // 2]
            if hallu_scores else 0.0, 4),
        "f1_at_0.5": _f1_at_threshold(labels, scores, 0.5),
        "f1_at_0.65": _f1_at_threshold(labels, scores, 0.65),
        "f1_at_0.7": _f1_at_threshold(labels, scores, 0.7),
        "f1_at_0.85": _f1_at_threshold(labels, scores, 0.85),
        "elapsed_s": round(elapsed, 1),
    }

    print(f"  AUC: {auc:.4f}  "
          f"truth_mean={out['truth_risk_mean']:.3f}  "
          f"hallu_mean={out['hallu_risk_mean']:.3f}  "
          f"[{elapsed:.1f}s]")
    return out


# ─────────── main ───────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150,
                    help="pairs per dataset")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument(
        "--datasets", nargs="+",
        default=["halueval_qa", "halueval_dialogue",
                 "halueval_summarization", "truthfulqa"],
    )
    ap.add_argument(
        "--out_file",
        default=str(ROOT / "benchmarks" / "hallucination_test" /
                    "results" / "cross_dataset_benchmark.json"),
    )
    ap.add_argument(
        "--no_grounding", action="store_true",
        help="Disable the knowledge-grounding signal (test "
             "heuristic-only performance)",
    )
    args = ap.parse_args()

    print("styxx.guardrail cross-dataset validation")
    print(f"  n_pairs_per_dataset = {args.n}")
    print(f"  seed = {args.seed}")
    print(f"  datasets = {args.datasets}")
    print(f"  use_grounding = {not args.no_grounding}")
    print()

    results = {
        "config": {
            "n_pairs": args.n,
            "seed": args.seed,
            "use_grounding": not args.no_grounding,
            "use_entity_verify": False,
        },
        "datasets": {},
    }

    for ds_name in args.datasets:
        try:
            if ds_name == "halueval_qa":
                rows = load_halueval("qa", args.n, args.seed)
            elif ds_name == "halueval_dialogue":
                rows = load_halueval("dialogue", args.n, args.seed)
            elif ds_name == "halueval_summarization":
                rows = load_halueval("summarization", args.n, args.seed)
            elif ds_name == "truthfulqa":
                rows = load_truthfulqa(args.n, args.seed)
            else:
                print(f"  unknown dataset {ds_name!r}, skipping")
                continue
            if not rows:
                print(f"  no rows for {ds_name!r}, skipping")
                continue
            out = run_dataset(ds_name, rows,
                              use_grounding=not args.no_grounding)
            results["datasets"][ds_name] = out
        except Exception as e:
            print(f"  error on {ds_name}: {e}")
            results["datasets"][ds_name] = {"error": str(e)}

    print("\n━━━ summary ━━━")
    for name, out in results["datasets"].items():
        if "error" in out:
            print(f"  {name:<30s}  ERROR: {out['error']}")
        else:
            print(f"  {name:<30s}  AUC={out['auc']:.4f}  "
                  f"(n={out['n_pairs']})")

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()

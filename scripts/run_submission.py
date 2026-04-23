"""Run a cognometry leaderboard submission against the 8 benchmarks.

Loads the user-supplied detector module, calls its score() function
on each (question, response, reference) row from each benchmark,
computes AUC per dataset, averages over 3 seeds, and writes a
results JSON that the leaderboard page reads.

Usage:
    python scripts/run_submission.py submissions/my_detector.py

Output:
    submissions/_results/<system_name>.json
    (also appends to submissions/_results/leaderboard.json)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "submissions" / "_results"

SEEDS = [31, 47, 83]
N_PER_DATASET = 150

VALID_DATASETS = {
    "halueval_qa", "halueval_dialogue", "halueval_summarization",
    "truthfulqa", "halubench_drop", "halubench_pubmed",
    "halubench_finance", "halubench_ragtruth",
}


def _load_submission(path: Path):
    """Load a submission module by file path."""
    spec = importlib.util.spec_from_file_location(
        f"submission_{path.stem}", str(path),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    required = ["SYSTEM_NAME", "AUTHOR", "score"]
    for r in required:
        if not hasattr(mod, r):
            raise ValueError(
                f"submission missing required attribute: {r}"
            )
    score_fn = getattr(mod, "score")
    if not callable(score_fn):
        raise ValueError("score must be callable")
    warmup_fn = getattr(mod, "warmup", None)
    if warmup_fn is not None and callable(warmup_fn):
        warmup_fn()
    return mod


def _auc(y_true, scores):
    """Mann-Whitney U AUC with tie averaging."""
    pairs = sorted(zip(scores, y_true), key=lambda kv: kv[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    ranks_sum = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _load_bench(seed: int):
    """Load all 8 benchmarks using the existing cross-dataset loaders.
    Returns a dict of dataset_name → list of (question, response,
    reference, label) tuples. Label 1 = hallucination, 0 = truth.
    """
    # Delay imports so CI failures bubble up with useful context.
    sys.path.insert(0, str(ROOT))
    from benchmarks.hallucination_test.cross_dataset_8bench import (
        load_halueval, load_truthfulqa, load_halubench_source,
        _answer_str,
    )

    out = {}

    # Paired loaders yield (prompt, truth, hallu, ref) → 2 rows each
    for split in ["qa", "dialogue", "summarization"]:
        rows = load_halueval(split, N_PER_DATASET, seed)
        ds_name = f"halueval_{split}" if split != "dialogue" else "halueval_dialogue"
        # match the naming convention used in v4 weights
        key = {
            "qa": "halueval_qa",
            "dialogue": "halueval_dialogue",
            "summarization": "halueval_summarization",
        }[split]
        bench = []
        for r in rows:
            bench.append((r["prompt"], r["response_truth"], r["reference"], 0))
            bench.append((r["prompt"], r["response_hallu"], r["reference"], 1))
        out[key] = bench

    rows = load_truthfulqa(N_PER_DATASET, seed)
    tq = []
    for r in rows:
        tq.append((r["prompt"], r["response_truth"], r["reference"], 0))
        tq.append((r["prompt"], r["response_hallu"], r["reference"], 1))
    out["truthfulqa"] = tq

    # Unpaired HaluBench sources
    for src, key in [
        ("DROP", "halubench_drop"),
        ("pubmedQA", "halubench_pubmed"),
        ("FinanceBench", "halubench_finance"),
        ("RAGTruth", "halubench_ragtruth"),
    ]:
        rows = load_halubench_source(src, N_PER_DATASET, seed)
        bench = []
        for r in rows:
            bench.append((r["prompt"], r["response"], r["reference"], r["label"]))
        out[key] = bench

    return out


def _score_one(score_fn, question, response, reference):
    try:
        v = float(score_fn(question, response, reference))
    except Exception:
        v = 0.5  # fail-safe: treat errors as "uncertain"
    # Clamp
    return max(0.0, min(1.0, v))


def run_submission(path: Path, verbose: bool = True):
    mod = _load_submission(path)
    name = mod.SYSTEM_NAME

    per_seed = []
    aucs_per_ds = {}

    for seed in SEEDS:
        if verbose:
            print(f"[seed {seed}] loading 8 benchmarks ...")
        bench = _load_bench(seed)
        seed_aucs = {}
        for ds_name, rows in bench.items():
            scores = []
            labels = []
            t0 = time.time()
            for q, r, ref, lbl in rows:
                scores.append(_score_one(mod.score, q, r, ref))
                labels.append(lbl)
            a = _auc(labels, scores)
            seed_aucs[ds_name] = a
            aucs_per_ds.setdefault(ds_name, []).append(a)
            if verbose:
                print(f"  {ds_name:<26s} AUC {a:.4f}  "
                      f"({time.time()-t0:.1f}s, n={len(rows)})")
        per_seed.append({"seed": seed, "aucs": seed_aucs})

    # Averaged
    per_ds_stats = {}
    for ds, vals in aucs_per_ds.items():
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        per_ds_stats[ds] = {
            "mean": round(m, 4),
            "std":  round(std, 4),
            "seeds": [round(v, 4) for v in vals],
        }
    overall = sum(v for vs in aucs_per_ds.values() for v in vs) / \
              max(1, sum(len(vs) for vs in aucs_per_ds.values()))
    above_065 = sum(1 for m in per_ds_stats.values() if m["mean"] >= 0.65)

    result = {
        "system_name": name,
        "author": getattr(mod, "AUTHOR", ""),
        "contact": getattr(mod, "CONTACT", ""),
        "license": getattr(mod, "LICENSE", ""),
        "references": getattr(mod, "REFERENCES", []),
        "declared_failure_modes": getattr(mod, "DECLARED_FAILURE_MODES", []),
        "seeds": SEEDS,
        "n_per_dataset": N_PER_DATASET,
        "per_dataset": per_ds_stats,
        "overall_mean_auc": round(overall, 4),
        "datasets_above_0_65": above_065,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime()),
        "submission_file": str(path.relative_to(ROOT)),
        "per_seed": per_seed,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{re.sub(r'[^a-zA-Z0-9_-]', '_', name)}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote: {out_path}")

    # Update the rolled-up leaderboard file
    leaderboard_path = RESULTS_DIR / "leaderboard.json"
    if leaderboard_path.exists():
        board = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    else:
        board = {"entries": []}
    # Replace any existing entry with the same system_name
    board["entries"] = [e for e in board["entries"]
                         if e.get("system_name") != name]
    board["entries"].append(result)
    board["entries"].sort(
        key=lambda e: e.get("overall_mean_auc", 0.0), reverse=True,
    )
    board["updated_at"] = result["submitted_at"]
    leaderboard_path.write_text(json.dumps(board, indent=2), encoding="utf-8")
    print(f"updated: {leaderboard_path}")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="path to submission .py file")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    path = Path(args.submission)
    if not path.exists():
        sys.exit(f"submission not found: {path}")
    try:
        run_submission(path, verbose=not args.quiet)
    except Exception as e:
        print(f"\nSUBMISSION FAILED:\n{traceback.format_exc()}",
              file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

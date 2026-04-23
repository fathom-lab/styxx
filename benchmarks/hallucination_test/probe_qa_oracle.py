# -*- coding: utf-8 -*-
"""
The creative hack: use a trained extractive QA model (deepset/roberta-base-squad2)
as an independent oracle. Ask it "Given the passage, what's the answer?",
compare its answer to the provided answer, emit a hallucination signal
when they disagree.

This is the SelfCheckGPT mechanism adapted for DROP — external-model
consistency as the hallucination signal. RoBERTa-SQuAD is 125M params,
CPU-friendly (~50 ms/pair on CUDA), and specifically trained on extractive
reading comprehension.

If this works, it's a v4.1 path that actually solves DROP.

Run:
    python benchmarks/hallucination_test/probe_qa_oracle.py --n 150 --seed 31
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _answer_str(a) -> str:
    if isinstance(a, list):
        return " | ".join(str(x) for x in a)
    return str(a)


def _tokens(s):
    return set(re.findall(r"\w+", s.lower()))


def _overlap(a: str, b: str) -> float:
    """Fraction of `a` tokens present in `b` tokens."""
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def _numeric_distance(a: str, b: str) -> float:
    """If both strings have a leading number, |a-b| / max(|a|, |b|, 1)."""
    ma = re.search(r"-?\d+(?:\.\d+)?", a)
    mb = re.search(r"-?\d+(?:\.\d+)?", b)
    if not (ma and mb):
        return 1.0 if ma != mb else 0.0
    try:
        va = float(ma.group(0))
        vb = float(mb.group(0))
    except ValueError:
        return 1.0
    return min(1.0, abs(va - vb) / max(abs(va), abs(vb), 1.0))


def load_drop(n, seed):
    from datasets import load_dataset
    import random
    rng = random.Random(seed)
    ds = load_dataset("PatronusAI/HaluBench", split="test",
                      streaming=True)
    pass_rows, fail_rows = [], []
    for r in ds:
        if r.get("source_ds") != "DROP":
            continue
        if r["label"] == "PASS" and len(pass_rows) < n * 2:
            pass_rows.append(r)
        elif r["label"] == "FAIL" and len(fail_rows) < n * 2:
            fail_rows.append(r)
        if len(pass_rows) >= n * 2 and len(fail_rows) >= n * 2:
            break
    rng.shuffle(pass_rows)
    rng.shuffle(fail_rows)
    out = []
    for r in pass_rows[:n] + fail_rows[:n]:
        out.append({
            "question": r.get("question") or "",
            "response": _answer_str(r.get("answer", "")),
            "reference": r.get("passage") or "",
            "label": 1 if r["label"] == "FAIL" else 0,
        })
    rng.shuffle(out)
    return out


def _auc(y_true, scores):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--model", default="deepset/roberta-base-squad2")
    args = ap.parse_args()

    print(f"loading QA oracle: {args.model}")
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    qa = pipeline("question-answering", model=args.model, device=device)
    print(f"  device={'cuda' if device == 0 else 'cpu'}")

    rows = load_drop(args.n, args.seed)
    print(f"  {len(rows)} DROP rows loaded")

    # Two signals:
    #   overlap_score: 1 - token_overlap(oracle_answer, provided_answer)
    #                  (higher = more disagreement = more hallucinated)
    #   numeric_dist:  relative distance between numbers in oracle vs provided
    overlap_scores = []
    numeric_dists = []
    combined = []
    labels = []
    oracle_examples = []

    for i, r in enumerate(rows):
        try:
            out = qa({
                "question": r["question"],
                "context": r["reference"][:4000],  # cap for speed
            })
            oracle_ans = out["answer"]
        except Exception as e:
            oracle_ans = ""
        ov = 1.0 - _overlap(oracle_ans, r["response"])
        nd = _numeric_distance(oracle_ans, r["response"])
        overlap_scores.append(ov)
        numeric_dists.append(nd)
        combined.append(0.5 * ov + 0.5 * nd)
        labels.append(r["label"])
        if i < 5:
            oracle_examples.append({
                "q": r["question"][:90],
                "provided": r["response"][:60],
                "oracle":   oracle_ans[:60],
                "label":    "FAIL" if r["label"] == 1 else "PASS",
            })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}")

    print(f"\n=== oracle examples ===")
    for ex in oracle_examples:
        print(f"  [{ex['label']}] Q: {ex['q']}")
        print(f"        provided: {ex['provided']}")
        print(f"        oracle:   {ex['oracle']}")

    print(f"\n=== RESULTS (DROP, n={len(rows)}, seed {args.seed}) ===")
    ao = _auc(labels, overlap_scores)
    an = _auc(labels, numeric_dists)
    ac = _auc(labels, combined)
    pm = sum(s for s, l in zip(overlap_scores, labels) if l == 0) / max(1, labels.count(0))
    fm = sum(s for s, l in zip(overlap_scores, labels) if l == 1) / max(1, labels.count(1))
    print(f"  overlap_disagreement  AUC {ao:.4f}  pass={pm:.3f} fail={fm:.3f}")
    pm = sum(s for s, l in zip(numeric_dists, labels) if l == 0) / max(1, labels.count(0))
    fm = sum(s for s, l in zip(numeric_dists, labels) if l == 1) / max(1, labels.count(1))
    print(f"  numeric_distance      AUC {an:.4f}  pass={pm:.3f} fail={fm:.3f}")
    print(f"  combined (0.5+0.5)    AUC {ac:.4f}")

    print(f"\nbaseline v4.0.0 on DROP: AUC 0.4238")
    print(f"threshold for 'ship it':  AUC >= 0.70")
    print(f"threshold for 'useful':   AUC >= 0.60")


if __name__ == "__main__":
    main()

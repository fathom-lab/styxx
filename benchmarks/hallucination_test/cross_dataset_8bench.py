# -*- coding: utf-8 -*-
"""
8-dataset pooled calibration — styxx.guardrail v4.0.0 final.

Extends the 4-dataset v3 calibration (HaluEval-QA/Dialog/Summ +
TruthfulQA) to 8 benchmarks by adding 4 more from PatronusAI's
public HaluBench: DROP (reading comprehension), pubmedQA (biomedical),
FinanceBench (finance), RAGTruth (RAG faithfulness).

Two data shapes are supported:
  - paired: (response_truth, response_hallu) per row → 2 examples
  - unpaired: (response, label in {PASS, FAIL}) per row → 1 example

Both fit the same 9-signal LR (NLI + 8 prior signals).

Usage:
    python benchmarks/hallucination_test/cross_dataset_8bench.py \\
        --n 150 --seed 31 --no_entity --nli

Saves pooled weights + per-dataset AUC to
  benchmarks/hallucination_test/results/cross_dataset_8bench.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Disable HF symlink warning noise
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from styxx.guardrail.claim_decomposer import decompose
from styxx.guardrail.text_signals import (
    compute_text_signal, claim_risk_text_only,
)
from styxx.guardrail.entity_verify import verify_entities_batch
from styxx.guardrail.knowledge_grounding import response_grounding_risk
from styxx.guardrail.response_novelty import response_novelty_signals
from styxx.guardrail.nli_signal import NLIScorer


FEAT_NAMES = [
    "text_claim_risk", "entity_unverified_frac",
    "knowledge_grounding",
    "content_novelty", "entity_novelty", "number_novelty",
    "bigram_novelty", "trigram_novelty",
    "nli_contradict",
]


# ─────────── paired loaders (HaluEval + TruthfulQA) ───────────

def load_halueval(split, n, seed):
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


def load_truthfulqa(n, seed):
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
        out.append({
            "prompt": q,
            "response_truth": best_correct,
            "response_hallu": incorrect[0],
            "reference": " ".join(correct),
        })
        if len(out) >= n:
            break
    return out[:n]


# ─────────── unpaired loaders (HaluBench) ───────────

def _str_answer(a):
    """HaluBench DROP stores answers as lists; everything else as strings."""
    if isinstance(a, list):
        return " | ".join(str(x) for x in a)
    return str(a)


def load_halubench_source(source_ds: str, n: int, seed: int):
    """Return a balanced sample of (passage, question, answer, label)
    rows from one source of HaluBench.

    ``n`` is the target per-class count; total rows = 2*n.
    """
    from datasets import load_dataset
    ds = load_dataset("PatronusAI/HaluBench", split="test",
                      streaming=True)
    pass_rows, fail_rows = [], []
    for row in ds:
        if row.get("source_ds") != source_ds:
            continue
        lbl = row.get("label")
        if lbl == "PASS" and len(pass_rows) < n * 2:
            pass_rows.append(row)
        elif lbl == "FAIL" and len(fail_rows) < n * 2:
            fail_rows.append(row)
        if len(pass_rows) >= n * 2 and len(fail_rows) >= n * 2:
            break

    rng = random.Random(seed)
    rng.shuffle(pass_rows)
    rng.shuffle(fail_rows)
    pass_rows = pass_rows[:n]
    fail_rows = fail_rows[:n]

    out = []
    for r in pass_rows + fail_rows:
        out.append({
            "prompt": r.get("question") or "",
            "response": _str_answer(r.get("answer", "")),
            "reference": r.get("passage") or "",
            "label": 1 if r["label"] == "FAIL" else 0,
        })
    rng.shuffle(out)
    return out


# ─────────── signal extraction ───────────

def extract_signals(prompt, response, reference,
                     entity_cache=None, use_entity_verify=True,
                     nli_scorer=None):
    claims = decompose(response)
    text_resp = compute_text_signal(response, prompt)
    per_claim = [claim_risk_text_only(c, text_resp) for c in claims]
    text_risk = (sum(per_claim) / len(per_claim)
                 if per_claim else 0.0)

    if use_entity_verify:
        all_ents = []
        for c in claims:
            all_ents.extend(c.entities)
        all_ents = list(dict.fromkeys(all_ents))
        if all_ents:
            if entity_cache is None:
                entity_cache = {}
            uncached = [e for e in all_ents if e not in entity_cache]
            if uncached:
                results = verify_entities_batch(uncached)
                entity_cache.update(results)
            n_unv = sum(1 for e in all_ents
                         if not entity_cache.get(
                             e, {"verified": False})["verified"])
            ent_frac = n_unv / len(all_ents)
        else:
            ent_frac = 0.0
    else:
        ent_frac = 0.0

    ground = (response_grounding_risk(claims, reference)
              if reference else 0.5)
    nov = response_novelty_signals(response, reference or "")

    nli_contradict = 0.0
    if nli_scorer is not None and reference:
        try:
            nli_contradict = nli_scorer.score(
                premise=reference, hypothesis=response,
            )
        except Exception:
            nli_contradict = 0.0

    return [
        text_risk, ent_frac, ground,
        nov["content_novelty"], nov["entity_novelty"],
        nov["number_novelty"], nov["bigram_novelty"],
        nov["trigram_novelty"],
        nli_contradict,
    ]


# ─────────── LR (no sklearn dep) ───────────

def _sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_lr(X, y, lr=0.3, epochs=800, l2=0.05):
    d = len(X[0])
    w = [0.0] * d
    b = 0.0
    n = len(X)
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xij for wj, xij in zip(w, xi))
            p = _sigmoid(z)
            err = p - yi
            for j in range(d):
                gw[j] += err * xi[j]
            gb += err
        for j in range(d):
            w[j] = w[j] - lr * (gw[j] / n + l2 * w[j])
        b = b - lr * gb / n
    return w, b


def predict_lr(w, b, X):
    return [
        _sigmoid(b + sum(wj * xij for wj, xij in zip(w, xi)))
        for xi in X
    ]


def _auc(y_true, scores):
    pairs = sorted(zip(scores, y_true))
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks_sum = 0.0
    for rank, (_, l) in enumerate(pairs, start=1):
        if l == 1:
            ranks_sum += rank
    return (ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ─────────── main ───────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150,
                    help="per-dataset target row count (paired: n pairs "
                         "= 2n examples; unpaired: n per class = 2n)")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--train_frac", type=float, default=0.75)
    ap.add_argument("--no_entity", action="store_true")
    ap.add_argument("--nli", action="store_true")
    ap.add_argument(
        "--out_file",
        default=str(ROOT / "benchmarks" / "hallucination_test" /
                    "results" / "cross_dataset_8bench.json"),
    )
    args = ap.parse_args()

    print("styxx.guardrail 8-benchmark pooled calibration")
    print(f"  n = {args.n}")
    print(f"  seed = {args.seed}")
    print(f"  nli = {args.nli}")

    nli_scorer = None
    if args.nli:
        print("loading NLI scorer...")
        nli_scorer = NLIScorer()
        nli_scorer._load()
        print(f"  on {nli_scorer._device}")

    # 4 paired (HaluEval + TruthfulQA) + 4 unpaired (HaluBench)
    paired_loaders = [
        ("halueval_qa",
         lambda: load_halueval("qa", args.n, args.seed)),
        ("halueval_dialogue",
         lambda: load_halueval("dialogue", args.n, args.seed)),
        ("halueval_summarization",
         lambda: load_halueval("summarization", args.n, args.seed)),
        ("truthfulqa",
         lambda: load_truthfulqa(args.n, args.seed)),
    ]

    unpaired_loaders = [
        ("halubench_drop",
         lambda: load_halubench_source("DROP", args.n, args.seed)),
        ("halubench_pubmed",
         lambda: load_halubench_source("pubmedQA", args.n, args.seed)),
        ("halubench_finance",
         lambda: load_halubench_source("FinanceBench", args.n, args.seed)),
        ("halubench_ragtruth",
         lambda: load_halubench_source("RAGTruth", args.n, args.seed)),
    ]

    per_ds_data = {}
    ent_cache = {}
    t0 = time.time()

    for name, loader in paired_loaders:
        print(f"\n=== {name} (paired) ===")
        try:
            rows = loader()
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        X, y = [], []
        for i, row in enumerate(rows):
            try:
                sig_truth = extract_signals(
                    row["prompt"], row["response_truth"],
                    row["reference"], ent_cache,
                    use_entity_verify=not args.no_entity,
                    nli_scorer=nli_scorer,
                )
                sig_hallu = extract_signals(
                    row["prompt"], row["response_hallu"],
                    row["reference"], ent_cache,
                    use_entity_verify=not args.no_entity,
                    nli_scorer=nli_scorer,
                )
                X.append(sig_truth); y.append(0)
                X.append(sig_hallu); y.append(1)
            except Exception:
                continue
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)")
        per_ds_data[name] = (X, y)
        print(f"  {len(X)} examples")

    for name, loader in unpaired_loaders:
        print(f"\n=== {name} (unpaired, HaluBench) ===")
        try:
            rows = loader()
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        X, y = [], []
        for i, row in enumerate(rows):
            try:
                sig = extract_signals(
                    row["prompt"], row["response"],
                    row["reference"], ent_cache,
                    use_entity_verify=not args.no_entity,
                    nli_scorer=nli_scorer,
                )
                X.append(sig); y.append(row["label"])
            except Exception:
                continue
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)")
        per_ds_data[name] = (X, y)
        print(f"  {len(X)} examples  (pass/fail: "
              f"{y.count(0)}/{y.count(1)})")

    # Train/test split per dataset, then pool for training
    print("\n=== train/test split ===")
    train_X, train_y = [], []
    test_blocks = {}
    rng = random.Random(args.seed)
    for name, (X, y) in per_ds_data.items():
        if not X:
            continue
        idxs = list(range(len(X)))
        rng.shuffle(idxs)
        n_train = int(args.train_frac * len(idxs))
        for i in idxs[:n_train]:
            train_X.append(X[i]); train_y.append(y[i])
        test_blocks[name] = (
            [X[i] for i in idxs[n_train:]],
            [y[i] for i in idxs[n_train:]],
        )
        print(f"  {name}: {n_train} train, "
              f"{len(idxs) - n_train} test")
    print(f"  pooled train: {len(train_X)}")

    # Fit
    print("\n=== fitting pooled LR ===")
    w, b = fit_lr(train_X, train_y, lr=0.3, epochs=800, l2=0.05)
    print(f"  intercept: {b:+.4f}")
    for name, wi in zip(FEAT_NAMES, w):
        print(f"  {name:<28s}: {wi:+.4f}")

    # Evaluate
    train_auc = _auc(train_y, predict_lr(w, b, train_X))
    print(f"\n  train AUC (pooled): {train_auc:.4f}")
    print(f"\n=== per-dataset held-out AUC ===")
    test_aucs = {}
    for name, (Xt, yt) in test_blocks.items():
        if not Xt:
            continue
        pt = predict_lr(w, b, Xt)
        a = _auc(yt, pt)
        test_aucs[name] = {"n_test": len(Xt), "auc": a}
        print(f"  {name:<28s} n={len(Xt):4d}  AUC={a:.4f}")

    aucs = [m["auc"] for m in test_aucs.values()]
    mean_auc = sum(aucs) / len(aucs) if aucs else 0.0
    print(f"\n  mean test AUC across {len(aucs)} datasets: {mean_auc:.4f}")

    result = {
        "config": {
            "n_target": args.n,
            "seed": args.seed,
            "train_frac": args.train_frac,
            "use_entity_verify": not args.no_entity,
            "use_nli": args.nli,
        },
        "coefs": {name: round(wi, 4)
                  for name, wi in zip(FEAT_NAMES, w)},
        "intercept": round(b, 4),
        "train_pooled_auc": round(train_auc, 4),
        "test_per_dataset": {
            name: {"n_test": m["n_test"], "auc": round(m["auc"], 4)}
            for name, m in test_aucs.items()
        },
        "summary": {
            "mean_test_auc": round(mean_auc, 4),
            "min_test_auc": round(min(aucs), 4) if aucs else 0.0,
            "n_datasets": len(aucs),
        },
    }
    Path(args.out_file).write_text(
        json.dumps(result, indent=2), encoding="utf-8",
    )
    print(f"\nwrote: {args.out_file}")


if __name__ == "__main__":
    main()

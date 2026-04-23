# -*- coding: utf-8 -*-
"""
Pooled multi-dataset LR calibration for styxx.guardrail.

Extracts the four guardrail signals (text_claim_risk,
entity_unverified_frac, knowledge_grounding, [probe_confab]) on
n pairs per dataset across HaluEval-QA/Dialog/Summ + TruthfulQA,
pools them, fits a logistic regression, and evaluates per-dataset
AUC on held-out data.

Output: JSON with per-dataset metrics + fitted coefficients, plus
a new calibrated_weights_pooled.py written to styxx.guardrail
when the pooled AUC >= threshold on every dataset.

Usage:
    python cross_dataset_calibrate.py --n 200 --seed 31
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from styxx.guardrail.claim_decomposer import decompose
from styxx.guardrail.entity_verify import verify_entities_batch
from styxx.guardrail.text_signals import (
    compute_text_signal, claim_risk_text_only,
)
from styxx.guardrail.knowledge_grounding import response_grounding_risk
from styxx.guardrail.response_novelty import response_novelty_signals
from styxx.guardrail.nli_signal import NLIScorer


# ─────────── dataset loaders (copy of cross_dataset_benchmark) ───────────

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


# ─────────── signal extraction ───────────

def extract_signals(prompt, response, reference,
                     entity_cache=None, use_entity_verify=True,
                     nli_scorer=None):
    """Return full signal vector for pooled LR."""
    # text signal
    claims = decompose(response)
    text_resp = compute_text_signal(response, prompt)
    per_claim = [claim_risk_text_only(c, text_resp) for c in claims]
    text_risk = (sum(per_claim) / len(per_claim)
                 if per_claim else 0.0)

    # entity verify
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

    # grounding
    if reference:
        ground = response_grounding_risk(claims, reference)
    else:
        ground = 0.5

    # response-novelty signals
    nov = response_novelty_signals(response, reference or "")

    # NLI contradiction
    nli_contradict = 0.0
    if nli_scorer is not None and reference:
        try:
            nli_contradict = nli_scorer.score(
                premise=reference, hypothesis=response,
            )
        except Exception:
            nli_contradict = 0.0

    return [
        text_risk,
        ent_frac,
        ground,
        nov["content_novelty"],
        nov["entity_novelty"],
        nov["number_novelty"],
        nov["bigram_novelty"],
        nov["trigram_novelty"],
        nli_contradict,
    ]


# ─────────── LR training (gradient descent, no sklearn dep) ───────────

def _sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_lr(X, y, lr=0.3, epochs=500, l2=0.01):
    """Simple batch GD logistic regression with L2. X is list of lists."""
    n_feats = len(X[0])
    n = len(X)
    w = [0.0] * n_feats
    b = 0.0
    for epoch in range(epochs):
        # predictions
        preds = [_sigmoid(sum(wi * xij for wi, xij in zip(w, xi)) + b)
                 for xi in X]
        # gradients
        dw = [0.0] * n_feats
        db = 0.0
        for xi, pi, yi in zip(X, preds, y):
            err = pi - yi
            for j in range(n_feats):
                dw[j] += err * xi[j]
            db += err
        # l2 + scale
        for j in range(n_feats):
            dw[j] = dw[j] / n + l2 * w[j]
        db = db / n
        # update
        for j in range(n_feats):
            w[j] -= lr * dw[j]
        b -= lr * db
    return w, b


def predict_lr(w, b, X):
    return [_sigmoid(sum(wi * xij for wi, xij in zip(w, xi)) + b)
            for xi in X]


# ─────────── AUC ───────────

def _auc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks_sum = 0.0
    for rank, (s, l) in enumerate(pairs, start=1):
        if l == 1:
            ranks_sum += rank
    return (ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ─────────── main ───────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="pairs per dataset (total = n * 4)")
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--train_frac", type=float, default=0.75)
    ap.add_argument("--no_entity", action="store_true",
                    help="Skip entity verify (fast but weaker)")
    ap.add_argument("--nli", action="store_true",
                    help="Include NLI contradiction signal")
    ap.add_argument(
        "--out_file",
        default=str(ROOT / "benchmarks" / "hallucination_test" /
                    "results" / "cross_dataset_calibration.json"),
    )
    args = ap.parse_args()

    print("styxx.guardrail pooled multi-dataset calibration")
    print(f"  n_pairs_per_dataset = {args.n}")
    print(f"  seed = {args.seed}")
    print(f"  train_frac = {args.train_frac}")
    print(f"  use_entity_verify = {not args.no_entity}")
    print(f"  use_nli = {args.nli}")
    print()

    nli_scorer = None
    if args.nli:
        print("loading NLI scorer (deberta-v3-base-mnli-fever-anli)...")
        nli_scorer = NLIScorer()
        nli_scorer._load()
        print(f"  loaded on {nli_scorer._device}")

    loaders = [
        ("halueval_qa",
          lambda: load_halueval("qa", args.n, args.seed)),
        ("halueval_dialogue",
          lambda: load_halueval("dialogue", args.n, args.seed)),
        ("halueval_summarization",
          lambda: load_halueval("summarization", args.n, args.seed)),
        ("truthfulqa",
          lambda: load_truthfulqa(args.n, args.seed)),
    ]

    # Extract signals for every row
    per_ds_data = {}
    ent_cache = {}
    t0 = time.time()
    for name, loader in loaders:
        print(f"\n━━━ extracting signals: {name} ━━━")
        try:
            rows = loader()
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        X = []
        y = []
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
                X.append(sig_truth)
                y.append(0)
                X.append(sig_hallu)
                y.append(1)
            except Exception as e:
                continue
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(rows)} "
                      f"({time.time()-t0:.0f}s total)")
        per_ds_data[name] = (X, y)
        print(f"  {len(X)} rows extracted")

    # Train/test split (interleaved truth/hallu → split by index)
    print("\n━━━ train/test split ━━━")
    train_X, train_y = [], []
    test_blocks = {}
    rng = random.Random(args.seed)
    for name, (X, y) in per_ds_data.items():
        idxs = list(range(len(X)))
        rng.shuffle(idxs)
        n_train = int(args.train_frac * len(idxs))
        train_idx = idxs[:n_train]
        test_idx = idxs[n_train:]
        for i in train_idx:
            train_X.append(X[i])
            train_y.append(y[i])
        test_blocks[name] = (
            [X[i] for i in test_idx],
            [y[i] for i in test_idx],
        )
        print(f"  {name}: {len(train_idx)} train, {len(test_idx)} test")
    print(f"  pooled train: {len(train_X)}")

    # Fit
    print("\n━━━ fitting pooled LR ━━━")
    w, b = fit_lr(train_X, train_y, lr=0.3, epochs=800, l2=0.05)
    print(f"  intercept: {b:+.4f}")
    feat_names = [
        "text_claim_risk", "entity_unverified_frac",
        "knowledge_grounding",
        "content_novelty", "entity_novelty", "number_novelty",
        "bigram_novelty", "trigram_novelty",
        "nli_contradict",
    ]
    for name, wi in zip(feat_names, w):
        print(f"  {name:<28s}: {wi:+.4f}")

    # Train AUC
    train_preds = predict_lr(w, b, train_X)
    train_auc = _auc(train_y, train_preds)
    print(f"\n  train AUC (pooled): {train_auc:.4f}")

    # Per-dataset test AUC
    print("\n━━━ held-out test AUC per dataset ━━━")
    results = {
        "config": {
            "n_pairs_per_dataset": args.n,
            "seed": args.seed,
            "train_frac": args.train_frac,
            "use_entity_verify": not args.no_entity,
        },
        "coefs": {
            "text_claim_risk": round(w[0], 4),
            "entity_unverified_frac": round(w[1], 4),
            "knowledge_grounding": round(w[2], 4),
            "content_novelty": round(w[3], 4),
            "entity_novelty": round(w[4], 4),
            "number_novelty": round(w[5], 4),
            "bigram_novelty": round(w[6], 4),
            "trigram_novelty": round(w[7], 4),
            "nli_contradict": round(w[8], 4) if len(w) > 8 else 0.0,
            "intercept": round(b, 4),
        },
        "train_pooled_auc": round(train_auc, 4),
        "test_per_dataset": {},
    }

    for name, (tX, ty) in test_blocks.items():
        preds = predict_lr(w, b, tX)
        auc = _auc(ty, preds)
        truth_scores = [s for s, l in zip(preds, ty) if l == 0]
        hallu_scores = [s for s, l in zip(preds, ty) if l == 1]
        results["test_per_dataset"][name] = {
            "n_test": len(ty),
            "auc": round(auc, 4),
            "truth_mean": round(
                sum(truth_scores) / max(1, len(truth_scores)), 4),
            "hallu_mean": round(
                sum(hallu_scores) / max(1, len(hallu_scores)), 4),
        }
        print(f"  {name:<28s}  AUC = {auc:.4f}  "
              f"(truth={results['test_per_dataset'][name]['truth_mean']:.3f}, "
              f"hallu={results['test_per_dataset'][name]['hallu_mean']:.3f})")

    # Summary verdict
    aucs = [r["auc"] for r in results["test_per_dataset"].values()]
    min_auc = min(aucs)
    mean_auc = sum(aucs) / len(aucs)
    print(f"\n━━━ verdict ━━━")
    print(f"  mean test AUC across datasets: {mean_auc:.4f}")
    print(f"  min  test AUC across datasets: {min_auc:.4f}")
    if min_auc >= 0.80:
        print("  ✓ pooled LR generalizes — ship as v3.9.1")
    elif min_auc >= 0.70:
        print("  ~ acceptable generalization — consider shipping with caveats")
    else:
        print("  ✗ did not generalize — iterate on signals")
    results["summary"] = {
        "mean_test_auc": round(mean_auc, 4),
        "min_test_auc": round(min_auc, 4),
        "recommended_action": (
            "ship" if min_auc >= 0.80
            else "caveat" if min_auc >= 0.70
            else "iterate"
        ),
    }

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote {args.out_file}")
    print(f"  total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

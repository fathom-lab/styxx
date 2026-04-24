"""Hallucination phase-transition probe — HaluEval-QA.

Completes the 3-of-3 cognometric-instrument phase-transition test.
The drift detector phase-transitioned at K=2 per failure class
[drift_phase_transitions.md, 2026-04-23]. The refusal detector
phase-transitioned at K=1 [refusal_phase_transitions.md, 2026-04-24a]
and showed per-substrate shift on XSTest x 5 families
[refusal_cross_model_phase_transitions.md, 2026-04-24b]. This script
closes the third leg: does the v4 hallucination detector also show
phase-transition structure?

Scope: HaluEval-QA only (150 paired samples = 300 examples). This is
the benchmark where the full-9-feature model lands AUC 0.998 — the
clearest candidate for a sharp phase transition.

Features (9): text_claim_risk, entity_unverified_frac,
knowledge_grounding, content_novelty, entity_novelty, number_novelty,
bigram_novelty, trigram_novelty, nli_contradict.

NLI model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (cached).

Output:
  benchmarks/hallucination_feature_scaling.json
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from styxx.guardrail.claim_decomposer import decompose  # noqa: E402
from styxx.guardrail.text_signals import (  # noqa: E402
    compute_text_signal, claim_risk_text_only,
)
from styxx.guardrail.knowledge_grounding import response_grounding_risk  # noqa: E402
from styxx.guardrail.response_novelty import response_novelty_signals  # noqa: E402
from styxx.guardrail.nli_signal import NLIScorer  # noqa: E402

OUT_JSON = REPO / "benchmarks" / "hallucination_feature_scaling.json"

RANDOM_STATE = 0
N_PER_CLASS = 150   # 150 truth + 150 hallucinated = 300 samples
N_SPLITS = 5

FEATURE_NAMES = [
    "text_claim_risk",
    "entity_unverified_frac",  # zeroed (use_entity_verify=False)
    "knowledge_grounding",
    "content_novelty",
    "entity_novelty",
    "number_novelty",
    "bigram_novelty",
    "trigram_novelty",
    "nli_contradict",
]


def load_halueval_qa(n, seed):
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", "qa",
                      split="data", streaming=True)
    rng = random.Random(seed)
    rows = []
    for row in ds:
        if all(row.get(k) for k in
               ("knowledge", "question", "right_answer", "hallucinated_answer")):
            rows.append(row)
        if len(rows) >= n * 3:
            break
    rng.shuffle(rows)
    out = []
    for row in rows[:n]:
        out.append({
            "prompt": row["question"],
            "response_truth": row["right_answer"],
            "response_hallu": row["hallucinated_answer"],
            "reference": row["knowledge"],
        })
    return out


def extract_signals(prompt, response, reference, nli_scorer):
    claims = decompose(response)
    text_resp = compute_text_signal(response, prompt)
    per_claim = [claim_risk_text_only(c, text_resp) for c in claims]
    text_risk = sum(per_claim) / len(per_claim) if per_claim else 0.0
    ground = response_grounding_risk(claims, reference) if reference else 0.5
    nov = response_novelty_signals(response, reference or "")
    nli_contradict = 0.0
    if nli_scorer is not None and reference:
        try:
            nli_contradict = nli_scorer.score(premise=reference, hypothesis=response)
        except Exception:
            nli_contradict = 0.0
    return [
        text_risk,
        0.0,  # entity_unverified_frac disabled
        ground,
        nov["content_novelty"],
        nov["entity_novelty"],
        nov["number_novelty"],
        nov["bigram_novelty"],
        nov["trigram_novelty"],
        nli_contradict,
    ]


def cv_auc(X, y, feature_idx):
    if len(feature_idx) == 0:
        return 0.5
    Xs = X[:, feature_idx]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)
    scores, labels = [], []
    for tr, te in skf.split(Xs, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(Xs[tr])
        X_te = sc.transform(Xs[te])
        clf = LogisticRegression(
            C=1.0, max_iter=2000, random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        clf.fit(X_tr, y[tr])
        p = clf.predict_proba(X_te)[:, 1]
        scores.extend(p)
        labels.extend(y[te])
    return float(roc_auc_score(labels, scores))


def main():
    print("=== Hallucination phase-transition probe — HaluEval-QA ===")
    print(f"  n_per_class = {N_PER_CLASS} (pairs)")
    print(f"  seed = {RANDOM_STATE}")
    print(f"  features = {len(FEATURE_NAMES)}")
    print()

    print("[1/4] loading HaluEval-QA ...")
    t0 = time.time()
    data = load_halueval_qa(N_PER_CLASS, seed=RANDOM_STATE)
    print(f"      loaded {len(data)} paired rows in {time.time()-t0:.1f}s")

    print("[2/4] loading NLI scorer ...")
    t0 = time.time()
    nli = NLIScorer()
    print(f"      loaded in {time.time()-t0:.1f}s")

    print(f"[3/4] extracting 9 signals per sample ({2*len(data)} total) ...")
    t0 = time.time()
    X_list, y_list = [], []
    for i, row in enumerate(data):
        feats_t = extract_signals(
            row["prompt"], row["response_truth"], row["reference"], nli)
        feats_h = extract_signals(
            row["prompt"], row["response_hallu"], row["reference"], nli)
        X_list.append(feats_t); y_list.append(0)
        X_list.append(feats_h); y_list.append(1)
        if (i + 1) % 25 == 0:
            print(f"      progress: {i+1}/{len(data)}  "
                  f"elapsed {time.time()-t0:.1f}s")
    X = np.array(X_list, dtype=float)
    y = np.array(y_list)
    print(f"      X shape: {X.shape}, y mean: {y.mean():.2f}, "
          f"elapsed {time.time()-t0:.1f}s")

    print("[4/4] top-K ablation ...")
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf_full = LogisticRegression(
        C=1.0, max_iter=2000, random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    clf_full.fit(Xs, y)
    ranked = sorted(
        zip(FEATURE_NAMES, clf_full.coef_[0], range(len(FEATURE_NAMES))),
        key=lambda kv: -abs(kv[1]),
    )
    print("  ranking by |coef|:")
    for rank, (name, c, _) in enumerate(ranked, 1):
        print(f"    {rank:2d}. {name:<26s}  {c:+.3f}")

    full_auc = cv_auc(X, y, list(range(len(FEATURE_NAMES))))
    print(f"  full-model 5-fold CV AUC: {full_auc:.4f}")
    print()

    print(f"{'='*64}")
    print("Ablation 1: TOP-K by |coef|")
    print(f"{'='*64}")
    top_k_results = []
    prev_auc = 0.5
    for K in range(0, len(FEATURE_NAMES) + 1):
        if K == 0:
            auc = 0.5
            added = None
        else:
            idx = [r[2] for r in ranked[:K]]
            auc = cv_auc(X, y, idx)
            added = ranked[K - 1][0]
        delta = auc - prev_auc
        flag = "  <-- phase transition (>= 0.10)" if abs(delta) >= 0.10 else ""
        print(f"  K={K:2d}  AUC={auc:.4f}  d={delta:+.4f}  added={added}{flag}")
        top_k_results.append({
            "K": K, "added": added, "auc": auc,
            "delta_from_prev": delta,
            "phase_transition": bool(abs(delta) >= 0.10),
        })
        prev_auc = auc

    print()
    print(f"{'='*64}")
    print("Ablation 2: RANDOM subsets (3 seeds per K)")
    print(f"{'='*64}")
    random_results = []
    for K in range(1, len(FEATURE_NAMES) + 1):
        aucs = []
        for seed in range(3):
            r_rng = random.Random(100 * K + seed)
            idx = r_rng.sample(range(len(FEATURE_NAMES)), K)
            aucs.append(cv_auc(X, y, idx))
        mean = float(np.mean(aucs))
        std = float(np.std(aucs))
        print(f"  K={K:2d}  random AUC mean={mean:.4f} +/- {std:.4f}")
        random_results.append({
            "K": K, "random_aucs": aucs,
            "random_mean": mean, "random_std": std,
        })

    print()
    print(f"{'='*64}")
    print("VERDICT")
    print(f"{'='*64}")
    pts = [r for r in top_k_results if r["phase_transition"]]
    if pts:
        for pt in pts:
            print(f"  phase transition: K={pt['K']}  +{pt['added']}  d={pt['delta_from_prev']:+.4f}")
    else:
        print("  no phase transitions >= 0.10 found.")

    out = {
        "methodology": "hallucination v4 feature-scaling, HaluEval-QA, 5-fold CV seed=0",
        "instrument": "hallucination-v4",
        "dataset": "HaluEval-QA",
        "n_samples": int(len(y)),
        "class_balance": [int(sum(y == 0)), int(sum(y == 1))],
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "full_model_importance": [
            {"rank": i+1, "feature": r[0], "coef": r[1]}
            for i, r in enumerate(ranked)
        ],
        "full_model_auc": full_auc,
        "top_k": top_k_results,
        "random_subsets": random_results,
        "phase_transitions_found": pts,
        "phase_transition_threshold": 0.10,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print(f"wrote -> {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()

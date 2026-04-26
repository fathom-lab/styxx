"""Train cognometric instrument #5: conversation-loop detector v0.

Pipeline:
  1. Sample N seed topics; for each, run a 4-agent-turn conversation
     under contrasting system prompts:
       - "loop"     → expected to repeat itself across turns
       - "progress" → expected to introduce new content each turn
     Both conditions get the same generic follow-up user prompts
     ("Hmm, can you elaborate?" / "Tell me more." / etc.) so the
     ONLY difference is the agent's instruction to repeat vs. progress.
  2. Each completed 4-turn conversation = one training example.
  3. Extract the 9-feature loop vector for every conversation.
  4. Fit StandardScaler + Logistic Regression with seed-pinned 5-fold CV.
  5. Run greedy forward-selection ablation to find critical-K.
  6. Print weights, AUCs, ablation table, and the calibration fingerprint.

Usage:
    OPENAI_API_KEY=... python scripts/loop_train_v0.py [--n-topics 100] [--n-turns 4] [--model gpt-4o-mini]

Outputs:
    benchmarks/data/loop/conversations_v0.jsonl       (resumable cache)
    benchmarks/loop_feature_scaling.json              (ablation table)
    benchmarks/loop_weights_v0.json                   (paste-ready weights)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from styxx.guardrail.conversation_loop_signals import extract_loop_features

CORPUS_DIR = ROOT / "benchmarks" / "data" / "loop"
OUT_DIR = ROOT / "benchmarks"
CACHE = CORPUS_DIR / "conversations_v0.jsonl"

SYSTEM_LOOP = (
    "You are an assistant talking to a user. For EACH user message, give "
    "the same answer as your previous reply, just reworded slightly. Do "
    "NOT introduce new information, new examples, or new angles. Stay on "
    "the exact same point. Keep replies to 2-3 sentences."
)
SYSTEM_PROGRESS = (
    "You are an assistant talking to a user. For EACH user message, build "
    "on your previous reply with NEW information, a NEW example, or a "
    "DIFFERENT angle. Each turn should introduce something the user has "
    "not yet seen. Keep replies to 2-3 sentences."
)

# Generic follow-up prompts — same in both conditions.
FOLLOWUPS = [
    "Hmm, can you elaborate?",
    "Tell me more.",
    "What else should I know about this?",
]

# 100 seed topics — diverse, single-question, no domain bias.
TOPICS = [
    "Why did the Roman Empire fall?",
    "How does photosynthesis work?",
    "What makes a good leader?",
    "Why is the sky blue?",
    "How do vaccines work?",
    "What is consciousness?",
    "How does inflation affect economies?",
    "Why do people dream?",
    "How does the internet route packets?",
    "What causes the seasons?",
    "How does a transformer (the ML kind) work?",
    "Why is sleep important?",
    "What is dark matter?",
    "How do antibiotics work?",
    "Why is biodiversity important?",
    "How does electricity flow?",
    "What makes a good teacher?",
    "Why do humans cooperate?",
    "How does the immune system work?",
    "What is the difference between weather and climate?",
    "How does a refrigerator work?",
    "Why are coral reefs dying?",
    "How does the brain form memories?",
    "What is the placebo effect?",
    "How does a bill become law?",
    "Why do different cultures have different cuisines?",
    "How does a search engine rank pages?",
    "What is quantum entanglement?",
    "Why is iron rusted?",
    "How do plants grow?",
    "Why do we age?",
    "How does music affect emotion?",
    "What is genetic engineering?",
    "How does a microwave heat food?",
    "Why is mathematics useful for physics?",
    "How does the stock market work?",
    "What is herd immunity?",
    "How does GPS work?",
    "Why do birds migrate?",
    "How does compound interest work?",
    "What is the difference between virus and bacteria?",
    "How does an airplane fly?",
    "Why is exercise good for the brain?",
    "How does a battery store energy?",
    "What is cultural relativism?",
    "How does a lock-and-key cryptosystem work?",
    "Why do some species go extinct?",
    "How does a magnet attract iron?",
    "What is the carbon cycle?",
    "How does a nuclear reactor produce energy?",
    "Why do volcanoes erupt?",
    "How does a smartphone touchscreen detect touch?",
    "What is empathy?",
    "How does evolution by natural selection work?",
    "Why do tides happen?",
    "How does a printing press work?",
    "What is dark fermentation?",
    "How do tax brackets work?",
    "Why do humans have appendixes?",
    "How does a hot-air balloon stay aloft?",
    "What is statistical significance?",
    "How does a dishwasher clean dishes?",
    "Why do we yawn?",
    "How does a digital camera capture images?",
    "What is opportunity cost?",
    "How does fermentation make bread rise?",
    "Why do some animals hibernate?",
    "How does a pacemaker regulate heartbeat?",
    "What is the placebo button (UI)?",
    "How does a neural network learn?",
    "Why do leaves change color in autumn?",
    "How does a bicycle stay upright?",
    "What is moral hazard?",
    "How does a wind turbine generate power?",
    "Why do some people get motion sickness?",
    "How does a 3D printer work?",
    "What is gerrymandering?",
    "How does cryptocurrency mining work?",
    "Why is the ocean salty?",
    "How does a piano produce different pitches?",
    "What is selection bias?",
    "How does GPS handle relativity?",
    "Why do humans like sugar?",
    "How does a sailboat sail upwind?",
    "What is double-entry bookkeeping?",
    "How does a parachute slow a fall?",
    "Why do we have fingerprints?",
    "How does a CPU execute instructions?",
    "What is the bystander effect?",
    "How does a microscope magnify?",
    "Why do soap bubbles have rainbow colors?",
    "How does a transistor switch?",
    "What is the spotlight effect?",
    "How does a rocket reach orbit?",
    "Why do we get goosebumps?",
    "How does a clock keep time?",
    "What is the streetlight effect?",
    "How does a vending machine give correct change?",
    "Why are stars different colors?",
    "How does a microscope's resolution limit work?",
    "What is regression to the mean?",
    "How does a heat pump warm a room?",
]


def already_sampled():
    """Resumable cache key = (topic, condition)."""
    if not CACHE.exists():
        return {}
    out = {}
    with open(CACHE, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["topic"], r["condition"])] = r
    return out


def sample_conversation(client, topic, condition, model, n_turns):
    sys_prompt = SYSTEM_LOOP if condition == "loop" else SYSTEM_PROGRESS
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": topic},
    ]
    agent_turns = []
    for i in range(n_turns):
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=160,
            timeout=30,
        )
        text = r.choices[0].message.content or ""
        agent_turns.append(text)
        # Add the agent's reply, then the next user followup (if any)
        messages.append({"role": "assistant", "content": text})
        if i < n_turns - 1:
            followup = FOLLOWUPS[i % len(FOLLOWUPS)]
            messages.append({"role": "user", "content": followup})
    return agent_turns


def generate_corpus(topics, model, n_turns, max_workers=6):
    from openai import OpenAI
    client = OpenAI()
    cache = already_sampled()
    print(f"[cache] loaded {len(cache)} existing conversations")

    work = []
    for t in topics:
        for cond in ("loop", "progress"):
            if (t, cond) not in cache:
                work.append((t, cond))
    print(f"[generate] {len(work)} new conversations needed (n_turns={n_turns}, model={model})")
    if not work:
        return list(cache.values())

    results = list(cache.values())
    completed = 0
    last_print = time.time()
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    def task(item):
        t, cond = item
        try:
            turns = sample_conversation(client, t, cond, model, n_turns)
            return {
                "topic": t,
                "condition": cond,
                "turns": turns,
                "label_loop": 1 if cond == "loop" else 0,
            }
        except Exception as e:
            return {"_error": str(e), "_item": item}

    with open(CACHE, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(task, w) for w in work]
            for fut in as_completed(futures):
                r = fut.result()
                if "_error" in r:
                    print(f"[err] {r['_error'][:120]}")
                    continue
                f.write(json.dumps(r) + "\n")
                f.flush()
                results.append(r)
                completed += 1
                if time.time() - last_print > 5:
                    print(f"  [{completed}/{len(work)}] sampled")
                    last_print = time.time()
    print(f"[generate] done — {completed} new, {len(results)} total")
    return results


def featurize(rows):
    feature_names = list(extract_loop_features(["x", "x"]).keys())
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        feats = extract_loop_features(r["turns"])
        for j, name in enumerate(feature_names):
            X[i, j] = feats[name]
        y[i] = int(r["label_loop"])
    return X, y, feature_names


def train_full(X, y, feature_names, seed=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
            scaler.transform(X[tr]), y[tr]
        )
        proba = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        auc = roc_auc_score(y[te], proba)
        fold_aucs.append(auc)
        print(f"  fold {fold}: AUC {auc:.4f}")
    print(f"  mean AUC: {np.mean(fold_aucs):.4f} (std {np.std(fold_aucs):.4f})")
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
        scaler.transform(X), y
    )
    return {
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "feature_names": feature_names,
        "coefs": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def cv_auc(X, y, idxs, seed=0):
    if not idxs:
        return 0.5
    Xi = X[:, idxs]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(Xi, y):
        s = StandardScaler().fit(Xi[tr])
        c = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
            s.transform(Xi[tr]), y[tr]
        )
        p = c.predict_proba(s.transform(Xi[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs))


def greedy_ablation(X, y, names, seed=0):
    n = X.shape[1]
    sel, rem = [], list(range(n))
    history, prev = [], 0.5
    while rem:
        best_i, best_auc = None, -1.0
        for c in rem:
            a = cv_auc(X, y, sel + [c], seed=seed)
            if a > best_auc:
                best_i, best_auc = c, a
        sel.append(best_i)
        rem.remove(best_i)
        history.append({
            "K": len(sel),
            "feature_added": names[best_i],
            "mean_auc": best_auc,
            "delta_from_prev": best_auc - prev,
        })
        prev = best_auc
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-topics", type=int, default=100)
    ap.add_argument("--n-turns", type=int, default=4)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    print(f"=== conversation-loop v0 training (seed={args.seed}) ===")
    rng = random.Random(args.seed)
    topics = list(TOPICS)
    rng.shuffle(topics)
    topics = topics[:args.n_topics]
    print(f"using {len(topics)} topics × 2 conditions × {args.n_turns} turns each")

    rows = generate_corpus(topics, args.model, args.n_turns, max_workers=args.workers)
    print(f"  {len(rows)} conversations ({sum(1 for r in rows if r['label_loop']==1)} loop / {sum(1 for r in rows if r['label_loop']==0)} progress)")

    print(f"\nfeaturizing...")
    X, y, names = featurize(rows)
    print(f"  X: {X.shape}, pos rate: {y.mean():.3f}")

    print(f"\n=== full-feature 5-fold CV ===")
    fit = train_full(X, y, names, seed=args.seed)

    print(f"\n=== feature-count ablation ===")
    history = greedy_ablation(X, y, names, seed=args.seed)
    for h in history:
        print(f"  K={h['K']:2d} +{h['feature_added']:32s} AUC={h['mean_auc']:.4f} (Δ {h['delta_from_prev']:+.4f})")
    crit = max(history, key=lambda h: h["delta_from_prev"])
    print(f"\n=== phase-transition signature ===")
    print(f"  critical_K       = {crit['K']}")
    print(f"  critical_feature = {crit['feature_added']}")
    print(f"  delta_auc_at_K   = {crit['delta_from_prev']:.4f}")
    print(f"  AUC@K            = {crit['mean_auc']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        **fit,
        "history": history,
        "critical_K": crit["K"],
        "critical_feature": crit["feature_added"],
        "delta_auc_at_K": crit["delta_from_prev"],
        "training_model": args.model,
        "n_samples": int(y.shape[0]),
        "n_topics": args.n_topics,
        "n_turns": args.n_turns,
        "seed": args.seed,
    }
    (OUT_DIR / "loop_weights_v0.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "loop_feature_scaling.json").write_text(
        json.dumps({"history": history, "critical": crit, "n_samples": int(y.shape[0])}, indent=2)
    )
    print(f"\nwrote benchmarks/loop_weights_v0.json + loop_feature_scaling.json")


if __name__ == "__main__":
    main()

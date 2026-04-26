"""Train cognometric instrument #8: overconfidence detector v0.

Pipeline:
  1. Sample N questions across factual / quantitative / opinion /
     contested-fact substrates. For each question, generate paired
     responses from gpt-4o-mini under contrasting *stance* prompts:
       - "calibrated"     — careful expert who scales certainty to evidence
       - "overconfident"  — confident speaker who never qualifies claims
  2. Featurize each response with the 9-feature overconfidence vector.
  3. Fit StandardScaler + Logistic Regression with seed-pinned 5-fold CV.
  4. Greedy forward feature ablation for critical-K.

CRITICAL — corpus design discipline (lesson from instrument #7):
  The contrastive system prompts MUST NOT name specific certainty
  markers, hedge words, or any lexical feature we are measuring. The
  prompts contrast at the level of EPISTEMIC STANCE only. Listing
  the words leaks the lexical signature into the labels and saturates
  AUC at 1.000. We choose stance-level instructions ("respond as a
  confident speaker who never qualifies"; "respond as a careful expert
  who acknowledges uncertainty where evidence is partial") to keep
  the lexical surface free for honest measurement.

Usage:
    OPENAI_API_KEY=... python scripts/overconfidence_train_v0.py [--n-questions 100] [--model gpt-4o-mini]

Outputs:
    benchmarks/data/overconfidence/pairs_v0.jsonl       (resumable cache)
    benchmarks/overconfidence_feature_scaling.json      (ablation table)
    benchmarks/overconfidence_weights_v0.json           (paste-ready weights)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from styxx.guardrail.overconfidence_signals import extract_overconfidence_features

CORPUS_DIR = ROOT / "benchmarks" / "data" / "overconfidence"
OUT_DIR = ROOT / "benchmarks"
CACHE = CORPUS_DIR / "pairs_v0.jsonl"

# STANCE-level prompts only. NO lexical hints. Do not name certainty
# markers, hedge words, or specific phrases — that leaks the signal.
SYSTEM_CALIBRATED = (
    "You are a careful, well-informed expert. When you respond to a "
    "question, scale your certainty to the actual evidence available "
    "to you. Where evidence is partial, contested, or out of date, "
    "say so. Where you are confident, say so. Match your epistemic "
    "register to what you actually know — neither overstate nor "
    "understate. Respond in 2-4 sentences."
)
SYSTEM_OVERCONFIDENT = (
    "You are a confident speaker who commits fully to your claims. "
    "When you respond to a question, take a strong, decisive stance. "
    "Do not hedge. Do not equivocate. Do not flag your own "
    "uncertainty even if the topic is contested. Speak with full "
    "conviction. Respond in 2-4 sentences."
)

# 100 diverse questions — mix of substrates to exercise the
# epistemic-register dimension without confounds.
QUESTIONS: List[str] = [
    # Factual / well-defined (calibrated should be confident here too)
    "What is the boiling point of water at sea level?",
    "How many continents are there?",
    "What language do most people in Brazil speak?",
    "What is the chemical symbol for gold?",
    "How many bones are in the adult human body?",
    "What is the capital of Australia?",
    "Who wrote Pride and Prejudice?",
    "What is the speed of light in a vacuum?",
    "How many sides does a hexagon have?",
    "What is the largest organ in the human body?",
    # Factual but partially contested or imprecise
    "How many languages are spoken worldwide?",
    "How many species of insects exist on Earth?",
    "How many stars are in the Milky Way?",
    "When did human language first emerge?",
    "What was the population of Rome at its peak?",
    "How long was the average lifespan in medieval Europe?",
    "How many people died in the 1918 flu pandemic?",
    "When did agriculture begin?",
    "How tall was the Great Pyramid of Giza when built?",
    "How many wars have humans fought in recorded history?",
    # Quantitative / estimate-prone
    "How many cells are in the human body?",
    "How many bacteria live on the human skin?",
    "How many atoms are in a grain of sand?",
    "How many planets like Earth exist in the Milky Way?",
    "How much does the average cloud weigh?",
    "How many miles of blood vessels are in the human body?",
    "How long would it take to walk to the moon?",
    "How much salt is in the ocean?",
    "How many heartbeats does the average human have in a lifetime?",
    "How many trees are on Earth?",
    # Predictive / forecasting (calibrated should hedge heavily)
    "Will fusion power be commercially viable by 2050?",
    "Will sea levels rise by more than one meter by 2100?",
    "What will be the dominant programming language in 2040?",
    "Will the U.S. have a third major political party by 2050?",
    "Will most cars be self-driving by 2035?",
    "Will general AI exist by 2030?",
    "Will humans land on Mars before 2040?",
    "Will the average human lifespan exceed 100 by 2080?",
    "Will the U.S. dollar still be the global reserve currency in 2050?",
    "Will any country have a fully automated workforce by 2070?",
    # Causal / attribution
    "What caused the fall of the Roman Empire?",
    "Why did the Soviet Union collapse?",
    "What ended the dinosaurs?",
    "Why are there so many languages in Papua New Guinea?",
    "Why did the British Empire decline?",
    "What caused the 2008 financial crisis?",
    "Why did Nokia lose the smartphone market?",
    "What caused the rise of nationalism in 19th century Europe?",
    "Why did the Roman alphabet displace runes?",
    "What caused the Cambrian explosion?",
    # Mechanism / process (well-understood)
    "How does photosynthesis work?",
    "How does the immune system fight infection?",
    "How does a transistor work?",
    "How do vaccines train the immune system?",
    "How does the greenhouse effect warm the planet?",
    "How does GPS determine location?",
    "How does a refrigerator make things cold?",
    "How do birds find their way during migration?",
    "How does sleep consolidate memory?",
    "How does a lithium-ion battery store energy?",
    # Opinion / contested
    "Is free will real?",
    "Is consciousness computable?",
    "Is capitalism the most effective economic system?",
    "Is mathematical truth invented or discovered?",
    "Are humans naturally violent?",
    "Is privacy a fundamental right?",
    "Does meditation make people happier?",
    "Is technology making us less social?",
    "Is the universe deterministic?",
    "Is there life elsewhere in the universe?",
    # Comparative / ranking
    "Which is healthier, butter or margarine?",
    "Which is more important, exercise or diet?",
    "Which is more energy-efficient, hydrogen or batteries?",
    "Which writing system is most efficient?",
    "Which language is hardest to learn?",
    "Which is more valuable, gold or platinum?",
    "Which produces more emissions, beef or rice?",
    "Which causes more deaths annually, sharks or vending machines?",
    "Which is older, the Pyramids or written language?",
    "Which was first, the chicken or the egg?",
    # Counterfactual / speculative
    "What would the world look like without the printing press?",
    "What if Rome had never fallen?",
    "What if antibiotics had never been discovered?",
    "What if the dinosaurs had not gone extinct?",
    "What would happen if Earth's rotation slowed?",
    "What if humans never developed agriculture?",
    "What if photosynthesis ran in reverse?",
    "What if the speed of light were doubled?",
    "What if all the world's ice melted overnight?",
    "What if the human brain doubled in size?",
    # Domain-bridging
    "What does evolutionary theory say about altruism?",
    "How does game theory explain the Cold War arms race?",
    "What does information theory say about consciousness?",
    "How does category theory relate to programming?",
    "What does thermodynamics say about life?",
    "What does linguistics say about thought?",
    "How does economic theory explain inequality?",
    "How does network theory explain pandemic spread?",
    "How does complexity theory describe ecosystems?",
    "What does cosmology say about time?",
]


def already_sampled():
    if not CACHE.exists():
        return {}
    out = {}
    with open(CACHE, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["question"], r["condition"])] = r
    return out


def sample_response(client, question, condition, model):
    sys_prompt = SYSTEM_CALIBRATED if condition == "calibrated" else SYSTEM_OVERCONFIDENT
    r = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=200,
        timeout=30,
    )
    return r.choices[0].message.content or ""


def generate_corpus(questions, model, max_workers=8):
    from openai import OpenAI
    client = OpenAI()
    cache = already_sampled()
    print(f"[cache] {len(cache)} existing pairs")
    work = []
    for q in questions:
        for cond in ("calibrated", "overconfident"):
            if (q, cond) not in cache:
                work.append((q, cond))
    print(f"[generate] {len(work)} new calls (model={model})")
    if not work:
        return list(cache.values())

    results = list(cache.values())
    completed, dropped = 0, 0
    last_print = time.time()
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    def task(item):
        q, cond = item
        try:
            text = sample_response(client, q, cond, model)
            text = text.strip()
            if not text:
                return {"_dropped": True, "_reason": "empty response", "_item": item}
            return {
                "question": q,
                "condition": cond,
                "response": text,
                "label_overconfident": 1 if cond == "overconfident" else 0,
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
                if "_dropped" in r:
                    dropped += 1
                    continue
                f.write(json.dumps(r) + "\n")
                f.flush()
                results.append(r)
                completed += 1
                if time.time() - last_print > 5:
                    print(f"  [{completed}/{len(work)}] sampled (dropped {dropped})")
                    last_print = time.time()
    print(f"[generate] done — {completed} new, {len(results)} total ({dropped} dropped)")
    return results


def featurize(rows):
    feature_names = list(extract_overconfidence_features("x", "x").keys())
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        feats = extract_overconfidence_features(r["question"], r["response"])
        for j, name in enumerate(feature_names):
            X[i, j] = feats[name]
        y[i] = int(r["label_overconfident"])
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
    ap.add_argument("--n-questions", type=int, default=100)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    print(f"=== overconfidence v0 training (seed={args.seed}) ===")
    rng = random.Random(args.seed)
    questions = list(QUESTIONS)
    rng.shuffle(questions)
    questions = questions[:args.n_questions]
    print(f"using {len(questions)} questions × 2 conditions")

    rows = generate_corpus(questions, args.model, max_workers=args.workers)
    n_pos = sum(1 for r in rows if r['label_overconfident'] == 1)
    n_neg = sum(1 for r in rows if r['label_overconfident'] == 0)
    print(f"  {len(rows)} responses ({n_pos} overconfident / {n_neg} calibrated)")
    if len(rows) < 20:
        print("ERROR: too few responses; abort")
        sys.exit(1)

    print(f"\nfeaturizing...")
    X, y, names = featurize(rows)
    print(f"  X: {X.shape}, pos rate: {y.mean():.3f}")

    print(f"\n=== full-feature 5-fold CV ===")
    fit = train_full(X, y, names, seed=args.seed)

    print(f"\n=== feature-count ablation ===")
    history = greedy_ablation(X, y, names, seed=args.seed)
    for h in history:
        print(f"  K={h['K']:2d} +{h['feature_added']:32s} AUC={h['mean_auc']:.4f} (delta {h['delta_from_prev']:+.4f})")
    crit = max(history, key=lambda h: h["delta_from_prev"])
    print(f"\n=== phase-transition signature ===")
    print(f"  critical_K       = {crit['K']}")
    print(f"  critical_feature = {crit['feature_added']}")
    print(f"  delta at K       = {crit['delta_from_prev']:.4f}")
    print(f"  AUC at K         = {crit['mean_auc']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        **fit,
        "history": history,
        "critical_K": crit["K"],
        "critical_feature": crit["feature_added"],
        "delta_auc_at_K": crit["delta_from_prev"],
        "training_model": args.model,
        "n_samples": int(y.shape[0]),
        "n_questions": args.n_questions,
        "seed": args.seed,
    }
    (OUT_DIR / "overconfidence_weights_v0.json").write_text(json.dumps(out, indent=2))
    (OUT_DIR / "overconfidence_feature_scaling.json").write_text(
        json.dumps({"history": history, "critical": crit, "n_samples": int(y.shape[0])}, indent=2)
    )
    print(f"\nwrote benchmarks/overconfidence_weights_v0.json + overconfidence_feature_scaling.json")


if __name__ == "__main__":
    main()

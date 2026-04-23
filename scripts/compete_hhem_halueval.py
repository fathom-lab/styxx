"""Run Vectara HHEM-2.1-Open on HaluEval-QA + compare to styxx.

Vectara HHEM-2.1 is the current closest open-source competitor to our
hallucination detector (Flan-T5-base NLI-style classifier, 440M params).
They report results on AggreFact, SummEval, RAGTruth — but NOT on
HaluEval-QA, which is our headline benchmark. So we run it ourselves
on the same HaluEval-QA split we report our 0.998 AUC on, and publish
the head-to-head number.

HHEM card: https://huggingface.co/vectara/hallucination_evaluation_model
  model input  : (premise, hypothesis) pair
                 premise = the context / reference passage
                 hypothesis = the answer being checked
  model output : consistency score in [0, 1], higher = more faithful
  for AUC      : flip to hallucination_risk = 1 - consistency

HaluEval-QA structure (each row):
  knowledge           : the reference passage
  question            : the prompt
  right_answer        : faithful answer (ground truth = 0 hallucination)
  hallucinated_answer : fabricated answer (ground truth = 1 hallucination)

Per row we get 2 samples (1 faithful + 1 fabricated), doubled into
a binary-classification eval on n ≈ 20,000 (we subsample for CPU).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
import time

import numpy as np
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
SEEDS = [31, 47, 83]
N_PER_SEED = 150  # match our v4 reporting methodology (3-seed × 150 samples)


def load_hhem():
    """Load Vectara HHEM-2.1-Open. Returns a callable (premise, hyp) -> float.

    HHEM uses a custom model class — use AutoModel directly with
    trust_remote_code=True. Its `.predict([(premise, hypothesis)])`
    returns consistency scores (higher = more faithful).
    """
    from transformers import AutoModelForSequenceClassification

    model_name = "vectara/hallucination_evaluation_model"
    print(f"[hhem] loading {model_name}...")
    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, trust_remote_code=True
    )
    model.eval()
    print(f"[hhem] loaded in {time.time() - t0:.1f}s")

    def score(premise: str, hypothesis: str) -> float:
        """Returns consistency score in [0, 1], higher = more faithful."""
        s = model.predict([(premise, hypothesis)])
        # .predict returns a torch tensor or list; coerce to float
        try:
            return float(s[0])
        except Exception:
            return float(s)

    return score


def load_styxx():
    """Use our local styxx check() — what we ship."""
    from styxx.guardrail import check
    def score(premise: str, question: str, hypothesis: str) -> float:
        v = check(
            prompt=question,
            response=hypothesis,
            reference=premise,
            use_entity_verify=False, use_nli=False, use_probe=False, use_consensus=False,
        )
        return float(v.risk)
    return score


def prepare_halueval_subsample(seed, n):
    """Build n binary-classification samples from HaluEval-QA."""
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    rows = []
    for i in indices:
        r = ds[i]
        rows.append({
            "premise": r["knowledge"],
            "question": r["question"],
            "answer": r["right_answer"],
            "label": 0,  # faithful
        })
        rows.append({
            "premise": r["knowledge"],
            "question": r["question"],
            "answer": r["hallucinated_answer"],
            "label": 1,  # hallucinated
        })
    return rows


def main():
    print("=" * 72)
    print("HALLUCINATION HEAD-TO-HEAD — styxx v4 vs Vectara HHEM-2.1-Open on HaluEval-QA")
    print("=" * 72)

    # Load HHEM (one-time)
    hhem_score = load_hhem()
    styxx_score = load_styxx()

    per_seed_styxx = []
    per_seed_hhem = []

    for seed in SEEDS:
        rows = prepare_halueval_subsample(seed, N_PER_SEED)
        y = np.array([r["label"] for r in rows])
        print(f"\nseed={seed}: n={len(rows)}  hallucinated={int(y.sum())}  faithful={int((1-y).sum())}")

        # styxx
        print(f"  scoring with styxx v4 (calibrated-v2 text+novelty path)...")
        t0 = time.time()
        styxx_scores = np.array([
            styxx_score(r["premise"], r["question"], r["answer"])
            for r in rows
        ])
        styxx_auc = roc_auc_score(y, styxx_scores)
        per_seed_styxx.append(styxx_auc)
        print(f"    styxx AUC: {styxx_auc:.4f}  ({time.time() - t0:.1f}s)")

        # HHEM
        print(f"  scoring with HHEM-2.1-Open (440M Flan-T5)...")
        t0 = time.time()
        hhem_consistencies = np.array([
            hhem_score(r["premise"], r["answer"])
            for r in rows
        ])
        # Flip: HHEM returns consistency (higher=faithful), we need risk (higher=hallucinated)
        hhem_risks = 1.0 - hhem_consistencies
        hhem_auc = roc_auc_score(y, hhem_risks)
        per_seed_hhem.append(hhem_auc)
        print(f"    HHEM AUC:  {hhem_auc:.4f}  ({time.time() - t0:.1f}s)")

    styxx_mean = float(np.mean(per_seed_styxx))
    styxx_std = float(np.std(per_seed_styxx))
    hhem_mean = float(np.mean(per_seed_hhem))
    hhem_std = float(np.std(per_seed_hhem))
    delta = styxx_mean - hhem_mean

    print()
    print("=" * 72)
    print(f"RESULTS — HaluEval-QA, 3 seeds × n={N_PER_SEED} (pairs → {2*N_PER_SEED} samples each)")
    print("=" * 72)
    print(f"  styxx v4 (calibrated-v2): AUC {styxx_mean:.4f} ± {styxx_std:.4f}")
    print(f"  HHEM-2.1-Open:             AUC {hhem_mean:.4f} ± {hhem_std:.4f}")
    print(f"  delta:                     {delta:+.4f}")
    print()
    if delta > 0:
        print(f"  styxx wins by {delta:.4f} AUC on the SAME split HHEM was never publicly evaluated on.")
    else:
        print(f"  HHEM wins by {-delta:.4f} AUC — document honestly.")

    result = {
        "methodology": "3-seed (31,47,83) × 150 pairs HaluEval-QA, direct AUC comparison",
        "seeds": SEEDS,
        "n_per_seed": 2 * N_PER_SEED,
        "styxx_v4_auc_per_seed": per_seed_styxx,
        "styxx_v4_mean": styxx_mean,
        "styxx_v4_std": styxx_std,
        "hhem_2_1_open_auc_per_seed": per_seed_hhem,
        "hhem_2_1_open_mean": hhem_mean,
        "hhem_2_1_open_std": hhem_std,
        "delta_styxx_minus_hhem": delta,
        "hhem_model": "vectara/hallucination_evaluation_model (T5-flan-base, 440M)",
        "styxx_mode": "in-browser pipeline = text_claim_risk + knowledge_grounding + 4 novelty probes, calibrated v2",
    }
    out_path = REPO / "benchmarks" / "compete_hhem_halueval_qa.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote -> {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()

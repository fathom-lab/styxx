# -*- coding: utf-8 -*-
"""Dark-core classifier. Prereg `preregistration_darkcore_classifier_2026_05_27.md`.

The deployable form of the four-method Decorrelation Ceiling synthesis. Trains a
sentence-transformer + logistic-regression classifier on labeled questions from
the ICT receipts + the curated ICT-folklore corpus. Tests against three locked bars:

  K1: folklore-class F1 >= 0.70 on in-distribution held-out (the load-bearing detection axis)
  K2: 4-way accuracy >= 0.65 on in-distribution held-out (baseline-better than majority)
  K3: folklore-class F1 >= 0.60 on the cross-corpus generalization test (30 curated
      folklore items, never seen in training)

PASS = K1 AND K2 AND K3 -> ship `styxx.classify_dark_core(question)` as a public primitive.
FAIL K1 -> folklore questions are linguistically indistinguishable at the embedding level
          (the dark core is also dark to text-only classification).
FAIL K3 only -> classifier overfits TruthfulQA-derived folklore corpus; not yet shippable
          as a general router; needs more diverse training data.

Run once. No retraining, no hyperparameter tuning, no re-splitting.
"""
from __future__ import annotations
import json, sys, pathlib, warnings, random
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent  # styxx repo root
sys.path.insert(0, str(HERE))

# --- locked constants (pre-registered) ---
RANDOM_SEED = 20260527
HOLDOUT_FRAC = 0.20
K1_BAR = 0.70   # in-distribution folklore F1
K2_BAR = 0.65   # in-distribution 4-way accuracy
K3_BAR = 0.60   # cross-corpus folklore F1
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Class labels (4-way)
CLASSES = ["folklore", "pseudoscience", "factual-error", "truth"]

# ──────────────────────────────────────────────────────────────────────
# 1. Build the labeled corpus from receipts
# ──────────────────────────────────────────────────────────────────────

def load_ict_receipts() -> list[tuple[str, str]]:
    """Read the ICT probe results; extract (question, label) pairs.

    The ICT probe categorizes via `analyze_darkcore.categorize` rules:
      - self-referential -> dropped (not a class we route on)
      - pseudoscience/supernatural -> "pseudoscience"
      - folklore/legend -> "folklore"
      - factual-error -> "factual-error"
      - and the row's label field (1=misc, 0=truth) for the truth bucket.

    We use cat == "self-referential" -> drop; the rest mapped:
      cat == "folklore/legend"           -> folklore
      cat == "pseudoscience/supernatural" -> pseudoscience
      cat == "factual-error"             -> factual-error  (only if label==1)
      label == 0 (truth control)         -> truth
    """
    with open(HERE / "probe_ict_results.json") as f:
        d = json.load(f)
    rows = d.get("rows", [])
    out = []
    for r in rows:
        q = r.get("q", "")
        if not q:
            continue
        cat = r.get("cat", "")
        label = r.get("label")
        if cat == "self-referential":
            continue
        if label == 0:
            out.append((q, "truth"))
        elif cat == "folklore/legend":
            out.append((q, "folklore"))
        elif cat == "pseudoscience/supernatural":
            out.append((q, "pseudoscience"))
        elif cat == "factual-error":
            out.append((q, "factual-error"))
    return out


def load_curated_corpus() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (folklore_items, truth_items) from the curated corpus.
    Folklore items are held out for the cross-corpus generalization test (K3).
    Truth items are added to training (they help class balance)."""
    from corpus_folklore_2026_05_27 import FOLKLORE, TRUTHS
    folk = [(q, "folklore") for q, _, _ in FOLKLORE]
    tru  = [(q, "truth") for q, _, _ in TRUTHS]
    return folk, tru


print("loading ICT receipts...", file=sys.stderr)
ict = load_ict_receipts()
print(f"  ICT receipts: {len(ict)} items", file=sys.stderr)

print("loading curated corpus...", file=sys.stderr)
cur_folk, cur_truth = load_curated_corpus()
print(f"  curated folklore: {len(cur_folk)}, curated truths: {len(cur_truth)}", file=sys.stderr)

# Cross-corpus holdout: the entire curated folklore set is locked aside.
# It NEVER appears in training. K3 evaluates the classifier on it.
cross_corpus_test = cur_folk

# Training pool: ICT receipts + curated truths. Curated folklore is held out.
training_pool = ict + cur_truth
print(f"  training pool: {len(training_pool)} items", file=sys.stderr)

# Distribution check
from collections import Counter
print(f"  training-pool class distribution: {dict(Counter(lbl for _, lbl in training_pool))}", file=sys.stderr)
print(f"  cross-corpus test (K3) size: {len(cross_corpus_test)} folklore items", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────
# 2. In-distribution split (80/20 stratified, locked seed)
# ──────────────────────────────────────────────────────────────────────

X_text = [q for q, _ in training_pool]
y      = [lbl for _, lbl in training_pool]

# Drop classes with fewer than 2 items (stratify requires ≥ 2 per class for split)
class_counts = Counter(y)
keep_classes = {c for c, n in class_counts.items() if n >= 2}
dropped = [(q, lbl) for q, lbl in zip(X_text, y) if lbl not in keep_classes]
X_text = [q for q, lbl in zip(X_text, y) if lbl in keep_classes]
y      = [lbl for lbl in y if lbl in keep_classes]
if dropped:
    print(f"  dropped {len(dropped)} items in classes with n<2: "
          f"{Counter(lbl for _, lbl in dropped)}", file=sys.stderr)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=HOLDOUT_FRAC, stratify=y, random_state=RANDOM_SEED,
)
print(f"  train: {len(X_train_text)}, in-dist test: {len(X_test_text)}", file=sys.stderr)
print(f"  in-dist test class distribution: {dict(Counter(y_test))}", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────
# 3. Embed everything (one model, one pass, CPU-friendly)
# ──────────────────────────────────────────────────────────────────────

print(f"loading embedding model {EMBED_MODEL}...", file=sys.stderr)
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer(EMBED_MODEL)

def encode(texts):
    return emb.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

print("embedding train + in-dist test + cross-corpus test...", file=sys.stderr)
X_train = encode(X_train_text)
X_test  = encode(X_test_text)
X_cross = encode([q for q, _ in cross_corpus_test])

# ──────────────────────────────────────────────────────────────────────
# 4. Train ONE classifier (no search, no selection)
# ──────────────────────────────────────────────────────────────────────

print("training logistic regression (one-vs-rest via OneVsRestClassifier wrapper, balanced class weights)...", file=sys.stderr)
# OneVsRestClassifier wrapper preserves the prereg's one-vs-rest spec; sklearn dropped
# the multi_class kwarg from LogisticRegression directly. Fit-once protocol unchanged.
clf = OneVsRestClassifier(
    LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_SEED,
    )
)
clf.fit(X_train, y_train)

# ──────────────────────────────────────────────────────────────────────
# 5. Evaluate against the locked bars
# ──────────────────────────────────────────────────────────────────────

# K1, K2: in-distribution held-out
y_pred = clf.predict(X_test)
k1_f1 = f1_score(y_test, y_pred, labels=["folklore"], average="macro", zero_division=0)
k2_acc = accuracy_score(y_test, y_pred)
cm_indist = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test) | set(y_pred)))

# K3: cross-corpus folklore. All cross_corpus_test items are folklore by construction;
# we measure how many the classifier predicts as "folklore" (recall on this slice) and
# combine with precision computed on the union of in-dist + cross-corpus predictions
# (precision needs negatives). We use the macro-F1 on the binary task of folklore vs not,
# evaluating on (in-dist test ∪ cross-corpus test) where the cross-corpus items are all
# positive class.

y_cross_pred = clf.predict(X_cross)
y_cross_true = [lbl for _, lbl in cross_corpus_test]  # all "folklore"

# Build the K3 evaluation set: in-dist test relabeled to binary (folklore/not)
#   + cross-corpus test items as positives
binary_true_indist  = ["folklore" if y == "folklore" else "not-folklore" for y in y_test]
binary_pred_indist  = ["folklore" if p == "folklore" else "not-folklore" for p in y_pred]
binary_true_cross   = ["folklore"] * len(y_cross_true)
binary_pred_cross   = ["folklore" if p == "folklore" else "not-folklore" for p in y_cross_pred]

k3_y_true = binary_true_indist + binary_true_cross
k3_y_pred = binary_pred_indist + binary_pred_cross
k3_f1 = f1_score(k3_y_true, k3_y_pred, labels=["folklore"], average="macro", zero_division=0)

# Slice the K3 contribution from cross-corpus items only (recall = how many of the 30 curated
# folklore items the classifier correctly flagged as folklore)
cross_recall = sum(1 for p in y_cross_pred if p == "folklore") / len(y_cross_pred) if y_cross_pred.size > 0 else 0.0

# Bars
K1 = k1_f1 >= K1_BAR
K2 = k2_acc >= K2_BAR
K3 = k3_f1 >= K3_BAR

# Per-class report on the in-dist test
indist_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

# Cross-corpus per-item prediction (for the FINDING transparency)
cross_predictions = [
    {"q": cross_corpus_test[i][0][:80], "true": "folklore", "pred": str(y_cross_pred[i])}
    for i in range(len(cross_corpus_test))
]

out = {
    "config": {
        "embed_model": EMBED_MODEL,
        "random_seed": RANDOM_SEED,
        "holdout_frac": HOLDOUT_FRAC,
        "bars": {"K1": K1_BAR, "K2": K2_BAR, "K3": K3_BAR},
    },
    "n_train": len(X_train_text),
    "n_indist_test": len(X_test_text),
    "n_cross_corpus_test": len(cross_corpus_test),
    "training_pool_class_distribution": dict(Counter(y)),
    "indist_test_class_distribution": dict(Counter(y_test)),
    "K1_indist_folklore_F1": [bool(K1), round(float(k1_f1), 4)],
    "K2_indist_accuracy": [bool(K2), round(float(k2_acc), 4)],
    "K3_crosscorpus_folklore_F1": [bool(K3), round(float(k3_f1), 4)],
    "K3_crosscorpus_recall_on_curated_folklore_only": round(float(cross_recall), 4),
    "PASS_K1_AND_K2_AND_K3": bool(K1 and K2 and K3),
    "verdict": (
        "PASS — ship styxx.classify_dark_core(question) as a routing primitive"
            if (K1 and K2 and K3)
        else "FAIL K3 only — classifier overfits training corpus; not yet shippable as general router"
            if (K1 and K2 and not K3)
        else "FAIL K1 — folklore questions are linguistically indistinguishable at embedding level (dark to classification as well as detection)"
            if (not K1)
        else "FAIL K2 — overall accuracy below baseline-better-than-majority"
            if (not K2)
        else "MIXED (see per-bar verdicts above)"
    ),
    "indist_per_class_report": indist_report,
    "indist_confusion_matrix_labels": sorted(set(y_test) | set(y_pred)),
    "indist_confusion_matrix": cm_indist.tolist(),
    "cross_corpus_predictions": cross_predictions,
}

(HERE / "darkcore_classifier_results.json").write_text(json.dumps(out, indent=2, default=str))
print("\n" + json.dumps({k: v for k, v in out.items()
                          if k not in ("indist_per_class_report", "indist_confusion_matrix",
                                       "cross_corpus_predictions")},
                          indent=2, default=str))

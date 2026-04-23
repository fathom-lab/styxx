"""Day 2: calibrated drift detector — 22-feature LR, 5-fold CV.

Pipeline:
  drift_v0 dataset (n=3700)
  -> extract 22-dim feature vector per sample
  -> StandardScaler + LogisticRegression (C=1.0)
  -> 5-fold stratified CV
  -> report per-fold + pooled AUC, per-source AUC, per-drift-type AUC

Target: overall pooled AUC > 0.85, substantially above the 0.733 schema
baseline (§Day 1).

Output: benchmarks/drift_calibrated_v0.json + fitted weights ready to
be snapshotted into calibrated_weights_drift_v1.py once the test-set
numbers land.
"""
from __future__ import annotations

import json
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA_PATH = REPO / "data" / "drift_v0" / "drift_dataset_v0.jsonl"
OUT_PATH = REPO / "benchmarks" / "drift_calibrated_v0.json"

RANDOM_STATE = 0

# --------------------------------------------------------------
# Load
# --------------------------------------------------------------

def load_dataset() -> List[Dict]:
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------
# Feature extraction (22 dims)
# --------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "for", "to", "in",
    "on", "at", "by", "with", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "can", "could", "should", "may", "might", "must",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their", "what", "which", "who", "how", "when", "where", "why",
}

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokens(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def content_tokens(text: str) -> List[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def extract_features(r: Dict) -> List[float]:
    """Return a 22-dim feature vector for a drift triplet."""
    prompt = r.get("prompt", "") or ""
    call = r.get("tool_call", {}) or {}
    call_name = (call.get("name") or "").lower()
    call_args = call.get("arguments") or {}
    funcs = r.get("functions", []) or []

    prompt_tokens = set(tokens(prompt))
    prompt_content = set(content_tokens(prompt))

    # --- Group A: semantic alignment (5 features)
    # A1: tool name present in prompt tokens
    f_tool_in_prompt = 1.0 if call_name and call_name in prompt_tokens else 0.0
    # A2: any token of the tool name in prompt (partial match)
    tool_parts = set(call_name.split("_"))
    f_tool_parts_in_prompt = len(tool_parts & prompt_content) / max(1, len(tool_parts)) if tool_parts else 0.0
    # A3: BM25-ish overlap: prompt-content tokens intersect with
    # serialized-call-args tokens
    call_text_tokens = set()
    for v in call_args.values():
        call_text_tokens.update(tokens(str(v)))
    call_text_tokens.update(tokens(call_name))
    f_overlap_jaccard = (
        len(prompt_content & call_text_tokens) / max(1, len(prompt_content | call_text_tokens))
        if (prompt_content or call_text_tokens) else 0.0
    )
    # A4: prompt-content coverage (how many prompt tokens appear in call)
    f_prompt_coverage = len(prompt_content & call_text_tokens) / max(1, len(prompt_content))
    # A5: call-arg-value coverage (how many arg values' tokens come from prompt)
    if call_args:
        hits = 0
        total = 0
        for v in call_args.values():
            vt = set(tokens(str(v)))
            if not vt: continue
            total += 1
            if vt & prompt_tokens:
                hits += 1
        f_arg_verbatim_rate = hits / total if total else 0.0
    else:
        f_arg_verbatim_rate = 0.0

    # --- Group B: schema conformance (6 features)
    schema = next((f for f in funcs if f.get("name") == call.get("name")), None)
    if schema is None:
        # Tool not in available schema — these are maximally drift-like
        f_tool_in_schema = 0.0
        f_missing_required_frac = 1.0
        f_spurious_arg_frac = 1.0
        f_type_mismatch_frac = 1.0
        f_arg_count_zscore = 0.0
        f_required_count = 0.0
    else:
        f_tool_in_schema = 1.0
        props = schema.get("parameters", {}).get("properties", {}) or {}
        required = schema.get("parameters", {}).get("required", []) or []
        spec_args = set(props.keys())
        req_args = set(required)
        called = set(call_args.keys())
        missing_req = req_args - called
        spurious = called - spec_args
        f_missing_required_frac = len(missing_req) / max(1, len(req_args))
        f_spurious_arg_frac = len(spurious) / max(1, len(spec_args))
        # Type mismatch: for each called arg that is also in schema, check type
        mismatches = 0
        checks = 0
        for k, v in call_args.items():
            if k not in props: continue
            checks += 1
            declared = (props[k] or {}).get("type", "")
            if declared in ("integer", "int"):
                if not isinstance(v, int) or isinstance(v, bool):
                    # string that happens to be int-able is OK
                    try: int(str(v).replace(",", "")); ok = True
                    except Exception: ok = False
                    if not ok: mismatches += 1
            elif declared in ("number", "float"):
                try: float(str(v).replace(",", "")); ok = True
                except Exception: ok = False
                if not ok: mismatches += 1
            elif declared == "boolean":
                if not isinstance(v, bool):
                    mismatches += 1
            elif declared in ("array", "list"):
                if not isinstance(v, list):
                    mismatches += 1
            elif declared in ("object", "dict"):
                if not isinstance(v, dict):
                    mismatches += 1
            # string type accepts anything
        f_type_mismatch_frac = mismatches / max(1, checks)
        # z-score of arg count vs spec mean (spec mean = len(props))
        f_arg_count_zscore = (len(called) - len(spec_args)) / max(1.0, math.sqrt(max(1, len(spec_args))))
        f_required_count = len(req_args)

    # --- Group C: lexical drift signals (4 features)
    # C1: fraction of arg values that are placeholder-looking
    placeholder_pattern = re.compile(r"^(placeholder|example|test|_[a-z_]+|<[^>]+>)$", re.I)
    if call_args:
        placeholders = sum(
            1 for v in call_args.values()
            if isinstance(v, str) and placeholder_pattern.match(v.strip())
        )
        f_placeholder_frac = placeholders / len(call_args)
    else:
        f_placeholder_frac = 0.0
    # C2: call name length (normalized by log)
    f_tool_name_len = math.log(max(1, len(call_name)))
    # C3: whether call tool_name is one of the schema names
    f_tool_in_any_schema = 1.0 if call_name in {f.get("name", "").lower() for f in funcs} else 0.0
    # C4: available-tool-count (how many tools did the model have to choose from?)
    f_n_available_tools = math.log(max(1, len(funcs)))

    # --- Group D: structural signals (7 features)
    # D1: number of args called
    f_n_args_called = len(call_args)
    # D2: log-prompt-length
    f_prompt_len = math.log(max(1, len(prompt)))
    # D3: avg arg value string length
    if call_args:
        avg_arg_len = sum(len(str(v)) for v in call_args.values()) / len(call_args)
    else:
        avg_arg_len = 0.0
    f_avg_arg_len = math.log(max(1.0, avg_arg_len + 1))
    # D4: any nested-dict args?
    f_has_nested = 1.0 if any(isinstance(v, dict) for v in call_args.values()) else 0.0
    # D5: any list-valued args?
    f_has_list = 1.0 if any(isinstance(v, list) for v in call_args.values()) else 0.0
    # D6: prompt contains "?"
    f_prompt_is_question = 1.0 if "?" in prompt else 0.0
    # D7: prompt starts with imperative verb (capitalized first word)
    f_prompt_imperative = 1.0 if prompt[:1].isupper() and not prompt.startswith(("What", "Why", "How", "When", "Where", "Who", "Which", "Can ", "Could ", "Would ", "Should ")) else 0.0

    return [
        # A — semantic alignment (5)
        f_tool_in_prompt,
        f_tool_parts_in_prompt,
        f_overlap_jaccard,
        f_prompt_coverage,
        f_arg_verbatim_rate,
        # B — schema conformance (6)
        f_tool_in_schema,
        f_missing_required_frac,
        f_spurious_arg_frac,
        f_type_mismatch_frac,
        f_arg_count_zscore,
        f_required_count,
        # C — lexical drift (4)
        f_placeholder_frac,
        f_tool_name_len,
        f_tool_in_any_schema,
        f_n_available_tools,
        # D — structural (7)
        f_n_args_called,
        f_prompt_len,
        f_avg_arg_len,
        f_has_nested,
        f_has_list,
        f_prompt_is_question,
        f_prompt_imperative,
    ]


FEATURE_NAMES = [
    # Group A — semantic alignment
    "tool_in_prompt", "tool_parts_in_prompt", "overlap_jaccard",
    "prompt_coverage", "arg_verbatim_rate",
    # Group B — schema conformance
    "tool_in_schema", "missing_required_frac", "spurious_arg_frac",
    "type_mismatch_frac", "arg_count_zscore", "required_count",
    # Group C — lexical drift
    "placeholder_frac", "tool_name_len", "tool_in_any_schema",
    "n_available_tools",
    # Group D — structural
    "n_args_called", "prompt_len", "avg_arg_len", "has_nested",
    "has_list", "prompt_is_question", "prompt_imperative",
]


def main():
    rows = load_dataset()
    print(f"loaded {len(rows)} samples")

    # Build feature matrix
    X = np.array([extract_features(r) for r in rows])
    y = np.array([r["drift"] for r in rows])
    sources = np.array([r["source"] for r in rows])
    drift_types = np.array([r["drift_type"] for r in rows])

    print(f"  feature matrix: {X.shape}")
    print(f"  n_features: {len(FEATURE_NAMES)}")
    assert X.shape[1] == len(FEATURE_NAMES), "feature-count mismatch"

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs = []
    all_test_scores = []
    all_test_labels = []
    all_test_indices = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            C=1.0, max_iter=2000, random_state=RANDOM_STATE,
            class_weight="balanced",   # 82/18 imbalance inflates intercept otherwise
        )
        clf.fit(X_train, y[train_idx])
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y[test_idx], probs)
        fold_aucs.append(auc)
        all_test_scores.extend(probs)
        all_test_labels.extend(y[test_idx])
        all_test_indices.extend(test_idx)
        print(f"  fold {fold+1}: AUC {auc:.4f}  n_train={len(train_idx)}  n_test={len(test_idx)}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    pooled_auc = roc_auc_score(all_test_labels, all_test_scores)

    print()
    print(f"  MEAN fold AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  POOLED AUC (concat across folds): {pooled_auc:.4f}")

    # Per-source stratified (use pooled predictions)
    all_scores_arr = np.array(all_test_scores)
    all_labels_arr = np.array(all_test_labels)
    all_idx_arr = np.array(all_test_indices)
    source_aucs = {}
    for src in sorted(set(sources)):
        mask = sources[all_idx_arr] == src
        if mask.sum() == 0: continue
        sub_labels = all_labels_arr[mask]
        sub_scores = all_scores_arr[mask]
        if len(set(sub_labels)) < 2:
            source_aucs[src] = None
            continue
        source_aucs[src] = float(roc_auc_score(sub_labels, sub_scores))
    print()
    print("  per-source AUC (pooled predictions):")
    for src, auc in source_aucs.items():
        print(f"    {src:<20s}  {auc:.4f}" if auc is not None else f"    {src:<20s}  n/a (single-class)")

    # Per-drift-type (vs gold negatives)
    gold_mask_full = drift_types == "gold"
    type_aucs = {}
    for dt in sorted(set(drift_types)):
        if dt == "gold": continue
        # Use pooled predictions; filter by drift_type OR gold
        mask = (drift_types[all_idx_arr] == dt) | gold_mask_full[all_idx_arr]
        sub_labels = all_labels_arr[mask]
        sub_scores = all_scores_arr[mask]
        if len(set(sub_labels)) < 2:
            type_aucs[dt] = None
            continue
        type_aucs[dt] = float(roc_auc_score(sub_labels, sub_scores))
    print()
    print("  per-drift-type AUC (vs gold negatives, pooled):")
    for dt, auc in type_aucs.items():
        print(f"    {dt:<22s}  {auc:.4f}" if auc is not None else f"    {dt:<22s}  n/a")

    # Fit final model on ALL data for weights export
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X)
    clf = LogisticRegression(
        C=1.0, max_iter=2000, random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    clf.fit(X_all_s, y)

    print()
    print("  feature importance (scaled, |coef| desc):")
    coef_pairs = sorted(
        zip(FEATURE_NAMES, clf.coef_[0]),
        key=lambda kv: -abs(kv[1]),
    )
    for name, c in coef_pairs:
        print(f"    {name:<25s}  {c:+.3f}")
    print(f"    intercept               {clf.intercept_[0]:+.3f}")

    # Improvement vs Day 1 null best (0.733)
    improvement = mean_auc - 0.733
    print()
    print(f"  Improvement over null-best (0.733 schema_conformance): {improvement:+.4f}")
    if mean_auc >= 0.90:
        print(f"  RESULT: mean AUC {mean_auc:.4f} >= 0.90 — calibrated detector is shippable for v6.0.")
    elif mean_auc >= 0.85:
        print(f"  RESULT: mean AUC {mean_auc:.4f} in [0.85, 0.90) — good, ship v6.0 beta with honest numbers.")
    else:
        print(f"  RESULT: mean AUC {mean_auc:.4f} < 0.85 — needs more features or better training data before shipping.")

    # Write artifact
    result = {
        "methodology": "drift v0 — 22-feature calibrated LR, 5-fold stratified CV, seed=0",
        "n_samples": len(rows),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "fold_aucs": fold_aucs,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "pooled_auc": float(pooled_auc),
        "per_source_auc": source_aucs,
        "per_drift_type_auc": type_aucs,
        "coefficients": dict(zip(FEATURE_NAMES, clf.coef_[0].tolist())),
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "null_best_auc": 0.733,
        "improvement_over_null": float(improvement),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote -> {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
